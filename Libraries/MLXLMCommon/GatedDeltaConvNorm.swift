//
//  GatedDeltaConvNorm.swift
//  mlx-swift-lm
//

import Foundation
import MLX

// MARK: - Fused conv + silu + q/k norm

/// One simdgroup per head row: the depthwise conv taps, silu, and for the
/// q|k rows MLX's `rms_single_row` reduction (32 lanes x 4 values), then the
/// head scale. The conv input is virtual — the `K - 1` state rows followed
/// by the qkv columns of the projection rows — and the kernel also writes
/// it out as the `[B, S + K - 1, D]` block the replay gathers from, plus
/// its last `K - 1` rows as the next conv state, from the same reads.
/// `w` is `[D, K, 1]`, q/k/v are `[B, S, H, HD]`. Compiled without fast
/// math, with the conv taps and the sum of squares accumulated as plain
/// statements and the silu in MLX's bf16 expression: the form that is
/// bitwise with the ops chain (explicit fma, contraction off and the
/// float-rounded silu intermediates were not).
private func makeGatedDeltaConvNormKernel() -> MLXFast.MLXFastKernel? {
    let source = """
        constexpr int N_READS = 4;
        constexpr int ROWS = 2 * HK + HV;
        const uint g = thread_position_in_grid.x;
        const uint lane = thread_index_in_simdgroup;
        const uint row = g / 32;
        const uint S_ = (uint)dims[0];
        const uint bs = row / ROWS;
        const uint r = row - bs * ROWS;
        const uint b = bs / S_;
        const uint s = bs - b * S_;
        const uint ch0 = r * HD + lane * N_READS;
        auto st = state + ((size_t)b * (K - 1)) * D + ch0;
        auto pr = rows + ((size_t)b * S_) * ROW + ROFF + ch0;
        auto wp = w + (size_t)ch0 * K;
        device T* cat = concat + ((size_t)b * (S_ + K - 1)) * D + ch0;
        device T* nx = next + ((size_t)b * (K - 1)) * D + ch0;

        // Virtual conv input row s + t: state rows first, then projection rows.
        T taps[K][N_READS];
        for (int t = 0; t < K; ++t) {
          const uint vr = s + t;
          const device T* src = (vr < (uint)(K - 1))
              ? (st + (size_t)vr * D)
              : (pr + (size_t)(vr - (K - 1)) * ROW);
          for (int i = 0; i < N_READS; ++i) {
            taps[t][i] = src[i];
          }
        }
        // The conv input block: every row writes its own projection row, row
        // 0 also the state rows. Block rows S .. S + K - 2 are the next state.
        for (int i = 0; i < N_READS; ++i) {
          cat[(size_t)(s + K - 1) * D + i] = taps[K - 1][i];
        }
        if (s + K - 1 >= S_) {
          for (int i = 0; i < N_READS; ++i) {
            nx[(size_t)(s + K - 1 - S_) * D + i] = taps[K - 1][i];
          }
        }
        if (s == 0) {
          for (int t = 0; t < K - 1; ++t) {
            for (int i = 0; i < N_READS; ++i) {
              cat[(size_t)t * D + i] = taps[t][i];
            }
            if ((uint)t >= S_) {
              for (int i = 0; i < N_READS; ++i) {
                nx[(size_t)(t - S_) * D + i] = taps[t][i];
              }
            }
          }
        }

        T val[N_READS];
        for (int i = 0; i < N_READS; ++i) {
          float acc = 0.0;
          for (int t = 0; t < K; ++t) {
            acc += static_cast<float>(taps[t][i]) * wp[i * K + t];
          }
          const T c = static_cast<T>(acc);
          const T e = static_cast<T>(metal::precise::exp(metal::abs(static_cast<float>(c))));
          auto y = 1 / (1 + e);
          const T sg = (c < 0) ? y : 1 - y;
          val[i] = c * sg;
        }

        if (r < (uint)(2 * HK)) {
          float sq = 0;
          for (int i = 0; i < N_READS; ++i) {
            float xi = val[i];
            sq += xi * xi;
          }
          sq = simd_sum(sq);
          const float inv = metal::precise::rsqrt(sq / HD + eps[0]);
          const bool isQ = r < (uint)HK;
          const T sc = isQ ? scales[0] : scales[1];
          device T* dst = isQ
              ? (q + (size_t)(bs * HK + r) * HD)
              : (k + (size_t)(bs * HK + (r - HK)) * HD);
          dst += lane * N_READS;
          for (int i = 0; i < N_READS; ++i) {
            const T y = static_cast<T>(val[i] * inv);
            dst[i] = sc * y;
          }
        } else {
          device T* dst = v + (size_t)(bs * HV + (r - 2 * HK)) * HD + lane * N_READS;
          for (int i = 0; i < N_READS; ++i) {
            dst[i] = val[i];
          }
        }
        """
    return MLXFast.metalKernel(
        name: "gdn_conv_norm_qkv",
        inputNames: ["state", "rows", "w", "scales", "eps", "dims"],
        outputNames: ["q", "k", "v", "concat", "next"],
        source: source
    )
}

private final class GatedDeltaConvNormKernelManager: Sendable {
    static let shared = GatedDeltaConvNormKernelManager()
    let kernel: MLXFast.MLXFastKernel?
    private init() {
        kernel = makeGatedDeltaConvNormKernel()
    }
}

/// The gated delta net's `silu(conv1d(concat([convState, qkv])))`, split into
/// the weightless RMS-normed and scaled `q`/`k` heads and the `v` heads, as
/// one launch that never builds the concat as an input: `convState` is
/// `[B, K - 1, D]` and the qkv columns sit at `rowOffset` of `rows`
/// `[B, S, row]` (the fused input projection, or a plain `[B, S, D]` at 0).
/// Also returns the conv input block `[B, S + K - 1, D]` (what a replay
/// gathers its conv state from) and the next conv state, its last `K - 1`
/// rows. Nil when the kernel does not cover the shape: bf16 only, a 128-wide
/// head dim shared by keys and values, `weight` `[D, K, 1]`, `scales` the
/// `[q, k]` pair in the activation dtype.
public func gatedDeltaConvNormQKV(
    convState: MLXArray, rows: MLXArray, rowOffset: Int, weight: MLXArray, numKHeads: Int,
    numVHeads: Int, headDim: Int, scales: MLXArray, eps: Float
) -> (q: MLXArray, k: MLXArray, v: MLXArray, convInput: MLXArray, nextConvState: MLXArray)? {
    guard rows.dtype == .bfloat16, convState.dtype == rows.dtype, weight.dtype == rows.dtype,
        scales.dtype == rows.dtype, headDim == 128, rows.ndim == 3, convState.ndim == 3,
        weight.ndim == 3, scales.size == 2
    else { return nil }
    let B = rows.dim(0)
    let S = rows.dim(1)
    let rowLength = rows.dim(2)
    let D = weight.dim(0)
    let K = weight.dim(1)
    guard S >= 1, K >= 2, weight.dim(2) == 1, D == (2 * numKHeads + numVHeads) * headDim,
        convState.dim(0) == B, convState.dim(1) == K - 1, convState.dim(2) == D,
        rowOffset >= 0, rowOffset + D <= rowLength
    else { return nil }
    guard let kernel = GatedDeltaConvNormKernelManager.shared.kernel else { return nil }
    let headRows = B * S * (2 * numKHeads + numVHeads)
    let outputs = kernel(
        [convState, rows, weight, scales, MLXArray([eps]), MLXArray([Int32(S)])],
        template: [
            ("T", rows.dtype), ("D", D), ("K", K), ("HK", numKHeads), ("HV", numVHeads),
            ("HD", headDim), ("ROW", rowLength), ("ROFF", rowOffset),
        ],
        grid: (32 * headRows, 1, 1),
        threadGroup: (128, 1, 1),
        outputShapes: [
            [B, S, numKHeads, headDim], [B, S, numKHeads, headDim], [B, S, numVHeads, headDim],
            [B, S + K - 1, D], [B, K - 1, D],
        ],
        outputDTypes: [rows.dtype, rows.dtype, rows.dtype, rows.dtype, rows.dtype]
    )
    return (outputs[0], outputs[1], outputs[2], outputs[3], outputs[4])
}
