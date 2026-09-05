//
//  GatedDeltaNormGate.swift
//  mlx-swift-lm
//

import Foundation
import MLX

// MARK: - Gated output norm

/// One simdgroup per head row: MLX's `rms_single_row` over the 128 values
/// (32 lanes x 4), then the f32 gate `silu(z) * normed` rounded once, as the
/// compiled `rmsNorm` + `silu(z.f32) * x.f32` chain does. The gate is read
/// straight out of its projection row (`gate + row * GROW + GOFF`), so the
/// strided gate view never needs a copy.
private func makeGatedDeltaNormGateKernel() -> MLXFast.MLXFastKernel? {
    let source = """
        constexpr int N_READS = 4;
        const uint g = thread_position_in_grid.x;
        const uint lane = thread_index_in_simdgroup;
        const uint row = g / 32;
        const uint bs = row / HV;
        const uint h = row - bs * HV;
        const uint base = lane * N_READS;
        auto xr = x + (size_t)row * HD + base;
        auto zr = gate + (size_t)bs * GROW + GOFF + (size_t)h * HD + base;
        const device T* wr = w + base;
        device T* outr = out + (size_t)row * HD + base;

        float acc = 0;
        for (int i = 0; i < N_READS; i++) {
          float xi = xr[i];
          acc += xi * xi;
        }
        acc = simd_sum(acc);
        const float inv = metal::precise::rsqrt(acc / HD + eps[0]);
        for (int i = 0; i < N_READS; i++) {
          const T normed = wr[i] * static_cast<T>(xr[i] * inv);
          const float zf = static_cast<float>(zr[i]);
          auto y = 1 / (1 + metal::exp(metal::abs(zf)));
          const float sg = (zf < 0) ? y : 1 - y;
          const float act = zf * sg;
          outr[i] = static_cast<T>(act * static_cast<float>(normed));
        }
        """
    return MLXFast.metalKernel(
        name: "gdn_norm_gate",
        inputNames: ["x", "gate", "w", "eps"],
        outputNames: ["out"],
        source: source
    )
}

private final class GatedDeltaNormGateKernelManager: Sendable {
    static let shared = GatedDeltaNormGateKernelManager()
    let kernel: MLXFast.MLXFastKernel?
    private init() {
        kernel = makeGatedDeltaNormGateKernel()
    }
}

/// `silu(gate.f32) * rmsNorm(x, weight, eps).f32` rounded to `x`'s dtype, as
/// one launch. `x` is `[B, S, H, 128]`; the gate rows live in `gateSource`
/// `[B, S, gateRowLength]` at column `gateOffset` (the `z` slice of the
/// fused input projection, or a plain `[B, S, H * 128]` at offset 0). Nil
/// when the kernel does not cover the shape (bf16, 128-wide heads).
public func gatedDeltaNormGate(
    _ x: MLXArray, gateSource: MLXArray, gateOffset: Int, gateRowLength: Int,
    weight: MLXArray, eps: Float
) -> MLXArray? {
    guard x.dtype == .bfloat16, gateSource.dtype == x.dtype, weight.dtype == x.dtype,
        x.ndim == 4, x.dim(3) == 128, gateSource.ndim == 3,
        gateSource.dim(2) == gateRowLength, weight.ndim == 1, weight.dim(0) == 128,
        gateSource.dim(0) * gateSource.dim(1) == x.dim(0) * x.dim(1),
        gateOffset >= 0, gateOffset + x.dim(2) * 128 <= gateRowLength
    else { return nil }
    guard let kernel = GatedDeltaNormGateKernelManager.shared.kernel else { return nil }
    let rows = x.dim(0) * x.dim(1) * x.dim(2)
    let outputs = kernel(
        [x, gateSource, weight, MLXArray([eps])],
        template: [
            ("T", x.dtype), ("HV", x.dim(2)), ("HD", 128), ("GOFF", gateOffset),
            ("GROW", gateRowLength),
        ],
        grid: (32 * rows, 1, 1),
        threadGroup: (128, 1, 1),
        outputShapes: [x.shape],
        outputDTypes: [x.dtype]
    )
    return outputs[0]
}
