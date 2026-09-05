//
//  AttentionNormRope.swift
//  mlx-swift-lm
//

import Foundation
import MLX
import MLXNN

// MARK: - Fused q/k RMS norm + RoPE

/// The parameters of the plain (non-traditional) `RoPE` layer the fused
/// kernel replicates. Built from the same values `initializeRope` gets, so
/// the vendor does not need the layer's internals: nil for a traditional
/// rope or any scaling type other than `default` / `linear` (those become
/// different layer classes with their own frequency tables).
public struct PlainRoPEParameters: Sendable {
    public let dimensions: Int
    public let base: Float
    public let scale: Float

    public init(dimensions: Int, base: Float, scale: Float) {
        self.dimensions = dimensions
        self.base = base
        self.scale = scale
    }

    public init?(
        dims: Int, base: Float, traditional: Bool, scalingConfig: [String: StringOrNumber]?
    ) {
        guard !traditional else { return nil }
        let ropeType: String = {
            if let config = scalingConfig,
                let typeValue = config["type"] ?? config["rope_type"],
                case .string(let s) = typeValue
            {
                return s
            }
            return "default"
        }()
        switch ropeType {
        case "default":
            scale = 1.0
        case "linear":
            if let factor = scalingConfig?["factor"]?.asFloat() {
                scale = 1 / factor
            } else {
                scale = 1.0
            }
        default:
            return nil
        }
        dimensions = dims
        self.base = base
    }
}

/// One threadgroup per (batch, position, head): MLX's `rms_single_row`
/// reduction (4 values per thread, simd sums, the cross-simd table) over
/// the head's columns of the projection row, the weight multiply, then the
/// `rope` kernel's rotation of the first `RD` dims from the same rounded
/// values (theta from `exp2(-d * log2(base))` and the fast cos/sin, as
/// MLX computes it). Heads below `HQ` are queries, the rest keys; each
/// output is head-major `[B, H, L, HD]`, what the transposed norm output
/// feeds the rope today. The `fastmath_` name prefix compiles the kernel
/// the way the package's AOT metallib is built (the `rope` kernel's `exp2`
/// is the fast one there); with the plain rotation below that is the
/// bitwise form, the other arithmetic orders (explicit fma, contraction
/// off) were not.
private func makeAttentionNormRopeKernel() -> MLXFast.MLXFastKernel? {
    let source = """
        constexpr int N_READS = 4;
        constexpr int SIMD_SIZE = 32;
        const uint gid = threadgroup_position_in_grid.x;
        const uint lid = thread_position_in_threadgroup.x;
        const uint simd_lane_id = thread_index_in_simdgroup;
        const uint simd_group_id = simdgroup_index_in_threadgroup;
        const uint seq = (uint)dims[0];
        const uint heads = (uint)(HQ + HK);
        const uint head = gid % heads;
        const uint bl = gid / heads;
        const uint l = bl % seq;
        const uint b = bl / seq;
        const bool isQ = head < (uint)HQ;
        const uint h = isQ ? head : head - (uint)HQ;
        const size_t col = isQ ? (size_t)(QOFF + h * QSTRIDE) : (size_t)(KOFF + h * KSTRIDE);
        const device T* xr = rows + (size_t)bl * ROW + col + lid * N_READS;
        const device T* wr = (isQ ? qw : kw) + lid * N_READS;

        threadgroup float local_inv_mean[1];
        threadgroup float local_sums[SIMD_SIZE];
        threadgroup T ybuf[HD];

        T xv[N_READS];
        float acc = 0;
        for (int i = 0; i < N_READS; i++) {
          xv[i] = xr[i];
          float xi = xv[i];
          acc += xi * xi;
        }
        acc = simd_sum(acc);
        if (simd_group_id == 0) {
          local_sums[simd_lane_id] = 0;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd_lane_id == 0) {
          local_sums[simd_group_id] = acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd_group_id == 0) {
          acc = simd_sum(local_sums[simd_lane_id]);
          if (simd_lane_id == 0) {
            local_inv_mean[0] = metal::precise::rsqrt(acc / (float)HD + eps[0]);
          }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int i = 0; i < N_READS; i++) {
          ybuf[lid * N_READS + i] = wr[i] * static_cast<T>(xv[i] * local_inv_mean[0]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const int batch_offset = pos[b * POS_STRIDE];
        const float lf = rope[1] * static_cast<float>(l + batch_offset);
        constexpr uint hlf = (uint)(RD / 2);
        device T* dst = (isQ ? (q + ((size_t)(b * HQ + h) * seq + l) * HD)
                             : (k + ((size_t)(b * HK + h) * seq + l) * HD)) + lid * N_READS;
        for (int i = 0; i < N_READS; i++) {
          const uint j = lid * N_READS + i;
          if (j < (uint)RD) {
            const bool first = j < hlf;
            const uint p = first ? j : j - hlf;
            const float d = static_cast<float>(p) / static_cast<float>(hlf);
            const float inv_freq = metal::exp2(-d * rope[0]);
            const float theta = lf * inv_freq;
            const float costheta = metal::fast::cos(theta);
            const float sintheta = metal::fast::sin(theta);
            const float x1 = static_cast<float>(ybuf[p]);
            const float x2 = static_cast<float>(ybuf[p + hlf]);
            const float r =
                first ? (x1 * costheta - x2 * sintheta) : (x1 * sintheta + x2 * costheta);
            dst[i] = static_cast<T>(r);
          } else {
            dst[i] = ybuf[j];
          }
        }
        """
    return MLXFast.metalKernel(
        name: "fastmath_attention_norm_rope",
        inputNames: ["rows", "qw", "kw", "eps", "pos", "rope", "dims"],
        outputNames: ["q", "k"],
        source: source
    )
}

private final class AttentionNormRopeKernelManager: Sendable {
    static let shared = AttentionNormRopeKernelManager()
    let kernel: MLXFast.MLXFastKernel?
    private init() {
        kernel = makeAttentionNormRopeKernel()
    }
}

/// The attention's `rmsNorm(q) → rope`, `rmsNorm(k) → rope` as one launch
/// reading the heads straight out of the stacked projection row `rows`
/// `[B, L, row]`: query head `h` occupies `queryOffset + h * queryHeadStride
/// ..< + headDim` (a stride of `2 * headDim` skips an interleaved gate),
/// key head `h` `keyOffset + h * keyHeadStride`. `offset` is the rope's
/// position offset, a `[1]` or `[B]` int array. Returns the rotated
/// queries `[B, HQ, L, headDim]` and keys `[B, HK, L, headDim]` (with
/// `queryHeads` 0 the queries output is a placeholder). Nil when the kernel
/// does not cover the shape: bf16/f16 rows, head dim a multiple of 4 up to
/// 4096, an even rope dimension up to the head dim.
public func attentionNormRope(
    rows: MLXArray, queryOffset: Int, queryHeadStride: Int, queryHeads: Int, keyOffset: Int,
    keyHeadStride: Int, keyHeads: Int, headDim: Int, queryWeight: MLXArray, keyWeight: MLXArray,
    eps: Float, rope: PlainRoPEParameters, offset: MLXArray
) -> (queries: MLXArray, keys: MLXArray)? {
    guard rows.dtype == .bfloat16 || rows.dtype == .float16, rows.ndim == 3,
        queryWeight.dtype == rows.dtype, keyWeight.dtype == rows.dtype,
        queryWeight.shape == [headDim], keyWeight.shape == [headDim],
        headDim % 4 == 0, headDim >= 4, headDim <= 4096,
        rope.dimensions >= 2, rope.dimensions % 2 == 0, rope.dimensions <= headDim,
        queryHeads >= 0, keyHeads >= 1, offset.ndim <= 1, offset.dtype.isInteger
    else { return nil }
    let B = rows.dim(0)
    let L = rows.dim(1)
    let rowLength = rows.dim(2)
    guard L >= 1, offset.size == 1 || offset.size == B,
        keyOffset >= 0, keyOffset + (keyHeads - 1) * keyHeadStride + headDim <= rowLength,
        queryHeads == 0
            || (queryOffset >= 0
                && queryOffset + (queryHeads - 1) * queryHeadStride + headDim <= rowLength)
    else { return nil }
    guard let kernel = AttentionNormRopeKernelManager.shared.kernel else { return nil }
    let threads = headDim / 4
    let groups = B * L * (queryHeads + keyHeads)
    let positions = offset.dtype == .int32 ? offset : offset.asType(.int32)
    let outputs = kernel(
        [
            rows, queryWeight, keyWeight, MLXArray([eps]), positions,
            MLXArray([log2(rope.base), rope.scale]), MLXArray([Int32(L)]),
        ],
        template: [
            ("T", rows.dtype), ("HD", headDim), ("HQ", queryHeads), ("HK", keyHeads),
            ("ROW", rowLength), ("QOFF", queryOffset), ("QSTRIDE", queryHeadStride),
            ("KOFF", keyOffset), ("KSTRIDE", keyHeadStride), ("RD", rope.dimensions),
            ("POS_STRIDE", offset.size == 1 ? 0 : 1),
        ],
        grid: (groups * threads, 1, 1),
        threadGroup: (threads, 1, 1),
        outputShapes: [[B, max(queryHeads, 1), L, headDim], [B, keyHeads, L, headDim]],
        outputDTypes: [rows.dtype, rows.dtype]
    )
    return (outputs[0], outputs[1])
}
