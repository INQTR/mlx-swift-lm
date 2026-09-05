//
//  RMSNormResidual.swift
//  mlx-swift-lm
//

import Foundation
import MLX

// MARK: - Residual add + RMS norm

/// `h = x + r` and `rmsNorm(h, weight, eps)` in one launch. The kernel is
/// MLX's `rms_looped` with the residual add folded into the row read: the
/// same thread geometry, the same per-thread accumulation order and the same
/// simd reductions, so `h` and the normed output are bitwise what the
/// separate `Add` and `RMSNorm` launches produce.
private func makeRMSNormResidualKernel() -> MLXFast.MLXFastKernel? {
    let source = """
        constexpr int N_READS = 4;
        constexpr int SIMD_SIZE = 32;
        const uint gid = threadgroup_position_in_grid.x;
        const uint lid = thread_position_in_threadgroup.x;
        const uint lsize = threads_per_threadgroup.x;
        const uint simd_lane_id = thread_index_in_simdgroup;
        const uint simd_group_id = simdgroup_index_in_threadgroup;
        const uint axis_size = (uint)AXIS;
        const float eps_value = eps[0];

        threadgroup float local_inv_mean[1];
        threadgroup float local_sums[SIMD_SIZE];

        const size_t row = gid * size_t(axis_size) + lid * N_READS;
        const device T* xr = x + row;
        const device T* rr = r + row;
        device T* hr = h + row;
        device T* outr = out + row;
        const device T* wr = w + lid * N_READS;

        float acc = 0;
        for (uint i0 = 0; i0 < axis_size; i0 += lsize * N_READS) {
          if (i0 + lid * N_READS + N_READS <= axis_size) {
            for (int i = 0; i < N_READS; i++) {
              const T hi = (T)((float)xr[i + i0] + (float)rr[i + i0]);
              hr[i + i0] = hi;
              const float xi = hi;
              acc += xi * xi;
            }
          } else {
            for (int i = 0; i < N_READS; i++) {
              if ((i0 + lid * N_READS + i) < axis_size) {
                const T hi = (T)((float)xr[i + i0] + (float)rr[i + i0]);
                hr[i + i0] = hi;
                const float xi = hi;
                acc += xi * xi;
              }
            }
          }
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
            local_inv_mean[0] = metal::precise::rsqrt(acc / axis_size + eps_value);
          }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint i0 = 0; i0 < axis_size; i0 += lsize * N_READS) {
          if (i0 + lid * N_READS + N_READS <= axis_size) {
            for (int i = 0; i < N_READS; i++) {
              outr[i0 + i] = wr[i + i0] * static_cast<T>(hr[i0 + i] * local_inv_mean[0]);
            }
          } else {
            for (int i = 0; i < N_READS; i++) {
              if ((i0 + lid * N_READS + i) < axis_size) {
                outr[i0 + i] = wr[i + i0] * static_cast<T>(hr[i0 + i] * local_inv_mean[0]);
              }
            }
          }
        }
        """
    return MLXFast.metalKernel(
        name: "fastmath_rms_norm_residual",
        inputNames: ["x", "r", "w", "eps"],
        outputNames: ["h", "out"],
        source: source
    )
}

private final class RMSNormResidualKernelManager: Sendable {
    static let shared = RMSNormResidualKernelManager()
    let kernel: MLXFast.MLXFastKernel?
    private init() {
        kernel = makeRMSNormResidualKernel()
    }
}

/// `(x + r, rmsNorm(x + r, weight, eps))` as one launch, bitwise the two
/// separate ops. `x` and `r` share a shape whose last axis is the normed
/// axis; `weight` is `[axis]`.
public func rmsNormResidual(
    _ x: MLXArray, _ r: MLXArray, weight: MLXArray, eps: Float
) -> (h: MLXArray, out: MLXArray) {
    precondition(x.shape == r.shape, "residual shapes differ")
    precondition(x.dtype == r.dtype && weight.dtype == x.dtype, "residual dtypes differ")
    let axis = x.dim(-1)
    precondition(weight.ndim == 1 && weight.dim(0) == axis, "weight must be [axis]")
    guard let kernel = RMSNormResidualKernelManager.shared.kernel else {
        fatalError("rms norm residual kernel unavailable")
    }
    let rows = x.size / axis
    // MLX's geometry: one threadgroup per row; rows up to 4096 wide use
    // ceil(axis / 4) threads rounded up to whole simdgroups, wider rows loop
    // over 1024 threads.
    let threads: Int
    if axis <= 4096 {
        let needed = (axis + 3) / 4
        threads = 32 * ((needed + 31) / 32)
    } else {
        threads = 1024
    }
    let outputs = kernel(
        [x, r, weight, MLXArray([eps])],
        template: [("T", x.dtype), ("AXIS", axis)],
        grid: (threads * rows, 1, 1),
        threadGroup: (threads, 1, 1),
        outputShapes: [x.shape, x.shape],
        outputDTypes: [x.dtype, x.dtype]
    )
    return (outputs[0], outputs[1])
}
