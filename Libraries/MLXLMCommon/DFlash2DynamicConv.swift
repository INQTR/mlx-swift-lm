//
//  DFlash2DynamicConv.swift
//  mlx-swift-lm
//

import Foundation
import MLX

// MARK: - Drafter dynamic conv

/// `Conv(x)_t = sum_tap (base[tap] + dyn[t, tap, group]) * x[t - tap]` as
/// one launch, in the separate ops' order and rounding: per tap `base * x`
/// rounded and added, then `dyn * x` rounded and added, all in the
/// activation dtype. Positions before the block read zeros and still run the
/// arithmetic, exactly as the zero-padded ops did.
private func makeDFlash2DynamicConvKernel() -> MLXFast.MLXFastKernel? {
    let source = """
        const uint idx = thread_position_in_grid.x;
        const uint h = idx % H;
        const uint bl = idx / H;
        const uint l = bl % L;
        const uint g = h / C;
        auto xr = x + (size_t)bl * H + h;
        auto dr = dyn + (size_t)bl * DROW + DOFF + g;
        T out = static_cast<T>(0);
        for (int tap = 0; tap < K; ++tap) {
          const T v = (l >= (uint)tap) ? xr[-(tap * (int)H)] : static_cast<T>(0);
          const T bv = base[tap * H + h] * v;
          out = (tap == 0) ? bv : (out + bv);
          const T dv = dr[tap * G] * v;
          out = out + dv;
        }
        y[idx] = out;
        """
    return MLXFast.metalKernel(
        name: "dflash2_dynamic_conv",
        inputNames: ["x", "dyn", "base"],
        outputNames: ["y"],
        source: source
    )
}

private final class DFlash2DynamicConvKernelManager: Sendable {
    static let shared = DFlash2DynamicConvKernelManager()
    let kernel: MLXFast.MLXFastKernel?
    private init() {
        kernel = makeDFlash2DynamicConvKernel()
    }
}

/// The drafter's grouped dynamic conv as one launch. `hidden` is
/// `[B, L, H]`; the per-position kernel taps sit at column `dynamicOffset`
/// of `dynamic` `[B, L, dynamicRowLength]` as `tap * groups + group`; `base`
/// is the slot's `[K, H]` base taps in the activation dtype. Nil when the
/// kernel does not cover the shape (bf16 or f16, 3-D inputs).
public func dflash2DynamicConv(
    _ hidden: MLXArray, dynamic: MLXArray, dynamicOffset: Int, dynamicRowLength: Int,
    base: MLXArray, kernelSize: Int, groupSize: Int
) -> MLXArray? {
    guard hidden.dtype == .bfloat16 || hidden.dtype == .float16, dynamic.dtype == hidden.dtype,
        base.dtype == hidden.dtype, hidden.ndim == 3, dynamic.ndim == 3, base.ndim == 2,
        kernelSize >= 1, groupSize >= 1
    else { return nil }
    let B = hidden.dim(0)
    let L = hidden.dim(1)
    let H = hidden.dim(2)
    guard H % groupSize == 0 else { return nil }
    let groups = H / groupSize
    guard dynamic.dim(0) == B, dynamic.dim(1) == L, dynamic.dim(2) == dynamicRowLength,
        dynamicOffset >= 0, dynamicOffset + kernelSize * groups <= dynamicRowLength,
        base.dim(0) == kernelSize, base.dim(1) == H
    else { return nil }
    guard let kernel = DFlash2DynamicConvKernelManager.shared.kernel else { return nil }
    let outputs = kernel(
        [hidden, dynamic, base],
        template: [
            ("T", hidden.dtype), ("H", H), ("L", L), ("C", groupSize), ("G", groups),
            ("K", kernelSize), ("DROW", dynamicRowLength), ("DOFF", dynamicOffset),
        ],
        grid: (B * L * H, 1, 1),
        threadGroup: (256, 1, 1),
        outputShapes: [hidden.shape],
        outputDTypes: [hidden.dtype]
    )
    return outputs[0]
}
