//
//  DFlash2GreedyWalk.swift
//  mlx-swift-lm
//

import Foundation
import MLX

// MARK: - Selector greedy walk

/// The candidate selector's greedy path as one launch: one simdgroup per
/// batch row, lane `j` holds candidate `j`. Per position the lane adds its
/// unary score and the edge from the previous pick (the anchor edge at
/// position 0) in the score dtype, as the separate ops do, and the pick is
/// the highest score with the lowest index on ties, MLX's `argMax` order.
private func makeDFlash2GreedyWalkKernel() -> MLXFast.MLXFastKernel? {
    let source = """
        const uint b = threadgroup_position_in_grid.x;
        const uint j = thread_position_in_threadgroup.x;
        const bool active = j < (uint)K;
        // `auto`: MLX binds inputs under 8 elements in the constant address space.
        auto un = unary + (size_t)b * L * K;
        auto ed = edges + (size_t)b * (L - 1) * K * K;
        auto an = anchor + (size_t)b * K;
        auto cand = candidates + (size_t)b * L * K;
        device IdxT* out = tokens + (size_t)b * L;
        uint prev = 0;
        for (int t = 0; t < L; ++t) {
          float sf = -metal::numeric_limits<float>::infinity();
          if (active) {
            const T e = (t == 0) ? an[j] : ed[((size_t)(t - 1) * K + prev) * K + j];
            const T s = un[(size_t)t * K + j] + e;
            sf = static_cast<float>(s);
          }
          const float m = simd_max(sf);
          const uint pick = (active && sf == m) ? j : (uint)K;
          prev = simd_min(pick);
          if (j == 0) {
            out[t] = cand[(size_t)t * K + prev];
          }
        }
        """
    return MLXFast.metalKernel(
        name: "dflash2_greedy_walk",
        inputNames: ["unary", "edges", "anchor", "candidates"],
        outputNames: ["tokens"],
        source: source
    )
}

private final class DFlash2GreedyWalkKernelManager: Sendable {
    static let shared = DFlash2GreedyWalkKernelManager()
    let kernel: MLXFast.MLXFastKernel?
    private init() {
        kernel = makeDFlash2GreedyWalkKernel()
    }
}

/// Greedy tokens `[B, L]` over `unary` `[B, L, K]` candidate scores,
/// `edges` `[B, L - 1, K, K]` (previous pick x next candidate), the anchor's
/// `[B, K]` edges into position 0, and the `[B, L, K]` candidate ids the
/// picks index. Nil when the kernel does not cover the shape (K up to 32,
/// at least two positions, one float dtype throughout).
public func dflash2GreedyWalk(
    unary: MLXArray, edges: MLXArray, anchorEdges: MLXArray, candidates: MLXArray
) -> MLXArray? {
    guard unary.ndim == 3, edges.ndim == 4, anchorEdges.ndim == 2, candidates.ndim == 3
    else { return nil }
    let B = unary.dim(0)
    let L = unary.dim(1)
    let K = unary.dim(2)
    guard L >= 2, K >= 1, K <= 32,
        [DType.bfloat16, .float16, .float32].contains(unary.dtype),
        edges.dtype == unary.dtype, anchorEdges.dtype == unary.dtype,
        edges.shape == [B, L - 1, K, K], anchorEdges.shape == [B, K],
        candidates.shape == [B, L, K], candidates.dtype == .uint32 || candidates.dtype == .int32,
        let kernel = DFlash2GreedyWalkKernelManager.shared.kernel
    else { return nil }
    let outputs = kernel(
        [unary, edges, anchorEdges, candidates],
        template: [("T", unary.dtype), ("IdxT", candidates.dtype), ("L", L), ("K", K)],
        grid: (32 * B, 1, 1),
        threadGroup: (32, 1, 1),
        outputShapes: [[B, L]],
        outputDTypes: [candidates.dtype]
    )
    return outputs[0]
}
