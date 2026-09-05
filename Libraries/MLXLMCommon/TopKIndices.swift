//
//  TopKIndices.swift
//  mlx-swift-lm
//

import Foundation
import MLX

// MARK: - Top-k indices

/// Elements per stage-1 threadgroup. 256 threads hold 16 each.
private let topKChunk = 4096
private let topKThreads = 256

/// One stage of the top-k selection: every threadgroup takes a `CHUNK`-wide
/// slice of one row and emits its `K` largest (key, index) pairs, ascending.
/// The key is the float's order-preserving bit pattern, so "largest" is the
/// sort order of `argPartition` (ascending value, ties by index). `KEYS`
/// reads keys instead of values and `HAS_IDX` reads indices instead of
/// positions: the second stage runs the same kernel over the first stage's
/// winners.
private func makeTopKKernel() -> MLXFast.MLXFastKernel? {
    let source = """
        constexpr int TPT = CHUNK / 256;
        const uint chunk = threadgroup_position_in_grid.x;
        const uint chunks = threadgroups_per_grid.x;
        const uint row = threadgroup_position_in_grid.y;
        const uint lid = thread_index_in_threadgroup;
        const uint lane = thread_index_in_simdgroup;
        const uint sg = simdgroup_index_in_threadgroup;
        const uint length = (uint)n[0];

        threadgroup uint tg_k[2][8];
        threadgroup uint tg_i[2][8];
        threadgroup uint res_k[K];
        threadgroup uint res_i[K];

        uint key[TPT];
        uint ix[TPT];
        const uint base = chunk * CHUNK;
        for (int j = 0; j < TPT; ++j) {
          const uint e = base + lid + j * 256;
          uint k = 0;
          uint i = 0xFFFFFFFFu;
          if (e < length) {
            const uint src = row * length + e;
            if (KEYS) {
              k = (uint)vals[src];
            } else {
              const uint u = as_type<uint>((float)vals[src]);
              k = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
            }
            i = HAS_IDX ? idx[src] : e;
          }
          key[j] = k;
          ix[j] = i;
        }

        for (int r = 0; r < K; ++r) {
          uint bk = 0;
          uint bi = 0;
          for (int j = 0; j < TPT; ++j) {
            if (key[j] > bk || (key[j] == bk && ix[j] > bi)) {
              bk = key[j];
              bi = ix[j];
            }
          }
          const uint mk = simd_max(bk);
          const uint mi = simd_max(bk == mk ? bi : 0u);
          if (lane == 0) {
            tg_k[r & 1][sg] = mk;
            tg_i[r & 1][sg] = mi;
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
          uint gk = 0;
          uint gi = 0;
          for (int s = 0; s < 8; ++s) {
            const uint sk = tg_k[r & 1][s];
            const uint si = tg_i[r & 1][s];
            if (sk > gk || (sk == gk && si > gi)) {
              gk = sk;
              gi = si;
            }
          }
          for (int j = 0; j < TPT; ++j) {
            if (key[j] == gk && ix[j] == gi) {
              key[j] = 0;
              ix[j] = 0;
            }
          }
          if (lid == 0) {
            res_k[r] = gk;
            res_i[r] = gi;
          }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lid < (uint)K) {
          const uint o = (row * chunks + chunk) * K + (K - 1 - lid);
          out_key[o] = res_k[lid];
          out_idx[o] = res_i[lid];
        }
        """
    return MLXFast.metalKernel(
        name: "dflash2_topk_stage",
        inputNames: ["vals", "idx", "n"],
        outputNames: ["out_key", "out_idx"],
        source: source
    )
}

private final class TopKKernelManager: Sendable {
    static let shared = TopKKernelManager()
    let kernel: MLXFast.MLXFastKernel?
    private init() {
        kernel = makeTopKKernel()
    }
}

/// The indices of the `k` largest values along the last axis, in the order
/// `argPartition(x, kth: V - k, axis: -1)[..., (V - k)...]` returns them
/// (ascending value, ties by index), as two small launches instead of a
/// full merge sort. Bitwise the partition's tail for finite inputs.
public func topKIndices(_ x: MLXArray, k: Int) -> MLXArray {
    let vocab = x.dim(-1)
    precondition(k >= 1 && k <= vocab && k <= topKThreads, "top-k out of range")
    let chunks = (vocab + topKChunk - 1) / topKChunk
    precondition(chunks * k <= topKChunk, "row too long for two stages")
    guard let kernel = TopKKernelManager.shared.kernel else {
        fatalError("top-k kernel unavailable")
    }
    let rows = x.size / vocab
    let flat = x.reshaped([rows, vocab])
    let resultShape = Array(x.shape.dropLast()) + [k]

    let stage1 = kernel(
        [flat, MLXArray([UInt32(0)]), MLXArray([Int32(vocab)])],
        template: [
            ("InT", x.dtype), ("CHUNK", topKChunk), ("HAS_IDX", 0), ("KEYS", 0), ("K", k),
        ],
        grid: (topKThreads * chunks, rows, 1),
        threadGroup: (topKThreads, 1, 1),
        outputShapes: [[rows, chunks, k], [rows, chunks, k]],
        outputDTypes: [.uint32, .uint32]
    )
    if chunks == 1 {
        return stage1[1].reshaped(resultShape)
    }
    let candidates = chunks * k
    let chunk2 = (candidates + topKThreads - 1) / topKThreads * topKThreads
    let stage2 = kernel(
        [stage1[0], stage1[1], MLXArray([Int32(candidates)])],
        template: [
            ("InT", DType.uint32), ("CHUNK", chunk2), ("HAS_IDX", 1), ("KEYS", 1), ("K", k),
        ],
        grid: (topKThreads, rows, 1),
        threadGroup: (topKThreads, 1, 1),
        outputShapes: [[rows, 1, k], [rows, 1, k]],
        outputDTypes: [.uint32, .uint32]
    )
    return stage2[1].reshaped(resultShape)
}
