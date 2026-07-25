// Copyright © 2026 Apple Inc.
//
// CI pins for the two bitwise contracts the Qwen3.5 decode optimizations
// rest on. Both were proven in standalone rigs before shipping
// (tesseract `benchmarks/gather-sweep/`, ledger C16/C18); these tests keep
// them proven across MLX pin bumps and future edits:
//
//  * C16 — the GDN decode step rewrites the depthwise conv1d as elementwise
//    multiply-adds with f32 accumulation. That is bit-identical to MLX's
//    `Convolution` kernel today; if a pin bump changes the conv kernel's
//    accumulation, decode silently diverges from prefill. The test holds the
//    whole fused decode body against the unfused `callAsFunction` body.
//
//  * C18 — the fused router top-k kernel replaces the
//    argPartition/takeAlong/normalise chain and must reproduce the stable
//    sort's selection order (ties, signed zeros, NaN-above-everything) and
//    the 8-wide sequential in-dtype sum, bit for bit, including the `uint32`
//    index dtype.

import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXLLM
@testable import MLXLMCommon

final class Qwen35BitwiseContractTests: XCTestCase {

    // MARK: - Helpers

    /// Bitwise equality, dtype and shape included. The f32 upcast is exact
    /// and injective for f16/bf16/f32 finite values, so bit-comparing the
    /// upcast compares the originals.
    private func assertBitIdentical(
        _ got: MLXArray, _ want: MLXArray, _ label: String,
        file: StaticString = #filePath, line: UInt = #line
    ) {
        XCTAssertEqual(got.dtype, want.dtype, "\(label): dtype", file: file, line: line)
        XCTAssertEqual(got.shape, want.shape, "\(label): shape", file: file, line: line)
        let a = got.asType(.float32).asArray(Float.self)
        let b = want.asType(.float32).asArray(Float.self)
        let mismatches = zip(a, b).filter { $0.bitPattern != $1.bitPattern }.count
        XCTAssertEqual(
            mismatches, 0, "\(label): \(mismatches)/\(a.count) elements differ bitwise",
            file: file, line: line)
    }

    // MARK: - C18: fused router top-k vs the chain it replaced

    /// The exact chain `routerTopK` falls back to at prefill.
    private func chainRouterTopK(
        _ gates: MLXArray, k: Int, normalize: Bool
    ) -> (MLXArray, MLXArray) {
        let kth = gates.dim(-1) - k
        let inds = MLX.argPartition(gates, kth: kth, axis: -1)[.ellipsis, (kth)...]
        var scores = MLX.takeAlong(gates, inds, axis: -1)
        if normalize {
            scores = scores / scores.sum(axis: -1, keepDims: true)
        }
        return (inds, scores)
    }

    private func assertRouterMatchesChain(
        _ gates: MLXArray, k: Int, normalize: Bool, _ label: String,
        indicesOnly: Bool = false,
        file: StaticString = #filePath, line: UInt = #line
    ) {
        let (chainInds, chainScores) = chainRouterTopK(gates, k: k, normalize: normalize)
        let (fusedInds, fusedScores) = fusedRouterTopK(gates, k: k, normalize: normalize)
        eval(chainInds, chainScores, fusedInds, fusedScores)

        XCTAssertEqual(
            fusedInds.dtype, chainInds.dtype,
            "\(label): index dtype must match argPartition's", file: file, line: line)
        let a = chainInds.reshaped(-1).asArray(UInt32.self)
        let b = fusedInds.reshaped(-1).asArray(UInt32.self)
        XCTAssertEqual(a, b, "\(label): expert selection order", file: file, line: line)

        // NaN score payloads are propagation-order-dependent and not part of
        // the contract (production gates are softmax outputs, NaN-free); the
        // NaN cases pin the *ordering* only.
        if !indicesOnly {
            assertBitIdentical(
                fusedScores.reshaped(-1, k), chainScores.reshaped(-1, k),
                "\(label): scores", file: file, line: line)
        }
    }

    func testFusedRouterTopKMatchesChain() {
        let rows = 64
        for (e, k) in [(256, 8), (128, 8)] {
            for dtype in [DType.float16, DType.bfloat16, DType.float32] {
                for normalize in [true, false] {
                    MLXRandom.seed(UInt64(e + k + (normalize ? 1 : 0)))
                    let tag = "E=\(e) K=\(k) \(dtype) norm=\(normalize)"

                    // The production distribution: softmax outputs.
                    let soft = MLX.softmax(
                        MLXRandom.normal([rows, e]), axis: -1, precise: true
                    ).asType(dtype)
                    assertRouterMatchesChain(soft, k: k, normalize: normalize, "softmax \(tag)")

                    // Heavy ties — the stable tie-break is the hard part.
                    let ties = (MLX.round(MLXRandom.normal([rows, e]) * 2) / 2).asType(dtype)
                    assertRouterMatchesChain(ties, k: k, normalize: normalize, "ties \(tag)")

                    // All-equal rows: pure index-order selection.
                    let equal = MLX.full([rows, e], values: MLXArray(Float(0.25))).asType(dtype)
                    assertRouterMatchesChain(equal, k: k, normalize: normalize, "all-equal \(tag)")

                    // Signed zeros compare equal but differ bitwise; the tie
                    // class must merge and the surviving values keep their sign.
                    let signs = MLX.where(
                        MLXRandom.uniform(low: Float(0), high: 1, [rows, e]) .< 0.5,
                        MLXArray(Float(-0.0)), MLXArray(Float(0.0)))
                    let zeros = MLX.where(
                        MLXRandom.uniform(low: Float(0), high: 1, [rows, e]) .< 0.5,
                        signs, MLXRandom.normal([rows, e])
                    ).asType(dtype)
                    assertRouterMatchesChain(zeros, k: k, normalize: normalize, "±0 \(tag)")

                    // NaN rows: sort.h's LessThan puts NaN above everything
                    // and ties all NaNs; order must match, scores exempt.
                    var nans = MLX.where(
                        MLXRandom.uniform(low: Float(0), high: 1, [rows, e]) .< 0.05,
                        MLXArray(Float.nan), MLXRandom.normal([rows, e])
                    ).asType(dtype)
                    nans[0] = MLX.full([e], values: MLXArray(Float.nan)).asType(dtype)
                    assertRouterMatchesChain(
                        nans, k: k, normalize: normalize, "NaN \(tag)", indicesOnly: true)
                }
            }
        }
    }

    // MARK: - C16/C14: fused GDN decode body vs the unfused body

    /// Wide enough conv (256 channels) that an accumulation-order change in
    /// MLX's Convolution kernel cannot slip through by coincidence (the rig
    /// measured ~47% of channels diverging under native-dtype accumulation).
    private func tinyGDNConfiguration() throws -> Qwen35TextConfiguration {
        let json = """
            {
                "model_type": "qwen3_5_moe",
                "hidden_size": 64,
                "num_hidden_layers": 2,
                "intermediate_size": 64,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 32,
                "linear_num_value_heads": 4,
                "linear_num_key_heads": 2,
                "linear_key_head_dim": 32,
                "linear_value_head_dim": 32,
                "linear_conv_kernel_dim": 4,
                "vocab_size": 32,
                "full_attention_interval": 2,
                "num_experts": 16,
                "num_experts_per_tok": 4,
                "moe_intermediate_size": 32,
                "shared_expert_intermediate_size": 32
            }
            """
        return try JSONDecoder().decode(
            Qwen35TextConfiguration.self, from: Data(json.utf8))
    }

    func testGDNDecodeForwardMatchesUnfusedBody() throws {
        let config = try tinyGDNConfiguration()
        for dtype in [DType.float16, DType.bfloat16] {
            MLXRandom.seed(11)
            let gdn = Qwen35GatedDeltaNet(config)
            gdn.update(parameters: gdn.parameters().mapValues { $0.asType(dtype) })

            let x = MLXRandom.normal([1, 1, config.hiddenSize]).asType(dtype)
            let convState = MLXRandom.normal(
                [1, gdn.convKernelSize - 1, gdn.convDim]
            ).asType(dtype)
            let recState = MLXRandom.normal(
                [1, gdn.numVHeads, gdn.headVDim, gdn.headKDim])
            eval(x, convState, recState)

            // Fused: the body every compiled decode trace replays.
            let (out, newConv, newRec) = gdn.decodeForward(
                x: x, convState: convState, recState: recState)

            // Reference: the unfused callAsFunction body (conv1d included).
            let cache = MambaCache()
            cache[0] = convState
            cache[1] = recState
            let refOut = gdn(x, mask: nil, cache: cache)

            eval(out, newConv, newRec, refOut, cache[0]!, cache[1]!)

            assertBitIdentical(out, refOut, "GDN decode output (\(dtype))")
            assertBitIdentical(newConv, cache[0]!, "conv state (\(dtype))")
            assertBitIdentical(newRec, cache[1]!, "recurrent state (\(dtype))")
        }
    }
}
