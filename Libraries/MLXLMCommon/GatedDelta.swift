//
//  GatedDelta.swift
//  mlx-swift-lm
//
//  Port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/gated_delta.py
//

import Foundation
import MLX
import MLXNN

// MARK: - Compute G

/// Fused form of the decay gate chain — elementwise, and MLX `compile`
/// preserves per-node dtype rounding (verified bitwise against the unfused
/// chain on the real decode/prefill shapes, bf16 and f16), so this is
/// bit-identical while cutting ~6 kernel launches per GDN layer per step.
private let computeGatedDeltaG: @Sendable (MLXArray, MLXArray, MLXArray) -> MLXArray = compile(
    shapeless: true
) { aLog, a, dtBias in
    exp(-exp(aLog.asType(.float32)) * softplus(a + dtBias))
}

// MARK: - Metal Kernel

/// Which of the scan's inputs and outputs a kernel variant carries.
enum GatedDeltaKernelVariant: Hashable {
    /// `y` and the final state, optionally masked per step.
    case full(masked: Bool)
    /// `y` only: the final state is not stored (verify passes discard it).
    case outputOnly
    /// The final state only, over the first `valid` steps (`valid` is an
    /// int32 array input): the replay that rewinds a verify pass. With
    /// `conv` it also copies the replay's conv state — rows `valid ..<
    /// valid + K - 1` of the conv input block — so the commit is one launch.
    case stateAfterValid(conv: Bool)
}

private func makeGatedDeltaKernel(
    _ variant: GatedDeltaKernelVariant, fusedGates: Bool
) -> MLXFast.MLXFastKernel? {
    let readsQuery: Bool
    let writesState: Bool
    let stepGuard: String
    var inputNames: [String]
    let outputNames: [String]
    let baseName: String
    var copiesConv = false
    switch variant {
    case .full(let masked):
        readsQuery = true
        writesState = true
        stepGuard = masked ? "mask[b_idx * T + t]" : "true"
        inputNames = ["q", "k", "v", "g", "beta", "state_in", "T"] + (masked ? ["mask"] : [])
        outputNames = ["y", "state_out"]
        baseName = masked ? "gated_delta_step_mask" : "gated_delta_step"
    case .outputOnly:
        readsQuery = true
        writesState = false
        stepGuard = "true"
        inputNames = ["q", "k", "v", "g", "beta", "state_in", "T"]
        outputNames = ["y"]
        baseName = "gated_delta_step_y"
    case .stateAfterValid(let conv):
        readsQuery = false
        writesState = true
        stepGuard = "t < valid[0]"
        inputNames = ["k", "v", "g", "beta", "state_in", "T", "valid"] + (conv ? ["conv_src"] : [])
        outputNames = ["state_out"] + (conv ? ["conv_out"] : [])
        baseName = conv ? "gated_delta_step_state_valid_conv" : "gated_delta_step_state_valid"
        copiesConv = conv
    }
    // Fused gates read the pre-activations straight out of their projection
    // rows (`a_src[b, t, AOFF + h]`, `b_src[b, t, BOFF + h]`) plus the
    // layer's `A_log` and `dt_bias`, instead of precomputed `g`/`beta`.
    let gateInputs = fusedGates ? ["a_src", "b_src", "a_log", "dt_bias"] : ["g", "beta"]
    inputNames = inputNames.flatMap { $0 == "g" ? gateInputs : ($0 == "beta" ? [] : [$0]) }
    let name = fusedGates ? baseName + "_fg" : baseName

    let source = """
            auto n = thread_position_in_grid.z;
            auto b_idx = n / Hv;
            auto hv_idx = n % Hv;
            auto hk_idx = hv_idx / (Hv / Hk);
            constexpr int n_per_t = Dk / 32;
            // Rows per thread: each thread scans RPT value rows (dv_idx plus
            // multiples of Dv / RPT) as independent chains, the same lanes
            // and simd reductions per row as one row per thread.
            constexpr int row_step = Dv / RPT;

            // q, k: [B, T, Hk, Dk]
            \(readsQuery ? "auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;" : "")
            auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;

            // v, y: [B, T, Hv, Dv]
            auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
            \(readsQuery ? "y += b_idx * T * Hv * Dv + hv_idx * Dv;" : "")

            auto dk_idx = thread_position_in_threadgroup.x;
            auto dv_idx = thread_position_in_grid.y;

            \(fusedGates ? """
                // Gates per step from the projection columns, as the compiled
                // ops did: `a + dt_bias` in the input dtype, softplus as MLX's
                // LogAddExp in that dtype, then f32 `exp(-exp(A_log) * sp)`;
                // beta is MLX's sigmoid in the input dtype, widened.
                // One thread per step fills the threadgroup's gate table up
                // front (T <= GT, host-checked), so the scan's latency chain
                // never waits on a transcendental.
                threadgroup float g_tg[GT];
                threadgroup float beta_tg[GT];
                {
                  const uint tl = thread_index_in_threadgroup;
                  if (tl < (uint)T) {
                    // `auto`: MLX binds inputs under 8 elements in the constant address space.
                    auto a_row = a_src + ((size_t)b_idx * T + tl) * AROW + AOFF + hv_idx;
                    auto b_row = b_src + ((size_t)b_idx * T + tl) * BROW + BOFF + hv_idx;
                    const float neg_exp_alog = -metal::precise::exp(static_cast<float>(a_log[hv_idx]));
                    const InT dt_b = dt_bias[hv_idx];
                    // MLX's Add, LogAddExp (softplus) and Sigmoid in the input
                    // dtype, expression for expression (bf16 operators round
                    // per op; `log1p` is MLX's utils.h overload).
                    const InT a_t = a_row[0];
                    const InT sum = a_t + dt_b;
                    const InT zero = static_cast<InT>(0);
                    const InT maxv = (sum > zero) ? sum : zero;
                    const InT minv = (sum > zero) ? zero : sum;
                    InT sp;
                    if (metal::isinf(static_cast<float>(minv)) || metal::isinf(static_cast<float>(maxv))) {
                      sp = maxv;
                    } else {
                      sp = maxv + log1p(metal::exp(minv - maxv));
                    }
                    g_tg[tl] = metal::precise::exp(neg_exp_alog * static_cast<float>(sp));
                    const InT b_t = b_row[0];
                    auto yb = 1 / (1 + metal::exp(metal::abs(b_t)));
                    const InT sg = (b_t < 0) ? yb : 1 - yb;
                    beta_tg[tl] = static_cast<float>(sg);
                  }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                """ : """
                // g, beta: [B, T, Hv]
                auto g_ = g + b_idx * T * Hv;
                auto beta_ = beta + b_idx * T * Hv;
                """)

            \(copiesConv ? """
                // The replay's conv state: rows valid ..< valid + CK of the
                // conv input block [B, CS, CD], one element per thread (the
                // launch has more threads per batch than elements).
                {
                  const uint tid = ((uint)hv_idx * (uint)(Dv / RPT) + (uint)dv_idx) * 32u + (uint)dk_idx;
                  if (tid < (uint)(CK * CD)) {
                    const uint cr = tid / (uint)CD;
                    const uint cc = tid % (uint)CD;
                    conv_out[((size_t)b_idx * CK + cr) * CD + cc] =
                        conv_src[((size_t)b_idx * CS + (size_t)valid[0] + cr) * CD + cc];
                  }
                }
                """ : "")

            // state_in, state_out: [B, Hv, Dv, Dk]
            auto i_state = state_in + (n * Dv + dv_idx) * Dk;
            \(writesState ? "auto o_state = state_out + (n * Dv + dv_idx) * Dk;" : "")

            float state[RPT][n_per_t];
            for (int r = 0; r < RPT; ++r) {
              for (int i = 0; i < n_per_t; ++i) {
                auto s_idx = n_per_t * dk_idx + i;
                state[r][i] = static_cast<float>(i_state[r * row_step * Dk + s_idx]);
              }
            }

            for (int t = 0; t < T; ++t) {
              \(fusedGates ? """
                  const float g_t = g_tg[t];
                  const float beta_t = beta_tg[t];
                  """ : """
                  const float g_t = g_[hv_idx];
                  const float beta_t = beta_[hv_idx];
                  """)
              if (\(stepGuard)) {
                float kv_mem[RPT];
                for (int r = 0; r < RPT; ++r) {
                  // Preserve Kahan summation under Metal's default fast math.
                  #pragma clang fp reassociate(off)
                  #pragma clang fp contract(off)
                  float kv_compensation = 0.0f;
                  kv_mem[r] = 0.0f;
                  for (int i = 0; i < n_per_t; ++i) {
                    auto s_idx = n_per_t * dk_idx + i;
                    state[r][i] = state[r][i] * g_t;
                    auto product = state[r][i] * k_[s_idx];
                    auto corrected = product - kv_compensation;
                    auto next_sum = kv_mem[r] + corrected;
                    kv_compensation = (next_sum - kv_mem[r]) - corrected;
                    kv_mem[r] = next_sum;
                  }
                }
                for (int r = 0; r < RPT; ++r) {
                  kv_mem[r] = simd_sum(kv_mem[r]);
                }

                \(readsQuery ? "float out[RPT];" : "")
                for (int r = 0; r < RPT; ++r) {
                  auto delta = (v_[dv_idx + r * row_step] - kv_mem[r]) * beta_t;
                  \(readsQuery ? "out[r] = 0.0f;" : "")
                  for (int i = 0; i < n_per_t; ++i) {
                    auto s_idx = n_per_t * dk_idx + i;
                    state[r][i] = state[r][i] + k_[s_idx] * delta;
                    \(readsQuery ? "out[r] += state[r][i] * q_[s_idx];" : "")
                  }
                }
                \(readsQuery ? """
                    for (int r = 0; r < RPT; ++r) {
                      out[r] = simd_sum(out[r]);
                    }
                    if (thread_index_in_simdgroup == 0) {
                      for (int r = 0; r < RPT; ++r) {
                        y[dv_idx + r * row_step] = static_cast<InT>(out[r]);
                      }
                    }
                    """ : "")
              } else {
                \(readsQuery ? """
                    for (int r = 0; r < RPT; ++r) {
                      y[dv_idx + r * row_step] = static_cast<InT>(0);
                    }
                    """ : "")
              }
              // Increment data pointers to next time step
              \(readsQuery ? "q_ += Hk * Dk;" : "")
              k_ += Hk * Dk;
              v_ += Hv * Dv;
              \(readsQuery ? "y += Hv * Dv;" : "")
              \(fusedGates ? "" : "g_ += Hv; beta_ += Hv;")
            }
            \(writesState ? """
                for (int r = 0; r < RPT; ++r) {
                  for (int i = 0; i < n_per_t; ++i) {
                    auto s_idx = n_per_t * dk_idx + i;
                    o_state[r * row_step * Dk + s_idx] = static_cast<StT>(state[r][i]);
                  }
                }
                """ : "")
        """

    return MLXFast.metalKernel(
        name: name,
        inputNames: inputNames,
        outputNames: outputNames,
        source: source
    )
}

private final class GatedDeltaKernelManager: Sendable {
    static let shared = GatedDeltaKernelManager()

    let kernel: MLXFast.MLXFastKernel?
    let kernelMasked: MLXFast.MLXFastKernel?
    let kernelOutputOnly: MLXFast.MLXFastKernel?
    let kernelStateAfterValid: MLXFast.MLXFastKernel?
    let kernelStateAfterValidConv: MLXFast.MLXFastKernel?
    let fusedGateKernel: MLXFast.MLXFastKernel?
    let fusedGateKernelMasked: MLXFast.MLXFastKernel?
    let fusedGateKernelOutputOnly: MLXFast.MLXFastKernel?
    let fusedGateKernelStateAfterValid: MLXFast.MLXFastKernel?
    let fusedGateKernelStateAfterValidConv: MLXFast.MLXFastKernel?

    private init() {
        kernel = makeGatedDeltaKernel(.full(masked: false), fusedGates: false)
        kernelMasked = makeGatedDeltaKernel(.full(masked: true), fusedGates: false)
        kernelOutputOnly = makeGatedDeltaKernel(.outputOnly, fusedGates: false)
        kernelStateAfterValid = makeGatedDeltaKernel(
            .stateAfterValid(conv: false), fusedGates: false)
        kernelStateAfterValidConv = makeGatedDeltaKernel(
            .stateAfterValid(conv: true), fusedGates: false)
        fusedGateKernel = makeGatedDeltaKernel(.full(masked: false), fusedGates: true)
        fusedGateKernelMasked = makeGatedDeltaKernel(.full(masked: true), fusedGates: true)
        fusedGateKernelOutputOnly = makeGatedDeltaKernel(.outputOnly, fusedGates: true)
        fusedGateKernelStateAfterValid = makeGatedDeltaKernel(
            .stateAfterValid(conv: false), fusedGates: true)
        fusedGateKernelStateAfterValidConv = makeGatedDeltaKernel(
            .stateAfterValid(conv: true), fusedGates: true)
    }

    func kernel(for variant: GatedDeltaKernelVariant, fusedGates: Bool) -> MLXFast.MLXFastKernel? {
        switch (variant, fusedGates) {
        case (.full(let masked), false): masked ? kernelMasked : kernel
        case (.outputOnly, false): kernelOutputOnly
        case (.stateAfterValid(let conv), false):
            conv ? kernelStateAfterValidConv : kernelStateAfterValid
        case (.full(let masked), true): masked ? fusedGateKernelMasked : fusedGateKernel
        case (.outputOnly, true): fusedGateKernelOutputOnly
        case (.stateAfterValid(let conv), true):
            conv ? fusedGateKernelStateAfterValidConv : fusedGateKernelStateAfterValid
        }
    }
}

/// Steps a fused-gate launch covers: the threadgroup's 128 threads fill one
/// gate table entry each. Longer passes (prefill) use precomputed gates.
let gatedDeltaFusedGateSteps = 128

/// Where a scan reads its gates. `.source` hands the kernel the `a`/`b`
/// pre-activations as column offsets into their `[B, S, row]` arrays (the
/// fused input projection, or standalone `[B, S, Hv]` arrays at offset 0)
/// plus the layer's `A_log`/`dt_bias`, and no gate array is materialised;
/// `.precomputed` is the `[B, S, Hv]` f32 pair from ``gatedDeltaGates(a:b:aLog:dtBias:)``.
public enum GatedDeltaGates {
    case precomputed(g: MLXArray, beta: MLXArray)
    case source(GatedDeltaGateSource)

    /// The `g`/`beta` pair, computed by the ops when the gates are a source.
    public var materialized: (g: MLXArray, beta: MLXArray) {
        switch self {
        case .precomputed(let g, let beta): (g, beta)
        case .source(let source):
            gatedDeltaGates(a: source.a, b: source.b, aLog: source.aLog, dtBias: source.dtBias)
        }
    }
}

public struct GatedDeltaGateSource {
    public var aSource: MLXArray
    public var aOffset: Int
    public var bSource: MLXArray
    public var bOffset: Int
    /// `[Hv]`, any float dtype (widened exactly).
    public var aLog: MLXArray
    /// `[Hv]`, the activation dtype (the ops add it to `a` in that dtype).
    public var dtBias: MLXArray

    public init(
        aSource: MLXArray, aOffset: Int, bSource: MLXArray, bOffset: Int, aLog: MLXArray,
        dtBias: MLXArray
    ) {
        self.aSource = aSource
        self.aOffset = aOffset
        self.bSource = bSource
        self.bOffset = bOffset
        self.aLog = aLog
        self.dtBias = dtBias
    }

    /// Standalone `[B, S, Hv]` pre-activations.
    public init(a: MLXArray, b: MLXArray, aLog: MLXArray, dtBias: MLXArray) {
        self.init(aSource: a, aOffset: 0, bSource: b, bOffset: 0, aLog: aLog, dtBias: dtBias)
    }

    public var heads: Int { aLog.dim(0) }
    public var a: MLXArray { aSource[.ellipsis, aOffset ..< (aOffset + heads)] }
    public var b: MLXArray { bSource[.ellipsis, bOffset ..< (bOffset + heads)] }

    /// The source over a range of sequence rows (axis 1 of both arrays).
    public func rows(_ range: some RangeExpression<Int>) -> GatedDeltaGateSource {
        let r = range.relative(to: 0 ..< aSource.dim(1))
        var copy = self
        copy.aSource = aSource[0..., r, 0...]
        copy.bSource = bSource[0..., r, 0...]
        return copy
    }

    /// Whether the kernel can compute the gates from this source with the
    /// ops' rounding: the activation dtype throughout, rows to index into.
    func servesKernel(inputType: DType) -> Bool {
        aSource.dtype == inputType && bSource.dtype == inputType && dtBias.dtype == inputType
            && aSource.ndim == 3 && bSource.ndim == 3 && aLog.ndim == 1 && dtBias.ndim == 1
            && aOffset + heads <= aSource.dim(2) && bOffset + heads <= bSource.dim(2)
            && [DType.float32, .float16, .bfloat16].contains(aLog.dtype)
    }
}

/// Launch one kernel variant. `extra` is the mask (`.full(masked: true)`) or
/// the int32 `valid` count (`.stateAfterValid`); `convInput` the `[B, S +
/// K - 1, D]` conv block a `.stateAfterValid(conv: true)` launch copies its
/// `K - 1` rows from.
private func gatedDeltaKernel(
    _ variant: GatedDeltaKernelVariant,
    q: MLXArray?,
    k: MLXArray,
    v: MLXArray,
    gates: GatedDeltaGates,
    state: MLXArray,
    extra: MLXArray? = nil,
    convInput: MLXArray? = nil,
    rowsPerThread: Int = gatedDeltaRowsPerThread
) -> [MLXArray] {
    let B = k.dim(0)
    let T = k.dim(1)
    let Hk = k.dim(2)
    let Dk = k.dim(3)
    let Hv = v.dim(2)
    let Dv = v.dim(3)
    let inputType = k.dtype
    let stateType = state.dtype
    // Rows per thread must divide Dv (the grid is Dv / RPT simdgroups per
    // head); odd or tiny value dims take one row per thread.
    let rowsPerThread = Dv % rowsPerThread == 0 ? rowsPerThread : 1

    // The gate inputs and their template entries; a source the kernel
    // cannot read exactly falls back to the ops' precomputed pair.
    let gateInputs: [MLXArray]
    var gateTemplate: [(String, any KernelTemplateArg)] = []
    var fusedGates = false
    switch gates {
    case .source(let source)
    where source.servesKernel(inputType: inputType) && T <= gatedDeltaFusedGateSteps:
        gateInputs = [source.aSource, source.bSource, source.aLog, source.dtBias]
        gateTemplate = [
            ("AOFF", source.aOffset), ("AROW", source.aSource.dim(2)),
            ("BOFF", source.bOffset), ("BROW", source.bSource.dim(2)),
            ("GT", gatedDeltaFusedGateSteps),
        ]
        fusedGates = true
    default:
        let (g, beta) = gates.materialized
        gateInputs = [g, beta]
    }

    guard
        let selectedKernel = GatedDeltaKernelManager.shared.kernel(
            for: variant, fusedGates: fusedGates)
    else {
        fatalError("gated delta kernel unavailable")
    }
    var inputs: [MLXArray] = []
    var outputShapes: [[Int]] = []
    var outputDTypes: [DType] = []
    switch variant {
    case .full(let masked):
        inputs = [q!, k, v] + gateInputs + [state, MLXArray(T)]
        if masked { inputs.append(extra!) }
        outputShapes = [[B, T, Hv, Dv], state.shape]
        outputDTypes = [inputType, stateType]
    case .outputOnly:
        inputs = [q!, k, v] + gateInputs + [state, MLXArray(T)]
        outputShapes = [[B, T, Hv, Dv]]
        outputDTypes = [inputType]
    case .stateAfterValid(let conv):
        inputs = [k, v] + gateInputs + [state, MLXArray(T), extra!.asType(.int32).reshaped([1])]
        outputShapes = [state.shape]
        outputDTypes = [stateType]
        if conv {
            let convInput = convInput!
            let kernelRows = convInput.dim(1) - T
            inputs.append(convInput)
            outputShapes.append([B, kernelRows, convInput.dim(2)])
            outputDTypes.append(convInput.dtype)
            gateTemplate += [
                ("CK", kernelRows), ("CS", convInput.dim(1)), ("CD", convInput.dim(2)),
            ]
        }
    }

    return selectedKernel(
        inputs,
        template: [
            ("InT", inputType),
            ("StT", stateType),
            ("Dk", Dk),
            ("Dv", Dv),
            ("Hk", Hk),
            ("Hv", Hv),
            ("RPT", rowsPerThread),
        ] + gateTemplate,
        grid: (32, Dv / rowsPerThread, B * Hv),
        threadGroup: (32, 4, 1),
        outputShapes: outputShapes,
        outputDTypes: outputDTypes
    )
}

/// The fused kernel: `y` and the final state.
func gatedDeltaKernel(
    q: MLXArray,
    k: MLXArray,
    v: MLXArray,
    gates: GatedDeltaGates,
    state: MLXArray,
    mask: MLXArray? = nil
) -> (MLXArray, MLXArray) {
    let outputs = gatedDeltaKernel(
        .full(masked: mask != nil), q: q, k: k, v: v, gates: gates, state: state,
        extra: mask)
    return (outputs[0], outputs[1])
}

// MARK: - Ops Fallback

private func gatedDeltaStepOps(
    q: MLXArray,
    k: MLXArray,
    v: MLXArray,
    g: MLXArray,
    beta: MLXArray,
    state: MLXArray,
    mask: MLXArray? = nil
) -> (MLXArray, MLXArray) {
    let oldState = state
    let decay: MLXArray
    if g.ndim == 2 {
        decay = expandedDimensions(g, axes: [2, 3])
    } else if g.ndim == 3 {
        decay = expandedDimensions(g, axis: -2)
    } else {
        fatalError("Unsupported gating shape \(g.shape)")
    }

    var state = state * decay
    let kvMem = (state * expandedDimensions(k, axis: -2)).sum(axis: -1)
    let delta = (v - kvMem) * expandedDimensions(beta, axis: -1)
    state = state + expandedDimensions(k, axis: -2) * expandedDimensions(delta, axis: -1)
    let y = (state * expandedDimensions(q, axis: -2)).sum(axis: -1)

    if let mask {
        let expandedMask: MLXArray
        if mask.ndim == 1 {
            expandedMask = expandedDimensions(mask, axes: [1, 2, 3])
        } else if mask.ndim == 2 {
            expandedMask = expandedDimensions(mask, axes: [2, 3])
        } else if mask.ndim == 3 {
            expandedMask = expandedDimensions(mask, axis: -1)
        } else {
            fatalError("Unsupported mask shape \(mask.shape)")
        }
        state = MLX.where(expandedMask, state, oldState)
    }

    return (y.asType(q.dtype), state)
}

func gatedDeltaOps(
    q: MLXArray,
    k: MLXArray,
    v: MLXArray,
    g: MLXArray,
    beta: MLXArray,
    state: MLXArray? = nil,
    mask: MLXArray? = nil
) -> (MLXArray, MLXArray) {
    let B = q.dim(0)
    let T = q.dim(1)
    let Hk = q.dim(2)
    let Dk = q.dim(3)
    let Hv = v.dim(2)
    let Dv = v.dim(3)

    var q = q
    var k = k

    let repeatFactor = Hv / Hk
    if repeatFactor > 1 {
        q = repeated(q, count: repeatFactor, axis: -2)
        k = repeated(k, count: repeatFactor, axis: -2)
    }

    var state = state ?? MLXArray.zeros([B, Hv, Dv, Dk], dtype: .float32)

    var ys = [MLXArray]()
    ys.reserveCapacity(T)

    for t in 0 ..< T {
        let qT = q[0..., t]
        let kT = k[0..., t]
        let vT = v[0..., t]
        let gT = g[0..., t]
        let betaT = beta[0..., t]
        let maskT = mask == nil ? nil : mask![0..., t]

        let (y, newState) = gatedDeltaStepOps(
            q: qT,
            k: kT,
            v: vT,
            g: gT,
            beta: betaT,
            state: state,
            mask: maskT
        )
        ys.append(y)
        state = newState
    }

    let y = MLX.stacked(ys, axis: 1)
    return (y, state)
}

// MARK: - Public API

/// The gates the scan consumes, both f32: `g = exp(-exp(A_log) * softplus(a + dt_bias))`
/// and `beta = sigmoid(b)`. Computed once per pass; a verify capture carries
/// them so its replay never recomputes them.
public func gatedDeltaGates(
    a: MLXArray, b: MLXArray, aLog: MLXArray, dtBias: MLXArray
) -> (g: MLXArray, beta: MLXArray) {
    (computeGatedDeltaG(aLog, a, dtBias), sigmoid(b).asType(.float32))
}

private func usesFusedKernel(keyDimension: Int) -> Bool {
    GatedDeltaKernelManager.shared.kernel != nil && keyDimension % 32 == 0
}

private func float32State(_ state: MLXArray?, batch: Int, v: MLXArray, k: MLXArray) -> MLXArray {
    // State kept in fp32 to match Python mlx-lm. Using q.dtype (bf16) loses
    // precision across T-step recurrence, compounding rounding error.
    let state = state ?? MLXArray.zeros([batch, v.dim(2), v.dim(3), k.dim(3)], dtype: .float32)
    return state.dtype == .float32 ? state : state.asType(.float32)
}

public func gatedDeltaUpdate(
    q: MLXArray,
    k: MLXArray,
    v: MLXArray,
    a: MLXArray,
    b: MLXArray,
    aLog: MLXArray,
    dtBias: MLXArray,
    state: MLXArray? = nil,
    mask: MLXArray? = nil
) -> (MLXArray, MLXArray) {
    let (g, beta) = gatedDeltaGates(a: a, b: b, aLog: aLog, dtBias: dtBias)
    return gatedDeltaUpdate(
        q: q, k: k, v: v, gates: .precomputed(g: g, beta: beta), state: state, mask: mask)
}

/// ``gatedDeltaUpdate(q:k:v:a:b:aLog:dtBias:state:mask:)`` with the gates as
/// ``GatedDeltaGates``: a `.source` lets the kernel compute them per step
/// (bitwise the ops' values; no gate launches, no gate arrays).
public func gatedDeltaUpdate(
    q: MLXArray,
    k: MLXArray,
    v: MLXArray,
    gates: GatedDeltaGates,
    state: MLXArray? = nil,
    mask: MLXArray? = nil
) -> (MLXArray, MLXArray) {
    let state = float32State(state, batch: q.dim(0), v: v, k: k)

    // The fused kernel distributes Dk over exactly 32 simd lanes
    // (`n_per_t = Dk / 32`, integer-truncating). A key head dim that is not a
    // multiple of 32 would silently drop its trailing `Dk % 32` state
    // dimensions and produce wrong output with no error. Stock Qwen3.5 uses
    // Dk = 192 (a multiple of 32), but the value is config-driven, so route any
    // non-multiple-of-32 Dk to the ops fallback, which handles an arbitrary key
    // dimension correctly (slower, but never truncating).
    if usesFusedKernel(keyDimension: q.dim(3)) {
        return gatedDeltaKernel(q: q, k: k, v: v, gates: gates, state: state, mask: mask)
    }
    let (g, beta) = gates.materialized
    return gatedDeltaOps(q: q, k: k, v: v, g: g, beta: beta, state: state, mask: mask)
}

/// The scan's output alone, from precomputed gates; the final state is never
/// stored. Same arithmetic as ``gatedDeltaUpdate(q:k:v:a:b:aLog:dtBias:state:mask:)``.
public func gatedDeltaOutput(
    q: MLXArray, k: MLXArray, v: MLXArray, g: MLXArray, beta: MLXArray, state: MLXArray
) -> MLXArray {
    gatedDeltaOutput(q: q, k: k, v: v, gates: .precomputed(g: g, beta: beta), state: state)
}

public func gatedDeltaOutput(
    q: MLXArray, k: MLXArray, v: MLXArray, gates: GatedDeltaGates, state: MLXArray
) -> MLXArray {
    let state = float32State(state, batch: q.dim(0), v: v, k: k)
    if usesFusedKernel(keyDimension: q.dim(3)) {
        return gatedDeltaKernel(.outputOnly, q: q, k: k, v: v, gates: gates, state: state)[0]
    }
    let (g, beta) = gates.materialized
    return gatedDeltaOps(q: q, k: k, v: v, g: g, beta: beta, state: state).0
}

/// Value rows each scan thread carries: two independent chains per thread
/// hide the step latency (bitwise the one-row geometry; travel 66.2 vs 66.4
/// ms/round). The arithmetic per row is identical for any value that
/// divides the value head dim.
let gatedDeltaRowsPerThread = 2

/// `gatedDeltaOutput` and the replay state with an explicit rows-per-thread
/// geometry, for the parity microbench.
public func gatedDeltaOutputVariant(
    q: MLXArray, k: MLXArray, v: MLXArray, gates: GatedDeltaGates, state: MLXArray,
    rowsPerThread: Int
) -> MLXArray {
    gatedDeltaKernel(
        .outputOnly, q: q, k: k, v: v, gates: gates, state: state,
        rowsPerThread: rowsPerThread)[0]
}

public func gatedDeltaStateAfterVariant(
    validCount: MLXArray, k: MLXArray, v: MLXArray, gates: GatedDeltaGates,
    state: MLXArray, rowsPerThread: Int
) -> MLXArray {
    gatedDeltaKernel(
        .stateAfterValid(conv: false), q: nil, k: k, v: v, gates: gates, state: state,
        extra: validCount, rowsPerThread: rowsPerThread)[0]
}

/// ``gatedDeltaStateAfter(validCount:k:v:gates:state:)`` plus the replay's
/// conv state, rows `validCount ..< validCount + K - 1` of `convInput`
/// `[B, S + K - 1, D]`, copied by the same launch when the fused kernel
/// runs (its threads outnumber the conv elements), else by a dynamic slice.
public func gatedDeltaStateAfter(
    validCount: MLXArray, k: MLXArray, v: MLXArray, gates: GatedDeltaGates, state: MLXArray,
    convInput: MLXArray
) -> (state: MLXArray, conv: MLXArray) {
    let B = k.dim(0)
    let T = k.dim(1)
    let kernelRows = convInput.dim(1) - T
    let threadsPerBatch = 32 * (v.dim(3) / gatedDeltaRowsPerThread) * v.dim(2)
    if usesFusedKernel(keyDimension: k.dim(3)), convInput.dtype == k.dtype, convInput.ndim == 3,
        convInput.dim(0) == B, kernelRows >= 1, kernelRows * convInput.dim(2) <= threadsPerBatch
    {
        let outputs = gatedDeltaKernel(
            .stateAfterValid(conv: true), q: nil, k: k, v: v, gates: gates,
            state: float32State(state, batch: B, v: v, k: k), extra: validCount,
            convInput: convInput)
        return (outputs[0], outputs[1])
    }
    return (
        gatedDeltaStateAfter(validCount: validCount, k: k, v: v, gates: gates, state: state),
        dynamicSlice(
            convInput, start: validCount.asType(.int32).reshaped([1]), axes: [1],
            sliceSize: [B, kernelRows, convInput.dim(2)])
    )
}

/// The state after the first `validCount` steps (an int32 array, possibly
/// lazy), from precomputed gates. Steps past `validCount` leave the state
/// untouched, so the result equals a prefix scan for every count. Same
/// arithmetic as the forward kernel; the query reduction is omitted.
public func gatedDeltaStateAfter(
    validCount: MLXArray, k: MLXArray, v: MLXArray, g: MLXArray, beta: MLXArray, state: MLXArray
) -> MLXArray {
    gatedDeltaStateAfter(
        validCount: validCount, k: k, v: v, gates: .precomputed(g: g, beta: beta), state: state)
}

public func gatedDeltaStateAfter(
    validCount: MLXArray, k: MLXArray, v: MLXArray, gates: GatedDeltaGates, state: MLXArray
) -> MLXArray {
    let state = float32State(state, batch: k.dim(0), v: v, k: k)
    if usesFusedKernel(keyDimension: k.dim(3)) {
        return gatedDeltaKernel(
            .stateAfterValid(conv: false), q: nil, k: k, v: v, gates: gates, state: state,
            extra: validCount)[0]
    }
    let (g, beta) = gates.materialized
    let mask = (MLXArray(Int32(0) ..< Int32(k.dim(1))) .< validCount.asType(.int32))
        .expandedDimensions(axis: 0)
    let q = MLXArray.zeros(k.shape, dtype: k.dtype)
    return gatedDeltaOps(q: q, k: k, v: v, g: g, beta: beta, state: state, mask: mask).1
}
