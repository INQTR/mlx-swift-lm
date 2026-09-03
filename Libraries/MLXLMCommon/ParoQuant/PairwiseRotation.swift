import Foundation
import MLX
import MLXNN

// MARK: - Metal Kernel Sources
//
// Both kernels are template kernels: `T` (activation dtype), `ROWS_PER_TILE`,
// `KROT` and — for the generic one — `GROUP_SIZE` arrive as MLX template
// arguments, so MLX instantiates and memoizes one specialisation per
// (dtype, geometry) itself, the same way the router top-k and GatedDelta
// kernels in this module work.

/// Pairwise Givens rotation kernel for Metal (Apple Silicon), groupSize 128.
///
/// One CTA is a single simdgroup (32 lanes) per (row-tile, channel-group):
///
/// - Each lane caches the cos/sin/pair coefficients of its two pair slots
///   (lane, lane+32) for every round in registers. `KROT` is a compile-time
///   constant, so these are constant-index register arrays — a runtime loop
///   bound would push them into local (DRAM-backed) memory.
/// - Per-round sync is `simdgroup_barrier(mem_threadgroup)` instead of
///   `threadgroup_barrier` — with a one-simdgroup CTA there is no
///   cross-simdgroup rendezvous to pay for. The tile layout is row-major
///   (`tile[row * 128 + ch]`), so pair accesses are bank-conflict-free
///   for any ROWS_PER_TILE (a channel-major `tile[ch * R + row]` layout
///   collapses onto 8 banks for R = 4).
/// - x/out IO is vectorized (`vec<T, 4>` per lane covers the 128-channel
///   group exactly); the f32 threadgroup tile is written/read as float4.
///   `channel_scales` is loaded scalar + converted so its dtype may
///   legitimately differ from the activation dtype.
/// - The write-back casts explicitly (`T(...)`), which is what lets the same
///   source instantiate for float16, bfloat16 and float32.
///
/// Correctness notes:
/// - All lanes execute every barrier (no early returns; `row < batch_size`
///   guards wrap memory accesses only and are CTA-uniform).
/// - The math is bit-identical to the generic kernel per element: f32 loads
///   of `float(x) * scale`, the same krot Givens rounds in order with the
///   same pairs/cos/sin (`a * c + b * s`, `b * c - a * s` in f32), then one
///   rounding to the element type on write-back.
/// - Assumes groupSize == 128 (64 pair slots per group = 2 per lane);
///   `dispatchPairwiseRotation` selects this kernel only for that size.
private let simdgroupRotationSource = """
    constexpr int GROUP_SIZE = 128;

    // `x_shape` is MLX's auto-injected shape buffer for input `x` ([batch, dim]).
    const int batch_size  = x_shape[0];
    const int hidden_size = x_shape[1];

    const int half_gs     = GROUP_SIZE / 2;
    const int half_hidden = hidden_size / 2;

    const int tile_idx  = threadgroup_position_in_grid.x;
    const int group_idx = threadgroup_position_in_grid.y;
    const int lane      = thread_index_in_threadgroup;
    const int gbase     = group_idx * GROUP_SIZE;

    // Rotation coefficients for this lane's two pair slots of every round
    float cos_vals[KROT][2], sin_vals[KROT][2];
    int   pair_vals[KROT][2];

    for (int k = 0; k < KROT; k++) {
        for (int u = 0; u < 2; u++) {
            int idx = k * half_hidden + group_idx * half_gs + lane + u * 32;
            cos_vals[k][u]  = float(cos_theta[idx]);
            sin_vals[k][u]  = float(sin_theta[idx]);
            pair_vals[k][u] = int(packed_pairs[idx]);
        }
    }

    threadgroup float tile[ROWS_PER_TILE * GROUP_SIZE];

    // Load activation tile into shared memory (fuse channel scales).
    // Lane owns channels lane*4 .. lane*4+3 of the group.
    float sc0 = float(channel_scales[gbase + lane * 4 + 0]);
    float sc1 = float(channel_scales[gbase + lane * 4 + 1]);
    float sc2 = float(channel_scales[gbase + lane * 4 + 2]);
    float sc3 = float(channel_scales[gbase + lane * 4 + 3]);
    for (int r = 0; r < ROWS_PER_TILE; r++) {
        int row = tile_idx * ROWS_PER_TILE + r;
        if (row < batch_size) {
            vec<T, 4> xh = ((const device vec<T, 4>*)(x + row * hidden_size + gbase))[lane];
            float4 tv;
            tv[0] = float(xh[0]) * sc0;
            tv[1] = float(xh[1]) * sc1;
            tv[2] = float(xh[2]) * sc2;
            tv[3] = float(xh[3]) * sc3;
            *(threadgroup float4*)(tile + r * GROUP_SIZE + lane * 4) = tv;
        }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    // Apply pairwise Givens rotations in-place
    for (int k = 0; k < KROT; k++) {
        for (int u = 0; u < 2; u++) {
            int i_local = pair_vals[k][u] & 0xFFFF;
            int j_local = pair_vals[k][u] >> 16;
            float c = cos_vals[k][u], s = sin_vals[k][u];
            for (int m = 0; m < ROWS_PER_TILE; m++) {
                float a = tile[m * GROUP_SIZE + i_local];
                float b = tile[m * GROUP_SIZE + j_local];
                tile[m * GROUP_SIZE + i_local] = a * c + b * s;
                tile[m * GROUP_SIZE + j_local] = b * c - a * s;
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write results back
    for (int r = 0; r < ROWS_PER_TILE; r++) {
        int row = tile_idx * ROWS_PER_TILE + r;
        if (row < batch_size) {
            float4 tv = *(threadgroup float4*)(tile + r * GROUP_SIZE + lane * 4);
            vec<T, 4> ov;
            ov[0] = T(tv[0]);
            ov[1] = T(tv[1]);
            ov[2] = T(tv[2]);
            ov[3] = T(tv[3]);
            *(device vec<T, 4>*)(out + row * hidden_size + gbase + lane * 4) = ov;
        }
    }
    """

/// Generic rotation kernel for any other group size: one thread per pair
/// slot (`GROUP_SIZE / 2` per CTA), a full CTA barrier per round, and a
/// channel-major threadgroup tile. Same per-element math as the simdgroup
/// kernel.
private let genericRotationSource = """
    // `x_shape` is MLX's auto-injected shape buffer for input `x` ([batch, dim]).
    const int batch_size  = x_shape[0];
    const int hidden_size = x_shape[1];

    const int half_gs     = GROUP_SIZE / 2;
    const int half_hidden = hidden_size / 2;

    const int tile_idx  = threadgroup_position_in_grid.x;
    const int group_idx = threadgroup_position_in_grid.y;
    const int tid       = thread_index_in_threadgroup;

    // Load rotation coefficients into registers
    float cos_vals[KROT], sin_vals[KROT];
    int   pair_vals[KROT];

    for (int k = 0; k < KROT; k++) {
        int idx = k * half_hidden + group_idx * half_gs + tid;
        cos_vals[k]  = float(cos_theta[idx]);
        sin_vals[k]  = float(sin_theta[idx]);
        pair_vals[k] = int(packed_pairs[idx]);
    }

    // Load activation tile into shared memory (fuse channel scales)
    threadgroup float tile[GROUP_SIZE * ROWS_PER_TILE];

    const int ch_lo = group_idx * GROUP_SIZE + tid;
    const int ch_hi = ch_lo + half_gs;
    float scale_lo = float(channel_scales[ch_lo]);
    float scale_hi = float(channel_scales[ch_hi]);

    for (int r = 0; r < ROWS_PER_TILE; r++) {
        int row = tile_idx * ROWS_PER_TILE + r;
        if (row < batch_size) {
            tile[tid * ROWS_PER_TILE + r]              = float(x[row * hidden_size + ch_lo]) * scale_lo;
            tile[(tid + half_gs) * ROWS_PER_TILE + r]  = float(x[row * hidden_size + ch_hi]) * scale_hi;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Apply pairwise Givens rotations in-place
    for (int k = 0; k < KROT; k++) {
        int i_local = pair_vals[k] & 0xFFFF;
        int j_local = pair_vals[k] >> 16;
        float c = cos_vals[k], s = sin_vals[k];

        for (int m = 0; m < ROWS_PER_TILE; m++) {
            float a = tile[i_local * ROWS_PER_TILE + m];
            float b = tile[j_local * ROWS_PER_TILE + m];
            tile[i_local * ROWS_PER_TILE + m] = a * c + b * s;
            tile[j_local * ROWS_PER_TILE + m] = b * c - a * s;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write results back
    for (int r = 0; r < ROWS_PER_TILE; r++) {
        int row = tile_idx * ROWS_PER_TILE + r;
        if (row < batch_size) {
            out[row * hidden_size + ch_lo] = T(tile[tid * ROWS_PER_TILE + r]);
            out[row * hidden_size + ch_hi] = T(tile[(tid + half_gs) * ROWS_PER_TILE + r]);
        }
    }
    """

/// The two rotation kernels, compiled once per process; MLX memoizes each
/// template instantiation behind them.
private final class RotationKernels: Sendable {
    static let shared = RotationKernels()
    let simdgroup: MLXFast.MLXFastKernel
    let generic: MLXFast.MLXFastKernel

    private init() {
        let inputNames = ["x", "packed_pairs", "cos_theta", "sin_theta", "channel_scales"]
        simdgroup = MLXFast.metalKernel(
            name: "paro_rotate", inputNames: inputNames, outputNames: ["out"],
            source: simdgroupRotationSource)
        generic = MLXFast.metalKernel(
            name: "paro_rotate_generic", inputNames: inputNames, outputNames: ["out"],
            source: genericRotationSource)
    }
}

// MARK: - Geometry

/// Largest group size the generic kernel accepts. One thread per pair slot
/// (Metal's 1024-thread CTA ceiling) and a 4-row f32 tile within the 32 KB
/// threadgroup-memory ceiling both allow 2048; 1024 keeps either bound at
/// half headroom.
private let maxGenericGroupSize = 1024

/// Validate a rotation's (dims, groupSize, krot) triple against what the
/// kernels assume, returning a reason on failure. The one silent case worth
/// naming: a `dims` that is not a whole number of groups would leave the
/// trailing partial group un-rotated (`numGroups` floors). The loader
/// surfaces a failure as a typed error with the key path; the module
/// initializers assert it for direct callers.
nonisolated func rotationGeometryProblem(dims: Int, groupSize: Int, krot: Int) -> String? {
    if krot < 1 {
        return "krot must be >= 1 (got \(krot))"
    }
    if groupSize < 2 || !groupSize.isMultiple(of: 2) {
        return "groupSize must be even and >= 2 (got \(groupSize))"
    }
    if groupSize > maxGenericGroupSize {
        return "groupSize must be <= \(maxGenericGroupSize) (got \(groupSize))"
    }
    if dims < groupSize || !dims.isMultiple(of: groupSize) {
        return
            "dims must be a positive multiple of groupSize (got dims \(dims), groupSize \(groupSize))"
    }
    return nil
}

/// `precondition` form of `rotationGeometryProblem`.
nonisolated func assertRotationGeometry(dims: Int, groupSize: Int, krot: Int) {
    if let problem = rotationGeometryProblem(dims: dims, groupSize: groupSize, krot: krot) {
        preconditionFailure("Pairwise rotation: \(problem)")
    }
}

// MARK: - Dispatch

/// Dispatch the pairwise rotation on a 2-D `[batch, dim]` activation.
///
/// groupSize == 128 takes the simdgroup-resident kernel (2 pair slots per
/// lane, no CTA rendezvous); any other groupSize the generic kernel. Both
/// instantiate for float16, bfloat16 and float32 activations. Shared by
/// `PairwiseRotation` and `RotateQuantizedLinear`, whose initializers have
/// already validated the geometry.
///
/// Zero-row inputs pass straight through: gathered MoE activations can be
/// legitimately empty, and a zero-sized grid dispatch is undefined. The
/// guard lives here, with the grid math, so no caller needs one.
nonisolated func dispatchPairwiseRotation(
    _ flat: MLXArray, state: RotationDerivedState, groupSize: Int, krot: Int
) -> MLXArray {
    let batch = flat.dim(0)
    if batch == 0 { return flat }
    precondition(
        flat.dtype == .float16 || flat.dtype == .bfloat16 || flat.dtype == .float32,
        "Pairwise rotation: unsupported activation dtype \(flat.dtype)")

    let dim = state.scalesFlat.dim(0)
    let numGroups = dim / groupSize
    let tile = batch <= 1 ? 1 : 4

    let kernels = RotationKernels.shared
    let (kernel, threads) =
        groupSize == 128 ? (kernels.simdgroup, 32) : (kernels.generic, groupSize / 2)
    var template: [(String, any KernelTemplateArg)] = [
        ("T", flat.dtype), ("ROWS_PER_TILE", tile), ("KROT", krot),
    ]
    if groupSize != 128 {
        template.append(("GROUP_SIZE", groupSize))
    }

    let gridX = ((batch + tile - 1) / tile) * threads
    return kernel(
        [flat, state.packedPairs, state.cosTheta, state.sinTheta, state.scalesFlat],
        template: template,
        grid: (gridX, numGroups, 1),
        threadGroup: (threads, 1, 1),
        outputShapes: [flat.shape],
        outputDTypes: [flat.dtype]
    )[0]
}

// MARK: - Pair Packing

/// Pack int16 pair indices into int32 for the Metal kernel.
///
/// Each pair `(i, j)` is packed as `i | (j << 16)` within each group.
///
/// Internal (not file-private): shared by `PairwiseRotation` and
/// `RotateQuantizedLinear`, and unit-tested directly.
nonisolated func packPairs(_ pairs: MLXArray, groupSize: Int) -> MLXArray {
    let krot = pairs.dim(0)
    let numGroups = pairs.dim(1) / groupSize

    // Reshape to [krot, numGroups, groupSize]
    let p = pairs.reshaped(krot, numGroups, groupSize).asType(.int32)

    // Even indices (lo) and odd indices (hi) within each group
    let lo = p[0..., 0..., .stride(by: 2)]
    let hi = p[0..., 0..., .stride(from: 1, by: 2)]
    return (lo | (hi << 16)).reshaped(krot, -1)
}

// MARK: - Derived Rotation State

/// The four kernel-ready tensors derived from a rotation's checkpoint
/// parameters (`theta` / `pairs` / `channel_scales`), shared by
/// `RotateQuantizedLinear` (dense) and `PairwiseRotation` (MoE shared) so
/// the derivation cannot diverge between the two paths.
///
/// Owners store this in an underscore-prefixed property: Module reflection
/// drops `_`-prefixed keys (`Module.parameterIsValid`), so the derived
/// tensors don't participate in weight loading — which keeps the loader's
/// strict `verify: [.allModelKeysSet]` contract intact — and are skipped by
/// `eval(model)`, which walks reflected parameters only.
struct RotationDerivedState {
    var cosTheta: MLXArray
    var sinTheta: MLXArray
    var packedPairs: MLXArray
    var scalesFlat: MLXArray

    /// Placeholder state — `prepare(...)` overwrites it after checkpoint
    /// load. Shapes are correct so a forward pass before finalize would be
    /// degenerate (identity-ish rotation) rather than crash.
    init(dims: Int, krot: Int) {
        cosTheta = MLXArray.ones([krot, dims / 2])
        sinTheta = MLXArray.zeros([krot, dims / 2])
        packedPairs = MLXArray.zeros([krot, dims / 2], type: Int32.self)
        scalesFlat = MLXArray.ones([dims])
    }

    /// Recompute from the loaded checkpoint parameters. The results are
    /// lazy graph nodes — materializing them is the caller's job (see
    /// `RotationStatePreparing` for why).
    mutating func prepare(
        theta: MLXArray, pairs: MLXArray, channelScales: MLXArray, groupSize: Int
    ) {
        cosTheta = MLX.cos(theta)
        sinTheta = MLX.sin(theta)
        packedPairs = packPairs(pairs, groupSize: groupSize)
        scalesFlat = channelScales.reshaped(-1)
    }

    /// The derived tensors, for batching one `eval` across many modules.
    var all: [MLXArray] { [cosTheta, sinTheta, packedPairs, scalesFlat] }
}

/// Load-time finalization hook shared by every rotation-carrying module.
///
/// The loader walks leaf modules and finalizes each conformer after the
/// checkpoint update — a protocol rather than a class enumeration so a new
/// rotation carrier cannot be silently skipped (a missed module would keep
/// its degenerate placeholder state and produce wrong numbers without ever
/// crashing).
///
/// `prepareDerivedRotationState()` returns the freshly derived tensors
/// *unmaterialized*: the loader batches a single `eval` over every module's
/// tensors instead of paying one GPU round-trip per module (~480 modules on
/// a 48-layer MoE). Discarding the result leaves the tensors lazy until
/// first use — harmless for tests, but the loader must eval them so
/// materialization stays out of the first forward pass's graph. Deriving
/// lazily *during* a forward pass was issue #157.
protocol RotationStatePreparing: AnyObject {
    /// Recompute rotation-derived tensors from the loaded checkpoint
    /// parameters. Must run after `update(parameters:)` and never
    /// concurrently with forward passes — the loader owns this call.
    @discardableResult
    func prepareDerivedRotationState() -> [MLXArray]
}

// MARK: - PairwiseRotation

/// Standalone pairwise Givens rotation over the last axis of an activation
/// tensor, fused with per-channel scaling in a single Metal kernel.
///
/// This is the rotation half of `RotateQuantizedLinear`, extracted as a
/// composable `Module` for layers whose rotation is *shared* across several
/// quantized projections instead of owned by one — the MoE `RotateSwitchGLU`
/// composes two of these (`gate_up_rot`, `down_rot`) around stock
/// `QuantizedSwitchLinear` experts.
///
/// Checkpoint contract: `theta` / `pairs` / `channel_scales` load via Module
/// reflection under this module's key prefix (e.g.
/// `switch_mlp.gate_up_rot.theta`). After loading, the owner must call
/// `prepareDerivedRotationState()` once, before any forward pass.
public class PairwiseRotation: Module, RotationStatePreparing {

    // Rotation parameters — discovered by Module reflection for update(parameters:).
    // `channelScales` uses @ParameterInfo so it can keep the snake_case checkpoint
    // key while having a Swift-idiomatic property name.
    let theta: MLXArray
    let pairs: MLXArray
    @ParameterInfo(key: "channel_scales") var channelScales: MLXArray

    let groupSize: Int

    // Populated once by `prepareDerivedRotationState()` after the checkpoint
    // parameters are loaded (see ParoQuantLoader), and never mutated
    // afterwards. See `RotationDerivedState` for why the underscore prefix
    // keeps it out of weight loading.
    private var _rotation: RotationDerivedState

    /// - Precondition: `dims` is a positive multiple of an even `groupSize`,
    ///   and `krot >= 1` — see `rotationGeometryProblem`.
    public init(dims: Int, groupSize: Int, krot: Int) {
        assertRotationGeometry(dims: dims, groupSize: groupSize, krot: krot)
        self.theta = MLXArray.zeros([krot, dims / 2])
        self.pairs = MLXArray.zeros([krot, dims], type: Int16.self)
        // Assign through `.wrappedValue` so the `@ParameterInfo(key:)` metadata
        // survives init — see the matching note in RotateQuantizedLinear.
        self._channelScales.wrappedValue = MLXArray.ones([1, dims])
        self.groupSize = groupSize
        self._rotation = RotationDerivedState(dims: dims, krot: krot)

        super.init()

        // Rotation parameters are inference-only checkpoint constants, never
        // trained — the same contract `QuantizedLinear` applies to its scales
        // and biases. Freezing keeps `trainableParameters()` empty on any
        // module that composes this one, which `SwitchGLU`'s direct weighted
        // reduction requires.
        self.freeze()
    }

    /// See `RotationStatePreparing` — loader-owned, results batched into a
    /// single load-time `eval` with every other rotation module's.
    @discardableResult
    public func prepareDerivedRotationState() -> [MLXArray] {
        _rotation.prepare(
            theta: theta, pairs: pairs, channelScales: channelScales, groupSize: groupSize)
        return _rotation.all
    }

    /// Apply channel scaling + pairwise Givens rotations to the last axis.
    ///
    /// Accepts any leading shape (the MoE path passes gathered 4-D
    /// activations); the input is flattened to 2-D for the kernel and the
    /// original shape is restored on return. Empty batches pass through
    /// unchanged (guarded in `dispatchPairwiseRotation`). No mutable state
    /// is read or written by this method.
    ///
    /// Kernel selection lives in `dispatchPairwiseRotation`.
    public func rotate(_ x: MLXArray) -> MLXArray {
        let shape = x.shape
        return dispatchPairwiseRotation(
            x.reshaped(-1, _rotation.scalesFlat.dim(0)),
            state: _rotation, groupSize: groupSize, krot: theta.dim(0)
        ).reshaped(shape)
    }
}
