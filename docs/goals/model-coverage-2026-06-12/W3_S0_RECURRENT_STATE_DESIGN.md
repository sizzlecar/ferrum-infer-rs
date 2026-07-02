# W3-S0 Recurrent State Cache Design Checkpoint

Date: 2026-06-17

This is a design checkpoint for the W3 Gated DeltaNet lane. It is not a
release-grade artifact, does not replace `RELEASE_GRADE_GOAL.md`, and does not
produce `MODEL_RELEASE_GRADE_W3 PASS`.

## Objective

W3 needs a recurrent state cache before DeltaNet can enter product paths.
DeltaNet state is compact per-layer recurrent state, not attention KV blocks.
The interface must therefore coexist with paged KV, ContinuousBatch,
preemption, prefix cache, and mixed attention/recurrent models without
overloading `KvCacheHandle`.

## Current Interface Constraints

- `crates/ferrum-interfaces` is the GPU-free contract crate. Recurrent state
  contracts belong there, next to the KV cache and model executor traits.
- `BlockTable::sequence_length` is explicitly not authoritative for engine
  position tracking. W3 must keep `SequenceState` / engine-side request state as
  the source of token position truth.
- `PrefillInput`, `DecodeInput`, and `UnifiedBatchItem` currently carry KV
  handles. W3 needs a parallel recurrent-state handle, because hybrid models may
  need both KV and recurrent state in the same forward.
- `DecodeInput` currently requires a KV handle. Pure recurrent models should not
  be forced to allocate fake KV just to satisfy the interface.
- Scheduler resource accounting currently exposes KV blocks and byte estimates.
  W3 needs separate recurrent-state bytes/slots so admission, cancellation, and
  preemption do not silently leak or over-admit state.
- Scheduler preemption defaults to unsupported. If W3 enables preemption later,
  recurrent state must be either recomputed on resume or explicitly snapshotted;
  dropping it silently is a correctness bug.

## Design Decision

Add a separate recurrent-state cache abstraction. Do not fold recurrent state
into `KvCacheHandle`.

Reasoning:

- KV cache grows with sequence length and is addressed by logical/physical
  blocks. Recurrent state is O(1) with respect to sequence length and is
  addressed by layer/state slots.
- KV and recurrent state have different copy, eviction, prefix, and preemption
  semantics.
- Hybrid models can require both: attention layers still need KV while DeltaNet
  layers need recurrent state.
- Treating recurrent state as KV would make active sequence position,
  memory-admission, and prefix-cache validity ambiguous.

## Minimal Contract Shape

The first implementation should keep this contract small and GPU-free:

```rust
pub struct RecurrentStateSpec {
    pub request_id: RequestId,
    pub num_layers: usize,
    pub state_shapes: Vec<Vec<usize>>,
    pub dtype: DataType,
    pub device: Device,
    pub max_batch_slots: usize,
}

pub trait RecurrentStateHandle: Send + Sync + std::fmt::Debug {
    fn request_id(&self) -> RequestId;
    fn device(&self) -> Device;
    fn num_layers(&self) -> usize;
    fn state_bytes(&self) -> usize;
    fn is_valid(&self) -> bool;
    fn cache_id(&self) -> String;
    fn clone_handle(&self) -> Result<Arc<dyn RecurrentStateHandle>>;
    fn as_any(&self) -> &dyn std::any::Any;
}

#[async_trait::async_trait]
pub trait RecurrentStateManager: Send + Sync {
    async fn allocate(
        &self,
        spec: &RecurrentStateSpec,
    ) -> Result<Arc<dyn RecurrentStateHandle>>;
    async fn deallocate(&self, request_id: RequestId) -> Result<()>;
    fn can_allocate(&self, spec: &RecurrentStateSpec) -> bool;
    fn get_handle(&self, request_id: RequestId)
        -> Option<Arc<dyn RecurrentStateHandle>>;
    fn list_handles(&self) -> Vec<(RequestId, Arc<dyn RecurrentStateHandle>)>;
    fn stats(&self) -> RecurrentStateManagerStats;
    async fn reset(&self) -> Result<()>;
}
```

Snapshot/restore should not be included in the first trait unless a concrete
preemption path needs it. The initial W3 policy should be:

```text
RecomputeOnResume
```

That means preempted or evicted recurrent state is discarded only when the
engine has enough prompt/generated-token history to rebuild it before the next
decode. If that cannot be guaranteed, W3 must stop and add explicit
snapshot/restore before enabling preemption for recurrent models.

## Model Executor Integration

Extend execution inputs with an optional recurrent state handle:

- `PrefillInput::recurrent_state: Option<Arc<dyn RecurrentStateHandle>>`
- `DecodeInput::recurrent_state: Option<Arc<dyn RecurrentStateHandle>>`
- `UnifiedBatchItem::recurrent_state: Option<Arc<dyn RecurrentStateHandle>>`

Then relax the decode contract so at least one state carrier is present:

```text
decode requires kv_cache.is_some() || recurrent_state.is_some()
```

Attention-only models continue to use KV only. Pure DeltaNet models use
recurrent state only. Hybrid models carry both handles.

## Scheduler And Lifecycle Rules

- Admission must account for both KV blocks and recurrent-state bytes/slots.
- Request completion, cancellation, error cleanup, and engine reset must
  deallocate recurrent state separately from KV.
- Metrics should expose recurrent-state active handles, bytes, allocation
  failures, and evictions separately from KV metrics.
- Prefix cache hits for recurrent models are valid only if the artifact includes
  a compatible recurrent-state snapshot or the engine explicitly recomputes the
  recurrent state from tokens before continuing.
- Preemption is unsupported until there is either a recompute-on-resume proof or
  an explicit snapshot/restore contract.

## S0 Microbench Contract

Before product integration, W3 must run a native CUDA/PTX microbench for
chunked delta-rule:

- reference: official/mainstream DeltaNet/fla implementation, with recorded
  commit/revision and dependency versions;
- kernel path: Triton-to-PTX or native CUDA loaded through the Ferrum CUDA path;
- inputs: fixed random seed, recorded tensor shapes, dtype, chunk size, and
  distribution;
- validation: reference vs Ferrum max absolute error and relative error, with
  target `max_abs <= 1e-3` unless the implementation document justifies a
  stricter dtype-specific bound;
- artifact evidence: command lines, PTX arch, build command, CUDA driver/toolkit,
  GPU snapshot, git SHA/dirty status, and saved stdout/stderr.

S0 failure stop condition:

```text
If the chunked delta-rule kernel cannot match the reference, or if recurrent
state cannot coexist with current paged-KV/ContinuousBatch lifecycle semantics,
stop W3 implementation and write a blocker report before touching product paths.
```

## Next Implementation Checklist

1. Add GPU-free recurrent-state contracts in `crates/ferrum-interfaces`.
2. Add CPU/mock manager tests for allocate/get/list/deallocate/reset.
3. Extend model executor inputs without breaking existing KV-only models.
4. Add scheduler resource fields for recurrent-state bytes/slots.
5. Add engine cleanup on complete/cancel/error/reset.
6. Build the S0 microbench and compare against the recorded reference.
7. Only after S0 passes, start W3-S1 single-layer correctness.
