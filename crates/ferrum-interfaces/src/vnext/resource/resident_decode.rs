//! Exclusive cold allocations and fresh whole-program execution authority.
//! Hot uses never construct an ordinary Step or submission Wave.
use super::*;
use crate::vnext::{
    CompletionReadbackBatchRequest, DeviceBufferRetention, ReusableExecutionBucketId,
    ReusableExecutionBucketSpec, SuccessfulWaveCompletionSeal,
};
use std::any::Any;
use std::ops::Range;
mod frame;
mod reservation;
mod use_;
use frame::ResidentFrameOwner;
pub use frame::{ResidentFrameFinalizationFailure, ResidentFrameRetirement};

const IDLE: u8 = 0;
const BUSY: u8 = 1;
const REVOKED: u8 = 2;

pub(super) struct ResidentUseGate {
    state: Arc<AtomicU8>,
    finalized: bool,
}
impl ResidentUseGate {
    fn acquire(state: &Arc<AtomicU8>) -> Result<Self, VNextError> {
        state
            .compare_exchange(IDLE, BUSY, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| invalid_resource("resident decode reservation is busy or revoked"))?;
        Ok(Self {
            state: Arc::clone(state),
            finalized: false,
        })
    }
    pub(super) fn mark_finalized(&mut self) {
        self.finalized = true;
    }
}
impl Drop for ResidentUseGate {
    fn drop(&mut self) {
        if self.finalized {
            let _ = self
                .state
                .compare_exchange(BUSY, IDLE, Ordering::AcqRel, Ordering::Acquire);
        } else {
            self.state.store(REVOKED, Ordering::Release);
        }
    }
}

pub enum ResidentBootstrapDecision<R: DeviceRuntime> {
    Prepared {
        step: Arc<StepResourceLease<R>>,
        wave: PreparedStepSubmissionWave<R>,
    },
    Fallback,
    RequestStateDeferred(RequestStateHazardDeferral),
    RequestStateSplitRequired(RequestStateHazardSplitRequired),
    RequestStatePoisoned(RequestStateHazardPoison),
}
pub enum ResidentDecodeDecision<R: DeviceRuntime> {
    Ready {
        use_: ResidentDecodeUse<R>,
        retirement: ResidentFrameRetirement<R>,
    },
    Fallback,
    RequestStateDeferred(RequestStateHazardDeferral),
    RequestStateSplitRequired(RequestStateHazardSplitRequired),
    RequestStatePoisoned(RequestStateHazardPoison),
}

#[must_use = "idle reservations must be released before unrelated admission"]
pub struct ResidentDecodeReservation<R: DeviceRuntime> {
    step_backing: ClaimedBackingTransaction,
    wave_backing: ClaimedSubmissionWaveBacking,
    backing_lookup: ResidentBackingLookup,
    used_pools: Vec<Arc<DynamicBackingPool<R>>>,
    pool_proofs: Vec<ReservedPoolProof<R::Buffer>>,
    // Only immutable physical append history, never fresh frame authority.
    // Advancing this after a checked append avoids rescanning the initial
    // prefix on every subsequent token.
    physical_snapshots: Mutex<Vec<Arc<SequenceBackingSnapshot<R>>>>,
    participants: ExecutionBatchParticipants<R>,
    lane: Arc<ExecutionLane<R>>,
    bucket: ReusableExecutionBucketSpec,
    bucket_id: ReusableExecutionBucketId,
    pressure_action: AdmissionPressureAction,
    state: Arc<AtomicU8>,
    full_nodes: Vec<usize>,
}

// Positions only: each fresh use still materializes and validates the current
// authority through the pool. No view, buffer address or participant is cached.
struct ResidentBackingLookup {
    nodes: BTreeSet<NodeId>,
    shared: BTreeMap<ResourceId, ResidentSharedBackingLocation>,
}

#[derive(Clone, Copy)]
enum ResidentSharedBackingLocation {
    Wave(usize),
    Step(usize),
}

impl ResidentBackingLookup {
    fn new(
        nodes: &[PlanNode],
        wave: &[LogicalBackingSliceAuthority],
        step: &[LogicalBackingSliceAuthority],
    ) -> Self {
        let mut shared = BTreeMap::new();
        for (index, slice) in wave.iter().enumerate() {
            shared
                .entry(slice.resource_id().clone())
                .or_insert(ResidentSharedBackingLocation::Wave(index));
        }
        for (index, slice) in step.iter().enumerate() {
            shared
                .entry(slice.resource_id().clone())
                .or_insert(ResidentSharedBackingLocation::Step(index));
        }
        Self {
            nodes: nodes.iter().map(|node| node.id().clone()).collect(),
            shared,
        }
    }
}

struct ReservedPoolProof<B> {
    epoch: AtomicU64,
    chunks: Vec<(u32, Arc<ResidentChunkBacking<B>>)>,
}
impl<R: DeviceRuntime> Drop for ResidentDecodeReservation<R> {
    fn drop(&mut self) {
        self.state.store(REVOKED, Ordering::Release);
    }
}

#[must_use = "a fresh resident use must be submitted or explicitly rolled back"]
pub struct ResidentDecodeUse<R: DeviceRuntime> {
    terminal_payload: Option<Arc<dyn Any + Send + Sync>>,
    initializations: Option<PreparedBackingInitializations>,
    hazards: Option<RequestStateHazardPermit<Arc<AdmittedRequestResources<R>>>>,
    flights: Vec<PreparedSubmissionWaveParticipantFlightHold>,
    active_wave: ActiveInvocationWaveGuard,
    frame: Arc<ResidentFrameOwner<R>>,
    attempt: BatchInvocationId,
    fingerprint: String,
    terminal_request: Option<CompletionReadbackBatchRequest>,
}

fn supported_request(request: &StepResourceAdmissionRequest) -> bool {
    request.fit_policy() == AdmissionFitPolicy::ImmediateOnly
        && request.reusable_execution_bucket_id().is_some()
        && !request.work_shape().participants().is_empty()
        && request
            .work_shape()
            .participant_token_ranges()
            .iter()
            .all(|range| range.immediate_tokens() == 1)
}

impl<R: DeviceRuntime> ExecutionBatchParticipants<R> {
    /// Cold reservation. Only this operation evaluates demand, claims the two
    /// resource scopes, binds their certificates and builds the program binding.
    pub fn try_reserve_resident_decode(
        &self,
        request: &StepResourceAdmissionRequest,
        lane: &Arc<ExecutionLane<R>>,
    ) -> Result<Option<Arc<ResidentDecodeReservation<R>>>, VNextError> {
        if !supported_request(request) {
            return Ok(None);
        }
        let step = match self.try_begin_step(request.clone(), lane)? {
            StepResourceAdmissionDecision::Admitted(step) => step,
            _ => return Ok(None),
        };
        // No invocation is prepared in this cold frame. Roll it back after
        // collecting the allocation owners, leaving the first use a fresh frame.
        let result = (|| {
            let plan = &self.sessions[0].resources().request.plan;
            let _lifecycle = plan.resources.read_lifecycle("reserve a decode session")?;
            let bucket = step.reusable_execution_bucket().cloned().ok_or_else(|| {
                invalid_resource("decode session requires a resolved reusable bucket")
            })?;
            let shape = request.immediate_shape();
            let (demand, slices) = plan.submission_wave_demand(
                shape,
                shape,
                Some(&bucket),
                request.fit_policy(),
                request.pressure_action(),
            )?;
            let prepared = match plan.prepare_lane_stable_backing_slices(lane, slices)? {
                LaneBackingPrepareDecision::Prepared(prepared) => prepared,
                LaneBackingPrepareDecision::Deferred(_) => return Ok(None),
            };
            let capacity = if demand.immediate_claim().is_empty() {
                None
            } else {
                let parents = self
                    .sessions
                    .iter()
                    .map(|session| session.resources().logical_lease())
                    .collect::<Vec<_>>();
                match plan
                    .logical_admission()
                    .try_claim_for_sequences(&parents, &demand)?
                {
                    BatchCapacityClaimDecision::Claimed(capacity) => {
                        if !plan
                            .logical_admission()
                            .owns_batch_capacity_claim(&capacity)
                        {
                            return Err(invalid_resource(
                                "decode session received foreign capacity",
                            ));
                        }
                        Some(capacity)
                    }
                    _ => return Ok(None),
                }
            };
            let layout = if plan
                .nodes()
                .iter()
                .any(|node| node.binding_resource().is_some())
            {
                Some(
                    plan.dynamic_pools()
                        .program_binding_layout(bucket.bucket_id())
                        .cloned()
                        .ok_or_else(|| {
                            invalid_resource("decode session has no compiled binding layout")
                        })?,
                )
            } else {
                None
            };
            let wave_backing = ClaimedSubmissionWaveBacking::new(
                plan.plan_hash().clone(),
                plan.nodes().len(),
                plan.nodes().len(),
                Arc::clone(&request.work_shape),
                demand,
                capacity,
                prepared.commit(),
                layout,
            )?;
            let used_pool_ids = step
                .backing_slices()
                .iter()
                .chain(wave_backing.backing_slices())
                .map(|slice| slice.evidence().pool_id().clone())
                .collect::<BTreeSet<_>>();
            let used_pools = used_pool_ids
                .iter()
                .map(|pool_id| {
                    plan.dynamic_pools()
                        .pools
                        .get(pool_id)
                        .cloned()
                        .ok_or_else(|| invalid_resource("decode session backing pool is missing"))
                })
                .collect::<Result<Vec<_>, _>>()?;
            let mut pool_proofs = Vec::with_capacity(used_pools.len());
            for (pool_id, pool) in used_pool_ids.iter().zip(&used_pools) {
                let state = pool
                    .state
                    .lock()
                    .map_err(|_| invalid_resource("resident backing pool lock poisoned"))?;
                if state.poisoned {
                    return Err(invalid_resource("resident backing pool poisoned"));
                }
                let ordinals = step
                    .backing_slices()
                    .iter()
                    .chain(wave_backing.backing_slices())
                    .filter(|slice| slice.evidence().pool_id() == pool_id)
                    .flat_map(|slice| {
                        slice
                            .evidence()
                            .segments()
                            .iter()
                            .map(|s| s.chunk_ordinal())
                    })
                    .collect::<BTreeSet<_>>();
                let chunks = ordinals
                    .into_iter()
                    .map(|ordinal| {
                        state
                            .chunks
                            .get(&ordinal)
                            .map(|chunk| (ordinal, Arc::clone(&chunk.backing)))
                            .ok_or_else(|| invalid_resource("resident backing chunk missing"))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                pool_proofs.push(ReservedPoolProof {
                    epoch: AtomicU64::new(state.chunks.epoch()),
                    chunks,
                });
            }
            // The participant constructor performs its own lifecycle check;
            // never recursively acquire a read lock while a writer may wait.
            drop(_lifecycle);
            let backing_lookup = ResidentBackingLookup::new(
                plan.nodes(),
                wave_backing.backing_slices(),
                step.backing_slices(),
            );
            Ok(Some(Arc::new(ResidentDecodeReservation {
                step_backing: step
                    .claimed_backing()
                    .borrow_reserved(Arc::clone(&request.work_shape))?,
                wave_backing,
                backing_lookup,
                used_pools,
                pool_proofs,
                participants: ExecutionBatchParticipants::new(self.sessions.clone())?,
                lane: Arc::clone(lane),
                bucket_id: bucket.bucket_id().clone(),
                bucket,
                pressure_action: request.pressure_action(),
                state: Arc::new(AtomicU8::new(IDLE)),
                full_nodes: (0..plan.nodes().len()).collect(),
                physical_snapshots: Mutex::new(
                    step.participants
                        .iter()
                        .map(|p| Arc::clone(&p.backing_snapshot))
                        .collect(),
                ),
            })))
        })();
        step.try_rollback_unsubmitted()
            .map_err(|failure| failure.error)?;
        result
    }
}
