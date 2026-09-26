use super::super::{cost_model::*, LogicalWorkGeneration};
use ferrum_interfaces::execution_cost::{ActualRowWork, ActualWaveKind, CanonicalWaveCostShape};
use ferrum_types::{RequestId, SloPlannerConfig};
use std::{
    num::{NonZeroU32, NonZeroU64, NonZeroUsize},
    sync::Arc,
};

/// Adapter of the existing owner's identity. The ordinal must come from that
/// owner and be rechecked there; constructing this value grants no authority.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RequestWorkKey {
    pub request_id: RequestId,
    pub incarnation: u64,
    pub work_generation: LogicalWorkGeneration,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlannerLatencyBudgets {
    pub ttft_ns: NonZeroU64,
    pub tpot_ns: NonZeroU64,
    pub itl_ns: NonZeroU64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RequestTimingView {
    pub ingress_at_ns: u64,
    pub first_commit_at_ns: Option<u64>,
    pub last_commit_at_ns: Option<u64>,
    pub committed_tokens: u32,
    pub maximum_output_tokens: NonZeroU32,
    pub budgets: PlannerLatencyBudgets,
    /// Historical violation survives every projection and simulation.
    pub slo_failed: bool,
}

impl RequestTimingView {
    pub fn next_deadline_ns(&self) -> Option<u64> {
        if self.committed_tokens == 0 {
            self.ingress_at_ns.checked_add(self.budgets.ttft_ns.get())
        } else {
            let itl = self
                .last_commit_at_ns?
                .checked_add(self.budgets.itl_ns.get())?;
            let prefix =
                u64::from(self.committed_tokens).checked_mul(self.budgets.tpot_ns.get())?;
            Some(itl.min(self.first_commit_at_ns?.checked_add(prefix)?))
        }
    }

    pub fn completed(&self) -> bool {
        self.committed_tokens >= self.maximum_output_tokens.get()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestReadiness {
    Ready,
    ResourceBlocked,
    OutputBlocked,
    StateBlocked,
    /// Missing evidence is not permission to silently omit this request.
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OutputCreditView {
    /// Token commits backed by queue/event capacity and bounded projection
    /// storage, without assuming consumer reads or future credit releases.
    /// The last command may still occupy its reserved projection arena. One
    /// command can produce multiple SSE text events.
    pub available_token_commands: u32,
    pub byte_backing: OutputByteBacking,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputByteBacking {
    /// Additional output needs both this request's budget and the shared
    /// unreserved bytes in `CapacityReadView`. No simulated refill is allowed.
    Incremental {
        available_bytes: u64,
        /// A conservative producer bound, not observed average output bytes.
        bytes_per_token_upper_bound: Option<NonZeroU64>,
    },
    /// The producer already owns a reservoir proved to cover this remaining
    /// command sequence. It is included in live pool usage, so spending it
    /// must not also consume shared free bytes. The remaining byte count is
    /// an ownership observation, not a per-token estimate to multiply.
    PrepaidLifetime {
        remaining_token_commands: u32,
        remaining_wire_bytes: u64,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReferenceWorkPoint {
    pub prompt_tokens: u32,
    pub cumulative_work_ns: u64,
}

/// Frozen scoring rule. Interpolation is a reference-work definition, never
/// an execution-time estimate or permission to execute a fragment.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum ReferenceWorkEvaluation {
    #[default]
    ExactV1,
    PiecewiseV2 {
        body: Arc<[ReferenceWorkPoint]>,
    },
}

/// Compact fixed checkpoints plus the explicitly versioned scoring rule.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrefillReferenceWork {
    pub version: u64,
    pub points: Vec<ReferenceWorkPoint>,
    pub evaluation: ReferenceWorkEvaluation,
}

impl PrefillReferenceWork {
    pub fn evidence_point_count(&self) -> usize {
        match &self.evaluation {
            ReferenceWorkEvaluation::ExactV1 => self.points.len(),
            ReferenceWorkEvaluation::PiecewiseV2 { body } => {
                self.points.len().saturating_add(body.len())
            }
        }
    }
    /// Public pure snapshots must validate the evaluator as well as their
    /// compact checkpoints; an arbitrary Arc is not a calibration receipt.
    pub fn evaluation_is_valid(&self) -> bool {
        let ReferenceWorkEvaluation::PiecewiseV2 { body } = &self.evaluation else {
            return true;
        };
        let Some(last) = self.points.last() else {
            return false;
        };
        let Some(before) = last.prompt_tokens.checked_sub(1) else {
            return false;
        };
        body.first()
            == Some(&ReferenceWorkPoint {
                prompt_tokens: 0,
                cumulative_work_ns: 0,
            })
            && body.windows(2).all(|p| {
                p[1].prompt_tokens > p[0].prompt_tokens
                    && p[1]
                        .cumulative_work_ns
                        .checked_sub(p[0].cumulative_work_ns)
                        .is_some_and(|delta| {
                            delta >= u64::from(p[1].prompt_tokens - p[0].prompt_tokens)
                        })
            })
            && interpolate_reference_work(body, before)
                .is_some_and(|work| work < last.cumulative_work_ns)
            && self
                .points
                .iter()
                .take(self.points.len().saturating_sub(1))
                .all(|point| {
                    interpolate_reference_work(body, point.prompt_tokens)
                        == Some(point.cumulative_work_ns)
                })
    }
    /// Greatest nonfinal integer prefix with score <= work. B is strictly
    /// increasing at integer-token resolution, validated at artifact load.
    pub fn nonfinal_prefix_at_or_below(&self, work: u64) -> Option<u32> {
        let ReferenceWorkEvaluation::PiecewiseV2 { body } = &self.evaluation else {
            return None;
        };
        let maximum = self.points.last()?.prompt_tokens.checked_sub(1)?;
        let index = body.partition_point(|p| p.cumulative_work_ns <= work);
        let left = body.get(index.checked_sub(1)?)?;
        let Some(right) = body.get(index) else {
            return Some(maximum);
        };
        let width = right.prompt_tokens.checked_sub(left.prompt_tokens)?;
        let height = right
            .cumulative_work_ns
            .checked_sub(left.cumulative_work_ns)?;
        if height == 0 {
            return None;
        }
        let numerator = u128::from(work.checked_sub(left.cumulative_work_ns)?)
            .checked_add(1)?
            .checked_mul(u128::from(width))?;
        let k = numerator.div_ceil(u128::from(height)).checked_sub(1)?;
        Some(
            left.prompt_tokens
                .checked_add(u32::try_from(k).ok()?)?
                .min(maximum),
        )
    }
    pub fn work_at(&self, prompt_tokens: u32) -> Option<u64> {
        if let ReferenceWorkEvaluation::PiecewiseV2 { body } = &self.evaluation {
            let last = self.points.last()?;
            return match prompt_tokens.cmp(&last.prompt_tokens) {
                std::cmp::Ordering::Greater => None,
                std::cmp::Ordering::Equal => Some(last.cumulative_work_ns),
                std::cmp::Ordering::Less => interpolate_reference_work(body, prompt_tokens),
            };
        }
        self.points
            .binary_search_by_key(&prompt_tokens, |point| point.prompt_tokens)
            .ok()
            .map(|index| self.points[index].cumulative_work_ns)
    }
}

/// Checked integer interpolation, also valid for positive nonmonotone terminal
/// anchor values. The body definition validates monotonicity separately.
pub fn interpolate_reference_work(points: &[ReferenceWorkPoint], p: u32) -> Option<u64> {
    match points.binary_search_by_key(&p, |point| point.prompt_tokens) {
        Ok(index) => Some(points[index].cumulative_work_ns),
        Err(index) => {
            let left = points.get(index.checked_sub(1)?)?;
            let right = points.get(index)?;
            let width = right.prompt_tokens.checked_sub(left.prompt_tokens)?;
            let a = u128::from(left.cumulative_work_ns)
                .checked_mul(u128::from(right.prompt_tokens.checked_sub(p)?))?;
            let b = u128::from(right.cumulative_work_ns)
                .checked_mul(u128::from(p.checked_sub(left.prompt_tokens)?))?;
            u64::try_from(a.checked_add(b)?.checked_div(u128::from(width))?).ok()
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrefillMilestone {
    pub at_ns: u64,
    /// Already includes any explicitly allowed one-granule quantization error.
    pub required_reference_work_ns: u64,
}

/// Build conservative discrete internal milestones from fixed work boundaries.
/// Final TTFT still requires an actual first-token commit, with no tolerance.
pub fn linear_prefill_milestones(
    admitted_at_ns: u64,
    first_deadline_ns: u64,
    total_reference_work_ns: NonZeroU64,
    checkpoints_at_ns: &[u64],
    quantum_allowance_ns: u64,
) -> Option<Vec<PrefillMilestone>> {
    if first_deadline_ns <= admitted_at_ns || checkpoints_at_ns.len() > 1024 {
        return None;
    }
    let duration = first_deadline_ns - admitted_at_ns;
    let mut previous = admitted_at_ns;
    let mut result = Vec::with_capacity(checkpoints_at_ns.len());
    for &at_ns in checkpoints_at_ns {
        if at_ns <= previous || at_ns > first_deadline_ns {
            return None;
        }
        let numerator =
            u128::from(total_reference_work_ns.get()) * u128::from(at_ns - admitted_at_ns);
        let required = numerator.div_ceil(u128::from(duration)) as u64;
        result.push(PrefillMilestone {
            at_ns,
            required_reference_work_ns: if at_ns == first_deadline_ns {
                required
            } else {
                required.saturating_sub(quantum_allowance_ns)
            },
        });
        previous = at_ns;
    }
    Some(result)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrefillProgressView {
    /// Original time promise anchor. Never reset to conceal prefill debt.
    pub admitted_at_ns: u64,
    /// Useful reference work already committed at that original anchor.
    pub reference_work_at_admission_ns: u64,
    /// Physical next fragment. Can trail logical_high_water during recompute.
    pub offset: u32,
    pub total_prompt_tokens: NonZeroU32,
    /// Previously committed useful logical prefix; never reset for recompute.
    pub logical_high_water: u32,
    pub reference: Arc<PrefillReferenceWork>,
    pub milestones: Arc<[PrefillMilestone]>,
    /// Backend/prefix-rendezvous ceiling for this projection, not an authority.
    pub executable_until: u32,
}

impl PrefillProgressView {
    /// Continuous ideal progress is a soft score target. Discrete hard
    /// milestones may allow a granule of quantization error independently.
    pub fn ideal_reference_work_at(&self, now_ns: u64, first_deadline_ns: u64) -> Option<u64> {
        let duration = first_deadline_ns.checked_sub(self.admitted_at_ns)?;
        if duration == 0 {
            return None;
        }
        let total = self.reference.points.last()?.cumulative_work_ns;
        let remaining = total.checked_sub(self.reference_work_at_admission_ns)?;
        let elapsed = now_ns.saturating_sub(self.admitted_at_ns).min(duration);
        let increment =
            (u128::from(remaining) * u128::from(elapsed)).div_ceil(u128::from(duration));
        self.reference_work_at_admission_ns
            .checked_add(u64::try_from(increment).ok()?)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RequestPhaseView {
    Decode,
    Prefill(PrefillProgressView),
}

/// A reliable optimistic bound across *all* supported alternatives under this
/// version. Empirical planning costs are not such lower bounds. The provider
/// must establish this certificate independently; absence is normal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OptimisticServiceLowerBound {
    pub model_version: u64,
    pub duration_ns: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RequestSchedulingView {
    pub key: RequestWorkKey,
    pub timing: RequestTimingView,
    pub phase: RequestPhaseView,
    pub readiness: RequestReadiness,
    pub context_tokens: u32,
    pub recurrent_state_bytes: u64,
    pub output_credit: OutputCreditView,
    /// Cached per-request host policy, before generated-history projection.
    /// The resolver receives the numeric history separately through `timing`.
    pub output_policy_signature: [u8; 32],
    pub fairness_rank: u64,
    pub recovery_service: super::obligations::RecoveryServiceDebt,
    /// Ordering hint under the snapshot model; not a feasibility certificate.
    pub ranking_service_cost_ns: Option<u64>,
    pub optimistic_next_service: Option<OptimisticServiceLowerBound>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendPlanningCapabilities {
    /// Product proposal bounds, re-evaluated on every successor ready domain.
    pub work_policy: super::super::work_policy::PlanningWorkPolicy,
    pub path: WaveExecutionPath,
    pub graph_state: WaveGraphState,
    pub order: BatchOrderSemantics,
    pub decode_batch_sizes: Vec<NonZeroUsize>,
    pub prefill_batch_sizes: Vec<NonZeroUsize>,
    pub prefill_chunk_sizes: Vec<NonZeroU32>,
    pub prefill_alignment: NonZeroU32,
    pub allow_final_short_chunk: bool,
    /// Means one actually unified wave. PlanRuntime Decode→Prefill is two waves.
    pub native_mixed: bool,
    pub max_wave_rows: NonZeroUsize,
    pub max_prefill_tokens_per_wave: NonZeroU64,
    /// Read-only conservative bound covering every declared supported shape.
    pub workspace_bytes_upper_bound: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CapacityReadView {
    /// Evidence for the scalar KV/workspace model used by `propose`. The
    /// explicit `propose_with_resources` path instead requires its resolver
    /// to initialize and advance a complete resource projection.
    pub evidence_known: bool,
    pub available_kv_tokens: u64,
    pub maximum_context_tokens: NonZeroU32,
    pub available_workspace_bytes: u64,
    /// Shared bytes not yet reserved by any request. Prepaid output reservoirs
    /// are excluded and cannot be charged to this balance a second time.
    pub available_output_bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlanningScope {
    /// Absolute time on the same monotonic clock as all ingress/commit values.
    pub horizon_end_ns: u64,
    /// Fixed for this reference version and for every compared candidate.
    pub reference_decode_token_ns: NonZeroU64,
    pub reference_work_version: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SchedulerSnapshot {
    pub observed_at_ns: u64,
    pub generation: u64,
    pub cost_model_version: u64,
    pub fingerprint: ExecutionFingerprint,
    pub requests: Vec<RequestSchedulingView>,
    pub capabilities: BackendPlanningCapabilities,
    pub capacity: CapacityReadView,
    pub scope: PlanningScope,
    /// Required restore/maintenance without an explicit bounded service model
    /// prevents a witness. It must not disappear from the obligation set.
    pub has_unmodeled_maintenance: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WaveAction {
    Decode,
    Prefill { offset: u32, count: NonZeroU32 },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CandidateWork {
    pub key: RequestWorkKey,
    pub action: WaveAction,
}

#[derive(Debug, Clone)]
pub struct WaveCandidate {
    /// Passive, per-alternative statistics; never resource/execution permission.
    pub cost_evidence: Option<PlanningShapeDomain<PlanningCostEvidence>>,
    pub work: Vec<CandidateWork>,
    pub execution_shape: PlanningShapeDomain<WaveExecutionShape>,
    pub based_on_generation: u64,
    pub cost_model_version: u64,
}

// Optional statistics do not redefine the old executable candidate identity.
impl PartialEq for WaveCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.work == other.work
            && self.execution_shape == other.execution_shape
            && self.based_on_generation == other.based_on_generation
            && self.cost_model_version == other.cost_model_version
    }
}
impl Eq for WaveCandidate {}

/// Privately bound to one validated canonical alternative. The expected shape
/// prevents a caller from rejoining a valid sidecar to different logical work.
#[derive(Debug, Clone)]
pub struct PlanningCostEvidence {
    expected: WaveExecutionShape,
    input: BoundCostInput,
}
#[derive(Debug, Clone)]
enum BoundCostInput {
    Selected(super::super::cost_model::statistical::StatisticalModelInputV1),
    Structured(Result<structured::StructuredInputV1, structured::StructuredUnknown>),
    StructuredV2(Result<structured_v2::StructuredQueryV2, structured_v2::StructuredUnknownV2>),
}
impl PlanningCostEvidence {
    pub(super) fn bind(
        exact: &CanonicalWaveCostShape,
        expected: &WaveExecutionShape,
        selected: &ferrum_interfaces::execution_cost::StatisticalWaveEvidenceV1,
        requirement: PlanningCostEvidenceRequirement,
    ) -> Option<Self> {
        Self::bind_with_forecast(exact, expected, selected, requirement, None)
    }
    pub(super) fn bind_with_forecast(
        exact: &CanonicalWaveCostShape,
        expected: &WaveExecutionShape,
        selected: &ferrum_interfaces::execution_cost::StatisticalWaveEvidenceV1,
        requirement: PlanningCostEvidenceRequirement,
        forecast: Option<&ferrum_interfaces::execution_cost::HostContentForecastV2>,
    ) -> Option<Self> {
        if super::cost_shape::canonical_cost_shape(exact).ok().as_ref() != Some(expected) {
            return None;
        }
        let input = match requirement {
            PlanningCostEvidenceRequirement::None => return None,
            PlanningCostEvidenceRequirement::Selected => BoundCostInput::Selected(
                super::super::cost_model::statistical::StatisticalModelInputV1::from_future(
                    exact, selected,
                )
                .ok()?,
            ),
            PlanningCostEvidenceRequirement::Structured => {
                // Cache the projection of this exact selected route once. A
                // missing/unsupported sidecar remains an explicit failed query;
                // it cannot fall back to a different cost protocol.
                let input = selected
                    .structured_capture()
                    .ok_or(structured::StructuredUnknown::MissingEvidence)
                    .and_then(|value| {
                        value.map_err(|_| structured::StructuredUnknown::MissingEvidence)
                    })
                    .and_then(|value| {
                        structured::StructuredInputV1::from_future(exact, selected, value)
                    });
                BoundCostInput::Structured(input)
            }
            PlanningCostEvidenceRequirement::StructuredV2 => {
                let input = forecast
                    .ok_or(structured_v2::StructuredUnknownV2::MissingEvidence)
                    .and_then(|forecast| {
                        let recipe = selected
                            .structured_capture()
                            .ok_or(structured_v2::StructuredUnknownV2::MissingEvidence)?
                            .map_err(|_| structured_v2::StructuredUnknownV2::MissingEvidence)?;
                        structured_v2::StructuredQueryV2::from_future(
                            exact, selected, recipe, forecast,
                        )
                    });
                BoundCostInput::StructuredV2(input)
            }
        };
        Some(Self {
            expected: expected.clone(),
            input,
        })
    }
    pub fn input_for(
        &self,
        shape: &WaveExecutionShape,
    ) -> Option<&super::super::cost_model::statistical::StatisticalModelInputV1> {
        match &self.input {
            BoundCostInput::Selected(input) if shape == &self.expected => Some(input),
            _ => None,
        }
    }
    pub fn structured_input_for(
        &self,
        shape: &WaveExecutionShape,
    ) -> Result<&structured::StructuredInputV1, structured::StructuredUnknown> {
        if shape != &self.expected {
            return Err(structured::StructuredUnknown::MissingEvidence);
        }
        match &self.input {
            BoundCostInput::Structured(input) => input.as_ref().map_err(|reason| *reason),
            _ => Err(structured::StructuredUnknown::MissingEvidence),
        }
    }
    pub fn structured_query_v2_for(
        &self,
        shape: &WaveExecutionShape,
    ) -> Result<&structured_v2::StructuredQueryV2, structured_v2::StructuredUnknownV2> {
        if shape != &self.expected {
            return Err(structured_v2::StructuredUnknownV2::MissingEvidence);
        }
        match &self.input {
            BoundCostInput::StructuredV2(query) => query.as_ref().map_err(|reason| *reason),
            _ => Err(structured_v2::StructuredUnknownV2::MissingEvidence),
        }
    }
}

/// An exact physical route, or the complete finite set of possible whole-wave
/// routes after unknown token content. Alternatives grant no execution permit:
/// a selected first wave must always be exact. Costs are never picked from a
/// cheap representative or formed by adding independent percentile estimates.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlanningShapeDomain<T> {
    Exact(T),
    HostContentAlternatives(Vec<T>),
}

impl<T> PlanningShapeDomain<T> {
    pub fn exact(&self) -> Option<&T> {
        match self {
            Self::Exact(shape) => Some(shape),
            Self::HostContentAlternatives(_) => None,
        }
    }
    pub fn shapes(&self) -> &[T] {
        match self {
            Self::Exact(shape) => std::slice::from_ref(shape),
            Self::HostContentAlternatives(shapes) => shapes,
        }
    }
}

/// An ordered row of legal logical work. Identity only correlates read views;
/// none of these values grants resource or execution authority.
#[derive(Debug, Clone, Copy)]
pub struct PlanningShapeRow<'a> {
    pub request: &'a RequestSchedulingView,
    pub work: ActualRowWork,
}

#[derive(Debug)]
pub struct PlanningShapeQuery<'a> {
    pub snapshot: &'a SchedulerSnapshot,
    /// Complete ordered prefix of this candidate sequence. A resolver with
    /// stateful initialization/staging rules must replay from its captured
    /// initial view; final row frontiers cannot establish execution order.
    /// Bounded by the planner's validated lookahead limit (at most 16 waves).
    pub prior_waves: &'a [WaveCandidate],
    pub kind: ActualWaveKind,
    pub rows: &'a [PlanningShapeRow<'a>],
    pub recurrent_state_bytes: u64,
}

/// Validation domain of graph labels emitted by the same immutable execution
/// projection. This is not graph residency or permission to capture/replay.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PlanningGraphDomain {
    /// Preserve the backend's exact snapshot label (including old backends).
    #[default]
    SnapshotExact,
    /// A captured configured catalog supports per-wave route selection.
    /// ConfiguredEager needs a core-proven eager route; Warm still requires
    /// an exact uploaded-program match. Disabled is outside this domain.
    ConfiguredPerWave,
    /// StartupReady permits only independently matched resident replay.
    ResidentReplayOnly,
}
impl PlanningGraphDomain {
    pub(super) fn accepts(self, expected: WaveGraphState, actual: WaveGraphState) -> bool {
        match self {
            Self::SnapshotExact => expected == actual,
            Self::ResidentReplayOnly => actual == WaveGraphState::Warm,
            Self::ConfiguredPerWave => {
                matches!(
                    actual,
                    WaveGraphState::ConfiguredEager | WaveGraphState::Warm
                )
            }
        }
    }
}

pub trait PlanningShapeResolver {
    fn graph_domain(&self) -> Result<PlanningGraphDomain, PlanningUnknownReason> {
        Ok(PlanningGraphDomain::SnapshotExact)
    }

    /// Pure physical ordering of an already chosen logical cohort. The caller
    /// verifies a full key+action permutation; this cannot add, remove, widen,
    /// or substitute work. It must obey the same bounded callback contract.
    fn order_work(
        &self,
        _snapshot: &SchedulerSnapshot,
        _work: &mut [CandidateWork],
        poll_budget: &mut dyn FnMut() -> Result<(), PlanningUnknownReason>,
    ) -> Result<(), PlanningUnknownReason> {
        poll_budget()
    }
    /// Pure, bounded route proof for this exact projection, including each
    /// row's output policy and numeric history. Past execution alone is not a
    /// proof of future selection. None means route evidence is unavailable.
    ///
    /// The caller bounds invocations and input/output sizes and polls before
    /// and after each call. A synchronous callback cannot be preempted; it must
    /// not block or perform I/O. Long implementations can additionally poll
    /// while processing their bounded route metadata.
    fn resolve(
        &self,
        query: &PlanningShapeQuery<'_>,
        poll_budget: &mut dyn FnMut() -> Result<(), PlanningUnknownReason>,
    ) -> Result<Option<CanonicalWaveCostShape>, PlanningUnknownReason>;

    /// A complete route domain, not a sample of likely routes. Implementations
    /// must preserve every reachable resource/residency state or return None.
    /// The planner independently bounds and validates the returned alternatives.
    fn resolve_domain(
        &self,
        query: &PlanningShapeQuery<'_>,
        poll_budget: &mut dyn FnMut() -> Result<(), PlanningUnknownReason>,
    ) -> Result<Option<PlanningShapeDomain<CanonicalWaveCostShape>>, PlanningUnknownReason> {
        self.resolve(query, poll_budget)
            .map(|shape| shape.map(PlanningShapeDomain::Exact))
    }
}

/// Read-only resource model backed by the executor's actual layout, budgets,
/// and allocation rules. A fresh projection is required for every candidate
/// sequence replay, including the final shared witness after search overhead.
pub trait PlanningResourceResolver {
    fn begin(
        &self,
        snapshot: &SchedulerSnapshot,
        poll_budget: &mut dyn FnMut() -> Result<(), PlanningUnknownReason>,
    ) -> Result<Box<dyn PlanningResourceProjection + '_>, PlanningUnknownReason>;
}

pub struct PlanningResourceQuery<'a> {
    pub snapshot: &'a SchedulerSnapshot,
    /// Complete projected obligations before this wave, including its peers.
    pub requests: &'a [RequestSchedulingView],
    pub wave: &'a WaveCandidate,
}

pub trait PlanningResourceProjection {
    /// Account for the entire ordered wave on this private projection. Retain
    /// persistent sequence growth across calls; release temporary work only
    /// at a modeled receipt. No allocation, maintenance, I/O, blocking locks,
    /// live reservations, or assumed future resource releases are permitted.
    /// Missing evidence must return Unknown, never infer a physical permit.
    fn apply(
        &mut self,
        query: &PlanningResourceQuery<'_>,
        poll_budget: &mut dyn FnMut() -> Result<(), PlanningUnknownReason>,
    ) -> Result<(), PlanningUnknownReason>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlanningCost {
    pub typical_ns: u64,
    pub planning_ns: u64,
    pub model_version: u64,
    /// Inclusive remaining freshness from this lookup's `now_ns`, expressed as
    /// a duration so local/imported clock epochs cannot be confused.
    pub valid_for_ns: u64,
}

/// The model's input protocol is fixed before projecting any candidate.
/// Unknown evidence never changes this selection or shrinks its alternatives.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlanningCostEvidenceRequirement {
    None,
    Selected,
    Structured,
    /// Whole-wave V2 with provider-bound uncertainty for every future branch.
    StructuredV2,
}

pub trait PlanningCostModel {
    /// New predictors explicitly require producer-bound selected statistics.
    /// The legacy callback remains unchanged for all existing implementations.
    fn requires_statistical_evidence(&self) -> bool {
        false
    }
    fn evidence_requirement(&self) -> PlanningCostEvidenceRequirement {
        if self.requires_statistical_evidence() {
            PlanningCostEvidenceRequirement::Selected
        } else {
            PlanningCostEvidenceRequirement::None
        }
    }
    fn predict_with_evidence(
        &self,
        fingerprint: &ExecutionFingerprint,
        shape: &WaveExecutionShape,
        _evidence: Option<&PlanningCostEvidence>,
        now_ns: u64,
    ) -> Option<PlanningCost> {
        if self.evidence_requirement() != PlanningCostEvidenceRequirement::None {
            None
        } else {
            self.predict(fingerprint, shape, now_ns)
        }
    }

    fn model_version(&self) -> u64;
    /// Only an explicitly calibrated whole-wave host-content model may consume
    /// hypothetical content branches. Legacy exact/numeric models cannot.
    fn supports_empirical_host_content(&self) -> bool {
        false
    }
    /// Bounded read-only lookup at this immutable model's declared full boundary.
    fn predict(
        &self,
        fingerprint: &ExecutionFingerprint,
        shape: &WaveExecutionShape,
        now_ns: u64,
    ) -> Option<PlanningCost>;
}

impl PlanningCostModel for CostModelSnapshot {
    fn model_version(&self) -> u64 {
        self.model_version()
    }
    fn supports_empirical_host_content(&self) -> bool {
        self.planning_boundary() == CostBoundary::PreparationToHostSettledV1
    }
    fn predict(
        &self,
        fingerprint: &ExecutionFingerprint,
        shape: &WaveExecutionShape,
        now_ns: u64,
    ) -> Option<PlanningCost> {
        match self.predict(fingerprint, shape, self.planning_boundary(), now_ns) {
            CostPrediction::Known(value) => Some(PlanningCost {
                typical_ns: value.typical_ns,
                planning_ns: value.planning_ns,
                model_version: value.model_version,
                valid_for_ns: value.valid_for_ns,
            }),
            CostPrediction::Unknown(_) => None,
        }
    }
}

impl PlanningCostModel for super::super::cost_profile::ImportedCostSnapshot {
    fn model_version(&self) -> u64 {
        self.model_version()
    }
    fn supports_empirical_host_content(&self) -> bool {
        self.planning_boundary() == CostBoundary::PreparationToHostSettledV1
    }
    fn predict(
        &self,
        fingerprint: &ExecutionFingerprint,
        shape: &WaveExecutionShape,
        now_ns: u64,
    ) -> Option<PlanningCost> {
        match self.predict(fingerprint, shape, self.planning_boundary(), now_ns) {
            CostPrediction::Known(value) => Some(PlanningCost {
                typical_ns: value.typical_ns,
                planning_ns: value.planning_ns,
                model_version: value.model_version,
                valid_for_ns: value.valid_for_ns,
            }),
            CostPrediction::Unknown(_) => None,
        }
    }
}

/// The original synchronous transaction, in the snapshot's monotonic epoch.
/// Capturing the snapshot and constructing an admission query consumes it too.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlanningBudgetWindow {
    pub started_at_ns: u64,
    pub deadline_ns: u64,
}

/// An earlier planner stop inside the original transaction. Phase percentages
/// still use `window`; adding a completion reserve must not scale them twice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlanningPhaseBudget {
    pub window: PlanningBudgetWindow,
    pub planner_deadline_ns: Option<u64>,
}

impl From<PlanningBudgetWindow> for PlanningPhaseBudget {
    fn from(window: PlanningBudgetWindow) -> Self {
        Self {
            window,
            planner_deadline_ns: None,
        }
    }
}

impl PlanningPhaseBudget {
    pub(super) fn phase_deadlines(
        self,
        settings: &SloPlannerConfig,
    ) -> Result<(u64, u64), PlanningUnknownReason> {
        let (search, replay) = self.window.phase_deadlines(settings)?;
        let capped_replay = replay.min(self.planner_deadline_ns.unwrap_or(replay));
        if capped_replay <= self.window.started_at_ns {
            return Err(PlanningUnknownReason::ComputeBudgetExhausted);
        }
        // Keep the configured final-replay allowance where it fits. A cap
        // before the old search endpoint must neither underflow nor remove
        // replay's reserve. Construct may still run to capped_replay; optional
        // improvement only begins after a complete common plan exists.
        let capped_search = search.min(
            capped_replay
                .saturating_sub(replay - search)
                .max(self.window.started_at_ns),
        );
        Ok((capped_search, capped_replay))
    }
}

impl PlanningBudgetWindow {
    pub(super) fn phase_deadlines(
        self,
        settings: &SloPlannerConfig,
    ) -> Result<(u64, u64), PlanningUnknownReason> {
        if settings.validate().is_err() {
            return Err(PlanningUnknownReason::InvalidConfiguration);
        }
        let configured_end = settings
            .max_planning_us
            .get()
            .checked_mul(1000)
            .and_then(|span| self.started_at_ns.checked_add(span))
            .ok_or(PlanningUnknownReason::ArithmeticOverflow)?;
        let deadline = self.deadline_ns.min(configured_end);
        let span = deadline
            .checked_sub(self.started_at_ns)
            .ok_or(PlanningUnknownReason::ComputeBudgetExhausted)?;
        let endpoint = |percent: u8| -> Result<u64, PlanningUnknownReason> {
            let offset = u64::try_from(u128::from(span) * u128::from(percent) / 100)
                .map_err(|_| PlanningUnknownReason::ArithmeticOverflow)?;
            self.started_at_ns
                .checked_add(offset)
                .ok_or(PlanningUnknownReason::ArithmeticOverflow)
        };
        let search = endpoint(settings.search_budget_percent)?;
        let replay = endpoint(100 - settings.publication_reserve_percent)?;
        if search <= self.started_at_ns || replay <= search || replay >= deadline {
            return Err(PlanningUnknownReason::ComputeBudgetExhausted);
        }
        Ok((search, replay))
    }
}

pub trait PlanningClock {
    /// Same monotonic epoch as the snapshot. Tests can supply a virtual clock.
    fn now_ns(&mut self) -> u64;
    /// Standalone callers may omit this; adapters preserve the outer start.
    fn planning_budget_window(&self) -> Option<PlanningBudgetWindow> {
        None
    }
    /// An optional earlier planner boundary, in the unchanged window's epoch.
    fn planning_phase_deadline_ns(&self) -> Option<u64> {
        None
    }
}

/// Exact first edge of the successful independent replay. Private fields and
/// construction prevent callers from relabeling a search projection as replay
/// evidence. This is immutable numeric evidence, never a submission permit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FinalReplayFirstWave {
    candidate: Arc<WaveCandidate>,
    canonical: Arc<CanonicalWaveCostShape>,
    statistics: Option<Arc<ferrum_interfaces::execution_cost::StatisticalWaveEvidenceV1>>,
    snapshot_observed_at_ns: u64,
}

impl FinalReplayFirstWave {
    pub(super) fn from_replay(
        snapshot: &SchedulerSnapshot,
        candidate: Arc<WaveCandidate>,
        canonical: Arc<CanonicalWaveCostShape>,
        statistics: Option<Arc<ferrum_interfaces::execution_cost::StatisticalWaveEvidenceV1>>,
    ) -> Self {
        Self {
            candidate,
            canonical,
            statistics,
            snapshot_observed_at_ns: snapshot.observed_at_ns,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SelectedWave {
    /// Missing evidence is rejected by production publication; no re-projection
    /// fallback. None remains available to legacy construction/negative tests.
    pub final_replay_first_wave: Option<Arc<FinalReplayFirstWave>>,
    /// Immutable partial-protection scope; never an all-request admission proof.
    pub protection: Option<Arc<super::obligations::PlanningObligationSet>>,
    pub candidate: WaveCandidate,
    pub predicted_wall_ns: u64,
    pub planning_observed_at_ns: u64,
    pub snapshot_observed_at_ns: u64,
    pub snapshot_generation: u64,
    pub cost_model_version: u64,
    /// Remaining inclusive start delay supported by the complete witness's
    /// deadlines and cost evidence. Final execution still requires authority.
    pub witness_valid_for_ns: u64,
}

impl SelectedWave {
    /// Capture-only immutable evidence from the successful independent replay.
    /// The public candidate sidecar is never consulted: candidate equality does
    /// not compare that sidecar. This accessor adds no projection or authority.
    pub fn replayed_first_wave_structured_v2(
        &self,
        snapshot: &SchedulerSnapshot,
    ) -> Option<(
        &Arc<CanonicalWaveCostShape>,
        &Arc<ferrum_interfaces::execution_cost::StatisticalWaveEvidenceV1>,
        &structured_v2::StructuredQueryV2,
    )> {
        self.replayed_first_wave(snapshot)?;
        let proof = self.final_replay_first_wave.as_ref()?;
        let query = proof
            .candidate
            .cost_evidence
            .as_ref()?
            .exact()?
            .structured_query_v2_for(proof.candidate.execution_shape.exact()?)
            .ok()?;
        Some((&proof.canonical, proof.statistics.as_ref()?, query))
    }

    /// Validate immutable candidate and snapshot identity before borrowing the
    /// canonical edge. Live route/resource/frontier/credit/time guards remain
    /// the publisher's responsibility. The canonical is never mutable here.
    pub fn replayed_first_wave(
        &self,
        snapshot: &SchedulerSnapshot,
    ) -> Option<&CanonicalWaveCostShape> {
        let proof = self.final_replay_first_wave.as_ref()?;
        (&self.candidate == proof.candidate.as_ref()
            && self.snapshot_generation == snapshot.generation
            && self.candidate.based_on_generation == snapshot.generation
            && self.cost_model_version == snapshot.cost_model_version
            && self.candidate.cost_model_version == snapshot.cost_model_version
            && self.snapshot_observed_at_ns == snapshot.observed_at_ns
            && proof.snapshot_observed_at_ns == snapshot.observed_at_ns)
            .then_some(proof.canonical.as_ref())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct PlanningWitnessSummary {
    pub waves: usize,
    pub completion_at_ns: u64,
    pub validated_through_ns: u64,
    pub predicted_output_tokens: u64,
    pub net_prefill_reference_work_ns: u64,
    pub terminal_prefill_debt_ns: u64,
    /// Local proxy units per nanosecond; never output throughput.
    pub proxy_score: f64,
    /// Active requests remain unpromised beyond the declared horizon.
    pub requests_with_obligations_beyond_horizon: usize,
}

/// Last phase reached, including on Unknown. Finalization includes common
/// ranking, independent replay and final clock/TTL checks, not just projection.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum PlanningSearchPhase {
    #[default]
    Construct,
    Improve,
    Finalization,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PlanningSearchStats {
    pub phase: PlanningSearchPhase,
    pub enumeration_attempts: usize,
    pub expanded_candidates: usize,
    pub generated_candidates: usize,
    pub max_depth_reached: usize,
    pub candidate_truncations: usize,
    /// A complete common witness existed when optional exploration stopped.
    /// This is a truncated search, never an optimality claim.
    pub search_soft_stops: usize,
    /// Subset of soft stops before the configured optional-search ceiling.
    pub replay_reserve_stops: usize,
    /// Largest observed complete-path begin/advance span; not a runtime bound.
    pub measured_replay_work_ns: u64,
    /// Configured final window plus that observed path high-water. Zero before
    /// any complete plan; never a guarantee that final replay will fit.
    pub replay_reserve_ns: u64,
    pub beam_pruned_nodes: usize,
    pub cost_unknown_candidates: usize,
    pub shape_unknown_candidates: usize,
    pub resource_unknown_candidates: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlanningImpossibleReason {
    HistoricalViolation {
        key: RequestWorkKey,
    },
    DeadlineAlreadyMissed {
        key: RequestWorkKey,
        deadline_ns: u64,
    },
    CertifiedOptimisticLowerBound {
        key: RequestWorkKey,
        deadline_ns: u64,
        earliest_completion_ns: u64,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlanningUnknownReason {
    InvalidConfiguration,
    InvalidSnapshot,
    ArithmeticOverflow,
    ClockMovedBackwards,
    ComputeBudgetExhausted,
    ModelVersionMismatch,
    UnknownResourceEvidence,
    UnknownReadiness,
    UnmodeledMaintenance,
    MissingReferenceWork,
    CostUnavailable,
    ShapeUnavailable,
    InvalidShapeEvidence,
    ShapeCapacity,
    OutputOrResourceBlocked,
    HorizonInsufficient,
    SearchIncomplete,
    NoWork,
    RecoveryConflict,
}

#[derive(Debug, Clone, PartialEq)]
pub enum PlanningDecision {
    ProtectedWithinHorizon {
        first_wave: SelectedWave,
        witness: PlanningWitnessSummary,
        protection: Arc<super::obligations::PlanningObligationSet>,
        search: PlanningSearchStats,
    },
    FeasibleWithinHorizon {
        first_wave: SelectedWave,
        witness: PlanningWitnessSummary,
        search: PlanningSearchStats,
    },
    ProvenImpossibleUnderModel {
        reason: PlanningImpossibleReason,
        model_version: u64,
        snapshot_generation: u64,
    },
    Unknown {
        reason: PlanningUnknownReason,
        search: PlanningSearchStats,
    },
}

#[derive(Debug, Clone)]
pub struct BoundedPlannerSettings {
    pub search: SloPlannerConfig,
    /// Time reserved between physical waves for the next controller transaction.
    /// The current transaction is accounted from its real clock separately.
    pub future_controller_time: FutureControllerTimeV1,
}

/// Prospective synchronous controller time, separate from execution-model labels.
/// This is a planning reservation, not a hard bound on queueing or failed retries.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum FutureControllerTimeV1 {
    /// Reserve the full configured budget for each later controller transaction.
    #[default]
    PlanningBudget,
    /// Explicit conservative estimate supplied by an integrating controller.
    ReservedNs(NonZeroU64),
    /// An ideal controller used only by abstract virtual-time unit fixtures.
    #[cfg(test)]
    InstantaneousVirtualController,
}

impl FutureControllerTimeV1 {
    pub fn reserved_ns(self, search: &SloPlannerConfig) -> Result<u64, PlanningUnknownReason> {
        match self {
            Self::PlanningBudget => search
                .max_planning_us
                .get()
                .checked_mul(1000)
                .ok_or(PlanningUnknownReason::ArithmeticOverflow),
            Self::ReservedNs(value) => Ok(value.get()),
            #[cfg(test)]
            Self::InstantaneousVirtualController => Ok(0),
        }
    }
}

impl Default for BoundedPlannerSettings {
    fn default() -> Self {
        Self {
            search: SloPlannerConfig::default(),
            future_controller_time: FutureControllerTimeV1::default(),
        }
    }
}
