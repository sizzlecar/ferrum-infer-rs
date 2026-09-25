//! One immutable execution successor carries both route and resource evidence.
//! These are numeric projections, never live reservations or execution permits.
use super::{shape, types::*};
use ferrum_interfaces::execution_cost::{ActualWaveKind, CanonicalWaveCostShape};
use std::{cell::Cell, sync::Arc};

pub struct PlanningExecutionInput<'a> {
    pub work: &'a [CandidateWork],
    pub requests: &'a [RequestSchedulingView],
    pub kind: ActualWaveKind,
    pub rows: &'a [PlanningShapeRow<'a>],
    pub recurrent_state_bytes: u64,
}

/// A synchronous, nonblocking projection factory for a captured epoch. Every
/// successor must retain the complete resource/route domain for its parent.
pub trait PlanningExecutionContext {
    fn begin<'epoch>(
        &'epoch self,
        snapshot: &'epoch SchedulerSnapshot,
        poll: &mut dyn FnMut() -> Result<(), PlanningUnknownReason>,
    ) -> Result<Arc<dyn PlanningExecutionState<'epoch> + 'epoch>, PlanningUnknownReason>;
}

pub trait PlanningExecutionState<'epoch> {
    fn graph_domain(&self) -> Result<PlanningGraphDomain, PlanningUnknownReason> {
        Ok(PlanningGraphDomain::SnapshotExact)
    }

    /// Pure branch expansion. Failure must leave the parent and siblings intact.
    /// The child borrows the epoch, never this temporary parent or input.
    /// All reachable physical branches must support the same logical action.
    fn project(
        &self,
        input: &PlanningExecutionInput<'_>,
        poll: &mut dyn FnMut() -> Result<(), PlanningUnknownReason>,
    ) -> Result<Option<ProjectedExecution<'epoch>>, PlanningUnknownReason>;
}

pub struct ProjectedExecution<'epoch> {
    /// Same ordered alternatives as canonical_domain; absence never implies legacy statistics.
    pub statistical_evidence:
        Option<PlanningShapeDomain<ferrum_interfaces::execution_cost::StatisticalWaveEvidenceV1>>,
    /// Only a complete permutation of the input work is permitted.
    pub ordered_work: Vec<CandidateWork>,
    pub canonical_domain: PlanningShapeDomain<CanonicalWaveCostShape>,
    pub successor: Arc<dyn PlanningExecutionState<'epoch> + 'epoch>,
}

pub(super) struct VerifiedExecution<'epoch> {
    pub wave: WaveCandidate,
    pub first_canonical: Option<Arc<CanonicalWaveCostShape>>,
    pub successor: Arc<dyn PlanningExecutionState<'epoch> + 'epoch>,
}

/// Enforce polls outside callbacks and preserve a failed poll even when a
/// callback ignores it. Synchronous callbacks cannot be forcibly preempted.
pub(super) fn checked<T>(
    poll: &mut dyn FnMut() -> Result<(), PlanningUnknownReason>,
    operation: impl FnOnce(
        &mut dyn FnMut() -> Result<(), PlanningUnknownReason>,
    ) -> Result<T, PlanningUnknownReason>,
) -> Result<T, PlanningUnknownReason> {
    poll()?;
    let mut failure = None;
    let result = operation(&mut || {
        if let Some(reason) = failure {
            return Err(reason);
        }
        let result = poll();
        if let Err(reason) = result {
            failure = Some(reason);
        }
        result
    });
    let after = poll();
    if let Some(reason) = failure {
        return Err(reason);
    }
    after?;
    result
}

pub(super) struct ExecutionSession<'a> {
    context: &'a dyn PlanningExecutionContext,
    remaining: Cell<usize>,
    max_alternatives: usize,
}

impl<'a> ExecutionSession<'a> {
    pub fn new(
        context: &'a dyn PlanningExecutionContext,
        settings: &BoundedPlannerSettings,
    ) -> Self {
        let depth = settings.search.lookahead_waves.get();
        Self {
            context,
            // Includes root initialization, expansion and independent replay.
            // Keep the original validated work ceiling while changing dataflow.
            remaining: Cell::new(
                settings.search.candidate_limit.get()
                    * settings.search.beam_width.get()
                    * depth
                    * (16 + 2 * depth),
            ),
            max_alternatives: settings.search.max_shape_alternatives.get(),
        }
    }
}

impl PlanningExecutionContext for ExecutionSession<'_> {
    fn begin<'epoch>(
        &'epoch self,
        snapshot: &'epoch SchedulerSnapshot,
        poll: &mut dyn FnMut() -> Result<(), PlanningUnknownReason>,
    ) -> Result<Arc<dyn PlanningExecutionState<'epoch> + 'epoch>, PlanningUnknownReason> {
        spend(&self.remaining)?;
        let state = checked(poll, |poll| self.context.begin(snapshot, poll))?;
        Ok(Arc::new(BoundedState {
            state,
            remaining: &self.remaining,
            max_alternatives: self.max_alternatives,
        }))
    }
}

struct BoundedState<'epoch> {
    state: Arc<dyn PlanningExecutionState<'epoch> + 'epoch>,
    remaining: &'epoch Cell<usize>,
    max_alternatives: usize,
}

fn spend(remaining: &Cell<usize>) -> Result<(), PlanningUnknownReason> {
    remaining.set(
        remaining
            .get()
            .checked_sub(1)
            .ok_or(PlanningUnknownReason::SearchIncomplete)?,
    );
    Ok(())
}

impl<'epoch> PlanningExecutionState<'epoch> for BoundedState<'epoch> {
    fn graph_domain(&self) -> Result<PlanningGraphDomain, PlanningUnknownReason> {
        self.state.graph_domain()
    }

    fn project(
        &self,
        input: &PlanningExecutionInput<'_>,
        poll: &mut dyn FnMut() -> Result<(), PlanningUnknownReason>,
    ) -> Result<Option<ProjectedExecution<'epoch>>, PlanningUnknownReason> {
        spend(self.remaining)?;
        let Some(projected) = checked(poll, |poll| self.state.project(input, poll))? else {
            return Ok(None);
        };
        if projected.canonical_domain.shapes().len() > self.max_alternatives {
            return Err(PlanningUnknownReason::ShapeCapacity);
        }
        Ok(Some(ProjectedExecution {
            statistical_evidence: projected.statistical_evidence,
            ordered_work: projected.ordered_work,
            canonical_domain: projected.canonical_domain,
            successor: Arc::new(Self {
                state: projected.successor,
                remaining: self.remaining,
                max_alternatives: self.max_alternatives,
            }),
        }))
    }
}

pub(super) fn project<'epoch>(
    snapshot: &SchedulerSnapshot,
    requests: &[RequestSchedulingView],
    work: &[CandidateWork],
    state: &dyn PlanningExecutionState<'epoch>,
    retain_first_canonical: bool,
    collect_statistics: bool,
    poll: &mut dyn FnMut() -> Result<(), PlanningUnknownReason>,
) -> Result<Option<VerifiedExecution<'epoch>>, PlanningUnknownReason> {
    if !super::candidates::within_work_envelope(&snapshot.capabilities, requests, work, poll)? {
        return Ok(None);
    }
    let mut failure = None;
    let prepared = shape::legal_rows(snapshot, requests, work, &mut || match poll() {
        Ok(()) => true,
        Err(reason) => {
            failure = Some(reason);
            false
        }
    });
    if let Some(reason) = failure {
        return Err(reason);
    }
    let Some((kind, rows, recurrent_state_bytes)) = prepared else {
        return Ok(None);
    };
    let projected = checked(poll, |poll| {
        state.project(
            &PlanningExecutionInput {
                work,
                requests,
                kind,
                rows: &rows,
                recurrent_state_bytes,
            },
            poll,
        )
    })?
    .ok_or(PlanningUnknownReason::ShapeUnavailable)?;
    shape::validate_permutation(work, &projected.ordered_work, poll)?;
    let ordered_rows: Vec<_> = projected
        .ordered_work
        .iter()
        .map(|row| {
            rows[work
                .iter()
                .position(|original| original == row)
                .expect("validated permutation")]
        })
        .collect();
    let execution_shape = shape::validate_domain(
        snapshot,
        kind,
        &ordered_rows,
        recurrent_state_bytes,
        &projected.canonical_domain,
        state.graph_domain()?,
        poll,
    )?;
    let cost_evidence = if collect_statistics {
        bind_statistics(
            &projected.canonical_domain,
            &execution_shape,
            projected.statistical_evidence.as_ref(),
            poll,
        )?
    } else {
        None
    };
    // Retain only the first exact edge, not every future alternative. Moving
    // its validated value preserves the provider result without cloning vectors.
    let first_canonical = if retain_first_canonical {
        match projected.canonical_domain {
            PlanningShapeDomain::Exact(canonical) => Some(Arc::new(canonical)),
            PlanningShapeDomain::HostContentAlternatives(_) => None,
        }
    } else {
        None
    };
    Ok(Some(VerifiedExecution {
        first_canonical,
        wave: WaveCandidate {
            cost_evidence,
            work: projected.ordered_work,
            execution_shape,
            based_on_generation: snapshot.generation,
            cost_model_version: snapshot.cost_model_version,
        },
        successor: projected.successor,
    }))
}

/// Compatibility for existing shape/resource providers. This adapter explicitly
/// replays bounded history; it is not the incremental production implementation.
pub(super) struct ReplayContext<'a> {
    pub resolver: &'a dyn PlanningShapeResolver,
    pub resources: Option<&'a dyn PlanningResourceResolver>,
}

struct ReplayState<'epoch> {
    context: &'epoch ReplayContext<'epoch>,
    snapshot: &'epoch SchedulerSnapshot,
    prefix: Vec<(WaveCandidate, Vec<RequestSchedulingView>)>,
}

impl PlanningExecutionContext for ReplayContext<'_> {
    fn begin<'epoch>(
        &'epoch self,
        snapshot: &'epoch SchedulerSnapshot,
        poll: &mut dyn FnMut() -> Result<(), PlanningUnknownReason>,
    ) -> Result<Arc<dyn PlanningExecutionState<'epoch> + 'epoch>, PlanningUnknownReason> {
        if let Some(resources) = self.resources {
            // Even an empty sequence requires known complete resource evidence.
            super::resources::begin(resources, snapshot, poll)?;
        }
        Ok(Arc::new(ReplayState {
            context: self,
            snapshot,
            prefix: Vec::new(),
        }))
    }
}

impl<'epoch> PlanningExecutionState<'epoch> for ReplayState<'epoch> {
    fn graph_domain(&self) -> Result<PlanningGraphDomain, PlanningUnknownReason> {
        self.context.resolver.graph_domain()
    }

    fn project(
        &self,
        input: &PlanningExecutionInput<'_>,
        poll: &mut dyn FnMut() -> Result<(), PlanningUnknownReason>,
    ) -> Result<Option<ProjectedExecution<'epoch>>, PlanningUnknownReason> {
        if self.prefix.len() >= 16 {
            return Err(PlanningUnknownReason::ShapeCapacity);
        }
        let mut work = input.work.to_vec();
        shape::order_work(self.snapshot, &mut work, self.context.resolver, poll)?;
        let rows: Vec<_> = work
            .iter()
            .map(|entry| {
                input.rows[input
                    .work
                    .iter()
                    .position(|old| old == entry)
                    .expect("validated ordering")]
            })
            .collect();
        let prior_waves: Vec<_> = self.prefix.iter().map(|(wave, _)| wave.clone()).collect();
        let domain = checked(poll, |poll| {
            self.context.resolver.resolve_domain(
                &PlanningShapeQuery {
                    snapshot: self.snapshot,
                    prior_waves: &prior_waves,
                    kind: input.kind,
                    rows: &rows,
                    recurrent_state_bytes: input.recurrent_state_bytes,
                },
                poll,
            )
        })?;
        let Some(canonical_domain) = domain else {
            return Ok(None);
        };
        let execution_shape = shape::validate_domain(
            self.snapshot,
            input.kind,
            &rows,
            input.recurrent_state_bytes,
            &canonical_domain,
            self.graph_domain()?,
            poll,
        )?;
        let wave = WaveCandidate {
            cost_evidence: None,
            work: work.clone(),
            execution_shape,
            based_on_generation: self.snapshot.generation,
            cost_model_version: self.snapshot.cost_model_version,
        };
        if let Some(resources) = self.context.resources {
            let mut projection = super::resources::begin(resources, self.snapshot, poll)?;
            for (wave, requests) in &self.prefix {
                super::resources::apply(
                    projection.as_mut(),
                    &PlanningResourceQuery {
                        snapshot: self.snapshot,
                        requests,
                        wave,
                    },
                    poll,
                )?;
            }
            super::resources::apply(
                projection.as_mut(),
                &PlanningResourceQuery {
                    snapshot: self.snapshot,
                    requests: input.requests,
                    wave: &wave,
                },
                poll,
            )?;
        }
        let mut prefix = self.prefix.clone();
        prefix.push((wave, input.requests.to_vec()));
        Ok(Some(ProjectedExecution {
            statistical_evidence: None,
            ordered_work: work,
            canonical_domain,
            successor: Arc::new(Self {
                context: self.context,
                snapshot: self.snapshot,
                prefix,
            }),
        }))
    }
}

pub(super) fn bind_statistics(
    canonical: &PlanningShapeDomain<CanonicalWaveCostShape>,
    shapes: &PlanningShapeDomain<super::super::cost_model::WaveExecutionShape>,
    statistics: Option<
        &PlanningShapeDomain<ferrum_interfaces::execution_cost::StatisticalWaveEvidenceV1>,
    >,
    poll: &mut dyn FnMut() -> Result<(), PlanningUnknownReason>,
) -> Result<Option<PlanningShapeDomain<PlanningCostEvidence>>, PlanningUnknownReason> {
    let Some(statistics) = statistics else {
        return Ok(None);
    };
    // Domain variants and every ordered alternative must correspond. A partial
    // producer cannot shrink the physical alternatives to a supported subset.
    let same_variant = matches!(
        (canonical, shapes, statistics),
        (
            PlanningShapeDomain::Exact(_),
            PlanningShapeDomain::Exact(_),
            PlanningShapeDomain::Exact(_)
        ) | (
            PlanningShapeDomain::HostContentAlternatives(_),
            PlanningShapeDomain::HostContentAlternatives(_),
            PlanningShapeDomain::HostContentAlternatives(_)
        )
    );
    if !same_variant
        || canonical.shapes().len() != statistics.shapes().len()
        || shapes.shapes().len() != statistics.shapes().len()
    {
        return Ok(None);
    }
    let mut evidence = Vec::new();
    evidence
        .try_reserve_exact(shapes.shapes().len())
        .map_err(|_| PlanningUnknownReason::ShapeCapacity)?;
    for ((exact, shape), selected) in canonical
        .shapes()
        .iter()
        .zip(shapes.shapes())
        .zip(statistics.shapes())
    {
        poll()?;
        let Some(bound) = PlanningCostEvidence::bind(exact, shape, selected) else {
            return Ok(None);
        };
        evidence.push(bound);
    }
    poll()?;
    Ok(Some(match canonical {
        PlanningShapeDomain::Exact(_) => PlanningShapeDomain::Exact(
            evidence
                .pop()
                .ok_or(PlanningUnknownReason::InvalidShapeEvidence)?,
        ),
        PlanningShapeDomain::HostContentAlternatives(_) => {
            PlanningShapeDomain::HostContentAlternatives(evidence)
        }
    }))
}
