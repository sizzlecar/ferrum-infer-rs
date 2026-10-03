//! Cold numerical inventory from real admitted roots. No model wave is prepared
//! or submitted; every result remains a hypothetical route, never a sample.
use super::cohort_driver::{ProbePrefillPlan, ProbeRequest};
use super::*;
use crate::continuous_engine::inner::{
    cost_observation::participant_host_features,
    slo_controller::{
        sampling::{self, FutureSamplingPolicy},
        shape::{projected_row_host, FutureHostMode},
    },
};
use ferrum_interfaces::{
    execution_cost::{
        host_history_cost_signature, ActualRowWork, ActualWaveKind, CostWorkloadDomainV1,
        HostContentForecastV2, HostCostFeaturesV1, HostPendingConstraintV2,
    },
    model_executor::{ExecutorResourcePlanningRequest, LogitsReturnPolicy},
    vnext::{
        ExecutionCostRouteAvailability as Availability, ExecutionCostRouteState,
        ExecutionCostRouteUnknown, ExecutionCostRouteView, FutureCostOutput,
        FutureHostPendingQueryV2, FutureHostPendingRowV2, FutureRepetitionRangeV3,
        FutureWaveCostQuery, FutureWaveCostRow, ResourcePlanningLimits, ResourcePlanningUnknown,
    },
};
use ferrum_scheduler::implementations::continuous::cost_model::structured_v2::{
    NumericalFamilyKeyV1, StructuredQueryV2, StructuredUnknownV2,
};
use std::num::NonZeroU32;
use tokio::time::Instant;

mod prefill_states;
use prefill_states::{PrefillReuse, PrefillStates};

mod known_prefix;
use known_prefix::PrefixRoots;
pub(in crate::continuous_engine::inner::calibration) use known_prefix::{
    GeometryPrefixCondition, GeometryPrefixConstraint,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::continuous_engine::inner::calibration) struct GeometryProjectionPoint {
    pub rows: usize,
    /// Includes the current decode token, as in DecodeContextBoundary.
    pub sequence_tokens: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::continuous_engine::inner::calibration) enum GeometryInputTarget {
    InitialPrefill {
        rows: usize,
    },
    /// A legal nonzero prompt offset reached from the original admitted root.
    PrefillSpan {
        rows: usize,
        offset: u32,
    },
    Decode(GeometryProjectionPoint),
}

pub(in crate::continuous_engine::inner::calibration) struct GeometryInputBranch {
    pub host_branch: Option<GeometryHostBranch>,
    /// Original checked projection state. A known configured-eager recipe is
    /// not a member of a population restricted to warm or disabled graphs.
    pub graph: ferrum_interfaces::execution_cost::ActualWaveGraphState,
    pub query: StructuredQueryV2,
}

pub(in crate::continuous_engine::inner::calibration) struct GeometryInputOutcome {
    pub scenario_index: usize,
    pub target: GeometryInputTarget,
    pub branches: Vec<GeometryInputBranch>,
    pub unknown: Option<GeometryProjectionUnknown>,
    pub prefix_condition: Option<GeometryPrefixCondition>,
}

pub(in crate::continuous_engine::inner::calibration) struct GeometryInputScenario<'a> {
    pub targets: &'a [GeometryInputTarget],
    pub prefixes: &'a [GeometryPrefixConstraint],
}

pub(in crate::continuous_engine::inner::calibration) struct GeometryInputReport {
    pub outcomes: Vec<GeometryInputOutcome>,
    pub admitted_requests: usize,
    pub projection_attempts: usize,
}

/// Only an uninterrupted initial traversal may retain complete checked inputs.
/// Partial traversals retain positions, never authority across a preparation.
pub(in crate::continuous_engine::inner::calibration) enum GeometryReadinessReport {
    Complete(GeometryInputReport),
    Progress {
        visited_targets: usize,
        gap: Option<(usize, GeometryInputTarget, GeometryProjectionUnknown)>,
        admitted_requests: usize,
        projection_attempts: usize,
    },
}

#[derive(Default)]
pub(in crate::continuous_engine::inner::calibration) struct GeometryProjectionCharge {
    pub admitted_requests: usize,
    pub projection_attempts: usize,
}

#[derive(Debug, Clone, Copy)]
pub(in crate::continuous_engine::inner::calibration) struct GeometryProjectionLimits {
    pub deadline: Instant,
    pub maximum_projections: usize,
    pub maximum_route_states: usize,
    /// Retained report and prefix trajectory payload, including validation
    /// scratch while binding prefixes. Temporary views/states obey the existing
    /// ResourcePlanningLimits and the explicit route-state count bound.
    pub maximum_retained_bytes: usize,
    pub prefill_chunk: NonZeroU32,
    pub prefill_row_ceiling: Option<NonZeroU32>,
}

/// The wave token budget is shared by all rows. A declared row ceiling caps
/// each row's share independently; it never reduces the legal wave width.
pub(in crate::continuous_engine::inner::calibration) fn prefill_chunk_for_width(
    whole_wave_tokens: NonZeroU32,
    prefill_row_ceiling: Option<NonZeroU32>,
    rows: usize,
) -> Option<NonZeroU32> {
    let rows = u32::try_from(rows).ok()?;
    let share = whole_wave_tokens.get().checked_div(rows)?;
    NonZeroU32::new(prefill_row_ceiling.map_or(share, |ceiling| share.min(ceiling.get())))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::continuous_engine::inner::calibration) enum GeometryHostBranch {
    Greedy,
    FullLogits,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::continuous_engine::inner::calibration) enum GeometryProjectionUnknown {
    BudgetExhausted,
    Capacity,
    Unreachable,
    MissingDomain,
    HostPolicyUnavailable,
    InvalidPrefix,
    Route(ExecutionCostRouteUnknown),
    Structured(StructuredUnknownV2),
    Admission(&'static str),
}
type GeometryResult<T> = std::result::Result<T, GeometryProjectionUnknown>;

pub(in crate::continuous_engine::inner::calibration) struct GeometryProjectionBranch {
    pub host_branch: GeometryHostBranch,
    pub query: StructuredQueryV2,
    pub family: NumericalFamilyKeyV1,
}

pub(in crate::continuous_engine::inner::calibration) struct GeometryProjectionOutcome {
    pub point: GeometryProjectionPoint,
    pub branches: Vec<GeometryProjectionBranch>,
    pub unknown: Option<GeometryProjectionUnknown>,
    pub prefix_condition: Option<GeometryPrefixCondition>,
}

pub(in crate::continuous_engine::inner::calibration) struct GeometryProjectionReport {
    pub outcomes: Vec<GeometryProjectionOutcome>,
    /// Successfully created real requests, including ones whose admission
    /// later fails. These consume the caller's original request budget.
    pub admitted_requests: usize,
    pub projection_attempts: usize,
}

struct HostRoot {
    participant: usize,
    prompt: u32,
    maximum_output: u32,
    host: HostCostFeaturesV1,
    policy_signature: [u8; 32],
    future: FutureSamplingPolicy,
}

/// One already checked hypothetical path, owned by a single scenario on one
/// captured view. It never crosses fresh owners, policies or prefix bindings.
/// The existing route-state limit still bounds both live state vectors.
struct GeometryTrajectory {
    width: usize,
    frontiers: Vec<u32>,
    states: Vec<Arc<ExecutionCostRouteState>>,
}

impl CalibrationSession {
    pub(in crate::continuous_engine::inner::calibration) async fn project_geometry(
        &mut self,
        requests: Vec<ProbeRequest>,
        points: &[GeometryProjectionPoint],
        limits: GeometryProjectionLimits,
    ) -> Result<GeometryProjectionReport> {
        self.project_geometry_with_prefixes(requests, points, limits, &[])
            .await
    }

    pub(in crate::continuous_engine::inner::calibration) async fn project_geometry_with_prefixes(
        &mut self,
        requests: Vec<ProbeRequest>,
        points: &[GeometryProjectionPoint],
        limits: GeometryProjectionLimits,
        prefixes: &[GeometryPrefixConstraint],
    ) -> Result<GeometryProjectionReport> {
        let overlap = std::mem::size_of::<GeometryProjectionReport>()
            .checked_add(
                points
                    .len()
                    .checked_mul(std::mem::size_of::<GeometryProjectionOutcome>())
                    .ok_or_else(|| FerrumError::resource_exhausted("geometry report capacity"))?,
            )
            .and_then(|n| {
                n.checked_add(
                    points
                        .len()
                        .checked_mul(std::mem::size_of::<GeometryInputTarget>())?,
                )
            })
            .filter(|n| *n <= limits.maximum_retained_bytes)
            .ok_or_else(|| FerrumError::resource_exhausted("geometry report capacity"))?;
        let targets: Vec<_> = points
            .iter()
            .copied()
            .map(GeometryInputTarget::Decode)
            .collect();
        let report = self
            .project_geometry_inputs_inner(
                requests,
                &[GeometryInputScenario {
                    targets: &targets,
                    prefixes,
                }],
                limits,
                overlap,
                PrefillReuse::Share,
            )
            .await?;
        let mut outcomes = Vec::with_capacity(report.outcomes.len());
        for outcome in report.outcomes {
            let GeometryInputTarget::Decode(point) = outcome.target else {
                unreachable!()
            };
            let branches: GeometryResult<Vec<_>> = outcome
                .branches
                .into_iter()
                .map(|branch| {
                    poll_deadline(&limits)?;
                    Ok(GeometryProjectionBranch {
                        family: branch
                            .query
                            .input()
                            .numerical_family_key()
                            .map_err(GeometryProjectionUnknown::Structured)?,
                        host_branch: branch
                            .host_branch
                            .ok_or(GeometryProjectionUnknown::HostPolicyUnavailable)?,
                        query: branch.query,
                    })
                })
                .collect();
            let (branches, unknown) = match branches {
                Ok(branches) => (branches, outcome.unknown),
                Err(reason) => (Vec::new(), Some(reason)),
            };
            outcomes.push(GeometryProjectionOutcome {
                point,
                branches,
                unknown,
                prefix_condition: outcome.prefix_condition,
            });
        }
        Ok(GeometryProjectionReport {
            outcomes,
            admitted_requests: report.admitted_requests,
            projection_attempts: report.projection_attempts,
        })
    }

    pub(in crate::continuous_engine::inner::calibration) async fn project_geometry_inputs(
        &mut self,
        requests: Vec<ProbeRequest>,
        targets: &[GeometryInputTarget],
        limits: GeometryProjectionLimits,
        prefixes: &[GeometryPrefixConstraint],
    ) -> Result<GeometryInputReport> {
        self.project_geometry_inputs_inner(
            requests,
            &[GeometryInputScenario { targets, prefixes }],
            limits,
            0,
            PrefillReuse::Share,
        )
        .await
    }

    pub(in crate::continuous_engine::inner::calibration) async fn project_geometry_input_scenarios(
        &mut self,
        requests: Vec<ProbeRequest>,
        scenarios: &[GeometryInputScenario<'_>],
        limits: GeometryProjectionLimits,
    ) -> Result<GeometryInputReport> {
        self.project_geometry_inputs_inner(requests, scenarios, limits, 0, PrefillReuse::Share)
            .await
    }

    async fn project_geometry_inputs_inner(
        &mut self,
        requests: Vec<ProbeRequest>,
        scenarios: &[GeometryInputScenario<'_>],
        limits: GeometryProjectionLimits,
        legacy_overlap: usize,
        prefill_reuse: PrefillReuse,
    ) -> Result<GeometryInputReport> {
        self.project_geometry_inputs_mode(
            requests,
            scenarios,
            limits,
            legacy_overlap,
            prefill_reuse,
            None,
            false,
            None,
            None,
        )
        .await
    }

    pub(in crate::continuous_engine::inner::calibration) async fn project_geometry_readiness_scenarios(
        &mut self,
        requests: Vec<ProbeRequest>,
        scenarios: &[GeometryInputScenario<'_>],
        limits: GeometryProjectionLimits,
        first_target: usize,
        retain_complete: bool,
        should_stop: &(dyn Fn(usize, GeometryInputTarget, &GeometryProjectionUnknown) -> bool
              + Sync),
        charge: &mut GeometryProjectionCharge,
    ) -> Result<GeometryReadinessReport> {
        if retain_complete && first_target != 0 {
            return Err(FerrumError::invalid_request(
                "only an initial readiness traversal can retain complete inputs",
            ));
        }
        let mut report = self
            .project_geometry_inputs_mode(
                requests,
                scenarios,
                limits,
                0,
                PrefillReuse::Share,
                Some(first_target),
                retain_complete,
                Some(should_stop),
                Some(charge),
            )
            .await?;
        if retain_complete
            && report.outcomes.len() == scenarios.iter().map(|s| s.targets.len()).sum::<usize>()
            && !report.outcomes.iter().any(|outcome| {
                outcome.unknown.as_ref().is_some_and(|reason| {
                    readiness_fatal(reason)
                        || should_stop(outcome.scenario_index, outcome.target, reason)
                })
            })
        {
            return Ok(GeometryReadinessReport::Complete(report));
        }
        Ok(GeometryReadinessReport::Progress {
            visited_targets: report.outcomes.len(),
            gap: report.outcomes.pop().and_then(|outcome| {
                outcome
                    .unknown
                    .map(|reason| (outcome.scenario_index, outcome.target, reason))
            }),
            admitted_requests: report.admitted_requests,
            projection_attempts: report.projection_attempts,
        })
    }

    pub(in crate::continuous_engine::inner::calibration) async fn project_geometry_input_scenarios_charged(
        &mut self,
        requests: Vec<ProbeRequest>,
        scenarios: &[GeometryInputScenario<'_>],
        limits: GeometryProjectionLimits,
        charge: &mut GeometryProjectionCharge,
    ) -> Result<GeometryInputReport> {
        self.project_geometry_inputs_mode(
            requests,
            scenarios,
            limits,
            0,
            PrefillReuse::Share,
            None,
            false,
            None,
            Some(charge),
        )
        .await
    }

    async fn project_geometry_inputs_mode(
        &mut self,
        requests: Vec<ProbeRequest>,
        scenarios: &[GeometryInputScenario<'_>],
        limits: GeometryProjectionLimits,
        legacy_overlap: usize,
        prefill_reuse: PrefillReuse,
        readiness_start: Option<usize>,
        retain_complete: bool,
        readiness_stop: Option<
            &(dyn Fn(usize, GeometryInputTarget, &GeometryProjectionUnknown) -> bool + Sync),
        >,
        charge: Option<&mut GeometryProjectionCharge>,
    ) -> Result<GeometryInputReport> {
        self.completed_owner_boundary()?;
        if !self.engine.inner.manual_calibration_driver
            || self.engine.inner.bg_loop_spawned.load(Ordering::Acquire)
            || self.engine.inner.is_running.load(Ordering::Acquire)
            || self.engine.inner.shutdown_started.load(Ordering::Acquire)
            || self.prefix_source5
            || (self.prefix_source8 && !self.startup_inventory_active())
            || self.prefix_preparation.is_some()
            || self.selected_capture_identity.is_some()
            || self.structured_capture.is_some()
            || self.structured_capture_v2.is_some()
            || self.structured_group_v2.is_some()
            || self.prepared_owner_capture.is_some()
            || requests.is_empty()
            || requests.len() > self.limits.maximum_requests().get()
            || requests.len() > 128
            || scenarios.is_empty()
            || scenarios.iter().any(|scenario| scenario.targets.is_empty())
            || limits.maximum_projections == 0
            || limits.maximum_route_states == 0
            || limits.maximum_route_states > 256
        {
            return Err(FerrumError::invalid_request(
                "geometry requires an unused bounded isolated session",
            ));
        }
        for (i, request) in requests.iter().enumerate() {
            if requests[..i]
                .iter()
                .any(|old| old.request.id == request.request.id)
            {
                return Err(FerrumError::invalid_request(
                    "geometry request identities must be distinct",
                ));
            }
        }
        let target_count = scenarios
            .iter()
            .try_fold(0usize, |n, scenario| n.checked_add(scenario.targets.len()))
            .ok_or_else(|| FerrumError::resource_exhausted("geometry report capacity"))?;
        if readiness_start.is_some_and(|start| start >= target_count) {
            return Err(FerrumError::invalid_request(
                "readiness cursor exceeds original targets",
            ));
        }
        let shared_state_overhead = PrefillStates::overhead_bytes(
            requests.len(),
            limits.maximum_route_states,
            prefill_reuse,
        )
        .ok_or_else(|| FerrumError::resource_exhausted("geometry shared-state capacity"))?;
        let base_bytes = target_count
            .checked_mul(std::mem::size_of::<GeometryInputOutcome>())
            .and_then(|n| n.checked_add(std::mem::size_of::<GeometryInputReport>()))
            .and_then(|n| n.checked_add(legacy_overlap))
            .and_then(|n| n.checked_add(shared_state_overhead))
            .filter(|n| *n <= limits.maximum_retained_bytes)
            .ok_or_else(|| FerrumError::resource_exhausted("geometry report capacity"))?;
        let mut report = GeometryInputReport {
            outcomes: Vec::with_capacity(target_count),
            admitted_requests: 0,
            projection_attempts: 0,
        };
        let mut outputs = Vec::with_capacity(requests.len());
        let ids: Vec<_> = requests.iter().map(|r| r.request.id.clone()).collect();
        // This async block returns GeometryResult; its Unknown values become
        // report outcomes after mandatory owner cleanup below.
        let result: GeometryResult<()> = tokio::time::timeout_at(limits.deadline, async {
            let domain = self
                .engine
                .inner
                .cost_runtime
                .as_ref()
                .and_then(|r| r.workload_domain())
                .cloned()
                .ok_or(GeometryProjectionUnknown::MissingDomain)?;
            for request in requests {
                if Instant::now() >= limits.deadline {
                    return Err(GeometryProjectionUnknown::BudgetExhausted);
                }
                outputs.push(
                    self.add_request(
                        request.request,
                        InferenceRequestContext::capture(),
                        request.contract,
                    )
                    .await
                    .map_err(|error| {
                        tracing::warn!(%error, admitted_requests = report.admitted_requests,
                            requested_requests = ids.len(), stage = "request_creation",
                            "Calibration input projection could not create its real request");
                        #[cfg(test)]
                        eprintln!("calibration geometry request_creation error={error:?} admitted={} requested={}", report.admitted_requests, ids.len());
                        GeometryProjectionUnknown::Admission("request_creation")
                    })?,
                );
                report.admitted_requests += 1;
            }
            let view = loop {
                if Instant::now() >= limits.deadline {
                    return Err(GeometryProjectionUnknown::BudgetExhausted);
                }
                self.step(CalibrationAction::AdmitOne)
                    .await
                    .map_err(|error| {
                        tracing::warn!(%error, admitted_requests = report.admitted_requests,
                            requested_requests = ids.len(), stage = "admission_turn",
                            "Calibration input projection could not admit its real requests");
                        GeometryProjectionUnknown::Admission("admission_turn")
                    })?;
                if self.engine.inner.scheduler.active_count() != ids.len()
                    || self.engine.inner.scheduler.waiting_count() != 0
                {
                    tokio::task::yield_now().await;
                    continue;
                }
                let requested: Vec<_> = ids
                    .iter()
                    .map(|id| ExecutorResourcePlanningRequest {
                        request_id: id,
                        cache_id: None,
                    })
                    .collect();
                match self.engine.inner.model_executor.execution_cost_route_view(
                    &requested,
                    ResourcePlanningLimits {
                        maximum_projected_waves: 256,
                        ..Default::default()
                    },
                    &mut || Instant::now() < limits.deadline,
                ) {
                    Availability::Known(view) => break view.with_structured_capture(true),
                    Availability::Unknown(ExecutionCostRouteUnknown::Resource(
                        ResourcePlanningUnknown::BusyOrUnavailable
                        | ResourcePlanningUnknown::ReadUnavailable(_),
                    )) => tokio::task::yield_now().await,
                    Availability::Unknown(reason) => {
                        return Err(GeometryProjectionUnknown::Route(reason))
                    }
                }
            };
            let roots = self.geometry_host_roots(&ids, &view)?;
            // Scope binds reuse to this exact captured view and original roots.
            let mut prefill_states = PrefillStates::new(
                roots.len(), limits.maximum_route_states, prefill_reuse,
            )?;
            let mut retained = base_bytes;
            let mut skip_targets = readiness_start.unwrap_or(0);
            'scenarios: for (scenario_index, scenario) in scenarios.iter().enumerate() {
                if skip_targets >= scenario.targets.len() {
                    skip_targets -= scenario.targets.len();
                    continue;
                }
                let first_target = std::mem::take(&mut skip_targets);
                let prefix_roots = match PrefixRoots::bind(
                    self.engine.inner.tokenizer.as_ref(),
                    &ids,
                    &roots,
                    scenario.prefixes,
                    &limits,
                    limits.maximum_retained_bytes - retained,
                ) {
                    Ok(prefixes) => prefixes,
                    Err(reason) => {
                        for &target in &scenario.targets[first_target..] {
                            report.outcomes.push(GeometryInputOutcome {
                                scenario_index,
                                target,
                                branches: Vec::new(),
                                unknown: Some(reason.clone()),
                                prefix_condition: None,
                            });
                            if readiness_stop.is_some_and(|stop| readiness_fatal(&reason) || stop(scenario_index, target, &reason)) {
                                break 'scenarios;
                            }
                        }
                        continue;
                    }
                };
                let prefix_bytes = prefix_roots.retained_payload_bytes();
                let mut trajectory = None;
                for (target_index, &target) in scenario.targets.iter().enumerate().skip(first_target) {
                    let condition = match target {
                        GeometryInputTarget::Decode(point) => prefix_roots.condition(&roots, point),
                        GeometryInputTarget::InitialPrefill { .. }
                        | GeometryInputTarget::PrefillSpan { .. } => Ok(None),
                    };
                    let prefix_condition = condition.as_ref().ok().copied().flatten();
                    let projected = condition.and_then(|_| {
                        project_point(
                            &self.engine.inner,
                            &view,
                            &roots,
                            &domain,
                            &prefix_roots,
                            target,
                            &limits,
                            &mut report.projection_attempts,
                            limits.maximum_retained_bytes - retained - prefix_bytes,
                            legacy_overlap != 0,
                            &mut trajectory,
                            &mut prefill_states,
                            scenario.targets.get(target_index + 1).is_some_and(|next| {
                                matches!((target, *next),
                                    (GeometryInputTarget::Decode(a), GeometryInputTarget::Decode(b))
                                        if a.rows == b.rows && a.sequence_tokens < b.sequence_tokens)
                            }),
                        )
                    });
                    let (branches, unknown) = match projected {
                        Ok((branches, bytes)) => {
                            if readiness_start.is_some() && !retain_complete {
                                // A resumed traversal carries no input authority across
                                // preparation. The final inventory uses a fresh capture.
                                (Vec::new(), None)
                            } else {
                                retained += bytes;
                                (branches, None)
                            }
                        }
                        Err(reason) => {
                            // Failed or terminal targets do not carry a
                            // partially advanced path to later targets.
                            trajectory = None;
                            (Vec::new(), Some(reason))
                        }
                    };
                    let stop = unknown.as_ref().is_some_and(|reason| {
                        readiness_stop.is_some_and(|stop| readiness_fatal(reason) || stop(scenario_index, target, reason))
                    });
                    report.outcomes.push(GeometryInputOutcome {
                        scenario_index,
                        target,
                        branches,
                        unknown,
                        prefix_condition,
                    });
                    if stop {
                        break 'scenarios;
                    }
                }
            }
            Ok::<(), GeometryProjectionUnknown>(())
        })
        .await
        .unwrap_or(Err(GeometryProjectionUnknown::BudgetExhausted));
        if let Some(charge) = charge {
            charge.admitted_requests = report.admitted_requests;
            charge.projection_attempts = report.projection_attempts;
        }
        // Dropping frames cancels only these real owners. Cleanup remains
        // outside the projection timeout; the startup caller also drains if
        // this whole future is cancelled by its enclosing preflight deadline.
        drop(outputs);
        self.drain_startup_geometry().await?;
        if let Err(reason) = result {
            let finished = report.outcomes.len() + readiness_start.unwrap_or(0);
            for (scenario_index, target) in scenarios
                .iter()
                .enumerate()
                .flat_map(|(index, scenario)| {
                    scenario.targets.iter().map(move |target| (index, *target))
                })
                .skip(finished)
            {
                report.outcomes.push(GeometryInputOutcome {
                    scenario_index,
                    target,
                    branches: Vec::new(),
                    unknown: Some(reason.clone()),
                    prefix_condition: None,
                });
                if readiness_stop.is_some_and(|stop| {
                    readiness_fatal(&reason) || stop(scenario_index, target, &reason)
                }) {
                    break;
                }
            }
        }
        Ok(report)
    }

    fn geometry_host_roots(
        &self,
        ids: &[RequestId],
        view: &ExecutionCostRouteView,
    ) -> GeometryResult<Vec<HostRoot>> {
        let sequences = self
            .engine
            .inner
            .sequences
            .try_read()
            .ok_or(GeometryProjectionUnknown::HostPolicyUnavailable)?;
        let mut roots = Vec::with_capacity(ids.len());
        for (participant, id) in ids.iter().enumerate() {
            let sequence = sequences
                .get(id)
                .ok_or(GeometryProjectionUnknown::HostPolicyUnavailable)?;
            // This entry only captures a new prefill-ready root. Actual decode
            // histories and known prefix constraints require a separate binding.
            if sequence.prefill_complete
                || sequence.prefill_tokens_processed != 0
                || !sequence.generated_tokens.is_empty()
            {
                return Err(GeometryProjectionUnknown::Unreachable);
            }
            let host = participant_host_features(sequence)
                .ok_or(GeometryProjectionUnknown::HostPolicyUnavailable)?;
            let actual = sequence.model_decode_logits_policy();
            let future = sampling::future_policy(
                sequence,
                &actual,
                self.engine.inner.model_executor.info().vocab_size as u64,
            )
            .map_err(|_| GeometryProjectionUnknown::HostPolicyUnavailable)?
            .ok_or(GeometryProjectionUnknown::HostPolicyUnavailable)?;
            roots.push(HostRoot {
                participant,
                prompt: u32::try_from(sequence.prefill_context_len())
                    .map_err(|_| GeometryProjectionUnknown::Capacity)?,
                maximum_output: u32::try_from(sequence.sampling_params.max_tokens)
                    .map_err(|_| GeometryProjectionUnknown::Capacity)?,
                host,
                future,
                policy_signature: sequence
                    .cost_policy_signature
                    .ok_or(GeometryProjectionUnknown::HostPolicyUnavailable)?,
            });
        }
        roots.sort_unstable_by_key(|r| view.participant_authority(r.participant));
        Ok(roots)
    }
}

fn claim_projection(limits: &GeometryProjectionLimits, attempts: &mut usize) -> GeometryResult<()> {
    if Instant::now() >= limits.deadline || *attempts >= limits.maximum_projections {
        return Err(GeometryProjectionUnknown::BudgetExhausted);
    }
    *attempts += 1;
    Ok(())
}

fn readiness_fatal(reason: &GeometryProjectionUnknown) -> bool {
    matches!(
        reason,
        GeometryProjectionUnknown::BudgetExhausted
            | GeometryProjectionUnknown::Capacity
            | GeometryProjectionUnknown::Admission(_)
    )
}

fn poll_deadline(limits: &GeometryProjectionLimits) -> GeometryResult<()> {
    if Instant::now() >= limits.deadline {
        Err(GeometryProjectionUnknown::BudgetExhausted)
    } else {
        Ok(())
    }
}

fn project_point(
    engine: &EngineInner,
    view: &ExecutionCostRouteView,
    roots: &[HostRoot],
    domain: &CostWorkloadDomainV1,
    prefixes: &PrefixRoots,
    target_input: GeometryInputTarget,
    limits: &GeometryProjectionLimits,
    attempts: &mut usize,
    retained_limit: usize,
    legacy_overlap: bool,
    trajectory: &mut Option<GeometryTrajectory>,
    prefill_states: &mut PrefillStates,
    keep_successor: bool,
) -> GeometryResult<(Vec<GeometryInputBranch>, usize)> {
    poll_deadline(limits)?;
    let (width, decode) = match target_input {
        GeometryInputTarget::InitialPrefill { rows }
        | GeometryInputTarget::PrefillSpan { rows, .. } => (rows, None),
        GeometryInputTarget::Decode(point) => (point.rows, Some(point)),
    };
    // Select original cohort slots before preserving provider authority order.
    // A smaller width must not accidentally borrow a different mixed-prefix row.
    let roots: Vec<_> = roots
        .iter()
        .filter(|root| root.participant < width)
        .collect();
    if roots.len() != width || width == 0 {
        return Err(GeometryProjectionUnknown::Unreachable);
    }
    let target = match decode {
        Some(point) => Some(
            point
                .sequence_tokens
                .checked_sub(1)
                .ok_or(GeometryProjectionUnknown::Unreachable)?,
        ),
        None => None,
    };
    for root in &roots {
        if root.prompt == 0 {
            return Err(GeometryProjectionUnknown::Unreachable);
        }
        if let Some(point) = decode {
            let generated = point
                .sequence_tokens
                .checked_sub(root.prompt)
                .ok_or(GeometryProjectionUnknown::Unreachable)?;
            if root.prompt == 0 || generated == 0 || generated >= root.maximum_output {
                return Err(GeometryProjectionUnknown::Unreachable);
            }
        }
    }
    u32::try_from(width).map_err(|_| GeometryProjectionUnknown::Capacity)?;
    let per_row_chunk =
        prefill_chunk_for_width(limits.prefill_chunk, limits.prefill_row_ceiling, width)
            .ok_or(GeometryProjectionUnknown::Unreachable)?
            .get();
    let prefill_offset = match target_input {
        GeometryInputTarget::InitialPrefill { .. } => Some(0),
        GeometryInputTarget::PrefillSpan { offset, .. } => {
            if offset == 0
                || offset % per_row_chunk != 0
                || roots.iter().any(|root| offset >= root.prompt)
            {
                return Err(GeometryProjectionUnknown::Unreachable);
            }
            Some(offset)
        }
        GeometryInputTarget::Decode(_) => None,
    };
    // The target is a joint decode, not a requirement that its preparation
    // was one batched prefill. Match the declared source8 prefix driver and
    // chain each real checked state; ordinary prefill targets stay joint.
    let prefill_plan = if decode.is_some() {
        ProbePrefillPlan::PreparedSequentialV1
    } else {
        ProbePrefillPlan::Joint
    };
    // A smaller/equal or different-width target starts from the same original
    // root. A later target may consume the exact successor states of the prior
    // projected wave; an earlier error never installs a partial successor.
    let resume = trajectory.take().filter(|previous| {
        previous.width == width
            && target.is_some_and(|target| previous.frontiers.iter().all(|kv| *kv <= target))
    });
    let (mut frontiers, mut states) = if let Some(previous) = resume {
        prefill_states.leave_room(previous.states.len());
        (previous.frontiers, previous.states)
    } else {
        prefill_states.leave_room(1);
        let sequential = decode.and_then(|_| prefill_states.sequential(&roots, limits));
        let joint =
            prefill_offset.and_then(|offset| prefill_states.joint(width, offset / per_row_chunk));
        let (mut frontiers, mut state) = if let Some((prepared_width, state)) = sequential {
            (
                roots
                    .iter()
                    .map(|root| {
                        if root.participant < prepared_width {
                            root.prompt
                        } else {
                            0
                        }
                    })
                    .collect(),
                state,
            )
        } else if let Some((completed_waves, state)) = joint {
            (
                roots
                    .iter()
                    .map(|root| {
                        (u64::from(completed_waves) * u64::from(per_row_chunk))
                            .min(u64::from(root.prompt)) as u32
                    })
                    .collect(),
                state,
            )
        } else {
            (vec![0u32; roots.len()], Arc::new(view.initial_state()))
        };
        // Advance the actual prompt only by legal prefill spans. A target context
        // never rewrites the captured frontier or invents a different prompt.
        while roots.iter().zip(&frontiers).any(|(r, p)| *p < r.prompt) {
            let mut rows = Vec::with_capacity(roots.len());
            // Original cohort slots also order the real driver. Authority order is
            // retained for joint queries; a single-row preparation cannot reorder
            // admission-derived allocation dependencies between slots.
            let serial_participant = decode
                .is_some()
                .then(|| {
                    roots
                        .iter()
                        .zip(&frontiers)
                        .filter(|(root, offset)| **offset < root.prompt)
                        .map(|(root, _)| root.participant)
                        .min()
                })
                .flatten();
            for (root, offset) in roots.iter().zip(&mut frontiers) {
                if serial_participant.is_some_and(|selected| selected != root.participant) {
                    continue;
                }
                let count = (root.prompt - *offset).min(per_row_chunk);
                if count == 0 {
                    continue;
                }
                rows.push(FutureWaveCostRow {
                    participant_index: root.participant,
                    work: ActualRowWork::Prefill {
                        offset: *offset,
                        count,
                        total_prompt_tokens: root.prompt,
                    },
                    host_policy_signature: host_history_cost_signature(root.policy_signature, 0),
                    host_features: Some(root.host),
                    output: FutureCostOutput::Prefill {
                        final_logits: *offset + count == root.prompt,
                    },
                });
                *offset += count;
                if rows.len() == prefill_plan.rows_per_wave(width) {
                    break;
                }
            }
            claim_projection(limits, attempts)?;
            let projected = match engine.model_executor.project_execution_cost_wave(
                view,
                &state,
                &FutureWaveCostQuery {
                    kind: ActualWaveKind::Prefill,
                    rows: &rows,
                },
                &mut || Instant::now() < limits.deadline,
            ) {
                Availability::Known(value) => value,
                Availability::Unknown(reason) => {
                    return Err(GeometryProjectionUnknown::Route(reason))
                }
            };
            poll_deadline(limits)?;
            if prefill_offset.is_some_and(|target_offset| {
                rows.iter().all(|row| {
                    matches!(row.work,
                    ActualRowWork::Prefill { offset, .. } if offset == target_offset)
                })
            }) {
                let selected = projected
                    .statistical_evidence
                    .as_ref()
                    .ok_or(GeometryProjectionUnknown::HostPolicyUnavailable)?;
                let recipe = selected
                    .structured_capture()
                    .ok_or(GeometryProjectionUnknown::HostPolicyUnavailable)?
                    .map_err(|_| GeometryProjectionUnknown::HostPolicyUnavailable)?;
                let query = StructuredQueryV2::from_future_with_domain(
                    &projected.shape,
                    selected,
                    recipe,
                    &HostContentForecastV2::Exact,
                    domain,
                )
                .map_err(GeometryProjectionUnknown::Structured)?;
                let retained = query
                    .retained_payload_bytes()
                    .and_then(|n| {
                        n.checked_add(
                            std::mem::size_of::<GeometryInputBranch>()
                                - std::mem::size_of::<StructuredQueryV2>(),
                        )
                    })
                    .filter(|n| *n <= retained_limit)
                    .ok_or(GeometryProjectionUnknown::Capacity)?;
                poll_deadline(limits)?;
                // Only the complete observed target may install a successor.
                // An ancestor projection followed by missing evidence, query
                // validation, capacity or deadline failure leaves no new path.
                let completed_waves =
                    prefill_offset.ok_or(GeometryProjectionUnknown::Unreachable)? / per_row_chunk
                        + 1;
                let successor = Arc::new(projected.state);
                prefill_states.insert_joint(width, completed_waves, &successor);
                return Ok((
                    vec![GeometryInputBranch {
                        host_branch: None,
                        graph: projected.shape.graph,
                        query,
                    }],
                    retained,
                ));
            }
            state = Arc::new(projected.state);
        }
        if decode.is_some() {
            prefill_states.insert_sequential(width, &state);
        }
        (frontiers, vec![state])
    };
    poll_deadline(limits)?;
    let target = target.ok_or(GeometryProjectionUnknown::Unreachable)?;
    loop {
        let final_wave = frontiers.iter().all(|n| *n == target);
        let selected: Vec<_> = frontiers
            .iter()
            .enumerate()
            .filter_map(|(i, &kv)| (final_wave || kv < target).then_some((i, kv)))
            .collect();
        if selected.is_empty() {
            return Err(GeometryProjectionUnknown::Unreachable);
        }
        let forced_full = selected.iter().any(|&(i, kv)| {
            roots[i].future.policy.requires_full_logits()
                || prefixes
                    .at(roots[i].participant, kv - roots[i].prompt + 1)
                    .is_some_and(|p| p.pending)
        });
        let all_known = selected.iter().all(|&(i, kv)| {
            prefixes
                .at(roots[i].participant, kv - roots[i].prompt + 1)
                .is_some()
        });
        let modes: &[FutureHostMode] = if forced_full {
            &[FutureHostMode::FullLogits]
        } else if all_known {
            &[FutureHostMode::Greedy]
        } else {
            &[FutureHostMode::Greedy, FutureHostMode::FullLogits]
        };
        let branch_capacity = states
            .len()
            .checked_mul(modes.len())
            .ok_or(GeometryProjectionUnknown::Capacity)?;
        let mut retained = if final_wave {
            branch_capacity
                .checked_mul(
                    std::mem::size_of::<GeometryInputBranch>()
                        + if legacy_overlap {
                            std::mem::size_of::<GeometryProjectionBranch>()
                        } else {
                            0
                        },
                )
                .filter(|n| *n <= retained_limit)
                .ok_or(GeometryProjectionUnknown::Capacity)?
        } else {
            0
        };
        let mut branches = Vec::with_capacity(if final_wave { branch_capacity } else { 0 });
        let mut next = Vec::with_capacity(limits.maximum_route_states);
        let mut can_resume = keep_successor;
        for previous in &states {
            for &mode in modes {
                let mut rows = Vec::with_capacity(selected.len());
                let mut eligible = Vec::with_capacity(selected.len());
                let full = LogitsReturnPolicy::FullLogits;
                for (position, &(index, kv)) in selected.iter().enumerate() {
                    let root = &roots[index];
                    let generated = kv - root.prompt + 1;
                    let known = prefixes.at(root.participant, generated);
                    let mut host = projected_row_host(
                        Some(root.host),
                        generated,
                        root.maximum_output,
                        true,
                        mode,
                    )
                    .flatten()
                    .ok_or(GeometryProjectionUnknown::HostPolicyUnavailable)?;
                    if let Some(known) = known {
                        host.state.pending_decoded_utf8 = known.pending;
                    }
                    // A fixed clean row retains its own clean policy even in
                    // a FullLogits wave forced by another pending/full peer.
                    let row_mode = if let Some(known) = known {
                        if known.pending || root.future.policy.requires_full_logits() {
                            FutureHostMode::FullLogits
                        } else {
                            FutureHostMode::Greedy
                        }
                    } else {
                        mode
                    };
                    let output = match row_mode {
                        FutureHostMode::FullLogits => {
                            if known.is_none() {
                                eligible.push(FutureHostPendingRowV2 {
                                    physical_position: position as u32,
                                    clean_policy: &root.future.policy,
                                });
                            }
                            FutureCostOutput::Decode { policy: &full }
                        }
                        FutureHostMode::Greedy => match root.future.repetition {
                            Some((_, penalty, vocabulary)) => {
                                let LogitsReturnPolicy::GreedyArgmax { token_mask, .. } =
                                    &root.future.policy
                                else {
                                    return Err(GeometryProjectionUnknown::HostPolicyUnavailable);
                                };
                                let (observed_unique, observed_generated) = known.map_or_else(
                                    || prefixes.repetition_anchor(root),
                                    |point| (point.unique, u64::from(generated)),
                                );
                                FutureCostOutput::ProjectedGreedy {
                                    token_mask: token_mask.as_ref(),
                                    repetition_penalty: penalty,
                                    repetition: FutureRepetitionRangeV3::from_observed_history(
                                        observed_unique,
                                        observed_generated,
                                        u64::from(generated),
                                        vocabulary,
                                    )
                                    .map_err(|_| {
                                        GeometryProjectionUnknown::HostPolicyUnavailable
                                    })?,
                                }
                            }
                            None => FutureCostOutput::Decode {
                                policy: &root.future.policy,
                            },
                        },
                        FutureHostMode::Exact => {
                            return Err(GeometryProjectionUnknown::HostPolicyUnavailable)
                        }
                    };
                    rows.push(FutureWaveCostRow {
                        participant_index: root.participant,
                        work: ActualRowWork::Decode { kv_tokens: kv },
                        host_features: Some(host),
                        output,
                        host_policy_signature: host_history_cost_signature(
                            root.policy_signature,
                            u64::from(generated),
                        ),
                    });
                }
                claim_projection(limits, attempts)?;
                let projected = match engine
                    .model_executor
                    .project_execution_cost_wave_with_host_content(
                        view,
                        previous.as_ref(),
                        &FutureWaveCostQuery {
                            kind: ActualWaveKind::Decode,
                            rows: &rows,
                        },
                        &FutureHostPendingQueryV2 {
                            eligible_rows: &eligible,
                            constraint: if forced_full || mode == FutureHostMode::Greedy {
                                HostPendingConstraintV2::AnySubset
                            } else {
                                HostPendingConstraintV2::NonEmptySubset
                            },
                        },
                        &mut || Instant::now() < limits.deadline,
                    ) {
                    Availability::Known(value) => value,
                    Availability::Unknown(reason) => {
                        return Err(GeometryProjectionUnknown::Route(reason))
                    }
                };
                poll_deadline(limits)?;
                if final_wave {
                    let selected = projected
                        .projection
                        .statistical_evidence
                        .as_ref()
                        .ok_or(GeometryProjectionUnknown::HostPolicyUnavailable)?;
                    let recipe = selected
                        .structured_capture()
                        .ok_or(GeometryProjectionUnknown::HostPolicyUnavailable)?
                        .map_err(|_| GeometryProjectionUnknown::HostPolicyUnavailable)?;
                    let query = StructuredQueryV2::from_future_with_domain(
                        &projected.projection.shape,
                        selected,
                        recipe,
                        &projected.host_content,
                        domain,
                    )
                    .map_err(GeometryProjectionUnknown::Structured)?;
                    retained = retained
                        .checked_add(
                            query
                                .retained_payload_bytes()
                                .ok_or(GeometryProjectionUnknown::Capacity)?
                                .checked_sub(std::mem::size_of::<StructuredQueryV2>())
                                .ok_or(GeometryProjectionUnknown::Capacity)?,
                        )
                        .filter(|n| *n <= retained_limit)
                        .ok_or(GeometryProjectionUnknown::Capacity)?;
                    poll_deadline(limits)?;
                    branches.push(GeometryInputBranch {
                        graph: projected.projection.shape.graph,
                        host_branch: Some(if mode == FutureHostMode::Greedy {
                            GeometryHostBranch::Greedy
                        } else {
                            GeometryHostBranch::FullLogits
                        }),
                        query,
                    });
                    if can_resume
                        && retain_state(&mut next, projected.projection.state, limits).is_err()
                    {
                        // A final query never previously required a successor
                        // state set. Preserve that query if an optional cache
                        // cannot represent every branch; the next target then
                        // replays from its original root under the same limits.
                        next.clear();
                        can_resume = false;
                    }
                } else {
                    retain_state(&mut next, projected.projection.state, limits)?;
                }
            }
        }
        // Cache occupies the original active-vector allowance; never shrink
        // the next branch set to keep a cache entry alive.
        prefill_states.leave_room(next.len().max(states.len()));
        if final_wave {
            poll_deadline(limits)?;
            if can_resume {
                for (index, _) in selected {
                    frontiers[index] = frontiers[index]
                        .checked_add(1)
                        .ok_or(GeometryProjectionUnknown::Capacity)?;
                }
                *trajectory = Some(GeometryTrajectory {
                    width,
                    frontiers,
                    states: next,
                });
            }
            return Ok((branches, retained));
        }
        states = next;
        for (index, _) in selected {
            frontiers[index] += 1;
        }
    }
}

fn retain_state(
    states: &mut Vec<Arc<ExecutionCostRouteState>>,
    next: ExecutionCostRouteState,
    limits: &GeometryProjectionLimits,
) -> GeometryResult<()> {
    for previous in states.iter() {
        if previous
            .as_ref()
            .same_future_state(&next, &mut || Instant::now() < limits.deadline)
            .map_err(GeometryProjectionUnknown::Route)?
        {
            return Ok(());
        }
    }
    if states.len() >= limits.maximum_route_states {
        return Err(GeometryProjectionUnknown::Capacity);
    }
    states.push(Arc::new(next));
    Ok(())
}

#[cfg(test)]
pub(in crate::continuous_engine::inner::calibration) mod tests;

#[cfg(test)]
mod input_tests;

#[cfg(test)]
mod scenario_tests;

#[cfg(test)]
mod trajectory_tests;

#[cfg(test)]
mod legal_prefill_tests;
