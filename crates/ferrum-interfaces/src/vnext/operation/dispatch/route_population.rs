//! Passive classification at the actual prepared-wave program selector.
use super::*;
use crate::execution_cost::{
    PreparedCostRouteClassV1 as C, PreparedCostRouteReasonV1 as U, PreparedCostRouteV1,
    PreparedReusableCostSelection,
};
use crate::vnext::{DeviceCostGraphCaptureCapability, IndexedExecutionLaneReusableCatalog};

impl OperationDispatch {
    /// The exact existing program lookup, retaining its privately produced
    /// catalog provenance. Missing topology/epoch is Unknown, never outside.
    pub fn select_reusable_execution_for_cost<'a, R: DeviceRuntime>(
        providers: &[BoundOperationProvider<'_, R>],
        resolved: &dyn ExecutablePlanView,
        wave: &PreparedStepSubmissionWave<R>,
        lane: &Arc<ExecutionLane<R>>,
        catalog: Option<&'a IndexedExecutionLaneReusableCatalog>,
        observe_route: bool,
    ) -> Result<PreparedReusableCostSelection<'a>, VNextError> {
        let authority = Self::reusable_execution_wave_authority(providers, resolved, wave, lane)?;
        let graph_snapshot = observe_route
            .then(|| {
                lane.try_with_resource_planning_lane(|epoch, state| Ok((epoch, state)))
                    .ok()
            })
            .flatten();
        let lane_epoch = lane.reusable_execution_epoch();
        let graph_state = graph_snapshot.and_then(|(_, state)| state);
        let configured_graph = graph_snapshot.is_some_and(|(epoch, _)| epoch == lane_epoch)
            && lane.runtime().cost_graph_capture_capability()
                == DeviceCostGraphCaptureCapability::Supported
            && graph_state.is_some_and(|state| state.is_ready());
        let (program_id, eager_boundaries) = match authority {
            Some(authority) => (
                Some(authority.program_id),
                authority.eager_boundary_node_indices,
            ),
            None => (None, Vec::new()),
        };
        let mut route = PreparedCostRouteV1 {
            class: C::Unknown,
            reason: U::ProgramIdentityUnavailable,
            program_id,
            non_reusable_wave: None,
            lane_id: lane.id().get(),
            lane_epoch,
            catalog_epoch: catalog.map(IndexedExecutionLaneReusableCatalog::epoch),
            graph_state,
            batch_step: Some(wave.batch_step_id().get()),
            batch_invocation: Some(wave.batch_invocation_id().get()),
        };
        let Some(program_id) = route.program_id.as_ref() else {
            // The authority builder returns None only after inspecting the
            // actual claimed backing: both program layout and lane slot are
            // absent. A known current graph catalogue makes this an explicit
            // eager wave outside the declared warm population. Preserve its
            // own original identity so submission cannot borrow another wave.
            if configured_graph
                && catalog.is_some_and(|catalog| {
                    catalog.is_observed()
                        && catalog.epoch() == lane_epoch
                        && catalog.lane_id() == Some(lane.id())
                })
                && wave.claimed_backing().program_binding_layout().is_none()
                && wave
                    .claimed_backing()
                    .program_binding_lane_slot_identity()
                    .is_none()
            {
                let work = wave.claimed_backing().work_shape();
                route.non_reusable_wave = Some(crate::execution_cost::NonReusableWaveIdentityV1 {
                    plan_hash: wave.claimed_backing().plan_hash().as_str().to_owned(),
                    runtime_implementation_fingerprint: lane
                        .descriptor()
                        .runtime_implementation_fingerprint
                        .clone(),
                    immediate_sequences: work.immediate_sequences(),
                    immediate_tokens: work.immediate_tokens(),
                    immediate_pages: work.immediate_pages(),
                });
                route.class = C::OutsideProgramLayoutAbsent;
                route.reason = U::ProgramLayoutAbsent;
            }
            return Ok(PreparedReusableCostSelection {
                program: None,
                route,
            });
        };
        let Some(catalog) = catalog else {
            route.reason = U::CatalogUnavailable;
            return Ok(PreparedReusableCostSelection {
                program: None,
                route,
            });
        };
        if catalog.epoch() != route.lane_epoch
            || (catalog.is_observed() && catalog.lane_id() != Some(lane.id()))
        {
            route.reason = U::CatalogEpochMismatch;
            return Ok(PreparedReusableCostSelection {
                program: None,
                route,
            });
        }
        let program = catalog.programs().get(program_id);
        match program {
            Some(program) if program.has_resident_segments() => {
                route.reason = U::ProgramPartial;
                if catalog.is_observed()
                    && configured_graph
                    && program.is_determinism_ready()
                    && program.node_count() as usize == wave.nodes().len()
                    && program.eager_boundary_node_indices() == eager_boundaries
                {
                    route.class = C::Warm;
                    route.reason = U::ResidentProgram;
                }
                Ok(PreparedReusableCostSelection {
                    program: Some(program),
                    route,
                })
            }
            value => {
                route.reason = if value.is_some() {
                    U::ProgramNonResident
                } else if catalog.programs().is_empty() {
                    U::CatalogEmpty
                } else {
                    U::ProgramAbsent
                };
                if catalog.is_observed() && configured_graph {
                    route.class = if value.is_some() {
                        C::OutsideProgramNonResident
                    } else {
                        C::OutsideProgramAbsent
                    };
                }
                Ok(PreparedReusableCostSelection {
                    program: None,
                    route,
                })
            }
        }
    }

    /// Used only when the actual selector will not select any reusable program.
    /// Nonblocking inspection cannot turn an unavailable stream into outside.
    pub fn observe_non_reusable_cost_route<R: DeviceRuntime>(
        lane: &Arc<ExecutionLane<R>>,
    ) -> PreparedCostRouteV1 {
        let mut route = PreparedCostRouteV1 {
            class: C::Unknown,
            reason: U::SelectionNotUsed,
            program_id: None,
            non_reusable_wave: None,
            lane_id: lane.id().get(),
            lane_epoch: lane.reusable_execution_epoch(),
            catalog_epoch: None,
            graph_state: None,
            batch_step: None,
            batch_invocation: None,
        };
        match lane.runtime().cost_graph_capture_capability() {
            DeviceCostGraphCaptureCapability::Unsupported => {
                route.class = C::GraphDisabled;
                route.reason = U::DeclaredGraphUnsupported;
            }
            DeviceCostGraphCaptureCapability::Supported => {
                let mut graph_state = None;
                let unconfigured = lane
                    .try_with_resource_planning_lane(|epoch, state| {
                        graph_state = state;
                        Ok(epoch == route.lane_epoch
                            && state.is_some_and(|state| state.is_unconfigured_empty()))
                    })
                    .unwrap_or(false);
                route.graph_state = graph_state;
                if unconfigured {
                    route.class = C::GraphDisabled;
                    route.reason = U::UnconfiguredStream;
                }
            }
            DeviceCostGraphCaptureCapability::Unknown => route.reason = U::RuntimeUnknown,
        }
        route
    }

    /// Same non-reusable classification, joined to the original prepared
    /// wave so a later guarded rollback cannot borrow another Step/Invocation.
    pub fn observe_non_reusable_cost_route_for_wave<R: DeviceRuntime>(
        lane: &Arc<ExecutionLane<R>>,
        wave: &PreparedStepSubmissionWave<R>,
    ) -> PreparedCostRouteV1 {
        let mut route = Self::observe_non_reusable_cost_route(lane);
        if wave.step_resources().execution_lane().id() != lane.id() {
            route.class = C::Unknown;
            route.reason = U::SelectionNotUsed;
        } else {
            route.batch_step = Some(wave.batch_step_id().get());
            route.batch_invocation = Some(wave.batch_invocation_id().get());
        }
        route
    }
}
