use super::*;
use ferrum_interfaces::execution_cost::ActualWaveKind;
use ferrum_interfaces::vnext::{ResourcePlanningRow, ResourcePlanningState};

pub(super) struct ExecutorResources<'a> {
    pub engine: &'a EngineInner,
    pub view: &'a ResourcePlanningView,
    pub fences: &'a [EngineFence],
}

struct Projection<'a> {
    source: &'a ExecutorResources<'a>,
    state: ResourcePlanningState,
}

impl PlanningResourceResolver for ExecutorResources<'_> {
    fn begin(
        &self,
        snapshot: &SchedulerSnapshot,
        poll: &mut dyn FnMut() -> std::result::Result<(), PlanningUnknownReason>,
    ) -> std::result::Result<Box<dyn PlanningResourceProjection + '_>, PlanningUnknownReason> {
        poll()?;
        if snapshot.requests.len() != self.fences.len()
            || self.view.participants().len() != self.fences.len()
        {
            return Err(PlanningUnknownReason::UnknownResourceEvidence);
        }
        Ok(Box::new(Projection {
            source: self,
            state: self.view.initial_state(),
        }))
    }
}

impl PlanningResourceProjection for Projection<'_> {
    fn apply(
        &mut self,
        query: &PlanningResourceQuery<'_>,
        poll: &mut dyn FnMut() -> std::result::Result<(), PlanningUnknownReason>,
    ) -> std::result::Result<(), PlanningUnknownReason> {
        let mut rows = Vec::with_capacity(query.wave.work.len());
        let mut has_prefill = false;
        let mut has_decode = false;
        for work in &query.wave.work {
            poll()?;
            let index = self
                .source
                .fences
                .iter()
                .position(|fence| {
                    fence.key.request_id == work.key.request_id
                        && fence.incarnation == work.key.incarnation
                        && fence.key.generation == work.key.work_generation
                })
                .ok_or(PlanningUnknownReason::InvalidSnapshot)?;
            let request = query
                .requests
                .iter()
                .find(|row| row.key == work.key)
                .ok_or(PlanningUnknownReason::InvalidSnapshot)?;
            let (start_token, token_count) = match work.action {
                WaveAction::Decode => {
                    has_decode = true;
                    (u64::from(request.context_tokens), 1)
                }
                WaveAction::Prefill { offset, count } => {
                    has_prefill = true;
                    (u64::from(offset), u64::from(count.get()))
                }
            };
            rows.push(ResourcePlanningRow {
                participant_index: index,
                start_token,
                token_count,
            });
        }
        let kind = match (has_prefill, has_decode) {
            (true, true) => ActualWaveKind::Mixed,
            (true, false) => ActualWaveKind::Prefill,
            (false, true) => ActualWaveKind::Decode,
            (false, false) => return Err(PlanningUnknownReason::InvalidSnapshot),
        };
        let mut failure = None;
        let result = self
            .source
            .engine
            .model_executor
            .project_execution_resource_wave_for_kind(
                self.source.view,
                &self.state,
                &rows,
                kind,
                &mut || match poll() {
                    Ok(()) if failure.is_none() => true,
                    Ok(()) => false,
                    Err(reason) => {
                        failure.get_or_insert(reason);
                        false
                    }
                },
            );
        let after = poll();
        if let Some(reason) = failure {
            return Err(reason);
        }
        after?;
        match result {
            ResourcePlanningAvailability::Known(projection) => {
                self.state = projection.state;
                Ok(())
            }
            ResourcePlanningAvailability::Unknown(reason) => Err(resource_reason(reason)),
        }
    }
}

pub(super) fn resource_reason(reason: ResourcePlanningUnknown) -> PlanningUnknownReason {
    match reason {
        ResourcePlanningUnknown::LogicalCapacity
        | ResourcePlanningUnknown::UnmaterializedCapacity
        | ResourcePlanningUnknown::PhysicalCapacity => {
            PlanningUnknownReason::OutputOrResourceBlocked
        }
        ResourcePlanningUnknown::BudgetExhausted => PlanningUnknownReason::ComputeBudgetExhausted,
        ResourcePlanningUnknown::MaintenanceRequired => PlanningUnknownReason::UnmodeledMaintenance,
        _ => PlanningUnknownReason::UnknownResourceEvidence,
    }
}

/// Static labels preserve the typed capture failure without formatting or I/O
/// on the controller thread. They do not change retry or scheduling policy.
pub(super) fn route_failure_checkpoint(
    reason: ferrum_interfaces::vnext::ExecutionCostRouteUnknown,
) -> &'static str {
    use ferrum_interfaces::vnext::{ExecutionCostRouteUnknown as Route, ResourcePlanningReadStage};
    match reason {
        Route::Unsupported => "route_failure_unsupported",
        Route::InvalidInput => "route_failure_invalid_input",
        Route::Capacity => "route_failure_capacity",
        Route::BudgetExhausted => "route_failure_budget_exhausted",
        Route::StaleView => "route_failure_stale_view",
        Route::ExecutionPolicy => "route_failure_execution_policy",
        Route::OnDemandResidentProgram => "route_failure_on_demand_resident_program",
        Route::ProviderRoute => "route_failure_provider_route",
        Route::CoreLayout => "route_failure_core_layout",
        Route::InitializationState => "route_failure_initialization_state",
        Route::ReadbackState => "route_failure_readback_state",
        Route::OutputBranch => "route_failure_output_branch",
        Route::Resource(reason) => match reason {
            ResourcePlanningUnknown::Unsupported => "resource_failure_unsupported",
            ResourcePlanningUnknown::BusyOrUnavailable => "resource_failure_busy_or_unavailable",
            ResourcePlanningUnknown::LimitExceeded => "resource_failure_limit_exceeded",
            ResourcePlanningUnknown::InvalidInput => "resource_failure_invalid_input",
            ResourcePlanningUnknown::StaleIdentity => "resource_failure_stale_identity",
            ResourcePlanningUnknown::ReusableExecution => "resource_failure_reusable_execution",
            ResourcePlanningUnknown::MaintenanceRequired => "resource_failure_maintenance_required",
            ResourcePlanningUnknown::LogicalCapacity => "resource_failure_logical_capacity",
            ResourcePlanningUnknown::UnmaterializedCapacity => {
                "resource_failure_unmaterialized_capacity"
            }
            ResourcePlanningUnknown::PhysicalCapacity => "resource_failure_physical_capacity",
            ResourcePlanningUnknown::InvalidDemand => "resource_failure_invalid_demand",
            ResourcePlanningUnknown::BudgetExhausted => "resource_failure_budget_exhausted",
            ResourcePlanningUnknown::ReadUnavailable(stage) => match stage {
                ResourcePlanningReadStage::Lifecycle => "resource_read_unavailable_lifecycle",
                ResourcePlanningReadStage::DeferredCleanup => "resource_read_unavailable_cleanup",
                ResourcePlanningReadStage::SequenceSession => "resource_read_unavailable_session",
                ResourcePlanningReadStage::SequenceBacking => "resource_read_unavailable_backing",
                ResourcePlanningReadStage::LogicalCapacity => "resource_read_unavailable_logical",
                ResourcePlanningReadStage::DeviceBudget => {
                    "resource_read_unavailable_device_budget"
                }
                ResourcePlanningReadStage::PhysicalPool => {
                    "resource_read_unavailable_physical_pool"
                }
                ResourcePlanningReadStage::ExecutionLane => "resource_read_unavailable_lane",
                ResourcePlanningReadStage::LaneWorkspace => "resource_read_unavailable_workspace",
                ResourcePlanningReadStage::ModelRegistry => "resource_read_unavailable_registry",
                ResourcePlanningReadStage::ModelRegistrySlot => {
                    "resource_read_unavailable_registry_slot"
                }
                ResourcePlanningReadStage::ModelSequenceOperation => {
                    "resource_read_unavailable_sequence_operation"
                }
            },
        },
    }
}
