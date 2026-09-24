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
        ResourcePlanningUnknown::LogicalCapacity | ResourcePlanningUnknown::PhysicalCapacity => {
            PlanningUnknownReason::OutputOrResourceBlocked
        }
        ResourcePlanningUnknown::BudgetExhausted => PlanningUnknownReason::ComputeBudgetExhausted,
        ResourcePlanningUnknown::MaintenanceRequired => PlanningUnknownReason::UnmodeledMaintenance,
        _ => PlanningUnknownReason::UnknownResourceEvidence,
    }
}
