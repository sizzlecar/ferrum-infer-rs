//! Passive current work bound to a separately sealed resident graph template.
//! None of these fields authorize execution or alter an execution identity.
use super::*;
use crate::execution_cost::{
    SelectedCommandCostEvidenceV1, SelectedReplayAlgorithmTemplateV1, MAX_COST_COMMANDS,
};

impl PartialEq for DeviceReusableExecutionInvocation {
    fn eq(&self, other: &Self) -> bool {
        self.program_id == other.program_id
            && self.segment == other.segment
            && self.participant_count == other.participant_count
            && self.token_count == other.token_count
    }
}
impl Eq for DeviceReusableExecutionInvocation {}

impl DeviceReusableExecutionInvocation {
    /// Only core dispatch can attach this ordered population, after producing
    /// it from actual invocations and successfully encoding dynamic bindings.
    /// A failed producer remains a None slot; it is never filtered/reindexed.
    pub(crate) fn with_selected_replay_cost(
        mut self,
        selected: Vec<Option<SelectedCommandCostEvidenceV1>>,
    ) -> Self {
        self.selected_replay_cost = None;
        if selected.len() <= MAX_COST_COMMANDS
            && selected.len() == self.segment.logical_command_count() as usize
            && self
                .segment
                .start_node_index()
                .checked_add(self.segment.logical_command_count())
                == Some(self.segment.end_node_index())
        {
            self.selected_replay_cost = Some(selected.into());
        }
        self
    }

    /// Runtime calls this only after matching the invocation to its resident
    /// program and segment. Metadata mismatch yields no passive evidence, not
    /// a different execution choice. The sealed rows retain no captured work.
    pub fn bind_replayed_cost_evidence(
        &self,
        sealed: &[DeviceReplayedLogicalCommandAttribution],
    ) -> Option<Vec<DeviceReplayedLogicalCommandAttribution>> {
        let selected = self.selected_replay_cost.as_deref()?;
        if sealed.len() != selected.len()
            || sealed.iter().enumerate().any(|(ordinal, row)| {
                row.logical_command_ordinal as usize != ordinal
                    || self
                        .segment
                        .start_node_index()
                        .checked_add(row.logical_command_ordinal)
                        != Some(row.node_index)
                    || row.participant_count != self.participant_count
            })
        {
            return None;
        }
        Some(
            sealed
                .iter()
                .zip(selected)
                .map(|(row, current)| {
                    let mut bound = row.clone();
                    bound.statistical_evidence = None;
                    bound.statistical_evidence =
                        row.bind_current_cost_evidence(current.as_ref()).cloned();
                    bound
                })
                .collect(),
        )
    }
}

// Match the pre-existing immutable execution identity exactly. Optional
// capture must not turn a cache publication or final execution guard into a
// different execution policy. Statistical consumers validate the sidecar.
impl PartialEq for DeviceReplayedLogicalCommandAttribution {
    fn eq(&self, other: &Self) -> bool {
        self.logical_command_ordinal == other.logical_command_ordinal
            && self.node_index == other.node_index
            && self.native_op_id == other.native_op_id
            && self.batching_form == other.batching_form
            && self.participant_count == other.participant_count
            && self.token_count == other.token_count
            && self.compute_dispatch_count == other.compute_dispatch_count
            && self.transfer_command_count == other.transfer_command_count
            && self.reusable_graph_node_count == other.reusable_graph_node_count
    }
}
impl Eq for DeviceReplayedLogicalCommandAttribution {}

impl DeviceReplayedLogicalCommandAttribution {
    /// This must come from the command that actually entered a successful
    /// native graph capture, never a subsequent eager candidate with same key.
    /// No captured numeric work/table is retained.
    pub fn with_captured_replay_template(
        mut self,
        template: Option<SelectedReplayAlgorithmTemplateV1>,
    ) -> Self {
        self.replay_template = template;
        self.statistical_evidence = None;
        self
    }

    /// Borrow current provider-selected work only after matching the sealed
    /// successful-capture template. Both actual replay and future projection
    /// use this checker; it never fabricates a live execution permission.
    pub fn bind_current_cost_evidence<'a>(
        &self,
        current: Option<&'a SelectedCommandCostEvidenceV1>,
    ) -> Option<&'a SelectedCommandCostEvidenceV1> {
        let current = current?;
        current
            .validate_command(
                self.token_count,
                self.compute_dispatch_count,
                self.transfer_command_count,
            )
            .ok()?;
        self.replay_template?.validate_binding(current).ok()?;
        Some(current)
    }

    pub fn statistical_evidence(&self) -> Option<&SelectedCommandCostEvidenceV1> {
        self.statistical_evidence.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution_cost::{
        KernelNumericWorkV1, KernelReplayGeometryV1, SelectedAlgorithmClassV1,
        SelectedCommandCostBuilderV1,
    };
    use crate::vnext::{
        ReusableExecutionBucketSpec, ReusableExecutionCapacity, ReusableExecutionClassId,
    };

    fn invocation() -> DeviceReusableExecutionInvocation {
        let bucket = ReusableExecutionBucketSpec::new(
            ReusableExecutionClassId::new("replay-current-work").unwrap(),
            ReusableExecutionCapacity::new(2, 3, 1).unwrap(),
        )
        .unwrap();
        let program = DeviceReusableExecutionProgramId::new(
            serde_json::from_value(serde_json::json!("a".repeat(64))).unwrap(),
            "b".repeat(64),
            ExecutionLaneId::mint().unwrap(),
            bucket.bucket_id().clone(),
            "c".repeat(64),
            "d".repeat(64),
            7,
            2,
            3,
            1,
        )
        .unwrap();
        DeviceReusableExecutionInvocation::new(
            program,
            DeviceReusableExecutionSegment::new(0, 4, 6, 2).unwrap(),
            2,
            3,
        )
        .unwrap()
    }
    fn evidence(entry: &str, context: u64, fixed: u64) -> SelectedCommandCostEvidenceV1 {
        let mut builder = SelectedCommandCostBuilderV1::new_with_algorithm_work(3);
        builder
            .kernel_with_replay_geometry(
                SelectedAlgorithmClassV1::new(entry, 1, [1; 32], [2; 32]).unwrap(),
                KernelNumericWorkV1 {
                    logical_units: 3,
                    padded_units: 3,
                    inner_units_per_logical_unit: context,
                    grid: [3, 1, 1],
                    scratch_bytes: 0,
                    staged_weight_bytes: 0,
                },
                KernelReplayGeometryV1 {
                    block: [128, 1, 1],
                    dynamic_shared_bytes: 0,
                    fixed_parameters: &[fixed],
                },
            )
            .unwrap();
        builder.finish().unwrap()
    }
    fn sealed(
        ordinal: u32,
        selected: &SelectedCommandCostEvidenceV1,
    ) -> DeviceReplayedLogicalCommandAttribution {
        DeviceReplayedLogicalCommandAttribution::new(
            ordinal,
            4 + ordinal,
            DeviceNativeOperationId::new("test.logical.compute").unwrap(),
            DeviceBatchingForm::Packed,
            2,
            3,
            1,
            0,
            1,
        )
        .unwrap()
        .with_captured_replay_template(Some(
            SelectedReplayAlgorithmTemplateV1::from_selected(selected, 3, 1, 0).unwrap(),
        ))
    }
    #[test]
    fn replay_current_work_preserves_population_and_uses_fresh_context() {
        let captured = [evidence("a", 32, 64), evidence("b", 32, 64)];
        let sealed = [sealed(0, &captured[0]), sealed(1, &captured[1])];
        assert!(sealed
            .iter()
            .all(|row| row.statistical_evidence().is_none()));
        let current = [evidence("a", 257, 64), evidence("b", 513, 64)];
        let invocation = invocation()
            .with_selected_replay_cost(vec![Some(current[0].clone()), Some(current[1].clone())]);
        let bound = invocation.bind_replayed_cost_evidence(&sealed).unwrap();
        assert_eq!(bound, sealed); // execution identity is unaffected by passive evidence.
        for (row, expected) in bound.iter().zip(&current) {
            let selected = row.statistical_evidence().unwrap();
            assert_eq!(selected.work(), expected.work());
            assert_ne!(selected.work(), captured[0].work());
            selected
                .algorithm_work()
                .unwrap()
                .unwrap()
                .validate_command(selected)
                .unwrap();
        }
        let missing = invocation
            .clone()
            .with_selected_replay_cost(vec![Some(current[0].clone()), None]);
        let bound = missing.bind_replayed_cost_evidence(&sealed).unwrap();
        assert_eq!(bound.len(), sealed.len());
        assert!(bound[0].statistical_evidence().is_some());
        assert!(bound[1].statistical_evidence().is_none());
        assert_eq!(bound[1].node_index(), 5);
    }
    #[test]
    fn replay_current_work_rejects_swapped_or_changed_fixed_launch_evidence() {
        let captured = [evidence("a", 32, 64), evidence("b", 32, 64)];
        let sealed = [sealed(0, &captured[0]), sealed(1, &captured[1])];
        let swapped = invocation()
            .with_selected_replay_cost(vec![Some(captured[1].clone()), Some(captured[0].clone())]);
        assert!(swapped
            .bind_replayed_cost_evidence(&sealed)
            .unwrap()
            .iter()
            .all(|r| r.statistical_evidence().is_none()));
        let changed = invocation().with_selected_replay_cost(vec![
            Some(evidence("a", 32, 65)),
            Some(evidence("b", 32, 65)),
        ]);
        assert!(changed
            .bind_replayed_cost_evidence(&sealed)
            .unwrap()
            .iter()
            .all(|r| r.statistical_evidence().is_none()));
        // Wrong logical identity cannot be made Known by otherwise correct work.
        let valid = invocation()
            .with_selected_replay_cost(vec![Some(captured[0].clone()), Some(captured[1].clone())]);
        let mut reordered = sealed.clone();
        reordered.swap(0, 1);
        assert!(valid.bind_replayed_cost_evidence(&reordered).is_none());
    }
    #[test]
    fn replay_current_work_missing_capture_or_incomplete_population_stays_unknown() {
        let selected = evidence("a", 32, 64);
        let rows = [sealed(0, &selected), sealed(1, &selected)];
        let plain = invocation();
        assert!(plain.bind_replayed_cost_evidence(&rows).is_none());
        let incomplete = plain
            .clone()
            .with_selected_replay_cost(vec![Some(selected.clone())]);
        assert_eq!(plain, incomplete);
        assert!(incomplete.bind_replayed_cost_evidence(&rows).is_none());
        let ready = plain.with_selected_replay_cost(vec![Some(selected.clone()), Some(selected)]);
        let no_capture = rows.map(|r| r.with_captured_replay_template(None));
        assert!(ready
            .bind_replayed_cost_evidence(&no_capture)
            .unwrap()
            .iter()
            .all(|r| r.statistical_evidence().is_none()));
    }
}
