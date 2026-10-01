//! Passive current work bound to a separately sealed resident graph template.
//! None of these fields authorize execution or alter an execution identity.
use super::*;
use crate::execution_cost::{
    SelectedCommandCostEvidenceV1, SelectedReplayAlgorithmTemplateV1, MAX_COST_COMMANDS,
};
use crate::vnext::BatchWorkShape;

/// Current numeric work minted from a core-validated complete invocation.
/// This passive value carries no buffer, lease or execution authority. The
/// resident projector may only use it for content-independent computation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceReplayCostWork {
    tokens: u64,
    participant_ranges: Arc<[std::ops::Range<u64>]>,
}
impl DeviceReplayCostWork {
    pub(super) fn from_observation(
        tokens: u64,
        participant_ranges: Arc<[std::ops::Range<u64>]>,
    ) -> Option<Self> {
        if participant_ranges.is_empty()
            || participant_ranges.len() > crate::execution_cost::MAX_COST_ROWS
        {
            return None;
        }
        let mut end = 0;
        for row in participant_ranges.iter() {
            if row.start != end || row.end <= row.start {
                return None;
            }
            end = row.end;
        }
        (end == tokens).then_some(Self {
            tokens,
            participant_ranges,
        })
    }
    pub fn retained_payload_bytes(&self) -> Option<usize> {
        self.participant_ranges
            .len()
            .checked_mul(std::mem::size_of::<std::ops::Range<u64>>())?
            .checked_add(std::mem::size_of::<Self>())
    }
    pub(crate) fn from_shape(shape: &BatchWorkShape) -> Option<Self> {
        let rows = shape.participant_token_ranges();
        if rows.is_empty() || rows.len() > crate::execution_cost::MAX_COST_ROWS {
            return None;
        }
        let mut end = 0;
        for row in rows {
            let range = row.immediate_token_range();
            if range.start != end || range.end <= range.start {
                return None;
            }
            end = range.end;
        }
        if end != shape.immediate_tokens() {
            return None;
        }
        Some(Self {
            tokens: end,
            participant_ranges: rows.iter().map(|row| row.immediate_token_range()).collect(),
        })
    }
    pub fn tokens(&self) -> u64 {
        self.tokens
    }
    pub fn participant_ranges(&self) -> &[std::ops::Range<u64>] {
        &self.participant_ranges
    }
}

#[derive(Debug, Clone)]
pub(crate) enum ReplayCostInput {
    Selected(Option<SelectedCommandCostEvidenceV1>),
    CapturedRecipe(DeviceReplayCostWork),
}

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
            self.selected_replay_cost = Some(
                selected
                    .into_iter()
                    .map(ReplayCostInput::Selected)
                    .collect(),
            );
        }
        self
    }

    pub(crate) fn with_replay_cost_inputs(mut self, inputs: Vec<ReplayCostInput>) -> Self {
        self.selected_replay_cost = None;
        if inputs.len() <= MAX_COST_COMMANDS
            && inputs.len() == self.segment.logical_command_count() as usize
            && self
                .segment
                .start_node_index()
                .checked_add(self.segment.logical_command_count())
                == Some(self.segment.end_node_index())
        {
            self.selected_replay_cost = Some(inputs.into());
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
        self.bind_replayed_cost_evidence_with(sealed, |_, _| None)
    }

    /// The backend supplies its recipe from the exact matched resident segment,
    /// sealed only after successful capture. Failed projection retains the
    /// ordinal as Unknown; it never requests a different provider or execution.
    pub fn bind_replayed_cost_evidence_with(
        &self,
        sealed: &[DeviceReplayedLogicalCommandAttribution],
        mut project: impl FnMut(u32, &DeviceReplayCostWork) -> Option<SelectedCommandCostEvidenceV1>,
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
                    bound.statistical_evidence = match current {
                        ReplayCostInput::Selected(current) => {
                            row.bind_current_cost_evidence(current.as_ref()).cloned()
                        }
                        ReplayCostInput::CapturedRecipe(work) => {
                            let current = project(row.logical_command_ordinal, work);
                            row.bind_current_cost_evidence(current.as_ref()).cloned()
                        }
                    };
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
    /// Whether a sealed, passive catalog row may retain its existing metadata.
    /// Unlike execution equality, this compares the entire captured template.
    /// A row carrying current numeric work is never reusable by this predicate,
    /// even against itself: sample equality is deliberately outside its scope.
    /// This does not authorize execution or replace resident-program checks.
    pub fn same_cost_catalog_metadata(&self, other: &Self) -> bool {
        self.statistical_evidence.is_none()
            && other.statistical_evidence.is_none()
            && self == other
            && self.replay_template == other.replay_template
    }

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

    fn shared_segment(
        invocation: &DeviceReusableExecutionInvocation,
        rows: Arc<[DeviceReplayedLogicalCommandAttribution]>,
    ) -> DeviceReplayedSegmentAttribution {
        DeviceReplayedSegmentAttribution::from_shared_logical_commands(
            0,
            invocation.program_id().clone(),
            invocation.segment().clone(),
            "e".repeat(64),
            rows,
        )
        .unwrap()
    }

    #[test]
    fn same_cost_catalog_metadata_distinguishes_templates_from_execution_equality() {
        let captured = evidence("a", 32, 64);
        let current = evidence("a", 257, 64);
        let original = sealed(0, &captured);
        let resealed = sealed(0, &current);
        // Different observed context is absent from a sealed template; fresh
        // context remains accepted without retaining the earlier numeric work.
        assert_ne!(captured.work(), current.work());
        assert!(original.same_cost_catalog_metadata(&resealed));
        assert!(resealed.same_cost_catalog_metadata(&original));
        assert!(original
            .bind_current_cost_evidence(Some(&current))
            .is_some());
        assert!(resealed
            .bind_current_cost_evidence(Some(&current))
            .is_some());

        let missing = original.clone().with_captured_replay_template(None);
        let changed_algorithm = sealed(0, &evidence("b", 32, 64));
        let changed_fixed_launch = sealed(0, &evidence("a", 32, 65));
        for changed in [missing, changed_algorithm, changed_fixed_launch] {
            // Existing Eq and wire omit the private cost template. Neither is
            // sufficient for deciding whether the numeric catalog is unchanged.
            assert_eq!(original, changed);
            assert_eq!(
                serde_json::to_vec(&original).unwrap(),
                serde_json::to_vec(&changed).unwrap()
            );
            assert!(!original.same_cost_catalog_metadata(&changed));
            assert!(!changed.same_cost_catalog_metadata(&original));
            assert!(changed.bind_current_cost_evidence(Some(&current)).is_none());
        }
        let mut changed_node = original.clone();
        changed_node.node_index += 1;
        assert!(!original.same_cost_catalog_metadata(&changed_node));
        assert!(!changed_node.same_cost_catalog_metadata(&original));
    }

    #[test]
    fn same_cost_catalog_metadata_rejects_actual_samples_even_for_equal_execution() {
        let captured = [evidence("a", 32, 64), evidence("b", 32, 64)];
        let rows = [sealed(0, &captured[0]), sealed(1, &captured[1])];
        let current = [evidence("a", 257, 64), evidence("b", 513, 64)];
        let invocation =
            invocation().with_selected_replay_cost(current.iter().cloned().map(Some).collect());
        let bound = invocation.bind_replayed_cost_evidence(&rows).unwrap();
        for ((sealed, actual), expected) in rows.iter().zip(&bound).zip(&current) {
            assert_eq!(sealed, actual);
            assert_eq!(
                actual.statistical_evidence().unwrap().work(),
                expected.work()
            );
            assert!(!sealed.same_cost_catalog_metadata(actual));
            assert!(!actual.same_cost_catalog_metadata(sealed));
            assert!(!actual.same_cost_catalog_metadata(actual));
            assert!(sealed.same_cost_catalog_metadata(&sealed.clone()));
        }
    }

    #[test]
    fn shared_replay_attribution_keeps_wire_and_constructor_rejections() {
        let invocation = invocation();
        let captured = [evidence("a", 32, 64), evidence("b", 32, 64)];
        let rows = vec![sealed(0, &captured[0]), sealed(1, &captured[1])];
        let legacy = DeviceReplayedSegmentAttribution::new(
            0,
            invocation.program_id().clone(),
            invocation.segment().clone(),
            "e".repeat(64),
            rows.clone(),
        )
        .unwrap();
        let shared = shared_segment(&invocation, Arc::from(rows.clone()));
        assert_eq!(
            serde_json::to_vec(&shared).unwrap(),
            serde_json::to_vec(&legacy).unwrap()
        );
        let wire = serde_json::to_value(&shared).unwrap();
        assert_eq!(wire["physical_command_index"], 0);
        assert_eq!(wire["logical_commands"][0]["node_index"], 4);
        assert_eq!(wire["logical_commands"][1]["node_index"], 5);
        assert!(wire["logical_commands"][0]
            .get("statistical_evidence")
            .is_none());

        let mut wrong_node = rows.clone();
        wrong_node[1].node_index = 6;
        let malformed = [
            Vec::new(),
            vec![rows[0].clone()],
            vec![rows[1].clone(), rows[0].clone()],
            vec![rows[0].clone(), rows[0].clone()],
            wrong_node,
        ];
        for bad_rows in malformed {
            assert!(DeviceReplayedSegmentAttribution::new(
                0,
                invocation.program_id().clone(),
                invocation.segment().clone(),
                "e".repeat(64),
                bad_rows.clone(),
            )
            .is_none());
            assert!(
                DeviceReplayedSegmentAttribution::from_shared_logical_commands(
                    0,
                    invocation.program_id().clone(),
                    invocation.segment().clone(),
                    "e".repeat(64),
                    Arc::from(bad_rows),
                )
                .is_none()
            );
        }
        for fingerprint in ["e".repeat(63), "E".repeat(64), "g".repeat(64)] {
            assert!(DeviceReplayedSegmentAttribution::new(
                0,
                invocation.program_id().clone(),
                invocation.segment().clone(),
                fingerprint.clone(),
                rows.clone(),
            )
            .is_none());
            assert!(
                DeviceReplayedSegmentAttribution::from_shared_logical_commands(
                    0,
                    invocation.program_id().clone(),
                    invocation.segment().clone(),
                    fingerprint,
                    Arc::from(rows.clone()),
                )
                .is_none()
            );
        }
    }

    #[test]
    fn shared_replay_attribution_does_not_relax_physical_submission_binding() {
        let invocation = invocation();
        let selected = evidence("a", 32, 64);
        let segment = shared_segment(
            &invocation,
            Arc::from([sealed(0, &selected), sealed(1, &selected)]),
        );
        for (path, node, participants, tokens, graphs, valid) in [
            (DeviceExecutionPath::Replayed, 4, 2, 3, Some(2), true),
            (DeviceExecutionPath::Eager, 4, 2, 3, None, false),
            (DeviceExecutionPath::Replayed, 5, 2, 3, Some(2), false),
            (DeviceExecutionPath::Replayed, 4, 1, 3, Some(2), false),
            (DeviceExecutionPath::Replayed, 4, 2, 4, Some(2), false),
            (DeviceExecutionPath::Replayed, 4, 2, 3, Some(3), false),
        ] {
            let physical = DeviceNativeWorkAttribution::new(
                0,
                Some(node),
                DeviceCommandPhase::Compute,
                DeviceNativeOperationId::new("vnext_reusable_execution").unwrap(),
                path,
                DeviceBatchingForm::Packed,
                participants,
                tokens,
                1,
                0,
                graphs,
            )
            .unwrap();
            assert_eq!(
                DeviceSubmissionAttribution::with_replayed_segments(
                    vec![physical],
                    vec![segment.clone()],
                )
                .is_some(),
                valid
            );
        }
    }

    #[test]
    fn shared_replay_attribution_isolates_requested_omitted_and_renewed_work() {
        let invocation = invocation();
        let captured = [evidence("a", 32, 64), evidence("b", 32, 64)];
        let sealed: Arc<[_]> = Arc::from([sealed(0, &captured[0]), sealed(1, &captured[1])]);
        let first = [evidence("a", 257, 64), evidence("b", 513, 64)];
        let renewed = [evidence("a", 1025, 64), evidence("b", 2049, 64)];
        let requested = |values: &[SelectedCommandCostEvidenceV1; 2]| {
            let current = invocation
                .clone()
                .with_selected_replay_cost(values.iter().cloned().map(Some).collect());
            let bound = current.bind_replayed_cost_evidence(&sealed).unwrap();
            shared_segment(&current, Arc::from(bound))
        };
        let first_segment = requested(&first);
        let retained_first = first_segment.clone();
        assert!(invocation.bind_replayed_cost_evidence(&sealed).is_none());
        let omitted = shared_segment(&invocation, Arc::clone(&sealed));
        let renewed_segment = requested(&renewed);
        assert!(sealed
            .iter()
            .all(|row| row.statistical_evidence().is_none()));
        drop(first_segment);
        drop(sealed);
        // Retained receipts still refer to their own original wave, even after
        // the cache population and original first-wave receipt are dropped.
        for (old, expected) in retained_first.logical_commands().iter().zip(&first) {
            assert_eq!(old.statistical_evidence().unwrap().work(), expected.work());
        }
        assert!(omitted
            .logical_commands()
            .iter()
            .all(|row| row.statistical_evidence().is_none()));
        for ((new, expected), old) in renewed_segment
            .logical_commands()
            .iter()
            .zip(&renewed)
            .zip(&first)
        {
            let evidence = new.statistical_evidence().unwrap();
            assert_eq!(evidence.work(), expected.work());
            assert_ne!(evidence.work(), old.work());
            evidence
                .algorithm_work()
                .unwrap()
                .unwrap()
                .validate_command(evidence)
                .unwrap();
        }
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
    fn omitted_actual_sample_keeps_cold_template_and_later_rebinds_fresh_work() {
        let captured = [evidence("a", 32, 64), evidence("b", 32, 64)];
        let rows = [sealed(0, &captured[0]), sealed(1, &captured[1])];
        let before = rows.clone();
        let mut projected = false;
        assert!(invocation()
            .bind_replayed_cost_evidence_with(&rows, |_, _| {
                projected = true;
                panic!("no actual sample must not project a captured recipe")
            })
            .is_none());
        assert!(!projected);
        assert_eq!(rows, before);
        assert!(rows.iter().all(|row| row.statistical_evidence().is_none()));
        // The same retained templates accept new exact work; the omitted wave
        // cannot copy the old numerical coordinates into the renewed sample.
        let current = [evidence("a", 257, 64), evidence("b", 513, 64)];
        let renewed = invocation()
            .with_selected_replay_cost(current.iter().cloned().map(Some).collect())
            .bind_replayed_cost_evidence(&rows)
            .unwrap();
        for ((row, expected), original) in renewed.iter().zip(&current).zip(&captured) {
            assert_eq!(row.statistical_evidence().unwrap().work(), expected.work());
            assert_ne!(expected.work(), original.work());
        }
        let missing = invocation()
            .with_selected_replay_cost(vec![None, Some(current[1].clone())])
            .bind_replayed_cost_evidence(&rows)
            .unwrap();
        assert_eq!(missing.len(), rows.len());
        assert!(missing[0].statistical_evidence().is_none());
        assert!(missing[1].statistical_evidence().is_some());
        assert_eq!(missing[0].node_index(), rows[0].node_index());
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
    #[test]
    fn captured_recipe_projection_preserves_mixed_population_and_fixed_launch_checks() {
        let current = evidence("a", 257, 64);
        let other = evidence("b", 513, 64);
        let captured = [evidence("a", 32, 64), evidence("b", 32, 64)];
        let rows = [sealed(0, &captured[0]), sealed(1, &captured[1])];
        let work = DeviceReplayCostWork {
            tokens: 3,
            participant_ranges: Arc::from([0..1, 1..3]),
        };
        let invocation = invocation().with_replay_cost_inputs(vec![
            ReplayCostInput::CapturedRecipe(work.clone()),
            ReplayCostInput::Selected(Some(other.clone())),
        ]);
        let bound = invocation
            .bind_replayed_cost_evidence_with(&rows, |ordinal, actual| {
                assert_eq!(ordinal, 0);
                assert_eq!(actual, &work);
                Some(current.clone())
            })
            .unwrap();
        assert_eq!(bound, rows);
        assert_eq!(
            bound[0].statistical_evidence().unwrap().algorithm_work(),
            current.algorithm_work()
        );
        assert_eq!(
            bound[1].statistical_evidence().unwrap().algorithm_work(),
            other.algorithm_work()
        );
        let unavailable = invocation.bind_replayed_cost_evidence(&rows).unwrap();
        assert!(unavailable[0].statistical_evidence().is_none());
        assert!(unavailable[1].statistical_evidence().is_some());
        let wrong_fixed = invocation
            .bind_replayed_cost_evidence_with(&rows, |_, _| Some(evidence("a", 257, 65)))
            .unwrap();
        assert!(wrong_fixed[0].statistical_evidence().is_none());
        assert!(wrong_fixed[1].statistical_evidence().is_some());
        let mut wrong_ordinal = rows.clone();
        wrong_ordinal.swap(0, 1);
        assert!(invocation
            .bind_replayed_cost_evidence_with(&wrong_ordinal, |_, _| panic!(
                "invalid resident population must not project"
            ))
            .is_none());
    }
}
