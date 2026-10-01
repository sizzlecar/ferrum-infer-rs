use super::*;
use crate::execution_cost::{
    KernelNumericWorkV1, SelectedAlgorithmClassV1, SelectedCommandCostBuilderV1,
};
use std::sync::atomic::{AtomicUsize, Ordering};

struct CountedCpuTemplate {
    calls: Arc<AtomicUsize>,
    output_tokens: u64,
    maximum: usize,
}
impl DeviceObservationTemplate for CountedCpuTemplate {
    fn command_count(&self) -> usize {
        1
    }
    fn retained_payload_bytes(&self) -> Option<usize> {
        Some(std::mem::size_of::<Self>())
    }
    fn projection_retained_bytes_upper_bound(&self) -> Option<usize> {
        Some(self.maximum)
    }
    fn project(
        &self,
        _: &FrozenObservationInput,
    ) -> Result<Vec<Option<SelectedCommandCostEvidenceV1>>, StatisticalEvidenceUnknown> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        let mut builder = SelectedCommandCostBuilderV1::new_with_algorithm_work(self.output_tokens);
        builder.kernel(
            SelectedAlgorithmClassV1::new("actual.kernel", 1, [1; 32], [2; 32])?,
            KernelNumericWorkV1 {
                logical_units: 1,
                padded_units: 1,
                inner_units_per_logical_unit: 1,
                grid: [1, 1, 1],
                scratch_bytes: 0,
                staged_weight_bytes: 0,
            },
        )?;
        Ok(vec![Some(builder.finish()?)])
    }
}
fn ledger(calls: &Arc<AtomicUsize>, tokens: u64, maximum: usize) -> DeviceSubmissionAttribution {
    let packet = DeviceObservationPacket::new(
        Arc::new(CountedCpuTemplate {
            calls: Arc::clone(calls),
            output_tokens: tokens,
            maximum,
        }),
        FrozenObservationInput::command(1),
    )
    .unwrap();
    ledger_with_packet(packet)
}
fn ledger_with_packet(packet: DeviceObservationPacket) -> DeviceSubmissionAttribution {
    let row = DeviceNativeWorkAttribution::new(
        0,
        Some(0),
        DeviceCommandPhase::Compute,
        DeviceNativeOperationId::new("actual.kernel").unwrap(),
        DeviceExecutionPath::Eager,
        DeviceBatchingForm::Packed,
        1,
        1,
        1,
        0,
        None,
    )
    .unwrap()
    .with_observation(packet)
    .unwrap();
    DeviceSubmissionAttribution::new(vec![row]).unwrap()
}

#[test]
fn observation_clone_inspection_and_formatting_never_project() {
    let calls = Arc::new(AtomicUsize::new(0));
    let raw = ledger(&calls, 1, 8192);
    let copy = raw.clone();
    assert!(Arc::ptr_eq(&raw.commands, &copy.commands));
    assert!(raw.has_unresolved_observation());
    assert!(raw.commands()[0].statistical_evidence().is_none());
    assert!(raw.retained_payload_bytes().unwrap() > 0);
    assert!(raw.maximum_resolved_bytes().unwrap() >= raw.retained_payload_bytes().unwrap());
    let _ = format!("{raw:?}");
    let _ = serde_json::to_vec(&raw).unwrap();
    assert_eq!(raw, copy);
    assert_eq!(calls.load(Ordering::Relaxed), 0);
    let resolved = copy.resolve_observation().unwrap();
    assert_eq!(calls.load(Ordering::Relaxed), 1);
    assert!(!resolved.has_unresolved_observation());
    assert!(resolved.commands()[0].statistical_evidence().is_some());
    assert!(raw.has_unresolved_observation());
    let _ = resolved.resolve_observation().unwrap();
    assert_eq!(calls.load(Ordering::Relaxed), 1);
}

fn logical(ordinal: u32, node: u32) -> DeviceReplayedLogicalCommandAttribution {
    DeviceReplayedLogicalCommandAttribution::new(
        ordinal,
        node,
        DeviceNativeOperationId::new("actual.kernel").unwrap(),
        DeviceBatchingForm::Packed,
        1,
        1,
        1,
        0,
        2,
    )
    .unwrap()
}

#[test]
fn observation_catalogue_preserves_order_and_uniform_participant_proof() {
    let segment = DeviceReusableExecutionSegment::new(0, 4, 6, 2).unwrap();
    let rows: Arc<[_]> = vec![logical(0, 4), logical(1, 5)].into();
    let catalogue =
        DeviceReplayedCommandCatalogue::new(segment.clone(), Arc::clone(&rows)).unwrap();
    assert!(Arc::ptr_eq(&catalogue.commands, &rows));
    assert_eq!(catalogue.logical_graph_node_count(), 4);
    assert!(catalogue.retained_payload_bytes().unwrap() >= std::mem::size_of_val(rows.as_ref()));
    let mut reversed = rows.to_vec();
    reversed.reverse();
    assert!(DeviceReplayedCommandCatalogue::new(segment.clone(), reversed.into()).is_none());
    let mut different_participants = rows.to_vec();
    different_participants[1].participant_count = 2;
    assert!(
        DeviceReplayedCommandCatalogue::new(segment.clone(), different_participants.into())
            .is_none()
    );
    let mut wrong_node = rows.to_vec();
    wrong_node[1].node_index = 6;
    assert!(DeviceReplayedCommandCatalogue::new(segment, wrong_node.into()).is_none());
}

#[test]
fn observation_exact_equality_retains_work_and_node_fields() {
    let row = logical(0, 4);
    let mut changed = row.clone();
    changed.node_index += 1;
    assert_ne!(row, changed);
    changed = row.clone();
    changed.token_count += 1;
    assert_ne!(row, changed);
    changed = row.clone();
    changed.reusable_graph_node_count += 1;
    assert_ne!(row, changed);
}

#[test]
fn observation_resolution_rejects_wrong_actual_command_and_understated_budget() {
    let calls = Arc::new(AtomicUsize::new(0));
    assert_eq!(
        ledger(&calls, 2, 8192).resolve_observation().unwrap_err(),
        StatisticalEvidenceUnknown::CommandMismatch
    );
    assert_eq!(
        ledger(&calls, 1, 0).resolve_observation().unwrap_err(),
        StatisticalEvidenceUnknown::Capacity
    );
    assert_eq!(calls.load(Ordering::Relaxed), 2);
}

#[test]
fn observation_drop_has_no_projection_or_retained_device_action() {
    let calls = Arc::new(AtomicUsize::new(0));
    let raw = ledger(&calls, 1, 8192);
    drop(raw);
    assert_eq!(calls.load(Ordering::Relaxed), 0);
    assert_eq!(Arc::strong_count(&calls), 1);
}

#[test]
fn observation_call_owned_excludes_only_live_template_lease_and_keeps_last_owner_charged() {
    let calls = Arc::new(AtomicUsize::new(0));
    let budget = DeviceObservationTemplateBudget::new(8192).unwrap();
    let retained = budget
        .retain(Arc::new(CountedCpuTemplate {
            calls: Arc::clone(&calls),
            output_tokens: 1,
            maximum: 8192,
        }))
        .unwrap();
    let charged = budget.retained_payload_bytes();
    let packet = retained.packet(FrozenObservationInput::command(1)).unwrap();
    assert_eq!(
        packet.retained_payload_bytes() - packet.call_owned_payload_bytes(),
        charged
    );
    let raw = ledger_with_packet(packet);
    let owned = raw.call_owned_payload_bytes().unwrap();
    assert_eq!(raw.retained_payload_bytes().unwrap() - owned, charged);
    assert_eq!(raw.maximum_working_bytes(), Some(owned * 2 + 8192));
    assert_eq!(
        raw.maximum_resolved_bytes().unwrap() - raw.maximum_working_bytes().unwrap(),
        charged * 2
    );
    let copy = raw.clone();
    drop(retained);
    drop(raw);
    assert_eq!(budget.retained_payload_bytes(), charged);
    assert_eq!(calls.load(Ordering::Relaxed), 0);
    let resolved = copy.resolve_observation().unwrap();
    assert_eq!(calls.load(Ordering::Relaxed), 1);
    assert_eq!(budget.retained_payload_bytes(), 0);
    assert!(resolved.commands()[0].statistical_evidence().is_some());
}

#[test]
fn observation_unleased_and_failed_packets_keep_accurate_charge_without_projection_getters() {
    let calls = Arc::new(AtomicUsize::new(0));
    let unleased = ledger(&calls, 1, 8192);
    assert_eq!(
        unleased.call_owned_payload_bytes(),
        unleased.retained_payload_bytes()
    );
    assert_eq!(
        unleased.maximum_working_bytes(),
        unleased.maximum_resolved_bytes()
    );
    let overflow = ledger(&calls, 1, usize::MAX);
    assert_eq!(overflow.maximum_working_bytes(), None);
    assert_eq!(calls.load(Ordering::Relaxed), 0);

    let budget = DeviceObservationTemplateBudget::new(8192).unwrap();
    let retained = budget
        .retain(Arc::new(CountedCpuTemplate {
            calls: Arc::clone(&calls),
            output_tokens: 2,
            maximum: 8192,
        }))
        .unwrap();
    let raw = ledger_with_packet(retained.packet(FrozenObservationInput::command(1)).unwrap());
    drop(retained);
    assert_eq!(
        raw.resolve_observation().unwrap_err(),
        StatisticalEvidenceUnknown::CommandMismatch
    );
    assert_eq!(budget.retained_payload_bytes(), 0);
}

#[test]
fn observation_call_owned_keeps_unleased_logical_catalogue_and_cow_bound() {
    use crate::vnext::{
        ReusableExecutionBucketSpec, ReusableExecutionCapacity, ReusableExecutionClassId,
    };
    let bucket = ReusableExecutionBucketSpec::new(
        ReusableExecutionClassId::new("observation-owned-catalogue").unwrap(),
        ReusableExecutionCapacity::new(1, 1, 1).unwrap(),
    )
    .unwrap();
    let program = DeviceReusableExecutionProgramId::new(
        serde_json::from_value(serde_json::json!("a".repeat(64))).unwrap(),
        "b".repeat(64),
        ExecutionLaneId::mint().unwrap(),
        bucket.bucket_id().clone(),
        "c".repeat(64),
        "d".repeat(64),
        1,
        1,
        1,
        1,
    )
    .unwrap();
    let descriptor = DeviceReusableExecutionSegment::new(0, 4, 5, 1).unwrap();
    let catalogue =
        DeviceReplayedCommandCatalogue::new(descriptor.clone(), vec![logical(0, 4)].into())
            .unwrap();
    let physical = DeviceNativeWorkAttribution::new(
        0,
        Some(4),
        DeviceCommandPhase::Compute,
        DeviceNativeOperationId::new("actual.kernel").unwrap(),
        DeviceExecutionPath::Replayed,
        DeviceBatchingForm::Packed,
        1,
        1,
        1,
        0,
        Some(2),
    )
    .unwrap();
    let without = DeviceSubmissionAttribution::new(vec![physical.clone()]).unwrap();
    let segment = DeviceReplayedSegmentAttribution::from_catalogue(
        0,
        program,
        descriptor,
        "e".repeat(64),
        &catalogue,
    )
    .unwrap();
    let raw =
        DeviceSubmissionAttribution::with_replayed_segments(vec![physical], vec![segment]).unwrap();
    let owned = raw.call_owned_payload_bytes().unwrap();
    assert!(owned - without.call_owned_payload_bytes().unwrap() >= catalogue.payload_bytes);
    assert_eq!(raw.retained_payload_bytes(), Some(owned));
    assert_eq!(raw.maximum_working_bytes(), Some(owned * 2));
    drop(catalogue);
    assert_eq!(raw.call_owned_payload_bytes(), Some(owned));
}
