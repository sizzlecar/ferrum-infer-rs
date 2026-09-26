//! Full native model fixture: real cold materializer, resolved provider,
//! preparation/submission/readback, current/future cost and retained replay.
//! This is backend conformance, not GGUF model-quality or serving-SLO evidence.
use super::*;
use ferrum_interfaces::execution_cost::AlgorithmWorkKindV1;
use ferrum_kernels::gguf_rn_fragment_materializer::{
    gguf_rn_fragment_inventory, GGUF_RN_FRAGMENT_FORMAT_ID, GGUF_RN_FRAGMENT_MATERIALIZER_ID,
};
use ferrum_types::SloStructuredCostCapture;
use std::sync::atomic::{AtomicU64, Ordering};

fn build(capture: SloStructuredCostCapture, mode: FixtureExecutionMode, rows: u64) -> Fixture {
    static NEXT: AtomicU64 = AtomicU64::new(1);
    // The existing real F16 embedding/recurrent trunk keeps its original math;
    // only the declared Q4K gate/up and Q6K down become dual representations.
    let kind = AttentionKind::GatedDeltaHadamardF16;
    let definition = ffn_family::Q8FfnFamily {
        base: Family::new(kind),
        policy: ffn_family::FfnPolicy::RnFragment,
    };
    let states = definition.base.states();
    let registration = TypedFamilyRegistration::new(Rounded(definition));
    let family = registration
        .prepare_with_profile(&serde_json::to_value(kind).unwrap(), &id(FRAGMENT_PROFILE))
        .unwrap();
    let inventory = gguf_rn_fragment_inventory(&family).unwrap();
    assert!(inventory.converted_f16_bytes > 0 && inventory.packed_fragment_bytes > 0);
    assert_eq!(
        inventory.unique_consumed_execution_bytes,
        inventory.converted_f16_bytes
            + inventory.packed_fragment_bytes
            + inventory.retained_consumed_bytes
    );
    let (runtime, registry, materializers, catalog) =
        CudaVNextComposition::create_with_observation(
            0,
            id(format!(
                "device.cuda.rn-fragment-model.{}",
                NEXT.fetch_add(1, Ordering::Relaxed)
            )),
            ferrum_types::AttentionExecutionPolicy::Portable,
            None,
            capture,
        )
        .unwrap()
        .into_parts();
    let materializer = cuda_weight_materializer_selection(&family).unwrap();
    assert_eq!(
        materializer.materializer_id().as_str(),
        GGUF_RN_FRAGMENT_MATERIALIZER_ID
    );
    let fixture = Fixture::from_prepared_family_with_composition(
        kind,
        family.clone(),
        states,
        mode,
        None,
        rows,
        BTreeMap::new(),
        (runtime, registry, materializers, materializer, catalog),
    );
    attribution::assert_product_witness_inventory(
        &registration,
        &family,
        &fixture,
        inventory.converted_f16_bytes,
        Some(inventory.packed_fragment_bytes),
    );
    let payload = fixture.compilation.executable().execution_plan().payload();
    let node_index = payload
        .nodes()
        .iter()
        .position(|n| n.id().as_str() == "node.ffn")
        .unwrap();
    let node = &payload.nodes()[node_index];
    assert_eq!(
        node.operation_id().as_str(),
        DENSE_SWIGLU_GGUF_RN_F16_FRAGMENT_M1_TO8_OPERATION_ID
    );
    let provider = fixture.providers.providers()[node_index].descriptor();
    let ProviderCheckpointCapability::CompletedBoundary(checkpoint) =
        provider.checkpoint_capability()
    else {
        panic!("actual RN fragment provider must declare checkpoint continuation");
    };
    assert_eq!(
        checkpoint.input_dependency(),
        CheckpointInputDependency::ExactTokenPrefix
    );
    assert_eq!(
        checkpoint.partition_numerics(),
        CheckpointPartitionNumerics::CapturedExecutionContinuation
    );
    assert!(checkpoint.state_ports().is_empty());
    assert!(
        matches!(
            fixture
                .compilation
                .executable()
                .execution_plan()
                .sequence_checkpoint_capability(),
            SequenceCheckpointCapability::Enabled(_)
        ),
        "actual selected providers must close the checkpoint contract: {:?}",
        fixture
            .compilation
            .executable()
            .execution_plan()
            .sequence_checkpoint_capability()
    );
    assert!(provider
        .accepted_weight_formats()
        .contains(&id(GGUF_RN_FRAGMENT_FORMAT_ID)));
    let mut physical_bytes = 0;
    for ordinal in [1, 2] {
        let value = node
            .values()
            .iter()
            .find(|v| v.role() == ResolvedValueRole::Input && v.ordinal() == ordinal)
            .unwrap();
        let weight = value.weight().unwrap();
        let PhysicalWeightLayout::RnF16DenseAndFragmentV1 {
            dense_values,
            fragment_values,
            source_format,
        } = weight.physical_layout()
        else {
            panic!("actual resolved provider must bind both representations");
        };
        let plan =
            RnF16FragmentPlanV1::from_dimensions(*source_format, value.tensor().dimensions())
                .unwrap();
        assert_ne!(dense_values.component_id, fragment_values.component_id);
        assert_eq!(weight.components().len(), 2);
        assert_eq!(value.storage().components().len(), 2);
        for (id, bytes) in [
            (&dense_values.component_id, plan.dense_bytes()),
            (&fragment_values.component_id, plan.packed_bytes()),
        ] {
            let storage = value
                .storage()
                .components()
                .iter()
                .find(|s| s.component_id() == Some(id))
                .unwrap();
            assert_eq!(storage.length_bytes(), bytes);
            physical_bytes += bytes;
        }
    }
    assert_eq!(
        physical_bytes,
        inventory.converted_f16_bytes + inventory.packed_fragment_bytes
    );
    fixture
}

fn assert_arm(fixture: &Fixture, counts: &[usize]) {
    let executable = fixture.compilation.executable();
    let index = executable
        .execution_plan()
        .payload()
        .nodes()
        .iter()
        .position(|n| n.id().as_str() == "node.ffn")
        .unwrap();
    let rows = counts
        .iter()
        .map(|&count| OperationCostWorkRow {
            offset: 0,
            count: NonZeroU64::new(count as u64).unwrap(),
            full_input_tokens: NonZeroU64::new(count as u64).unwrap(),
        })
        .collect::<Vec<_>>();
    let route = fixture.providers.providers()[index]
        .eager_cost_route(executable, &rows)
        .unwrap()
        .unwrap();
    let m = counts.iter().sum::<usize>();
    let libraries = route
        .commands()
        .iter()
        .flat_map(|command| {
            command
                .statistical_evidence()
                .unwrap()
                .algorithm_work()
                .unwrap()
                .unwrap()
                .entries()
        })
        .filter(|entry| entry.kind() == AlgorithmWorkKindV1::LibraryCall)
        .count();
    assert_eq!(
        libraries,
        if m <= 8 { 0 } else { 2 },
        "whole M={m}, partition={counts:?}"
    );
}
fn assert_finite(outputs: &[Vec<u8>]) {
    let values = outputs
        .iter()
        .flat_map(|row| row.chunks_exact(2))
        .map(|bytes| f16::from_le_bytes([bytes[0], bytes[1]]).to_f32())
        .collect::<Vec<_>>();
    assert!(!values.is_empty() && values.iter().all(|v| v.is_finite()));
    assert!(values.iter().any(|v| *v != 0.0));
}

#[test]
fn rn_fragment_model_actual_future_and_dual_receipt_cover_whole_m_boundary() {
    let disabled = build(
        SloStructuredCostCapture::Disabled,
        FixtureExecutionMode::Eager,
        9,
    );
    let enabled = build(
        SloStructuredCostCapture::HostSettledV1,
        FixtureExecutionMode::Eager,
        9,
    );
    for counts in [&[1][..], &[8][..], &[9][..], &[4, 4][..], &[4, 5][..]] {
        let expected = full_cost_route::run_with_output(&disabled, counts, false, "node.ffn").0;
        // This submits the actual OperationProvider and compares its original
        // device attribution to the pre-submission query, including sealed
        // algorithm assignments and launch geometry. It also consumes the
        // real completion/readback receipt and checks resource freshness.
        let actual = full_cost_route::run_with_ffn_statistics(&enabled, counts, true);
        assert_eq!(actual, expected, "passive capture changed actual FFN math");
        assert_finite(&actual);
        assert_arm(&enabled, counts);
    }
}

#[test]
fn rn_fragment_model_changed_inputs_and_checkpoint_continue_in_retained_graph() {
    for rows in [1usize, 8, 9] {
        let eager = build(
            SloStructuredCostCapture::Disabled,
            FixtureExecutionMode::Eager,
            rows as u64,
        );
        let replay = build(
            SloStructuredCostCapture::HostSettledV1,
            FixtureExecutionMode::Replay,
            rows as u64,
        );
        // Reuse existing full-model graph checks: changed actual tokens, two
        // independent owners, A parked while B overwrites invocation scratch,
        // and subsequent A resumption. ReplayedOnly is required after capture.
        graph::compare_changed_inputs(
            &eager,
            &replay,
            rows,
            "node.ffn",
            graph::ReplayExpectation::Resident,
        );
        // The cross-owner scenario may grow backing and then release it.
        // First-capture maintenance has a separate, fresh foreground-only
        // setup, so free backing from that scenario cannot satisfy capture.
        drop(replay);
        drop(eager);
        let eager = build(
            SloStructuredCostCapture::Disabled,
            FixtureExecutionMode::Eager,
            rows as u64,
        );
        let replay = build(
            SloStructuredCostCapture::HostSettledV1,
            FixtureExecutionMode::Replay,
            rows as u64,
        );
        let tokens: Arc<[u32]> = (0..4 * rows)
            .map(|i| 1 + ((i * 7 + i / rows) % 29) as u32)
            .collect();
        let expected_owner = eager.admit("fragment-checkpoint-eager", Arc::clone(&tokens));
        let source = replay.admit("fragment-checkpoint-source", Arc::clone(&tokens));
        let execute = |fixture: &Fixture,
                       owner: &Arc<SequenceSession<Runtime>>,
                       index: usize,
                       resident: bool| {
            let observed = fixture
                .execute_checked_output_node(
                    owner,
                    Arc::clone(&tokens),
                    index * rows..(index + 1) * rows,
                    resident,
                    false,
                    resident,
                    "node.ffn",
                )
                .unwrap();
            observed.assert_state_nonzero();
            observed
        };
        let expected = (0..4)
            .map(|i| execute(&eager, &expected_owner, i, false))
            .collect::<Vec<_>>();
        for i in 0..2 {
            expected[i].assert_same(
                // Real cold warm/capture before requiring resident replay.
                &execute(&replay, &source, i, false),
                "fragment checkpoint owner warm/capture prefix",
            );
        }
        let checkpoint = replay.capture(&source);
        assert_eq!(checkpoint.completed_tokens(), 2 * rows);
        source.try_abort_if_quiescent().unwrap();
        drop(source);
        let restored = replay.admit("fragment-checkpoint-restored", Arc::clone(&tokens));
        replay.restore(&restored, &checkpoint, Arc::clone(&tokens));
        for i in 2..4 {
            // This helper also compares retained current/future selected cost
            // and fixed parameters, while requiring this FFN node's resident
            // segment and the actual ReplayedOnly submission policy.
            let actual = execute(&replay, &restored, i, true);
            expected[i].assert_same(&actual, "fragment restored current-token suffix");
            actual.assert_different_output(&expected[i - 1]);
        }
        restored.try_complete().unwrap();
        expected_owner.try_complete().unwrap();
    }
}
