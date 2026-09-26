use super::*;
use crate::gguf_blocks::GgufBlockFormat;
use ferrum_interfaces::execution_cost::SelectedReplayAlgorithmTemplateV1;
use ferrum_interfaces::vnext::WeightId;

fn part(format: GgufBlockFormat, offset: u32) -> weights::MatrixPart {
    weights::MatrixPart {
        component_id: WeightId::new(format!("component.ffn.{offset}.{format:?}")).unwrap(),
        format: weights::MatrixFormat::Block(format),
        rows: 256,
        columns: 256,
        output_offset: offset,
        transform: None,
        signs_region: None,
    }
}

fn matrices() -> ([weights::MatrixPart; 2], [weights::MatrixPart; 1]) {
    (
        [
            part(GgufBlockFormat::Q4K, 0),
            part(GgufBlockFormat::Q4K, 256),
        ],
        [part(GgufBlockFormat::Q6K, 0)],
    )
}

#[test]
fn cuda_selected_native_ffn_complete_strict_and_q8_leaf_work_is_ordered_and_auditable() {
    let (gate, down) = matrices();
    for policy in [None, Some(Q8SumPolicy::Quantized), Some(Q8SumPolicy::Input)] {
        let workspace = policy.map_or(0, |p| {
            q8_f32scale::matrix_plan_from_parts(&gate, 8, p)
                .unwrap()
                .workspace_bytes(8)
                .unwrap()
        });
        let packed = swiglu(
            &gate,
            &down,
            [8],
            8,
            256,
            256,
            0,
            workspace,
            policy,
            None,
            false,
            SloStructuredCostCapture::HostSettledV1,
        )
        .unwrap();
        let leaves = swiglu(
            &gate,
            &down,
            [1; 8],
            8,
            256,
            256,
            0,
            workspace,
            policy,
            None,
            false,
            SloStructuredCostCapture::HostSettledV1,
        )
        .unwrap();
        let expected = if policy.is_some() { 6 } else { 4 };
        packed.validate_command(8, expected, 0).unwrap();
        leaves.validate_command(8, expected * 8, 0).unwrap();
        for evidence in [&packed, &leaves] {
            evidence
                .algorithm_work()
                .unwrap()
                .unwrap()
                .validate_command(evidence)
                .unwrap();
        }
        let template =
            SelectedReplayAlgorithmTemplateV1::from_selected(&packed, 8, expected, 0).unwrap();
        assert!(template.validate_binding(&leaves).is_err());
    }
}

#[test]
fn cuda_selected_native_ffn_replay_rejects_changed_fixed_dimensions_and_sum_policy() {
    let (gate, down) = matrices();
    let make = |gate: &[weights::MatrixPart], p| {
        let mut down = down.clone();
        down[0].rows = gate[0].columns;
        swiglu(
            gate,
            &down,
            [8],
            8,
            gate[0].columns,
            256,
            0,
            8192,
            Some(p),
            None,
            false,
            SloStructuredCostCapture::HostSettledV1,
        )
        .unwrap()
    };
    let original = make(&gate, Q8SumPolicy::Quantized);
    let template = SelectedReplayAlgorithmTemplateV1::from_selected(&original, 8, 6, 0).unwrap();
    let mut wider = gate.clone();
    for part in &mut wider {
        part.columns = 512;
    }
    assert!(template
        .validate_binding(&make(&wider, Q8SumPolicy::Quantized))
        .is_err());
    assert!(template
        .validate_binding(&make(&gate, Q8SumPolicy::Input))
        .is_err());
    template
        .validate_binding(&make(&gate, Q8SumPolicy::Quantized))
        .unwrap();
}

#[test]
fn cuda_selected_native_ffn_disabled_is_lazy_and_missing_or_invalid_evidence_stays_unknown() {
    let (gate, down) = matrices();
    let lazy = std::iter::from_fn(|| -> Option<u32> { panic!("Off must not enumerate leaves") });
    assert!(swiglu(
        &gate,
        &down,
        lazy,
        8,
        256,
        256,
        0,
        0,
        None,
        None,
        false,
        SloStructuredCostCapture::Disabled
    )
    .is_none());
    for rows in [vec![], vec![0], vec![7], vec![4, 5]] {
        assert!(swiglu(
            &gate,
            &down,
            rows,
            8,
            256,
            256,
            0,
            0,
            None,
            None,
            false,
            SloStructuredCostCapture::HostSettledV1
        )
        .is_none());
    }
    assert!(swiglu(
        &gate,
        &down,
        [8],
        8,
        256,
        256,
        0,
        1,
        Some(Q8SumPolicy::Quantized),
        None,
        false,
        SloStructuredCostCapture::HostSettledV1
    )
    .is_none());
    assert!(swiglu(
        &gate,
        &down,
        [8],
        8,
        256,
        256,
        0,
        0,
        None,
        None,
        true,
        SloStructuredCostCapture::HostSettledV1
    )
    .is_none());
    // A provider that already selected strict fallback remains identical to strict.
    let strict = swiglu(
        &gate,
        &down,
        [2, 3],
        5,
        256,
        256,
        0,
        0,
        None,
        None,
        false,
        SloStructuredCostCapture::HostSettledV1,
    )
    .unwrap();
    strict.validate_command(5, 8, 0).unwrap();
}
