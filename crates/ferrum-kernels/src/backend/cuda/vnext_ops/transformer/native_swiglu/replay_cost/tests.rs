use super::*;
use crate::gguf_blocks::GgufBlockFormat;
use ferrum_interfaces::execution_cost::SelectedReplayAlgorithmTemplateV1;
use ferrum_interfaces::vnext::WeightId;

fn prepared(
    ranges: &[Range<u64>],
    packed: bool,
    q8: Option<Q8SumPolicy>,
    mmq: Option<&StreamMmq>,
    capture: SloStructuredCostCapture,
) -> prepared::Prepared {
    let part = |format, offset| weights::MatrixPart {
        component_id: WeightId::new(format!("ffn.recipe.{offset}.{format:?}")).unwrap(),
        format: weights::MatrixFormat::Block(format),
        rows: 256,
        columns: 256,
        output_offset: offset,
        transform: None,
        signs_region: None,
    };
    let gate_up = vec![
        part(GgufBlockFormat::Q4K, 0),
        part(GgufBlockFormat::Q4K, 256),
    ];
    let down = vec![part(GgufBlockFormat::Q4K, 0)];
    let tokens = ranges.last().unwrap().end;
    let arithmetic = match (q8, mmq) {
        (_, Some(mmq)) if mmq.is_residual2() => route_selection::Arithmetic::Residual2M2To8,
        (_, Some(_)) => route_selection::Arithmetic::StreamMmq,
        (Some(_), None) => route_selection::Arithmetic::Q8,
        _ => route_selection::Arithmetic::Strict,
    };
    let mut selection = route_selection::select(
        &gate_up,
        &down,
        256,
        256,
        tokens,
        ranges.len() as u32,
        packed,
        packed,
        arithmetic,
    )
    .unwrap();
    let launches = match selection.packed_rows {
        Some(rows) => vec![(0, 1, rows, 0)],
        None => ranges
            .iter()
            .map(|range| (0, 1, (range.end - range.start) as u32, range.start))
            .collect(),
    };
    let q8_bytes = q8.map_or(0, |policy| {
        q8_part_workspace_per_token_with_policy(&gate_up, policy).unwrap() * tokens
    });
    let evidence = selected::swiglu(
        &gate_up,
        &down,
        launches.iter().map(|leaf| leaf.2),
        tokens,
        256,
        256,
        0,
        q8_bytes,
        q8,
        mmq,
        selection.mmq_hit,
        capture,
    );
    selection.command = selected::attach(selection.command, evidence);
    // The recipe consumes only the selector's numeric fields. No fake device
    // buffers or resource authority are minted for these pure contract tests.
    prepared::Prepared {
        regions: vec![],
        gate_up,
        down,
        launches,
        scratch_index: 0,
        scratch_layout: ScratchLayout::new(tokens, 256).unwrap(),
        hidden: 256,
        intermediate: 256,
        transform_bytes: 0,
        q8_bytes,
        mmq_bytes: 0,
        selection,
        key: CudaCommandReplayKeyBuilder::new("recipe-test", "ffn").finish(),
    }
}

fn assert_same_table(prepared: &prepared::Prepared, projected: &SelectedCommandCostEvidenceV1) {
    let command = &prepared.selection.command;
    let actual = command.statistical_evidence().unwrap();
    assert_eq!(actual, projected);
    assert_eq!(actual.algorithm_work(), projected.algorithm_work());
    SelectedReplayAlgorithmTemplateV1::from_selected(
        actual,
        command.token_count(),
        command.compute_dispatch_count(),
        command.transfer_command_count(),
    )
    .unwrap()
    .validate_binding(projected)
    .unwrap();
}

#[test]
fn cuda_ffn_recipe_preserves_packed_m_and_ordered_participant_launches() {
    let ranges = [0..2, 2..5];
    for q8 in [None, Some(Q8SumPolicy::Quantized), Some(Q8SumPolicy::Input)] {
        for packed in [false, true] {
            let prepared = prepared(
                &ranges,
                packed,
                q8,
                None,
                SloStructuredCostCapture::HostSettledV1,
            );
            let recipe = Recipe::from_prepared(&prepared, q8, None).unwrap();
            assert_same_table(&prepared, &recipe.project(5, &ranges).unwrap());
            // Equal-total repartition is legal only for the captured packed
            // launch. Ordered leaves fix both each M and its scratch offset.
            let repartition = recipe.project(5, &[0..3, 3..5]);
            assert_eq!(repartition.is_some(), packed);
            if let Some(projected) = repartition {
                assert_same_table(&prepared, &projected);
            }
            for (tokens, invalid) in [
                (4, vec![0..2, 2..4]),
                (5, vec![0..2, 3..5]),
                (5, vec![0..3, 2..5]),
                (5, vec![0..0, 0..5]),
                (5, vec![]),
                (5, vec![1..5]),
            ] {
                assert!(recipe.project(tokens, &invalid).is_none());
            }
        }
    }
}

#[test]
fn cuda_ffn_recipe_missing_actual_capture_and_inconsistent_launches_stay_unknown() {
    let ranges = [0..2, 2..5];
    let disabled = prepared(
        &ranges,
        true,
        None,
        None,
        SloStructuredCostCapture::Disabled,
    );
    assert!(Recipe::from_prepared(&disabled, None, None).is_none());
    let mut actual = prepared(
        &ranges,
        true,
        None,
        None,
        SloStructuredCostCapture::HostSettledV1,
    );
    actual.launches[0].3 = 1;
    assert!(Recipe::from_prepared(&actual, None, None).is_none());

    // A total M beyond the one-launch CUDA row bound stays on its captured
    // participant path; it cannot be reassigned to an oversized owner.
    let ranges = [0..32_768, 32_768..65_536];
    let actual = prepared(
        &ranges,
        true,
        None,
        None,
        SloStructuredCostCapture::HostSettledV1,
    );
    assert!(actual.selection.packed_rows.is_none());
    let recipe = Recipe::from_prepared(&actual, None, None).unwrap();
    assert_same_table(&actual, &recipe.project(65_536, &ranges).unwrap());
    assert!(recipe.project(65_536, &[0..65_536]).is_none());
    assert!(recipe.project(65_536, &[0..1, 1..65_536]).is_none());
}

#[test]
#[ignore = "requires an actual CUDA device"]
fn cuda_ffn_recipe_retains_real_mmq_workspace_and_whole_m_policy() {
    let context = cudarc::driver::CudaContext::new(0).unwrap();
    for mmq in [
        StreamMmq::load(&context).unwrap(),
        StreamMmq::load_residual2(&context).unwrap(),
    ] {
        for tokens in [1, 2, 3, 7, 8, 9] {
            let ranges = (0..tokens)
                .map(|start| start..start + 1)
                .collect::<Vec<_>>();
            for packed in [false, true] {
                let actual = prepared(
                    &ranges,
                    packed,
                    None,
                    Some(&mmq),
                    SloStructuredCostCapture::HostSettledV1,
                );
                let expected_hit = if mmq.is_residual2() {
                    (2..=8).contains(&tokens)
                } else {
                    tokens == 8 && packed
                };
                assert_eq!(actual.selection.mmq_hit, expected_hit);
                let recipe = Recipe::from_prepared(&actual, None, Some(&mmq)).unwrap();
                assert_same_table(&actual, &recipe.project(tokens, &ranges).unwrap());
                // A previously selected hit cannot recreate cost with the
                // strict fallback or an unavailable MMQ implementation.
                if expected_hit {
                    assert!(Recipe::from_prepared(&actual, None, None).is_none());
                }
            }
        }
    }
}
