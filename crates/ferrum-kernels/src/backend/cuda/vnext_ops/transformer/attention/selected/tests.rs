use super::*;
use crate::gguf_blocks::GgufBlockFormat;
use ferrum_interfaces::execution_cost::SelectedReplayAlgorithmTemplateV1;
use ferrum_interfaces::vnext::WeightId;

fn shape() -> AttentionShape {
    AttentionShape {
        hidden_size: 4096,
        key_heads: 16,
        value_heads: 32,
        key_head_dim: 128,
        value_head_dim: 128,
        qkv_features: 8192,
        value_features: 4096,
        qkvz_features: 12288,
        ba_features: 64,
        qkvzba_features: 12352,
        conv_kernel: 4,
        conv_state_width: 3,
        epsilon: 1.0e-6,
        layer_index: 0,
        decay_parameterization: GatedDeltaDecayParameterization::NegativeRate,
        value_head_mapping: GatedDeltaValueHeadMapping::GroupedByKeyHead,
    }
}
fn matrices(quantized: bool) -> (Vec<weights::MatrixPart>, Vec<weights::MatrixPart>) {
    let part = |rows, offset, format| weights::MatrixPart {
        component_id: WeightId::new(format!("weight.gdn.{rows}.{offset}.{format:?}")).unwrap(),
        format: weights::MatrixFormat::Block(format),
        rows,
        columns: 4096,
        output_offset: offset,
        transform: None,
        signs_region: None,
    };
    let format = if quantized {
        GgufBlockFormat::Q4K
    } else {
        GgufBlockFormat::Q8_0
    };
    (
        vec![
            part(8192, 0, format),
            part(4096, 8192, format),
            part(32, 12288, GgufBlockFormat::Q8_0),
            part(32, 12320, GgufBlockFormat::Q8_0),
        ],
        vec![part(4096, 0, format)],
    )
}
fn projection(parts: &[weights::MatrixPart], quantized: bool) -> AttentionProjection {
    if quantized {
        AttentionProjection::NativeQ8 {
            pack_bytes_per_token: q8_f32scale::matrix_plan_from_parts(
                parts,
                1,
                q8_f32scale::Q8SumPolicy::Quantized,
            )
            .unwrap()
            .pack_bytes_per_row(),
        }
    } else {
        AttentionProjection::Native {
            transform_bytes_per_token: 0,
        }
    }
}

#[test]
fn cuda_selected_gdn_matches_shared_dispatch_selectors_for_packed_and_leaf_commands() {
    for quantized in [false, true] {
        let (qkv, output) = matrices(quantized);
        let precision = if quantized {
            AttentionPrecision::F32MasterQ8Projections
        } else {
            AttentionPrecision::F32Master
        };
        for counts in [vec![8], vec![1; 8], vec![4, 4], vec![3, 5]] {
            let packed = counts.len() == 1;
            let leaves = counts
                .iter()
                .map(|&n| (n, if packed { 8 } else { 1 }, packed));
            let participants = if packed { 8 } else { counts.len() };
            let observed = compute(
                shape(),
                precision,
                projection(&qkv, quantized),
                ProjectionEvidence::Native {
                    input: &qkv,
                    output: &output,
                },
                leaves,
                8,
                participants,
                true,
                SloStructuredCostCapture::HostSettledV1,
            )
            .unwrap();
            let expected = counts
                .iter()
                .map(|&n| {
                    8 + cost_route::native_projection_dispatches(&qkv, n, quantized).unwrap()
                        + cost_route::native_projection_dispatches(&output, n, quantized).unwrap()
                })
                .sum();
            observed
                .validate_command(8, expected, counts.len() as u64 * 2)
                .unwrap();
            observed
                .algorithm_work()
                .unwrap()
                .unwrap()
                .validate_command(&observed)
                .unwrap();
            SelectedReplayAlgorithmTemplateV1::from_selected(
                &observed,
                8,
                expected,
                counts.len() as u64 * 2,
            )
            .unwrap()
            .validate_binding(&observed)
            .unwrap();
        }
    }
}

#[test]
fn cuda_selected_gdn_replay_binds_same_grid_fixed_parameters_and_pair_selection() {
    let (qkv, output) = matrices(false);
    let make = |shape, pair| {
        compute(
            shape,
            AttentionPrecision::F32Master,
            projection(&qkv, false),
            ProjectionEvidence::Native {
                input: &qkv,
                output: &output,
            },
            [(8, 8, true)],
            8,
            8,
            pair,
            SloStructuredCostCapture::HostSettledV1,
        )
        .unwrap()
    };
    let original = make(shape(), true);
    original.validate_command(8, 12, 2).unwrap();
    let template = SelectedReplayAlgorithmTemplateV1::from_selected(&original, 8, 12, 2).unwrap();
    let unpaired = make(shape(), false);
    unpaired.validate_command(8, 13, 2).unwrap();
    assert!(template.validate_binding(&unpaired).is_err());
    let mut epsilon = shape();
    epsilon.epsilon *= 2.0;
    assert!(
        template.validate_binding(&make(epsilon, true)).is_err(),
        "same grid and work do not authorize different captured epsilon"
    );
    let mut mapping = shape();
    mapping.value_head_mapping = GatedDeltaValueHeadMapping::InterleavedByKeyHead;
    assert!(template.validate_binding(&make(mapping, true)).is_err());
}

#[test]
fn cuda_selected_gdn_disabled_is_lazy_and_uncovered_projection_or_population_is_unknown() {
    let (qkv, output) = matrices(false);
    let lazy = std::iter::from_fn(|| -> Option<(u64, u32, bool)> {
        panic!("disabled enumerated GDN leaves")
    });
    assert!(compute(
        shape(),
        AttentionPrecision::F32Master,
        projection(&qkv, false),
        ProjectionEvidence::Native {
            input: &qkv,
            output: &output
        },
        lazy,
        8,
        8,
        true,
        SloStructuredCostCapture::Disabled
    )
    .is_none());
    assert!(compute(
        shape(),
        AttentionPrecision::F32Master,
        AttentionProjection::F16,
        ProjectionEvidence::Native {
            input: &qkv,
            output: &output
        },
        [(8, 8, true)],
        8,
        8,
        true,
        SloStructuredCostCapture::HostSettledV1
    )
    .is_none());
    for leaves in [
        vec![],
        vec![(0, 8, true)],
        vec![(8, 0, true)],
        vec![(8, 8, false)],
        vec![(7, 8, true)],
        vec![(9, 8, true)],
    ] {
        assert!(compute(
            shape(),
            AttentionPrecision::F32Master,
            projection(&qkv, false),
            ProjectionEvidence::Native {
                input: &qkv,
                output: &output
            },
            leaves,
            8,
            8,
            true,
            SloStructuredCostCapture::HostSettledV1
        )
        .is_none());
    }
    let evidence = bindings(8, 8, SloStructuredCostCapture::HostSettledV1).unwrap();
    evidence.validate_command(8, 0, 8).unwrap();
    assert!(bindings(8, 8, SloStructuredCostCapture::Disabled).is_none());
}

#[test]
fn cuda_selected_gdn_projection_row_chunks_preserve_pair_tail_and_current_work() {
    let (qkv, _) = matrices(false);
    let rows = super::super::super::native_matrix::MAX_ROWS + 8;
    let mut builder = SelectedCommandCostBuilderV1::new_with_algorithm_work(rows);
    projection_work(&mut builder, &qkv, rows, 12352, None, true, 0, 0).unwrap();
    let evidence = builder.finish().unwrap();
    let dispatches = cost_route::native_projection_dispatches(&qkv, rows, false).unwrap();
    assert_eq!(
        dispatches, 7,
        "four ordinary projections and three tail projections"
    );
    evidence.validate_command(rows, dispatches, 0).unwrap();
    evidence
        .algorithm_work()
        .unwrap()
        .unwrap()
        .validate_command(&evidence)
        .unwrap();
}

#[test]
fn cublas_gdn_complete_sequence_matches_two_library_calls_and_all_native_work() {
    use ferrum_interfaces::execution_cost::AlgorithmWorkKindV1;
    let identity = CublasHandleApiIdentity::fixture_identity();
    for (tokens, participants, leaves) in [
        (1, 1, vec![(1, 1, false)]),
        (7, 1, vec![(7, 1, false)]),
        (8, 8, vec![(8, 8, true)]),
        (8, 2, vec![(3, 1, false), (5, 1, false)]),
    ] {
        let leaf_count = leaves.len() as u64;
        let selected = compute(
            shape(),
            AttentionPrecision::F32MasterGgufF16Projections,
            AttentionProjection::F16,
            ProjectionEvidence::Library(identity),
            leaves,
            tokens,
            participants,
            true,
            SloStructuredCostCapture::HostSettledV1,
        )
        .unwrap();
        selected
            .validate_command(tokens, leaf_count * 10, leaf_count * 2)
            .unwrap();
        let table = selected.algorithm_work().unwrap().unwrap();
        table.validate_command(&selected).unwrap();
        let commands = |kind| {
            table
                .entries()
                .iter()
                .filter(|entry| entry.kind() == kind)
                .map(|entry| entry.commands())
                .sum::<u64>()
        };
        assert_eq!(commands(AlgorithmWorkKindV1::LibraryCall), leaf_count * 2);
        assert_eq!(commands(AlgorithmWorkKindV1::Kernel), leaf_count * 8);
        assert_eq!(
            commands(AlgorithmWorkKindV1::HostToDevice) + commands(AlgorithmWorkKindV1::Fill),
            leaf_count * 2
        );
        // Library API work has no invented vendor grid, while the surrounding
        // recurrent kernels retain their real launch geometry.
        assert!(table
            .entries()
            .iter()
            .filter(|entry| entry.kind() == AlgorithmWorkKindV1::LibraryCall)
            .all(|entry| entry.work().grid_blocks == 0));
    }
}

#[test]
fn cublas_gdn_does_not_authorize_old_dense_contract_or_changed_replay_parameters() {
    let identity = CublasHandleApiIdentity::fixture_identity();
    let make = |shape, precision, rows, capture| {
        compute(
            shape,
            precision,
            AttentionProjection::F16,
            ProjectionEvidence::Library(identity),
            [(rows, 1, false)],
            rows,
            1,
            false,
            capture,
        )
    };
    let original = make(
        shape(),
        AttentionPrecision::F32MasterGgufF16Projections,
        3,
        SloStructuredCostCapture::HostSettledV1,
    )
    .unwrap();
    let template = SelectedReplayAlgorithmTemplateV1::from_selected(&original, 3, 10, 2).unwrap();
    template.validate_binding(&original).unwrap();
    let changed_rows = make(
        shape(),
        AttentionPrecision::F32MasterGgufF16Projections,
        7,
        SloStructuredCostCapture::HostSettledV1,
    )
    .unwrap();
    assert_eq!(original.family_signature(), changed_rows.family_signature());
    assert_ne!(
        original.algorithm_work().unwrap(),
        changed_rows.algorithm_work().unwrap()
    );
    assert!(template.validate_binding(&changed_rows).is_err());
    let mut changed = shape();
    changed.epsilon *= 2.0;
    assert!(template
        .validate_binding(
            &make(
                changed,
                AttentionPrecision::F32MasterGgufF16Projections,
                3,
                SloStructuredCostCapture::HostSettledV1
            )
            .unwrap()
        )
        .is_err());
    assert!(make(
        shape(),
        AttentionPrecision::F32Master,
        3,
        SloStructuredCostCapture::HostSettledV1
    )
    .is_none());
    assert!(make(
        shape(),
        AttentionPrecision::F16,
        3,
        SloStructuredCostCapture::HostSettledV1
    )
    .is_none());
    let lazy = std::iter::from_fn(|| -> Option<(u64, u32, bool)> {
        panic!("disabled enumerated library leaves")
    });
    assert!(compute(
        shape(),
        AttentionPrecision::F32MasterGgufF16Projections,
        AttentionProjection::F16,
        ProjectionEvidence::Library(identity),
        lazy,
        8,
        8,
        true,
        SloStructuredCostCapture::Disabled
    )
    .is_none());
    assert!(make(
        shape(),
        AttentionPrecision::F32MasterGgufF16Projections,
        0,
        SloStructuredCostCapture::HostSettledV1
    )
    .is_none());
    assert!(make(
        shape(),
        AttentionPrecision::F32MasterGgufF16Projections,
        u64::MAX,
        SloStructuredCostCapture::HostSettledV1
    )
    .is_none());
}
