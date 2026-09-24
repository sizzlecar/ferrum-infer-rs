use super::*;
use ferrum_interfaces::vnext::{
    BlockQuantizationSpec, CompositeWeightPart, ContractVersion, HadamardApplication,
    HadamardSigns, HadamardTransformSpec, PhysicalWeightComponentBinding, PhysicalWeightPadding,
    WeightComponentRole, WeightComponentSpec, WeightEncoding, WeightSchema, WeightTensorSpec,
};
use std::num::{NonZeroU32, NonZeroU64};

fn id<T>(text: &str) -> T
where
    T: TryFrom<String>,
    T::Error: std::fmt::Debug,
{
    T::try_from(text.to_owned()).unwrap()
}

#[derive(Clone, Copy)]
enum Format {
    Dense,
    Q4K,
    Q6K,
}

fn schema(parts: &[(u64, Format)], input: u64) -> WeightSchema {
    let mut offset = 0;
    let mut components = Vec::new();
    let mut layouts = Vec::new();
    for (index, &(output, format)) in parts.iter().enumerate() {
        let name = format!("component.projection.{index}");
        let quantized = !matches!(format, Format::Dense);
        components.push(WeightComponentSpec {
            id: id(&name),
            role: if quantized {
                WeightComponentRole::PackedValues
            } else {
                WeightComponentRole::Values
            },
            external_names: vec![name.clone()],
            dimensions: vec![output, if quantized { input / 256 } else { input }],
            encoding: match format {
                Format::Dense => WeightEncoding::Dense {
                    element_type: ElementType::F16,
                },
                Format::Q4K | Format::Q6K => {
                    WeightEncoding::BlockQuantized(BlockQuantizationSpec {
                        format_id: id(if matches!(format, Format::Q4K) {
                            "quantization.gguf.q4-k"
                        } else {
                            "quantization.gguf.q6-k"
                        }),
                        logical_values_per_block: 256,
                        bytes_per_block: if matches!(format, Format::Q4K) {
                            144
                        } else {
                            210
                        },
                    })
                }
            },
            required: true,
        });
        let layout = if quantized {
            PhysicalWeightLayout::BlockQuantized {
                blocks: PhysicalWeightComponentBinding::exact_contiguous(id(&name)),
                block_axis: 1,
                block_padding: PhysicalWeightPadding::Exact,
            }
        } else {
            PhysicalWeightLayout::Dense {
                component_id: id(&name),
            }
        };
        layouts.push(CompositeWeightPart {
            layout: Box::new(layout),
            logical_offsets: vec![offset, 0],
            extents: vec![output, input],
        });
        offset += output;
    }
    let physical_layout = if layouts.len() == 1 {
        *layouts.remove(0).layout
    } else {
        PhysicalWeightLayout::Composite { parts: layouts }
    };
    WeightSchema {
        format_id: id("weight-format.gguf.native-block"),
        layout_id: id("weight-layout.gdn-cost-test"),
        version: ContractVersion::new(1, 0),
        components,
        tensors: vec![WeightTensorSpec {
            id: id("weight.projection"),
            dimensions: vec![offset, input],
            logical_element_type: ElementType::F16,
            physical_layout,
            required: true,
        }],
    }
}

fn resolved(schema: &WeightSchema) -> ResolvedWeightBinding {
    ResolvedWeightBinding::from_schema(schema, &id("weight.projection")).unwrap()
}

fn shape() -> AttentionShape {
    AttentionShape {
        hidden_size: 256,
        key_heads: 2,
        value_heads: 4,
        key_dim: 64,
        value_dim: 64,
        qkv_features: 512,
        value_features: 256,
        qkvz_features: 768,
        ba_features: 8,
        qkvzba_features: 776,
        conv_kernel: 4,
        conv_state_width: 3,
        epsilon: 1e-6,
        layer_index: 0,
        decay_parameterization: GatedDeltaDecayParameterization::NegativeRate,
        value_head_mapping: GatedDeltaValueHeadMapping::InterleavedByKeyHead,
    }
}

fn capabilities() -> GatedDeltaExecutionCapabilities {
    GatedDeltaExecutionCapabilities::with_chunked_scan(64).unwrap()
}

fn row(count: u64) -> OperationCostWorkRow {
    OperationCostWorkRow {
        offset: 0,
        count: NonZeroU64::new(count).unwrap(),
        full_input_tokens: NonZeroU64::new(count).unwrap(),
    }
}

fn route(rows: &[OperationCostWorkRow], packed: bool, simd_width: u64) -> OperationCostCommand {
    let shape = shape();
    let input = plain_projection(
        &resolved(&schema(&[(776, Format::Q4K)], 256)),
        776,
        256,
        true,
    )
    .unwrap()
    .unwrap();
    let output = plain_projection(
        &resolved(&schema(&[(256, Format::Q6K)], 256)),
        256,
        256,
        false,
    )
    .unwrap()
    .unwrap()[0];
    let total = rows.iter().map(|row| row.count.get()).sum();
    project(
        shape,
        ElementType::F16,
        rows,
        total,
        packed,
        &input,
        output,
        ScratchLayout::new(shape, total).unwrap(),
        0,
        capabilities(),
        MetalGatedDeltaExecutionCostModel::initial_c64(simd_width, 128),
    )
    .unwrap()
}

#[test]
fn plain_q4_q6_rows_select_packed_and_participant_loop_accounting() {
    let rows = [row(1), row(3)];
    let packed = route(&rows, true, 32);
    assert_eq!(packed.batching(), DeviceBatchingForm::Packed);
    assert_eq!(packed.participant_count(), 2);
    assert_eq!(packed.token_count(), 4);
    // Five shared non-projection kernels, two shared projections, and four
    // local kernels per recurrent row. No projection runs once per row here.
    assert_eq!(packed.compute_dispatch_count(), 15);
    let separate = route(&rows, false, 32);
    assert_eq!(separate.batching(), DeviceBatchingForm::ParticipantLoop);
    assert_eq!(separate.compute_dispatch_count(), 22);
    assert_eq!(
        route(&rows[..1], false, 32).batching(),
        DeviceBatchingForm::Scalar
    );
}

#[test]
fn real_selector_uses_pipeline_capability_and_each_rows_chunk_threshold() {
    let rows = [row(63), row(64)];
    assert_eq!(
        route(&rows, true, 32).native_operation(),
        "vnext_gated_delta_recurrent_attention"
    );
    let mixed = route(&rows, true, 16);
    assert_eq!(
        mixed.native_operation(),
        "vnext_gated_delta_mixed_attention"
    );
    assert_eq!(mixed.compute_dispatch_count(), 19);
    assert_eq!(
        route(&[row(64)], false, 16).native_operation(),
        "vnext_gated_delta_chunked_attention"
    );
    // Repeated K128 heads insert the gram kernel into the selected chunk path.
    let mut larger = shape();
    larger.key_dim = 128;
    larger.value_dim = 128;
    larger.value_features = 512;
    larger.qkv_features = 1024;
    larger.qkvz_features = 1536;
    larger.qkvzba_features = 1544;
    let form = larger
        .execution_form(
            64,
            capabilities(),
            MetalGatedDeltaExecutionCostModel::initial_c64(16, 128),
        )
        .unwrap();
    assert_eq!(delta_dispatch_count(form, &larger.params(64).unwrap()), 6);
}

#[test]
fn plain_partition_uses_actual_leaf_abi_and_full_row_coverage() {
    let definition = schema(&[(512, Format::Q4K), (264, Format::Dense)], 256);
    let parts = plain_projection(&resolved(&definition), 776, 256, true)
        .unwrap()
        .unwrap();
    assert_eq!(parts.len(), 2);
    assert!(plain_projection(&resolved(&definition), 776, 256, false)
        .unwrap()
        .is_none());
    assert!(plain_projection(&resolved(&definition), 775, 256, true).is_err());
    assert!(plain_projection(&resolved(&definition), 776, 255, true).is_err());
    let mut reordered = definition.clone();
    let PhysicalWeightLayout::Composite { parts } = &mut reordered.tensors[0].physical_layout
    else {
        unreachable!()
    };
    parts.reverse();
    assert_eq!(
        plain_projection(&resolved(&reordered), 776, 256, true)
            .unwrap()
            .unwrap()
            .len(),
        2
    );
    let oversized = schema(&vec![(16, Format::Q4K); MAX_PLAIN_PARTS + 1], 256);
    assert!(plain_projection(
        &resolved(&oversized),
        16 * (MAX_PLAIN_PARTS as u64 + 1),
        256,
        true
    )
    .unwrap()
    .is_none());
}

#[test]
fn transformed_projection_remains_unknown_instead_of_counting_plain_work() {
    let mut transformed = schema(&[(776, Format::Q4K)], 256);
    let values = transformed.tensors[0].physical_layout.clone();
    transformed.tensors[0].physical_layout = PhysicalWeightLayout::Hadamard {
        values: Box::new(values),
        transform: HadamardTransformSpec {
            block_size: NonZeroU32::new(256).unwrap(),
            signs: HadamardSigns::Identity,
            application: HadamardApplication::BeforeMatmul {
                input_permutation: None,
            },
        },
    };
    assert!(plain_projection(&resolved(&transformed), 776, 256, true)
        .unwrap()
        .is_none());
}

#[test]
fn gdn_staging_policy_preserves_q4_threshold_and_q6_non_staged_route() {
    let shape = AttentionShape {
        hidden_size: 2048,
        ..shape()
    };
    let output = plain_projection(
        &resolved(&schema(&[(2048, Format::Q6K)], 256)),
        2048,
        256,
        false,
    )
    .unwrap()
    .unwrap()[0];
    for (format, staged_input_dispatches) in [(Format::Q4K, 2), (Format::Q6K, 1)] {
        let input = plain_projection(&resolved(&schema(&[(776, format)], 2048)), 776, 2048, true)
            .unwrap()
            .unwrap();
        let layout = ScratchLayout::new(shape, 768).unwrap();
        for (tokens, expected) in [(767, 1), (768, staged_input_dispatches)] {
            let projected = project_dispatches(
                shape,
                &input,
                output,
                layout,
                0,
                tokens,
                Some(staged_prefill::StagingPolicy::GatedDelta),
            )
            .unwrap();
            assert_eq!(projected.input, expected);
            assert_eq!(projected.output, 1);
        }
        // No retained staging owner means the same plain quantized route.
        assert_eq!(
            project_dispatches(shape, &input, output, layout, 0, 768, None)
                .unwrap()
                .input,
            1
        );
    }
}

#[test]
fn command_retains_f32_master_identity_and_actual_local_kernel_count() {
    let recurrent = || {
        Ok(RowDispatches {
            projections: ProjectionDispatches {
                input: 2,
                output: 1,
            },
            delta: 1,
            chunked: false,
        })
    };
    let chunked = || {
        Ok(RowDispatches {
            projections: ProjectionDispatches {
                input: 2,
                output: 1,
            },
            delta: 6,
            chunked: true,
        })
    };
    let mixed = command(
        ElementType::F32,
        65,
        None,
        [recurrent(), chunked()].into_iter(),
    )
    .unwrap();
    assert_eq!(
        mixed.native_operation(),
        "vnext_gated_delta_mixed_attention_f32_master"
    );
    assert_eq!(mixed.compute_dispatch_count(), 29);
    assert_eq!(mixed.phase(), DeviceCommandPhase::Compute);
    assert_eq!(mixed.transfer_command_count(), 0);
    assert_eq!(
        command(ElementType::F32, 1, None, [recurrent()].into_iter())
            .unwrap()
            .native_operation(),
        "vnext_gated_delta_recurrent_attention_f32_master"
    );
    assert_eq!(
        command(ElementType::F32, 64, None, [chunked()].into_iter())
            .unwrap()
            .native_operation(),
        "vnext_gated_delta_chunked_attention_f32_master"
    );
}

#[test]
fn numerical_scratch_kernel_integer_and_dispatch_overflow_are_rejected() {
    let shape = shape();
    let input = plain_projection(
        &resolved(&schema(&[(776, Format::Q4K)], 256)),
        776,
        256,
        true,
    )
    .unwrap()
    .unwrap();
    let output = plain_projection(
        &resolved(&schema(&[(256, Format::Q6K)], 256)),
        256,
        256,
        false,
    )
    .unwrap()
    .unwrap()[0];
    let layout = ScratchLayout::new(shape, 2).unwrap();
    assert!(project_dispatches(shape, &input, output, layout, u64::MAX, 1, None).is_err());
    assert!(project_dispatches(shape, &input, output, layout, 2, 3, None).is_err());
    assert!(shape
        .execution_form(
            u64::from(u32::MAX) / 776 + 1,
            capabilities(),
            MetalGatedDeltaExecutionCostModel::initial_c64(32, 128)
        )
        .is_err());
    let row = || {
        Ok(RowDispatches {
            projections: ProjectionDispatches {
                input: u64::MAX,
                output: 1,
            },
            delta: 1,
            chunked: false,
        })
    };
    assert!(command(ElementType::F16, 1, None, [row()].into_iter()).is_err());
    assert!(command(ElementType::F16, 1, None, std::iter::empty()).is_err());
    assert!(command(
        ElementType::F16,
        1,
        None,
        [Err("actual projection failed".into())].into_iter()
    )
    .is_err());
}
