use super::*;
use crate::fast_h3::{FastH3FfnConfig, FastH3FfnFamily, FastH3FfnLimits, FastH3FfnSource};
use ferrum_runtime::*;
use std::sync::Arc;

fn ffn_limits() -> FastH3FfnLimits {
    FastH3FfnLimits {
        maximum_rows: 129,
        maximum_plan_nodes: 128,
        maximum_metadata_bytes: 32768,
        maximum_stored_bytes: 1 << 20,
    }
}
fn host(bytes: usize) -> HostMemoryAccount {
    HostMemoryAccount::new(
        HostMemoryLimits::new(
            bytes,
            HostMemoryClassLimits {
                input_preprocess_bytes: 0,
                transfer_staging_bytes: 0,
                output_bytes: 0,
                weight_backing_bytes: bytes,
            },
        )
        .unwrap(),
    )
}
fn source(
    fixture: &Fixture,
    account: &HostMemoryAccount,
) -> (tempfile::TempDir, FastH3FfnConfig, FastH3FfnSource) {
    let (dir, package) = fixture.open(limits());
    let (config, source) =
        FastH3FfnSource::prepare(Arc::new(package.unwrap()), 0, 3, account, ffn_limits()).unwrap();
    (dir, config, source)
}

#[test]
fn static_qad_ffn_derives_original_alpha_and_retains_only_managed_scalar_backing() {
    let account = host(8);
    let (_dir, config, source) = source(&Fixture::new(), &account);
    let family = FastH3FfnFamily::new(ffn_limits()).unwrap();
    let ModelWeights::Stored(schema) = family.weights(&config).unwrap() else {
        panic!("Stored FFN")
    };
    assert_eq!(account.usage().unwrap().allocated_bytes(), 8);
    for recipe in config.scalar_derivations() {
        // Independent literal fixture: alpha=.3f32, SF=8.0f32. The rounded
        // BF16 weight-global reciprocal would give a different result.
        assert_eq!(recipe.input_bits(), [0x3e99_999a, 0x4100_0000]);
        assert_eq!(recipe.result_bits(), 0x3d19_999a);
        assert_ne!(recipe.result_bits(), (1.0f32 / 3.328125 / 8.0).to_bits());
        assert_eq!(recipe.source_alpha().role, WeightComponentRole::Metadata);
        assert!(recipe.source_alpha().dimensions.is_empty());
        assert_eq!(
            recipe.source_files(),
            &[
                "nvfp4_weights.safetensors".to_owned(),
                "nvfp4_weights.safetensors".to_owned()
            ]
        );
    }
    assert_eq!(schema.components.len(), 8);
    let mut alpha_payload = None;
    for c in &schema.components {
        let payload = source.component(c).unwrap();
        assert!(payload.retained_host_memory().is_some());
        assert_eq!(payload.bytes().len() as u64, c.physical_bytes().unwrap());
        if c.role == WeightComponentRole::Values {
            assert_eq!(payload.bytes(), 0x3d19_999au32.to_le_bytes());
            alpha_payload = Some(payload);
        } else {
            let tensor=schema.tensors.iter().find(|t|matches!(&t.physical_layout,PhysicalWeightLayout::PhysicalComponent(v) if v.component_id()==&c.id)).unwrap();
            assert!(tensor.logical_elements().is_err());
            let bound = ResolvedWeightBinding::from_schema(&schema, &tensor.id).unwrap();
            assert_eq!(
                bound.physical_component_view().unwrap().encoding(),
                &c.encoding
            );
        }
        let mut forged = c.clone();
        forged.external_names = vec!["unrelated.weight".into()];
        assert!(source.component(&forged).is_err());
    }
    // A retained upload backing holds its original account until released.
    let retained = alpha_payload
        .unwrap()
        .retained_host_memory()
        .unwrap()
        .clone();
    drop(source);
    assert_eq!(account.usage().unwrap().allocated_bytes(), 4);
    drop(retained);
    assert_eq!(account.usage().unwrap().allocated_bytes(), 0);
}

#[test]
fn static_qad_ffn_program_preserves_value_first_split_and_declared_rounding_boundaries() {
    let (_dir, config, _source) = source(&Fixture::new(), &host(8));
    let family = FastH3FfnFamily::new(ffn_limits()).unwrap();
    let raw = serde_json::to_value(&config).unwrap();
    let prepared = TypedFamilyRegistration::new(family)
        .prepare_with_profile(
            &raw,
            &NumericalProfileId::new(FastH3FfnFamily::PROFILE_ID).unwrap(),
        )
        .unwrap();
    let graph = prepared.program();
    let nodes = &graph.blocks()[0].nodes;
    assert_eq!(
        nodes
            .iter()
            .map(|n| n.operation_id.as_str())
            .collect::<Vec<_>>(),
        [
            NVFP4_QUANTIZE_BF16_OPERATION_ID,
            NVFP4_MATMUL_BF16_OPERATION_ID,
            PACKED_SWIGLU_BF16_OPERATION_ID,
            NVFP4_QUANTIZE_BF16_OPERATION_ID,
            NVFP4_MATMUL_BF16_OPERATION_ID,
        ]
    );
    assert_eq!(
        nodes[2].attributes,
        PackedSwiGluBf16Profile::Exp2SigmoidF32SingleRound
            .attributes()
            .unwrap()
    );
    assert_eq!(nodes[2].inputs, nodes[1].outputs);
    assert_eq!(nodes[3].inputs[0], nodes[2].outputs[0]);
    for index in [0, 3] {
        assert_eq!(
            nodes[index].attributes,
            Nvfp4QuantizeProfile::CudaRcpApproxFtzV1
                .attributes()
                .unwrap()
        );
    }
    let options = config.compile_options().unwrap();
    for index in [1, 2, 4] {
        assert_eq!(
            options.tensor_specs()[&nodes[index].outputs[0]].element_type,
            ElementType::Bf16
        );
    }
    assert_eq!(
        options.tensor_specs()[&nodes[4].outputs[0]].dimensions,
        [3, 32]
    );
    let ModelWeights::Stored(schema) = prepared.weights() else {
        panic!("stored")
    };
    validate_graph_bindings(graph, schema, &options);
    // Quantizer global SF is the source Metadata view. GEMM alpha is a
    // different numerical Dense Values producer, never the raw alpha view.
    for (node_index, ordinal, is_physical) in
        [(0, 1, true), (3, 1, true), (1, 4, false), (4, 4, false)]
    {
        let operand = &nodes[node_index].inputs[ordinal];
        let reference = graph
            .weights()
            .iter()
            .find(|w| &w.value_id == operand)
            .unwrap();
        let bound = ResolvedWeightBinding::from_schema(schema, &reference.weight_id).unwrap();
        assert_eq!(bound.physical_component_view().is_some(), is_physical);
    }
}

#[test]
fn static_qad_ffn_admission_rolls_back_and_does_not_replace_missing_static_scale() {
    let fixture = Fixture::new();
    let (_dir, package) = fixture.open(limits());
    let package = Arc::new(package.unwrap());
    let account = host(7);
    let error = FastH3FfnSource::prepare(package.clone(), 0, 3, &account, ffn_limits())
        .err()
        .unwrap();
    assert!(matches!(
        error.downcast_ref::<HostMemoryError>(),
        Some(HostMemoryError::Capacity { .. })
    ));
    assert_eq!(account.usage().unwrap().committed_bytes(), 0);
    assert!(FastH3FfnSource::prepare(package.clone(), 1, 3, &account, ffn_limits()).is_err());
    assert!(FastH3FfnSource::prepare(package, 0, 130, &account, ffn_limits()).is_err());
    assert_eq!(account.usage().unwrap().committed_bytes(), 0);
    let mut fixture = Fixture::new();
    fixture
        .packed
        .remove("minimax_h3.transformer_blocks.0.ff.fc_in::_nvfp4_input_global_sf");
    let (_dir, package) = fixture.open(limits());
    assert!(
        FastH3FfnSource::prepare(Arc::new(package.unwrap()), 0, 3, &account, ffn_limits()).is_err()
    );
    assert_eq!(account.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn static_qad_ffn_rejects_overflowing_derived_alpha_and_tampered_recipe() {
    let mut fixture = Fixture::new();
    fixture
        .packed
        .get_mut("minimax_h3.transformer_blocks.0.ff.fc_in::_nvfp4_alpha")
        .unwrap()
        .2 = f32::MAX.to_le_bytes().to_vec();
    fixture
        .packed
        .get_mut("minimax_h3.transformer_blocks.0.ff.fc_in::_nvfp4_input_global_sf")
        .unwrap()
        .2 = 0.5f32.to_le_bytes().to_vec();
    let (_dir, package) = fixture.open(limits());
    let account = host(8);
    assert!(
        FastH3FfnSource::prepare(Arc::new(package.unwrap()), 0, 3, &account, ffn_limits()).is_err()
    );
    assert_eq!(account.usage().unwrap().allocated_bytes(), 0);
    let (_dir, config, _source) = source(&Fixture::new(), &account);
    let family = FastH3FfnFamily::new(ffn_limits()).unwrap();
    let mut json = serde_json::to_value(&config).unwrap();
    json["projections"][0]["derivation"]["result_bits"] = 0u32.into();
    assert!(family.parse_config(&json).is_err());
    let mut json = serde_json::to_value(&config).unwrap();
    json["projections"][0]["global_sf"]["role"] = "Values".into();
    assert!(family.parse_config(&json).is_err());
    let mut json = serde_json::to_value(&config).unwrap();
    json["block"] = 1usize.into();
    assert!(family.parse_config(&json).is_err());
}

#[test]
fn row_partition_preserves_weights_profiles_and_short_tail_geometry() {
    let (_dir, whole, _source) = source(&Fixture::new(), &host(8));
    let whole_prepared = TypedFamilyRegistration::new(FastH3FfnFamily::new(ffn_limits()).unwrap())
        .prepare_with_profile(
            &serde_json::to_value(&whole).unwrap(),
            &NumericalProfileId::new(FastH3FfnFamily::PROFILE_ID).unwrap(),
        )
        .unwrap();
    for rows in [1, 2, 3, 4] {
        let config = whole
            .clone()
            .with_chunk_rows(std::num::NonZeroU64::new(rows));
        let prepared = TypedFamilyRegistration::new(FastH3FfnFamily::new(ffn_limits()).unwrap())
            .prepare_with_profile(
                &serde_json::to_value(&config).unwrap(),
                &NumericalProfileId::new(FastH3FfnFamily::PROFILE_ID).unwrap(),
            )
            .unwrap();
        assert_eq!(prepared.weights(), whole_prepared.weights());
        let ModelWeights::Stored(schema) = prepared.weights() else {
            panic!("stored")
        };
        let options = config.compile_options().unwrap();
        validate_graph_bindings(prepared.program(), schema, &options);
        let output =
            &options.tensor_specs()[&ProgramValueId::new(FastH3FfnFamily::OUTPUT_ID).unwrap()];
        assert_eq!(output.dimensions, [3, whole.hidden()]);
        let nodes = &prepared.program().blocks()[0].nodes;
        for node in nodes {
            if node.operation_id.as_str() == PACKED_SWIGLU_BF16_OPERATION_ID {
                assert_eq!(
                    node.attributes,
                    PackedSwiGluBf16Profile::Exp2SigmoidF32SingleRound
                        .attributes()
                        .unwrap()
                );
            }
        }
        if rows < whole.rows() {
            let gathers: Vec<_> = nodes
                .iter()
                .filter(|n| n.operation_id.as_str() == ROW_GATHER_BF16_OPERATION_ID)
                .collect();
            let lengths: Vec<_> = gathers
                .iter()
                .map(|n| n.attributes[&AttributeId::new("run_lengths").unwrap()].clone())
                .collect();
            let expected: Vec<_> = std::iter::once(whole.rows() as i64)
                .chain(
                    match rows {
                        1 => vec![1, 1, 1],
                        2 => vec![2, 1],
                        _ => unreachable!(),
                    }
                    .into_iter(),
                )
                .map(|n| SemanticValue::Integers(vec![n]))
                .collect();
            assert_eq!(lengths, expected);
        }
    }
    let constrained = FastH3FfnLimits {
        maximum_plan_nodes: 5,
        ..ffn_limits()
    };
    let family = FastH3FfnFamily::new(constrained).unwrap();
    assert!(family.weights(&whole).is_ok());
    assert!(family
        .weights(&whole.with_chunk_rows(std::num::NonZeroU64::new(2)))
        .is_err());
}

// Validate the real graph's operation boundaries against the production
// descriptors, including Stored interpretations. These metadata resources do
// not claim execution; the device owner must still compile/admit/complete it.
pub(super) fn validate_graph_bindings(
    graph: &ModelProgram,
    schema: &WeightSchema,
    options: &ProgramPlanCompileOptions,
) {
    for node in graph.blocks().iter().flat_map(|b| &b.nodes) {
        let contract = match node.operation_id.as_str() {
            AFFINE_LINEAR_F32_OPERATION_ID => affine_linear_f32_contract(),
            AFFINE_LINEAR_F32_TO_BF16_OPERATION_ID => affine_linear_f32_to_bf16_contract(),
            CAST_F32_TO_BF16_OPERATION_ID => cast_f32_to_bf16_contract(),
            ROW_GATHER_BF16_OPERATION_ID => row_gather_bf16_contract(),
            ROW_WRITE_BF16_OPERATION_ID => row_write_bf16_contract(),
            NVFP4_QUANTIZE_BF16_OPERATION_ID => nvfp4_quantize_bf16_contract(),
            NVFP4_MATMUL_BF16_OPERATION_ID => nvfp4_matmul_bf16_contract(),
            PACKED_SWIGLU_BF16_OPERATION_ID => packed_swiglu_bf16_contract(),
            ADDCMUL_FMA_BF16_OPERATION_ID => addcmul_fma_bf16_contract(),
            ROW_INDEX_GATHER_OR_ZERO_BF16_OPERATION_ID => row_index_gather_or_zero_bf16_contract(),
            MERGE_LEADING_BATCH_BF16_VIEW_OPERATION_ID => {
                LeadingBatchBf16Kind::Merge.view_contract()
            }
            SPLIT_LEADING_BATCH_BF16_VIEW_OPERATION_ID => {
                LeadingBatchBf16Kind::Split.view_contract()
            }
            HEAD_RMS_PARTIAL_ROPE_SINGLE_ROUND_BF16_OPERATION_ID => {
                head_rms_partial_rope_single_round_bf16_contract()
            }
            MASKED_TILE_MEAN_BF16_F32_OPERATION_ID => masked_tile_mean_bf16_f32_contract(),
            BATCHED_MATMUL_F32_OPERATION_ID => batched_matmul_f32_contract(),
            SCALE_RANK4_F32_OPERATION_ID => scale_rank4_f32_contract(),
            ROW_SOFTMAX_RANK4_F32_OPERATION_ID => row_softmax_rank4_f32_contract(),
            ROW_CENTER_BF16_OPERATION_ID => row_center_bf16_contract(),
            HEAD_FP8_ROUNDTRIP_BF16_OPERATION_ID => head_fp8_roundtrip_bf16_contract(),
            INDEXED_HEAD_FP8_ROUNDTRIP_BF16_OPERATION_ID => {
                indexed_head_fp8_roundtrip_bf16_contract()
            }
            CROSS_ATTENTION_BF16_OPERATION_ID => cross_attention_bf16_contract(),
            CROSS_ATTENTION_FUSED_BF16_OPERATION_ID => cross_attention_fused_bf16_contract(),
            SPARSE_FP4_QKV_QUANTIZE_BF16_OPERATION_ID => sparse_fp4_qkv_quantize_bf16_contract(),
            SPARSE_FP4_ATTENTION_REGION_TOPK_OPERATION_ID => {
                sparse_fp4_attention_region_topk_contract()
            }
            PACK_HEAD_AXIS_BF16_OPERATION_ID => HeadAxisPackKind::Bf16.contract(),
            PACK_HEAD_AXIS_F32_TO_BF16_OPERATION_ID => HeadAxisPackKind::F32ToBf16.contract(),
            BATCHED_ROW_CONCAT_BF16_OPERATION_ID => batched_row_concat_bf16_contract(),
            ROW_SELECTED_RMS_MODULATION_BF16_OPERATION_ID => {
                row_selected_rms_modulation_bf16_contract()
            }
            ROW_SELECTED_RESIDUAL_RMS_MODULATION_BF16_OPERATION_ID => {
                row_selected_residual_rms_modulation_bf16_contract()
            }
            ROW_SELECTED_ADDCMUL_FMA_BF16_OPERATION_ID => row_selected_addcmul_fma_bf16_contract(),
            _ => panic!("unqualified model operation {}", node.operation_id),
        }
        .unwrap();
        contract
            .descriptor()
            .validate_attributes(&node.attributes)
            .unwrap();
        let mut bindings: Vec<ResolvedValueBinding> = Vec::new();
        for (input, values) in [(true, &node.inputs), (false, &node.outputs)] {
            for (ordinal, id) in values.iter().enumerate() {
                let port = if input {
                    &contract.descriptor().inputs[ordinal]
                } else {
                    &contract.descriptor().outputs[ordinal]
                };
                let weight = graph.weights().iter().find(|w| &w.value_id == id);
                let spec = weight
                    .map(|w| &w.tensor)
                    .unwrap_or_else(|| &options.tensor_specs()[id]);
                let resolved = weight
                    .map(|w| ResolvedWeightBinding::from_schema(schema, &w.weight_id).unwrap());
                let resource = ResourceId::new(format!("resource.{id}")).unwrap();
                let storage = if let AliasPolicy::ReadOnlyView { tensor_index }
                | AliasPolicy::MustAlias { tensor_index } = port.alias()
                {
                    bindings[*tensor_index as usize].storage().clone()
                } else if let Some(w) = &resolved {
                    assert_eq!(w.components().len(), 1);
                    let component = w.components()[0].component_id();
                    let original = schema
                        .components
                        .iter()
                        .find(|c| &c.id == component)
                        .unwrap();
                    ResolvedValueStorage::composite(vec![ResolvedStorageComponent::new(
                        Some(component.clone()),
                        resource,
                        0,
                        original.physical_bytes().unwrap(),
                        original.physical_element_type(),
                    )
                    .unwrap()])
                    .unwrap()
                } else {
                    ResolvedValueStorage::single(
                        resource,
                        0,
                        spec.byte_len().unwrap(),
                        spec.element_type,
                    )
                    .unwrap()
                };
                bindings.push(
                    ResolvedValueBinding::new(
                        id.clone(),
                        if input {
                            ResolvedValueRole::Input
                        } else {
                            ResolvedValueRole::Output
                        },
                        ordinal as u32,
                        ResolvedTensorSpec::new(
                            spec.dimensions.clone(),
                            spec.element_type,
                            spec.layout.clone(),
                        )
                        .unwrap(),
                        port.access(),
                        port.alias().clone(),
                        if resolved.is_some() {
                            BufferUsage::Weights
                        } else {
                            BufferUsage::Activations
                        },
                        resolved,
                        storage,
                    )
                    .unwrap(),
                );
            }
        }
        contract
            .descriptor()
            .validate_resolved_bindings(&bindings)
            .unwrap();
    }
}
