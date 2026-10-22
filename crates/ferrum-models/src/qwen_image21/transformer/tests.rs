use super::*;
use serde_json::json;
type F = QwenImage21TransformerFamily;

#[test]
fn fp8_transformer_scope_groups_qkv_but_keeps_all_boundary_projections_bf16() {
    let config = config(true);
    let family = F::new().unwrap();
    let profiles = family.numerical_profiles(&config).unwrap();
    let bf16 = profiles
        .resolve(&profile_id(F::CUDA_PROFILE_ID).unwrap())
        .unwrap();
    let fp8 = profiles
        .resolve(&profile_id(F::CUDA_FP8_PROFILE_ID).unwrap())
        .unwrap();
    let a = family.semantic_program(&config, bf16).unwrap();
    let b = family.semantic_program(&config, fp8).unwrap();
    let groups = F::fp8_weight_groups(&config).unwrap();
    assert_eq!(groups.len(), 10);
    for (layer, group) in groups.chunks_exact(5).enumerate() {
        assert_eq!(
            group[0],
            ["attn.to_q", "attn.to_k", "attn.to_v"].map(|name| WeightId::new(format!(
                "weight.qwen_image21.transformer_blocks.{layer}.{name}.weight"
            ))
            .unwrap())
        );
        assert!(group[1..].iter().all(|g| g.len() == 1));
    }
    assert_eq!(a.weights(), b.weights());
    assert_eq!(a.input_tensor_specs(), b.input_tensor_specs());
    assert_eq!(bf16.boundaries, fp8.boundaries);
    assert!(bf16.validate_program(&b).is_err());
    assert!(fp8.validate_program(&a).is_err());
    let selected: BTreeSet<_> = groups.iter().flatten().collect();
    let mut converted = 0;
    let mut unchanged_dense = 0;
    for (left, right) in a
        .blocks()
        .iter()
        .flat_map(|b| &b.nodes)
        .zip(b.blocks().iter().flat_map(|b| &b.nodes))
    {
        let mut restored = right.clone();
        if right.operation_id.as_str() == DENSE_LINEAR_FP8_BF16_OPERATION_ID {
            let weight = a
                .weights()
                .iter()
                .find(|w| w.value_id == right.inputs[1])
                .unwrap();
            assert!(selected.contains(&weight.weight_id));
            restored.operation_id = left.operation_id.clone();
            converted += 1;
        } else if right.operation_id.as_str() == DENSE_LINEAR_BF16_OPERATION_ID {
            unchanged_dense += 1;
        }
        assert_eq!(&restored, left);
    }
    assert_eq!(converted, 14);
    assert!(unchanged_dense > 0);
    assert_ne!(a.fingerprint().unwrap(), b.fingerprint().unwrap());
}

#[test]
fn fused_transformer_profile_selects_every_dit_attention_without_changing_transport() {
    let config = QwenImage21TransformerConfig::new(
        transformer(2, true, true),
        QwenImage21T2iLayout::new(3, 2, 2, vec![3, 1]).unwrap(),
    )
    .unwrap();
    let family = F::new().unwrap();
    let profiles = family.numerical_profiles(&config).unwrap();
    let fused_id = profile_id(F::CUDA_FUSED_PROFILE_ID).unwrap();
    let legacy = profiles
        .resolve(&profile_id(F::CUDA_PROFILE_ID).unwrap())
        .unwrap();
    let fused = profiles.resolve(&fused_id).unwrap();
    assert!(!profiles.auto_preference().contains(&fused_id));
    assert_eq!(legacy.boundaries, fused.boundaries);
    assert_ne!(legacy.fingerprint().unwrap(), fused.fingerprint().unwrap());
    let original = family.semantic_program(&config, legacy).unwrap();
    let candidate = family.semantic_program(&config, fused).unwrap();
    assert_eq!(original.inputs(), candidate.inputs());
    assert_eq!(
        original.input_tensor_specs(),
        candidate.input_tensor_specs()
    );
    assert_eq!(original.weights(), candidate.weights());
    assert_eq!(candidate.outputs(), &[id(F::VELOCITY_ID)]);
    let mut changed = 0;
    for (left, right) in original.blocks()[0]
        .nodes
        .iter()
        .zip(&candidate.blocks()[0].nodes)
    {
        let mut restored = right.clone();
        if right.operation_id.as_str() == STRUCTURED_ATTENTION_FUSED_BF16_OPERATION_ID {
            assert_eq!(
                left.operation_id.as_str(),
                STRUCTURED_ATTENTION_BF16_OPERATION_ID
            );
            restored.operation_id = left.operation_id.clone();
            changed += 1;
        }
        assert_eq!(&restored, left);
    }
    assert_eq!(changed, config.transformer().layers());
    assert!(legacy.validate_program(&candidate).is_err());
    assert!(fused.validate_program(&original).is_err());
    let prepared = TypedFamilyRegistration::new(F::new().unwrap())
        .prepare_with_profile(&serde_json::to_value(config).unwrap(), &fused_id)
        .unwrap();
    assert_eq!(prepared.program(), &candidate);
}

fn transformer(layers: u64, causal_condition: bool, real: bool) -> QwenImage21Config {
    QwenImage21Config::from_hf_json(
        &serde_json::to_vec(&json!({
            "_class_name":"QwenImage21Transformer2DModel","num_layers":layers,
            "num_attention_heads":if real {32} else {2},"attention_head_dim":if real {128} else {8},
            "context_in_dim":if real {4096} else {8},"in_channels":if real {64} else {3},
            "out_channels":if real {64} else {4},"patch_size":1,"mlp_ratio":3,
            "axes_dims_rope":if real {vec![16,56,56]} else {vec![2,2,4]},
            "causal_condition":causal_condition,"eps":0.000001,
        }))
        .unwrap(),
        StructureLimits::default(),
    )
    .unwrap()
}
pub(super) fn config(causal: bool) -> QwenImage21TransformerConfig {
    QwenImage21TransformerConfig::new(
        transformer(2, causal, false),
        QwenImage21T2iLayout::new(3, 2, 4, vec![3, 1]).unwrap(),
    )
    .unwrap()
}
fn id(name: &str) -> ProgramValueId {
    ProgramValueId::new(name).unwrap()
}
fn attr<'a>(attrs: &'a BTreeMap<AttributeId, SemanticValue>, name: &str) -> &'a SemanticValue {
    &attrs[&AttributeId::new(name).unwrap()]
}

#[test]
fn homogeneous_geometry_keeps_padded_text_positions_target_grid_and_per_sample_key_masks() {
    let config = config(true);
    let layout = config.layout();
    assert_eq!(layout.batch(), 2);
    assert_eq!(layout.target_tokens(), 8);
    assert_eq!(layout.sequence_length(), 11);
    assert_eq!(layout.rotary_position(0), Some([0, 0, 0]));
    assert_eq!(layout.rotary_position(2), Some([2, 2, 2]));
    assert_eq!(layout.rotary_position(3), Some([3, -1, -2]));
    assert_eq!(layout.rotary_position(6), Some([3, -1, 1]));
    assert_eq!(layout.rotary_position(10), Some([3, 0, 1]));
    assert_eq!(layout.rotary_position(11), None);
    let attention = config.block_config().unwrap();
    let tensor = ResolvedTensorSpec::new(
        vec![22, 16],
        ElementType::Bf16,
        ResolvedTensorLayout::Contiguous,
    )
    .unwrap();
    let geometry = StructuredAttentionBf16Geometry::new(
        &tensor,
        &tensor,
        &tensor,
        &tensor,
        &attention.attention().attributes().unwrap(),
    )
    .unwrap();
    assert_eq!(geometry.mask().visible_end(0), Some(1));
    assert_eq!(geometry.mask().visible_end(2), Some(3));
    assert_eq!(geometry.mask().visible_end(3), Some(11));
    assert_eq!(geometry.mask().key_is_valid(0, 2), Some(true));
    assert_eq!(geometry.mask().key_is_valid(1, 1), Some(false));
    assert_eq!(geometry.mask().key_is_valid(1, 3), Some(true));
    let specs = config.input_tensor_specs().unwrap();
    assert_eq!(specs[&id(F::LATENTS_ID)].dimensions, vec![16, 3]);
    assert_eq!(specs[&id(F::TEXT_ID)].dimensions, vec![6, 8]);
    assert_eq!(specs[&id(F::TIME_FEATURES_ID)].dimensions, vec![3, 256]);
    assert_eq!(specs[&id(F::FREQUENCIES_ID)].dimensions, vec![11, 4, 2]);
    assert_eq!(specs[&id(F::FREQUENCIES_ID)].element_type, ElementType::F32);
}

#[test]
fn conditioning_transport_reads_one_shared_t0_row_and_each_targets_own_sample_row() {
    let checked = config(true);
    for block in 0..4 {
        let attrs = program::conditioning_attributes(&checked, block * 16).unwrap();
        assert_eq!(attr(&attrs, "batch"), &SemanticValue::Unsigned(2));
        assert_eq!(
            attr(&attrs, "column_start"),
            &SemanticValue::Unsigned(block * 16)
        );
        assert_eq!(attr(&attrs, "column_count"), &SemanticValue::Unsigned(16));
        assert_eq!(
            attr(&attrs, "run_lengths"),
            &SemanticValue::Integers(vec![3, 8])
        );
        assert_eq!(
            attr(&attrs, "source_starts"),
            &SemanticValue::Integers(vec![2, 0])
        );
        assert_eq!(
            attr(&attrs, "source_batch_strides"),
            &SemanticValue::Integers(vec![0, 1])
        );
        assert_eq!(
            attr(&attrs, "source_row_strides"),
            &SemanticValue::Integers(vec![0, 0])
        );
    }
    let noncausal = config(false);
    assert_eq!(noncausal.time_feature_rows(), 2);
    let attrs = program::conditioning_attributes(&noncausal, 0).unwrap();
    assert_eq!(
        attr(&attrs, "run_lengths"),
        &SemanticValue::Integers(vec![11])
    );
    assert_eq!(
        attr(&attrs, "source_starts"),
        &SemanticValue::Integers(vec![0])
    );
    assert_eq!(
        attr(&attrs, "source_batch_strides"),
        &SemanticValue::Integers(vec![1])
    );
}

#[test]
fn full_graph_reuses_two_distinct_layers_and_only_retains_target_velocity() {
    let family = F::new().unwrap();
    let config = config(true);
    let profiles = family.numerical_profiles(&config).unwrap();
    for profile in profiles.profiles() {
        let graph = family.semantic_program(&config, profile).unwrap();
        assert_eq!(graph.inputs(), &F::INPUT_IDS.map(id));
        assert_eq!(
            graph.input_tensor_specs(),
            &config.input_tensor_specs().unwrap()
        );
        assert_eq!(graph.outputs(), &[id(F::VELOCITY_ID)]);
        assert!(graph.states().is_empty());
        assert_eq!(graph.weights().len(), 27);
        let nodes = &graph.blocks()[0].nodes;
        assert_eq!(profile.boundaries.len(), nodes.len());
        assert_shared_gates(&graph, config.transformer().layers());
        let target = nodes
            .iter()
            .find(|node| node.outputs == [id(program::TARGET_HIDDEN)])
            .unwrap();
        assert_eq!(
            attr(&target.attributes, "source_starts"),
            &SemanticValue::Integers(vec![3])
        );
        assert_eq!(
            attr(&target.attributes, "source_batch_strides"),
            &SemanticValue::Integers(vec![11])
        );
        assert_eq!(
            attr(&target.attributes, "source_row_strides"),
            &SemanticValue::Integers(vec![1])
        );
        let block = Block::new().unwrap();
        let block_config = config.block_config().unwrap();
        let sources = block.numerical_profiles(&block_config).unwrap();
        let source = block
            .semantic_program(&block_config, &sources.profiles()[0])
            .unwrap();
        let mut previous = id(program::JOINT);
        for layer in 0..config.transformer().layers() {
            let prefix = format!("node.qwen_image21.transformer.layer{layer}.");
            let layer_nodes = nodes
                .iter()
                .filter(|node| node.id.as_str().starts_with(&prefix))
                .collect::<Vec<_>>();
            assert_eq!(layer_nodes.first().unwrap().inputs[0], previous);
            previous = layer_nodes.last().unwrap().outputs[0].clone();
            let unchanged = source.blocks()[0]
                .nodes
                .iter()
                .filter(|node| remap::shared_gate_destination(node).unwrap().is_none())
                .collect::<Vec<_>>();
            assert_eq!(layer_nodes.len(), unchanged.len());
            for (actual, original) in layer_nodes.into_iter().zip(unchanged) {
                if is_fp8(profile.id.as_str())
                    && original.operation_id.as_str() == DENSE_LINEAR_BF16_OPERATION_ID
                {
                    assert_eq!(
                        actual.operation_id.as_str(),
                        DENSE_LINEAR_FP8_BF16_OPERATION_ID
                    );
                } else {
                    assert_eq!(actual.operation_id, original.operation_id);
                }
                assert_eq!(actual.attributes, original.attributes);
                assert_ne!(actual.id, original.id);
                assert!(actual
                    .inputs
                    .iter()
                    .all(|value| !value.as_str().contains("block0")));
            }
        }
        assert_eq!(target.inputs, [previous]);
        let referenced = nodes
            .iter()
            .flat_map(|node| &node.inputs)
            .collect::<BTreeSet<_>>();
        assert!(graph
            .weights()
            .iter()
            .all(|weight| referenced.contains(&weight.value_id)));
        for operation in &profile.operations {
            if [
                ROW_GATHER_BF16_OPERATION_ID,
                BATCHED_ROW_CONCAT_BF16_OPERATION_ID,
            ]
            .contains(&operation.operation_id.as_str())
            {
                assert_eq!(operation.multiplication_type, None);
                assert_eq!(operation.accumulation_type, None);
            }
        }
    }
}

fn assert_shared_gates(graph: &ModelProgram, layers: u64) {
    let nodes = &graph.blocks()[0].nodes;
    let gates = nodes
        .iter()
        .enumerate()
        .filter(|(_, node)| node.operation_id.as_str() == TANH_BF16_OPERATION_ID)
        .collect::<Vec<_>>();
    assert_eq!(gates.len(), program::TANH_GATES.len());
    for (gate_index, (producer_index, producer)) in gates.into_iter().enumerate() {
        assert_eq!(
            producer.inputs,
            [id(program::CONDITIONING[1 + 2 * gate_index])]
        );
        assert_eq!(producer.outputs, [id(program::TANH_GATES[gate_index])]);
        assert!(producer.attributes.is_empty());
        let consumers = nodes
            .iter()
            .enumerate()
            .filter(|(_, node)| node.inputs.contains(&producer.outputs[0]))
            .collect::<Vec<_>>();
        assert_eq!(consumers.len() as u64, layers);
        for (consumer_index, consumer) in consumers {
            assert!(consumer_index > producer_index);
            assert_eq!(
                consumer.operation_id.as_str(),
                GATED_RESIDUAL_BF16_OPERATION_ID
            );
            assert_eq!(consumer.inputs[1], producer.outputs[0]);
        }
    }
}

#[test]
fn shared_gate_remapping_rejects_changed_arithmetic_or_input() {
    let family = Block::new().unwrap();
    let config = config(true).block_config().unwrap();
    let profiles = family.numerical_profiles(&config).unwrap();
    let source = family
        .semantic_program(&config, &profiles.profiles()[0])
        .unwrap();
    let gate = source.blocks()[0]
        .nodes
        .iter()
        .find(|node| node.outputs == [id(Block::TANH_GATE1_ID)])
        .unwrap();
    assert_eq!(
        remap::shared_gate_destination(gate).unwrap(),
        Some(program::TANH_GATES[0])
    );
    for mutation in 0..5 {
        let mut changed = gate.clone();
        match mutation {
            0 => changed.operation_id = OperationId::new(SILU_BF16_OPERATION_ID).unwrap(),
            1 => changed.inputs[0] = id(Block::GATE2_ID),
            2 => changed.required_version = ContractVersion::new(2, 0),
            3 => changed.outputs.push(id("value.unexpected.gate")),
            _ => {
                changed.attributes.insert(
                    AttributeId::new("unexpected").unwrap(),
                    SemanticValue::Unsigned(1),
                );
            }
        }
        assert!(remap::shared_gate_destination(&changed).is_err());
    }
}

#[test]
fn real_full_schema_keeps_all_297_tensor_components_without_diagnostic_completion_outputs() {
    let config = QwenImage21TransformerConfig::new(
        transformer(32, true, true),
        QwenImage21T2iLayout::new(2, 2, 2, vec![2]).unwrap(),
    )
    .unwrap();
    let family = F::new().unwrap();
    let ModelWeights::Stored(schema) = family.weights(&config).unwrap() else {
        panic!("Stored required")
    };
    assert_eq!(schema.components.len(), 297);
    assert_eq!(schema.tensors.len(), 297);
    assert_eq!(
        schema
            .components
            .iter()
            .map(|c| c.physical_bytes().unwrap())
            .sum::<u64>(),
        13 * 1024 * 1024 * 1024 + 259 * 1024 * 1024 + 24 * 1024
    );
    let profiles = family.numerical_profiles(&config).unwrap();
    let graph = family
        .semantic_program(&config, &profiles.profiles()[0])
        .unwrap();
    assert_shared_gates(&graph, config.transformer().layers());
    assert_eq!(graph.outputs(), &[id(F::VELOCITY_ID)]);
}

#[test]
fn geometry_and_conditioning_policy_are_checked_canonical_identity() {
    for (text, height, width, lengths) in [
        (0, 2, 2, vec![0]),
        (8192, 2, 2, vec![1]),
        (3, 0, 2, vec![1]),
        (3, 3, 2, vec![1]),
        (3, 2, 2050, vec![1]),
        (3, 2, 2, vec![]),
        (3, 2, 2, vec![4]),
        (3, 2, 2, vec![1; 17]),
    ] {
        assert!(QwenImage21T2iLayout::new(text, height, width, lengths).is_err());
    }
    let config = config(true);
    let encoded = serde_json::to_value(&config).unwrap();
    assert_eq!(
        serde_json::from_value::<QwenImage21TransformerConfig>(encoded.clone()).unwrap(),
        config
    );
    let mut changed = encoded.clone();
    changed["layout"]["frames"] = json!(2);
    assert!(serde_json::from_value::<QwenImage21TransformerConfig>(changed).is_err());
    let registration = TypedFamilyRegistration::new(F::new().unwrap());
    let prepared = registration
        .prepare_with_profile(&encoded, &profile_id(F::REFERENCE_PROFILE_ID).unwrap())
        .unwrap();
    let mut changed = encoded;
    changed["layout"]["valid_text_lengths"] = json!([3, 2]);
    let other = registration
        .prepare_with_profile(&changed, &profile_id(F::REFERENCE_PROFILE_ID).unwrap())
        .unwrap();
    assert_ne!(
        prepared.fingerprint().unwrap(),
        other.fingerprint().unwrap()
    );
    assert_eq!(prepared.weights(), other.weights());
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct TensorPolicy {
    capacity: u64,
    profiles: Vec<DynamicStorageProfile>,
}
impl RuntimePolicy for TensorPolicy {
    fn version(&self) -> ContractVersion {
        ContractVersion::new(1, 0)
    }
    fn memory_capacity_bytes(&self) -> u64 {
        self.capacity
    }
    fn memory_reserve_bytes(&self) -> u64 {
        0
    }
    fn maximum_active_sequences(&self) -> u32 {
        1
    }
    fn work_limits(&self) -> RuntimeWorkLimits {
        RuntimeWorkLimits::FixedTensor
    }
    fn attention_execution_policy(&self) -> ferrum_types::AttentionExecutionPolicy {
        ferrum_types::AttentionExecutionPolicy::Portable
    }
    fn execution_determinism_requirement(&self) -> ExecutionDeterminismRequirement {
        ExecutionDeterminismRequirement::BitwiseSameRuntime
    }
    fn dynamic_storage_profile_order(&self) -> &[DynamicStorageProfile] {
        &self.profiles
    }
    fn reusable_execution_policy(&self) -> Option<&ReusableExecutionPolicy> {
        None
    }
    fn validate(&self) -> Result<(), VNextError> {
        if self.capacity == 0 || self.profiles.is_empty() {
            return Err(invalid("policy", "missing fixed-tensor capacity/storage"));
        }
        Ok(())
    }
}

#[test]
fn typed_compile_options_supply_transport_shapes_and_retain_only_velocity() {
    use ferrum_kernels::backend::reference::ReferenceVNextComposition;
    let config = config(true);
    let family = TypedFamilyRegistration::new(F::new().unwrap())
        .prepare_with_profile(
            &serde_json::to_value(&config).unwrap(),
            &profile_id(F::REFERENCE_PROFILE_ID).unwrap(),
        )
        .unwrap();
    let composition = ReferenceVNextComposition::create(
        DeviceId::new("device.reference.transformer.compile").unwrap(),
    )
    .unwrap();
    let descriptor = composition.runtime().descriptor();
    let policy = TensorPolicy {
        capacity: descriptor.total_memory_bytes,
        profiles: descriptor
            .dynamic_storage_profiles
            .iter()
            .copied()
            .collect(),
    };
    let options = config.compile_options().unwrap();
    assert_eq!(options.tensor_specs().len(), 11);
    assert_eq!(
        options.tensor_specs()[&id(program::JOINT)].dimensions,
        vec![22, 16]
    );
    for value in program::CONDITIONING {
        assert_eq!(options.tensor_specs()[&id(value)].dimensions, vec![22, 16]);
    }
    for value in [program::TARGET_HIDDEN, program::FINAL_SCALE] {
        assert_eq!(options.tensor_specs()[&id(value)].dimensions, vec![16, 16]);
    }
    let compile = |options: &ProgramPlanCompileOptions| {
        ProgramPlanCompiler::compile_with_weight_materializers(
            &family,
            composition.catalog(),
            &policy,
            &composition.registry().planning(),
            composition.weight_materializers(),
            options,
        )
    };
    let compiled = compile(&options).unwrap();
    let plan = compiled.executable().execution_plan();
    let ports = plan.payload().tensor_ports().unwrap();
    assert_eq!(ports.inputs().len(), 4);
    assert_eq!(ports.outputs().len(), 1);
    assert_eq!(ports.outputs()[0].value_id(), &id(F::VELOCITY_ID));
    assert_eq!(ports.outputs()[0].tensor().dimensions(), [16, 4]);
    let gate_outputs = plan
        .payload()
        .nodes()
        .iter()
        .filter(|node| node.operation_id().as_str() == TANH_BF16_OPERATION_ID)
        .flat_map(|node| node.values())
        .filter(|value| value.role() == ResolvedValueRole::Output)
        .collect::<Vec<_>>();
    assert_eq!(gate_outputs.len(), program::TANH_GATES.len());
    for (gate, name) in gate_outputs.iter().zip(program::TANH_GATES) {
        assert_eq!(gate.value_id(), &id(name));
        assert_eq!(gate.tensor().dimensions(), [22, 16]);
    }
    assert_eq!(composition.runtime().snapshot().live_allocations, 0);
    // A wrong caller override is rechecked against the row-map, not trusted.
    let mut wrong = options.clone();
    wrong
        .insert_tensor_spec(
            id(program::FINAL_SCALE),
            ProgramTensorSpec {
                dimensions: vec![15, 16],
                element_type: ElementType::Bf16,
                layout: ResolvedTensorLayout::Contiguous,
            },
        )
        .unwrap();
    assert!(compile(&wrong).is_err());
    // Supplying only inputs cannot silently invent a different transport ABI.
    assert!(compile(
        &ProgramPlanCompileOptions::new(config.input_tensor_specs().unwrap()).unwrap()
    )
    .is_err());
}
