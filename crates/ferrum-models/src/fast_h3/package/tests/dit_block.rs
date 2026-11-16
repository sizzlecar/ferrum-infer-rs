use super::*;
use crate::fast_h3::{vsa::*, *};
use ferrum_runtime::*;
use half::bf16;
use std::sync::Arc;
type F = FastH3DitBlockFamily;
mod batch;
#[cfg(all(feature = "cuda-cutlass-nvfp4", feature = "cuda-sage-vsa-fp4"))]
mod cuda;
fn id(s: &str) -> ProgramValueId {
    ProgramValueId::new(s).unwrap()
}
fn capacities() -> FastH3DitBlockLimits {
    FastH3DitBlockLimits {
        modulation: FastH3ModulationLimits {
            maximum_rows: 32768,
            maximum_time_rows: 64,
            maximum_metadata_bytes: 32768,
            maximum_stored_bytes: 4096,
        },
        attention_input: super::attention_input::capacities(),
        attention_core: super::attention_core::capacities(),
        ffn: FastH3FfnLimits {
            maximum_rows: 32768,
            maximum_plan_nodes: 128,
            maximum_metadata_bytes: 32768,
            maximum_stored_bytes: 1 << 20,
        },
        maximum_plan_nodes: 128,
        maximum_stored_bytes: 4 << 20,
        maximum_metadata_bytes: 2 << 20,
    }
}
fn fixture() -> (tempfile::TempDir, Arc<FastH3TransformerPackage>) {
    let c = config_with_attention(128, 2, 128, 128);
    let mut f = Fixture::with_attention(&c);
    for (name, (_, _, bytes)) in &mut f.dense {
        if name.ends_with("norm1.weight")
            || name.ends_with("norm2.weight")
            || name.ends_with("attn.norm_q.weight")
            || name.ends_with("attn.norm_k.weight")
        {
            for b in bytes.chunks_exact_mut(2) {
                b.copy_from_slice(&bf16::ONE.to_bits().to_le_bytes());
            }
        }
    }
    let (dir, p) = f.open_with_config(
        &c,
        FastH3WeightLimits {
            maximum_source_bytes: 8 << 20,
            ..limits()
        },
    );
    (dir, Arc::new(p.unwrap()))
}
fn schedule() -> FastH3ConditioningSchedule {
    let c = FastH3Checkpoint::parse(
        include_bytes!("../../fixtures/inference.json"),
        include_bytes!("../../fixtures/transformer.json"),
        include_bytes!("../../fixtures/video_scheduler.json"),
        include_bytes!("../../fixtures/audio_scheduler.json"),
        FastH3MetadataLimits {
            max_file_bytes: 4096,
            max_pdd_steps: 64,
        },
    )
    .unwrap();
    FastH3ConditioningSchedule::all_modalities(&FastH3PddPlan::new(&c).unwrap()).unwrap()
}
fn tags() -> Vec<FastH3TextTagRun> {
    vec![
        FastH3TextTagRun {
            rows: 1,
            tag: FastH3TextTokenTag::Vision,
        },
        FastH3TextTagRun {
            rows: 2,
            tag: FastH3TextTokenTag::Text,
        },
    ]
}
fn host() -> HostMemoryAccount {
    HostMemoryAccount::new(
        HostMemoryLimits::new(
            128 << 20,
            HostMemoryClassLimits {
                input_preprocess_bytes: 64 << 20,
                output_bytes: 32 << 20,
                transfer_staging_bytes: 16 << 20,
                weight_backing_bytes: 16 << 20,
            },
        )
        .unwrap(),
    )
}
#[test]
fn actual_dit_block_joins_original_sources_and_reuses_one_compact_table_through_final_fma() {
    let (_dir, p) = fixture();
    let layout = super::attention_core::layout();
    let schedule = schedule();
    let tags = tags();
    let account = host();
    let (c, source, regions) = FastH3DitBlockSource::prepare(
        p,
        0,
        &layout,
        &schedule,
        &[(1, &tags)],
        FastH3VsaSelection::Trained,
        FastH3PrefixAttentionArithmetic::VendorFused,
        &account,
        capacities(),
    )
    .unwrap();
    let prepared = TypedFamilyRegistration::new(F::new(capacities()).unwrap())
        .prepare_with_profile(
            &serde_json::to_value(&c).unwrap(),
            &NumericalProfileId::new(c.profile_id()).unwrap(),
        )
        .unwrap();
    let graph = prepared.program();
    let options = c.compile_options(capacities()).unwrap();
    assert_eq!(
        graph.inputs(),
        &[
            id(F::HIDDEN_ID),
            id(F::COSINE_ID),
            id(F::SINE_ID),
            id(F::TABLE_ID)
        ]
    );
    assert_eq!(
        options.tensor_specs()[&id(F::HIDDEN_ID)],
        options.tensor_specs()[&id(F::OUTPUT_ID)]
    );
    assert_eq!(c.batches(), 1);
    assert_eq!(c.rows().unwrap(), layout.packed().sequence_rows() as u64);
    let nodes = &graph.blocks()[0].nodes;
    let only = |name: &str| {
        let values = nodes
            .iter()
            .filter(|n| n.operation_id.as_str() == name)
            .collect::<Vec<_>>();
        assert_eq!(values.len(), 1, "{name}");
        values[0]
    };
    let norm1 = only(ROW_SELECTED_RMS_MODULATION_BF16_OPERATION_ID);
    let norm2 = only(ROW_SELECTED_RESIDUAL_RMS_MODULATION_BF16_OPERATION_ID);
    let final_fma = only(ROW_SELECTED_ADDCMUL_FMA_BF16_OPERATION_ID);
    assert_eq!(norm1.inputs[0], id(F::HIDDEN_ID));
    assert_eq!(norm2.inputs[0], id(F::HIDDEN_ID));
    assert_eq!(norm2.inputs[3], id(FastH3AttentionCoreFamily::OUTPUT_ID));
    for n in [norm1, norm2, final_fma] {
        assert_eq!(n.inputs[1], id(F::TABLE_ID));
        for name in ["run_lengths", "parameter_rows"] {
            assert_eq!(
                n.attributes[&AttributeId::new(name).unwrap()],
                norm1.attributes[&AttributeId::new(name).unwrap()]
            );
        }
    }
    assert_eq!(
        final_fma.attributes[&AttributeId::new("gate_column").unwrap()],
        SemanticValue::Unsigned(5 * c.hidden_size())
    );
    assert_eq!(final_fma.inputs[0], norm2.outputs[0]);
    assert_eq!(final_fma.inputs[2], id(FastH3FfnFamily::OUTPUT_ID));
    let ffn_quant = nodes
        .iter()
        .find(|n| {
            n.operation_id.as_str() == NVFP4_QUANTIZE_BF16_OPERATION_ID
                && n.inputs[0] == norm2.outputs[1]
        })
        .unwrap();
    assert_eq!(
        ffn_quant.attributes,
        Nvfp4QuantizeProfile::CudaRcpApproxFtzV1
            .attributes()
            .unwrap()
    );
    let ModelWeights::Stored(schema) = prepared.weights() else {
        panic!("Stored")
    };
    super::ffn::validate_graph_bindings(graph, schema, &options);
    for component in &schema.components {
        let data = source.component(component).unwrap();
        assert_eq!(
            data.bytes().len() as u64,
            component.physical_bytes().unwrap()
        );
        assert!(data.retained_host_memory().is_some());
        if component.id.as_str().contains("unit_f32") {
            assert_eq!(data.bytes(), 1.0f32.to_le_bytes());
        }
    }
    let raw = serde_json::to_value(&c).unwrap();
    let family = F::new(capacities()).unwrap();
    let mut wrong = raw.clone();
    wrong["norm2"]["routing"]["runs"][0]["time_row"] = serde_json::json!(0);
    if wrong != raw {
        assert!(family.parse_config(&wrong).is_err());
    }
    let mut wrong = raw;
    wrong["norm2"]["stage"] = serde_json::json!("attention");
    assert!(family.parse_config(&wrong).is_err());
    drop(source);
    drop(regions);
    assert_eq!(account.usage().unwrap().allocated_bytes(), 0);
}

#[test]
fn complete_checkpoint_stack_has_stable_role_inputs_and_original_physical_sources() {
    // The package-wide fixture intentionally keeps block1 dense. This consumer
    // requires the actual release's all-NVFP4 block inventory instead.
    let geometry = config_with_attention(128, 2, 128, 128);
    let mut fixture = Fixture::with_attention(&geometry);
    fixture.metadata.remove("skip_blocks");
    for (name, tensor) in fixture.packed.clone() {
        fixture.packed.insert(
            name.replace("transformer_blocks.0", "transformer_blocks.1"),
            tensor,
        );
    }
    for name in [
        "attn.to_q",
        "attn.to_k",
        "attn.to_v",
        "attn.to_out.0",
        "attn.to_gate_compress",
        "ff.net.0.proj",
        "ff.net.2",
    ] {
        fixture
            .dense
            .remove(&format!("transformer_blocks.1.{name}.weight"));
    }
    let (_dir, package) = fixture.open_with_config(
        &geometry,
        FastH3WeightLimits {
            maximum_tensors: 150,
            maximum_source_bytes: 8 << 20,
            ..limits()
        },
    );
    let package = Arc::new(package.unwrap());
    let layers = package.config().layers();
    let layout = super::attention_core::layout();
    let tags = tags();
    let account = host();
    let limits = FastH3DitBackboneLimits {
        block: capacities(),
        maximum_layers: layers,
        maximum_plan_nodes: layers * 128,
        maximum_stored_bytes: 16 << 20,
        maximum_metadata_bytes: 8 << 20,
    };
    let (c, source, regions) = FastH3DitBackboneSource::prepare(
        package.clone(),
        &layout,
        &schedule(),
        &[&tags],
        FastH3VsaSelection::Trained,
        FastH3PrefixAttentionArithmetic::VendorFused,
        &account,
        limits,
    )
    .unwrap();
    type Stack = FastH3DitBackboneFamily;
    let prepared = TypedFamilyRegistration::new(Stack::new(limits).unwrap())
        .prepare_with_profile(
            &serde_json::to_value(&c).unwrap(),
            &NumericalProfileId::new(c.profile_id()).unwrap(),
        )
        .unwrap();
    let ModelWeights::Stored(schema) = prepared.weights() else {
        panic!("Stored")
    };
    assert_eq!(c.layers(), layers);
    assert_eq!(prepared.program().inputs().len(), 3 + layers);
    // Reusing an existing validated composition must retain the exact mapping
    // produced by the generic family entrance, including physical source IDs.
    let first = &c.blocks()[0];
    let built = first.composition(limits.block).unwrap();
    let bindings = built
        .program
        .inputs()
        .iter()
        .map(|id| (id.clone(), id.clone()))
        .collect();
    let generic = crate::tensor_program_fragment::Fragment::family_instance(
        &FastH3DitBlockFamily::new(limits.block).unwrap(),
        first,
        first.profile_id(),
        built.options.tensor_specs(),
        bindings,
        Some("identity_check"),
    )
    .unwrap();
    let bindings = built
        .program
        .inputs()
        .iter()
        .map(|id| (id.clone(), id.clone()))
        .collect();
    let reused = crate::tensor_program_fragment::Fragment::composition(
        built,
        bindings,
        Some("identity_check"),
    )
    .unwrap();
    assert_eq!(generic.schema, reused.schema);
    assert_eq!(generic.nodes, reused.nodes);
    assert_eq!(generic.weights, reused.weights);
    assert_eq!(generic.specs, reused.specs);
    assert_eq!(generic.operations, reused.operations);
    assert_eq!(generic.boundaries, reused.boundaries);
    for (i, block) in c.blocks().iter().enumerate() {
        assert_eq!(block.block(), i);
        assert_eq!(
            block.modulation_config().conditioning_request_count(),
            Some(1)
        );
        assert_eq!(
            c.compile_options(limits).unwrap().tensor_specs()[&c.table_input_id(i).unwrap()]
                .dimensions,
            [5, 6 * c.hidden_size()]
        );
    }
    super::ffn::validate_graph_bindings(
        prepared.program(),
        schema,
        &c.compile_options(limits).unwrap(),
    );
    for component in &schema.components {
        assert_eq!(
            source.component(component).unwrap().bytes().len() as u64,
            component.physical_bytes().unwrap()
        );
    }
    let mut wrong = serde_json::to_value(&c).unwrap();
    wrong["blocks"][0]["norm2"]["routing"]["time_rows"][0][0] = serde_json::json!(0);
    assert!(Stack::new(limits).unwrap().parse_config(&wrong).is_err());
    drop(source);
    drop(regions);
    let denoising_limits = FastH3DenoisingLimits {
        input: FastH3InputLimits {
            maximum_rows: 32768,
            maximum_segments: 32,
            maximum_metadata_bytes: 1 << 20,
            maximum_stored_bytes: 16 << 20,
        },
        backbone: limits,
        maximum_plan_nodes: layers * 128 + 7,
        maximum_stored_bytes: 32 << 20,
        maximum_metadata_bytes: 16 << 20,
    };
    let (denoising, source, regions) = FastH3DenoisingSource::prepare(
        &package,
        &layout,
        &schedule(),
        &tags,
        FastH3VsaSelection::Trained,
        FastH3PrefixAttentionArithmetic::VendorFused,
        &account,
        denoising_limits,
    )
    .unwrap();
    let definition =
        TypedFamilyRegistration::new(FastH3DenoisingFamily::new(denoising_limits).unwrap())
            .prepare_with_profile(
                &serde_json::to_value(&denoising).unwrap(),
                &NumericalProfileId::new(denoising.profile_id()).unwrap(),
            )
            .unwrap();
    let ModelWeights::Stored(schema) = definition.weights() else {
        panic!("Stored")
    };
    let options = denoising.compile_options(denoising_limits).unwrap();
    super::ffn::validate_graph_bindings(definition.program(), schema, &options);
    assert_eq!(definition.program().inputs().len(), 5 + layers);
    assert_eq!(
        options.tensor_specs()[&id(FastH3DenoisingFamily::VIDEO_INPUT_ID)].element_type,
        ElementType::F32
    );
    assert_eq!(
        options.tensor_specs()[&id(FastH3DenoisingFamily::TEXT_INPUT_ID)].element_type,
        ElementType::Bf16
    );
    for component in &schema.components {
        assert_eq!(
            source.component(component).unwrap().bytes().len() as u64,
            component.physical_bytes().unwrap()
        );
    }
    drop(source);
    drop(regions);
    assert_eq!(account.usage().unwrap().allocated_bytes(), 0);
}
