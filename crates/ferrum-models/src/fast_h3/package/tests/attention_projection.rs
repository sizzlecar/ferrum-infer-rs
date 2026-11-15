use super::*;
use crate::fast_h3::*;
use ferrum_runtime::*;
use std::sync::Arc;

#[cfg(feature = "cuda-cutlass-nvfp4")]
mod cuda;
pub(in crate::fast_h3::package::tests) mod dyadic;

fn projection_limits() -> FastH3AttentionProjectionLimits {
    FastH3AttentionProjectionLimits {
        maximum_rows: 129,
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
    f: &Fixture,
    account: &HostMemoryAccount,
) -> (
    tempfile::TempDir,
    FastH3AttentionProjectionConfig,
    FastH3AttentionProjectionSource,
) {
    let (dir, package) = f.open(limits());
    let (c, s) = FastH3AttentionProjectionSource::prepare(
        Arc::new(package.unwrap()),
        0,
        3,
        account,
        projection_limits(),
    )
    .unwrap();
    (dir, c, s)
}

#[test]
fn attention_source_keeps_original_packed_identity_and_authoritative_unit_alpha() {
    let account = host(20);
    let (_dir, c, source) = source(&Fixture::new(), &account);
    let family = FastH3AttentionProjectionFamily::new(projection_limits()).unwrap();
    let ModelWeights::Stored(schema) = family.weights(&c).unwrap() else {
        panic!("Stored projections")
    };
    assert_eq!(account.usage().unwrap().allocated_bytes(), 20);
    assert_eq!(schema.components.len(), 13);
    for recipe in c.scalar_derivations() {
        assert_eq!(recipe.stored_bits(), 0x3e99_999a);
        assert_eq!(recipe.result_bits().unwrap(), recipe.stored_bits());
        assert_ne!(recipe.result_bits().unwrap(), (1.0f32 / 3.328125).to_bits());
        assert_eq!(recipe.source_file(), "nvfp4_weights.safetensors");
        assert_eq!(recipe.source_alpha().role, WeightComponentRole::Metadata);
    }
    let mut retained = None;
    for component in &schema.components {
        let payload = source.component(component).unwrap();
        assert_eq!(
            payload.bytes().len() as u64,
            component.physical_bytes().unwrap()
        );
        assert!(payload.retained_host_memory().is_some());
        if component.role == WeightComponentRole::Values {
            let unit = component.id.as_str().contains("unit_f32");
            let expected = if unit { 1.0f32.to_bits() } else { 0x3e99_999a };
            assert_eq!(payload.bytes(), expected.to_le_bytes());
            if !unit {
                retained = Some(payload.retained_host_memory().unwrap().clone());
            }
        } else {
            let tensor=schema.tensors.iter().find(|t|matches!(&t.physical_layout,PhysicalWeightLayout::PhysicalComponent(v) if v.component_id()==&component.id)).unwrap();
            let binding = ResolvedWeightBinding::from_schema(&schema, &tensor.id).unwrap();
            assert_eq!(
                binding.physical_component_view().unwrap().encoding(),
                &component.encoding
            );
            assert!(tensor.logical_elements().is_err());
        }
        let mut changed = component.clone();
        changed.external_names = vec!["other.weight".into()];
        assert!(source.component(&changed).is_err());
    }
    drop(source);
    assert_eq!(account.usage().unwrap().allocated_bytes(), 4);
    drop(retained);
    assert_eq!(account.usage().unwrap().allocated_bytes(), 0);
}

#[test]
fn attention_graph_shares_one_quantization_for_four_non_square_raw_projections() {
    let (_dir, c, _source) = source(&Fixture::new(), &host(20));
    assert_eq!(
        (
            c.hidden(),
            c.heads(),
            c.head_dim(),
            c.output_width().unwrap()
        ),
        (32, 3, 16, 48)
    );
    let family = FastH3AttentionProjectionFamily::new(projection_limits()).unwrap();
    let prepared = TypedFamilyRegistration::new(family)
        .prepare_with_profile(
            &serde_json::to_value(&c).unwrap(),
            &NumericalProfileId::new(FastH3AttentionProjectionFamily::PROFILE_ID).unwrap(),
        )
        .unwrap();
    let graph = prepared.program();
    let nodes = &graph.blocks()[0].nodes;
    assert_eq!(nodes.len(), 5);
    assert_eq!(
        nodes[0].operation_id.as_str(),
        NVFP4_QUANTIZE_BF16_OPERATION_ID
    );
    assert_eq!(
        nodes[0].attributes,
        Nvfp4QuantizeProfile::CudaRcpApproxFtzV1
            .attributes()
            .unwrap()
    );
    let options = c.compile_options().unwrap();
    for (node, kind) in nodes[1..].iter().zip(FastH3AttentionProjection::ALL) {
        assert_eq!(node.operation_id.as_str(), NVFP4_MATMUL_BF16_OPERATION_ID);
        assert_eq!(node.inputs[0], nodes[0].outputs[0]);
        assert_eq!(node.inputs[2], nodes[0].outputs[1]);
        assert_eq!(node.outputs[0].as_str(), kind.output_id());
        assert_eq!(options.tensor_specs()[&node.outputs[0]].dimensions, [3, 48]);
        assert_eq!(
            options.tensor_specs()[&node.outputs[0]].element_type,
            ElementType::Bf16
        );
    }
    assert_eq!(graph.outputs().len(), 4);
    let ModelWeights::Stored(schema) = prepared.weights() else {
        panic!("Stored")
    };
    super::ffn::validate_graph_bindings(graph, schema, &options);
    // Unit SF is a real numerical constant; only B and scaleB are raw views.
    let global = &nodes[0].inputs[1];
    let w = graph
        .weights()
        .iter()
        .find(|w| &w.value_id == global)
        .unwrap();
    let binding = ResolvedWeightBinding::from_schema(schema, &w.weight_id).unwrap();
    assert!(binding.physical_component_view().is_none());
    assert_eq!(binding.components()[0].role(), WeightComponentRole::Values);
}

#[test]
fn attention_source_rejects_qad_and_tampered_recipes_and_rolls_back_true_capacity() {
    let fixture = Fixture::new();
    let (_dir, package) = fixture.open(limits());
    let package = Arc::new(package.unwrap());
    let limited = host(19);
    let error = FastH3AttentionProjectionSource::prepare(
        package.clone(),
        0,
        3,
        &limited,
        projection_limits(),
    )
    .err()
    .unwrap();
    assert!(matches!(
        error.downcast_ref::<HostMemoryError>(),
        Some(HostMemoryError::Capacity { .. })
    ));
    assert_eq!(limited.usage().unwrap().committed_bytes(), 0);
    assert!(FastH3AttentionProjectionSource::prepare(
        package.clone(),
        1,
        3,
        &limited,
        projection_limits()
    )
    .is_err());
    assert!(FastH3AttentionProjectionSource::prepare(
        package,
        0,
        130,
        &limited,
        projection_limits()
    )
    .is_err());

    let mut fixture = Fixture::new();
    fixture.packed.insert(
        "minimax_h3.transformer_blocks.0.attn.to_q::_nvfp4_input_global_sf".into(),
        (Dtype::F32, vec![], 8.0f32.to_le_bytes().to_vec()),
    );
    let (_dir, package) = fixture.open(limits());
    assert!(FastH3AttentionProjectionSource::prepare(
        Arc::new(package.unwrap()),
        0,
        3,
        &limited,
        projection_limits()
    )
    .is_err());
    assert_eq!(limited.usage().unwrap().committed_bytes(), 0);

    let (_dir, c, source) = source(&Fixture::new(), &host(20));
    let family = FastH3AttentionProjectionFamily::new(projection_limits()).unwrap();
    let mut raw = serde_json::to_value(&c).unwrap();
    raw["block"] = 1.into();
    assert!(family.parse_config(&raw).is_err());
    let mut raw = serde_json::to_value(&c).unwrap();
    raw["projections"][0]["alpha"]["stored_bits"] = 0u32.into();
    assert!(family.parse_config(&raw).is_err());
    // A different otherwise-valid alpha recipe gets a distinct source ID, so
    // the original source cannot satisfy a wire-mutated component declaration.
    for (key, value) in [
        ("stored_bits", serde_json::json!(0.5f32.to_bits())),
        ("source_file", serde_json::json!("other.safetensors")),
    ] {
        let mut raw = serde_json::to_value(&c).unwrap();
        raw["projections"][0]["alpha"][key] = value;
        let changed = family.parse_config(&raw).unwrap();
        let ModelWeights::Stored(schema) = family.weights(&changed).unwrap() else {
            panic!("Stored")
        };
        assert!(schema
            .components
            .iter()
            .any(|component| source.component(component).is_err()));
    }
}

#[test]
fn typed_projection_groups_preserve_selected_originals_and_choose_real_primary_output() {
    let (_dir, p) = Fixture::new().open(limits());
    let p = Arc::new(p.unwrap());
    for group in [
        FastH3AttentionProjectionGroup::Qkv,
        FastH3AttentionProjectionGroup::Gate,
    ] {
        let account = host(20);
        let (c, source) = FastH3AttentionProjectionSource::prepare_group(
            p.clone(),
            0,
            3,
            group,
            &account,
            projection_limits(),
        )
        .unwrap();
        assert_eq!(c.group(), group);
        assert_eq!(
            account.usage().unwrap().allocated_bytes(),
            4 * (group.projections().len() + 1)
        );
        let prepared = TypedFamilyRegistration::new(
            FastH3AttentionProjectionFamily::new(projection_limits()).unwrap(),
        )
        .prepare_with_profile(
            &serde_json::to_value(&c).unwrap(),
            &NumericalProfileId::new(FastH3AttentionProjectionFamily::PROFILE_ID).unwrap(),
        )
        .unwrap();
        assert_eq!(
            prepared
                .program()
                .outputs()
                .iter()
                .map(|v| v.as_str())
                .collect::<Vec<_>>(),
            group
                .projections()
                .iter()
                .map(|v| v.output_id())
                .collect::<Vec<_>>()
        );
        let ModelWeights::Stored(schema) = prepared.weights() else {
            panic!("Stored")
        };
        super::ffn::validate_graph_bindings(
            prepared.program(),
            schema,
            &c.compile_options().unwrap(),
        );
        for component in &schema.components {
            source.component(component).unwrap();
        }
        drop(source);
        assert_eq!(account.usage().unwrap().committed_bytes(), 0);
    }
}
