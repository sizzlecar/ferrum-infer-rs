use super::*;
use crate::fast_h3::*;
use ferrum_runtime::*;
use std::sync::Arc;
#[cfg(feature = "cuda-cutlass-nvfp4")]
mod cuda;
mod dyadic;

fn output_limits() -> FastH3AttentionOutputLimits {
    FastH3AttentionOutputLimits {
        maximum_rows: 129,
        maximum_metadata_bytes: 16384,
        maximum_stored_bytes: 8 << 20,
    }
}
fn host(bytes: usize) -> HostMemoryAccount {
    HostMemoryAccount::new(
        HostMemoryLimits::new(
            bytes,
            HostMemoryClassLimits {
                input_preprocess_bytes: 0,
                output_bytes: 0,
                transfer_staging_bytes: 0,
                weight_backing_bytes: bytes,
            },
        )
        .unwrap(),
    )
}
fn prepare(
    account: &HostMemoryAccount,
) -> (
    tempfile::TempDir,
    FastH3AttentionOutputConfig,
    FastH3AttentionOutputSource,
) {
    let (dir, package) = Fixture::new().open(limits());
    let (config, source) = FastH3AttentionOutputSource::prepare(
        Arc::new(package.unwrap()),
        0,
        3,
        account,
        output_limits(),
    )
    .unwrap();
    (dir, config, source)
}
#[test]
fn attention_output_source_preserves_packed_views_and_uses_authoritative_alpha() {
    let account = host(8);
    let (_dir, c, source) = prepare(&account);
    assert_eq!(account.usage().unwrap().allocated_bytes(), 8);
    assert_eq!(c.scalar_derivation().stored_bits(), 0x3e99_999a);
    assert_eq!(c.scalar_derivation().result_bits().unwrap(), 0x3e99_999a);
    let ModelWeights::Stored(schema) = FastH3AttentionOutputFamily::new(output_limits())
        .unwrap()
        .weights(&c)
        .unwrap()
    else {
        panic!("Stored")
    };
    assert_eq!(schema.components.len(), 4);
    let mut retained = None;
    for component in &schema.components {
        let payload = source.component(component).unwrap();
        assert_eq!(
            payload.bytes().len() as u64,
            component.physical_bytes().unwrap()
        );
        assert!(payload.retained_host_memory().is_some());
        if component.role == WeightComponentRole::Values {
            let bits = if component.id.as_str().contains("unit_f32") {
                1.0f32.to_bits()
            } else {
                0x3e99_999a
            };
            assert_eq!(payload.bytes(), bits.to_le_bytes());
            retained = Some(payload.retained_host_memory().unwrap().clone());
        } else {
            let tensor = schema.tensors.iter().find(|t| matches!(&t.physical_layout, PhysicalWeightLayout::PhysicalComponent(v) if v.component_id()==&component.id)).unwrap();
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
fn attention_output_graph_requires_raw_gate_then_fma_quantize_and_original_to_out() {
    let (_dir, c, _source) = prepare(&host(8));
    assert_eq!(
        (c.rows(), c.attention_width().unwrap(), c.hidden()),
        (3, 48, 32)
    );
    assert_eq!(c.merge(), FastH3AttentionMerge::ActiveCompressionFma);
    let prepared =
        TypedFamilyRegistration::new(FastH3AttentionOutputFamily::new(output_limits()).unwrap())
            .prepare_with_profile(
                &serde_json::to_value(&c).unwrap(),
                &NumericalProfileId::new(FastH3AttentionOutputFamily::PROFILE_ID).unwrap(),
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
            ADDCMUL_FMA_BF16_OPERATION_ID,
            NVFP4_QUANTIZE_BF16_OPERATION_ID,
            NVFP4_MATMUL_BF16_OPERATION_ID
        ]
    );
    assert_eq!(
        nodes[0]
            .inputs
            .iter()
            .map(|v| v.as_str())
            .collect::<Vec<_>>(),
        [
            FastH3AttentionOutputFamily::FINE_ID,
            FastH3AttentionOutputFamily::POOLED_ID,
            FastH3AttentionOutputFamily::GATE_ID
        ]
    );
    assert_eq!(nodes[1].inputs[0], nodes[0].outputs[0]);
    assert_eq!(nodes[2].inputs[0], nodes[1].outputs[0]);
    assert_eq!(nodes[2].inputs[2], nodes[1].outputs[1]);
    assert_eq!(
        nodes[1].attributes,
        Nvfp4QuantizeProfile::CudaRcpApproxFtzV1
            .attributes()
            .unwrap()
    );
    let options = c.compile_options().unwrap();
    assert_eq!(
        options.tensor_specs()[&nodes[2].outputs[0]].dimensions,
        [3, 32]
    );
    assert_eq!(
        options.tensor_specs()[&nodes[2].outputs[0]].element_type,
        ElementType::Bf16
    );
    let ModelWeights::Stored(schema) = prepared.weights() else {
        panic!("Stored")
    };
    super::ffn::validate_graph_bindings(graph, schema, &options);
    assert!(schema
        .components
        .iter()
        .filter(|c| c.role != WeightComponentRole::Values)
        .all(|c| c.external_names[0].contains("attn.to_out::")));
}
#[test]
fn attention_output_rejects_source_mixing_and_rolls_back_capacity() {
    let (dir, package) = Fixture::new().open(limits());
    let package = Arc::new(package.unwrap());
    let account = host(7);
    assert!(
        FastH3AttentionOutputSource::prepare(package.clone(), 0, 3, &account, output_limits())
            .is_err()
    );
    assert_eq!(account.usage().unwrap().allocated_bytes(), 0);
    for (block, rows, bound) in [
        (1, 3, output_limits()),
        (0, 130, output_limits()),
        (
            0,
            3,
            FastH3AttentionOutputLimits {
                maximum_stored_bytes: 1,
                ..output_limits()
            },
        ),
    ] {
        assert!(FastH3AttentionOutputSource::prepare(
            package.clone(),
            block,
            rows,
            &host(8),
            bound
        )
        .is_err());
    }
    drop(dir);
    let (_dir, c, _source) = prepare(&host(8));
    let family = FastH3AttentionOutputFamily::new(output_limits()).unwrap();
    for (key, replacement) in [
        ("block", serde_json::json!(1)),
        ("merge", serde_json::json!("clamped_gate")),
        ("hidden", serde_json::json!(48)),
    ] {
        let mut wire = serde_json::to_value(&c).unwrap();
        wire[key] = replacement;
        assert!(family.parse_config(&wire).is_err());
    }
    let mut wire = serde_json::to_value(&c).unwrap();
    wire["alpha"]["arithmetic_profile"] = serde_json::json!("recomputed_bf16_reciprocal");
    assert!(family.parse_config(&wire).is_err());
}
