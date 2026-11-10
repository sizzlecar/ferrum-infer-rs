use super::*;
use ferrum_execution::{
    component::{
        CompiledTensorComponent, ComponentInitializationConfig, ComponentWeightSource,
        TensorComponentPolicy,
    },
    TensorExecutorConfig,
};
use ferrum_kernels::backend::reference::ReferenceVNextComposition;
use ferrum_quantization::TensorArchive;
#[cfg(feature = "cuda")]
mod cuda;
pub(in crate::qwen3_vl_vision) mod fixture;
pub(in crate::qwen3_vl_vision) mod oracle;
use fixture::*;
#[test]
fn fused_f32_profile_changes_only_attention_contract_and_keeps_original_default() {
    let config = config(4, 6);
    let family = Qwen3VLVisionComfyFamily::new(limits()).unwrap();
    let profiles = family.numerical_profiles(&config).unwrap();
    let strict = profiles
        .resolve(&id(Qwen3VLVisionComfyFamily::CUDA_PROFILE_ID))
        .unwrap();
    let fused = profiles
        .resolve(&id(Qwen3VLVisionComfyFamily::CUDA_FUSED_F32_PROFILE_ID))
        .unwrap();
    assert_eq!(strict.boundaries, fused.boundaries);
    let strict_graph = family.semantic_program(&config, strict).unwrap();
    let fused_graph = family.semantic_program(&config, fused).unwrap();
    fused.validate_program(&fused_graph).unwrap();
    assert_eq!(strict_graph.inputs(), fused_graph.inputs());
    assert_eq!(strict_graph.outputs(), fused_graph.outputs());
    assert_eq!(strict_graph.weights(), fused_graph.weights());
    let mut expected = strict_graph.blocks().to_vec();
    let mut found_attention = false;
    for node in expected.iter_mut().flat_map(|block| &mut block.nodes) {
        if node.operation_id.as_str() == STRUCTURED_GQA_ATTENTION_F32_OPERATION_ID {
            found_attention = true;
            node.operation_id = id(STRUCTURED_GQA_ATTENTION_FUSED_F32_OPERATION_ID);
        }
    }
    assert!(found_attention);
    assert_eq!(expected, fused_graph.blocks());
    assert_ne!(
        strict_graph.fingerprint().unwrap(),
        fused_graph.fingerprint().unwrap()
    );
    assert_eq!(strict_graph, program::build(&config).unwrap().0);
}
fn compile(
    c: &Qwen3VLVisionComfyConfig,
    r: &ReferenceVNextComposition,
) -> CompiledTensorComponent<ferrum_kernels::backend::reference::ReferenceDeviceRuntime> {
    let p = TypedFamilyRegistration::new(Qwen3VLVisionComfyFamily::new(limits()).unwrap())
        .prepare_with_profile(
            &serde_json::to_value(c).unwrap(),
            &id(Qwen3VLVisionComfyFamily::REFERENCE_PROFILE_ID),
        )
        .unwrap();
    let d = r.runtime().descriptor();
    let policy = TensorComponentPolicy::new(
        d.total_memory_bytes,
        d.dynamic_storage_profiles.iter().copied().collect(),
    )
    .unwrap();
    CompiledTensorComponent::compile(
        p,
        r.registry(),
        r.catalog(),
        &policy,
        r.weight_materializers(),
        &c.compile_options().unwrap(),
    )
    .unwrap()
}
fn init(name: &str) -> ComponentInitializationConfig {
    ComponentInitializationConfig {
        run_id: id(format!("run.vision.{name}")),
        request_id: id(format!("request.vision.{name}")),
        transaction_id: id(format!("transaction.vision.{name}")),
        static_policy: StaticInitializationPolicy::new(1 << 20, 256).unwrap(),
        executor: TensorExecutorConfig::default(),
    }
}

/// Reuse the original tiny three-layer Stored fixture for consumers of the
/// four real visual ports. Native sessions/weights retire before returning;
/// the caller retains only independently accounted F32 readbacks.
pub(crate) fn reference_outputs(
    height: u64,
    width: u64,
    which: usize,
    host: &HostMemoryAccount,
) -> Vec<ferrum_execution::TensorExecutionOutput> {
    let c = config(height, width);
    let original = Original::new(&c);
    let (_directory, archive) = original.archive();
    let checkpoint = Qwen3VLVisionComfyCheckpoint::new(&c, &archive, limits()).unwrap();
    let r = ReferenceVNextComposition::create(id(format!(
        "device.reference.vision.comfy.bridge.{height}.{width}.{which}"
    )))
    .unwrap();
    let component = compile(&c, &r);
    let source = checkpoint.weight_source(host);
    let (mut session, weights) = component
        .initialize_shared_weights(
            r.runtime().clone(),
            host.clone(),
            ComponentWeightSource::Stored(&source),
            init("bridge"),
        )
        .unwrap();
    let positions = c
        .managed_position_inputs(
            host,
            &TensorArchive::tensor(&archive, Qwen3VLVisionFamily::POSITION_EXTERNAL_NAME).unwrap(),
        )
        .unwrap();
    let mut inputs = vec![input(host, &c, &patches(&c, which))];
    inputs.extend(positions);
    let outputs = session
        .execute(
            id("run.vision.comfy.bridge"),
            id("request.vision.comfy.bridge"),
            inputs,
        )
        .unwrap();
    assert!(matches!(
        session
            .close()
            .unwrap_or_else(|e| panic!("visual bridge close: {:?}", e.failure())),
        PlanRuntimeCloseOutcome::Closed(_)
    ));
    drop(weights);
    outputs
}
#[test]
fn all_original_weights_keep_rank_and_explicit_f32_graph_has_four_real_merger_outputs() {
    let c = config(4, 6);
    let (g, specs) = program::build(&c).unwrap();
    assert_eq!(g.outputs().len(), 4);
    for output in g.outputs() {
        assert_eq!(specs[output].dimensions, [6, 256]);
        assert_eq!(specs[output].element_type, ElementType::F32);
    }
    for n in g.blocks().iter().flat_map(|b| &b.nodes) {
        for out in &n.outputs {
            assert!(specs.contains_key(out));
        }
        assert!(!n.operation_id.as_str().ends_with(".bf16"));
    }
    let schema = program::schema(&c).unwrap();
    let patch = schema
        .components
        .iter()
        .find(|c| c.external_names == ["model.visual.patch_embed.proj.weight"])
        .unwrap();
    assert_eq!(patch.dimensions, [8, 3, 2, 2, 2]);
    assert_eq!(schema.components.len(), 62);
    let base = Qwen3VLVisionFamily::new()
        .unwrap()
        .weights(&c.base)
        .unwrap();
    assert!(base
        .stored_schema()
        .unwrap()
        .components
        .iter()
        .all(|w| w.encoding
            == WeightEncoding::Dense {
                element_type: ElementType::Bf16
            }));
}
#[test]
fn original_bf16_position_products_and_sums_differ_from_the_legacy_final_round_only_adapter() {
    let c = config(4, 6);
    let mut original = Original::new(&c);
    let table = original
        .values
        .get_mut(Qwen3VLVisionFamily::POSITION_EXTERNAL_NAME)
        .unwrap();
    for row in table.chunks_exact_mut(8) {
        row.fill(0.);
    }
    table[..8].fill(-2.);
    table[8..16].fill(1.);
    let (_dir, archive) = original.archive();
    let h = host(1 << 20);
    let p = TensorArchive::tensor(&archive, Qwen3VLVisionFamily::POSITION_EXTERNAL_NAME).unwrap();
    let inputs = c.managed_position_inputs(&h, &p).unwrap();
    // Grid x=1 samples 0.4/0.6. B(0.4)=0.400390625 and
    // B(0.6)=0.6015625; separately rounded products add to -0.19921875.
    assert_eq!(floats(&inputs[0])[8].to_bits(), (-0.19921875_f32).to_bits());
    let legacy = c.base.managed_position_inputs(&h, &p).unwrap();
    let bits = u16::from_le_bytes(legacy[0].bytes().bytes()[16..18].try_into().unwrap());
    assert_eq!(half::bf16::from_bits(bits).to_f32(), -0.2001953125);
    drop(inputs);
    drop(legacy);
    assert_eq!(h.usage().unwrap().committed_bytes(), 0);
    let denied = host(1);
    assert!(c.managed_position_inputs(&denied, &p).is_err());
    assert_eq!(denied.usage().unwrap().committed_bytes(), 0);
}
#[test]
fn actual_original_stored_three_layers_execute_main_and_all_deepstack_then_rebind_shared_weights() {
    exercise_original_stored_component(config(4, 6), config(6, 4));
}

#[test]
fn actual_qwen35_stored_vision_without_deepstack_matches_oracle_and_retires_owners() {
    let c = qwen35_config(4, 6);
    let original = Original::new(&c);
    assert!(original.values.keys().all(|key| !key.contains("deepstack")));
    let (graph, _) = program::build(&c).unwrap();
    assert_eq!(graph.outputs(), &[id(Qwen3VLVisionComfyFamily::OUTPUT_ID)]);
    // A main-only program must not bind the incompatible three-merger Stored
    // schema even when all common layer dimensions happen to match.
    let other = config(4, 6);
    assert_ne!(
        program::schema(&c).unwrap(),
        program::schema(&other).unwrap()
    );
    exercise_original_stored_component(c, qwen35_config(6, 4));
}

fn exercise_original_stored_component(
    c: Qwen3VLVisionComfyConfig,
    alternate: Qwen3VLVisionComfyConfig,
) {
    let original = Original::new(&c);
    let (_dir, archive) = original.archive();
    let checkpoint = Qwen3VLVisionComfyCheckpoint::new(&c, &archive, limits()).unwrap();
    let r = ReferenceVNextComposition::create(id("device.reference.vision_comfy.stored")).unwrap();
    let h = host(8 << 20);
    let source = checkpoint.weight_source(&h);
    let component = compile(&c, &r);
    let first = &component
        .family()
        .stored_weight_schema()
        .unwrap()
        .components[0];
    let payload = source.component(first).unwrap();
    assert!(payload.retained_host_memory().is_some());
    assert!(h.usage().unwrap().committed_bytes() > 0);
    drop(payload);
    assert_eq!(h.usage().unwrap().committed_bytes(), 0);
    let (mut session, weights) = component
        .initialize_shared_weights(
            r.runtime().clone(),
            h.clone(),
            ComponentWeightSource::Stored(&source),
            init("cold"),
        )
        .unwrap();
    let mut baseline = None;
    for which in [0, 1, 0] {
        let x = patches(&c, which);
        let pos = c
            .managed_position_inputs(
                &h,
                &TensorArchive::tensor(&archive, Qwen3VLVisionFamily::POSITION_EXTERNAL_NAME)
                    .unwrap(),
            )
            .unwrap();
        let expected = oracle::forward(
            &c,
            &x,
            &floats(&pos[0]),
            &floats(&pos[1]),
            &floats(&pos[2]),
            &original,
        );
        let mut inputs = vec![input(&h, &c, &x)];
        inputs.extend(pos);
        let outputs = session
            .execute(
                id(format!("run.vision.execute.{which}")),
                id(format!("request.vision.execute.{which}")),
                inputs,
            )
            .unwrap();
        assert_eq!(outputs.len(), 1 + c.architecture().deepstack_layers().len());
        let mut snapshot = BTreeMap::new();
        for out in &outputs {
            assert_eq!(out.tensor().element_type(), ElementType::F32);
            assert_eq!(out.tensor().dimensions(), [6, 256]);
            assert!(h.owns(out.bytes()));
            let values = out
                .bytes()
                .bytes()
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect::<Vec<_>>();
            let target = &expected[out.value_id().as_str()];
            for (i, (&actual, &expected)) in values.iter().zip(target).enumerate() {
                assert!(
                    (actual - expected).abs() <= 3e-5,
                    "{} cell{i}: {actual} vs {expected}",
                    out.value_id()
                );
            }
            assert!(values.iter().any(|v| v.to_bits() & 0xffff != 0));
            snapshot.insert(
                out.value_id().to_string(),
                values.into_iter().map(f32::to_bits).collect::<Vec<_>>(),
            );
        }
        if which == 0 {
            if let Some(before) = &baseline {
                assert_eq!(before, &snapshot);
            } else {
                baseline = Some(snapshot);
            }
        } else {
            assert_ne!(baseline.as_ref().unwrap(), &snapshot);
        }
        drop(outputs);
    }
    assert!(matches!(
        session
            .close()
            .unwrap_or_else(|e| panic!("visual close: {:?}", e.failure())),
        PlanRuntimeCloseOutcome::Closed(_)
    ));
    let component = compile(&alternate, &r);
    let mut session = component
        .initialize_with_shared_weights(r.runtime().clone(), h.clone(), &weights, init("rebound"))
        .unwrap();
    let x = patches(&alternate, 0);
    let pos = alternate
        .managed_position_inputs(
            &h,
            &TensorArchive::tensor(&archive, Qwen3VLVisionFamily::POSITION_EXTERNAL_NAME).unwrap(),
        )
        .unwrap();
    let mut inputs = vec![input(&h, &alternate, &x)];
    inputs.extend(pos);
    let out = session
        .execute(
            id("run.vision.rebound"),
            id("request.vision.rebound"),
            inputs,
        )
        .unwrap();
    assert_eq!(
        out.len(),
        1 + alternate.architecture().deepstack_layers().len()
    );
    let retained = out[0].bytes().bytes().to_vec();
    assert!(matches!(
        session
            .close()
            .unwrap_or_else(|e| panic!("visual rebind close: {:?}", e.failure())),
        PlanRuntimeCloseOutcome::Closed(_)
    ));
    assert_eq!(out[0].bytes().bytes(), retained);
    drop(out);
    assert!(matches!(
        weights.close(),
        Ok(SharedComponentWeightsCloseOutcome::Closed(_))
    ));
    assert_eq!(h.usage().unwrap().committed_bytes(), 0);
    assert_eq!(r.runtime().snapshot().live_allocations, 0);
}
