//! Real CUDA three-layer Stored visual execution. The fixture bound qualifies
//! selected scaled tensors; it is not an official-checkpoint image quality or
//! an upstream Torch binary-parity claim.
use super::*;
use ferrum_kernels::backend::cuda::{
    vnext_ops::{
        cuda_tensor_runtime_config, cuda_vnext_affine_linear_f32_provider,
        cuda_vnext_patch_projection_f32_provider, cuda_vnext_tensor_providers, CudaTensorOperation,
    },
    vnext_runtime::{CudaDeviceRuntime, CudaTensorRuntimeConfig},
};
use std::sync::Arc;

#[test]
#[ignore = "requires configured CUDA host with admitted F32 visual fixture capacity"]
fn original_stored_f32_visual_three_layers_emit_main_and_deepstack_with_shared_weights() {
    exercise_cuda_stored_vision(config(4, 6), config(6, 4));
}

#[test]
#[ignore = "requires configured CUDA host with admitted F32 Qwen3.5 visual fixture capacity"]
fn original_stored_qwen35_visual_emits_only_main_with_shared_weights() {
    exercise_cuda_stored_vision(qwen35_config(4, 6), qwen35_config(6, 4));
}

fn exercise_cuda_stored_vision(c: Qwen3VLVisionComfyConfig, other: Qwen3VLVisionComfyConfig) {
    let original = Original::new(&c);
    let (_dir, archive) = original.archive();
    let checkpoint = Qwen3VLVisionComfyCheckpoint::new(&c, &archive, limits()).unwrap();
    let runtime = Arc::new(
        CudaDeviceRuntime::new_tensor_eager(CudaTensorRuntimeConfig {
            device: cuda_tensor_runtime_config(0, id("device.cuda.0")).unwrap(),
            usable_memory_bytes: 1 << 30,
            bootstrap_memory_bytes: 256 << 20,
            stream_memory_bytes: 16 << 20,
            maximum_streams: 1,
            maximum_in_flight_per_stream: 1,
            maximum_commands_per_submission: 256,
            opaque_memory_profile: "f32-vision-scaled-fixture-unqualified-capacity".into(),
        })
        .unwrap(),
    );
    let mut providers = cuda_vnext_tensor_providers(
        &runtime,
        &[
            CudaTensorOperation::PlainResidualAddF32,
            CudaTensorOperation::GroupedColumnSliceF32,
            CudaTensorOperation::HalfSplitRopeF32,
            CudaTensorOperation::StructuredGqaAttentionF32,
            CudaTensorOperation::LayerNormF32,
            CudaTensorOperation::GeluTanhF32,
            CudaTensorOperation::GeluErfF32,
            CudaTensorOperation::ReshapeCopyF32,
        ],
    )
    .unwrap();
    providers.push(cuda_vnext_affine_linear_f32_provider(&runtime).unwrap());
    providers.push(cuda_vnext_patch_projection_f32_provider(&runtime).unwrap());
    let registry = OperationRuntimeRegistry::new(
        vec![
            Box::new(plain_residual_add_f32_contract().unwrap()),
            Box::new(grouped_column_slice_f32_contract().unwrap()),
            Box::new(half_split_rope_f32_contract().unwrap()),
            Box::new(structured_gqa_attention_f32_contract().unwrap()),
            Box::new(layer_norm_f32_contract().unwrap()),
            Box::new(gelu_tanh_f32_contract().unwrap()),
            Box::new(gelu_erf_f32_contract().unwrap()),
            Box::new(reshape_copy_f32_contract().unwrap()),
            Box::new(affine_linear_f32_contract().unwrap()),
            Box::new(patch_projection_f32_contract().unwrap()),
        ],
        providers,
    )
    .unwrap();
    let materializers = WeightMaterializerRegistry::identity_only().unwrap();
    let d = runtime.descriptor();
    let engine = EngineProviderDescriptor::new(
        id("provider.engine.cuda.vision_comfy.fixture"),
        ContractVersion::new(1, 0),
        d.runtime_implementation_fingerprint.clone(),
        d.id.clone(),
        d.capabilities.clone(),
    )
    .unwrap();
    let catalog = materializers
        .augment_catalog(
            registry
                .capability_catalog(d.clone(), vec![engine])
                .unwrap(),
        )
        .unwrap();
    let policy = TensorComponentPolicy::new(
        d.total_memory_bytes,
        d.dynamic_storage_profiles.iter().copied().collect(),
    )
    .unwrap();
    let compile = |c: &Qwen3VLVisionComfyConfig| {
        let family = TypedFamilyRegistration::new(Qwen3VLVisionComfyFamily::new(limits()).unwrap())
            .prepare_with_profile(
                &serde_json::to_value(c).unwrap(),
                &id(Qwen3VLVisionComfyFamily::CUDA_PROFILE_ID),
            )
            .unwrap();
        CompiledTensorComponent::compile(
            family,
            &registry,
            &catalog,
            &policy,
            &materializers,
            &c.compile_options().unwrap(),
        )
        .unwrap()
    };
    let h = host(8 << 20);
    let source = checkpoint.weight_source(&h);
    let component = compile(&c);
    let (mut session, weights) = component
        .initialize_shared_weights(
            runtime.clone(),
            h.clone(),
            ComponentWeightSource::Stored(&source),
            init("cuda_cold"),
        )
        .unwrap();
    let mut baseline = None;
    for (trial, which) in [0, 1, 0].into_iter().enumerate() {
        let x = patches(&c, which);
        let table =
            TensorArchive::tensor(&archive, Qwen3VLVisionFamily::POSITION_EXTERNAL_NAME).unwrap();
        let positions = c.managed_position_inputs(&h, &table).unwrap();
        let expected = oracle::forward(
            &c,
            &x,
            &floats(&positions[0]),
            &floats(&positions[1]),
            &floats(&positions[2]),
            &original,
        );
        let mut inputs = vec![input(&h, &c, &x)];
        inputs.extend(positions);
        let outputs = session
            .execute(
                id(format!("run.cuda.visual.{trial}")),
                id(format!("request.cuda.visual.{trial}")),
                inputs,
            )
            .unwrap();
        assert_eq!(outputs.len(), 1 + c.architecture().deepstack_layers().len());
        let mut snapshot = BTreeMap::new();
        for out in &outputs {
            assert_eq!(out.tensor().dimensions(), [6, 256]);
            assert_eq!(out.tensor().element_type(), ElementType::F32);
            assert!(h.owns(out.bytes()));
            let actual = out
                .bytes()
                .bytes()
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect::<Vec<_>>();
            for (i, (&x, &y)) in actual
                .iter()
                .zip(&expected[out.value_id().as_str()])
                .enumerate()
            {
                assert!(
                    x.is_finite() && (x - y).abs() <= 3e-5,
                    "{} cell{i}: {x} vs {y}",
                    out.value_id()
                );
            }
            snapshot.insert(
                out.value_id().to_string(),
                actual.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
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
            .unwrap_or_else(|e| panic!("CUDA vision close: {:?}", e.failure())),
        PlanRuntimeCloseOutcome::Closed(_)
    ));
    let component = compile(&other);
    let mut attached = component
        .initialize_with_shared_weights(runtime, h.clone(), &weights, init("cuda_rebind"))
        .unwrap();
    let table =
        TensorArchive::tensor(&archive, Qwen3VLVisionFamily::POSITION_EXTERNAL_NAME).unwrap();
    let positions = other.managed_position_inputs(&h, &table).unwrap();
    let mut inputs = vec![input(&h, &other, &patches(&other, 0))];
    inputs.extend(positions);
    let outputs = attached
        .execute(
            id("run.cuda.visual.rebound"),
            id("request.cuda.visual.rebound"),
            inputs,
        )
        .unwrap();
    assert_eq!(
        outputs.len(),
        1 + other.architecture().deepstack_layers().len()
    );
    assert!(matches!(
        attached
            .close()
            .unwrap_or_else(|e| panic!("CUDA visual rebind close: {:?}", e.failure())),
        PlanRuntimeCloseOutcome::Closed(_)
    ));
    assert!(h.owns(outputs[0].bytes()));
    drop(outputs);
    assert!(matches!(
        weights.close(),
        Ok(SharedComponentWeightsCloseOutcome::Closed(_))
    ));
    assert_eq!(h.usage().unwrap().committed_bytes(), 0);
}
