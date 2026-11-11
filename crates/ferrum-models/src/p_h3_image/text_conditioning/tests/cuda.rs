//! Scaled Stored vertical fixture; not a real 36-layer checkpoint/image test.
use super::*;
use ferrum_execution::{
    component::{
        CompiledTensorComponent, ComponentInitializationConfig, ComponentWeightSource,
        TensorComponentPolicy,
    },
    TensorExecutorConfig, TensorSerialActor,
};
use ferrum_kernels::backend::cuda::{
    vnext_ops::{
        cuda_tensor_runtime_config, cuda_vnext_dense_f32_provider, cuda_vnext_tensor_providers,
        CudaTensorOperation,
    },
    vnext_runtime::{CudaDeviceRuntime, CudaTensorRuntimeConfig},
};
use std::sync::{Arc, Mutex};
static GPU: Mutex<()> = Mutex::new(());

#[test]
#[ignore = "requires configured CUDA host and pinned native operator-set lock"]
fn actual_stored_tokens_five_taps_and_merged_connector_reuse_sessions_and_refund_host() {
    let _guard = GPU.lock().unwrap_or_else(|p| p.into_inner());
    let p = prompt();
    let c = config(p.layout().clone());
    let (_directory, source) = archive(&original(&c));
    let source = Arc::new(source);
    let weight_adapter = PH3ImageTextWeightAdapter::new(&c, source.clone(), limits()).unwrap();
    let runtime = Arc::new(
        CudaDeviceRuntime::new_tensor_eager(CudaTensorRuntimeConfig {
            device: cuda_tensor_runtime_config(0, id("device.cuda.0")).unwrap(),
            usable_memory_bytes: 1 << 30,
            bootstrap_memory_bytes: 256 << 20,
            stream_memory_bytes: 16 << 20,
            maximum_streams: 2,
            maximum_in_flight_per_stream: 1,
            maximum_commands_per_submission: 256,
            opaque_memory_profile: "unqualified-ph3-text-conditioning-fixture-v1".into(),
        })
        .unwrap(),
    );
    let mut providers = cuda_vnext_tensor_providers(
        &runtime,
        &[
            CudaTensorOperation::ComfyConvRotInt8EmbeddingF32,
            CudaTensorOperation::ComfyConvRotW4A8F32,
            CudaTensorOperation::ComfyConvRotW8A8F32,
            CudaTensorOperation::HeadRmsNormF32,
            CudaTensorOperation::HalfSplitRopeF32,
            CudaTensorOperation::StructuredGqaAttentionF32,
            CudaTensorOperation::PlainResidualAddF32,
            CudaTensorOperation::SiluMulF32,
        ],
    )
    .unwrap();
    providers.push(cuda_vnext_dense_f32_provider(&runtime).unwrap());
    let registry = OperationRuntimeRegistry::new(
        vec![
            Box::new(comfy_convrot_int8_embedding_f32_contract().unwrap()),
            Box::new(comfy_convrot_w4a8_f32_contract().unwrap()),
            Box::new(comfy_convrot_w8a8_f32_contract().unwrap()),
            Box::new(head_rms_norm_f32_contract().unwrap()),
            Box::new(half_split_rope_f32_contract().unwrap()),
            Box::new(structured_gqa_attention_f32_contract().unwrap()),
            Box::new(plain_residual_add_f32_contract().unwrap()),
            Box::new(silu_mul_f32_contract().unwrap()),
            Box::new(dense_linear_f32_contract().unwrap()),
        ],
        providers,
    )
    .unwrap();
    let materializers = WeightMaterializerRegistry::identity_only().unwrap();
    let d = runtime.descriptor();
    let engine = EngineProviderDescriptor::new(
        id("provider.engine.ph3_text_fixture"),
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
    let encoder_family =
        TypedFamilyRegistration::new(PH3ImageTextConditioningFamily::new(limits()).unwrap())
            .prepare_with_profile(
                &serde_json::to_value(&c).unwrap(),
                &id(PH3ImageTextConditioningFamily::CUDA_PROFILE_ID),
            )
            .unwrap();
    let encoder = CompiledTensorComponent::compile(
        encoder_family,
        &registry,
        &catalog,
        &policy,
        &materializers,
        &c.compile_options().unwrap(),
    )
    .unwrap();
    let connector_config = c.connector_config().unwrap();
    let projection_config = c.projection_config().unwrap();
    let projection_family = TypedFamilyRegistration::new(
        PH3ImageTextConnectorProjectionFamily::new(PH3ImageTextConnectorProjectionLimits {
            max_projection_bytes: 1 << 20,
        })
        .unwrap(),
    )
    .prepare_with_profile(
        &serde_json::to_value(&projection_config).unwrap(),
        &id(PH3ImageTextConnectorProjectionFamily::CUDA_PROFILE_ID),
    )
    .unwrap();
    let projection_options = ProgramPlanCompileOptions::new(BTreeMap::from([(
        id(PH3ImageTextConnectorProjectionFamily::INPUT_ID),
        projection_config
            .input_spec(c.encoder().text().layout().rows())
            .unwrap(),
    )]))
    .unwrap();
    let projection = CompiledTensorComponent::compile(
        projection_family,
        &registry,
        &catalog,
        &policy,
        &materializers,
        &projection_options,
    )
    .unwrap();
    let host = host(32 << 20);
    assert_eq!(
        weight_adapter
            .maximum_group_backing_bytes(
                encoder.family(),
                encoder.executable().execution_plan(),
                &host,
            )
            .unwrap(),
        host.backing_bytes(
            usize::try_from(c.encoder().text().architecture().hidden_size() * 4).unwrap()
        )
        .unwrap(),
    );
    let weight_source = weight_adapter.weight_source(&host);
    let connector_checkpoint = PH3ImageTextConnectorCheckpoint::new_with_layout(
        &connector_config,
        source.archive(),
        PH3ImageTextConnectorProjectionLimits {
            max_projection_bytes: 1 << 20,
        },
        PH3ImageTextConnectorCheckpointLayout::ComfyMergedTextEncoder,
    )
    .unwrap();
    let init = |name| ComponentInitializationConfig {
        request_id: id(format!("request.ph3_text.{name}.init")),
        run_id: id(format!("run.ph3_text.{name}.init")),
        transaction_id: id(format!("transaction.ph3_text.{name}.init")),
        static_policy: StaticInitializationPolicy::new(8 << 20, 512).unwrap(),
        executor: TensorExecutorConfig::default(),
    };
    let mut encoder = encoder
        .initialize(
            runtime.clone(),
            host.clone(),
            ComponentWeightSource::Stored(&weight_source),
            init("encoder"),
        )
        .unwrap_or_else(|e| panic!("encoder init: {e:?}"));
    let mut projection = projection
        .initialize(
            runtime.clone(),
            host.clone(),
            ComponentWeightSource::Stored(source.as_ref()),
            init("connector"),
        )
        .unwrap_or_else(|e| panic!("projection init: {e:?}"));
    let baseline = host.usage().unwrap().committed_bytes();
    let mut actor = TensorSerialActor::new(runtime.clone()).unwrap();
    {
        let mut conditioning = PH3ImageTextConditioning::bind(
            &c,
            limits(),
            &connector_checkpoint,
            &mut encoder,
            &mut projection,
            PH3ImageConditioningPrecision::F32,
        )
        .unwrap();
        for repeat in 0..2 {
            let run = PH3ImageTextConditioningRun {
                encoder_run: id(format!("run.ph3_text.encoder.{repeat}")),
                encoder_request: id(format!("request.ph3_text.encoder.{repeat}")),
                projection_run: id(format!("run.ph3_text.connector.{repeat}")),
                projection_request: id(format!("request.ph3_text.connector.{repeat}")),
            };
            let output = if repeat == 0 {
                conditioning.execute(&p, run)
            } else {
                conditioning.execute_on_actor(&mut actor, &p, run)
            }
            .unwrap();
            assert_eq!(output.valid_lengths(), [2, 1]);
            assert_eq!(output.conditioning().tensor().dimensions(), [4, 4]);
            let values = output
                .conditioning()
                .bytes()
                .bytes()
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect::<Vec<_>>();
            // Every projection is exact zero, so residuals preserve embedding.
            // H256 of an impulse16 at scale0.5 is all0.5; impulse-32 -> all-1.
            // Four pre-final taps remain -1; final tap is -1/sqrt(1+epsilon).
            // Connector selects these channels independently; each sample's
            // first row is replaced by the original sink, including batch1.
            let expected = [
                -3.0,
                2.0,
                7.0,
                1.0,
                0.5,
                -1.0 - 0.25 / (1.25_f64).sqrt(),
                -0.5,
                1.25,
                -3.0,
                2.0,
                7.0,
                1.0,
            ];
            for (&actual, expected) in values[..12].iter().zip(expected) {
                assert!(
                    (f64::from(actual) - expected).abs() <= 2.0e-6,
                    "conditioning {actual} != {expected}"
                );
            }
            drop(output);
            assert_eq!(
                host.usage().unwrap().committed_bytes(),
                baseline,
                "request retained temporary Host owners"
            );
        }
        let wrong = PH3ImageTextPromptBatch::from_rows(vec![vec![0]], 15, 16).unwrap();
        let failed = conditioning.execute(
            &wrong,
            PH3ImageTextConditioningRun {
                encoder_run: id("run.ph3_text.reject.encoder"),
                encoder_request: id("request.ph3_text.reject.encoder"),
                projection_run: id("run.ph3_text.reject.connector"),
                projection_request: id("request.ph3_text.reject.connector"),
            },
        );
        assert!(matches!(
            failed,
            Err(PH3ImageTextConditioningError::Preparation(_))
        ));
        assert_eq!(host.usage().unwrap().committed_bytes(), baseline);
    }
    assert!(matches!(
        encoder
            .close()
            .unwrap_or_else(|f| panic!("encoder close: {:?}", f.failure())),
        PlanRuntimeCloseOutcome::Closed(_)
    ));
    assert!(matches!(
        projection
            .close()
            .unwrap_or_else(|f| panic!("projection close: {:?}", f.failure())),
        PlanRuntimeCloseOutcome::Closed(_)
    ));
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}
