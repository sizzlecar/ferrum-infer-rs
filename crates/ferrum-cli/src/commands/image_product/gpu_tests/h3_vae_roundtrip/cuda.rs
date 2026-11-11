//! Test-only composition of the existing VAE and native F32 noise providers.
use super::*;
use ferrum_execution::component::{
    CompiledTensorComponent, ComponentInitializationConfig, ComponentWeightSource,
    TensorComponentPolicy, TensorComponentSession,
};
pub(super) use ferrum_kernels::backend::cuda::vnext_runtime::CudaDeviceRuntime;
use ferrum_kernels::backend::cuda::{vnext_ops::*, vnext_runtime::CudaTensorRuntimeConfig};
use ferrum_models::normal_noise::{NormalNoiseConfig, NormalNoiseFamily, NormalNoisePrecision};
use std::sync::{Arc, Mutex};

pub(super) struct Execution {
    pub runtime: Arc<CudaDeviceRuntime>,
    registry: OperationRuntimeRegistry<CudaDeviceRuntime>,
    catalog: CapabilityCatalog,
    materializers: WeightMaterializerRegistry,
    initialization: ComponentInitializationConfig,
}

impl Execution {
    pub fn new(
        config: &ferrum_engine::image_generation::p_h3::cuda::PH3ImageCudaConfig,
    ) -> Result<Self> {
        config.validate()?;
        let runtime = Arc::new(CudaDeviceRuntime::new_tensor_eager(
            CudaTensorRuntimeConfig {
                device: cuda_tensor_runtime_config(
                    config.ordinal,
                    DeviceId::new(format!("device.cuda.{}", config.ordinal))?,
                )?,
                usable_memory_bytes: match config.memory {
                    ferrum_engine::image_generation::cuda::ImageCudaMemoryPolicy::ExplicitTotal(
                        n,
                    ) => n.get(),
                    _ => anyhow::bail!("roundtrip requires the product's explicit CUDA capacity"),
                },
                bootstrap_memory_bytes: config.bootstrap_memory_bytes,
                stream_memory_bytes: config.stream_memory_bytes,
                maximum_streams: config.stream_plan()?.resolved_streams(),
                maximum_in_flight_per_stream: config.maximum_in_flight_per_stream,
                maximum_commands_per_submission: config.maximum_commands_per_submission,
                opaque_memory_profile: config.opaque_memory_profile.clone(),
            },
        )?);
        let pairs: Vec<(CudaTensorOperation, Box<dyn OperationContract>)> = vec![
            (
                CudaTensorOperation::ComfyConvRotW8A8F32,
                Box::new(comfy_convrot_w8a8_f32_contract()?),
            ),
            (
                CudaTensorOperation::HeadRmsNormF32,
                Box::new(head_rms_norm_f32_contract()?),
            ),
            (
                CudaTensorOperation::UnitHeadRmsPartialRopeF32,
                Box::new(unit_head_rms_partial_rope_f32_contract()?),
            ),
            (
                CudaTensorOperation::StructuredGqaAttentionF32,
                Box::new(structured_gqa_attention_f32_contract()?),
            ),
            (
                CudaTensorOperation::SiluMulF32,
                Box::new(silu_mul_f32_contract()?),
            ),
            (
                CudaTensorOperation::GroupedColumnSliceF32,
                Box::new(grouped_column_slice_f32_contract()?),
            ),
            (
                CudaTensorOperation::ChannelBiasAddF32,
                Box::new(channel_bias_add_f32_contract()?),
            ),
            (
                CudaTensorOperation::ChannelScaledResidualF32,
                Box::new(channel_scaled_residual_f32_contract()?),
            ),
            (
                CudaTensorOperation::NanToNumF32,
                Box::new(nan_to_num_f32_contract()?),
            ),
            (
                CudaTensorOperation::BatchedSuffixAppendF32,
                Box::new(batched_suffix_append_f32_contract()?),
            ),
            (
                CudaTensorOperation::LayerNormF32,
                Box::new(layer_norm_f32_contract()?),
            ),
            (
                CudaTensorOperation::GroupNormSiluNchwF32,
                Box::new(group_norm_silu_nchw_f32_contract()?),
            ),
            (
                CudaTensorOperation::PlainResidualAddNchwF32,
                Box::new(plain_residual_add_nchw_f32_contract()?),
            ),
            (
                CudaTensorOperation::PhiloxNormalF32,
                Box::new(philox_normal_f32_contract()?),
            ),
        ];
        let (selected, mut contracts): (Vec<_>, Vec<_>) = pairs.into_iter().unzip();
        // One factory loads the native module once and owns the real cuDNN
        // allowance. A separate noise factory would duplicate that allowance.
        let mut providers = cuda_vnext_cnn_f32_providers(&runtime, &config.cudnn, &selected)?;
        providers.push(cuda_vnext_affine_linear_f32_provider(&runtime)?);
        contracts.push(Box::new(conv2d_f32_contract()?));
        contracts.push(Box::new(affine_linear_f32_contract()?));
        let registry = OperationRuntimeRegistry::new(contracts, providers)?;
        let descriptor = runtime.descriptor();
        let materializers = WeightMaterializerRegistry::identity_only()?;
        let engine = EngineProviderDescriptor::new(
            id("provider.engine.cuda.vae_roundtrip")?,
            ContractVersion::new(1, 0),
            descriptor.runtime_implementation_fingerprint.clone(),
            descriptor.id.clone(),
            descriptor.capabilities.clone(),
        )?;
        let catalog = materializers
            .augment_catalog(registry.capability_catalog(descriptor.clone(), vec![engine])?)?;
        ferrum_kernels::native_ops::validate_compiled_native_operator_provider_catalog_for_units(
            &catalog.native_operator_provider_catalog(ferrum_types::NativeOperatorBackend::Cuda)?,
            ferrum_kernels::native_ops::compiled_native_operator_artifacts(),
            &[],
        )
        .map_err(anyhow::Error::msg)?;
        Ok(Self {
            runtime,
            registry,
            catalog,
            materializers,
            initialization: ComponentInitializationConfig {
                request_id: id("request.vae_roundtrip.init")?,
                run_id: id("run.vae_roundtrip.init")?,
                transaction_id: id("transaction.vae_roundtrip.init")?,
                static_policy: config.pipeline.static_initialization,
                executor: config.pipeline.executor,
            },
        })
    }

    pub fn initialize(
        &self,
        name: &str,
        family: PreparedModelFamily,
        options: &ProgramPlanCompileOptions,
        weights: ComponentWeightSource<'_>,
        host: &HostMemoryAccount,
    ) -> Result<(TensorComponentSession<CudaDeviceRuntime>, Value)> {
        let start = Instant::now();
        let d = self.runtime.descriptor();
        let component = CompiledTensorComponent::compile(
            family,
            &self.registry,
            &self.catalog,
            &TensorComponentPolicy::new(
                d.total_memory_bytes,
                d.dynamic_storage_profiles.iter().copied().collect(),
            )?,
            &self.materializers,
            options,
        )?;
        let compile_s = start.elapsed().as_secs_f64();
        let mut config = self.initialization.clone();
        config.request_id = id(format!("request.vae_roundtrip.{name}.init"))?;
        config.run_id = id(format!("run.vae_roundtrip.{name}.init"))?;
        config.transaction_id = id(format!("transaction.vae_roundtrip.{name}.init"))?;
        let start = Instant::now();
        let session = component
            .initialize(self.runtime.clone(), host.clone(), weights, config)
            .map_err(|e| retained(format!("{name} initialization: {e}"), e))?;
        let record = json!({"component":name,"compile_s":compile_s,"initialize_s":start.elapsed().as_secs_f64(),
            "initialization":session.initialization_receipt(),
            "profile":component.family().numerical_profile().id.as_str()});
        Ok((session, record))
    }

    pub fn posterior_noise(
        &self,
        spec: &ResolvedTensorSpec,
        host: &HostMemoryAccount,
        actor: &mut TensorSerialActor<CudaDeviceRuntime>,
    ) -> Result<(ManagedHostTensor, Value)> {
        let config = NormalNoiseConfig::new_with_precision(
            1,
            1,
            spec.minimum_storage_bytes()? / 4,
            NormalNoisePrecision::F32,
        )?;
        let family = TypedFamilyRegistration::new(NormalNoiseFamily::new_with_precision(
            NormalNoisePrecision::F32,
            CUDA_PHILOX_NORMAL_F32_NUMERICAL_PROFILE_ID,
        )?)
        .prepare_with_profile(
            &serde_json::to_value(&config)?,
            &id(CUDA_PHILOX_NORMAL_F32_NUMERICAL_PROFILE_ID)?,
        )?;
        let (mut session, initialization) = self.initialize(
            "posterior_noise",
            family,
            &config.compile_options()?,
            ComponentWeightSource::None,
            host,
        )?;
        let mut state = host.allocate_zeroed(HostMemoryClass::InputPreprocess, 16)?;
        state
            .as_mut()
            .copy_from_slice(&PhiloxSeedOffset::fresh(42).little_endian_bytes());
        let state = ManagedHostTensor::new(
            id(NormalNoiseFamily::STATE_ID)?,
            NormalNoiseConfig::state_tensor()?,
            state.freeze(),
        )?;
        let start = Instant::now();
        let result = session.execute_on_actor(
            actor,
            id("run.vae_roundtrip.posterior_noise")?,
            id("request.vae_roundtrip.posterior_noise")?,
            vec![state],
        );
        let execute_s = start.elapsed().as_secs_f64();
        let output = result.map_err(|e| retained(format!("posterior noise execution: {e}"), e))?;
        ensure!(
            output.len() == 1
                && output[0].tensor() == &config.output_tensor()?
                && output[0].value_id().as_str() == NormalNoiseFamily::OUTPUT_ID,
            "native posterior output differs from its explicit F32 recipe"
        );
        let mut bytes =
            host.allocate_zeroed(HostMemoryClass::InputPreprocess, output[0].bytes().len())?;
        bytes.as_mut().copy_from_slice(output[0].bytes().bytes());
        let noise = ManagedHostTensor::new(
            id("value.vae_roundtrip.posterior_noise")?,
            spec.clone(),
            bytes.freeze(),
        )?;
        drop(output);
        let close = close(session)?;
        Ok((
            noise,
            json!({"initialization":initialization,"execute_s":execute_s,"close":close,
            "profile":CUDA_PHILOX_NORMAL_F32_NUMERICAL_PROFILE_ID,"seed":42,"torch_cpu_noise_equivalent":false}),
        ))
    }
}

pub(super) fn close(session: TensorComponentSession<CudaDeviceRuntime>) -> Result<Value> {
    let start = Instant::now();
    let outcome = session
        .close()
        .map_err(|e| retained(format!("VAE component close failed: {:?}", e.failure()), e))?;
    match outcome {
        PlanRuntimeCloseOutcome::Closed(receipt) => Ok(
            json!({"outcome":"closed","close_s":start.elapsed().as_secs_f64(),"receipt":receipt}),
        ),
        other => Err(retained(
            "VAE component is still referenced; retirement is not complete".into(),
            other,
        )),
    }
}

// Keep the original resource/fence owner in the returned error. Printing an
// error string or dropping it is never reported as successful retirement.
pub(super) fn retained<T: Send + 'static>(message: String, owner: T) -> anyhow::Error {
    struct Failure<T> {
        message: String,
        _owner: Mutex<T>,
    }
    impl<T> std::fmt::Debug for Failure<T> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str(&self.message)
        }
    }
    impl<T> std::fmt::Display for Failure<T> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str(&self.message)
        }
    }
    impl<T> std::error::Error for Failure<T> {}
    anyhow::Error::new(Failure {
        message,
        _owner: Mutex::new(owner),
    })
}
