//! Native Comfy composition under the product's one CUDA runtime, Host parent,
//! catalog and compiled native artifact set. Construction is not numerical or
//! concurrency qualification of the complete released model.
use super::{PH3ImageAssets, PH3ImagePipeline, PH3ImagePipelineOptions};
use crate::image_generation::{
    cuda::ImageCudaMemoryPolicy, ImageCudaAttentionPolicy, ImageCudaConvolutionWorkspacePolicy,
    ImageDecoderBatchPolicy, ImageEditPrefixPolicy, ImagePipelineLimits, ImageStreamLimit,
    ImageStreamPlan, ImageStreamRequirements,
};
use anyhow::{ensure, Context, Result};
use ferrum_kernels::backend::cuda::{
    vnext_ops::*,
    vnext_runtime::{
        CudaDeviceCommand, CudaDeviceRuntime, CudaNativeAllocationReuse, CudaNativeLaneReuse,
        CudaTensorRuntimeConfig, CudaTensorRuntimeOptions,
    },
    CudaCudnnConfig, CudaDenseF16CausalProfile, CudaTensorAndCausalRuntimeAutoConfig,
    CudaTensorAndCausalRuntimeConfig, CudaTensorMemoryObservation, CudaTensorRuntimeAutoConfig,
    CudnnSdpaPlanPolicy,
};
use ferrum_runtime::*;
use std::{num::NonZeroU64, sync::Arc};
mod operations;
mod preflight;

pub const PH3_IMAGE_CUDA_PRESET_ID: &str = "comfy_w4a8_f32_v1";

fn validate_decode_replay_limits(
    engine: Option<&ferrum_types::EngineConfig>,
    profile: Option<&CudaDenseF16CausalProfile>,
) -> Result<()> {
    let requested = engine.filter(|engine| {
        engine.backend.enable_reusable_execution
            && engine.backend.reusable_execution_capture.scope
                == ferrum_types::ReusableExecutionCaptureScope::DecodeOnly
    });
    let limits = profile.and_then(CudaDenseF16CausalProfile::decode_replay_limits);
    ensure!(
        requested.is_some() == limits.is_some(),
        "H3 expansion decode replay and its explicit CUDA graph allowance must agree"
    );
    if let (Some(engine), Some((maximum, _))) = (requested, limits) {
        let capture = &engine.backend.reusable_execution_capture;
        ensure!(
            capture.preparation == ferrum_types::ReusableExecutionPreparationMode::OnDemand
                && capture.maximum_executables_per_lane == Some(maximum as u64),
            "H3 expansion core and CUDA per-lane executable caps must agree under on-demand preparation"
        );
    } else {
        ensure!(
            engine.is_none_or(|engine| engine
                .backend
                .reusable_execution_capture
                .maximum_executables_per_lane
                .is_none()),
            "H3 expansion executable cap requires explicit CUDA decode replay"
        );
    }
    Ok(())
}

#[derive(Clone, Debug)]
pub struct PH3ImageCudaConfig {
    pub ordinal: usize,
    pub memory: ImageCudaMemoryPolicy,
    pub bootstrap_memory_bytes: u64,
    pub stream_memory_bytes: u64,
    pub maximum_streams: ImageStreamLimit,
    /// Native idle lanes remain charged against this runtime's stream quota.
    pub native_lane_reuse: CudaNativeLaneReuse,
    /// Fixed opaque reservation; reused bytes still require fresh Core permits.
    pub native_allocation_reuse: CudaNativeAllocationReuse,
    pub pinned_weight_upload: Option<super::PH3ImagePinnedWeightUpload>,
    pub maximum_in_flight_per_stream: usize,
    pub maximum_commands_per_submission: usize,
    pub opaque_memory_profile: String,
    /// An explicit extra native-module allowance. Presence must match the
    /// validated assets' declared prompt-expansion capability.
    pub prompt_expansion_causal: Option<CudaDenseF16CausalProfile>,
    pub cudnn: CudaCudnnConfig,
    pub attention_policy: ImageCudaAttentionPolicy,
    pub convolution_workspace: ImageCudaConvolutionWorkspacePolicy,
    pub host: HostMemoryLimits,
    pub pipeline: ImagePipelineLimits,
    pub options: PH3ImagePipelineOptions,
}
#[derive(Debug, Clone, serde::Serialize)]
pub struct PH3ImageCudaCompositionReceipt {
    pub preset: &'static str,
    /// Exact native affine/patch implementation identities in this composition.
    pub affine_f32_numerical_profile: &'static str,
    pub patch_projection_f32_numerical_profile: &'static str,
    pub memory_policy: &'static str,
    pub stream_plan: ImageStreamPlan,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub native_lane_reuse_maximum_idle_lanes: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub native_allocation_reuse: Option<PH3ImageNativeAllocationReuseReceipt>,
    pub pinned_weight_upload: Option<super::PH3ImagePinnedWeightUpload>,
    pub device_capacity_bytes: u64,
    pub observed_memory: Option<CudaTensorMemoryObservation>,
    pub runtime_opaque_profile: String,
    pub prompt_expansion_causal_module_memory_bytes: Option<u64>,
    pub prompt_expansion_causal_opaque_profile: Option<String>,
    pub prompt_expansion_causal_mode: Option<&'static str>,
    pub prompt_expansion_decode_replay_limits: Option<(usize, u64)>,
    pub prompt_expansion_vision_profile: Option<String>,
    pub cudnn_library: std::path::PathBuf,
    pub cudnn_opaque_profile: String,
    pub cudnn_convolution_workspace_policy: ImageCudaConvolutionWorkspacePolicy,
    pub cudnn_convolution_workspace_limit_bytes: u64,
    pub attention_policy: ImageCudaAttentionPolicy,
    pub attention_opaque_memory_bytes: u64,
    pub operation_ids: Vec<OperationId>,
    pub largest_component_execution_weight_bytes: u64,
    pub initial_noise_policy: &'static str,
    pub vae_decoder_tile_batch: ferrum_models::p_h3_image_vae::PH3ImageVaeTileBatch,
}
#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct PH3ImageNativeAllocationReuseReceipt {
    pub maximum_idle_bytes: u64,
    pub maximum_entry_bytes: u64,
    pub maximum_entries: usize,
}
impl PH3ImageCudaConfig {
    fn native_allocation_reuse_receipt(&self) -> Option<PH3ImageNativeAllocationReuseReceipt> {
        match self.native_allocation_reuse {
            CudaNativeAllocationReuse::Off => None,
            CudaNativeAllocationReuse::Bounded {
                maximum_idle_bytes,
                maximum_entry_bytes,
                maximum_entries,
            } => Some(PH3ImageNativeAllocationReuseReceipt {
                maximum_idle_bytes: maximum_idle_bytes.get(),
                maximum_entry_bytes: maximum_entry_bytes.get(),
                maximum_entries: maximum_entries.get(),
            }),
        }
    }

    fn native_lane_reuse_maximum_idle_lanes(&self) -> Option<usize> {
        match self.native_lane_reuse {
            CudaNativeLaneReuse::Off => None,
            CudaNativeLaneReuse::Bounded { maximum_idle_lanes } => Some(maximum_idle_lanes.get()),
        }
    }

    pub fn stream_plan(&self) -> Result<ImageStreamPlan> {
        let requirements = ImageStreamRequirements::for_serial_pipeline(
            self.pipeline.static_initialization,
            self.pipeline.executor,
        );
        self.maximum_streams
            .resolve(if self.prompt_expansion_causal.is_some() {
                requirements.with_serial_causal_phase()
            } else {
                requirements
            })
    }
    /// Pure declared-policy checks, before source reads, native loads or a
    /// driver context. Actual request shapes still require ordinary admission.
    pub fn validate(&self) -> Result<()> {
        self.pipeline.validate()?;
        self.native_allocation_reuse.validate()?;
        if let Some(policy) = self.pinned_weight_upload {
            policy.validate()?;
        }
        ensure!(
            self.pipeline.decoder_batch_policy == ImageDecoderBatchPolicy::SingleImage
                && self.pipeline.edit_prefix_policy == ImageEditPrefixPolicy::Recompute,
            "H3 requires request-local single-image decoding and recomputed bidirectional editing"
        );
        ensure!(
            self.options.noise_profile.as_str() == CUDA_PHILOX_NORMAL_F32_NUMERICAL_PROFILE_ID,
            "Comfy F32 preset requires the explicit native F32 Philox profile"
        );
        ensure!(
            matches!(
                self.options.vae_encoder,
                ferrum_models::p_h3_image_vae::PH3ImageVaeEncoderPreset::CudaF32Fma
            ),
            "CUDA Comfy preset requires its explicit F32/FMA VAE encoder profile"
        );
        i32::try_from(self.ordinal).context("CUDA ordinal exceeds driver range")?;
        ensure!(
            self.bootstrap_memory_bytes > 0 && self.stream_memory_bytes > 0,
            "CUDA bootstrap and per-stream allowances must be positive"
        );
        profile_name(&self.opaque_memory_profile, "CUDA runtime")?;
        profile_name(&self.cudnn.opaque_memory_profile, "cuDNN")?;
        ensure!(
            self.cudnn.library_path.is_absolute()
                && self.cudnn.dependency_libraries.len() <= 16
                && self
                    .cudnn
                    .dependency_libraries
                    .iter()
                    .all(|p| p.is_absolute()),
            "H3 cuDNN paths must be absolute with at most 16 explicit dependencies"
        );
        ensure!(
            self.cudnn.opaque_memory_bytes > 0
                && self.cudnn.workspace_limit_bytes <= isize::MAX as u64,
            "cuDNN needs positive opaque bytes and addressable workspace"
        );
        ensure!(
            self.maximum_in_flight_per_stream > 0
                && self.maximum_commands_per_submission > 0
                && self.maximum_commands_per_submission <= u32::MAX as usize,
            "CUDA flights and submission command limits must be positive and bounded"
        );
        let streams = self.stream_plan()?.resolved_streams();
        ensure!(
            self.native_lane_reuse_maximum_idle_lanes()
                .is_none_or(|maximum| maximum <= streams),
            "native lane reuse exceeds the resolved CUDA stream allowance"
        );
        streams
            .checked_mul(self.maximum_in_flight_per_stream)
            .and_then(|n| n.checked_mul(self.maximum_commands_per_submission))
            .and_then(|n| {
                n.checked_mul(std::mem::size_of::<DeviceCommandEntry<CudaDeviceCommand>>())
            })
            .filter(|&n| n <= isize::MAX as usize)
            .context("CUDA command metadata exceeds address space")?;
        ensure!(
            self.pipeline
                .static_initialization
                .maximum_commands_per_batch()
                <= self.maximum_commands_per_submission,
            "static command batch exceeds CUDA submission limit"
        );
        let staging = usize::try_from(self.pipeline.static_initialization.maximum_staging_bytes())?;
        ensure!(
            staging <= isize::MAX as usize
                && staging <= self.host.class_bytes(HostMemoryClass::TransferStaging),
            "static initialization staging exceeds its Host ceiling"
        );
        ensure!(
            self.host.class_bytes(HostMemoryClass::InputPreprocess) >= 16
                && self.host.class_bytes(HostMemoryClass::Output) > 0
                && self.host.class_bytes(HostMemoryClass::WeightBacking) > 0,
            "H3 needs explicit seed/input, readback and source-repacking Host allowances"
        );
        if let Some(capacity) = self.declared_maximum() {
            self.validate_capacity(capacity.get(), 0)?;
        }
        Ok(())
    }
    pub fn create_pipeline(
        self,
        assets: PH3ImageAssets,
    ) -> Result<PH3ImagePipeline<CudaDeviceRuntime>> {
        self.create_pipeline_with_receipt(assets)
            .map(|(pipeline, _)| pipeline)
    }
    pub fn create_pipeline_with_receipt(
        mut self,
        mut assets: PH3ImageAssets,
    ) -> Result<(
        PH3ImagePipeline<CudaDeviceRuntime>,
        PH3ImageCudaCompositionReceipt,
    )> {
        let visual_expansion = assets.has_prompt_expansion_visual();
        let bf16_expansion = self
            .prompt_expansion_causal
            .as_ref()
            .is_some_and(|p| p.is_bf16());
        ensure!(
            assets
                .expansion
                .as_ref()
                .is_some_and(|e| e.assets.is_bf16())
                == bf16_expansion,
            "causal CUDA precision differs from the prepared model definition"
        );
        if visual_expansion {
            self.prompt_expansion_causal = self
                .prompt_expansion_causal
                .take()
                .map(CudaDenseF16CausalProfile::with_visual_prompt_v1);
        } else {
            ensure!(
                !self
                    .prompt_expansion_causal
                    .as_ref()
                    .is_some_and(CudaDenseF16CausalProfile::is_visual_prompt_v1),
                "visual causal CUDA mode requires validated visual prompt-expansion assets"
            );
        }
        self.validate()?;
        use ferrum_models::p_h3_image::PH3ImageDenoiserAttentionPolicy as DenoiserAttention;
        assets
            .denoiser
            .set_attention_policy(match self.attention_policy {
                ImageCudaAttentionPolicy::Sequential => DenoiserAttention::Sequential,
                ImageCudaAttentionPolicy::CudnnSdpaExperimentalV1 => {
                    DenoiserAttention::CudnnSdpaExperimentalV1
                }
            })?;
        ensure!(
            assets.has_prompt_expansion() == self.prompt_expansion_causal.is_some(),
            "H3 prompt-expansion assets and explicit CUDA causal-module allowance must agree"
        );
        validate_decode_replay_limits(
            assets
                .expansion
                .as_ref()
                .map(|expansion| &expansion.options.engine),
            self.prompt_expansion_causal.as_ref(),
        )?;
        let pairs = operations::selected()?;
        let (selection, mut contracts): (Vec<_>, Vec<_>) = pairs.into_iter().unzip();
        contracts.extend(operations::independent_contracts()?);
        if self.prompt_expansion_causal.is_some() {
            contracts.extend(if bf16_expansion {
                if visual_expansion {
                    cuda_vnext_dense_bf16_visual_causal_contracts()?
                } else {
                    cuda_vnext_dense_bf16_causal_contracts()?
                }
            } else if visual_expansion {
                cuda_vnext_dense_f16_visual_causal_contracts()?
            } else {
                cuda_vnext_dense_f16_causal_contracts()?
            });
        }
        let families = preflight::families(&assets, &self.options)?;
        let largest =
            preflight::validate(&families, &contracts, self.maximum_commands_per_submission)?;
        let minimum_pixels = assets
            .spatial_scale()
            .checked_mul(2)
            .context("H3 minimum output grid overflows")?;
        ensure!(
            u64::from(self.pipeline.maximum_width) >= minimum_pixels
                && u64::from(self.pipeline.maximum_height) >= minimum_pixels,
            "configured output bounds admit no supported even latent grid"
        );
        if let Some(capacity) = self.declared_maximum() {
            self.validate_capacity(capacity.get(), largest)?;
        }
        let stream_plan = self.stream_plan()?;
        let device_id = DeviceId::new(format!("device.cuda.{}", self.ordinal))?;
        let mut device = if self.prompt_expansion_causal.is_some() {
            if bf16_expansion {
                if visual_expansion {
                    cuda_tensor_and_dense_bf16_visual_causal_runtime_config(
                        self.ordinal,
                        device_id,
                    )?
                } else {
                    cuda_tensor_and_dense_bf16_causal_runtime_config(self.ordinal, device_id)?
                }
            } else if visual_expansion {
                cuda_tensor_and_dense_f16_visual_causal_runtime_config(self.ordinal, device_id)?
            } else {
                cuda_tensor_and_dense_f16_causal_runtime_config(self.ordinal, device_id)?
            }
        } else {
            cuda_tensor_runtime_config(self.ordinal, device_id)?
        };
        if self
            .prompt_expansion_causal
            .as_ref()
            .is_some_and(|p| p.decode_replay_limits().is_some())
        {
            for id in [
                DEVICE_REUSABLE_EXECUTION_CAPABILITY_ID,
                DEVICE_ON_DEMAND_REUSABLE_EXECUTION_CAPABILITY_ID,
            ] {
                device.capabilities.insert(CapabilityId::new(id)?);
            }
        }
        let runtime_options = CudaTensorRuntimeOptions {
            native_lane_reuse: self.native_lane_reuse,
            native_allocation_reuse: self.native_allocation_reuse,
            pinned_weight_upload: self.pinned_weight_upload.map(|policy| {
                ferrum_kernels::backend::cuda::vnext_runtime::CudaPinnedWeightUploadConfig {
                    maximum_allocation_bytes: policy.maximum_allocation_bytes,
                    maximum_registered_bytes: policy.maximum_registered_bytes,
                }
            }),
        };
        // Exactly one context, and automatic capacity observes this same one.
        let runtime = match self.memory {
            ImageCudaMemoryPolicy::ExplicitTotal(bytes) => {
                let tensor = CudaTensorRuntimeConfig {
                    device,
                    usable_memory_bytes: bytes.get(),
                    bootstrap_memory_bytes: self.bootstrap_memory_bytes,
                    stream_memory_bytes: self.stream_memory_bytes,
                    maximum_streams: stream_plan.resolved_streams(),
                    maximum_in_flight_per_stream: self.maximum_in_flight_per_stream,
                    maximum_commands_per_submission: self.maximum_commands_per_submission,
                    opaque_memory_profile: self.opaque_memory_profile.clone(),
                };
                match &self.prompt_expansion_causal {
                    Some(causal) => CudaDeviceRuntime::new_tensor_and_causal_eager_with_options(
                        CudaTensorAndCausalRuntimeConfig {
                            tensor,
                            causal: causal.clone(),
                        },
                        runtime_options,
                    )?,
                    None => {
                        CudaDeviceRuntime::new_tensor_eager_with_options(tensor, runtime_options)?
                    }
                }
            }
            ImageCudaMemoryPolicy::AutoAvailable {
                utilization,
                maximum_bytes,
            } => {
                let tensor = CudaTensorRuntimeAutoConfig {
                    device,
                    utilization,
                    maximum_memory_bytes: maximum_bytes.map(NonZeroU64::get),
                    bootstrap_memory_bytes: self.bootstrap_memory_bytes,
                    stream_memory_bytes: self.stream_memory_bytes,
                    maximum_streams: stream_plan.resolved_streams(),
                    maximum_in_flight_per_stream: self.maximum_in_flight_per_stream,
                    maximum_commands_per_submission: self.maximum_commands_per_submission,
                    opaque_memory_profile: self.opaque_memory_profile.clone(),
                };
                match &self.prompt_expansion_causal {
                    Some(causal) => {
                        CudaDeviceRuntime::new_tensor_and_causal_eager_auto_with_options(
                            CudaTensorAndCausalRuntimeAutoConfig {
                                tensor,
                                causal: causal.clone(),
                            },
                            runtime_options,
                        )?
                    }
                    None => CudaDeviceRuntime::new_tensor_eager_auto_with_options(
                        tensor,
                        runtime_options,
                    )?,
                }
            }
        };
        let runtime = Arc::new(runtime);
        let observed_memory = runtime.tensor_memory_observation().copied();
        let capacity = match self.memory {
            ImageCudaMemoryPolicy::ExplicitTotal(bytes) => bytes,
            ImageCudaMemoryPolicy::AutoAvailable { .. } => NonZeroU64::new(
                observed_memory
                    .context("automatic runtime omitted memory observation")?
                    .effective_ceiling_bytes(),
            )
            .context("automatic runtime selected zero capacity")?,
        };
        self.validate_capacity(capacity.get(), largest)?;
        let mut convolution = self.cudnn.clone();
        convolution.workspace_limit_bytes = self.convolution_workspace.resolve(
            self.cudnn.workspace_limit_bytes,
            capacity.get(),
            self.opaque_bytes()?,
            largest,
        )?;
        // This factory loads the union's one native module and adds its one
        // admitted cuDNN F32/FMA convolution owner. Calling a second tensor or
        // standalone Philox factory would acquire a duplicate module permit.
        let mut providers = match self.attention_policy {
            ImageCudaAttentionPolicy::Sequential => {
                cuda_vnext_cnn_f32_providers(&runtime, &convolution, &selection)?
            }
            ImageCudaAttentionPolicy::CudnnSdpaExperimentalV1 => {
                cuda_vnext_cnn_f32_providers_with_segmented_sdpa(
                    &runtime,
                    &convolution,
                    CudnnSdpaPlanPolicy {
                        maximum_plans: 64,
                        maximum_live_bindings: 2048,
                        opaque_memory_bytes: self.attention_policy.additional_opaque_bytes(),
                        workspace_limit_bytes: self.cudnn.workspace_limit_bytes,
                    },
                    &selection,
                )?
            }
        };
        providers.push(cuda_vnext_dense_f32_provider(&runtime)?);
        // Selected affine/patch providers share the union module above.
        // The independent bias-free dense provider still needs only lane BLAS.
        if self.prompt_expansion_causal.is_some() {
            providers.extend(if bf16_expansion {
                if visual_expansion {
                    cuda_vnext_dense_bf16_visual_causal_providers(&runtime)?
                } else {
                    cuda_vnext_dense_bf16_causal_providers(&runtime)?
                }
            } else if visual_expansion {
                cuda_vnext_dense_f16_visual_causal_providers(&runtime)?
            } else {
                cuda_vnext_dense_f16_causal_providers(&runtime)?
            });
        }
        let operation_ids = contracts
            .iter()
            .map(|c| c.descriptor().id.clone())
            .collect();
        let registry = OperationRuntimeRegistry::new(contracts, providers)?;
        let descriptor = runtime.descriptor();
        let materializers = WeightMaterializerRegistry::identity_only()?;
        let engine = EngineProviderDescriptor::new(
            ProviderId::new("provider.engine.cuda.p_h3_image")?,
            ContractVersion::new(1, 0),
            descriptor.runtime_implementation_fingerprint.clone(),
            descriptor.id.clone(),
            descriptor.capabilities.clone(),
        )?;
        let catalog = materializers
            .augment_catalog(registry.capability_catalog(descriptor.clone(), vec![engine])?)?;
        // Use the same compiled artifact-set validation as the existing image
        // factory. No unselected FP8/Marlin provider is manufactured or waived.
        ferrum_kernels::native_ops::validate_compiled_native_operator_provider_catalog_for_units(
            &catalog.native_operator_provider_catalog(ferrum_types::NativeOperatorBackend::Cuda)?,
            ferrum_kernels::native_ops::compiled_native_operator_artifacts(),
            &[],
        )
        .map_err(anyhow::Error::msg)?;
        let receipt = PH3ImageCudaCompositionReceipt {
            affine_f32_numerical_profile: CUDA_AFFINE_LINEAR_F32_ROW_BIAS_NUMERICAL_PROFILE_ID,
            patch_projection_f32_numerical_profile:
                CUDA_PATCH_PROJECTION_F32_ROW_BIAS_NUMERICAL_PROFILE_ID,
            preset: PH3_IMAGE_CUDA_PRESET_ID,
            memory_policy: match self.memory {
                ImageCudaMemoryPolicy::ExplicitTotal(_) => "explicit_total",
                ImageCudaMemoryPolicy::AutoAvailable { .. } => "auto_available",
            },
            stream_plan,
            native_lane_reuse_maximum_idle_lanes: self.native_lane_reuse_maximum_idle_lanes(),
            native_allocation_reuse: self.native_allocation_reuse_receipt(),
            pinned_weight_upload: self.pinned_weight_upload,
            device_capacity_bytes: capacity.get(),
            observed_memory,
            runtime_opaque_profile: self.opaque_memory_profile.clone(),
            prompt_expansion_causal_module_memory_bytes: self
                .prompt_expansion_causal
                .as_ref()
                .map(CudaDenseF16CausalProfile::module_memory_bytes),
            prompt_expansion_causal_opaque_profile: self
                .prompt_expansion_causal
                .as_ref()
                .map(|profile| profile.opaque_memory_profile().to_owned()),
            prompt_expansion_decode_replay_limits: self
                .prompt_expansion_causal
                .as_ref()
                .and_then(CudaDenseF16CausalProfile::decode_replay_limits),
            prompt_expansion_causal_mode: self.prompt_expansion_causal.as_ref().map(|_| {
                if bf16_expansion {
                    if visual_expansion {
                        "qwen3_5.bf16.visual_prompt_v1"
                    } else {
                        "qwen3_5.bf16"
                    }
                } else if visual_expansion {
                    "qwen3_5.f16.visual_prompt_v1"
                } else {
                    "qwen3_5.f16"
                }
            }),
            prompt_expansion_vision_profile: assets
                .prompt_expansion_vision_family()
                .map(|family| family.numerical_profile().id.as_str().to_owned()),
            cudnn_library: self.cudnn.library_path.clone(),
            cudnn_opaque_profile: self.cudnn.opaque_memory_profile.clone(),
            cudnn_convolution_workspace_policy: self.convolution_workspace,
            cudnn_convolution_workspace_limit_bytes: convolution.workspace_limit_bytes,
            attention_policy: self.attention_policy,
            attention_opaque_memory_bytes: self.attention_policy.additional_opaque_bytes(),
            operation_ids,
            largest_component_execution_weight_bytes: largest,
            initial_noise_policy: self.options.initial_noise.identity(),
            vae_decoder_tile_batch: self.options.vae_decoder_tile_batch,
        };
        let opaque = self.opaque_bytes()?;
        let host = if self.pinned_weight_upload.is_some() {
            HostMemoryAccount::with_allocation_alignment(
                self.host,
                CudaDeviceRuntime::pinned_weight_upload_alignment()?,
            )?
        } else {
            HostMemoryAccount::new(self.host)
        };
        let mut pipeline = PH3ImagePipeline::new(
            runtime,
            host,
            registry,
            catalog,
            materializers,
            assets,
            self.pipeline.with_device_capacity(capacity),
            self.options,
        )?;
        pipeline.execution.device_opaque_allowance_bytes = opaque;
        Ok((pipeline, receipt))
    }
    fn declared_maximum(&self) -> Option<NonZeroU64> {
        match self.memory {
            ImageCudaMemoryPolicy::ExplicitTotal(bytes) => Some(bytes),
            ImageCudaMemoryPolicy::AutoAvailable { maximum_bytes, .. } => maximum_bytes,
        }
    }
    fn opaque_bytes(&self) -> Result<u64> {
        self.stream_memory_bytes
            .checked_add(self.prompt_expansion_causal.as_ref().map_or(
                0,
                CudaDenseF16CausalProfile::decode_replay_memory_bytes_per_stream,
            ))
            .and_then(|bytes| bytes.checked_mul(self.stream_plan().ok()?.resolved_streams() as u64))
            .and_then(|n| n.checked_add(self.bootstrap_memory_bytes))
            .and_then(|n| n.checked_add(self.cudnn.opaque_memory_bytes))
            .and_then(|n| n.checked_add(self.attention_policy.additional_opaque_bytes()))
            .and_then(|n| {
                n.checked_add(
                    self.prompt_expansion_causal
                        .as_ref()
                        .map_or(0, CudaDenseF16CausalProfile::module_memory_bytes),
                )
            })
            .and_then(|n| n.checked_add(self.native_allocation_reuse.maximum_idle_bytes()))
            .context("H3 CUDA opaque allowance total overflows")
    }
    fn validate_capacity(&self, capacity: u64, largest: u64) -> Result<()> {
        let storage = capacity
            .checked_sub(self.opaque_bytes()?)
            .filter(|&n| n > 0)
            .context("CUDA opaque allowances leave no component storage")?;
        ensure!(
            largest < storage,
            "largest H3 component weights leave no activation storage"
        );
        if self.convolution_workspace == ImageCudaConvolutionWorkspacePolicy::ConfiguredLimit {
            ensure!(
                self.cudnn.workspace_limit_bytes < storage,
                "cuDNN workspace ceiling leaves no tensor storage"
            );
        }
        Ok(())
    }
}
fn profile(value: &str) -> Result<NumericalProfileId> {
    NumericalProfileId::new(value).map_err(anyhow::Error::msg)
}
fn profile_name(value: &str, label: &str) -> Result<()> {
    ensure!(
        !value.trim().is_empty()
            && value.len() <= 256
            && value.is_ascii()
            && !value.bytes().any(|b| b.is_ascii_control()),
        "{label} requires a bounded ASCII opaque-memory profile"
    );
    Ok(())
}

#[cfg(test)]
mod tests;
