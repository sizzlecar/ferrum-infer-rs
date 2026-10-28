//! Metal composition for the shared managed image pipeline.
use super::*;
use anyhow::{ensure, Context, Result};
use ferrum_kernels::backend::metal::{
    vnext_ops::tensor::{
        metal_tensor_providers, metal_tensor_runtime_config, MetalTensorOperation,
    },
    vnext_runtime::{
        MetalDeviceRuntime, MetalTensorMemoryObservation, MetalTensorRuntimeAutoConfig,
        MetalTensorRuntimeConfig,
    },
};
use ferrum_models::qwen3_vl_text::{Qwen3VLTextKeyRun, Qwen3VLTextLayout};
use ferrum_models::qwen3_vl_vision::Qwen3VLVisionFamily;
use ferrum_models::qwen_image21::QwenImage21VaeEncoderFamily;
use std::num::NonZeroU64;

#[derive(Debug, Clone, Copy)]
pub enum ImageMetalMemoryPolicy {
    ExplicitTotal(NonZeroU64),
    /// Fraction of the recommended working set, bounded by its remaining
    /// process allowance and an optional caller ceiling; not system free RAM.
    AutoRecommended {
        utilization: f64,
        maximum_bytes: Option<NonZeroU64>,
    },
}

#[derive(Debug, Clone)]
pub struct QwenImage21MetalConfig {
    pub memory: ImageMetalMemoryPolicy,
    pub precision_policy: ImagePrecisionPolicy,
    pub attention_policy: ImageMetalAttentionPolicy,
    pub bootstrap_memory_bytes: u64,
    pub stream_memory_bytes: u64,
    pub maximum_streams: ImageStreamLimit,
    pub maximum_in_flight_per_stream: usize,
    pub maximum_commands_per_submission: usize,
    pub opaque_memory_profile: String,
    pub host: HostMemoryLimits,
    pub pipeline: ImagePipelineLimits,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct ImageMetalCompositionReceipt {
    pub memory_policy: &'static str,
    pub precision_policy: ImagePrecisionPolicy,
    pub attention_policy: ImageMetalAttentionPolicy,
    pub stream_plan: ImageStreamPlan,
    pub device_capacity_bytes: u64,
    pub observed_memory: Option<MetalTensorMemoryObservation>,
    pub required_component_commands: usize,
    pub runtime_opaque_profile: String,
    pub native_block_denoiser: bool,
    /// Physical execution schemas, including FP8 scales; not peak residency.
    pub execution_weight_bytes: [u64; 3],
}

struct PreparedComponents {
    commands: usize,
    text: Qwen3VLTextConfig,
    denoiser: QwenImage21TransformerConfig,
    families: Vec<PreparedModelFamily>,
}

impl QwenImage21MetalConfig {
    pub fn stream_plan(&self) -> Result<ImageStreamPlan> {
        self.maximum_streams
            .resolve(ImageStreamRequirements::for_serial_pipeline(
                self.pipeline.static_initialization,
                self.pipeline.executor,
            ))
    }
    fn explicit(&self, capacity: u64) -> Result<MetalTensorRuntimeConfig> {
        Ok(MetalTensorRuntimeConfig {
            device: metal_tensor_runtime_config(DeviceId::new("device.metal.0")?)?,
            usable_memory_bytes: capacity,
            bootstrap_memory_bytes: self.bootstrap_memory_bytes,
            stream_memory_bytes: self.stream_memory_bytes,
            maximum_streams: self.stream_plan()?.resolved_streams(),
            maximum_in_flight_per_stream: self.maximum_in_flight_per_stream,
            maximum_commands_per_submission: self.maximum_commands_per_submission,
            opaque_memory_profile: self.opaque_memory_profile.clone(),
        })
    }
    fn automatic(
        &self,
        utilization: f64,
        maximum: Option<NonZeroU64>,
    ) -> Result<MetalTensorRuntimeAutoConfig> {
        let explicit = self.explicit(maximum.map_or(u64::MAX, NonZeroU64::get))?;
        Ok(MetalTensorRuntimeAutoConfig {
            device: explicit.device,
            utilization,
            maximum_memory_bytes: maximum.map(NonZeroU64::get),
            bootstrap_memory_bytes: explicit.bootstrap_memory_bytes,
            stream_memory_bytes: explicit.stream_memory_bytes,
            maximum_streams: explicit.maximum_streams,
            maximum_in_flight_per_stream: explicit.maximum_in_flight_per_stream,
            maximum_commands_per_submission: explicit.maximum_commands_per_submission,
            opaque_memory_profile: explicit.opaque_memory_profile,
        })
    }
    pub fn validate(&self) -> Result<()> {
        self.pipeline.validate()?;
        ensure!(
            !self.precision_policy.uses_fp8()
                || self.host.class_bytes(HostMemoryClass::WeightBacking) > 0,
            "Metal FP8 conversion requires a nonzero host weight backing allowance"
        );
        if self.pipeline.edit_prefix_policy == ImageEditPrefixPolicy::RequestCache {
            let contracts = operation_contracts(self.attention_policy)?;
            let device = metal_tensor_runtime_config(DeviceId::new("device.metal.0")?)?;
            validate_request_cache_support(
                &contracts,
                &device.capabilities,
                self.attention_policy,
            )?;
        }
        match self.memory {
            ImageMetalMemoryPolicy::ExplicitTotal(capacity) => {
                self.explicit(capacity.get())?.validate()?
            }
            ImageMetalMemoryPolicy::AutoRecommended {
                utilization,
                maximum_bytes,
            } => self.automatic(utilization, maximum_bytes)?.validate()?,
        }
        ensure!(
            self.host.class_bytes(HostMemoryClass::InputPreprocess) >= 16
                && self.host.class_bytes(HostMemoryClass::Output) > 0,
            "Metal needs explicit seed/input and output Host capacity"
        );
        ensure!(
            self.pipeline
                .static_initialization
                .maximum_commands_per_batch()
                <= self.maximum_commands_per_submission,
            "static initialization command batch exceeds Metal submission limit"
        );
        Ok(())
    }
    fn prepare(
        &self,
        assets: &QwenImage21Assets,
        profiles: &QwenImage21Profiles,
        edit_profiles: &QwenImage21EditProfiles,
    ) -> Result<PreparedComponents> {
        let minimum = assets
            .spatial_scale()?
            .checked_mul(2)
            .context("minimum image size overflows")?;
        ensure!(
            self.pipeline.maximum_width >= minimum && self.pipeline.maximum_height >= minimum,
            "image limits admit no supported even latent grid"
        );
        let text = Qwen3VLTextConfig::new(
            assets.text.clone(),
            Qwen3VLTextLayout::new(1, vec![vec![Qwen3VLTextKeyRun::new(1, true)]])?,
        )?;
        let dit = QwenImage21TransformerConfig::new(
            assets.transformer.clone(),
            QwenImage21T2iLayout::new(1, 2, 2, vec![1])?,
        )?
        .with_weight_encodings(assets.transformer_encodings.clone())?;
        let mut families = vec![
            TypedFamilyRegistration::new(Qwen3VLTextEncoderFamily::new()?)
                .prepare_with_profile(&serde_json::to_value(&text)?, &profiles.text_encoder)?,
            TypedFamilyRegistration::new(NormalNoiseFamily::new(profiles.noise.as_str())?)
                .prepare_with_profile(
                    &serde_json::to_value(NormalNoiseConfig::new(64, 2, 2)?)?,
                    &profiles.noise,
                )?,
            TypedFamilyRegistration::new(QwenImage21TransformerFamily::new()?)
                .prepare_with_profile(&serde_json::to_value(&dit)?, &profiles.denoiser)?,
            TypedFamilyRegistration::new(QwenImage21VaeDecoderFamily::new()?)
                .prepare_with_profile(
                    &serde_json::to_value(assets.decoder(2, 2)?)?,
                    &profiles.decoder,
                )?,
        ];
        let mut prefix_commands = 0;
        if let Some(edit) = &assets.editing {
            families.extend(super::composition::editing_command_families(
                &text,
                &dit,
                &edit.vision,
                &edit.encoder,
                profiles,
                edit_profiles,
            )?);
            if self.pipeline.edit_prefix_policy == ImageEditPrefixPolicy::RequestCache {
                let (prefix, copies) = super::composition::prefix_command_families(&dit, profiles)?;
                prefix_commands = super::composition::validate_prefix_command_limit(
                    &prefix[0],
                    copies,
                    self.maximum_commands_per_submission,
                )?;
                families.extend(prefix);
            }
        }
        super::composition::validate_family_command_limits(
            &families,
            self.maximum_commands_per_submission,
        )?;
        let commands = families
            .iter()
            .map(super::composition::component_command_count)
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .max()
            .context("missing image components")?
            .max(prefix_commands);
        // Check the selected real programs before creating the device. A wire
        // policy cannot substitute a missing operation or advertise its capability.
        let contracts = operation_contracts(self.attention_policy)?;
        let device = metal_tensor_runtime_config(DeviceId::new("device.metal.0")?)?;
        validate_family_support(&families, &contracts, &device.capabilities)?;
        Ok(PreparedComponents {
            commands,
            text,
            denoiser: dit,
            families,
        })
    }
    pub fn create_pipeline(
        self,
        assets: QwenImage21Assets,
    ) -> Result<QwenImage21Pipeline<MetalDeviceRuntime>> {
        self.create_pipeline_with_receipt(assets)
            .map(|(pipeline, _)| pipeline)
    }
    pub fn create_pipeline_with_receipt(
        self,
        assets: QwenImage21Assets,
    ) -> Result<(
        QwenImage21Pipeline<MetalDeviceRuntime>,
        ImageMetalCompositionReceipt,
    )> {
        ensure!(
            !assets.has_lora(),
            "image LoRA is currently qualified only for CUDA"
        );
        self.validate()?;
        let native_block_denoiser = !assets.transformer_encodings.is_empty();
        self.precision_policy
            .validate_denoiser_source(native_block_denoiser)?;
        let profiles = profiles(
            native_block_denoiser,
            self.precision_policy,
            self.attention_policy,
        )?;
        let editing_profiles = editing_profiles()?;
        let components = self.prepare(&assets, &profiles, &editing_profiles)?;
        let required_component_commands = components.commands;
        let stream_plan = self.stream_plan()?;
        let (runtime, observed_memory) = match self.memory {
            ImageMetalMemoryPolicy::ExplicitTotal(bytes) => (
                MetalDeviceRuntime::new_tensor_eager(self.explicit(bytes.get())?)?,
                None,
            ),
            ImageMetalMemoryPolicy::AutoRecommended {
                utilization,
                maximum_bytes,
            } => {
                let (runtime, observation) = MetalDeviceRuntime::new_tensor_eager_auto(
                    self.automatic(utilization, maximum_bytes)?,
                )?;
                (runtime, Some(observation))
            }
        };
        let runtime = Arc::new(runtime);
        let capacity = runtime
            .tensor_capacity_bytes()
            .context("Metal runtime has no admitted tensor capacity")?;
        let opaque = self
            .stream_memory_bytes
            .checked_mul(stream_plan.resolved_streams() as u64)
            .and_then(|v| v.checked_add(self.bootstrap_memory_bytes))
            .context("Metal opaque allowance overflows")?;
        let storage = capacity
            .checked_sub(opaque)
            .context("Metal allowances exceed capacity")?;
        let host = runtime.tensor_host_account(self.host)?;
        let mut execution_weight_bytes = assets.receipt().required_weight_bytes;
        let mut selected: Vec<Box<dyn WeightMaterializer>> = Vec::new();
        for (index, stage, family, materializer_id) in [
            (
                0,
                ImageStage::TextEncoder,
                &components.families[0],
                precision::TEXT_MATERIALIZER,
            ),
            (
                1,
                ImageStage::Denoiser,
                &components.families[2],
                precision::DIT_MATERIALIZER,
            ),
        ] {
            if !self.precision_policy.stage_uses_fp8(stage) {
                continue;
            }
            let groups = match stage {
                ImageStage::TextEncoder => {
                    Qwen3VLTextEncoderFamily::fp8_weight_groups(&components.text)?
                }
                ImageStage::Denoiser => {
                    QwenImage21TransformerFamily::fp8_weight_groups(&components.denoiser)?
                }
                _ => unreachable!("only materialized stages are enumerated"),
            };
            let materializer = precision::materializer(materializer_id, &host, groups)?;
            precision::validate_backing(family, &materializer, &host)?;
            execution_weight_bytes[index] = materializer.execution_weight_bytes(family)?;
            selected.push(Box::new(materializer));
        }
        let materializers = WeightMaterializerRegistry::new(selected)?;
        for bytes in execution_weight_bytes
            .into_iter()
            .chain(assets.editing_weight_bytes().into_iter().flatten())
        {
            ensure!(bytes < storage, "component weight bytes {bytes} leave no activation capacity under Metal storage ceiling {storage}");
        }
        let contracts = operation_contracts(self.attention_policy)?
            .into_iter()
            .map(|contract| Box::new(contract) as Box<dyn OperationContract>)
            .collect();
        let registry = OperationRuntimeRegistry::new(
            contracts,
            metal_tensor_providers(&runtime, &operation_kinds(self.attention_policy))?,
        )?;
        let descriptor = runtime.descriptor();
        let engine = EngineProviderDescriptor::new(
            ProviderId::new("provider.engine.metal.image_generation")?,
            ContractVersion::new(1, 0),
            descriptor.runtime_implementation_fingerprint.clone(),
            descriptor.id.clone(),
            descriptor.capabilities.clone(),
        )?;
        let catalog = materializers
            .augment_catalog(registry.capability_catalog(descriptor.clone(), vec![engine])?)?;
        let editing = assets.editing.is_some();
        let mut pipeline = QwenImage21Pipeline::new(
            runtime,
            host,
            registry,
            catalog,
            materializers,
            assets,
            profiles,
            self.pipeline
                .with_device_capacity(NonZeroU64::new(capacity).context("zero Metal capacity")?),
        )?
        .with_device_opaque_allowance(opaque)?;
        if editing {
            pipeline = pipeline.with_editing_profiles(editing_profiles)?;
        }
        Ok((
            pipeline,
            ImageMetalCompositionReceipt {
                precision_policy: self.precision_policy,
                attention_policy: self.attention_policy,
                memory_policy: if observed_memory.is_some() {
                    "auto_recommended"
                } else {
                    "explicit_total"
                },
                stream_plan,
                device_capacity_bytes: capacity,
                observed_memory,
                required_component_commands,
                runtime_opaque_profile: self.opaque_memory_profile,
                native_block_denoiser,
                execution_weight_bytes,
            },
        ))
    }
}

fn operation_kinds(policy: ImageMetalAttentionPolicy) -> Vec<MetalTensorOperation> {
    let mut kinds = MetalTensorOperation::ALL.to_vec();
    if policy == ImageMetalAttentionPolicy::FusedDense {
        kinds.extend([
            MetalTensorOperation::StructuredAttentionFused,
            MetalTensorOperation::CrossAttentionFused,
        ]);
    }
    kinds
}

fn operation_contracts(
    policy: ImageMetalAttentionPolicy,
) -> Result<Vec<StandardOperationContract>> {
    operation_kinds(policy)
        .into_iter()
        .map(|kind| kind.contract().map_err(Into::into))
        .collect()
}

fn validate_request_cache_support(
    contracts: &[StandardOperationContract],
    capabilities: &std::collections::BTreeSet<CapabilityId>,
    policy: ImageMetalAttentionPolicy,
) -> Result<()> {
    let required = match policy {
        ImageMetalAttentionPolicy::Reference => CROSS_ATTENTION_BF16_OPERATION_ID,
        ImageMetalAttentionPolicy::FusedDense => CROSS_ATTENTION_FUSED_BF16_OPERATION_ID,
    };
    let contract = contracts
        .iter()
        .find(|contract| contract.descriptor().id.as_str() == required)
        .context("Metal request-cache editing requires the cross-attention provider")?;
    ensure!(
        contract
            .descriptor()
            .provider
            .required_capabilities
            .is_subset(capabilities),
        "Metal request-cache editing requires declared cross-attention capabilities"
    );
    Ok(())
}

fn validate_family_support(
    families: &[PreparedModelFamily],
    contracts: &[StandardOperationContract],
    capabilities: &std::collections::BTreeSet<CapabilityId>,
) -> Result<()> {
    for contract in contracts {
        let descriptor = contract.descriptor();
        ensure!(
            descriptor
                .provider
                .required_capabilities
                .is_subset(capabilities),
            "Metal operation {} requires undeclared device capabilities",
            descriptor.id
        );
    }
    for family in families {
        for node in family
            .program()
            .blocks()
            .iter()
            .flat_map(|block| &block.nodes)
        {
            ensure!(
                contracts.iter().any(|contract| {
                    let descriptor = contract.descriptor();
                    descriptor.id == node.operation_id
                        && descriptor.version.satisfies(node.required_version)
                }),
                "Metal family {} requires unavailable operation {} version {}",
                family.family_id(),
                node.operation_id,
                node.required_version
            );
        }
    }
    Ok(())
}

fn profiles(
    native_block: bool,
    precision: ImagePrecisionPolicy,
    attention: ImageMetalAttentionPolicy,
) -> Result<QwenImage21Profiles> {
    let id = |name| NumericalProfileId::new(name).map_err(anyhow::Error::msg);
    Ok(QwenImage21Profiles {
        text_encoder: id(if precision.stage_uses_fp8(ImageStage::TextEncoder) {
            Qwen3VLTextEncoderFamily::METAL_FP8_PROFILE_ID
        } else {
            Qwen3VLTextEncoderFamily::METAL_PROFILE_ID
        })?,
        noise: id(MetalTensorOperation::PhiloxNormal.profile())?,
        denoiser: id(
            if attention == ImageMetalAttentionPolicy::FusedDense && native_block {
                QwenImage21TransformerFamily::METAL_NATIVE_BLOCK_FUSED_PROFILE_ID
            } else if attention == ImageMetalAttentionPolicy::FusedDense
                && precision.stage_uses_fp8(ImageStage::Denoiser)
            {
                QwenImage21TransformerFamily::METAL_FP8_FUSED_PROFILE_ID
            } else if attention == ImageMetalAttentionPolicy::FusedDense {
                QwenImage21TransformerFamily::METAL_FUSED_PROFILE_ID
            } else if native_block {
                QwenImage21TransformerFamily::METAL_NATIVE_BLOCK_PROFILE_ID
            } else if precision.stage_uses_fp8(ImageStage::Denoiser) {
                QwenImage21TransformerFamily::METAL_FP8_PROFILE_ID
            } else {
                QwenImage21TransformerFamily::METAL_PROFILE_ID
            },
        )?,
        decoder: id(QwenImage21VaeDecoderFamily::METAL_PROFILE_ID)?,
        euler: Bf16EulerProfile::CudaScalarBf16StagedRneV1,
    })
}
fn editing_profiles() -> Result<QwenImage21EditProfiles> {
    Ok(QwenImage21EditProfiles {
        vision: NumericalProfileId::new(Qwen3VLVisionFamily::METAL_PROFILE_ID)
            .map_err(anyhow::Error::msg)?,
        encoder: NumericalProfileId::new(QwenImage21VaeEncoderFamily::METAL_PROFILE_ID)
            .map_err(anyhow::Error::msg)?,
    })
}

#[cfg(test)]
mod metadata_tests {
    use super::*;

    // No Metal device is opened: these are real selected model programs and
    // checked-in operation descriptors, not a declaration that kernels ran.
    #[test]
    fn metal_request_cache_preflight_requires_registered_operations_and_capabilities() {
        let device = metal_tensor_runtime_config(DeviceId::new("device.metal.0").unwrap()).unwrap();

        let architecture = QwenImage21Config::from_hf_json(
            &serde_json::to_vec(&serde_json::json!({
                "_class_name":"QwenImage21Transformer2DModel",
                "num_layers":2,"num_attention_heads":1,"attention_head_dim":8,
                "context_in_dim":8,"in_channels":64,"out_channels":64,
                "patch_size":1,"mlp_ratio":2,"axes_dims_rope":[2,2,4],
                "causal_condition":true,"eps":0.000001
            }))
            .unwrap(),
            StructureLimits::default(),
        )
        .unwrap();
        let config = QwenImage21TransformerConfig::new(
            architecture,
            QwenImage21T2iLayout::new(1, 2, 2, vec![1]).unwrap(),
        )
        .unwrap();
        for (policy, cross_operation, cross_capability) in [
            (
                ImageMetalAttentionPolicy::Reference,
                CROSS_ATTENTION_BF16_OPERATION_ID,
                CROSS_ATTENTION_BF16_CAPABILITY_ID,
            ),
            (
                ImageMetalAttentionPolicy::FusedDense,
                CROSS_ATTENTION_FUSED_BF16_OPERATION_ID,
                CROSS_ATTENTION_FUSED_BF16_CAPABILITY_ID,
            ),
        ] {
            let contracts = operation_contracts(policy).unwrap();
            validate_request_cache_support(&contracts, &device.capabilities, policy).unwrap();
            let mut missing_capability = device.capabilities.clone();
            assert!(missing_capability.remove(&CapabilityId::new(cross_capability).unwrap()));
            assert!(
                validate_request_cache_support(&contracts, &missing_capability, policy).is_err()
            );
            for precision in [
                ImagePrecisionPolicy::Bf16,
                ImagePrecisionPolicy::Fp8LinearExperimentalV1,
            ] {
                let selected_profiles = profiles(false, precision, policy).unwrap();
                let reference_profiles =
                    profiles(false, precision, ImageMetalAttentionPolicy::Reference).unwrap();
                assert_eq!(
                    selected_profiles.text_encoder,
                    reference_profiles.text_encoder
                );
                assert_eq!(selected_profiles.decoder, reference_profiles.decoder);
                let (families, _) =
                    super::super::composition::prefix_command_families(&config, &selected_profiles)
                        .unwrap();
                assert!(families
                    .iter()
                    .flat_map(|family| family.program().blocks())
                    .flat_map(|block| &block.nodes)
                    .any(|node| node.operation_id.as_str() == cross_operation));
                validate_family_support(&families, &contracts, &device.capabilities).unwrap();
                assert!(
                    validate_family_support(&families, &contracts, &missing_capability).is_err()
                );
                let without_concat: Vec<_> = operation_contracts(policy)
                    .unwrap()
                    .into_iter()
                    .filter(|contract| {
                        contract.descriptor().id.as_str() != BATCHED_ROW_CONCAT_BF16_OPERATION_ID
                    })
                    .collect();
                assert!(
                    validate_family_support(&families, &without_concat, &device.capabilities)
                        .is_err()
                );
            }
            let without_cross: Vec<_> = contracts
                .into_iter()
                .filter(|contract| contract.descriptor().id.as_str() != cross_operation)
                .collect();
            assert!(
                validate_request_cache_support(&without_cross, &device.capabilities, policy)
                    .is_err()
            );
        }
    }
}
