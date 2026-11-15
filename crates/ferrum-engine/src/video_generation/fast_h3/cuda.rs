//! One CUDA composition for every FastH3 stage and transient batch shape.
use anyhow::{ensure, Context, Result};
use ferrum_execution::component::TensorComponentPolicy;
use ferrum_kernels::{
    backend::cuda::{vnext_ops::*, vnext_runtime::*},
    native_ops::sage_vsa_fp4::{CheckedVsaRegions, SageVsaFp4Config},
};
use ferrum_runtime::*;
use serde::{Deserialize, Serialize};
use std::{
    num::{NonZeroU64, NonZeroUsize},
    path::PathBuf,
    sync::Arc,
};
mod operations;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FastH3CudaConfig {
    pub device: usize,
    pub host_bytes: usize,
    pub device_bytes: u64,
    pub bootstrap_bytes: u64,
    pub stream_bytes: u64,
    pub maximum_commands_per_submission: usize,
    /// The product executes each component on one serial lane. Declaring this
    /// in the plan lets Core reuse branch temporaries with explicit ordering.
    #[serde(default = "serial_execution_order")]
    pub execution_order: PlanExecutionOrder,
    pub staging_bytes: usize,
    /// Physical backing threshold for optional pinned weight uploads. Zero
    /// preserves pinning of every eligible upload; native metadata stays pinned.
    #[serde(default)]
    pub minimum_pinned_weight_upload_bytes: usize,
    /// Optional temporary private-pool allocation after cold initialization.
    /// Its startup cost is explicit; insufficient capacity skips preparation.
    /// Later tensors retain ordinary admission and may still wait for backing.
    #[serde(default)]
    pub initial_allocation_cache_bytes: Option<NonZeroU64>,
    pub native_bytes: u64,
    pub cudnn_library: PathBuf,
    pub cudnn_dependencies: Vec<PathBuf>,
    pub cudnn_opaque_bytes: u64,
    pub workspace_bytes: u64,
    pub plan_cache_bytes: u64,
    pub maximum_native_plans: usize,
    pub maximum_native_bindings: usize,
}
impl FastH3CudaConfig {
    /// Explicit first deployment preset; library selection remains the caller's.
    pub fn rtx5090(cudnn_library: PathBuf, cudnn_dependencies: Vec<PathBuf>) -> Self {
        Self {
            device: 0,
            host_bytes: 4 << 30,
            device_bytes: 28 << 30,
            bootstrap_bytes: 1 << 30,
            stream_bytes: 16 << 20,
            // The pipeline admits graphs with up to 8192 nodes. Command
            // records also include uploads and workspace initialization.
            // This Host record bound does not enlarge the device budget.
            maximum_commands_per_submission: 16384,
            execution_order: PlanExecutionOrder::Serial,
            staging_bytes: 8 << 20,
            minimum_pinned_weight_upload_bytes: 0,
            initial_allocation_cache_bytes: None,
            native_bytes: 64 << 20,
            cudnn_library,
            cudnn_dependencies,
            cudnn_opaque_bytes: 512 << 20,
            workspace_bytes: 256 << 20,
            plan_cache_bytes: 512 << 20,
            maximum_native_plans: 128,
            maximum_native_bindings: 1024,
        }
    }
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.initial_allocation_cache_bytes.is_none_or(|bytes| {
                bytes.get() < self.device_bytes && bytes.get() < isize::MAX as u64
            }),
            "initial CUDA allocation cache must be smaller than the device budget"
        );
        ensure!(
            self.host_bytes > 0
                && self.host_bytes < isize::MAX as usize
                && self.staging_bytes > 0
                && self.staging_bytes <= self.host_bytes / 2
                && self.minimum_pinned_weight_upload_bytes <= self.staging_bytes
                && self.maximum_commands_per_submission > 0
                && self.maximum_native_plans > 0
                && self.maximum_native_bindings > 0,
            "invalid FastH3 Host, command or native plan capacity"
        );
        ensure!(
            [
                self.device_bytes,
                self.bootstrap_bytes,
                self.stream_bytes,
                self.native_bytes,
                self.cudnn_opaque_bytes,
                self.plan_cache_bytes
            ]
            .into_iter()
            .all(|n| n > 0 && n < isize::MAX as u64)
                && self.workspace_bytes < isize::MAX as u64,
            "invalid FastH3 device/native capacity"
        );
        ensure!(
            self.cudnn_library.is_absolute()
                && self.cudnn_dependencies.iter().all(|p| p.is_absolute()),
            "cuDNN libraries require explicit absolute paths"
        );
        Ok(())
    }
}

/// These are the original runtime, account and provider objects. Registries
/// differ only by their immutable VSA region interpretation, never by a new
/// device allocator or an additional NVFP4/PTX/cuDNN module installation.
pub struct FastH3CudaComposition {
    pub runtime: Arc<CudaDeviceRuntime>,
    pub host: HostMemoryAccount,
    pub policy: TensorComponentPolicy,
    /// The existing device ledger, observed without claiming additional bytes.
    pub capacity: DeviceCapacityOwner,
    pub device_bytes: u64,
    /// An initialization lane can overlap the worker's retained execution lane.
    pub initialization_stream_bytes: u64,
    pub materializers: WeightMaterializerRegistry,
    providers: Vec<Arc<dyn OperationProvider<CudaDeviceRuntime>>>,
    sage: SageVsaFp4Config,
}
impl FastH3CudaComposition {
    pub fn new(c: &FastH3CudaConfig) -> Result<Self> {
        c.validate()?;
        let mut device = cuda_tensor_runtime_config(
            c.device,
            DeviceId::new(format!("device.cuda.{}", c.device))?,
        )?;
        cuda_tensor_runtime_enable_cutlass_nvfp4(&mut device)?;
        cuda_tensor_runtime_enable_sage_vsa_fp4_regions(&mut device)?;
        cuda_tensor_runtime_enable_sage_vsa_fp4_quantize(&mut device)?;
        let runtime = Arc::new(CudaDeviceRuntime::new_tensor_eager_with_options(
            CudaTensorRuntimeConfig {
                device,
                usable_memory_bytes: c.device_bytes,
                bootstrap_memory_bytes: c.bootstrap_bytes,
                stream_memory_bytes: c.stream_bytes,
                // The worker retains one serial lane while initialization uses
                // another. There is still only one inference worker.
                maximum_streams: 2,
                maximum_in_flight_per_stream: 1,
                maximum_commands_per_submission: c.maximum_commands_per_submission,
                opaque_memory_profile: "fast_h3.product.cuda.v1".into(),
            },
            CudaTensorRuntimeOptions {
                pinned_weight_upload: Some(CudaPinnedWeightUploadConfig {
                    minimum_allocation_bytes: c.minimum_pinned_weight_upload_bytes,
                    maximum_allocation_bytes: NonZeroUsize::new(c.staging_bytes)
                        .context("empty staging")?,
                    maximum_registered_bytes: NonZeroUsize::new(
                        c.staging_bytes.checked_mul(2).context("staging overflow")?,
                    )
                    .context("empty registration")?,
                }),
                ..Default::default()
            },
        )?);
        let host = HostMemoryAccount::with_allocation_alignment(
            HostMemoryLimits::new(
                c.host_bytes,
                HostMemoryClassLimits {
                    input_preprocess_bytes: c.host_bytes,
                    transfer_staging_bytes: c.host_bytes,
                    output_bytes: c.host_bytes,
                    weight_backing_bytes: c.host_bytes,
                },
            )?,
            CudaDeviceRuntime::pinned_weight_upload_alignment()?,
        )?;
        let vendor = [
            CudaCudnnOperation::Conv2dBf16,
            CudaCudnnOperation::Conv2dF32,
            CudaCudnnOperation::Conv3dF32,
            CudaCudnnOperation::ConvTranspose2dF32,
            CudaCudnnOperation::StructuredAttentionBf16,
            CudaCudnnOperation::SegmentedAttentionBf16,
            CudaCudnnOperation::CrossAttentionBf16,
            CudaCudnnOperation::CausalGqaBf16,
        ];
        let mut providers = cuda_vnext_tensor_providers_with_cudnn_selection(
            &runtime,
            &CudaCudnnConfig {
                library_path: c.cudnn_library.clone(),
                dependency_libraries: c.cudnn_dependencies.clone(),
                opaque_memory_bytes: c.cudnn_opaque_bytes,
                workspace_limit_bytes: c.workspace_bytes,
                opaque_memory_profile: "fast_h3.product.cudnn.v1".into(),
            },
            Some(CudnnSdpaPlanPolicy {
                maximum_plans: c.maximum_native_plans,
                maximum_live_bindings: c.maximum_native_bindings,
                opaque_memory_bytes: c.plan_cache_bytes,
                workspace_limit_bytes: c.workspace_bytes,
            }),
            &vendor,
            operations::ORDINARY,
        )?;
        providers.push(cuda_vnext_dense_bf16_provider(&runtime)?);
        let sage = SageVsaFp4Config {
            opaque_memory_bytes: c.native_bytes,
            opaque_memory_profile: "fast_h3.product.sage.v1".into(),
        };
        providers.push(cuda_vnext_sparse_fp4_qkv_quantize_bf16_provider(
            &runtime, &sage,
        )?);
        providers.extend(cuda_vnext_nvfp4_bf16_providers(
            &runtime,
            &CutlassNvfp4Config {
                opaque_memory_bytes: c.native_bytes,
                opaque_memory_profile: "fast_h3.product.nvfp4.v1".into(),
            },
        )?);
        let policy = TensorComponentPolicy::new(
            c.device_bytes,
            runtime
                .descriptor()
                .dynamic_storage_profiles
                .iter()
                .copied()
                .collect(),
        )?
        .with_execution_order(c.execution_order);
        let capacity = DeviceCapacityOwner::open(runtime.descriptor(), c.device_bytes)?;
        Ok(Self {
            capacity,
            device_bytes: c.device_bytes,
            initialization_stream_bytes: c.stream_bytes,
            runtime,
            host,
            policy,
            materializers: WeightMaterializerRegistry::identity_only()?,
            providers: providers.into_iter().map(Arc::from).collect(),
            sage,
        })
    }
    pub fn registry(
        &self,
        regions: Option<Arc<CheckedVsaRegions>>,
    ) -> Result<Arc<OperationRuntimeRegistry<CudaDeviceRuntime>>> {
        let with_regions = regions.is_some();
        let mut providers = self.providers.clone();
        if let Some(regions) = regions {
            providers.push(Arc::from(
                cuda_vnext_sparse_fp4_attention_region_topk_provider(
                    &self.runtime,
                    &self.sage,
                    regions,
                )?,
            ));
        }
        Ok(Arc::new(OperationRuntimeRegistry::from_shared_providers(
            operations::contracts(with_regions)?,
            providers,
        )?))
    }
    pub fn catalog(
        &self,
        registry: &OperationRuntimeRegistry<CudaDeviceRuntime>,
    ) -> Result<CapabilityCatalog> {
        let d = self.runtime.descriptor();
        let engine = EngineProviderDescriptor::new(
            ProviderId::new("provider.engine.fast_h3.product")?,
            ContractVersion::new(1, 0),
            d.runtime_implementation_fingerprint.clone(),
            d.id.clone(),
            d.capabilities.clone(),
        )?;
        Ok(self
            .materializers
            .augment_catalog(registry.capability_catalog(d.clone(), vec![engine])?)?)
    }
}

fn serial_execution_order() -> PlanExecutionOrder {
    PlanExecutionOrder::Serial
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn initial_allocation_cache_is_explicit_bounded_and_round_trips() {
        let mut config = FastH3CudaConfig::rtx5090("/fixture/libcudnn.so".into(), vec![]);
        assert_eq!(config.initial_allocation_cache_bytes, None);
        config.initial_allocation_cache_bytes = NonZeroU64::new(25 << 30);
        config.validate().unwrap();
        let value = serde_json::to_value(&config).unwrap();
        let decoded: FastH3CudaConfig = serde_json::from_value(value.clone()).unwrap();
        assert_eq!(
            decoded.initial_allocation_cache_bytes,
            config.initial_allocation_cache_bytes
        );
        let mut invalid = value;
        invalid["initial_allocation_cache_bytes"] = serde_json::json!(0);
        assert!(serde_json::from_value::<FastH3CudaConfig>(invalid).is_err());
        config.initial_allocation_cache_bytes = NonZeroU64::new(config.device_bytes);
        assert!(config.validate().is_err());
    }
    #[test]
    #[ignore = "requires CUDA SM120, pinned native operator lock and explicit cuDNN libraries; validates shared provider installation, not inference"]
    fn shared_cuda_composition_initializes_all_stage_provider_owners() {
        let library = std::env::var_os("FERRUM_TEST_CUDNN_LIBRARY")
            .expect("explicit test cuDNN library")
            .into();
        let dependencies = std::env::var_os("FERRUM_TEST_CUDNN_DEPENDENCIES")
            .map(|value| std::env::split_paths(&value).collect())
            .unwrap_or_default();
        let config = FastH3CudaConfig::rtx5090(library, dependencies);
        let composition = FastH3CudaComposition::new(&config).unwrap();
        let registry = composition.registry(None).unwrap();
        let _catalog = composition.catalog(&registry).unwrap();
        let actor = ferrum_execution::TensorSerialActor::new(composition.runtime.clone()).unwrap();
        assert!(!actor.is_failed());
        assert_eq!(composition.host.usage().unwrap().committed_bytes(), 0);
        drop((actor, registry));
        drop(composition);
    }
}
