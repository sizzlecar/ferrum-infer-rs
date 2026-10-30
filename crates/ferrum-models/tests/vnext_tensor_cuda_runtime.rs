//! Actual CUDA managed H2D -> BF16 SiLU -> managed D2H execution smoke.
//! Run explicitly under an exclusive GPU window. This is not image-model parity.
#![cfg(feature = "cuda")]

#[path = "vnext_tensor_cuda_runtime/attention.rs"]
mod attention;
#[path = "vnext_tensor_cuda_runtime/normalization.rs"]
mod normalization;
#[path = "vnext_tensor_cuda_runtime/philox.rs"]
mod philox;
#[path = "vnext_tensor_cuda_runtime/rotary.rs"]
mod rotary;
#[path = "vnext_tensor_cuda_runtime/spatial.rs"]
mod spatial;
#[path = "vnext_tensor_cuda_runtime/spatial_decoder.rs"]
mod spatial_decoder;
#[path = "vnext_tensor_cuda_runtime/spatial_matmul.rs"]
mod spatial_matmul;
#[path = "vnext_tensor_cuda_runtime/text.rs"]
mod text;
#[path = "vnext_tensor_cuda_runtime/torch_philox.rs"]
mod torch_philox;
#[path = "vnext_tensor_cuda_runtime/transport.rs"]
mod transport;

use ferrum_execution::{PreparedTensorExecutor, TensorExecutorConfig};
use ferrum_kernels::backend::cuda::{
    vnext_ops::{
        cuda_tensor_runtime_config, cuda_vnext_silu_bf16_provider,
        CUDA_SILU_BF16_NUMERICAL_PROFILE_ID,
    },
    vnext_runtime::{CudaDeviceRuntime, CudaTensorRuntimeConfig},
};
use ferrum_runtime::*;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, OnceLock};

const INPUT_ID: &str = "value.tensor.input";
const OUTPUT_ID: &str = "value.tensor.output";
const PROFILE: &str = CUDA_SILU_BF16_NUMERICAL_PROFILE_ID;
const USABLE_BYTES: u64 = 2 << 30;

fn id<T>(value: impl Into<String>) -> T
where
    T: TryFrom<String, Error = VNextError>,
{
    T::try_from(value.into()).unwrap()
}

fn input_spec(dimensions: Vec<u64>, element_type: ElementType) -> ResolvedTensorSpec {
    ResolvedTensorSpec::new(dimensions, element_type, ResolvedTensorLayout::Contiguous).unwrap()
}

const BYTE_LEN: usize = 24;
const INPUT_BITS: [u16; 12] = [
    0x0000, 0x8000, 0x3f80, 0xc2b4, 0x0001, 0x007f, 0x8001, 0x807f, 0x7f80, 0xff80, 0x7fc1, 0x42b4,
];
// Prepared independently before candidate execution: signed zero, SiLU(1),
// expf overflow at -90, tiny exp=1 followed by FP32 division and BF16 RNE,
// infinities/NaN, and saturation at +90. Full primitive qualification remains
// pending; these selected singleton cases do not claim Torch parity.
const EXPECTED_BITS: [u16; 12] = [
    0x0000, 0x8000, 0x3f3b, 0x8000, 0x0000, 0x0040, 0x8000, 0x8040, 0x7f80, 0x7fc0, 0x7fc0, 0x42b4,
];

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    rows: u64,
    width: u64,
}

impl Config {
    fn standard() -> Self {
        Self { rows: 2, width: 6 }
    }

    fn spec(&self) -> ProgramTensorSpec {
        ProgramTensorSpec {
            dimensions: vec![self.rows, self.width],
            element_type: ElementType::Bf16,
            layout: ResolvedTensorLayout::Contiguous,
        }
    }
}

struct Family;

impl ModelFamilyProvider for Family {
    type Config = Config;

    fn family_id(&self) -> &ModelFamilyId {
        static ID: OnceLock<ModelFamilyId> = OnceLock::new();
        ID.get_or_init(|| id("family.cuda.tensor.weightless.silu.libdevice"))
    }

    fn external_metadata_ids(&self) -> BTreeSet<ExternalModelMetadataId> {
        BTreeSet::from([id("metadata.cuda.tensor.weightless.silu.libdevice")])
    }

    fn validate_config_identity(&self, raw: &Value, config: &Config) -> Result<(), VNextError> {
        if config.rows == 0
            || config.width == 0
            || config.rows.checked_mul(config.width) != Some(INPUT_BITS.len() as u64)
            || serde_json::to_value(config).unwrap() != *raw
        {
            return Err(VNextError::InvalidModelConfig {
                family_id: self.family_id().to_string(),
                field: "shape".into(),
                reason: "weightless fixture requires a canonical twelve-element matrix".into(),
            });
        }
        Ok(())
    }

    fn validated_external_metadata_id(
        &self,
        raw: &Value,
        config: &Config,
    ) -> Result<ExternalModelMetadataId, VNextError> {
        self.validate_config_identity(raw, config)?;
        Ok(id("metadata.cuda.tensor.weightless.silu.libdevice"))
    }

    fn parse_config(&self, raw: &Value) -> Result<Config, VNextError> {
        let config = serde_json::from_value(raw.clone()).map_err(|error| {
            VNextError::InvalidModelConfig {
                family_id: self.family_id().to_string(),
                field: "config".into(),
                reason: error.to_string(),
            }
        })?;
        self.validate_config_identity(raw, &config)?;
        Ok(config)
    }

    fn weights(&self, _: &Config) -> Result<ModelWeights, VNextError> {
        Ok(ModelWeights::None)
    }

    fn numerical_profiles(&self, _: &Config) -> Result<FamilyNumericalProfiles, VNextError> {
        let profile_id = NumericalProfileId::new(PROFILE).unwrap();
        FamilyNumericalProfiles::new(
            self.family_id(),
            ContractVersion::new(1, 0),
            vec![NumericalExecutionProfile {
                id: profile_id.clone(),
                family_id: self.family_id().clone(),
                version: ContractVersion::new(1, 0),
                kv_storage: vec![],
                primary_activation: id(OUTPUT_ID),
                boundaries: BTreeMap::from([(id(OUTPUT_ID), ElementType::Bf16)]),
                states: vec![],
                operations: vec![NumericalOperationContract {
                    operation_id: id(SILU_BF16_OPERATION_ID),
                    version: ContractVersion::new(1, 0),
                    multiplication_type: Some(ElementType::F32),
                    accumulation_type: Some(ElementType::F32),
                    staged_arithmetic: None,
                    composite_arithmetic: None,
                }],
            }],
            vec![profile_id],
        )
    }

    fn semantic_program(
        &self,
        _: &Config,
        _: &NumericalExecutionProfile,
    ) -> Result<ModelProgram, VNextError> {
        ModelProgram::new(
            self.family_id().clone(),
            vec![id(INPUT_ID)],
            vec![ProgramBlock {
                id: "block.weightless.silu".into(),
                nodes: vec![ProgramNode {
                    id: id("node.weightless.silu"),
                    operation_id: id(SILU_BF16_OPERATION_ID),
                    required_version: ContractVersion::new(1, 0),
                    work: ProgramNodeWorkSpec::Fixed,
                    inputs: vec![id(INPUT_ID)],
                    outputs: vec![id(OUTPUT_ID)],
                    attributes: BTreeMap::new(),
                }],
            }],
            vec![],
            vec![],
            vec![id(OUTPUT_ID)],
        )
    }

    fn semantic_metadata(&self, _: &Config) -> Result<ModelSemanticMetadata, VNextError> {
        Ok(ModelSemanticMetadata::TensorComponent)
    }
}

fn family(config: &Config) -> PreparedModelFamily {
    TypedFamilyRegistration::new(Family)
        .prepare_with_profile(
            &serde_json::to_value(config).unwrap(),
            &NumericalProfileId::new(PROFILE).unwrap(),
        )
        .unwrap()
}

fn options(config: &Config) -> ProgramPlanCompileOptions {
    let mut options =
        ProgramPlanCompileOptions::new(BTreeMap::from([(id(INPUT_ID), config.spec())])).unwrap();
    options.retain_completion_value(id(OUTPUT_ID));
    options
}

fn policy<R: DeviceRuntime>(runtime: &R) -> TensorPolicy {
    TensorPolicy {
        capacity_bytes: USABLE_BYTES,
        storage_profiles: runtime
            .descriptor()
            .dynamic_storage_profiles
            .iter()
            .copied()
            .collect(),
    }
}

fn account(total: usize, output: usize) -> HostMemoryAccount {
    HostMemoryAccount::new(
        HostMemoryLimits::new(
            total,
            HostMemoryClassLimits {
                input_preprocess_bytes: BYTE_LEN,
                transfer_staging_bytes: 0,
                output_bytes: output,
                weight_backing_bytes: 0,
            },
        )
        .unwrap(),
    )
}

fn input(host: &HostMemoryAccount, config: &Config) -> ManagedHostTensor {
    let mut allocation = host
        .allocate_zeroed(HostMemoryClass::InputPreprocess, BYTE_LEN)
        .unwrap();
    for (destination, bits) in allocation.as_mut().chunks_exact_mut(2).zip(INPUT_BITS) {
        destination.copy_from_slice(&bits.to_le_bytes());
    }
    ManagedHostTensor::new(
        id(INPUT_ID),
        input_spec(vec![config.rows, config.width], ElementType::Bf16),
        allocation.freeze(),
    )
    .unwrap()
}

fn check_output(bytes: &[u8]) {
    assert_eq!(bytes.len(), BYTE_LEN);
    for (index, (bytes, expected)) in bytes.chunks_exact(2).zip(EXPECTED_BITS).enumerate() {
        let actual = u16::from_le_bytes([bytes[0], bytes[1]]);
        if expected & 0x7f80 == 0x7f80 && expected & 0x007f != 0 {
            assert!(
                actual & 0x7f80 == 0x7f80 && actual & 0x007f != 0,
                "index {index}"
            );
        } else {
            assert_eq!(actual, expected, "index {index}");
        }
    }
}

fn no_static<R: DeviceRuntime>(
    runtime: Arc<R>,
    plan: &ExecutionPlan,
) -> Arc<PlanRuntimeResources<R>> {
    match plan
        .provision_static(runtime, id("request.weightless.provision"))
        .unwrap()
        .into_provisioning()
    {
        StaticProvisioning::NoStatic(provisioned) => provisioned.into_plan_runtime(),
        StaticProvisioning::Required(_) => panic!("weightless SiLU requires no static allocation"),
    }
}

fn close_resources<R: DeviceRuntime>(resources: Arc<PlanRuntimeResources<R>>) {
    match PlanRuntimeResources::close(resources) {
        Ok(PlanRuntimeCloseOutcome::Closed(receipt)) => {
            assert_eq!(receipt.released_static_resources(), 0)
        }
        Ok(PlanRuntimeCloseOutcome::Referenced { strong_count, .. }) => {
            panic!("weightless root retains {strong_count} owners")
        }
        Err(error) => panic!("weightless close failed: {:?}", error.failure()),
    }
}

/// A tensor policy carries no text scheduling ceiling or placeholder token.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct TensorPolicy {
    capacity_bytes: u64,
    storage_profiles: Vec<DynamicStorageProfile>,
}

impl RuntimePolicy for TensorPolicy {
    fn version(&self) -> ContractVersion {
        ContractVersion::new(1, 0)
    }
    fn memory_capacity_bytes(&self) -> u64 {
        self.capacity_bytes
    }
    fn memory_reserve_bytes(&self) -> u64 {
        0
    }
    fn maximum_active_sequences(&self) -> u32 {
        1
    }
    fn work_limits(&self) -> RuntimeWorkLimits {
        RuntimeWorkLimits::FixedTensor
    }
    fn attention_execution_policy(&self) -> ferrum_types::AttentionExecutionPolicy {
        ferrum_types::AttentionExecutionPolicy::Portable
    }
    fn execution_determinism_requirement(&self) -> ExecutionDeterminismRequirement {
        ExecutionDeterminismRequirement::BitwiseSameRuntime
    }
    fn dynamic_storage_profile_order(&self) -> &[DynamicStorageProfile] {
        &self.storage_profiles
    }
    fn reusable_execution_policy(&self) -> Option<&ReusableExecutionPolicy> {
        None
    }
    fn validate(&self) -> Result<(), VNextError> {
        if self.capacity_bytes == 0 || self.storage_profiles.is_empty() {
            return Err(VNextError::InvalidExecutionPlan {
                reason: "tensor fixture requires positive capacity and a storage profile"
                    .to_owned(),
            });
        }
        Ok(())
    }
}

#[test]
#[ignore = "requires exclusive CUDA; actual managed buffers and production BF16 SiLU PTX"]
fn managed_cuda_silu_upload_execute_readback_and_retirement() {
    assert_eq!(
        TensorExecutorConfig::default().device_timing_mode,
        DeviceTimingMode::Off
    );
    for mode in [DeviceTimingMode::Off, DeviceTimingMode::Kernel] {
        managed_cuda_silu_with_timing(mode);
    }
}

fn managed_cuda_silu_with_timing(mode: DeviceTimingMode) {
    // These are explicit conservative smoke-test reservations, not a measured
    // CUDA memory profile or a throughput/capacity qualification claim.
    let runtime = Arc::new(
        CudaDeviceRuntime::new_tensor_eager(CudaTensorRuntimeConfig {
            device: cuda_tensor_runtime_config(0, id("device.cuda.0")).unwrap(),
            usable_memory_bytes: USABLE_BYTES,
            bootstrap_memory_bytes: 1 << 30,
            stream_memory_bytes: 64 << 20,
            maximum_streams: 1,
            maximum_in_flight_per_stream: 1,
            maximum_commands_per_submission: 64,
            opaque_memory_profile: "experimental-unqualified-smoke-v1".into(),
        })
        .unwrap(),
    );
    assert!(runtime.supports_managed_tensor_io());
    let registry = OperationRuntimeRegistry::new(
        vec![Box::new(silu_bf16_contract().unwrap())],
        vec![cuda_vnext_silu_bf16_provider(&runtime).unwrap()],
    )
    .unwrap();
    // The fixed bootstrap reservation covers one live native provider bundle.
    assert!(cuda_vnext_silu_bf16_provider(&runtime).is_err());
    let engine = EngineProviderDescriptor::new(
        id("provider.engine.cuda.tensor.smoke"),
        ContractVersion::new(1, 0),
        runtime
            .descriptor()
            .runtime_implementation_fingerprint
            .clone(),
        runtime.descriptor().id.clone(),
        runtime.descriptor().capabilities.clone(),
    )
    .unwrap();
    let catalog = registry
        .capability_catalog(runtime.descriptor().clone(), vec![engine])
        .unwrap();
    let config = Config::standard();
    let prepared = family(&config);
    let compilation = ProgramPlanCompiler::compile(
        &prepared,
        &catalog,
        &policy(runtime.as_ref()),
        &registry.planning(),
        &options(&config),
    )
    .unwrap();
    let executable = compilation.executable();
    let plan = executable.execution_plan();
    assert!(matches!(
        plan.payload().execution_weights(),
        ExecutionWeights::None
    ));
    assert!(plan.payload().memory().static_allocations().is_empty());
    let resources = no_static(runtime.clone(), plan);
    let host = account(2 * BYTE_LEN, BYTE_LEN);
    let mut executor = PreparedTensorExecutor::new(
        executable.clone(),
        registry.bind_plan(executable).unwrap(),
        resources.clone(),
        host.clone(),
        TensorExecutorConfig {
            device_timing_mode: mode,
            ..TensorExecutorConfig::default()
        },
    )
    .unwrap();
    // A second execution exercises retirement and reuse of the one-flight lane.
    let mut final_output = None;
    for ordinal in 0..2 {
        let outputs = executor
            .execute(
                id(format!("run.cuda.tensor.silu.{ordinal}")),
                id(format!("request.cuda.tensor.silu.{ordinal}")),
                vec![input(&host, &config)],
            )
            .unwrap();
        let profile = executor.take_execution_profile();
        if mode == DeviceTimingMode::Off {
            assert!(profile.is_none());
        } else {
            let profile = profile.expect("CUDA profile");
            let attribution = profile.attribution.expect("CUDA command attribution");
            let timing = attribution
                .terminal_timing()
                .measured()
                .expect("CUDA event measurement");
            assert!(!attribution.device().commands().is_empty());
            assert_eq!(timing.spans().len(), attribution.device().commands().len());
            for command in attribution.device().commands() {
                let span = timing
                    .span_for_command(command.command_index())
                    .expect("each command has a physical interval");
                assert_eq!(span.command_count(), 1);
                assert!(span.measurement().elapsed_ns().is_some());
            }
            eprintln!(
                "CUDA tensor profile preflight: execution={ordinal}, measured_commands={}",
                timing.spans().len()
            );
        }
        assert!(executor.take_execution_profile().is_none());
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].tensor().dimensions(), [2, 6]);
        assert!(host.owns(outputs[0].bytes()));
        check_output(outputs[0].bytes().bytes());
        let retained_output = outputs[0].bytes().clone();
        drop(outputs);
        assert_eq!(host.usage().unwrap().committed_bytes(), BYTE_LEN);
        check_output(retained_output.bytes());
        if ordinal == 1 {
            final_output = Some(retained_output);
        } else {
            drop(retained_output);
            assert_eq!(host.usage().unwrap().committed_bytes(), 0);
        }
    }
    drop((executor, compilation, registry));
    close_resources(resources);
    // The previous module and lane must release their reservations on retirement.
    drop(cuda_vnext_silu_bf16_provider(&runtime).unwrap());
    drop(runtime.create_stream().unwrap());
    check_output(final_output.as_ref().unwrap().bytes());
    drop(final_output);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    assert_eq!(host.usage().unwrap().allocated_bytes(), 0);
    eprintln!("CUDA managed SiLU: two actual executions and host retirement completed; no image or throughput claim");
}
