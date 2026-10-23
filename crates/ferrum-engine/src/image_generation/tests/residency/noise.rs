//! Weightless session/backing lifecycle using real Reference BF16 addition.
//! These tests exercise the Noise residency slot, not a simulated RNG provider.
use super::*;
use ferrum_kernels::backend::reference::reference_vnext_operation_registry;
use serde_json::Value;
use std::{
    collections::BTreeSet,
    num::NonZeroU64,
    sync::atomic::{AtomicU64, Ordering},
};

const X: &str = "value.weightless.residency.input";
const Y: &str = "value.weightless.residency.output";
const PROFILE: &str = "profile.weightless.residency.bf16";

fn id<T: TryFrom<String, Error = VNextError>>(value: impl Into<String>) -> Result<T, VNextError> {
    T::try_from(value.into())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Config {
    elements: u64,
}
struct Family;
fn spec(elements: u64) -> ProgramTensorSpec {
    ProgramTensorSpec {
        dimensions: vec![1, elements],
        element_type: ElementType::Bf16,
        layout: ResolvedTensorLayout::Contiguous,
    }
}
impl ModelFamilyProvider for Family {
    type Config = Config;
    fn family_id(&self) -> &ModelFamilyId {
        static ID: std::sync::OnceLock<ModelFamilyId> = std::sync::OnceLock::new();
        ID.get_or_init(|| id("family.weightless.residency").unwrap())
    }
    fn external_metadata_ids(&self) -> BTreeSet<ExternalModelMetadataId> {
        BTreeSet::from([id("metadata.weightless.residency").unwrap()])
    }
    fn validate_config_identity(&self, raw: &Value, config: &Config) -> Result<(), VNextError> {
        if config.elements == 0 || *raw != serde_json::to_value(config).unwrap() {
            return Err(VNextError::InvalidExecutionPlan {
                reason: "invalid weightless fixture config".into(),
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
        id("metadata.weightless.residency")
    }
    fn parse_config(&self, raw: &Value) -> Result<Config, VNextError> {
        let config =
            serde_json::from_value(raw.clone()).map_err(|e| VNextError::InvalidExecutionPlan {
                reason: format!("weightless fixture: {e}"),
            })?;
        self.validate_config_identity(raw, &config)?;
        Ok(config)
    }
    fn weights(&self, _: &Config) -> Result<ModelWeights, VNextError> {
        Ok(ModelWeights::None)
    }
    fn numerical_profiles(&self, _: &Config) -> Result<FamilyNumericalProfiles, VNextError> {
        let profile = NumericalProfileId::new(PROFILE).unwrap();
        FamilyNumericalProfiles::new(
            self.family_id(),
            ContractVersion::new(1, 0),
            vec![NumericalExecutionProfile {
                id: profile.clone(),
                family_id: self.family_id().clone(),
                version: ContractVersion::new(1, 0),
                kv_storage: vec![],
                primary_activation: id(Y).unwrap(),
                boundaries: BTreeMap::from([(id(Y).unwrap(), ElementType::Bf16)]),
                states: vec![],
                operations: vec![NumericalOperationContract {
                    operation_id: id(PLAIN_RESIDUAL_ADD_BF16_OPERATION_ID).unwrap(),
                    version: ContractVersion::new(1, 0),
                    multiplication_type: Some(ElementType::F32),
                    accumulation_type: Some(ElementType::F32),
                    staged_arithmetic: None,
                    composite_arithmetic: None,
                }],
            }],
            vec![profile],
        )
    }
    fn semantic_program(
        &self,
        config: &Config,
        _: &NumericalExecutionProfile,
    ) -> Result<ModelProgram, VNextError> {
        ModelProgram::new(
            self.family_id().clone(),
            vec![id(X)?],
            vec![ProgramBlock {
                id: "weightless.residency".into(),
                nodes: vec![ProgramNode {
                    id: id("node.weightless.residency.add")?,
                    operation_id: id(PLAIN_RESIDUAL_ADD_BF16_OPERATION_ID)?,
                    required_version: ContractVersion::new(1, 0),
                    work: ProgramNodeWorkSpec::Fixed,
                    inputs: vec![id(X)?, id(X)?],
                    outputs: vec![id(Y)?],
                    attributes: BTreeMap::new(),
                }],
            }],
            vec![],
            vec![],
            vec![id(Y)?],
        )?
        .with_input_tensor_specs(BTreeMap::from([(id(X)?, spec(config.elements))]))
    }
    fn semantic_metadata(&self, _: &Config) -> Result<ModelSemanticMetadata, VNextError> {
        Ok(ModelSemanticMetadata::TensorComponent)
    }
}

fn pipeline() -> (
    QwenImage21Pipeline<ReferenceDeviceRuntime>,
    tempfile::TempDir,
) {
    let (original, directory) = numeric_pipeline();
    static NEXT_DEVICE: AtomicU64 = AtomicU64::new(0);
    let composition = composition(&format!(
        "device.image.weightless.residency.{}",
        NEXT_DEVICE.fetch_add(1, Ordering::Relaxed)
    ));
    let mut pipeline = QwenImage21Pipeline::new(
        composition.runtime().clone(),
        original.host,
        reference_vnext_operation_registry(composition.runtime()).unwrap(),
        composition.catalog().clone(),
        WeightMaterializerRegistry::identity_only().unwrap(),
        original.assets,
        original.profiles,
        original.config,
    )
    .unwrap();
    pipeline.config.residency = ImageResidencyPolicy::AllStages;
    set_ceiling(&mut pipeline, 4096);
    (pipeline, directory)
}
fn set_ceiling(pipeline: &mut QwenImage21Pipeline<ReferenceDeviceRuntime>, bytes: u64) {
    pipeline.config.dynamic_residency = ImageDynamicResidencyPolicy::RetainStageBudgets {
        maximum_bytes_by_stage: ImageStageRetentionBudgets {
            noise: NonZeroU64::new(bytes),
            ..ImageStageRetentionBudgets::default()
        },
    };
}
fn stage(
    pipeline: &QwenImage21Pipeline<ReferenceDeviceRuntime>,
    elements: u64,
) -> PreparedStage<ReferenceDeviceRuntime> {
    let family = TypedFamilyRegistration::new(Family)
        .prepare_with_profile(
            &serde_json::to_value(Config { elements }).unwrap(),
            &NumericalProfileId::new(PROFILE).unwrap(),
        )
        .unwrap();
    let options =
        ProgramPlanCompileOptions::new(BTreeMap::from([(id(X).unwrap(), spec(elements))])).unwrap();
    let prepared =
        PreparedStage::compile(ImageStage::Noise, 1, family, &options, pipeline).unwrap();
    assert_eq!(prepared.plan.static_bytes, 0);
    prepared
}
fn execute(
    pipeline: &mut QwenImage21Pipeline<ReferenceDeviceRuntime>,
    active: ActiveImageStage<ReferenceDeviceRuntime>,
    elements: u64,
    offset: u64,
    job: u64,
) -> (
    ActiveImageStage<ReferenceDeviceRuntime>,
    TensorExecutionOutput,
) {
    let mut bytes = pipeline
        .host
        .allocate_zeroed(HostMemoryClass::InputPreprocess, elements as usize * 2)
        .unwrap();
    for (index, part) in bytes.as_mut().chunks_exact_mut(2).enumerate() {
        part.copy_from_slice(
            &bf16::from_f32((offset + index as u64 + 1) as f32)
                .to_bits()
                .to_le_bytes(),
        );
    }
    let input = ManagedHostTensor::new(
        id(X).unwrap(),
        ResolvedTensorSpec::new(
            vec![1, elements],
            ElementType::Bf16,
            ResolvedTensorLayout::Contiguous,
        )
        .unwrap(),
        bytes.freeze(),
    )
    .unwrap();
    let (active, outputs) = pipeline
        .execute_stage(active, job, vec![input])
        .unwrap_or_else(|e| panic!("weightless execution: {e}"));
    let output = only_output(outputs, Y).unwrap();
    for (index, part) in output.bytes().bytes().chunks_exact(2).enumerate() {
        assert_eq!(
            u16::from_le_bytes([part[0], part[1]]),
            bf16::from_f32((2 * (offset + index as u64 + 1)) as f32).to_bits()
        );
    }
    (active, output)
}

#[test]
fn weightless_noise_reuses_session_and_backing_without_claiming_weight_reuse() {
    let (mut pipeline, _directory) = pipeline();
    let capacity = DeviceCapacityOwner::open(pipeline.runtime.descriptor(), DEVICE_BYTES).unwrap();
    let mut receipts = Vec::new();
    let mut outputs = Vec::new();
    let mut allocations = 0;
    for job in 0..2 {
        let prepared = stage(&pipeline, 8);
        let active = pipeline
            .start_stage(prepared, job, &mut receipts)
            .unwrap_or_else(|e| panic!("weightless start: {e}"));
        assert_eq!(active.receipt.session_reused, job == 1);
        assert!(!active.receipt.weights_reused);
        let (active, output) = execute(&mut pipeline, active, 8, job * 8, job);
        pipeline
            .finish_stage(active, &mut receipts)
            .unwrap_or_else(|e| panic!("weightless retain: {e}"));
        let ImageStageRetirement::ResidentBackingRetained { retention } =
            &receipts.last().unwrap().retirement
        else {
            panic!("retained Noise backing")
        };
        assert!(retention.retained_bytes() > 0);
        let remaining = capacity
            .reserve(
                DeviceOpaqueMemoryKind::Bootstrap,
                DEVICE_BYTES - retention.retained_bytes(),
            )
            .unwrap();
        assert!(capacity
            .reserve(DeviceOpaqueMemoryKind::Bootstrap, 1)
            .is_err());
        drop(remaining);
        let snapshot = pipeline.runtime.snapshot();
        if job == 0 {
            allocations = snapshot.allocations;
        } else {
            assert_eq!(snapshot.allocations, allocations);
        }
        assert_eq!(snapshot.uploaded_bytes, (job + 1) * 16);
        outputs.push(output);
    }
    let closed = pipeline.shutdown().unwrap();
    assert_eq!(
        closed.len(),
        1,
        "weightless sessions have no snapshot owner to close"
    );
    assert!(matches!(closed[0].retirement, ImageStageRetirement::Closed));
    assert_eq!(pipeline.runtime.snapshot().live_allocations, 0);
    assert_eq!(
        &outputs[0].bytes().bytes()[..2],
        &bf16::from_f32(2.).to_bits().to_le_bytes()
    );
    drop(outputs);
    assert_eq!(pipeline.host.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn weightless_noise_shape_replacement_and_tighter_cap_retire_real_backing() {
    let (mut pipeline, _directory) = pipeline();
    let mut receipts = Vec::new();
    let first = stage(&pipeline, 8);
    let old_hash = first.plan.plan_hash.clone();
    let active = pipeline
        .start_stage(first, 0, &mut receipts)
        .unwrap_or_else(|e| panic!("start: {e}"));
    let (active, output) = execute(&mut pipeline, active, 8, 0, 0);
    drop(output);
    pipeline
        .finish_stage(active, &mut receipts)
        .unwrap_or_else(|e| panic!("retain: {e}"));
    assert!(pipeline.runtime.snapshot().live_allocations > 0);
    let changed = stage(&pipeline, 4);
    assert_ne!(changed.plan.plan_hash, old_hash);
    let active = pipeline
        .start_stage(changed, 1, &mut receipts)
        .unwrap_or_else(|e| panic!("replace: {e}"));
    assert!(!active.receipt.session_reused);
    assert!(!active.receipt.weights_reused);
    assert!(matches!(
        receipts.last().unwrap().retirement,
        ImageStageRetirement::Closed
    ));
    assert_eq!(receipts.last().unwrap().execution.plan.plan_hash, old_hash);
    assert_eq!(
        pipeline.runtime.snapshot().live_allocations,
        0,
        "old dynamic owner closes before replacement executes"
    );
    let (active, output) = execute(&mut pipeline, active, 4, 16, 1);
    drop(output);
    set_ceiling(&mut pipeline, 1);
    pipeline
        .finish_stage(active, &mut receipts)
        .unwrap_or_else(|e| panic!("trim: {e}"));
    let ImageStageRetirement::Resident { idle_release } = &receipts.last().unwrap().retirement
    else {
        panic!("over-cap chunks must retire")
    };
    assert!(idle_release.reclaimed_bytes() > 1);
    assert_eq!(pipeline.runtime.snapshot().live_allocations, 0);
    pipeline.shutdown().unwrap();
    assert_eq!(pipeline.host.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn weightless_noise_actual_backing_counts_after_lowering_its_declared_cap() {
    let (mut pipeline, _directory) = pipeline();
    let mut receipts = Vec::new();
    let initial = stage(&pipeline, 8);
    let active = pipeline
        .start_stage(initial, 0, &mut receipts)
        .unwrap_or_else(|e| panic!("start: {e}"));
    let (active, output) = execute(&mut pipeline, active, 8, 0, 0);
    drop(output);
    pipeline
        .finish_stage(active, &mut receipts)
        .unwrap_or_else(|e| panic!("retain: {e}"));
    let ImageStageRetirement::ResidentBackingRetained { retention } =
        &receipts.last().unwrap().retirement
    else {
        panic!("retained Noise backing")
    };
    let actual = retention.retained_bytes();
    assert!(actual > 1);
    set_ceiling(&mut pipeline, 1);
    let replacement = stage(&pipeline, 2);
    let denoiser = prepared(&pipeline, 1);
    let demand = |s: &PreparedStage<ReferenceDeviceRuntime>| {
        s.component
            .executable()
            .execution_plan()
            .payload()
            .memory()
            .minimum_runnable_request_bytes()
    };
    let required =
        denoiser.plan.static_bytes + demand(&replacement).max(demand(&denoiser) + actual);
    let before = pipeline.runtime.snapshot();
    pipeline.config.device_capacity_bytes = required - 1;
    assert!(pipeline
        .validate_resident_capacity(&[&replacement, &denoiser])
        .is_err());
    pipeline.config.device_capacity_bytes = required;
    pipeline
        .validate_resident_capacity(&[&replacement, &denoiser])
        .unwrap();
    assert_eq!(
        pipeline.runtime.snapshot(),
        before,
        "preflight never allocates or erases old receipts"
    );
    pipeline.shutdown().unwrap();
    assert_eq!(pipeline.runtime.snapshot().live_allocations, 0);
    assert_eq!(pipeline.host.usage().unwrap().committed_bytes(), 0);
}
