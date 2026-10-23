//! A real four-node graph distinguishes liveness reuse from preamble input
//! clobbering. Arithmetic is independently hand computed as 8*x + late.

use super::*;

const LATE: &str = "value.liveness.late";
const A: &str = "value.liveness.a";
const B: &str = "value.liveness.b";
const C: &str = "value.liveness.c";
const PROFILE: &str = "reference.tensor.fixed_liveness.bf16";

struct ReuseFamily;

impl ModelFamilyProvider for ReuseFamily {
    type Config = Config;

    fn family_id(&self) -> &ModelFamilyId {
        static ID: OnceLock<ModelFamilyId> = OnceLock::new();
        ID.get_or_init(|| id("family.reference.tensor.fixed-liveness"))
    }

    fn external_metadata_ids(&self) -> BTreeSet<ExternalModelMetadataId> {
        Family.external_metadata_ids()
    }

    fn validate_config_identity(&self, raw: &Value, config: &Config) -> Result<(), VNextError> {
        Family.validate_config_identity(raw, config)
    }

    fn validated_external_metadata_id(
        &self,
        raw: &Value,
        config: &Config,
    ) -> Result<ExternalModelMetadataId, VNextError> {
        Family.validated_external_metadata_id(raw, config)
    }

    fn parse_config(&self, raw: &Value) -> Result<Config, VNextError> {
        Family.parse_config(raw)
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
                primary_activation: id(OUTPUT_ID),
                boundaries: [A, B, C, OUTPUT_ID]
                    .into_iter()
                    .map(|value| (id(value), ElementType::Bf16))
                    .collect(),
                states: vec![],
                operations: vec![NumericalOperationContract {
                    operation_id: id(PLAIN_RESIDUAL_ADD_BF16_OPERATION_ID),
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
        let node = |name: &str, lhs: &str, rhs: &str, output: &str| ProgramNode {
            id: id(name),
            operation_id: id(PLAIN_RESIDUAL_ADD_BF16_OPERATION_ID),
            required_version: ContractVersion::new(1, 0),
            work: ProgramNodeWorkSpec::Fixed,
            inputs: vec![id(lhs), id(rhs)],
            outputs: vec![id(output)],
            attributes: BTreeMap::new(),
        };
        ModelProgram::new(
            self.family_id().clone(),
            vec![id(INPUT_ID), id(LATE)],
            vec![ProgramBlock {
                id: "block.fixed-liveness".into(),
                nodes: vec![
                    node("node.fixed-liveness.a", INPUT_ID, INPUT_ID, A),
                    node("node.fixed-liveness.b", A, A, B),
                    node("node.fixed-liveness.c", B, B, C),
                    node("node.fixed-liveness.output", C, LATE, OUTPUT_ID),
                ],
            }],
            vec![],
            vec![],
            vec![id(OUTPUT_ID)],
        )?
        .with_input_tensor_specs(BTreeMap::from([
            (id(INPUT_ID), config.spec()),
            (id(LATE), config.spec()),
        ]))
    }

    fn semantic_metadata(&self, _: &Config) -> Result<ModelSemanticMetadata, VNextError> {
        Ok(ModelSemanticMetadata::TensorComponent)
    }
}

fn execute(retain_a: bool) -> (Vec<u8>, u64) {
    // All additions and these expected BF16 values are exact. Late input has
    // distinct values and is uploaded before A is produced, but consumed last.
    let x = [
        1.0, 2.0, -1.0, 0.5, -0.5, 3.0, -3.0, 4.0, 0.0, 0.25, -0.25, 8.0,
    ];
    let late = [
        10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0,
    ];
    let expected = [
        18.0, 36.0, 22.0, 44.0, 46.0, 84.0, 46.0, 112.0, 90.0, 102.0, 108.0, 184.0,
    ];
    let expected = expected
        .into_iter()
        .flat_map(|value| half::bf16::from_f32(value).to_bits().to_le_bytes())
        .collect::<Vec<_>>();
    let config = Config::standard();
    let family = TypedFamilyRegistration::new(ReuseFamily)
        .prepare_with_profile(
            &serde_json::to_value(&config).unwrap(),
            &NumericalProfileId::new(PROFILE).unwrap(),
        )
        .unwrap();
    let composition =
        ReferenceVNextComposition::create(id("device.reference.fixed-liveness")).unwrap();
    let policy = policy(composition.runtime().as_ref());
    let mut options = ProgramPlanCompileOptions::new(BTreeMap::from([
        (id(INPUT_ID), config.spec()),
        (id(LATE), config.spec()),
    ]))
    .unwrap();
    if retain_a {
        options.retain_completion_value(id(A));
    }
    let compilation = ProgramPlanCompiler::compile(
        &family,
        composition.catalog(),
        &policy,
        &composition.registry().planning(),
        &options,
    )
    .unwrap();
    let plan = compilation.executable().execution_plan();
    let memory = plan.payload().memory();
    let dedicated_bytes = memory
        .dynamic_descriptors()
        .iter()
        .map(|descriptor| descriptor.minimum_request_bytes().unwrap())
        .sum::<u64>();
    let planned = memory.minimum_step_bytes();
    assert_eq!(memory.static_bytes(), 0);
    assert_eq!(memory.dynamic_descriptors().len(), 6);
    if retain_a {
        assert_eq!(planned, dedicated_bytes);
    } else {
        assert!(planned < dedicated_bytes);
        assert_eq!(
            memory
                .dynamic_pools()
                .iter()
                .flat_map(|pool| pool.step_resource_slots())
                .filter(|slot| slot.kind() == StepResourceSlotKind::OrderedSingleFenceStepWave)
                .count(),
            1
        );
    }
    // Trusted reconstruction re-derives the same resource-origin and lifetime
    // proof; the public JSON does not grant an independent alias authority.
    let restored = ExecutionPlan::from_json_validated_with_completion_retention(
        &plan.to_json().unwrap(),
        &family,
        composition.catalog(),
        &policy,
        compilation.node_resolutions().to_vec(),
        compilation.completion_retention().clone(),
    )
    .unwrap();
    assert_eq!(
        restored.payload().memory().dynamic_pools(),
        memory.dynamic_pools()
    );

    let resources = no_static(composition.runtime().clone(), plan);
    let host = HostMemoryAccount::new(
        HostMemoryLimits::new(
            3 * BYTE_LEN,
            HostMemoryClassLimits {
                input_preprocess_bytes: 2 * BYTE_LEN,
                transfer_staging_bytes: 0,
                output_bytes: BYTE_LEN,
                weight_backing_bytes: 0,
            },
        )
        .unwrap(),
    );
    let input = |name: &str, values: [f32; 12]| {
        let mut bytes = host
            .allocate_zeroed(HostMemoryClass::InputPreprocess, BYTE_LEN)
            .unwrap();
        for (destination, value) in bytes.as_mut().chunks_exact_mut(2).zip(values) {
            destination.copy_from_slice(&half::bf16::from_f32(value).to_bits().to_le_bytes());
        }
        ManagedHostTensor::new(
            id(name),
            input_spec(vec![2, 6], ElementType::Bf16),
            bytes.freeze(),
        )
        .unwrap()
    };
    let mut executor = PreparedTensorExecutor::new(
        compilation.executable().clone(),
        composition
            .registry()
            .bind_plan(compilation.executable())
            .unwrap(),
        resources.clone(),
        host.clone(),
        TensorExecutorConfig::default(),
    )
    .unwrap();
    let outputs = executor
        .execute(
            id("run.fixed-liveness"),
            id("request.fixed-liveness"),
            vec![input(INPUT_ID, x), input(LATE, late)],
        )
        .unwrap();
    assert_eq!(outputs.len(), 1);
    assert_eq!(outputs[0].bytes().bytes(), expected);
    let status = resources.dynamic_pool_status().unwrap();
    let resident = status
        .pools()
        .iter()
        .map(|pool| pool.resident_bytes())
        .sum::<u64>();
    assert_eq!(resident, planned);
    assert!(status.pools().iter().all(|pool| pool.live_segments() == 0));
    assert_eq!(
        composition.runtime().snapshot().plain_residual_add_launches,
        4
    );
    let output = outputs[0].bytes().clone();
    let actual = output.bytes().to_vec();
    drop((outputs, executor, compilation));
    close_resources(resources);
    assert_eq!(composition.runtime().snapshot().live_allocations, 0);
    assert_eq!(host.usage().unwrap().committed_bytes(), BYTE_LEN);
    assert_eq!(output.bytes(), expected);
    drop(output);
    assert_eq!(host.usage().unwrap().allocated_bytes(), 0);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    (actual, resident)
}

#[test]
fn fixed_liveness_preserves_late_input_and_output_while_reducing_real_pool_residency() {
    let (reused_output, reused_bytes) = execute(false);
    let (dedicated_output, dedicated_bytes) = execute(true);
    assert_eq!(reused_output, dedicated_output);
    assert!(reused_bytes < dedicated_bytes);
}
