use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::{Arc, OnceLock};

use ferrum_interfaces::vnext::*;
use ferrum_kernels::backend::reference::{
    ReferenceDeviceRuntime, ReferenceVNextComposition, REFERENCE_DENSE_SAFETENSORS_FORMAT_ID,
};
use ferrum_quantization::SafetensorsArchive;
use half::f16;
use safetensors::tensor::{serialize_to_file, Dtype, TensorView};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

const ROWS: u64 = 1;
const MIXED_ROWS: u64 = 8;
const MIXED_ROWS_USIZE: usize = 8;
const IN_FEATURES: u64 = 2;
const OUT_FEATURES: u64 = 2;
const MAX_MAINTENANCE_ATTEMPTS: usize = 3;
const MIXED_REFERENCE_SEEDS: [usize; 3] = [0, 1, 2];

fn id<T>(value: impl Into<String>) -> T
where
    T: TryFrom<String, Error = VNextError>,
{
    T::try_from(value.into()).unwrap()
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct TinyDenseConfig {
    rows: u64,
    in_features: u64,
    out_features: u64,
}

struct TinyDenseFamily;

impl ModelFamilyProvider for TinyDenseFamily {
    type Config = TinyDenseConfig;

    fn family_id(&self) -> &ModelFamilyId {
        static FAMILY_ID: OnceLock<ModelFamilyId> = OnceLock::new();
        FAMILY_ID.get_or_init(|| id("family.reference.tiny-dense"))
    }

    fn external_metadata_ids(&self) -> BTreeSet<ExternalModelMetadataId> {
        BTreeSet::from([id("metadata.reference.tiny-dense")])
    }

    fn validate_config_identity(
        &self,
        _raw: &Value,
        config: &Self::Config,
    ) -> Result<(), VNextError> {
        if !(1..=MIXED_ROWS).contains(&config.rows)
            || config.in_features != IN_FEATURES
            || config.out_features != OUT_FEATURES
        {
            return Err(VNextError::InvalidModelConfig {
                family_id: self.family_id().to_string(),
                field: "shape".to_owned(),
                reason: "L1 tiny-dense shape differs from the bounded fixture".to_owned(),
            });
        }
        Ok(())
    }

    fn validated_external_metadata_id(
        &self,
        raw: &Value,
        config: &Self::Config,
    ) -> Result<ExternalModelMetadataId, VNextError> {
        self.validate_config_identity(raw, config)?;
        Ok(id("metadata.reference.tiny-dense"))
    }

    fn parse_config(&self, raw: &Value) -> Result<Self::Config, VNextError> {
        let config: TinyDenseConfig = serde_json::from_value(raw.clone()).map_err(|error| {
            VNextError::InvalidModelConfig {
                family_id: self.family_id().to_string(),
                field: "config".to_owned(),
                reason: error.to_string(),
            }
        })?;
        self.validate_config_identity(raw, &config)?;
        Ok(config)
    }

    fn weight_schema(&self, config: &Self::Config) -> Result<WeightSchema, VNextError> {
        Ok(WeightSchema {
            format_id: id(REFERENCE_DENSE_SAFETENSORS_FORMAT_ID),
            layout_id: id("weight-layout.reference.tiny-dense"),
            version: ContractVersion::new(1, 0),
            components: vec![WeightComponentSpec {
                id: id("weight.component.reference.linear"),
                role: WeightComponentRole::Values,
                external_names: vec!["linear.weight".to_owned()],
                dimensions: vec![config.out_features, config.in_features],
                encoding: WeightEncoding::Dense {
                    element_type: ElementType::F16,
                },
                required: true,
            }],
            tensors: vec![WeightTensorSpec {
                id: id("weight.reference.linear"),
                dimensions: vec![config.out_features, config.in_features],
                logical_element_type: ElementType::F16,
                physical_layout: PhysicalWeightLayout::Dense {
                    component_id: id("weight.component.reference.linear"),
                },
                required: true,
            }],
        })
    }

    fn semantic_program(&self, config: &Self::Config) -> Result<ModelProgram, VNextError> {
        ModelProgram::new(
            self.family_id().clone(),
            vec![
                id("value.reference.input.fixed"),
                id("value.reference.input.tokens"),
            ],
            vec![ProgramBlock {
                id: "block.reference.tiny-dense".to_owned(),
                nodes: vec![
                    ProgramNode {
                        id: id("node.reference.dense-linear.fixed"),
                        operation_id: id(DENSE_LINEAR_OPERATION_ID),
                        required_version: ContractVersion::new(1, 0),
                        work: ProgramNodeWorkSpec::Fixed,
                        inputs: vec![
                            id("value.reference.input.fixed"),
                            id("value.reference.weight"),
                        ],
                        outputs: vec![id("value.reference.output.fixed")],
                        attributes: BTreeMap::from([
                            (
                                id("in_features"),
                                SemanticValue::Unsigned(config.in_features),
                            ),
                            (
                                id("out_features"),
                                SemanticValue::Unsigned(config.out_features),
                            ),
                        ]),
                    },
                    ProgramNode {
                        id: id("node.reference.dense-linear.tokens"),
                        operation_id: id(DENSE_LINEAR_OPERATION_ID),
                        required_version: ContractVersion::new(1, 0),
                        work: ProgramNodeWorkSpec::tokens(id("value.reference.input.tokens"), 0),
                        inputs: vec![
                            id("value.reference.input.tokens"),
                            id("value.reference.weight"),
                        ],
                        outputs: vec![id("value.reference.output.tokens")],
                        attributes: BTreeMap::from([
                            (
                                id("in_features"),
                                SemanticValue::Unsigned(config.in_features),
                            ),
                            (
                                id("out_features"),
                                SemanticValue::Unsigned(config.out_features),
                            ),
                        ]),
                    },
                ],
            }],
            Vec::new(),
            vec![WeightReference {
                weight_id: id("weight.reference.linear"),
                value_id: id("value.reference.weight"),
                tensor: ProgramTensorSpec {
                    dimensions: vec![config.out_features, config.in_features],
                    element_type: ElementType::F16,
                    layout: ResolvedTensorLayout::Contiguous,
                },
            }],
            vec![
                id("value.reference.output.fixed"),
                id("value.reference.output.tokens"),
            ],
        )
    }

    fn semantic_metadata(
        &self,
        _config: &Self::Config,
    ) -> Result<ModelSemanticMetadata, VNextError> {
        Ok(ModelSemanticMetadata {
            template: TemplateMetadata {
                template: "{{ messages }}".to_owned(),
                source_file: "reference-template.json".to_owned(),
                sha256: sha256(b"{{ messages }}"),
            },
            special_tokens: SpecialTokenMetadata {
                bos_token_id: Some(1),
                eos_token_ids: BTreeSet::from([2]),
                pad_token_id: Some(0),
                collision_policy: SpecialTokenCollisionPolicy::require_distinct(),
            },
        })
    }
}

fn runtime_policy(runtime: &ReferenceDeviceRuntime) -> ResolvedRuntimePolicy {
    ResolvedRuntimePolicy::new(
        "runtime-policy.reference.l1",
        ContractVersion::new(1, 0),
        SchedulingDiscipline::FirstReady,
        RuntimeMemoryPolicy {
            capacity_bytes: runtime.descriptor().total_memory_bytes,
            reserve_bytes: 1 << 20,
            maximum_active_sequences: 4,
            dynamic_storage_profile_order: runtime
                .descriptor()
                .dynamic_storage_profiles
                .iter()
                .copied()
                .collect(),
        },
        AdmissionPolicy {
            maximum_queue_depth: 4,
            maximum_scheduled_tokens: 16,
            sequence_fit_policy: AdmissionFitPolicy::ImmediateOnly,
            allow_defer: true,
            cancellation_check_interval_steps: 1,
        },
        ferrum_types::AttentionExecutionPolicy::Portable,
        ExecutionDeterminismRequirement::BitwiseSameRuntime,
        None,
    )
    .unwrap()
}

fn tiny_work() -> (TokenSpanWork, ResourceWorkShape) {
    let span = TokenSpanWork::from_token_ids(&[7], 0..1).unwrap();
    let work = ResourceWorkShape::single(span.clone()).unwrap();
    (span, work)
}

fn admit_sequence(
    resources: &Arc<PlanRuntimeResources<ReferenceDeviceRuntime>>,
    work: &ResourceWorkShape,
    run_id: &str,
    request_id: &str,
) -> Arc<AdmittedSequenceResources<ReferenceDeviceRuntime>> {
    let request = RequestResourceAdmissionRequest::new(
        work.clone(),
        AdmissionFitPolicy::ImmediateOnly,
        AdmissionPressureAction::WaitForRelease,
    )
    .unwrap();
    let binding = resources.trusted_runtime_binding().unwrap();
    let request_resources = (0..=MAX_MAINTENANCE_ATTEMPTS)
        .find_map(|attempt| {
            match binding
                .try_admit_request(request.clone(), id(run_id), id(request_id))
                .unwrap()
            {
                RequestResourceAdmissionDecision::Admitted(resources) => Some(resources),
                RequestResourceAdmissionDecision::BackingDeferred(deferred)
                    if attempt < MAX_MAINTENANCE_ATTEMPTS =>
                {
                    deferred.maintain().unwrap();
                    None
                }
                _ => panic!("reference L1 request admission did not converge"),
            }
        })
        .expect("bounded request backing maintenance must converge");
    let request = SequenceResourceAdmissionRequest::new(
        work.clone(),
        AdmissionFitPolicy::ImmediateOnly,
        AdmissionPressureAction::WaitForRelease,
    )
    .unwrap();
    (0..=MAX_MAINTENANCE_ATTEMPTS)
        .find_map(|attempt| {
            match request_resources
                .try_admit_sequence(request.clone())
                .unwrap()
            {
                SequenceResourceAdmissionDecision::Admitted(resources) => Some(resources),
                SequenceResourceAdmissionDecision::BackingDeferred(deferred)
                    if attempt < MAX_MAINTENANCE_ATTEMPTS =>
                {
                    deferred.maintain().unwrap();
                    None
                }
                _ => panic!("reference L1 sequence admission did not converge"),
            }
        })
        .expect("bounded sequence backing maintenance must converge")
}

fn begin_step(
    batch: &ExecutionBatchParticipants<ReferenceDeviceRuntime>,
    lane: &Arc<ExecutionLane<ReferenceDeviceRuntime>>,
    spans: &[TokenSpanWork],
) -> Arc<StepResourceLease<ReferenceDeviceRuntime>> {
    let request = StepResourceAdmissionRequest::new(
        batch.bind_work_shape(spans.to_vec()).unwrap(),
        AdmissionFitPolicy::ImmediateOnly,
        AdmissionPressureAction::WaitForRelease,
    )
    .unwrap();
    (0..=MAX_MAINTENANCE_ATTEMPTS)
        .find_map(
            |attempt| match batch.try_begin_step(request.clone(), lane).unwrap() {
                StepResourceAdmissionDecision::Admitted(step) => Some(step),
                StepResourceAdmissionDecision::BackingDeferred(deferred)
                    if attempt < MAX_MAINTENANCE_ATTEMPTS =>
                {
                    deferred.maintain().unwrap();
                    None
                }
                _ => panic!("reference L1 step admission did not converge"),
            },
        )
        .expect("bounded step backing maintenance must converge")
}

fn prepare_wave(
    plan: &ExecutionPlan,
    step: &Arc<StepResourceLease<ReferenceDeviceRuntime>>,
    spans: &[TokenSpanWork],
) -> PreparedStepSubmissionWave<ReferenceDeviceRuntime> {
    let requests = plan
        .payload()
        .nodes()
        .iter()
        .map(|node| {
            InvocationResourceAdmissionRequest::for_all_step_participants(
                node.id().clone(),
                step.bind_all_invocation_work_shape(spans.to_vec()).unwrap(),
                AdmissionFitPolicy::ImmediateOnly,
                AdmissionPressureAction::WaitForRelease,
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    (0..=MAX_MAINTENANCE_ATTEMPTS)
        .find_map(
            |attempt| match step.try_prepare_submission_wave(requests.clone()).unwrap() {
                StepSubmissionWaveAdmissionDecision::Prepared(wave) => Some(wave),
                StepSubmissionWaveAdmissionDecision::BackingDeferred(deferred)
                    if attempt < MAX_MAINTENANCE_ATTEMPTS =>
                {
                    deferred.maintain().unwrap();
                    None
                }
                _ => panic!("reference L1 wave admission did not converge"),
            },
        )
        .expect("bounded wave backing maintenance must converge")
}

struct MixedReferenceFixture {
    composition: ReferenceVNextComposition,
    compilation: ProgramPlanCompilation,
    providers: BoundOperationProviderSet<ReferenceDeviceRuntime>,
    plan_resources: Arc<PlanRuntimeResources<ReferenceDeviceRuntime>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MixedSpanClass {
    FinalChunk,
    NonFinalChunk,
    DecodeTail,
}

#[derive(Clone)]
struct MixedReferenceCase {
    logical_index: usize,
    span_class: MixedSpanClass,
    span: TokenSpanWork,
    rows: Vec<[f32; 2]>,
}

struct MixedReferenceBatchResult {
    outputs: BTreeMap<usize, Vec<u8>>,
    physical_submissions: u64,
}

fn build_mixed_reference_fixture() -> MixedReferenceFixture {
    let model_dir = tempfile::tempdir().unwrap();
    let weight_values = [2.0_f32, 1.0, -1.0, 3.0];
    let weight_bytes = weight_values
        .iter()
        .flat_map(|value| f16::from_f32(*value).to_bits().to_le_bytes())
        .collect::<Vec<_>>();
    let tensor = TensorView::new(Dtype::F16, vec![2, 2], &weight_bytes).unwrap();
    let weight_path = model_dir.path().join("model.safetensors");
    serialize_to_file(
        vec![("linear.weight".to_owned(), tensor)],
        &None::<HashMap<String, String>>,
        &weight_path,
    )
    .unwrap();
    let weight_source = SafetensorsArchive::open(model_dir.path()).unwrap();

    let family = TypedFamilyRegistration::new(TinyDenseFamily)
        .prepare(&json!({
            "rows": MIXED_ROWS,
            "in_features": IN_FEATURES,
            "out_features": OUT_FEATURES
        }))
        .unwrap();
    let program_fingerprint = family.program().fingerprint().unwrap();
    let composition = ReferenceVNextComposition::create(id("device.reference.l1.mixed")).unwrap();
    let policy = runtime_policy(composition.runtime());
    let input_tensor = ProgramTensorSpec {
        dimensions: vec![MIXED_ROWS, IN_FEATURES],
        element_type: ElementType::F16,
        layout: ResolvedTensorLayout::Contiguous,
    };
    let mut options = ProgramPlanCompileOptions::new(BTreeMap::from([
        (id("value.reference.input.fixed"), input_tensor.clone()),
        (id("value.reference.input.tokens"), input_tensor),
    ]))
    .unwrap();
    options.require_weight_materializer(composition.weight_materializer_id().clone());
    assert!(options.retain_completion_value(id("value.reference.output.tokens")));
    let compilation = ProgramPlanCompiler::compile_with_weight_materializers(
        &family,
        composition.catalog(),
        &policy,
        &composition.registry().planning(),
        composition.weight_materializers(),
        &options,
    )
    .unwrap();
    assert_eq!(family.program().fingerprint().unwrap(), program_fingerprint);
    let executable = compilation.executable();
    let plan = executable.execution_plan();
    let providers = composition.registry().bind_plan(executable).unwrap();

    let provisioned = plan
        .provision_static(
            Arc::clone(composition.runtime()),
            id("request.reference.l1.mixed.provision"),
        )
        .unwrap();
    let permit = match provisioned.into_provisioning() {
        StaticProvisioning::Required(permit) => permit,
        StaticProvisioning::NoStatic(_) => {
            panic!("mixed L1 real weights require static provisioning")
        }
    };
    let identity = ResourceTransactionIdentity::for_admission(
        permit.binding(),
        id("run.reference.l1.mixed.provision"),
        id("transaction.reference.l1.mixed.provision"),
    );
    let driver = RuntimeResourceDriver::new(Arc::clone(composition.runtime())).unwrap();
    let reserved = ResourceTransaction::begin(driver, identity, permit)
        .unwrap()
        .reserve()
        .unwrap();
    let committed = match reserved.commit() {
        Ok(committed) => committed,
        Err(ResourceCommitTransitionError::Recoverable(error)) => {
            panic!("mixed L1 static commit failed: {:?}", error.failure())
        }
        Err(ResourceCommitTransitionError::Poisoned(error)) => {
            panic!(
                "mixed L1 static commit was indeterminate: {:?}",
                error.failure()
            )
        }
    };
    let pool_ids = committed
        .maintenance_controller()
        .pool_ids()
        .cloned()
        .collect::<Vec<_>>();
    for pool_id in pool_ids {
        committed
            .maintenance_controller()
            .initialize_pool(&pool_id)
            .unwrap();
    }
    let initialized = committed
        .initialize_static(
            &family,
            plan,
            &weight_source,
            StaticInitializationPolicy::new(1 << 20, 8).unwrap(),
        )
        .unwrap();
    assert_eq!(initialized.receipt().uploaded_component_count(), 1);
    assert_eq!(initialized.receipt().uploaded_bytes(), 8);
    let plan_resources = match initialized.into_plan_runtime() {
        Ok(resources) => resources,
        Err(error) => panic!("mixed L1 runtime handoff failed: {}", error.error()),
    };

    MixedReferenceFixture {
        composition,
        compilation,
        providers,
        plan_resources,
    }
}

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut value = *state;
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn mixed_cases(seed: usize) -> Vec<MixedReferenceCase> {
    let width = 2 + seed % 3;
    (0..width)
        .map(|logical_index| {
            let span_class = match (seed + logical_index) % 3 {
                0 => MixedSpanClass::FinalChunk,
                1 => MixedSpanClass::NonFinalChunk,
                _ => MixedSpanClass::DecodeTail,
            };
            let immediate_range = match span_class {
                MixedSpanClass::FinalChunk => 6..8,
                MixedSpanClass::NonFinalChunk => 2..4,
                MixedSpanClass::DecodeTail => 7..8,
            };
            let token_base = u32::try_from((seed + 1) * 10_000 + logical_index * 100).unwrap();
            let token_ids = (0..MIXED_ROWS_USIZE)
                .map(|row| token_base + u32::try_from(row).unwrap())
                .collect::<Vec<_>>();
            let span = TokenSpanWork::from_token_ids(&token_ids, immediate_range).unwrap();
            let mut state = 0x6a09_e667_f3bc_c909
                ^ (u64::try_from(seed).unwrap() << 32)
                ^ u64::try_from(logical_index).unwrap();
            let rows = (0..MIXED_ROWS_USIZE)
                .map(|_| {
                    let x = (i32::try_from(splitmix64(&mut state) % 65).unwrap() - 32) as f32 / 8.0;
                    let y = (i32::try_from(splitmix64(&mut state) % 65).unwrap() - 32) as f32 / 8.0;
                    [x, y]
                })
                .collect();
            MixedReferenceCase {
                logical_index,
                span_class,
                span,
                rows,
            }
        })
        .collect()
}

fn encode_f16_rows(rows: &[[f32; 2]], range: std::ops::Range<u64>) -> Vec<u8> {
    let start = usize::try_from(range.start).unwrap();
    let end = usize::try_from(range.end).unwrap();
    rows[start..end]
        .iter()
        .flat_map(|row| row.iter())
        .flat_map(|value| f16::from_f32(*value).to_bits().to_le_bytes())
        .collect()
}

fn analytic_output(case: &MixedReferenceCase) -> Vec<u8> {
    let range = case.span.immediate_token_range();
    let start = usize::try_from(range.start).unwrap();
    let end = usize::try_from(range.end).unwrap();
    case.rows[start..end]
        .iter()
        .flat_map(|[x, y]| {
            let x = f16::from_f32(*x).to_f32();
            let y = f16::from_f32(*y).to_f32();
            [f16::from_f32(2.0 * x + y), f16::from_f32(-x + 3.0 * y)]
        })
        .flat_map(|value| value.to_bits().to_le_bytes())
        .collect()
}

fn activation_resource(
    plan: &ExecutionPlan,
    node: &PlanNode,
    role: ResolvedValueRole,
    ordinal: u32,
) -> (ResourceId, AllocationLifetime) {
    let binding = node
        .values()
        .iter()
        .find(|binding| binding.role() == role && binding.ordinal() == ordinal)
        .expect("mixed L1 node must retain its activation binding");
    let [component] = binding.storage().components() else {
        panic!("mixed L1 activation must use one resource")
    };
    let descriptor = plan
        .payload()
        .memory()
        .dynamic_descriptors()
        .iter()
        .find(|descriptor| descriptor.base_resource_id() == component.resource_id())
        .expect("mixed L1 activation must use dynamic backing");
    assert_eq!(descriptor.usage(), BufferUsage::Activations);
    (component.resource_id().clone(), descriptor.lifetime())
}

fn token_upload_logical_offset(range: &BatchParticipantTokenRange) -> u64 {
    let start = range.source_token_range().start;
    start
        .checked_mul(IN_FEATURES)
        .and_then(|elements| elements.checked_mul(ElementType::F16.size_bytes()))
        .expect("mixed L1 token offset must fit u64")
}

fn token_readback_logical_offset(
    lifetime: AllocationLifetime,
    range: &BatchParticipantTokenRange,
) -> u64 {
    let start = match lifetime {
        AllocationLifetime::Step => 0,
        AllocationLifetime::Invocation => range.immediate_token_range().start,
        AllocationLifetime::Request | AllocationLifetime::Sequence => {
            range.source_token_range().start
        }
        AllocationLifetime::Plan => panic!("mixed L1 token activation cannot be plan-lifetime"),
    };
    start
        .checked_mul(OUT_FEATURES)
        .and_then(|elements| elements.checked_mul(ElementType::F16.size_bytes()))
        .expect("mixed L1 token readback offset must fit u64")
}

fn execute_reference_batch(
    fixture: &MixedReferenceFixture,
    cases: &[MixedReferenceCase],
    namespace: &str,
) -> MixedReferenceBatchResult {
    assert!(!cases.is_empty() && cases.len() <= 4);
    let executable = fixture.compilation.executable();
    let plan = executable.execution_plan();
    let admitted = cases
        .iter()
        .enumerate()
        .map(|(slot, case)| {
            let work = ResourceWorkShape::single(case.span.clone()).unwrap();
            let run_id = format!("run.reference.l1.{namespace}.{slot}");
            let request_id = format!("request.reference.l1.{namespace}.{slot}");
            let sequence = admit_sequence(&fixture.plan_resources, &work, &run_id, &request_id);
            (sequence.sequence_authority(), case.clone(), sequence)
        })
        .collect::<Vec<_>>();
    let sessions = admitted
        .iter()
        .map(|(_, _, sequence)| sequence.open_session().unwrap())
        .collect::<Vec<_>>();
    let batch = ExecutionBatchParticipants::new(sessions.iter().cloned().collect()).unwrap();
    let canonical_cases = batch
        .sessions()
        .iter()
        .map(|session| {
            admitted
                .iter()
                .find(|(authority, _, _)| *authority == session.sequence_authority())
                .map(|(_, case, _)| case.clone())
                .expect("canonical batch participant must map to its source case")
        })
        .collect::<Vec<_>>();
    let spans = canonical_cases
        .iter()
        .map(|case| case.span.clone())
        .collect::<Vec<_>>();
    let active_bindings = batch
        .sessions()
        .iter()
        .map(|session| TrustedActiveSequenceBinding::from_session(session).unwrap())
        .collect::<Vec<_>>();
    let lane = fixture.plan_resources.create_execution_lane().unwrap();
    let step = begin_step(&batch, &lane, &spans);
    let wave = prepare_wave(plan, &step, &spans);
    let token_wave_node = wave
        .nodes()
        .iter()
        .find(|node| node.node_id().as_str() == "node.reference.dense-linear.tokens")
        .expect("mixed L1 token node must be in the prepared wave");
    let token_ranges = token_wave_node
        .work_shape()
        .participant_token_ranges()
        .to_vec();
    assert_eq!(token_ranges.len(), canonical_cases.len());
    let batch_identity = OperationDispatch::bind_submission_wave_identity(
        executable,
        active_bindings.iter(),
        &wave,
        &lane,
    )
    .unwrap();

    let [fixed_node, token_node] = plan.payload().nodes() else {
        panic!("mixed L1 plan must contain fixed and token nodes")
    };
    let (_, token_input_lifetime) =
        activation_resource(plan, token_node, ResolvedValueRole::Input, 0);
    let (token_output_resource, token_output_lifetime) =
        activation_resource(plan, token_node, ResolvedValueRole::Output, 0);
    assert_eq!(token_input_lifetime, AllocationLifetime::Step);
    assert_eq!(token_output_lifetime, AllocationLifetime::Step);
    assert!(plan
        .payload()
        .terminal_output_resources()
        .contains(&token_output_resource));
    let terminal_slots = plan
        .payload()
        .memory()
        .dynamic_pools()
        .iter()
        .flat_map(|pool| pool.step_resource_slots())
        .filter(|slot| slot.resource_ids().contains(&token_output_resource))
        .collect::<Vec<_>>();
    assert_eq!(terminal_slots.len(), 1);
    assert_eq!(terminal_slots[0].kind(), StepResourceSlotKind::Dedicated);
    assert_eq!(
        terminal_slots[0].resource_ids(),
        std::slice::from_ref(&token_output_resource)
    );
    let fixed_bytes = encode_f16_rows(&vec![[0.25, -0.5]; MIXED_ROWS_USIZE], 0..MIXED_ROWS);
    let mut uploads = Vec::with_capacity(canonical_cases.len() * 2);
    for (participant_index, (case, token_range)) in
        canonical_cases.iter().zip(&token_ranges).enumerate()
    {
        let participant_index = u32::try_from(participant_index).unwrap();
        uploads.push(
            SubmissionWaveInputUpload::new(
                fixed_node.id().clone(),
                participant_index,
                0,
                0,
                HostTransferLayout::new(ElementType::F16, MIXED_ROWS * IN_FEATURES).unwrap(),
                fixed_bytes.clone(),
            )
            .unwrap(),
        );
        let source_range = case.span.immediate_token_range();
        uploads.push(
            SubmissionWaveInputUpload::new(
                token_node.id().clone(),
                participant_index,
                0,
                token_upload_logical_offset(token_range),
                HostTransferLayout::new(
                    ElementType::F16,
                    case.span.immediate_tokens() * IN_FEATURES,
                )
                .unwrap(),
                encode_f16_rows(&case.rows, source_range),
            )
            .unwrap(),
        );
    }
    let readbacks = CompletionReadbackBatchRequest::new(
        token_ranges
            .iter()
            .enumerate()
            .map(|(participant_index, token_range)| {
                CompletionReadbackRequest::new(
                    token_node.id().clone(),
                    u32::try_from(participant_index).unwrap(),
                    token_output_resource.clone(),
                    token_readback_logical_offset(token_output_lifetime, token_range),
                    HostTransferLayout::new(
                        ElementType::F16,
                        token_range.immediate_tokens() * OUT_FEATURES,
                    )
                    .unwrap(),
                )
                .unwrap()
            })
            .collect(),
    )
    .unwrap();
    let before = fixture.composition.runtime().snapshot();
    let reaper = CompletionReaper::new();
    let handle = OperationDispatch::encode_and_submit_wave_with_inputs(
        fixture.providers.providers(),
        executable,
        &batch_identity,
        active_bindings.iter(),
        DeviceTimingMode::Off,
        &uploads,
        wave,
        &lane,
        &reaper,
    )
    .unwrap();
    let receipt = match handle.wait_with_readbacks(readbacks).unwrap() {
        CompletionReadbackBatchObservation::Terminal(receipt) => receipt,
        other => panic!("mixed L1 batch readback did not terminate: {other:?}"),
    };
    let outputs = canonical_cases
        .iter()
        .zip(receipt.dispositions())
        .map(|(case, disposition)| match disposition {
            CompletionReadbackDisposition::Succeeded(output) => {
                (case.logical_index, output.bytes().to_vec())
            }
            other => panic!("mixed L1 participant readback failed: {other:?}"),
        })
        .collect::<BTreeMap<_, _>>();
    let after = fixture.composition.runtime().snapshot();
    let physical_submissions = after
        .submissions
        .checked_sub(before.submissions)
        .expect("reference runtime submission counter must be monotonic");
    assert_eq!(physical_submissions, 1);

    drop(receipt);
    drop(handle);
    drop(reaper);
    drop(batch_identity);
    drop(active_bindings);
    step.try_retire_normal().unwrap();
    drop(lane);
    drop(batch);
    for session in &sessions {
        session.try_complete().unwrap();
    }
    drop(sessions);
    drop(admitted);
    let pool_status = fixture.plan_resources.dynamic_pool_status().unwrap();
    assert!(pool_status.pools().iter().all(|pool| {
        let total = pool.live_occupancy().total();
        total.claim_count() == 0 && total.segment_count() == 0 && total.physical_bytes() == 0
    }));

    MixedReferenceBatchResult {
        outputs,
        physical_submissions,
    }
}

#[test]
fn tiny_real_safetensors_executes_through_reference_vnext_runtime() {
    let model_dir = tempfile::tempdir().unwrap();
    let weight_values = [2.0_f32, 1.0, -1.0, 3.0];
    let weight_bytes = weight_values
        .iter()
        .flat_map(|value| f16::from_f32(*value).to_bits().to_le_bytes())
        .collect::<Vec<_>>();
    let tensor = TensorView::new(Dtype::F16, vec![2, 2], &weight_bytes).unwrap();
    let weight_path = model_dir.path().join("model.safetensors");
    serialize_to_file(
        vec![("linear.weight".to_owned(), tensor)],
        &None::<HashMap<String, String>>,
        &weight_path,
    )
    .unwrap();
    let weight_source = SafetensorsArchive::open(model_dir.path()).unwrap();

    let family = TypedFamilyRegistration::new(TinyDenseFamily)
        .prepare(&json!({
            "rows": ROWS,
            "in_features": IN_FEATURES,
            "out_features": OUT_FEATURES
        }))
        .unwrap();
    let program_fingerprint = family.program().fingerprint().unwrap();
    let composition = ReferenceVNextComposition::create(id("device.reference.l1.0")).unwrap();
    let policy = runtime_policy(composition.runtime());
    let input_tensor = ProgramTensorSpec {
        dimensions: vec![ROWS, IN_FEATURES],
        element_type: ElementType::F16,
        layout: ResolvedTensorLayout::Contiguous,
    };
    let mut options = ProgramPlanCompileOptions::new(BTreeMap::from([
        (id("value.reference.input.fixed"), input_tensor.clone()),
        (id("value.reference.input.tokens"), input_tensor),
    ]))
    .unwrap();
    options.require_weight_materializer(composition.weight_materializer_id().clone());
    assert!(options.retain_completion_value(id("value.reference.output.tokens")));
    let compilation = ProgramPlanCompiler::compile_with_weight_materializers(
        &family,
        composition.catalog(),
        &policy,
        &composition.registry().planning(),
        composition.weight_materializers(),
        &options,
    )
    .unwrap();
    assert_eq!(family.program().fingerprint().unwrap(), program_fingerprint);
    let executable = compilation.executable();
    let plan = executable.execution_plan();
    let [fixed_node, token_node] = plan.payload().nodes() else {
        panic!("reference L1 plan must contain its fixed and token-projected nodes")
    };
    assert!(matches!(fixed_node.work(), NodeWorkContract::Fixed));
    assert!(matches!(token_node.work(), NodeWorkContract::Tokens { .. }));
    assert_eq!(token_node.work().token_projections().len(), 2);
    let providers = composition.registry().bind_plan(executable).unwrap();

    let provisioned = plan
        .provision_static(
            Arc::clone(composition.runtime()),
            id("request.reference.l1.provision"),
        )
        .unwrap();
    let permit = match provisioned.into_provisioning() {
        StaticProvisioning::Required(permit) => permit,
        StaticProvisioning::NoStatic(_) => panic!("tiny real weights require static provisioning"),
    };
    let identity = ResourceTransactionIdentity::for_admission(
        permit.binding(),
        id("run.reference.l1.provision"),
        id("transaction.reference.l1.provision"),
    );
    let driver = RuntimeResourceDriver::new(Arc::clone(composition.runtime())).unwrap();
    let reserved = ResourceTransaction::begin(driver, identity, permit)
        .unwrap()
        .reserve()
        .unwrap();
    let committed = match reserved.commit() {
        Ok(committed) => committed,
        Err(ResourceCommitTransitionError::Recoverable(error)) => {
            panic!("reference L1 static commit failed: {:?}", error.failure())
        }
        Err(ResourceCommitTransitionError::Poisoned(error)) => {
            panic!(
                "reference L1 static commit was indeterminate: {:?}",
                error.failure()
            )
        }
    };
    let pool_ids = committed
        .maintenance_controller()
        .pool_ids()
        .cloned()
        .collect::<Vec<_>>();
    for pool_id in pool_ids {
        committed
            .maintenance_controller()
            .initialize_pool(&pool_id)
            .unwrap();
    }
    let initialized = committed
        .initialize_static(
            &family,
            plan,
            &weight_source,
            StaticInitializationPolicy::new(1 << 20, 8).unwrap(),
        )
        .unwrap();
    assert_eq!(initialized.receipt().uploaded_component_count(), 1);
    assert_eq!(initialized.receipt().uploaded_bytes(), 8);
    assert_eq!(
        initialized.receipt().source_files(),
        &BTreeSet::from(["model.safetensors".to_owned()])
    );
    let plan_resources = match initialized.into_plan_runtime() {
        Ok(resources) => resources,
        Err(error) => panic!("reference L1 runtime handoff failed: {}", error.error()),
    };

    let (span, work) = tiny_work();
    let sequence = admit_sequence(
        &plan_resources,
        &work,
        "run.reference.l1.single",
        "request.reference.l1.single",
    );
    let session = sequence.open_session().unwrap();
    let active = TrustedActiveSequenceBinding::from_session(&session).unwrap();
    let batch = ExecutionBatchParticipants::new(vec![Arc::clone(&session)]).unwrap();
    let lane = plan_resources.create_execution_lane().unwrap();
    let step = begin_step(&batch, &lane, std::slice::from_ref(&span));
    let wave = prepare_wave(plan, &step, std::slice::from_ref(&span));
    let batch_identity = OperationDispatch::bind_submission_wave_identity(
        executable,
        std::iter::once(&active),
        &wave,
        &lane,
    )
    .unwrap();

    let input_values = [1.5_f32, -2.0];
    let input_bytes = input_values
        .iter()
        .flat_map(|value| f16::from_f32(*value).to_bits().to_le_bytes())
        .collect::<Vec<_>>();
    let fixed_upload = SubmissionWaveInputUpload::new(
        id("node.reference.dense-linear.fixed"),
        0,
        0,
        0,
        HostTransferLayout::new(ElementType::F16, 2).unwrap(),
        input_bytes.clone(),
    )
    .unwrap();
    let token_upload = SubmissionWaveInputUpload::new(
        id("node.reference.dense-linear.tokens"),
        0,
        0,
        0,
        HostTransferLayout::new(ElementType::F16, 2).unwrap(),
        input_bytes,
    )
    .unwrap();
    let reaper = CompletionReaper::new();
    let handle = OperationDispatch::encode_and_submit_wave_with_inputs(
        providers.providers(),
        executable,
        &batch_identity,
        std::iter::once(&active),
        DeviceTimingMode::Off,
        &[fixed_upload, token_upload],
        wave,
        &lane,
        &reaper,
    )
    .unwrap();
    let readback = plan
        .completion_checkpoint_readback_for_work(&id("value.reference.output.tokens"), 0, &work)
        .unwrap();
    let receipt = match handle.wait_with_readback(readback).unwrap() {
        CompletionReadbackObservation::Terminal(receipt) => receipt,
        other => panic!("reference L1 readback did not terminate: {other:?}"),
    };
    let output = match receipt.disposition() {
        CompletionReadbackDisposition::Succeeded(output) => output.bytes(),
        other => panic!("reference L1 readback failed: {other:?}"),
    };
    let output_values = output
        .chunks_exact(2)
        .map(|bytes| f16::from_bits(u16::from_le_bytes([bytes[0], bytes[1]])).to_f32())
        .collect::<Vec<_>>();
    assert_eq!(output_values, vec![1.0, -7.5]);
    let snapshot = composition.runtime().snapshot();
    assert!(snapshot.allocations >= 2);
    assert!(snapshot.live_allocations >= 2);
    assert!(snapshot.submissions >= 2);
    assert!(snapshot.commands >= 2);
    assert_eq!(snapshot.dense_linear_launches, 2);
    assert_eq!(snapshot.readback_bytes, 4);

    drop(receipt);
    drop(handle);
    drop(reaper);
    drop(batch_identity);
    drop(active);
    step.try_retire_normal().unwrap();
    drop(lane);
    drop(batch);
    session.try_complete().unwrap();
    drop(session);
    drop(sequence);
    drop(providers);
    let close_receipt = match PlanRuntimeResources::close(plan_resources) {
        Ok(PlanRuntimeCloseOutcome::Closed(receipt)) => receipt,
        Ok(PlanRuntimeCloseOutcome::Referenced {
            strong_count,
            deferred_cleanup,
            ..
        }) => panic!(
            "reference L1 runtime retained {strong_count} roots after teardown; deferred={deferred_cleanup:?}"
        ),
        Err(error) => panic!("reference L1 runtime close failed: {:?}", error.failure()),
    };
    assert_eq!(close_receipt.released_static_resources(), 1);
    let closed_snapshot = composition.runtime().snapshot();
    assert_eq!(closed_snapshot.live_allocations, 0);
}

#[test]
fn mixed_batch_matches_independent_per_request_reference() {
    let fixture = build_mixed_reference_fixture();
    let runtime_before = fixture.composition.runtime().snapshot();
    let mut unique_seed_fingerprints = BTreeSet::new();
    let mut final_chunk_cases = 0_usize;
    let mut non_final_chunk_cases = 0_usize;
    let mut decode_tail_cases = 0_usize;
    let mut mixed_submissions = 0_u64;
    let mut scalar_submissions = 0_u64;
    let mut participant_cases = 0_usize;

    for seed in MIXED_REFERENCE_SEEDS {
        let cases = mixed_cases(seed);
        let mut seed_material = Vec::new();
        for case in &cases {
            seed_material.extend_from_slice(case.span.fingerprint().as_bytes());
            seed_material.extend_from_slice(&encode_f16_rows(&case.rows, 0..MIXED_ROWS));
            match case.span_class {
                MixedSpanClass::FinalChunk => final_chunk_cases += 1,
                MixedSpanClass::NonFinalChunk => non_final_chunk_cases += 1,
                MixedSpanClass::DecodeTail => decode_tail_cases += 1,
            }
        }
        assert!(unique_seed_fingerprints.insert(sha256(&seed_material)));

        let mixed_namespace = format!("mixed.seed{seed:03}");
        let mixed = execute_reference_batch(&fixture, &cases, &mixed_namespace);
        mixed_submissions += mixed.physical_submissions;
        assert_eq!(mixed.outputs.len(), cases.len());
        for case in &cases {
            participant_cases += 1;
            let expected = analytic_output(case);
            let mixed_output = mixed
                .outputs
                .get(&case.logical_index)
                .expect("mixed L1 result must contain every participant");
            assert_eq!(
                mixed_output, &expected,
                "mixed L1 analytic mismatch for seed {seed} participant {}",
                case.logical_index
            );

            let scalar_namespace =
                format!("scalar.seed{seed:03}.participant{:02}", case.logical_index);
            let scalar =
                execute_reference_batch(&fixture, std::slice::from_ref(case), &scalar_namespace);
            scalar_submissions += scalar.physical_submissions;
            let scalar_output = scalar
                .outputs
                .get(&case.logical_index)
                .expect("scalar L1 result must contain its participant");
            assert_eq!(
                scalar_output, &expected,
                "scalar L1 analytic mismatch for seed {seed} participant {}",
                case.logical_index
            );
            assert_eq!(
                mixed_output, scalar_output,
                "mixed/scalar L1 mismatch for seed {seed} participant {}",
                case.logical_index
            );
        }
    }

    assert_eq!(unique_seed_fingerprints.len(), MIXED_REFERENCE_SEEDS.len());
    assert_eq!(mixed_submissions, MIXED_REFERENCE_SEEDS.len() as u64);
    assert_eq!(scalar_submissions, participant_cases as u64);
    assert!(final_chunk_cases > 0 && non_final_chunk_cases > 0 && decode_tail_cases > 0);
    let pool_status = fixture.plan_resources.dynamic_pool_status().unwrap();
    let final_live_claims = pool_status
        .pools()
        .iter()
        .map(|pool| pool.live_occupancy().total().claim_count())
        .sum::<u64>();
    assert_eq!(final_live_claims, 0);
    let runtime_after_batches = fixture.composition.runtime().snapshot();
    assert_eq!(
        runtime_after_batches.submissions - runtime_before.submissions,
        mixed_submissions + scalar_submissions
    );

    let MixedReferenceFixture {
        composition,
        compilation,
        providers,
        plan_resources,
    } = fixture;
    drop(providers);
    drop(compilation);
    let close_receipt = match PlanRuntimeResources::close(plan_resources) {
        Ok(PlanRuntimeCloseOutcome::Closed(receipt)) => receipt,
        Ok(PlanRuntimeCloseOutcome::Referenced {
            strong_count,
            deferred_cleanup,
            ..
        }) => panic!(
            "mixed L1 runtime retained {strong_count} roots after teardown; deferred={deferred_cleanup:?}"
        ),
        Err(error) => panic!("mixed L1 runtime close failed: {:?}", error.failure()),
    };
    assert_eq!(close_receipt.released_static_resources(), 1);
    let closed_snapshot = composition.runtime().snapshot();
    assert_eq!(closed_snapshot.live_allocations, 0);
}
