use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs;
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
const IN_FEATURES: u64 = 2;
const OUT_FEATURES: u64 = 2;
const MAX_MAINTENANCE_ATTEMPTS: usize = 3;

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
        if config.rows != ROWS
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
                .try_admit_request(
                    request.clone(),
                    id("run.reference.l1"),
                    id("request.reference.l1"),
                )
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
    span: &TokenSpanWork,
) -> Arc<StepResourceLease<ReferenceDeviceRuntime>> {
    let request = StepResourceAdmissionRequest::new(
        batch.bind_work_shape(vec![span.clone()]).unwrap(),
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
    span: &TokenSpanWork,
) -> PreparedStepSubmissionWave<ReferenceDeviceRuntime> {
    let requests = plan
        .payload()
        .nodes()
        .iter()
        .map(|node| {
            InvocationResourceAdmissionRequest::for_all_step_participants(
                node.id().clone(),
                step.bind_all_invocation_work_shape(vec![span.clone()])
                    .unwrap(),
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
    let weight_file_bytes = fs::read(&weight_path).unwrap();
    let weight_file_sha256 = sha256(&weight_file_bytes);
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
    let sequence = admit_sequence(&plan_resources, &work);
    let session = sequence.open_session().unwrap();
    let active = TrustedActiveSequenceBinding::from_session(&session).unwrap();
    let batch = ExecutionBatchParticipants::new(vec![Arc::clone(&session)]).unwrap();
    let lane = plan_resources.create_execution_lane().unwrap();
    let step = begin_step(&batch, &lane, &span);
    let wave = prepare_wave(plan, &step, &span);
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
    let output_sha256 = sha256(output);

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

    println!(
        "FERRUM RUNTIME VNEXT G02 L1 TEST PASS: weight_sha256={weight_file_sha256} output_sha256={output_sha256} allocations={} released_static_resources={} live_allocations_after_close={} submissions={} commands={}",
        snapshot.allocations,
        close_receipt.released_static_resources(),
        closed_snapshot.live_allocations,
        snapshot.submissions,
        snapshot.commands
    );
}
