//! Generic ownership over a real nonzero Stored graph. The logical roles are
//! declared aliases of the tiny Qwen graph, not an assertion that it is an H3
//! connector/refiner. All installs, attachments and forwards are actual ones.
use super::*;
use crate::image_generation::denoising::fixture_pipeline;
use ferrum_kernels::backend::reference::ReferenceDeviceRuntime as Runtime;
use std::sync::atomic::{AtomicUsize, Ordering};

mod auxiliary_suspension;
mod trace_capacity;

#[derive(Clone)]
struct Sources {
    qwen: QwenImage21SourceAdapter,
    reads: Arc<AtomicUsize>,
}
impl Sources {
    fn mapped(d: &ImageStageDeclaration) -> ImageStageDeclaration {
        ImageStageDeclaration {
            stage: match d.stage {
                ImageStage::TextConnector | ImageStage::TextRefiner => ImageStage::Denoiser,
                other => other,
            },
            ..*d
        }
    }
}
impl ImagePipelineSourceAdapter for Sources {
    type Source<'a> = QwenImage21StageSource<'a>;
    fn stage_source<'a>(
        &'a self,
        declaration: &ImageStageDeclaration,
        profile: &NumericalProfileId,
        host: &'a HostMemoryAccount,
    ) -> anyhow::Result<Option<Self::Source<'a>>> {
        self.reads.fetch_add(1, Ordering::AcqRel);
        self.qwen
            .stage_source(&Self::mapped(declaration), profile, host)
    }
    fn weight_owner(&self, stage: ImageStage) -> Option<ImageWeightOwnerKey> {
        match stage {
            ImageStage::TextConnector => Some(ImageWeightOwnerKey::TextConnector),
            ImageStage::TextRefiner => Some(ImageWeightOwnerKey::Denoiser),
            other => self.qwen.weight_owner(other),
        }
    }
    fn supports_stage(&self, stage: ImageStage) -> bool {
        stage != ImageStage::VisionEncoder
            && (stage == ImageStage::Noise || self.weight_owner(stage).is_some())
    }
    fn prepare_host_backing(
        &self,
        declaration: &ImageStageDeclaration,
        profile: &NumericalProfileId,
        family: &PreparedModelFamily,
        plan: &ExecutionPlan,
        host: &HostMemoryAccount,
    ) -> anyhow::Result<(usize, HostBackingPreparation)> {
        self.qwen
            .prepare_host_backing(&Self::mapped(declaration), profile, family, plan, host)
    }
}
type Execution = ImagePipelineExecution<Runtime, Sources>;
fn fixture() -> (
    Execution,
    QwenImage21TransformerConfig,
    tempfile::TempDir,
    Arc<AtomicUsize>,
) {
    let (pipeline, directory) = fixture_pipeline();
    let model = QwenImage21TransformerConfig::new(
        pipeline.execution.sources.assets.transformer.clone(),
        QwenImage21T2iLayout::new(2, 2, 2, vec![2]).unwrap(),
    )
    .unwrap();
    let QwenImage21Pipeline { execution, .. } = pipeline;
    // Transfer the fresh, empty owner in place: retain its exact Runtime,
    // Host, policy and original actor. No second runtime/lane is constructed.
    let ImagePipelineExecution {
        runtime,
        host,
        execution_host,
        registry,
        catalog,
        materializers,
        mut config,
        policy,
        next_job,
        continuous_trace,
        stopped,
        service_preparation_identity,
        plan_cache,
        preparation_only,
        resident_stages: old,
        device_opaque_allowance_bytes,
        sources: qwen,
        serial_actor,
    } = execution;
    assert!(!old.has_weights(ImageStage::Denoiser));
    assert!(!old.has_weights(ImageStage::TextEncoder));
    assert_eq!(runtime.snapshot().live_allocations, 0);
    let reads = Arc::new(AtomicUsize::new(0));
    let sources = Sources {
        qwen,
        reads: reads.clone(),
    };
    config.residency = ImageResidencyPolicy::Adaptive;
    let execution = ImagePipelineExecution {
        runtime,
        host,
        execution_host,
        registry,
        catalog,
        materializers,
        config,
        policy,
        next_job,
        continuous_trace,
        stopped,
        service_preparation_identity,
        plan_cache,
        preparation_only,
        resident_stages: residency::ResidentStages::new(&sources),
        device_opaque_allowance_bytes,
        sources,
        serial_actor,
    };
    (execution, model, directory, reads)
}
fn prepared(
    execution: &Execution,
    config: &QwenImage21TransformerConfig,
    stage: ImageStage,
    attachment: ImageStageAttachment,
) -> PreparedStage<Runtime> {
    let family = TypedFamilyRegistration::new(QwenImage21TransformerFamily::new().unwrap())
        .prepare_with_profile(
            &serde_json::to_value(config).unwrap(),
            &NumericalProfileId::new(QwenImage21TransformerFamily::REFERENCE_PROFILE_ID).unwrap(),
        )
        .unwrap();
    execution
        .compile_stage(
            ImageStageDeclaration {
                stage,
                role: ImageStagePlanRole::WholeStage,
                batch_size: 1,
                weight_owner: execution.sources.weight_owner(stage),
                attachment,
            },
            family,
            &config.compile_options().unwrap(),
        )
        .unwrap()
}
fn inputs(
    host: &HostMemoryAccount,
    config: &QwenImage21TransformerConfig,
) -> Vec<ManagedHostTensor> {
    let specs = config.input_tensor_specs().unwrap();
    let mut values = Vec::new();
    for name in [
        QwenImage21TransformerFamily::LATENTS_ID,
        QwenImage21TransformerFamily::TEXT_ID,
    ] {
        let spec = specs.get(&id::<ProgramValueId>(name).unwrap()).unwrap();
        let spec = ResolvedTensorSpec::new(
            spec.dimensions.clone(),
            spec.element_type,
            spec.layout.clone(),
        )
        .unwrap();
        let mut bytes = host
            .allocate_zeroed(
                HostMemoryClass::InputPreprocess,
                spec.minimum_storage_bytes().unwrap() as usize,
            )
            .unwrap();
        for (i, out) in bytes.as_mut().chunks_exact_mut(2).enumerate() {
            let value = half::bf16::from_f32(((i * 3) % 19) as f32 * 0.125 - 1.125);
            out.copy_from_slice(&value.to_bits().to_le_bytes());
        }
        values.push(ManagedHostTensor::new(id(name).unwrap(), spec, bytes.freeze()).unwrap());
    }
    values.push(config.managed_time_features(host, 500.0).unwrap());
    values.push(config.managed_rotary_frequencies(host).unwrap());
    values
}
fn run(
    execution: &mut Execution,
    stage: PreparedStage<Runtime>,
    job: u64,
    inputs: Vec<ManagedHostTensor>,
    receipts: &mut Vec<ImageStageReceipt>,
) -> (Vec<u8>, ImageStageExecutionReceipt) {
    let active = execution
        .start_stage(stage, job, receipts)
        .unwrap_or_else(|e| panic!("actual install/attach: {e}"));
    let (active, outputs) = active
        .execute_on_actor(execution.serial_actor.as_mut().unwrap(), job, inputs)
        .unwrap_or_else(|e| panic!("actual forward: {e}"));
    let output = only_output(outputs, QwenImage21TransformerFamily::VELOCITY_ID).unwrap();
    let bytes = output.bytes().bytes().to_vec();
    drop(output);
    let receipt = active.receipt.clone();
    execution
        .finish_stage(active, receipts)
        .unwrap_or_else(|e| panic!("actual fence/retire: {e}"));
    (bytes, receipt)
}

#[test]
fn actual_snapshot_pin_keeps_refiner_attachment_open_and_counts_the_owner_once() {
    let (mut execution, config, _source, reads) = fixture();
    let scope = execution
        .host
        .reserve_scope(&[
            (HostMemoryClass::InputPreprocess, 64 << 10),
            (HostMemoryClass::Output, 64 << 10),
            (HostMemoryClass::TransferStaging, 64 << 10),
            (HostMemoryClass::WeightBacking, 0),
        ])
        .unwrap();
    execution.execution_host = Some(scope.account());
    let first = prepared(
        &execution,
        &config,
        ImageStage::Denoiser,
        ImageStageAttachment::InitializeShared,
    );
    let refiner = prepared(
        &execution,
        &config,
        ImageStage::TextRefiner,
        ImageStageAttachment::RequireShared,
    );
    let report = execution
        .capacity_with_prefix(&[&first, &refiner], None)
        .unwrap();
    assert_eq!(report.retained_static_bytes, first.plan.static_bytes);
    let mut receipts = Vec::new();
    let x = inputs(&scope.account(), &config);
    let (a, first_receipt) = run(&mut execution, first, 0, x.clone(), &mut receipts);
    assert!(!first_receipt.weights_reused);
    assert!(a.iter().any(|&v| v != 0));
    let pin = execution
        .pin_weight_owner(ImageWeightOwnerKey::Denoiser)
        .unwrap();
    assert!(!execution
        .release_weight_owner(ImageWeightOwnerKey::Denoiser, &mut receipts)
        .unwrap_or_else(|e| panic!("pin: {e}")));
    pin.weights()
        .validate_for(refiner.executable().execution_plan(), execution.runtime())
        .unwrap();
    let before_reads = reads.load(Ordering::Acquire);
    let (b, refiner_receipt) = run(&mut execution, refiner, 1, x.clone(), &mut receipts);
    assert!(refiner_receipt.weights_reused);
    assert_eq!(
        reads.load(Ordering::Acquire),
        before_reads,
        "RequireShared never reads a cold source"
    );
    assert_eq!(
        a, b,
        "separate logical role executes the same actual Stored graph"
    );
    drop(pin);
    assert!(execution
        .release_weight_owner(ImageWeightOwnerKey::Denoiser, &mut receipts)
        .unwrap_or_else(|e| panic!("release: {e}")));
    assert_eq!(execution.runtime.snapshot().live_allocations, 0);
    drop(x);
    execution.execution_host = None;
    drop(scope);
    execution
        .shutdown()
        .unwrap_or_else(|e| panic!("shutdown: {e}"));
    assert_eq!(execution.host.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn composed_failure_retains_actual_peer_sessions_snapshot_and_host_grants() {
    let (mut execution, config, _source, _) = fixture();
    let scope = execution
        .host
        .reserve_scope(&[
            (HostMemoryClass::InputPreprocess, 64 << 10),
            (HostMemoryClass::Output, 64 << 10),
            (HostMemoryClass::TransferStaging, 64 << 10),
            (HostMemoryClass::WeightBacking, 0),
        ])
        .unwrap();
    execution.execution_host = Some(scope.account());
    let installed = prepared(
        &execution,
        &config,
        ImageStage::Denoiser,
        ImageStageAttachment::InitializeShared,
    );
    let refinement = prepared(
        &execution,
        &config,
        ImageStage::TextRefiner,
        ImageStageAttachment::RequireShared,
    );
    let mut receipts = Vec::new();
    let peer = execution
        .start_stage(installed, 0, &mut receipts)
        .unwrap_or_else(|e| panic!("install: {e}"));
    let pin = execution
        .pin_weight_owner(ImageWeightOwnerKey::Denoiser)
        .unwrap();
    let active = execution
        .start_stage(refinement, 1, &mut receipts)
        .unwrap_or_else(|e| panic!("attach: {e}"));
    let before = execution.runtime.snapshot();
    let host_before = execution.host.usage().unwrap().committed_bytes();
    assert!(before.live_allocations > 0);
    let source =
        match active.execute_on_actor(execution.serial_actor.as_mut().unwrap(), 1, Vec::new()) {
            Err(failure) => failure,
            Ok(_) => panic!("missing physical inputs must fail"),
        };
    let failure = ImageGenerationFailure::Pipeline {
        source: Box::new(source),
        stages: vec![peer],
        weights: vec![pin],
        host_scopes: vec![scope],
        host_backing: Vec::new(),
    };
    execution.execution_host = None;
    assert_eq!(execution.runtime.snapshot().submissions, before.submissions);
    assert_eq!(
        execution.runtime.snapshot().live_allocations,
        before.live_allocations
    );
    assert_eq!(
        execution.host.usage().unwrap().committed_bytes(),
        host_before
    );
    assert!(!execution
        .release_weight_owner(ImageWeightOwnerKey::Denoiser, &mut receipts)
        .unwrap_or_else(|e| panic!("pin: {e}")));
    // Recover the real owners: a diagnostic string cannot provide any of them.
    let ImageGenerationFailure::Pipeline {
        source,
        stages,
        weights,
        host_scopes,
        host_backing,
    } = failure
    else {
        unreachable!()
    };
    let ImageGenerationFailure::Execution { source, active } = *source else {
        panic!("original execution failure was not preserved")
    };
    assert!(matches!(source, TensorExecutionError::Runtime(_)));
    drop(source);
    // This rejected input never submitted work, so both sessions can close
    // explicitly. Indeterminate execution failures still expose their reaper.
    active
        .close()
        .unwrap_or_else(|e| panic!("close failed stage: {e}"));
    for peer in stages {
        peer.close().unwrap_or_else(|e| panic!("close peer: {e}"));
    }
    assert!(!execution
        .release_weight_owner(ImageWeightOwnerKey::Denoiser, &mut receipts)
        .unwrap_or_else(|e| panic!("pin after close: {e}")));
    drop(weights);
    assert!(execution
        .release_weight_owner(ImageWeightOwnerKey::Denoiser, &mut receipts)
        .unwrap_or_else(|e| panic!("release: {e}")));
    drop(host_backing);
    drop(host_scopes);
    execution.shutdown().unwrap();
    assert_eq!(execution.runtime.snapshot().live_allocations, 0);
    assert_eq!(execution.host.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn absent_shared_owner_and_wrong_source_binding_fail_before_any_device_work() {
    let (mut execution, config, _source, reads) = fixture();
    let refiner = prepared(
        &execution,
        &config,
        ImageStage::TextRefiner,
        ImageStageAttachment::RequireShared,
    );
    assert_eq!(execution.stage_host_backing(&refiner).unwrap().0, 0);
    let before = execution.runtime.snapshot();
    let mut receipts = Vec::new();
    assert!(matches!(
        execution.start_stage(refiner.clone(), 0, &mut receipts),
        Err(ImageGenerationFailure::Preflight(_))
    ));
    let mut wrong = *refiner.declaration();
    wrong.weight_owner = Some(ImageWeightOwnerKey::TextConnector);
    assert!(execution
        .compile_stage(
            wrong,
            refiner.family().clone(),
            &config.compile_options().unwrap()
        )
        .is_err());
    let unsupported = ImageStageDeclaration {
        stage: ImageStage::VisionEncoder,
        weight_owner: execution.sources.weight_owner(ImageStage::VisionEncoder),
        attachment: ImageStageAttachment::MayInitializeStored,
        ..*refiner.declaration()
    };
    assert!(execution
        .compile_stage(
            unsupported,
            refiner.family().clone(),
            &config.compile_options().unwrap()
        )
        .is_err());
    assert_eq!(execution.runtime.snapshot().submissions, before.submissions);
    assert_eq!(execution.runtime.snapshot().allocations, before.allocations);
    assert_eq!(reads.load(Ordering::Acquire), 0);
    assert!(receipts.is_empty());
    execution
        .shutdown()
        .unwrap_or_else(|e| panic!("shutdown: {e}"));
    assert_eq!(execution.host.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn source_budget_prepared_while_warm_still_cold_installs_after_actual_eviction() {
    use safetensors::tensor::{Dtype, TensorView};
    let (mut execution, config, directory, reads) = fixture();
    execution.host = HostMemoryAccount::with_allocation_alignment(
        HostMemoryLimits::new(
            1 << 20,
            HostMemoryClassLimits {
                input_preprocess_bytes: 256 << 10,
                output_bytes: 256 << 10,
                transfer_staging_bytes: 256 << 10,
                weight_backing_bytes: 64 << 10,
            },
        )
        .unwrap(),
        64,
    )
    .unwrap();
    let stage = prepared(
        &execution,
        &config,
        ImageStage::Denoiser,
        ImageStageAttachment::MayInitializeStored,
    );
    // Same deterministic small weights, with real F32 physical carriers that
    // the Qwen adapter explicitly normalizes into the declared BF16 schema.
    let original = execution.sources.qwen.assets.sources.transformer.clone();
    let tensors = stage
        .family()
        .stored_weight_schema()
        .unwrap()
        .components
        .iter()
        .map(|component| {
            let [name] = component.external_names.as_slice() else {
                panic!("fixture needs one tensor per component")
            };
            let source = original.tensor(name).unwrap();
            assert_eq!(source.element_type(), Some(ElementType::Bf16));
            let data = source
                .bytes()
                .chunks_exact(2)
                .flat_map(|b| {
                    half::bf16::from_bits(u16::from_le_bytes([b[0], b[1]]))
                        .to_f32()
                        .to_le_bytes()
                })
                .collect::<Vec<_>>();
            (
                name.clone(),
                component
                    .dimensions
                    .iter()
                    .map(|&n| n as usize)
                    .collect::<Vec<_>>(),
                data,
            )
        })
        .collect::<Vec<_>>();
    let path = directory.path().join("cast-source.safetensors");
    safetensors::tensor::serialize_to_file(
        tensors.iter().map(|(name, shape, bytes)| {
            (
                name.as_str(),
                TensorView::new(Dtype::F32, shape.clone(), bytes).unwrap(),
            )
        }),
        &None,
        &path,
    )
    .unwrap();
    Arc::get_mut(&mut execution.sources.qwen.assets)
        .unwrap()
        .sources
        .transformer = Arc::new(
        SafetensorsArchive::open_single_file(directory.path(), "cast-source.safetensors").unwrap(),
    );
    let (cold_bytes, cold_retention) = execution.stage_host_backing(&stage).unwrap();
    assert!(
        cold_bytes > 0,
        "real source normalization needs managed backing"
    );
    let grant = |bytes| {
        execution
            .host
            .reserve_scope(&[
                (HostMemoryClass::InputPreprocess, 64 << 10),
                (HostMemoryClass::Output, 64 << 10),
                (HostMemoryClass::TransferStaging, 64 << 10),
                (HostMemoryClass::WeightBacking, bytes),
            ])
            .unwrap()
    };
    let first_scope = grant(cold_bytes);
    execution.execution_host = Some(first_scope.account());
    let mut receipts = Vec::new();
    let x = inputs(&first_scope.account(), &config);
    let (a, _) = run(&mut execution, stage.clone(), 0, x, &mut receipts);
    let (warm_bytes, warm_retention) = execution.stage_host_backing(&stage).unwrap();
    assert_eq!(
        warm_bytes, cold_bytes,
        "native warmth alone cannot reduce future source credit"
    );
    execution.execution_host = None;
    drop(first_scope);
    // Build a fresh grant from the budget observed while native weights were
    // warm, then really retire those weights before reusing that preparation.
    let replacement_scope = execution
        .host
        .reserve_scope(&[
            (HostMemoryClass::InputPreprocess, 64 << 10),
            (HostMemoryClass::Output, 64 << 10),
            (HostMemoryClass::TransferStaging, 64 << 10),
            (HostMemoryClass::WeightBacking, warm_bytes),
        ])
        .unwrap();
    assert!(execution
        .release_weight_owner(ImageWeightOwnerKey::Denoiser, &mut receipts)
        .unwrap_or_else(|e| panic!("evict: {e}")));
    assert_eq!(execution.runtime.snapshot().live_allocations, 0);
    let before = reads.load(Ordering::Acquire);
    execution.execution_host = Some(replacement_scope.account());
    let x = inputs(&replacement_scope.account(), &config);
    let (b, receipt) = run(&mut execution, stage, 1, x, &mut receipts);
    assert!(!receipt.weights_reused);
    assert!(
        reads.load(Ordering::Acquire) > before,
        "actual cold source was reread"
    );
    assert_eq!(a, b);
    execution.execution_host = None;
    drop((replacement_scope, cold_retention, warm_retention));
    execution
        .shutdown()
        .unwrap_or_else(|e| panic!("shutdown: {e}"));
    assert_eq!(execution.host.usage().unwrap().committed_bytes(), 0);
    assert_eq!(execution.runtime.snapshot().live_allocations, 0);
}

#[test]
fn retained_executor_rebinds_a_fresh_exact_output_grant_after_old_scope_closes() {
    let (mut execution, config, _source, reads) = fixture();
    // One actual output exhausts the parent Output class. A reused executor
    // cannot bypass the new scope by reserving another parent readback.
    let stage = prepared(
        &execution,
        &config,
        ImageStage::Denoiser,
        ImageStageAttachment::MayInitializeStored,
    );
    let output_ports = host_budget::ports(&stage).unwrap().outputs();
    assert_eq!(output_ports.len(), 1);
    let logical_output =
        usize::try_from(output_ports[0].tensor().minimum_storage_bytes().unwrap()).unwrap();
    let alignment = 64;
    let physical_output = HostMemoryAccount::with_allocation_alignment(
        HostMemoryLimits::new(
            1 << 20,
            HostMemoryClassLimits {
                input_preprocess_bytes: 1 << 20,
                output_bytes: 1 << 20,
                transfer_staging_bytes: 1 << 20,
                weight_backing_bytes: 1 << 20,
            },
        )
        .unwrap(),
        alignment,
    )
    .unwrap()
    .backing_bytes(logical_output)
    .unwrap();
    execution.host = HostMemoryAccount::with_allocation_alignment(
        HostMemoryLimits::new(
            1 << 20,
            HostMemoryClassLimits {
                input_preprocess_bytes: 128 << 10,
                output_bytes: physical_output,
                transfer_staging_bytes: 64 << 10,
                weight_backing_bytes: 64 << 10,
            },
        )
        .unwrap(),
        alignment,
    )
    .unwrap();
    let input_bytes = host_budget::sum_ports(
        &execution.host,
        host_budget::ports(&stage).unwrap().inputs().iter(),
    )
    .unwrap();
    assert_eq!(
        host_budget::sum_ports(
            &execution.host,
            host_budget::ports(&stage).unwrap().outputs().iter(),
        )
        .unwrap(),
        physical_output,
    );
    let claims = [
        (HostMemoryClass::InputPreprocess, input_bytes),
        (HostMemoryClass::Output, physical_output),
        (HostMemoryClass::TransferStaging, 64 << 10),
        (HostMemoryClass::WeightBacking, 0),
    ];
    let old_scope = execution.host.reserve_scope(&claims).unwrap();
    let old_view = old_scope.account();
    execution.execution_host = Some(old_view.clone());
    let mut receipts = Vec::new();
    let x = inputs(&old_view, &config);
    let (expected, first) = run(&mut execution, stage.clone(), 0, x, &mut receipts);
    assert!(!first.session_reused);
    assert!(expected.iter().any(|&v| v != 0));
    execution.execution_host = None;
    drop(old_scope);
    assert!(old_view
        .allocate_zeroed(HostMemoryClass::Output, logical_output)
        .is_err());
    assert_eq!(execution.host.usage().unwrap().committed_bytes(), 0);

    let scope = execution.host.reserve_scope(&claims).unwrap();
    execution.execution_host = Some(scope.account());
    let x = inputs(&scope.account(), &config);
    let before_reads = reads.load(Ordering::Acquire);
    let active = execution
        .start_stage(stage, 1, &mut receipts)
        .unwrap_or_else(|e| panic!("reuse: {e}"));
    assert!(active.receipt.session_reused);
    assert!(active.receipt.weights_reused);
    let (active, outputs) = active
        .execute_on_actor(execution.serial_actor.as_mut().unwrap(), 1, x)
        .unwrap_or_else(|e| panic!("forward under exact new Output grant: {e}"));
    let output = only_output(outputs, QwenImage21TransformerFamily::VELOCITY_ID).unwrap();
    assert_eq!(expected, output.bytes().bytes());
    assert_eq!(reads.load(Ordering::Acquire), before_reads);
    assert_eq!(
        execution
            .host
            .usage()
            .unwrap()
            .class_allocated_bytes(HostMemoryClass::Output),
        physical_output,
    );
    assert!(scope
        .account()
        .allocate_zeroed(HostMemoryClass::Output, logical_output)
        .is_err());
    drop(output);
    // Dropping the readback refunds the new scope's credit. Its reservation
    // still blocks the parent, so success here proves the refund destination.
    assert!(execution
        .host
        .allocate_zeroed(HostMemoryClass::Output, logical_output)
        .is_err());
    let refunded = scope
        .account()
        .allocate_zeroed(HostMemoryClass::Output, logical_output)
        .unwrap();
    drop(refunded);
    execution
        .finish_stage(active, &mut receipts)
        .unwrap_or_else(|e| panic!("retire: {e}"));
    execution.execution_host = None;
    drop(scope);
    assert_eq!(execution.host.usage().unwrap().committed_bytes(), 0);
    execution
        .shutdown()
        .unwrap_or_else(|e| panic!("shutdown: {e}"));
    assert_eq!(execution.runtime.snapshot().live_allocations, 0);
    assert_eq!(execution.host.usage().unwrap().committed_bytes(), 0);
}
