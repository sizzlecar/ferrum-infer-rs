use super::artifacts::{inspect, routing, save};
use super::{arguments::Arguments, package, retained_failure::Failure};
use ferrum_execution::{
    component::{
        CompiledTensorComponent, ComponentInitializationConfig, ComponentWeightSource,
        TensorComponentPolicy,
    },
    TensorExecutorConfig,
};
use ferrum_kernels::backend::cuda::{
    vnext_ops::{cuda_tensor_runtime_config, cuda_vnext_tensor_providers, CudaTensorOperation},
    vnext_runtime::{CudaDeviceRuntime, CudaTensorRuntimeConfig},
};
use ferrum_models::fast_h3::*;
use ferrum_runtime::*;

use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    fs::OpenOptions,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    time::Instant,
};

fn id<T: TryFrom<String, Error = VNextError>>(s: impl Into<String>) -> T {
    T::try_from(s.into()).expect("fixed diagnostic ID")
}
fn failure<E: std::fmt::Display + 'static>(stage: &'static str, error: E) -> Failure {
    Failure::retain(stage, error, ())
}
struct CountedSource<'a> {
    inner: FastH3TimeConditioningSource<'a>,
    reads: AtomicUsize,
}
impl WeightComponentSource for CountedSource<'_> {
    fn component<'a>(
        &'a self,
        c: &WeightComponentSpec,
    ) -> std::result::Result<WeightComponentPayload<'a>, VNextError> {
        self.reads.fetch_add(1, Ordering::Relaxed);
        self.inner.component(c)
    }
}
struct Phase {
    _runtime: Arc<CudaDeviceRuntime>,
    _registry: Arc<OperationRuntimeRegistry<CudaDeviceRuntime>>,
    _compiled: Arc<CompiledTensorComponent<CudaDeviceRuntime>>,
    _package: Arc<FastH3TransformerPackage>,
    input: ManagedHostTensor,
}

pub(super) fn run(args: Arguments) -> std::result::Result<Value, Failure> {
    let checkpoint = package::load_checkpoint(
        &args.model_dir,
        FastH3MetadataLimits {
            max_file_bytes: args.max_metadata_bytes,
            max_pdd_steps: args.max_pdd_steps,
        },
    )
    .map_err(|e| failure("checkpoint", e))?;
    let sampling = FastH3PddPlan::new(&checkpoint).map_err(|e| failure("schedule", e))?;
    let schedule =
        FastH3ConditioningSchedule::all_modalities(&sampling).map_err(|e| failure("clock", e))?;
    let package = Arc::new(
        package::load(
            &args.model_dir,
            FastH3Nvfp4Profile::DiTWithVsa,
            args.max_metadata_bytes,
            FastH3WeightLimits {
                maximum_tensors: args.max_tensors,
                maximum_source_bytes: args.max_source_bytes,
                maximum_skip_metadata_bytes: args.max_metadata_bytes,
            },
        )
        .map_err(|e| failure("package", e))?,
    );
    if args.first_block + args.block_count > package.config().layers() {
        return Err(failure("block range", "selected range exceeds checkpoint"));
    }
    // Refuse to mix partial tables with a prior run. Completion is published
    // only after every selected block has a successful close receipt.
    std::fs::create_dir(&args.output_dir).map_err(|e| failure("new output directory", e))?;
    let host = HostMemoryAccount::new(
        HostMemoryLimits::new(
            args.host_bytes,
            HostMemoryClassLimits {
                input_preprocess_bytes: args.host_bytes,
                transfer_staging_bytes: args.host_bytes,
                output_bytes: args.host_bytes,
                weight_backing_bytes: args.host_bytes,
            },
        )
        .map_err(|e| failure("Host capacity", e))?,
    );
    let device_index = usize::try_from(args.device).map_err(|e| failure("device index", e))?;
    let device =
        cuda_tensor_runtime_config(device_index, id(format!("device.cuda.{}", args.device)))
            .map_err(|e| failure("runtime descriptor", e))?;
    let runtime = Arc::new(
        CudaDeviceRuntime::new_tensor_eager(CudaTensorRuntimeConfig {
            device,
            usable_memory_bytes: args.device_bytes,
            bootstrap_memory_bytes: args.bootstrap_bytes,
            stream_memory_bytes: args.stream_bytes,
            maximum_streams: 1,
            maximum_in_flight_per_stream: 1,
            maximum_commands_per_submission: 64,
            opaque_memory_profile: "fast-h3-time-precompute-v1".into(),
        })
        .map_err(|e| failure("runtime", e))?,
    );
    let providers = cuda_vnext_tensor_providers(
        &runtime,
        &[
            CudaTensorOperation::AffineLinearF32,
            CudaTensorOperation::SiluF32,
            CudaTensorOperation::CastF32ToBf16,
            CudaTensorOperation::AffineLinear,
        ],
    )
    .map_err(|e| Failure::retain("providers", e, runtime.clone()))?;
    let contracts: Vec<Box<dyn OperationContract>> = vec![
        Box::new(affine_linear_f32_contract().map_err(|e| failure("contract", e))?),
        Box::new(silu_f32_contract().map_err(|e| failure("contract", e))?),
        Box::new(cast_f32_to_bf16_contract().map_err(|e| failure("contract", e))?),
        Box::new(affine_linear_bf16_contract().map_err(|e| failure("contract", e))?),
    ];
    let registry = Arc::new(
        OperationRuntimeRegistry::new(contracts, providers)
            .map_err(|e| Failure::retain("registry", e, runtime.clone()))?,
    );
    let materializers =
        WeightMaterializerRegistry::identity_only().map_err(|e| failure("materializers", e))?;
    let d = runtime.descriptor();
    let engine = EngineProviderDescriptor::new(
        id("provider.engine.cuda.fast_h3.time_precompute"),
        ContractVersion::new(1, 0),
        d.runtime_implementation_fingerprint.clone(),
        d.id.clone(),
        d.capabilities.clone(),
    )
    .map_err(|e| failure("engine", e))?;
    let catalog = materializers
        .augment_catalog(
            registry
                .capability_catalog(d.clone(), vec![engine])
                .map_err(|e| failure("catalog", e))?,
        )
        .map_err(|e| failure("catalog", e))?;
    let policy = TensorComponentPolicy::new(
        args.device_bytes,
        d.dynamic_storage_profiles.iter().copied().collect(),
    )
    .map_err(|e| failure("policy", e))?;
    let limits = FastH3TimeConditioningLimits {
        maximum_time_rows: schedule.timesteps().len() as u64,
        maximum_metadata_bytes: args.max_metadata_bytes,
        maximum_stored_bytes: args.device_bytes,
    };
    let mut blocks = Vec::new();
    for block in args.first_block..args.first_block + args.block_count {
        let config = FastH3TimeConditioningConfig::from_package(
            &package,
            block,
            schedule.timesteps().len() as u64,
            limits,
        )
        .map_err(|e| failure("time configuration", e))?;
        let family = TypedFamilyRegistration::new(
            FastH3TimeConditioningFamily::new(limits).map_err(|e| failure("family", e))?,
        )
        .prepare_with_profile(
            &serde_json::to_value(&config).map_err(|e| failure("configuration", e))?,
            &NumericalProfileId::new(FastH3TimeConditioningFamily::PROFILE_ID)
                .map_err(|e| failure("profile ID", e))?,
        )
        .map_err(|e| failure("family", e))?;
        let ModelWeights::Stored(schema) = family.weights() else {
            return Err(failure("schema", "Stored required"));
        };
        let source = CountedSource {
            inner: FastH3TimeConditioningSource::new(&package, &config, &host, limits)
                .map_err(|e| failure("source", e))?,
            reads: AtomicUsize::new(0),
        };
        let mut components = Vec::new();
        let mut selected_bytes = 0u64;
        for c in &schema.components {
            let payload = source
                .component(c)
                .map_err(|e| failure("source identity", e))?;
            selected_bytes = selected_bytes
                .checked_add(c.physical_bytes().map_err(|e| failure("source bytes", e))?)
                .ok_or_else(|| failure("source bytes", "overflow"))?;
            components.push(json!({"id":c.id,"external_names":payload.external_names(),"files":payload.source_files(),
                "encoding":c.encoding,"dimensions":c.dimensions,"bytes":payload.bytes().len(),
                "sha256":format!("{:x}",Sha256::digest(payload.bytes()))}));
        }
        let source_identity = json!({"schema_fingerprint":schema.fingerprint().map_err(|e| failure("schema",e))?,
            "selected_payload_bytes":selected_bytes,"components":components});
        source.reads.store(0, Ordering::Relaxed);
        let compiled = Arc::new(
            CompiledTensorComponent::compile(
                family,
                &registry,
                &catalog,
                &policy,
                &materializers,
                &config
                    .compile_options()
                    .map_err(|e| failure("compile options", e))?,
            )
            .map_err(|e| Failure::retain("compile", e, (runtime.clone(), registry.clone())))?,
        );
        let plan = compiled.executable().execution_plan();
        let plan_receipt = json!({"hash":plan.plan_hash(),"static_bytes":plan.payload().memory().static_bytes(),
            "minimum_step_bytes":plan.payload().memory().minimum_step_bytes(),"providers":plan.payload().nodes().iter()
                .map(|n|json!({"provider":n.provider_resources().provider_id(),"fingerprint":n.provider_implementation_fingerprint()})).collect::<Vec<_>>()});
        let input = config
            .managed_schedule_features(&host, &schedule)
            .map_err(|e| failure("features", e))?;
        let input_sha = format!("{:x}", Sha256::digest(input.bytes().bytes()));
        let phase = Arc::new(Phase {
            _runtime: runtime.clone(),
            _registry: registry.clone(),
            _compiled: compiled.clone(),
            _package: package.clone(),
            input,
        });
        let started = Instant::now();
        let mut session = compiled
            .initialize_retained(
                runtime.clone(),
                host.clone(),
                ComponentWeightSource::Stored(&source),
                ComponentInitializationConfig {
                    run_id: id(format!("run.time.init.{block}")),
                    request_id: id(format!("request.time.init.{block}")),
                    transaction_id: id(format!("transaction.time.init.{block}")),
                    static_policy: StaticInitializationPolicy::new(args.staging_bytes, 64)
                        .map_err(|e| failure("staging", e))?,
                    executor: TensorExecutorConfig::default(),
                },
                phase.clone(),
            )
            .map_err(|e| Failure::retain("initialize", e, phase.clone()))?;
        let initialization_seconds = started.elapsed().as_secs_f64();
        let initialization = match serde_json::to_value(session.session().initialization_receipt())
        {
            Ok(v) => v,
            Err(e) => {
                return Err(Failure::retain(
                    "initialization receipt",
                    e,
                    (session, phase),
                ))
            }
        };
        let source_reads = source.reads.load(Ordering::Relaxed);
        let complete_upload = match session.session().initialization_receipt() {
            ferrum_execution::component::ComponentInitializationReceipt::Initialized {
                receipt,
            } => {
                receipt
                    .uploaded_bytes()
                    .checked_add(receipt.imported_bytes())
                    == Some(selected_bytes)
            }
            _ => false,
        };
        let mut valid = source_reads == components.len() && complete_upload;
        let mut trials = Vec::new();
        let mut first = None;
        let mut retained = None;
        for iteration in 0..args.iterations {
            let started = Instant::now();
            let outputs = match session.session_mut().execute(
                id(format!("run.time.{block}.{iteration}")),
                id(format!("request.time.{block}.{iteration}")),
                vec![phase.input.clone()],
            ) {
                Ok(v) => v,
                Err(e) => return Err(Failure::retain("execute", e, (session, phase))),
            };
            let seconds = started.elapsed().as_secs_f64();
            let evidence = match inspect(&outputs, &config) {
                Ok(v) => v,
                Err(e) => return Err(Failure::retain("output", e, (session, phase, outputs))),
            };
            let same = first.as_ref().is_none_or(|v| v == &evidence);
            first.get_or_insert(evidence.clone());
            valid &= same && source.reads.load(Ordering::Relaxed) == source_reads;
            trials.push(
                json!({"iteration":iteration,"component_wall_seconds":seconds,"outputs":evidence,
                "same_as_first":(iteration>0).then_some(same),"source_reads":source.reads.load(Ordering::Relaxed)}),
            );
            retained = Some(outputs);
        }
        let (returned, close) = session
            .close()
            .map_err(|e| Failure::retain("close", e, phase.clone()))?;
        let outputs = retained.expect("validated positive iterations");
        valid &= outputs.iter().all(|v| host.owns(v.bytes()));
        drop(returned);
        drop(phase);
        drop(compiled);
        drop(source);
        if !valid {
            return Err(failure(
                "qualification",
                "nondeterminism, static source reread, or ownership mismatch",
            ));
        }
        let output_path = args
            .output_dir
            .join(format!("block-{block:03}.safetensors"));
        let table =
            FastH3ConditioningTable::from_execution(config.clone(), &schedule, outputs, limits)
                .map_err(|e| failure("bind time table", e))?;
        save(
            &output_path,
            table.outputs(),
            &config,
            &schedule,
            &source_identity,
            &plan_receipt,
        )
        .map_err(|e| failure("write output", e))?;
        let retained_bytes = host
            .usage()
            .map_err(|e| failure("Host usage", e))?
            .committed_bytes();
        drop(table);
        let remaining = host
            .usage()
            .map_err(|e| failure("Host retirement", e))?
            .committed_bytes();
        if remaining != 0 {
            return Err(failure(
                "Host retirement",
                format!("{remaining} bytes remain"),
            ));
        }
        blocks.push(json!({"block":block,"configuration":config,"source":source_identity,"plan":plan_receipt,
            "input_sha256":input_sha,"initialization":initialization,"initialization_wall_seconds":initialization_seconds,
            "trials":trials,"repeat_comparison_count":args.iterations-1,"artifact":output_path,"retirement":{"receipt":close,
                "host_bytes_with_outputs":retained_bytes,"host_bytes_after_output_drop":remaining}}));
    }
    let report = json!({"schema":"ferrum.fast_h3.time_precompute.v1","arguments":args,
        "timesteps":schedule.timesteps(),"routing":routing(&schedule),"blocks":blocks,
        "runtime_fingerprint":d.runtime_implementation_fingerprint,"full_source_payload_bytes":package.source_bytes(),
        "numerical_scope":"CPU libm-0.2.15 sinusoid, CUDA pedantic F32 time MLP, libdevice SiLU, BF16 AdaLN; finite outputs and separately reported repetition comparisons, not Torch bitwise or full video quality qualification",
        "timing_scope":"cold component initialization and execute wall times including upload/readback; no serving throughput or latency SLO claim",
        "memory_scope":"one shared Host account and managed device runtime; declared capacities are not measured peak RSS/VRAM"});
    let temporary = args.output_dir.join("manifest.json.pending");
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)
        .map_err(|e| failure("manifest", e))?;
    serde_json::to_writer_pretty(&mut file, &report).map_err(|e| failure("manifest", e))?;
    file.sync_all().map_err(|e| failure("manifest", e))?;
    std::fs::rename(temporary, args.output_dir.join("manifest.json"))
        .map_err(|e| failure("publish manifest", e))?;
    Ok(report)
}
