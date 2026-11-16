use super::{
    config::{Config, HiddenInput, LatentInput, Rotary, TensorFile, Workload},
    limits, package,
    retained_failure::Failure,
};
use ferrum_execution::{
    component::{
        CompiledTensorComponent, ComponentInitializationConfig, ComponentWeightSource,
        TensorComponentPolicy,
    },
    TensorExecutorConfig,
};
use ferrum_kernels::backend::cuda::vnext_runtime::CudaDeviceRuntime;
use ferrum_models::fast_h3::{vsa::*, *};
use ferrum_runtime::*;
use half::bf16;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    io::Write,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    time::Instant,
};
#[path = "inputs.rs"]
mod inputs;
#[path = "pdd.rs"]
mod pdd;
#[path = "runtime.rs"]
mod runtime;
#[path = "time.rs"]
mod time;
fn id<T: TryFrom<String, Error = VNextError>>(s: impl Into<String>) -> T {
    T::try_from(s.into()).expect("fixed component identity")
}
fn error<E: std::fmt::Display + 'static>(stage: &'static str, e: E) -> Failure {
    Failure::retain(stage, e, ())
}
fn finite(bytes: &[u8]) -> bool {
    bytes.chunks_exact(2).all(|v| {
        bf16::from_bits(u16::from_le_bytes([v[0], v[1]]))
            .to_f32()
            .is_finite()
    })
}
fn initialization(a: &Config, suffix: &str) -> Result<ComponentInitializationConfig, VNextError> {
    Ok(ComponentInitializationConfig {
        run_id: id(format!("run.dit.init.{suffix}")),
        request_id: id(format!("request.dit.init.{suffix}")),
        transaction_id: id(format!("transaction.dit.init.{suffix}")),
        static_policy: StaticInitializationPolicy::new(a.staging_bytes as u64, 64)?,
        executor: TensorExecutorConfig::default(),
    })
}
struct Counted {
    inner: FastH3DitBackboneSource,
    reads: AtomicUsize,
}
impl WeightComponentSource for Counted {
    fn component<'a>(
        &'a self,
        c: &WeightComponentSpec,
    ) -> Result<WeightComponentPayload<'a>, VNextError> {
        self.reads.fetch_add(1, Ordering::Relaxed);
        self.inner.component(c)
    }
}
struct Phase {
    _context: Arc<runtime::Context>,
    _compiled: Arc<CompiledTensorComponent<CudaDeviceRuntime>>,
    source: Arc<Counted>,
    input: [ManagedHostTensor; 3],
    tables: Vec<FastH3ConditioningTable>,
}
pub(super) fn run(a: Config) -> Result<Value, Failure> {
    let checkpoint = package::load_checkpoint(
        &a.model_dir,
        FastH3MetadataLimits {
            max_file_bytes: a.maximum_metadata_bytes,
            max_pdd_steps: a.maximum_steps,
        },
    )
    .map_err(|e| error("checkpoint", e))?;
    let sampling = FastH3PddPlan::new(&checkpoint).map_err(|e| error("sampling schedule", e))?;
    let schedule = FastH3ConditioningSchedule::all_modalities(&sampling)
        .map_err(|e| error("conditioning schedule", e))?;
    let package = Arc::new(
        package::load(
            &a.model_dir,
            FastH3Nvfp4Profile::DiTWithVsa,
            a.maximum_metadata_bytes,
            FastH3WeightLimits {
                maximum_tensors: a.maximum_tensors,
                maximum_source_bytes: a.maximum_source_bytes,
                maximum_skip_metadata_bytes: a.maximum_metadata_bytes,
            },
        )
        .map_err(|e| error("package", e))?,
    );
    let target = FastH3VideoGeometry::new(a.width, a.height, a.frames)
        .map_err(|e| error("video geometry", e))?;
    let references = [FastH3Reference::Image {
        latent_height: a.reference_latent_height,
        latent_width: a.reference_latent_width,
    }];
    let layout = FastH3VsaLayout::ref2va(
        &checkpoint,
        a.text_rows,
        &references,
        &target,
        FastH3VsaLayoutLimits {
            maximum_packed_rows: a.maximum_rows,
            maximum_tiles: a.maximum_tiles,
            maximum_metadata_bytes: a.maximum_metadata_bytes,
        },
    )
    .map_err(|e| error("VSA layout", e))?;
    let positions = FastH3PositionPlan::ref2va(
        a.text_rows,
        &references,
        &target,
        FastH3PositionLimits {
            max_sequence_rows: a.maximum_rows,
            max_regions: 4,
        },
    )
    .map_err(|e| error("positions", e))?;
    std::fs::create_dir(&a.output_dir).map_err(|e| error("new output directory", e))?;
    let host = HostMemoryAccount::with_allocation_alignment(
        HostMemoryLimits::new(
            a.host_bytes,
            HostMemoryClassLimits {
                input_preprocess_bytes: a.host_bytes,
                output_bytes: a.host_bytes,
                transfer_staging_bytes: a.host_bytes,
                weight_backing_bytes: a.host_bytes,
            },
        )
        .map_err(|e| error("Host limits", e))?,
        CudaDeviceRuntime::pinned_weight_upload_alignment()
            .map_err(|e| error("Host alignment", e))?,
    )
    .map_err(|e| error("Host account", e))?;
    if !matches!(a.workload, Workload::Backbone { .. }) {
        return pdd::run(
            a,
            package,
            Arc::new(sampling),
            schedule,
            layout,
            positions,
            host,
        );
    }
    let limits = limits::model_limits(&a, &package);
    let started = Instant::now();
    let (config, source, regions) = FastH3DitBackboneSource::prepare(
        package.clone(),
        &layout,
        &schedule,
        &[&a.text_tags],
        FastH3VsaSelection::Trained,
        FastH3PrefixAttentionArithmetic::VendorFused,
        &host,
        limits,
    )
    .map_err(|e| error("backbone source", e))?;
    let source_prepare_seconds = started.elapsed().as_secs_f64();
    let context = runtime::create(&a, regions)?;
    let started = Instant::now();
    let tables = time::precompute(&a, &context, &package, &schedule, &host)?;
    let time_precompute_seconds = started.elapsed().as_secs_f64();
    let input = inputs::prepare(&a, &config, &package, &positions, &host)
        .map_err(|e| error("backbone inputs", e))?;
    let input_sha256=input.iter().map(|t|json!({"id":t.value_id(),"shape":t.tensor().dimensions(),"sha256":format!("{:x}",Sha256::digest(t.bytes().bytes()))})).collect::<Vec<_>>();
    let started = Instant::now();
    let family = TypedFamilyRegistration::new(
        FastH3DitBackboneFamily::new(limits).map_err(|e| error("family", e))?,
    )
    .prepare_with_profile(
        &serde_json::to_value(&config).map_err(|e| error("config", e))?,
        &NumericalProfileId::new(config.profile_id()).map_err(|e| error("profile", e))?,
    )
    .map_err(|e| error("definition", e))?;
    let ModelWeights::Stored(schema) = family.weights() else {
        return Err(error("weights", "requires original Stored schema"));
    };
    let components = schema.components.len();
    let selected_bytes = schema
        .components
        .iter()
        .try_fold(0u64, |n, c| {
            n.checked_add(c.physical_bytes()?)
                .ok_or_else(|| VNextError::InvalidExecutionPlan {
                    reason: "physical byte sum overflow".into(),
                })
        })
        .map_err(|e| error("source bytes", e))?;
    let schema_fingerprint = schema
        .fingerprint()
        .map_err(|e| error("schema fingerprint", e))?;
    let compiled = Arc::new(
        CompiledTensorComponent::compile(
            family,
            &context.registry,
            &context.catalog,
            &context.policy,
            &context.materializers,
            &config
                .compile_options(limits)
                .map_err(|e| error("compile options", e))?,
        )
        .map_err(|e| Failure::retain("compile", e, context.clone()))?,
    );
    let compile_seconds = started.elapsed().as_secs_f64();
    let plan = compiled.executable().execution_plan();
    let plan_report = json!({"hash":plan.plan_hash(),"nodes":plan.payload().nodes().len(),"static_bytes":plan.payload().memory().static_bytes(),
        "minimum_step_bytes":plan.payload().memory().minimum_step_bytes()});
    let source = Arc::new(Counted {
        inner: source,
        reads: AtomicUsize::new(0),
    });
    let phase = Arc::new(Phase {
        _context: context.clone(),
        _compiled: compiled.clone(),
        source: source.clone(),
        input,
        tables,
    });
    let started = Instant::now();
    let mut session = compiled
        .initialize_retained(
            context.runtime.clone(),
            host.clone(),
            ComponentWeightSource::Stored(source.as_ref()),
            initialization(&a, "backbone").map_err(|e| error("static policy", e))?,
            phase.clone(),
        )
        .map_err(|e| Failure::retain("initialize", e, phase.clone()))?;
    let initialization_seconds = started.elapsed().as_secs_f64();
    let complete_upload = match session.session().initialization_receipt() {
        ferrum_execution::component::ComponentInitializationReceipt::Initialized { receipt } => {
            receipt
                .uploaded_bytes()
                .checked_add(receipt.imported_bytes())
                == Some(selected_bytes)
        }
        _ => false,
    };
    if !complete_upload || source.reads.load(Ordering::Relaxed) != components {
        return Err(Failure::retain(
            "initialization coverage",
            "source reads or uploaded bytes differ",
            (session, phase),
        ));
    }
    let mut evaluations = Vec::with_capacity(schedule.evaluation_count());
    for evaluation in 0..schedule.evaluation_count() {
        let mut inputs =
            match config.managed_role_tables(&phase.tables, &schedule, &[evaluation], &host) {
                Ok(inputs) => inputs,
                Err(e) => return Err(Failure::retain("role inputs", e, (session, phase))),
            };
        inputs.extend(phase.input.iter().cloned());
        let started = Instant::now();
        let outputs = match session.session_mut().execute(
            id(format!("run.dit.evaluation{evaluation}")),
            id(format!("request.dit.evaluation{evaluation}")),
            inputs,
        ) {
            Ok(outputs) => outputs,
            Err(e) => return Err(Failure::retain("execute", e, (session, phase))),
        };
        let seconds = started.elapsed().as_secs_f64();
        if outputs.len() != 1
            || outputs[0].tensor().dimensions() != [config.packed_rows(), config.hidden_size()]
            || outputs[0].tensor().element_type() != ElementType::Bf16
            || !finite(outputs[0].bytes().bytes())
            || source.reads.load(Ordering::Relaxed) != components
        {
            return Err(Failure::retain(
                "output",
                "output geometry/finite or source reuse check failed",
                (session, phase, outputs),
            ));
        }
        let sha = format!("{:x}", Sha256::digest(outputs[0].bytes().bytes()));
        let path = a
            .output_dir
            .join(format!("evaluation-{evaluation:02}.hidden.bf16"));
        let write = (|| -> std::io::Result<()> {
            let mut file = std::fs::OpenOptions::new()
                .create_new(true)
                .write(true)
                .open(&path)?;
            file.write_all(outputs[0].bytes().bytes())?;
            file.sync_all()
        })();
        if let Err(e) = write {
            return Err(Failure::retain(
                "output artifact",
                e,
                (session, phase, outputs),
            ));
        }
        let row = json!({"evaluation":evaluation,"component_wall_seconds":seconds,"finite":true,"shape":outputs[0].tensor().dimensions(),
            "sha256":sha,"path":path,"source_component_reads":source.reads.load(Ordering::Relaxed)});
        eprintln!("{}", json!({"stage":"backbone_forward","result":row}));
        evaluations.push(row);
    }
    let (returned, close) = session
        .close()
        .map_err(|e| Failure::retain("close", e, phase.clone()))?;
    let close = serde_json::to_value(close).map_err(|e| error("close receipt", e))?;
    drop(returned);
    drop(phase);
    drop(source);
    drop(compiled);
    drop(context);
    drop(package);
    let host_after = host
        .usage()
        .map_err(|e| error("Host final", e))?
        .allocated_bytes();
    if host_after != 0 {
        return Err(error(
            "Host retirement",
            format!("{host_after} bytes still owned"),
        ));
    }
    let report = json!({"schema":"ferrum.fast_h3.dit_backbone.execution.v1","configuration":a,"layers":config.layers(),"batches":config.batches(),
        "packed_rows":config.packed_rows(),"precision":"NVFP4 projections, BF16 boundaries, explicit operator profiles",
        "scope":"complete checkpoint backbone with fixed hidden input at each evaluation; no PDD update, media projection, video quality or performance comparison claim",
        "hidden_progression":"fixed_input_each_evaluation","rotary_profile":match a.workload.rotary() {Rotary::AnalyticPositions=>FastH3PositionPlan::ROTARY_NUMERICAL_PROFILE_ID,Rotary::DiagnosticIdentity=>"diagnostic.identity_bf16"},
        "weight_payload_hashes_computed":false,"schema_fingerprint":schema_fingerprint,"source_components":components,"selected_weight_bytes":selected_bytes,
        "source_prepare_seconds":source_prepare_seconds,"time_precompute_seconds":time_precompute_seconds,"compile_seconds":compile_seconds,
        "initialization_seconds":initialization_seconds,"static_initializations":1,"source_reads_after_all_evaluations":components,
        "inputs":input_sha256,"plan":plan_report,"evaluations":evaluations,"close":close,"host_bytes_after_drop":host_after});
    let bytes = serde_json::to_vec_pretty(&report).map_err(|e| error("manifest", e))?;
    let mut file = std::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(a.output_dir.join("manifest.json"))
        .map_err(|e| error("manifest", e))?;
    file.write_all(&bytes).map_err(|e| error("manifest", e))?;
    Ok(report)
}
