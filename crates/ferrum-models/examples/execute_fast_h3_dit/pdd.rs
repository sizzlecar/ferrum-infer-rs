//! Execution wiring for the shared model stages. Solver math and stage
//! provenance remain in FastH3DenoisingRequest/ManagedContinuation.
use super::*;
use ferrum_execution::TensorExecutionOutput;

fn definition<F: ModelFamilyProvider>(
    family: F,
    c: &F::Config,
    profile: &str,
) -> Result<PreparedModelFamily, Failure> {
    TypedFamilyRegistration::new(family)
        .prepare_with_profile(
            &serde_json::to_value(c).map_err(|e| error("stage config", e))?,
            &NumericalProfileId::new(profile).map_err(|e| error("stage profile", e))?,
        )
        .map_err(|e| error("stage definition", e))
}
fn cold(
    a: &Config,
    context: &Arc<runtime::Context>,
    package: &Arc<FastH3TransformerPackage>,
    host: &HostMemoryAccount,
    name: &str,
    family: PreparedModelFamily,
    options: ProgramPlanCompileOptions,
    source: &dyn WeightComponentSource,
    inputs: Vec<ManagedHostTensor>,
) -> Result<Vec<TensorExecutionOutput>, Failure> {
    let compiled = Arc::new(
        CompiledTensorComponent::compile(
            family,
            &context.registry,
            &context.catalog,
            &context.policy,
            &context.materializers,
            &options,
        )
        .map_err(|e| Failure::retain("stage compile", e, context.clone()))?,
    );
    let phase = Arc::new((
        context.clone(),
        package.clone(),
        compiled.clone(),
        inputs.clone(),
    ));
    let mut session = compiled
        .initialize_retained(
            context.runtime.clone(),
            host.clone(),
            ComponentWeightSource::Stored(source),
            initialization(a, name).map_err(|e| error("stage policy", e))?,
            phase.clone(),
        )
        .map_err(|e| Failure::retain("stage initialize", e, phase.clone()))?;
    let outputs = match session.session_mut().execute(
        id(format!("run.dit.{name}")),
        id(format!("request.dit.{name}")),
        inputs,
    ) {
        Ok(outputs) => outputs,
        Err(e) => return Err(Failure::retain("stage execute", e, (session, phase))),
    };
    let (returned, _) = session
        .close()
        .map_err(|e| Failure::retain("stage close", e, phase.clone()))?;
    drop(returned);
    drop(phase);
    Ok(outputs)
}
struct Counted<'a> {
    source: &'a dyn WeightComponentSource,
    reads: Arc<AtomicUsize>,
}
impl WeightComponentSource for Counted<'_> {
    fn component<'a>(
        &'a self,
        c: &WeightComponentSpec,
    ) -> Result<WeightComponentPayload<'a>, VNextError> {
        self.reads.fetch_add(1, Ordering::Relaxed);
        self.source.component(c)
    }
}
pub(super) fn run(
    a: Config,
    package: Arc<FastH3TransformerPackage>,
    plan: Arc<FastH3PddPlan>,
    schedule: FastH3ConditioningSchedule,
    layout: FastH3VsaLayout,
    positions: FastH3PositionPlan,
    host: HostMemoryAccount,
) -> Result<Value, Failure> {
    let text_features = match &a.workload {
        Workload::Pdd { text_features, .. } | Workload::PddSeededKeyframe { text_features, .. } => {
            text_features
        }
        _ => return Err(error("workload", "requires explicit PDD inputs")),
    };
    let input_limits = FastH3InputLimits {
        maximum_rows: a.maximum_rows as u64,
        maximum_segments: 32,
        maximum_metadata_bytes: a.maximum_metadata_bytes,
        maximum_stored_bytes: a.device_bytes,
    };
    let limits = FastH3DenoisingLimits {
        input: input_limits,
        backbone: limits::model_limits(&a, &package),
        maximum_plan_nodes: a.maximum_plan_nodes,
        maximum_stored_bytes: a.device_bytes,
        maximum_metadata_bytes: a.maximum_metadata_bytes,
    };
    let (config, source, regions) = FastH3DenoisingSource::prepare(
        &package,
        &layout,
        &schedule,
        &a.text_tags,
        FastH3VsaSelection::Trained,
        FastH3PrefixAttentionArithmetic::VendorFused,
        &host,
        limits,
    )
    .map_err(|e| error("denoising source", e))?;
    let context = runtime::create(&a, regions)?;
    eprintln!(
        "{}",
        json!({"stage":"denoising_source_ready","layers":config.backbone().layers()})
    );

    let text_config =
        FastH3InputProjectionConfig::text_from_package(&package, a.text_rows as u64, input_limits)
            .map_err(|e| error("text projection config", e))?;
    let text_source = FastH3InputProjectionSource::new(&package, &text_config, &host, input_limits)
        .map_err(|e| error("text projection source", e))?;
    let text = inputs::tensor_file(
        text_features,
        ResolvedTensorSpec::new(
            vec![a.text_rows as u64, package.config().text_dim() as u64],
            ElementType::Bf16,
            ResolvedTensorLayout::Contiguous,
        )
        .map_err(|e| error("text spec", e))?,
        FastH3InputProjectionFamily::TEXT_INPUT_ID,
        &host,
    )
    .map_err(|e| error("text features", e))?;
    let projected_outputs = cold(
        &a,
        &context,
        &package,
        &host,
        "context_projection",
        definition(
            FastH3InputProjectionFamily::new(input_limits).map_err(|e| error("text family", e))?,
            &text_config,
            FastH3InputProjectionFamily::TEXT_PROFILE_ID,
        )?,
        text_config
            .compile_options()
            .map_err(|e| error("text options", e))?,
        &text_source,
        vec![text],
    )?;
    let refiner_limits = FastH3TextRefinerLimits {
        maximum_rows: a.maximum_rows as u64,
        maximum_requests: 1,
        maximum_layers: a.maximum_layers,
        maximum_metadata_bytes: a.maximum_metadata_bytes,
        maximum_stored_bytes: a.device_bytes,
    };
    let refiner =
        FastH3TextRefinerConfig::from_package(&package, vec![a.text_rows as u64], refiner_limits)
            .map_err(|e| error("refiner config", e))?;
    let refiner_source = FastH3TextRefinerSource::new(&package, &refiner, refiner_limits)
        .map_err(|e| error("refiner source", e))?;
    let projected = projected_outputs
        .first()
        .ok_or_else(|| error("text output", "missing projected text"))?;
    let refined = cold(
        &a,
        &context,
        &package,
        &host,
        "text_refiner",
        definition(
            FastH3TextRefinerFamily::new(refiner_limits).map_err(|e| error("refiner family", e))?,
            &refiner,
            FastH3TextRefinerFamily::PROFILE_ID,
        )?,
        refiner
            .compile_options()
            .map_err(|e| error("refiner options", e))?,
        &refiner_source,
        vec![
            FastH3DenoisingRequest::projected_text_input(&refiner, projected, &host)
                .map_err(|e| error("refiner input", e))?,
        ],
    )?;
    drop(projected_outputs);
    eprintln!("{}", json!({"stage":"text_conditioning_complete"}));
    let tables = time::precompute(&a, &context, &package, &schedule, &host)?;
    let output_limits = FastH3OutputConditioningLimits {
        maximum_rows: a.maximum_rows as u64,
        maximum_time_rows: a.maximum_steps * 4,
        maximum_metadata_bytes: a.maximum_metadata_bytes,
        maximum_stored_bytes: a.device_bytes,
    };
    let output_time = FastH3OutputTimeConfig::from_package(&package, &schedule, output_limits)
        .map_err(|e| error("output time config", e))?;
    let output_time_source =
        FastH3OutputConditioningSource::for_time_projection(&package, &output_time, output_limits)
            .map_err(|e| error("output time source", e))?;
    let output_table = cold(
        &a,
        &context,
        &package,
        &host,
        "output_time",
        definition(
            FastH3OutputTimeFamily::new(output_limits)
                .map_err(|e| error("output time family", e))?,
            &output_time,
            FastH3OutputTimeFamily::PROFILE_ID,
        )?,
        output_time
            .compile_options()
            .map_err(|e| error("output time options", e))?,
        &output_time_source,
        vec![output_time
            .embedding_input(&tables[0], &host)
            .map_err(|e| error("output embedding", e))?],
    )?;
    let output_table = Arc::new(
        FastH3OutputConditioningTable::from_execution(output_time, output_table, output_limits)
            .map_err(|e| error("output table", e))?,
    );
    let ([video, audio], diagnostic_latents, input_profile) =
        inputs::initial_media(&a, &plan, layout.packed(), &host)
            .map_err(|e| error("initial media", e))?;
    let (cos, sin) = inputs::rotary(&a, config.backbone(), &package, &positions, &host)
        .map_err(|e| error("rotary", e))?;
    let mut request = FastH3DenoisingRequest::initialize(
        config.clone(),
        limits,
        id("request.fast_h3.pdd"),
        plan.clone(),
        layout.packed(),
        host.clone(),
        &video,
        &audio,
        refined
            .first()
            .ok_or_else(|| error("refiner output", "missing refined text"))?,
        [cos, sin],
        tables.into(),
        output_table,
    )
    .map_err(|e| error("sampling request", e))?;
    drop(refined);
    drop(video);
    drop(audio);
    let prepared = definition(
        FastH3DenoisingFamily::new(limits).map_err(|e| error("denoising family", e))?,
        &config,
        config.profile_id(),
    )?;
    let ModelWeights::Stored(schema) = prepared.weights() else {
        return Err(error("source", "Stored required"));
    };
    let components = schema.components.len();
    let compiled = Arc::new(
        CompiledTensorComponent::compile(
            prepared,
            &context.registry,
            &context.catalog,
            &context.policy,
            &context.materializers,
            &config
                .compile_options(limits)
                .map_err(|e| error("denoising options", e))?,
        )
        .map_err(|e| Failure::retain("denoising compile", e, context.clone()))?,
    );
    eprintln!(
        "{}",
        json!({"stage":"denoising_compiled","nodes":compiled.executable().execution_plan().payload().nodes().len()})
    );
    let reads = Arc::new(AtomicUsize::new(0));
    let counted = Counted {
        source: &source,
        reads: reads.clone(),
    };
    let phase = Arc::new((
        context.clone(),
        compiled.clone(),
        package.clone(),
        reads.clone(),
    ));
    let mut session = compiled
        .initialize_retained(
            context.runtime.clone(),
            host.clone(),
            ComponentWeightSource::Stored(&counted),
            initialization(&a, "denoising").map_err(|e| error("denoising policy", e))?,
            phase.clone(),
        )
        .map_err(|e| Failure::retain("denoising initialize", e, phase.clone()))?;
    drop(counted);
    drop(source);
    eprintln!(
        "{}",
        json!({"stage":"denoising_initialized","source_reads":reads.load(Ordering::Relaxed)})
    );
    if reads.load(Ordering::Relaxed) != components {
        return Err(Failure::retain(
            "source reads",
            "initial source inventory mismatch",
            (session, phase, request),
        ));
    }
    let final_limits = FastH3FinalOutputLimits {
        conditioning: output_limits,
        projection: FastH3PddProjectionLimits {
            maximum_rows: a.maximum_rows as u64,
            maximum_metadata_bytes: a.maximum_metadata_bytes,
            maximum_source_bytes: a.maximum_source_bytes as u64,
            maximum_fused_bytes: a.host_bytes as u64,
        },
        maximum_metadata_bytes: a.maximum_metadata_bytes,
        maximum_plan_nodes: a.maximum_plan_nodes,
        maximum_stored_bytes: a.device_bytes,
        maximum_pdd_steps: a.maximum_steps,
    };
    let mut evaluations = Vec::new();
    while !request.is_complete() {
        let index = request.evaluation();
        let input = match request.denoising_inputs() {
            Ok(v) => v,
            Err(e) => return Err(Failure::retain("PDD inputs", e, (session, phase, request))),
        };
        let started = Instant::now();
        let hidden = match session.session_mut().execute(
            id(format!("run.pdd.dit{index}")),
            request.request().clone(),
            input,
        ) {
            Ok(v) => v,
            Err(e) => {
                return Err(Failure::retain(
                    "PDD denoising",
                    e,
                    (session, phase, request),
                ))
            }
        };
        let result = (|| -> Result<Vec<TensorExecutionOutput>, Failure> {
            let (c, source) = FastH3FinalOutputSource::prepare(
                &package,
                &plan,
                layout.packed(),
                index,
                &host,
                final_limits,
            )
            .map_err(|e| error("PDD head source", e))?;
            let inputs = request
                .final_output_inputs(
                    &c,
                    hidden
                        .first()
                        .ok_or_else(|| error("backbone output", "missing hidden"))?,
                )
                .map_err(|e| error("PDD head input", e))?;
            cold(
                &a,
                &context,
                &package,
                &host,
                &format!("pdd_head{index}"),
                definition(
                    FastH3FinalOutputFamily::new(final_limits)
                        .map_err(|e| error("PDD head family", e))?,
                    &c,
                    FastH3FinalOutputFamily::PROFILE_ID,
                )?,
                c.compile_options(final_limits)
                    .map_err(|e| error("PDD head options", e))?,
                &source,
                inputs,
            )
        })();
        let velocities = match result {
            Ok(v) => v,
            Err(mut e) => {
                e.owner = Box::new((e.owner, session, phase, request));
                return Err(e);
            }
        };
        if let Err(e) = request.advance(&velocities) {
            return Err(Failure::retain(
                "PDD commit",
                e,
                (session, phase, request, velocities),
            ));
        }
        if reads.load(Ordering::Relaxed) != components {
            return Err(Failure::retain(
                "PDD source reuse",
                "denoising source reread",
                (session, phase, request),
            ));
        }
        let entry = json!({"evaluation":index,"component_and_head_seconds":started.elapsed().as_secs_f64(),"committed_evaluations":request.evaluation(),
            "video_sha256":format!("{:x}",Sha256::digest(request.video().bytes().bytes())),"audio_sha256":format!("{:x}",Sha256::digest(request.audio().bytes().bytes())),"source_reads":reads.load(Ordering::Relaxed)});
        eprintln!("{}", json!({"stage":"pdd_committed","result":entry}));
        evaluations.push(entry);
    }
    let (returned, close) = session
        .close()
        .map_err(|e| Failure::retain("denoising close", e, phase.clone()))?;
    drop(returned);
    drop(phase);
    let mut outputs = Vec::new();
    for (name, sample, rows) in [
        (
            "video",
            request.video(),
            layout.packed().target_video_rows(),
        ),
        (
            "audio",
            request.audio(),
            layout.packed().target_audio_rows(),
        ),
    ] {
        let width = sample.tensor().dimensions()[1] as usize;
        let bytes = &sample.bytes().bytes()[rows.start * width * 4..rows.end * width * 4];
        let path = a.output_dir.join(format!("target-{name}.f32"));
        let mut file = std::fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&path)
            .map_err(|e| error("final latent artifact", e))?;
        file.write_all(bytes)
            .map_err(|e| error("final latent artifact", e))?;
        file.sync_all()
            .map_err(|e| error("final latent artifact", e))?;
        outputs.push(json!({"modality":name,"dtype":"F32","layout":"original packed channel-major output rows","shape":[rows.len(),width],"path":path,"sha256":format!("{:x}",Sha256::digest(bytes))}));
    }
    drop(request);
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
    let report = json!({"schema":"ferrum.fast_h3.pdd.execution.v1","configuration":a,"scope":"explicit text features and declared media inputs through actual media/text/backbone/PDD; final packed target latents; diagnostic zero inputs do not establish generation quality; no performance comparison claim",
        "diagnostic_latents":diagnostic_latents,
        "initial_media_profile":input_profile,
        "denoising_static_initializations":1,"denoising_source_components":components,"evaluations":evaluations,"outputs":outputs,"close":close,"host_bytes_after_drop":host_after,
        "solver_profile":FastH3ManagedContinuation::NUMERICAL_PROFILE_ID});
    let mut file = std::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(a.output_dir.join("manifest.json"))
        .map_err(|e| error("manifest", e))?;
    file.write_all(&serde_json::to_vec_pretty(&report).map_err(|e| error("manifest", e))?)
        .map_err(|e| error("manifest", e))?;
    Ok(report)
}
