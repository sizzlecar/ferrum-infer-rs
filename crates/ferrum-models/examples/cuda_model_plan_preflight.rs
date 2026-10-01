//! Read-only product compilation. Never initializes an executor or uploads weights.
//! The CUDA composition keeps the product's native artifact validation intact.
#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("cuda_model_plan_preflight requires --features cuda");
    std::process::exit(1);
}

#[cfg(feature = "cuda")]
fn main() {
    std::process::exit(cuda::run());
}

#[cfg(feature = "cuda")]
mod cuda {
    use std::{collections::BTreeMap, error::Error, path::PathBuf, sync::Arc};

    use ferrum_interfaces::vnext::{
        DeviceId, ExecutablePlanView, ModelSourceKind, OriginalModelSource, OriginalModelSources,
        SequenceCheckpointCapability,
    };
    use ferrum_kernels::backend::cuda::vnext_ops::{
        cuda_weight_materializer_selection, CudaVNextComposition,
    };
    use ferrum_models::{
        executor::{VNextExecutorConfig, VNextRuntimeComposition},
        vnext::{
            open_registered_product_sources, resolve_registered_model_from_sources,
            ProductionWeightArtifact,
        },
    };
    use ferrum_types::{
        AttentionExecutionPolicy, Device, EngineConfig, KvCacheDtype, KvStorageFormat, ModelId,
        NumericalExecutionPolicy, NumericalProfileId,
    };
    use serde_json::{json, Value};

    type Result<T> = std::result::Result<T, Box<dyn Error>>;

    const USAGE: &str = "cuda_model_plan_preflight --model GGUF --semantic-source DIR \
        --numerical-profile ID --cuda-ordinal N --attention-policy POLICY \
        --max-model-len N --max-num-seqs N --max-num-batched-tokens N \
        --runtime-memory-budget-bytes N --reusable-execution true|false \
        --boundaries N[,N...] [--engine-config ENGINE_CONFIG_JSON]";

    struct Args {
        model: PathBuf,
        semantic: PathBuf,
        profile: NumericalProfileId,
        ordinal: usize,
        attention: AttentionExecutionPolicy,
        context: usize,
        sequences: usize,
        batch: usize,
        capacity: u64,
        reusable: bool,
        boundaries: Vec<u64>,
        engine_config: Option<PathBuf>,
    }

    impl Args {
        fn parse() -> Result<Self> {
            let raw = std::env::args().skip(1).collect::<Vec<_>>();
            if raw.is_empty() || raw.iter().any(|s| s == "--help") {
                return Err(USAGE.into());
            }
            if raw.len() % 2 != 0 {
                return Err(format!("every option requires one value: {USAGE}").into());
            }
            let mut values = BTreeMap::new();
            for pair in raw.chunks_exact(2) {
                if values.insert(pair[0].clone(), pair[1].clone()).is_some() {
                    return Err(format!("duplicate option {}", pair[0]).into());
                }
            }
            let mut take = |name: &str| -> Result<String> {
                values
                    .remove(name)
                    .ok_or_else(|| format!("missing {name}: {USAGE}").into())
            };
            let model = PathBuf::from(take("--model")?);
            let semantic = PathBuf::from(take("--semantic-source")?);
            let profile = NumericalProfileId::new(take("--numerical-profile")?)?;
            let ordinal = take("--cuda-ordinal")?.parse()?;
            let attention =
                AttentionExecutionPolicy::parse_runtime_value(&take("--attention-policy")?)?;
            let context = take("--max-model-len")?.parse()?;
            let sequences = take("--max-num-seqs")?.parse()?;
            let batch = take("--max-num-batched-tokens")?.parse()?;
            let capacity = take("--runtime-memory-budget-bytes")?.parse()?;
            let reusable = take("--reusable-execution")?.parse()?;
            let boundaries = take("--boundaries")?
                .split(',')
                .map(str::parse::<u64>)
                .collect::<std::result::Result<Vec<_>, _>>()?;
            let engine_config = values.remove("--engine-config").map(PathBuf::from);
            if !values.is_empty() {
                return Err(format!("unknown options: {:?}", values.keys()).into());
            }
            if context == 0
                || sequences == 0
                || batch == 0
                || capacity == 0
                || boundaries.is_empty()
                || boundaries.iter().any(|n| *n == 0 || *n > context as u64)
            {
                return Err(
                    "positive capacity/shape and boundaries within declared context required"
                        .into(),
                );
            }
            Ok(Self {
                model,
                semantic,
                profile,
                ordinal,
                attention,
                context,
                sequences,
                batch,
                capacity,
                reusable,
                boundaries,
                engine_config,
            })
        }
    }

    fn source(kind: ModelSourceKind, path: &std::path::Path) -> OriginalModelSource {
        OriginalModelSource {
            kind,
            location: path.display().to_string(),
            requested_revision: None,
        }
    }

    fn inspect(report: &mut Value) -> Result<bool> {
        let args = Args::parse()?;
        let usable_capacity_bytes = usize::try_from(args.capacity).map_err(|_| {
            format!(
                "--runtime-memory-budget-bytes {} exceeds this platform's {}-bit usize range",
                args.capacity,
                usize::BITS,
            )
        })?;
        report["inputs"] = json!({
            "model": args.model, "semantic_source": args.semantic,
            "numerical_profile": args.profile, "requested_boundaries": args.boundaries,
            "engine_config": args.engine_config,
        });
        report["stage"] = json!("native_registry_validation");
        // This is the original product constructor. No unvalidated composition
        // or alternative executable registry is exposed by this diagnostic.
        let (runtime, registry, materializers, catalog) = CudaVNextComposition::create(
            args.ordinal,
            DeviceId::new(format!("cuda:{}", args.ordinal))?,
            args.attention,
        )?
        .into_parts();
        report["native_validation"] = json!("passed");
        report["capability_catalog_fingerprint"] = json!(catalog.fingerprint()?);
        report["device"] = serde_json::to_value(catalog.device())?;

        report["stage"] = json!("registered_source_metadata");
        // The original source bundle streams the artifact SHA for provenance.
        // Family preparation reads tensor headers, not device tensor payloads.
        let semantic = source(ModelSourceKind::LocalDirectory, &args.semantic);
        let sources = Arc::new(open_registered_product_sources(
            &args.semantic,
            &args.semantic,
            ProductionWeightArtifact::gguf_file(&args.model),
            OriginalModelSources {
                semantic: semantic.clone(),
                tokenizer: semantic,
                weights: source(ModelSourceKind::LocalFile, &args.model),
            },
        )?);
        report["resolved_sources"] = serde_json::to_value(sources.resolved_sources())?;
        let registration = resolve_registered_model_from_sources(&sources)?.into_required()?;
        let defined = registration.define_from_sources(sources)?;
        let mut engine: EngineConfig = match &args.engine_config {
            Some(path) => serde_json::from_reader(std::fs::File::open(path)?)?,
            None => EngineConfig::default(),
        };
        engine.backend.device = Device::CUDA(args.ordinal);
        engine.backend.dtype = defined.descriptor().execution_dtype();
        engine.backend.enable_reusable_execution = args.reusable;
        engine.runtime.attention_execution_policy = args.attention;
        engine.runtime.max_model_len = Some(args.context);
        engine.runtime.prefix_state_cache_enabled = true;
        engine.scheduler.max_running_requests = args.sequences;
        engine.batching.max_num_batched_tokens = args.batch;
        engine.memory.usable_capacity_bytes = Some(usable_capacity_bytes);
        engine.kv_cache.dtype = KvCacheDtype::Fp16;
        engine.numerical_execution = NumericalExecutionPolicy::Require(args.profile.clone());
        // Preserve the original declared profile/KV compatibility validator.
        let profiles = defined
            .definition()
            .numerical_profiles()
            .candidates(&engine.numerical_execution, KvStorageFormat::F16)?;
        if profiles.len() != 1 || profiles[0].id != args.profile {
            return Err("Require did not resolve exactly the requested numerical profile".into());
        }
        let prepared = defined.prepare(&args.profile)?;
        report["prepared_family_fingerprint"] = json!(prepared.family().fingerprint()?);
        report["program_fingerprint"] = json!(prepared.family().program().fingerprint()?);
        report["numerical_profile"] = serde_json::to_value(prepared.family().numerical_profile())?;
        let selection = cuda_weight_materializer_selection(prepared.family())?;
        report["materializer_selection"] = json!({
            "materializer_id": selection.materializer_id(),
            "has_numeric_quality_artifact": selection.has_numeric_quality_artifact(),
        });
        let info = prepared.model_info(
            ModelId::new("metadata-preflight"),
            Device::CUDA(args.ordinal),
        );
        let config = VNextExecutorConfig::from_engine_config(&engine, &info, runtime.as_ref())?;
        report["engine_configuration"] = serde_json::to_value(&engine)?;
        report["resolved_runtime_policy"] = serde_json::to_value(&config.runtime_policy)?;
        report["effective_maximum_model_tokens"] = json!(config.maximum_model_tokens);
        report["stage"] = json!("product_plan_compile");
        let composition = VNextRuntimeComposition::new(runtime, registry, materializers, catalog);
        let compiled = composition.compile_model(&prepared, info, &engine, config, selection)?;
        let plan = compiled.compilation().executable().execution_plan();
        report["plan_hash"] = serde_json::to_value(plan.plan_hash())?;
        report["memory_plan"] = serde_json::to_value(plan.payload().memory())?;
        let catalog = composition.catalog();
        let mut nodes = Vec::new();
        let mut missing = BTreeMap::<String, Vec<String>>::new();
        for node in plan.payload().nodes() {
            let provider = catalog
                .providers_for(node.operation_id())?
                .iter()
                .find(|p| p.provider_id() == node.selection().selected_provider())
                .ok_or("compiled selected provider absent from exact catalog")?;
            if provider.checkpoint_capability().is_unsupported() {
                missing
                    .entry(provider.provider_id().to_string())
                    .or_default()
                    .push(node.id().to_string());
            }
            nodes.push(json!({
                "node_id": node.id(), "operation_id": node.operation_id(),
                "operation_fingerprint": node.operation_fingerprint(),
                "provider_id": node.selection().selected_provider(),
                "implementation_fingerprint": node.provider_implementation_fingerprint(),
                "checkpoint_capability": provider.checkpoint_capability(),
            }));
        }
        report["selected_nodes"] = json!(nodes);
        report["undeclared_selected_providers"] = json!(missing);
        let mut supported = match plan.sequence_checkpoint_capability() {
            SequenceCheckpointCapability::Enabled(layout) => {
                report["checkpoint"] = json!({"kind":"enabled", "layout":layout,
                    "layout_fingerprint":layout.fingerprint()?});
                true
            }
            SequenceCheckpointCapability::Unsupported(reasons) => {
                report["checkpoint"] = json!({"kind":"unsupported", "reasons":reasons});
                false
            }
        };
        let mut byte_plans = Vec::new();
        for boundary in args.boundaries {
            match plan.checkpoint_byte_plan(boundary) {
                Ok(byte_plan) => byte_plans.push(json!({"boundary":boundary,"result":byte_plan})),
                Err(error) => {
                    supported = false;
                    byte_plans.push(json!({"boundary":boundary,"error":error.to_string()}));
                }
            }
        }
        report["checkpoint_byte_plans"] = json!(byte_plans);
        report["stage"] = json!("complete_before_initialize");
        // Drop compiled metadata here. Deliberately never call initialize(),
        // prepare_startup(), static allocation/materialization or dispatch.
        Ok(supported)
    }

    pub(super) fn run() -> i32 {
        let mut report = json!({
            "schema":"ferrum.cuda_model_plan_preflight.v1",
            "stage":"arguments", "native_validation":"not_reached",
            "scope":"Original product metadata/static plan only; no executor initialization, weight upload, dispatch, transfer, model qualification or SLO claim.",
            "source_io":"Original product source SHA reads the GGUF bytes; this is not a promise of header-only host IO.",
        });
        let code = match inspect(&mut report) {
            Ok(true) => 0,
            Ok(false) => 2,
            Err(error) => {
                report["error"] = json!(error.to_string());
                1
            }
        };
        report["exit_code"] = json!(code);
        println!(
            "{}",
            serde_json::to_string_pretty(&report).expect("JSON report")
        );
        code
    }
}
