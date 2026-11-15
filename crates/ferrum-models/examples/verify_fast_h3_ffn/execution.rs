use super::{arguments::Arguments, package, Failure};
use anyhow::{ensure, Context, Result};
use ferrum_execution::{
    component::{
        CompiledTensorComponent, ComponentInitializationConfig, ComponentWeightSource,
        TensorComponentPolicy,
    },
    TensorExecutorConfig,
};
use ferrum_kernels::backend::cuda::{
    vnext_ops::{
        cuda_tensor_runtime_config, cuda_tensor_runtime_enable_cutlass_nvfp4,
        cuda_vnext_nvfp4_bf16_providers, cuda_vnext_tensor_providers, CudaTensorOperation,
        CutlassNvfp4Config,
    },
    vnext_runtime::{CudaDeviceRuntime, CudaTensorRuntimeConfig},
};
use ferrum_models::fast_h3::*;
use ferrum_runtime::*;
use half::bf16;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    time::Instant,
};

fn id<T: TryFrom<String, Error = VNextError>>(s: impl Into<String>) -> T {
    T::try_from(s.into()).expect("fixed qualification ID")
}
fn error<E: std::fmt::Display + 'static>(stage: &'static str, e: E) -> Failure {
    Failure::retain(stage, e, ())
}

struct CountedSource {
    inner: FastH3FfnSource,
    reads: AtomicUsize,
}
impl WeightComponentSource for CountedSource {
    fn component<'a>(
        &'a self,
        c: &WeightComponentSpec,
    ) -> std::result::Result<WeightComponentPayload<'a>, VNextError> {
        self.reads.fetch_add(1, Ordering::Relaxed);
        self.inner.component(c)
    }
}
/// These owners remain one retained caller phase until the original component
/// returns its checked close receipt. Failures retain the typed error as well.
struct Phase {
    runtime: Arc<CudaDeviceRuntime>,
    registry: Arc<OperationRuntimeRegistry<CudaDeviceRuntime>>,
    compiled: Arc<CompiledTensorComponent<CudaDeviceRuntime>>,
    source: Arc<CountedSource>,
    input: ManagedHostTensor,
}

pub(super) fn run(args: Arguments) -> std::result::Result<Value, Failure> {
    let (package, _fixture_directory) = load(&args).map_err(|e| error("package", e))?;
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
        .map_err(|e| error("Host capacity", e))?,
    );
    let (config, source) = FastH3FfnSource::prepare(
        package.clone(),
        args.block,
        args.rows,
        &host,
        FastH3FfnLimits {
            maximum_rows: args.rows,
            maximum_plan_nodes: args.max_plan_nodes,
            maximum_metadata_bytes: args.max_metadata_bytes,
            maximum_stored_bytes: args.device_bytes,
        },
    )
    .map_err(|e| error("FFN source", e))?;
    let config = config.with_chunk_rows(args.chunk_rows);
    let limits = FastH3FfnLimits {
        maximum_rows: args.rows,
        maximum_plan_nodes: args.max_plan_nodes,
        maximum_metadata_bytes: args.max_metadata_bytes,
        maximum_stored_bytes: args.device_bytes,
    };
    let family =
        TypedFamilyRegistration::new(FastH3FfnFamily::new(limits).map_err(|e| error("family", e))?)
            .prepare_with_profile(
                &serde_json::to_value(&config).map_err(|e| error("configuration", e))?,
                &NumericalProfileId::new(FastH3FfnFamily::PROFILE_ID)
                    .map_err(|e| error("profile", e))?,
            )
            .map_err(|e| error("family", e))?;
    let ModelWeights::Stored(schema) = family.weights() else {
        return Err(error("schema", "FFN must own Stored weights"));
    };
    let selected_bytes = schema
        .components
        .iter()
        .try_fold(0u64, |sum, c| {
            sum.checked_add(c.physical_bytes()?)
                .ok_or_else(|| VNextError::InvalidExecutionPlan {
                    reason: "selected byte count overflow".into(),
                })
        })
        .map_err(|e| error("source bytes", e))?;
    let mut source_receipt = Vec::new();
    for c in &schema.components {
        let payload = source
            .component(c)
            .map_err(|e| error("source identity", e))?;
        source_receipt.push(json!({"id":c.id,"external_names":payload.external_names(),"files":payload.source_files(),"dimensions":c.dimensions,"encoding":c.encoding,"role":c.role,"bytes":payload.bytes().len(),"sha256":format!("{:x}",Sha256::digest(payload.bytes()))}));
    }
    let source_identity = json!({"schema_fingerprint":schema.fingerprint().map_err(|e|error("schema identity",e))?,"selected_payload_bytes":selected_bytes,"components":source_receipt,"derived_scalars":config.scalar_derivations().collect::<Vec<_>>(),"full_package_source_payload_bytes":package.source_bytes()});
    let input = input(&config, args.fixture, &host).map_err(|e| error("input", e))?;
    let input_sha = format!("{:x}", Sha256::digest(input.bytes().bytes()));
    let expected = oracle(&args, &package).map_err(|e| error("independent oracle", e))?;
    let source = Arc::new(CountedSource {
        inner: source,
        reads: AtomicUsize::new(0),
    });
    let device_index = usize::try_from(args.device).map_err(|e| error("device index", e))?;
    let mut device =
        cuda_tensor_runtime_config(device_index, id(format!("device.cuda.{}", args.device)))
            .map_err(|e| error("runtime descriptor", e))?;
    cuda_tensor_runtime_enable_cutlass_nvfp4(&mut device)
        .map_err(|e| error("runtime capability", e))?;
    let runtime = Arc::new(
        CudaDeviceRuntime::new_tensor_eager(CudaTensorRuntimeConfig {
            device,
            usable_memory_bytes: args.device_bytes,
            bootstrap_memory_bytes: args.bootstrap_bytes,
            stream_memory_bytes: args.stream_bytes,
            maximum_streams: 1,
            maximum_in_flight_per_stream: 1,
            maximum_commands_per_submission: 64,
            opaque_memory_profile: "explicit-fast-h3-ffn-qualification-v1".into(),
        })
        .map_err(|e| error("runtime construction", e))?,
    );
    let mut providers = cuda_vnext_nvfp4_bf16_providers(
        &runtime,
        &CutlassNvfp4Config {
            opaque_memory_bytes: args.native_bytes,
            opaque_memory_profile: "explicit-fast-h3-ffn-native-qualification-v1".into(),
        },
    )
    .map_err(|e| Failure::retain("NVFP4 providers", e, runtime.clone()))?;
    let mut ordinary = vec![CudaTensorOperation::PackedSwiGlu];
    if args.chunk_rows.is_some_and(|n| n.get() < args.rows) {
        ordinary.extend([
            CudaTensorOperation::RowGather,
            CudaTensorOperation::RowWrite,
        ]);
    }
    providers.extend(
        cuda_vnext_tensor_providers(&runtime, &ordinary).map_err(|e| {
            Failure::retain("BF16 providers", e, (runtime.clone(), providers.len()))
        })?,
    );
    let mut contracts: Vec<Box<dyn OperationContract>> = vec![
        Box::new(nvfp4_quantize_bf16_contract().map_err(|e| error("quantize contract", e))?),
        Box::new(nvfp4_matmul_bf16_contract().map_err(|e| error("matmul contract", e))?),
        Box::new(packed_swiglu_bf16_contract().map_err(|e| error("SwiGLU contract", e))?),
    ];
    if args.chunk_rows.is_some_and(|n| n.get() < args.rows) {
        contracts.push(Box::new(
            row_gather_bf16_contract().map_err(|e| error("row gather contract", e))?,
        ));
        contracts.push(Box::new(
            row_write_bf16_contract().map_err(|e| error("row write contract", e))?,
        ));
    }
    let registry = Arc::new(
        OperationRuntimeRegistry::new(contracts, providers)
            .map_err(|e| Failure::retain("registry", e, runtime.clone()))?,
    );
    let materializers = WeightMaterializerRegistry::identity_only()
        .map_err(|e| error("identity materializer", e))?;
    let d = runtime.descriptor();
    let engine = EngineProviderDescriptor::new(
        id("provider.engine.cuda.fast_h3.ffn.qualification"),
        ContractVersion::new(1, 0),
        d.runtime_implementation_fingerprint.clone(),
        d.id.clone(),
        d.capabilities.clone(),
    )
    .map_err(|e| error("catalog", e))?;
    let catalog = materializers
        .augment_catalog(
            registry
                .capability_catalog(d.clone(), vec![engine])
                .map_err(|e| error("catalog", e))?,
        )
        .map_err(|e| error("catalog", e))?;
    let policy = TensorComponentPolicy::new(
        args.device_bytes,
        d.dynamic_storage_profiles.iter().copied().collect(),
    )
    .map_err(|e| error("policy", e))?
    .with_execution_order(PlanExecutionOrder::Serial);
    let compiled = Arc::new(
        CompiledTensorComponent::compile(
            family,
            &registry,
            &catalog,
            &policy,
            &materializers,
            &config
                .compile_options()
                .map_err(|e| error("compile options", e))?,
        )
        .map_err(|e| Failure::retain("compile", e, (runtime.clone(), registry.clone())))?,
    );
    let plan = compiled.executable().execution_plan();
    let plan_receipt = json!({"hash":plan.plan_hash(),"static_bytes":plan.payload().memory().static_bytes(),"minimum_step_bytes":plan.payload().memory().minimum_step_bytes(),"providers":plan.payload().nodes().iter().map(|n|json!({"provider":n.provider_resources().provider_id(),"fingerprint":n.provider_implementation_fingerprint()})).collect::<Vec<_>>(),"runtime_fingerprint":d.runtime_implementation_fingerprint});
    let phase = Arc::new(Phase {
        runtime: runtime.clone(),
        registry,
        compiled: compiled.clone(),
        source: source.clone(),
        input,
    });
    let mut session = compiled
        .initialize_retained(
            runtime,
            host.clone(),
            ComponentWeightSource::Stored(source.as_ref()),
            ComponentInitializationConfig {
                run_id: id("run.fast_h3.ffn.initialize"),
                request_id: id("request.fast_h3.ffn.initialize"),
                transaction_id: id("transaction.fast_h3.ffn.initialize"),
                static_policy: StaticInitializationPolicy::new(args.staging_bytes as u64, 64)
                    .map_err(|e| error("staging policy", e))?,
                executor: TensorExecutorConfig::default(),
            },
            phase.clone(),
        )
        .map_err(|e| Failure::retain("initialize", e, phase.clone()))?;
    let initialization = serde_json::to_value(session.session().initialization_receipt())
        .map_err(|e| Failure::retain("initialization receipt", e, phase.clone()))?;
    let source_reads = source.reads.load(Ordering::Relaxed);
    let complete_upload = match session.session().initialization_receipt() {
        ferrum_execution::component::ComponentInitializationReceipt::Initialized { receipt } => {
            receipt
                .uploaded_bytes()
                .checked_add(receipt.imported_bytes())
                == Some(selected_bytes)
        }
        _ => false,
    };
    let mut trials = Vec::new();
    let mut retained = None;
    let mut first_sha = None;
    let mut valid = complete_upload && source_reads == source_receipt.len();
    for trial in 0..args.iterations {
        let started = Instant::now();
        let result = session.session_mut().execute(
            id(format!("run.fast_h3.ffn.{trial}")),
            id(format!("request.fast_h3.ffn.{trial}")),
            vec![phase.input.clone()],
        );
        let wall_seconds = started.elapsed().as_secs_f64();
        let outputs = match result {
            Ok(output) => output,
            Err(e) => return Err(Failure::retain("execute", e, (session, phase))),
        };
        let report = inspect(&outputs, expected.as_deref())
            .map_err(|e| Failure::retain("output", e, phase.clone()))?;
        let sha = report["sha256"]
            .as_str()
            .expect("local SHA report")
            .to_owned();
        let repeated = first_sha.as_ref().is_none_or(|prior| prior == &sha);
        first_sha.get_or_insert(sha);
        let no_reupload = source.reads.load(Ordering::Relaxed) == source_reads;
        valid &= report["finite"].as_bool() == Some(true)
            && report["oracle_matches"].as_bool() != Some(false)
            && repeated
            && no_reupload;
        trials.push(json!({"trial":trial,"component_wall_seconds":wall_seconds,"output":report,"same_output_as_first":(trial>0).then_some(repeated),"source_component_reads":source.reads.load(Ordering::Relaxed),"static_source_not_reread":no_reupload}));
        retained = Some(outputs);
    }
    let (returned_phase, receipt) = session
        .close()
        .map_err(|e| Failure::retain("close", e, phase.clone()))?;
    let retained = retained.expect("positive iteration count");
    let retained_after_close = retained.iter().all(|o| host.owns(o.bytes()));
    drop(returned_phase);
    drop(phase);
    drop(source);
    drop(compiled);
    let after_close = host
        .usage()
        .map_err(|e| error("Host after close", e))?
        .committed_bytes();
    drop(retained);
    let after_drop = host
        .usage()
        .map_err(|e| error("Host retirement", e))?
        .committed_bytes();
    if !valid || !retained_after_close || after_drop != 0 {
        return Err(error("qualification",format!("output/retention checks failed: valid={valid}, retained={retained_after_close}, remaining_Host={after_drop}")));
    }
    Ok(
        json!({"schema":"ferrum.fast_h3.ffn.qualification.v1","arguments":args,"source":source_identity,"plan":plan_receipt,"initialization":initialization,"input_sha256":input_sha,"trials":trials,"repeat_comparison_count":args.iterations-1,"retirement":{"kind":"closed","receipt":receipt,"host_committed_bytes_with_retained_output":after_close,"host_committed_bytes_after_output_drop":after_drop},"numerical_scope":if expected.is_some(){"complete FFN independent CPU FP64/BF16 oracle within dyadic/global1/amax6/SiLU16 exact subset"}else{"real checkpoint finite output; repeated-execution agreement is reported separately; no full numerical oracle"},"timing_scope":"session.execute wall time including input upload and output readback; not GPU kernel timing or throughput/SLO evidence","memory_scope":"one Host payload account; source mmap and metadata are outside this account; capacities are not peak RSS/VRAM measurements"}),
    )
}

fn input(
    c: &FastH3FfnConfig,
    fixture: bool,
    host: &HostMemoryAccount,
) -> Result<ManagedHostTensor> {
    let spec = ResolvedTensorSpec::new(
        vec![c.rows(), c.hidden()],
        ElementType::Bf16,
        ResolvedTensorLayout::Contiguous,
    )?;
    let mut bytes = host.allocate_zeroed(
        HostMemoryClass::InputPreprocess,
        spec.minimum_storage_bytes()?.try_into()?,
    )?;
    let hidden = usize::try_from(c.hidden())?;
    for (index, b) in bytes.as_mut().chunks_exact_mut(2).enumerate() {
        let (r, k) = (index / hidden, index % hidden);
        let value = if fixture {
            if k == 0 {
                4.
            } else {
                [2., 4., 6.][(r + k) % 3]
            }
        } else {
            ((r.wrapping_mul(17) + k.wrapping_mul(7)) % 33) as f32 / 16. - 1.
        };
        b.copy_from_slice(&bf16::from_f32(value).to_bits().to_le_bytes());
    }
    Ok(ManagedHostTensor::new(
        id(FastH3FfnFamily::INPUT_ID),
        spec,
        bytes.freeze(),
    )?)
}
fn inspect(
    outputs: &[ferrum_execution::TensorExecutionOutput],
    expected: Option<&[u16]>,
) -> Result<Value> {
    ensure!(
        outputs.len() == 1
            && outputs[0].value_id() == &id::<ProgramValueId>(FastH3FfnFamily::OUTPUT_ID),
        "FFN requires its declared single output"
    );
    let bytes = outputs[0].bytes().bytes();
    ensure!(bytes.len() % 2 == 0, "BF16 output byte count");
    let mut finite = true;
    let mut maximum = 0f32;
    let mut matches = true;
    if let Some(expected) = expected {
        ensure!(expected.len() == bytes.len() / 2, "oracle shape differs");
    }
    for (index, b) in bytes.chunks_exact(2).enumerate() {
        let bits = u16::from_le_bytes(b.try_into()?);
        let value = bf16::from_bits(bits).to_f32();
        finite &= value.is_finite();
        maximum = maximum.max(value.abs());
        if let Some(expected) = expected {
            matches &= bits == expected[index];
        }
    }
    Ok(
        json!({"bytes":bytes.len(),"sha256":format!("{:x}",Sha256::digest(bytes)),"finite":finite,"maximum_abs":maximum,"oracle_matches":expected.map(|_|matches)}),
    )
}
fn load(args: &Arguments) -> Result<(Arc<FastH3TransformerPackage>, Option<tempfile::TempDir>)> {
    if args.fixture {
        #[cfg(feature = "test-support")]
        {
            let (directory, package) = super::dyadic::fixture()?;
            return Ok((package, Some(directory)));
        }
        #[cfg(not(feature = "test-support"))]
        {
            anyhow::bail!("--fixture dyadic requires test-support");
        }
    }
    let profile = match args.profile.as_str() {
        "ffn" => FastH3Nvfp4Profile::FeedForward,
        "dit" => FastH3Nvfp4Profile::DiT,
        _ => FastH3Nvfp4Profile::DiTWithVsa,
    };
    Ok((
        Arc::new(package::load(
            args.model_dir
                .as_deref()
                .context("missing model directory")?,
            profile,
            args.max_metadata_bytes,
            FastH3WeightLimits {
                maximum_tensors: args.max_tensors,
                maximum_source_bytes: args.max_source_bytes,
                maximum_skip_metadata_bytes: args.max_metadata_bytes,
            },
        )?),
        None,
    ))
}
fn oracle(args: &Arguments, package: &FastH3TransformerPackage) -> Result<Option<Vec<u16>>> {
    if !args.fixture {
        return Ok(None);
    }
    // Bound the independent diagnostic's temporary arrays too; these are not
    // runtime activation owners and are reported separately from Host payloads.
    ensure!(
        args.rows
            .checked_mul(128 * 32)
            .is_some_and(|n| n <= args.host_bytes as u64),
        "tiny oracle temporary arrays exceed explicit Host-sized diagnostic bound"
    );
    #[cfg(feature = "test-support")]
    {
        Ok(Some(super::dyadic::expected(
            package,
            args.rows.try_into()?,
        )?))
    }
    #[cfg(not(feature = "test-support"))]
    {
        let _ = package;
        anyhow::bail!("tiny oracle requires test-support")
    }
}
