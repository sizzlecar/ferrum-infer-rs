use super::*;
use ferrum_execution::{
    component::{
        CompiledTensorComponent, ComponentInitializationConfig, ComponentWeightSource,
        TensorComponentPolicy, TensorComponentSession,
    },
    TensorExecutorConfig,
};
use ferrum_kernels::backend::cuda::{
    vnext_ops::{
        cuda_tensor_runtime_config_with_cutlass_fp8, cuda_vnext_dense_linear_fp8_bf16_provider,
        CudaCutlassFp8Config,
    },
    vnext_runtime::{CudaDeviceRuntime, CudaTensorRuntimeConfig},
};
use std::{io::Write, sync::Arc, time::Instant};

type BenchResult<T> = std::result::Result<T, Failure>;
fn failure<E: std::fmt::Display + 'static>(stage: &'static str, error: E) -> Failure {
    Failure::retain(stage, error, ())
}
fn emit(value: &Value) -> BenchResult<()> {
    let mut stdout = std::io::stdout().lock();
    serde_json::to_writer(&mut stdout, value).map_err(|e| failure("report", e))?;
    writeln!(stdout).map_err(|e| failure("report", e))
}

/// Kept intact on any initialization/execution/retirement failure. Field order
/// drops the session before its compiled provider bindings on the success path.
struct Owners {
    session: Option<TensorComponentSession<CudaDeviceRuntime>>,
    compiled: Option<CompiledTensorComponent<CudaDeviceRuntime>>,
    registry: Option<OperationRuntimeRegistry<CudaDeviceRuntime>>,
    runtime: Arc<CudaDeviceRuntime>,
    host: HostMemoryAccount,
    input: Option<ManagedHostTensor>,
    payload: Option<pattern::Payload>,
}

pub(super) fn run(args: arguments::Arguments) -> BenchResult<()> {
    let extents = args.validate().map_err(|e| failure("preflight", e))?;
    let selected = pattern::expected(args.shape).map_err(|e| failure("oracle preparation", e))?;
    let prepared = family::prepare(&args.shape).map_err(|e| failure("family", e))?;
    let host = HostMemoryAccount::new(args.host_limits().map_err(|e| failure("host limits", e))?);
    // These are the only large CPU payload allocations. No input-sized Vec,
    // duplicate weights, staging outside the account, or full O(M*N*K) oracle.
    let input = pattern::input(args.shape, &host).map_err(|e| failure("input preparation", e))?;
    let payload =
        pattern::Payload::new(args.shape, &host).map_err(|e| failure("weight preparation", e))?;
    let device = cuda_tensor_runtime_config_with_cutlass_fp8(
        args.device,
        id(format!("device.cuda.{}", args.device)),
    )
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
            opaque_memory_profile: "explicit-managed-fp8-benchmark-v1".into(),
        })
        .map_err(|e| failure("runtime construction", e))?,
    );
    let mut owners = Owners {
        session: None,
        compiled: None,
        registry: None,
        runtime,
        host,
        input: Some(input),
        payload: Some(payload),
    };

    let result = (|| -> BenchResult<()> {
        let provider = cuda_vnext_dense_linear_fp8_bf16_provider(
            &owners.runtime,
            &CudaCutlassFp8Config {
                opaque_memory_bytes: args.native_bytes,
                maximum_plans: 1,
                opaque_memory_profile: "explicit-managed-fp8-benchmark-native-v1".into(),
            },
        )
        .map_err(|e| failure("provider", e))?;
        owners.registry = Some(
            OperationRuntimeRegistry::new(
                vec![Box::new(
                    dense_linear_fp8_bf16_contract().map_err(|e| failure("contract", e))?,
                )],
                vec![provider],
            )
            .map_err(|e| failure("registry", e))?,
        );
        let registry = owners.registry.as_ref().expect("registry installed");
        let descriptor = owners.runtime.descriptor();
        let engine = EngineProviderDescriptor::new(
            id("provider.engine.cuda.fp8.benchmark"),
            ContractVersion::new(1, 0),
            descriptor.runtime_implementation_fingerprint.clone(),
            descriptor.id.clone(),
            descriptor.capabilities.clone(),
        )
        .map_err(|e| failure("engine catalog", e))?;
        let catalog = registry
            .capability_catalog(descriptor.clone(), vec![engine])
            .map_err(|e| failure("capability catalog", e))?;
        let policy = TensorComponentPolicy::new(
            args.device_bytes,
            descriptor
                .dynamic_storage_profiles
                .iter()
                .copied()
                .collect(),
        )
        .map_err(|e| failure("component policy", e))?;
        let options = ProgramPlanCompileOptions::new(BTreeMap::from([(
            id(X),
            tensor(vec![args.shape.m, args.shape.k]),
        )]))
        .map_err(|e| failure("compile options", e))?;
        let materializers = WeightMaterializerRegistry::identity_only()
            .map_err(|e| failure("identity materializer", e))?;
        let started = Instant::now();
        let compiled = CompiledTensorComponent::compile(
            prepared,
            registry,
            &catalog,
            &policy,
            &materializers,
            &options,
        )
        .map_err(|e| failure("compile", e))?;
        let compile_us = started.elapsed().as_micros();
        let plan = compiled.executable().execution_plan();
        let plan_receipt = json!({
            "hash":plan.plan_hash(),
            "static_bytes":plan.payload().memory().static_bytes(),
            "minimum_step_bytes":plan.payload().memory().minimum_step_bytes(),
            "scratch_bytes":plan.payload().nodes()[0].provider_resources().scratch()
                .and_then(|s|s.fixed_bytes()),
            "provider_id":plan.payload().nodes()[0].provider_resources().provider_id(),
            "provider_implementation_fingerprint":plan.payload().nodes()[0].provider_implementation_fingerprint(),
            "runtime_implementation_fingerprint":descriptor.runtime_implementation_fingerprint,
        });
        owners.compiled = Some(compiled);
        let started = Instant::now();
        owners.session = Some(
            owners
                .compiled
                .as_ref()
                .expect("compiled installed")
                .initialize(
                    Arc::clone(&owners.runtime),
                    owners.host.clone(),
                    ComponentWeightSource::Stored(
                        owners.payload.as_ref().expect("payload installed"),
                    ),
                    ComponentInitializationConfig {
                        request_id: id("request.fp8.benchmark.init"),
                        run_id: id("run.fp8.benchmark.init"),
                        transaction_id: id("transaction.fp8.benchmark.init"),
                        static_policy: StaticInitializationPolicy::new(
                            args.staging_bytes as u64,
                            64,
                        )
                        .map_err(|e| failure("static policy", e))?,
                        executor: TensorExecutorConfig::default(),
                    },
                )
                .map_err(|e| failure("initialize", e))?,
        );
        let initialize_us = started.elapsed().as_micros();
        let initialization =
            serde_json::to_value(owners.session.as_ref().unwrap().initialization_receipt())
                .map_err(|e| failure("initialization receipt", e))?;
        drop(owners.payload.take());
        let usage = owners
            .host
            .usage()
            .map_err(|e| failure("Host observation", e))?;
        if usage.class_committed_bytes(HostMemoryClass::WeightBacking) != 0 {
            return Err(failure(
                "Host retirement",
                anyhow::anyhow!("source WeightBacking remained after initialization"),
            ));
        }
        emit(&json!({
            "schema":"ferrum.managed_fp8_benchmark.declaration.v1",
            "arguments":args,"extents":extents,"plan":plan_receipt,
            "profile":PROFILE,"compile_us":compile_us,"initialize_us":initialize_us,
            "initialization":initialization,
            "data":{
                "identity":"finite-dyadic-v1",
                "input":"X[r,k] = sign(r,k)*(448 if k=0 or (r+k)%3=0 else 224); sign=-1 iff k>0 and (r+3*k)%7=0",
                "weight":"E4M3 W[n,k] = sign(n,k)*(1/16 if (n+2*k)%3=0 else 1/32); sign=-1 iff k>0 and (3*n+5*k)%11=0",
                "weight_scale":pattern::WEIGHT_SCALE,
                "activation_scale":"each row amax=448, native row scale exactly 1",
                "selected_oracle":selected,
                "oracle_domain":"each product is an exact integer |p|<=28; 28*K<=2^24 bounds every partial sum; scale=2^-8 then BF16 RNE",
                "qualification":"synthetic shape diagnostic; selected dots, not full tensor numeric or checkpoint/image quality validation"
            },
            "timing_scope":"session.execute component wall: admission, uploads, encode, submit/wait, readback; excludes compile/init, construction, checks and logging; no GPU event or separate quant/GEMM timings",
            "memory_scope":"tensor source/input/output/staging payloads use one Host account; metadata/allocator overhead excluded; capacities are not peak VRAM/RSS measurements",
            "native_retirement_scope":"CUTLASS opaque module claim conservatively lasts until process exit; successful session close only certifies its managed resources",
        }))?;
        let mut measured = 0_u32;
        let mut sum_us = 0_f64;
        let mut min_us = f64::INFINITY;
        let mut max_us = 0_f64;
        for (warmup, count) in [(true, args.warmup), (false, args.iterations)] {
            for iteration in 0..count {
                let phase = if warmup { "warmup" } else { "measured" };
                let run_id = id(format!("run.fp8.benchmark.{phase}.{iteration}"));
                let request_id = id(format!("request.fp8.benchmark.{phase}.{iteration}"));
                let inputs = vec![owners.input.as_ref().expect("input retained").clone()];
                let started = Instant::now();
                let outputs = owners
                    .session
                    .as_mut()
                    .expect("session installed")
                    .execute(run_id, request_id, inputs);
                let wall_us = started.elapsed().as_secs_f64() * 1e6;
                let outputs = outputs.map_err(|e| failure("execute", e))?;
                if outputs.len() != 1 || outputs[0].value_id() != &id::<ProgramValueId>(Y) {
                    return Err(failure(
                        "output identity",
                        anyhow::anyhow!("one declared output required"),
                    ));
                }
                let validation = pattern::check(args.shape, outputs[0].bytes().bytes(), &selected)
                    .map_err(|e| failure("output validation", e))?;
                let host_observed = owners
                    .host
                    .usage()
                    .map_err(|e| failure("Host observation", e))?
                    .committed_bytes();
                drop(outputs);
                if !warmup {
                    measured += 1;
                    sum_us += wall_us;
                    min_us = min_us.min(wall_us);
                    max_us = max_us.max(wall_us);
                }
                emit(&json!({"schema":"ferrum.managed_fp8_benchmark.sample.v1",
                    "phase":phase,"iteration":iteration,"component_wall_us":wall_us,
                    "validation":validation,"observed_live_host_payload_bytes":host_observed}))?;
            }
        }
        let session = owners.session.take().expect("session installed");
        let close = match session.close() {
            Ok(PlanRuntimeCloseOutcome::Closed(receipt)) => receipt,
            Ok(outcome @ PlanRuntimeCloseOutcome::Referenced { .. }) => {
                return Err(Failure::retain(
                    "close",
                    anyhow::anyhow!("runtime still referenced"),
                    outcome,
                ))
            }
            Err(error) => {
                let message = format!("{:?}", error.failure());
                return Err(Failure::retain("close", anyhow::anyhow!(message), error));
            }
        };
        drop(owners.input.take());
        let final_usage = owners
            .host
            .usage()
            .map_err(|e| failure("final Host observation", e))?;
        if final_usage.committed_bytes() != 0 {
            return Err(failure(
                "final Host retirement",
                anyhow::anyhow!("managed Host payloads remain live"),
            ));
        }
        emit(&json!({"schema":"ferrum.managed_fp8_benchmark.summary.v1",
            "measured_iterations":measured,
            "component_wall_us":{"mean":sum_us/f64::from(measured),"min":min_us,"max":max_us},
            "retirement":{"kind":"closed","receipt":close},
            "host_committed_bytes_after_close":final_usage.committed_bytes(),
            "not_measured":["kernel durations","quantization/GEMM split","peak VRAM","process RSS","image throughput"],
        }))?;
        Ok(())
    })();
    match result {
        Ok(()) => Ok(()),
        Err(mut error) => {
            error.owner = Box::new((error.owner, owners));
            Err(error)
        }
    }
}
