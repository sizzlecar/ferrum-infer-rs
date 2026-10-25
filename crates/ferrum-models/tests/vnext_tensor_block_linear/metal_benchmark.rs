//! Opt-in paired native Q4_K decode+GEMM versus the identical decoded BF16 OI
//! weight. Both weights are installed before timing. Each sample measures
//! managed execute/admission/encode/wait/readback, not GPU kernel timestamps.
//! Run the ignored `managed_metal_q4k_vs_dense_microbench` test with exclusive
//! GPU access and --nocapture --test-threads=1; preserve its JSON sample lines.
use super::*;
use sha2::{Digest, Sha256};
use std::time::Instant;

#[path = "metal_benchmark/dense.rs"]
mod dense;

const DEVICE_BYTES: u64 = 1 << 30;
const HOST_BYTES: usize = 2304 << 20;

fn context() -> (
    Arc<MetalDeviceRuntime>,
    HostMemoryAccount,
    OperationRuntimeRegistry<MetalDeviceRuntime>,
    CapabilityCatalog,
) {
    let runtime = Arc::new(
        MetalDeviceRuntime::new_tensor_eager(MetalTensorRuntimeConfig {
            device: metal_tensor_runtime_config(id("device.metal.0")).unwrap(),
            usable_memory_bytes: DEVICE_BYTES,
            bootstrap_memory_bytes: 16 << 20,
            stream_memory_bytes: 1 << 20,
            maximum_streams: 1,
            maximum_in_flight_per_stream: 1,
            maximum_commands_per_submission: 8,
            opaque_memory_profile: "metal.block-dense.microbench.declared.v1".into(),
        })
        .unwrap(),
    );
    let host = runtime
        .tensor_host_account(
            HostMemoryLimits::new(
                HOST_BYTES,
                HostMemoryClassLimits {
                    input_preprocess_bytes: 128 << 20,
                    output_bytes: 128 << 20,
                    transfer_staging_bytes: 0,
                    weight_backing_bytes: 2 << 30,
                },
            )
            .unwrap(),
        )
        .unwrap();
    let kinds = [
        MetalTensorOperation::DenseLinear,
        MetalTensorOperation::BlockLinear,
    ];
    let registry = OperationRuntimeRegistry::new(
        kinds
            .iter()
            .map(|k| Box::new(k.contract().unwrap()) as Box<dyn OperationContract>)
            .collect(),
        metal_tensor_providers(&runtime, &kinds).unwrap(),
    )
    .unwrap();
    let engine = EngineProviderDescriptor::new(
        id("provider.engine.metal.block-dense.microbench"),
        ContractVersion::new(1, 0),
        runtime
            .descriptor()
            .runtime_implementation_fingerprint
            .clone(),
        runtime.descriptor().id.clone(),
        runtime.descriptor().capabilities.clone(),
    )
    .unwrap();
    let catalog = registry
        .capability_catalog(runtime.descriptor().clone(), vec![engine])
        .unwrap();
    (runtime, host, registry, catalog)
}

fn activation(host: &HostMemoryAccount, shape: &Shape) -> ManagedHostTensor {
    let mut bytes = host
        .allocate_zeroed(
            HostMemoryClass::InputPreprocess,
            (shape.m * shape.k * 2) as usize,
        )
        .unwrap();
    for row in 0..shape.m {
        let inner = (row * 17 + 3) % shape.k;
        let scale = [0.25_f32, -0.5, 1.0, -2.0][(row % 4) as usize];
        let offset = ((row * shape.k + inner) * 2) as usize;
        bytes.as_mut()[offset..offset + 2].copy_from_slice(&bf16::from_f32(scale).to_le_bytes());
    }
    ManagedHostTensor::new(
        id(X),
        ResolvedTensorSpec::new(
            vec![shape.m, shape.k],
            ElementType::Bf16,
            ResolvedTensorLayout::Contiguous,
        )
        .unwrap(),
        bytes.freeze(),
    )
    .unwrap()
}

fn check_output(shape: &Shape, decoded: &[bf16], bytes: &[u8]) -> Result<(), String> {
    if bytes.len() as u64 != shape.m * shape.n * 2 {
        return Err("output byte length differs from M*N BF16".into());
    }
    // One signed power-of-two activation per row makes the exact independent
    // result a Candle-decoded weight scaled in FP64. All outputs are checked;
    // the provider still executes a full dense GEMM (no sparse operation).
    // Odd stride 17 permutes all 4096 K lanes in the full-row cases.
    for (index, word) in bytes.chunks_exact(2).enumerate() {
        let (row, column) = (index as u64 / shape.n, index as u64 % shape.n);
        let inner = (row * 17 + 3) % shape.k;
        let scale = [0.25_f64, -0.5, 1.0, -2.0][(row % 4) as usize];
        let expected = f64::from(decoded[(column * shape.k + inner) as usize].to_f32()) * scale;
        let actual = bf16::from_bits(u16::from_le_bytes(word.try_into().unwrap())).to_f32();
        if !actual.is_finite() || f64::from(actual) != expected {
            return Err(format!(
                "[{row},{column}]: actual={actual} exact decoded-FP64={expected}"
            ));
        }
    }
    Ok(())
}

#[test]
#[ignore = "large paired Metal Q4_K/dense timing; requires exclusive GPU access"]
fn managed_metal_q4k_vs_dense_microbench() {
    // The final row count is a representative previously measured R210 tail,
    // not a claim about a particular real edit's tokenized sequence length.
    for (case, (m, n, k)) in [(4096, 4096, 4096), (4096, 12288, 4096), (8208, 4096, 4096)]
        .into_iter()
        .enumerate()
    {
        let shape = Shape {
            m,
            n,
            k,
            blocks: Format::Q4K.blocks(),
        };
        eprintln!(
            "block_linear_microbench={}",
            serde_json::json!({
                "event":"case_started", "case":case, "shape":{"m":m,"n":n,"k":k},
                "device_budget_bytes":DEVICE_BYTES, "host_budget_bytes":HOST_BYTES,
                "status":"preparation_pending"
            })
        );
        let (runtime, host, registry, catalog) = context();
        let capacity = DeviceCapacityOwner::open(runtime.descriptor(), DEVICE_BYTES).unwrap();
        // Explicitly reserve the external CPU fixture's temporary working set:
        // eight F32-equivalent matrix copies plus 16MiB for quantizer/writer
        // staging. This is an admission reservation, not process RSS telemetry.
        // It bounds this diagnostic's chosen shapes without changing production
        // Host defaults, and remains live throughout Candle quantize/decode.
        let preparation_bytes = (32 * n * k + (16 << 20)) as usize;
        let preparation = host
            .reserve(HostMemoryClass::WeightBacking, preparation_bytes)
            .unwrap();
        let fixture = Fixture::new(&shape, Format::Q4K);
        let retained_fixture_bytes =
            fixture.decoded.len() * 2 + fixture._file.as_file().metadata().unwrap().len() as usize;
        let retained_fixture = host
            .reserve(HostMemoryClass::WeightBacking, retained_fixture_bytes)
            .unwrap();
        let mut decoded = host
            .allocate_zeroed(HostMemoryClass::WeightBacking, fixture.decoded.len() * 2)
            .unwrap();
        for (word, value) in decoded.as_mut().chunks_exact_mut(2).zip(&fixture.decoded) {
            word.copy_from_slice(&value.to_le_bytes());
        }
        let dense = dense::DenseSource(decoded.freeze());
        drop(preparation);
        let input = activation(&host, &shape);
        let policy = TensorComponentPolicy::new(
            DEVICE_BYTES,
            runtime
                .descriptor()
                .dynamic_storage_profiles
                .iter()
                .copied()
                .collect(),
        )
        .unwrap();
        let options =
            ProgramPlanCompileOptions::new(BTreeMap::from([(id(X), tensor(vec![m, k]))])).unwrap();
        let dense_kind = MetalTensorOperation::DenseLinear;
        let block_kind = MetalTensorOperation::BlockLinear;
        let dense_prepared = TypedFamilyRegistration::new(dense::DenseFamily(family::Family {
            profile: dense_kind.profile(),
        }))
        .prepare_with_profile(
            &serde_json::to_value(&shape).unwrap(),
            &id(dense_kind.profile()),
        )
        .unwrap();
        let materializers = WeightMaterializerRegistry::identity_only().unwrap();
        let compile = |family| {
            CompiledTensorComponent::compile(
                family,
                &registry,
                &catalog,
                &policy,
                &materializers,
                &options,
            )
            .unwrap()
        };
        let dense_component = compile(dense_prepared);
        let block_component = compile(family::prepare_with_profile(&shape, block_kind.profile()));
        let memory = |c: &CompiledTensorComponent<MetalDeviceRuntime>| {
            let p = c.executable().execution_plan().payload();
            serde_json::json!({"static_bytes":p.memory().static_bytes(),
                "minimum_runnable_bytes":p.memory().minimum_runnable_request_bytes(),
                "scratch_bytes":p.nodes()[0].provider_resources().scratch().and_then(|s|s.fixed_bytes())})
        };
        let dense_memory = memory(&dense_component);
        let block_memory = memory(&block_component);
        assert_eq!(
            block_component
                .executable()
                .execution_plan()
                .payload()
                .nodes()[0]
                .provider_resources()
                .scratch()
                .unwrap()
                .fixed_bytes()
                .unwrap(),
            n * k * 2
        );
        let init = |name: &str| ComponentInitializationConfig {
            request_id: id(format!("request.block-dense.init.{case}.{name}")),
            run_id: id(format!("run.block-dense.init.{case}.{name}")),
            transaction_id: id(format!("transaction.block-dense.init.{case}.{name}")),
            static_policy: StaticInitializationPolicy::new(16 << 20, 16).unwrap(),
            executor: TensorExecutorConfig::default(),
        };
        let mut sessions = [
            dense_component
                .initialize(
                    runtime.clone(),
                    host.clone(),
                    ComponentWeightSource::Stored(&dense),
                    init("dense"),
                )
                .unwrap(),
            block_component
                .initialize(
                    runtime.clone(),
                    host.clone(),
                    ComponentWeightSource::Stored(&fixture.source),
                    init("q4k"),
                )
                .unwrap(),
        ];
        eprintln!(
            "block_linear_microbench={}",
            serde_json::json!({
                "event":"setup", "case":case, "shape":{"m":m,"n":n,"k":k},
                "weight_layout":"OI", "native_format":"GGUF Q4_K", "dense_source":"same Candle Q4_K decode rounded BF16",
                "dense_weight_sha256":format!("{:x}",Sha256::digest(dense.0.bytes())),
                "packed_weight_sha256":format!("{:x}",Sha256::digest(fixture.source.tensor("projection.weight").unwrap().bytes())),
                "dense_plan":dense_memory,"q4k_plan":block_memory,
                "device_budget_bytes":DEVICE_BYTES,"host_budget_bytes":HOST_BYTES,
                "fixture_preparation_reservation_bytes":preparation_bytes,"fixture_retained_reservation_bytes":retained_fixture_bytes,
                "runtime_fingerprint":runtime.descriptor().runtime_implementation_fingerprint,
                "timing_scope":"managed_execute_admission_encode_wait_readback",
                "native_memory_scope":"sampled_device_handle_peak_since_this_shape_runtime_creation_not_per_arm_or_RSS",
            })
        );
        let mut timings = [Vec::new(), Vec::new()];
        let mut hashes = [None::<String>, None::<String>];
        for pair in 0..5 {
            let order = if pair % 2 == 0 { [0, 1] } else { [1, 0] };
            for (position, arm) in order.into_iter().enumerate() {
                let inputs = vec![input.clone()];
                let run = id(format!("run.block-dense.{case}.{pair}.{arm}"));
                let request = id(format!("request.block-dense.{case}.{pair}.{arm}"));
                let start = Instant::now();
                let result = sessions[arm].execute(run, request, inputs);
                let ms = start.elapsed().as_secs_f64() * 1000.0;
                let output = match result {
                    Ok(output) => output,
                    Err(error) => {
                        eprintln!(
                            "block_linear_microbench={}",
                            serde_json::json!({"event":"sample","case":case,"arm":arm,"pair":pair,"position":position,"warmup":pair==0,"elapsed_ms":ms,"status":"execution_failed","error":error.to_string(),"native_memory":runtime.native_memory_snapshot()})
                        );
                        panic!("paired block/dense execution failed: {error}");
                    }
                };
                assert_eq!(output.len(), 1);
                let bytes = output[0].bytes().bytes();
                let validation = check_output(&shape, &fixture.decoded, bytes);
                let hash = format!("{:x}", Sha256::digest(bytes));
                eprintln!(
                    "block_linear_microbench={}",
                    serde_json::json!({
                        "event":"sample","case":case,"arm":if arm==0 {"dense_bf16"} else {"native_q4k"},
                        "pair":pair,"position":position,"warmup":pair==0,"elapsed_ms":ms,
                        "status":if validation.is_ok() {"pass"} else {"invalid_output"},"validation_error":validation.as_ref().err(),
                        "output_sha256":hash,"native_memory":runtime.native_memory_snapshot(),
                        "host_committed_bytes":host.usage().unwrap().committed_bytes(),
                    })
                );
                validation.unwrap();
                if let Some(previous) = &hashes[arm] {
                    assert_eq!(&hash, previous, "repeat changed bytes");
                }
                hashes[arm] = Some(hash);
                timings[arm].push(ms);
            }
        }
        eprintln!(
            "block_linear_microbench={}",
            serde_json::json!({"event":"summary","case":case,
            "shape":{"m":m,"n":n,"k":k},"dense_warmup_ms":timings[0][0],"q4k_warmup_ms":timings[1][0],
            "dense_measured_ms":&timings[0][1..],"q4k_measured_ms":&timings[1][1..],
            "identical_output_bytes":hashes[0]==hashes[1],"native_memory":runtime.native_memory_snapshot(),
            "interpretation":"diagnostic same-weight managed cost; no kernel-only or trained-model performance claim"})
        );
        for session in sessions {
            assert!(matches!(
                session
                    .close()
                    .unwrap_or_else(|e| panic!("close: {:?}", e.failure())),
                PlanRuntimeCloseOutcome::Closed(_)
            ));
        }
        drop((
            dense_component,
            block_component,
            dense,
            input,
            fixture,
            retained_fixture,
        ));
        assert_eq!(host.usage().unwrap().committed_bytes(), 0);
        drop(registry);
        // A new provider must be admitted after both old component owners retire.
        drop(metal_tensor_providers(&runtime, &[MetalTensorOperation::DenseLinear]).unwrap());
        let native_after_close = runtime.native_memory_snapshot();
        drop(runtime);
        let all = capacity
            .reserve(DeviceOpaqueMemoryKind::Bootstrap, DEVICE_BYTES)
            .unwrap();
        assert!(capacity.reserve(DeviceOpaqueMemoryKind::Stream, 1).is_err());
        drop(all);
        eprintln!(
            "block_linear_microbench={}",
            serde_json::json!({
                "event":"retirement", "case":case, "host_committed_bytes":host.usage().unwrap().committed_bytes(),
                "device_capacity_fully_reclaimable":true, "native_before_runtime_drop":native_after_close
            })
        );
    }
}
