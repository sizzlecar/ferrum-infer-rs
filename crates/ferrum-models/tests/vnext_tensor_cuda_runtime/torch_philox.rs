//! Optional deployment-specific byte oracle; invoke its ignored GPU test only
//! in an exclusive validation window.
//! Run the ignored build test in the configured Linux compiler container, then
//! use executable.mode=prebuilt on the GPU host. No Python, NVCC, tch or product
//! build hook is involved. The C++ child uses its own explicit Torch loader path.
use super::*;
use anyhow::{ensure, Context, Result};
use ferrum_kernels::backend::cuda::vnext_ops::{
    cuda_vnext_philox_normal_bf16_provider, CudaPhiloxDeviceGeometry,
    CUDA_PHILOX_BUILD_IDENTITY_SHA256, CUDA_PHILOX_NORMAL_BF16_NUMERICAL_PROFILE_ID,
};
use ferrum_models::normal_noise::{NormalNoiseConfig, NormalNoiseFamily};

#[path = "../support/torch_native_oracle.rs"]
mod native;

#[test]
#[ignore = "explicit FERRUM_TEST_TORCH_ORACLE_CONFIG Build mode; Linux C++ compiler, no GPU needed"]
fn build_deployed_torch_philox_oracle_binary() {
    let config = native::Config::from_test_environment().unwrap();
    assert!(
        matches!(&config.executable, native::Executable::Build { .. }),
        "build-only test requires Build mode"
    );
    let oracle = native::Oracle::prepare(config).unwrap();
    eprintln!(
        "{}",
        serde_json::json!({
            "executable": oracle.executable,
            "executable_sha256": oracle.executable_sha256,
            "build_receipt": oracle.build_receipt,
            "gpu_executed": false,
        })
    );
}

fn prepare_family(config: &NormalNoiseConfig) -> PreparedModelFamily {
    TypedFamilyRegistration::new(
        NormalNoiseFamily::new(CUDA_PHILOX_NORMAL_BF16_NUMERICAL_PROFILE_ID).unwrap(),
    )
    .prepare_with_profile(
        &serde_json::to_value(config).unwrap(),
        &NumericalProfileId::new(CUDA_PHILOX_NORMAL_BF16_NUMERICAL_PROFILE_ID).unwrap(),
    )
    .unwrap()
}

fn state(host: &HostMemoryAccount, seed: u64, offset: u64) -> ManagedHostTensor {
    let mut storage = host
        .allocate_zeroed(HostMemoryClass::InputPreprocess, 16)
        .unwrap();
    storage
        .as_mut()
        .copy_from_slice(&PhiloxSeedOffset::new(seed, offset).little_endian_bytes());
    ManagedHostTensor::new(
        id(NormalNoiseFamily::STATE_ID),
        NormalNoiseConfig::state_tensor().unwrap(),
        storage.freeze(),
    )
    .unwrap()
}

fn compare_bytes(expected: &[u8], actual: &[u8]) -> Result<()> {
    ensure!(
        !expected.is_empty() && expected.len() % 2 == 0 && expected.len() == actual.len(),
        "output extent mismatch"
    );
    let mut first = None;
    let mut mismatches = 0_usize;
    for (index, (left, right)) in expected
        .chunks_exact(2)
        .zip(actual.chunks_exact(2))
        .enumerate()
    {
        if left != right {
            mismatches += 1;
            first.get_or_insert((
                index,
                u16::from_le_bytes([left[0], left[1]]),
                u16::from_le_bytes([right[0], right[1]]),
            ));
        }
    }
    ensure!(mismatches == 0, "Torch/Ferrum BF16 mismatch: count={mismatches}, first(index,Torch,Ferrum)={first:?}, Torch SHA256={}, Ferrum SHA256={}", native::bytes_sha(expected), native::bytes_sha(actual));
    Ok(())
}

fn compare_case(
    oracle: &mut native::Oracle,
    runtime: &Arc<CudaDeviceRuntime>,
    registry: &OperationRuntimeRegistry<CudaDeviceRuntime>,
    catalog: &CapabilityCatalog,
    case: native::Case,
    label: &str,
) -> Result<()> {
    let config = NormalNoiseConfig::new(case.channels, case.height, case.width)?;
    let output_bytes = usize::try_from(case.payload_bytes()?)?;
    let host = HostMemoryAccount::new(HostMemoryLimits::new(
        3 * output_bytes + 16,
        HostMemoryClassLimits {
            input_preprocess_bytes: 16,
            output_bytes: 3 * output_bytes,
            transfer_staging_bytes: 0,
            weight_backing_bytes: 0,
        },
    )?);
    // Prepare independent expected bytes before submitting the candidate. The
    // actual Torch process allocates outside Ferrum and exits before comparison.
    let expected = oracle.generate(case, &host)?;
    ensure!(
        host.usage()?.allocated_bytes() == usize::try_from(case.draws)? * output_bytes,
        "oracle Host accounting mismatch"
    );
    let family = prepare_family(&config);
    let compiled = ProgramPlanCompiler::compile(
        &family,
        catalog,
        &policy(runtime.as_ref()),
        &registry.planning(),
        &config.compile_options()?,
    )?;
    let executable = compiled.executable();
    let resources = no_static(runtime.clone(), executable.execution_plan());
    let result = (|| -> Result<()> {
        let mut executor = PreparedTensorExecutor::new(
            executable.clone(),
            registry.bind_plan(executable)?,
            resources.clone(),
            host.clone(),
            TensorExecutorConfig::default(),
        )
        .map_err(|error| anyhow::anyhow!("executor construction: {error}"))?;
        let launch = CudaPhiloxDeviceGeometry::query(runtime)?.launch(case.payload_bytes()? / 2)?;
        for (draw_index, expected_draw) in expected.draws.iter().enumerate() {
            ensure!(
                expected_draw
                    .offset_before
                    .checked_add(launch.offset_advance())
                    == Some(expected_draw.offset_after),
                "Torch offset advance differs from declared Ferrum geometry"
            );
            let outputs = executor
                .execute(
                    id(format!("run.cuda.torch_oracle.{label}.{draw_index}")),
                    id(format!("request.cuda.torch_oracle.{label}.{draw_index}")),
                    vec![state(&host, case.seed, expected_draw.offset_before)],
                )
                .unwrap_or_else(|error| panic!("managed candidate execution failed: {error:?}"));
            ensure!(
                outputs.len() == 1
                    && outputs[0].value_id().as_str() == NormalNoiseFamily::OUTPUT_ID,
                "unexpected candidate output ports"
            );
            ensure!(
                outputs[0].tensor() == &config.output_tensor()? && host.owns(outputs[0].bytes()),
                "candidate storage is not the declared managed NCHW output"
            );
            let comparison = compare_bytes(expected_draw.bytes.bytes(), outputs[0].bytes().bytes());
            eprintln!(
                "{}",
                serde_json::json!({
                    "case": label, "draw": draw_index, "seed": case.seed,
                    "offset_before": expected_draw.offset_before, "offset_after": expected_draw.offset_after,
                    "shape": [1, case.channels, case.height, case.width],
                    "blocks": launch.blocks(), "rounds": launch.rounds(),
                    "torch_sha256": native::bytes_sha(expected_draw.bytes.bytes()),
                    "ferrum_sha256": native::bytes_sha(outputs[0].bytes().bytes()),
                    "exact_bytes_equal": comparison.is_ok(),
                })
            );
            drop(outputs);
            ensure!(
                host.usage()?.allocated_bytes() == usize::try_from(case.draws)? * output_bytes,
                "candidate owners did not retire"
            );
            comparison.with_context(|| format!("{label}, draw {draw_index}"))?;
        }
        eprintln!(
            "{}",
            serde_json::json!({
                "deployment_profile": oracle.config.deployment_profile,
                "oracle_executable_sha256": oracle.executable_sha256,
                "oracle_metadata": expected.metadata,
                "actual_library_sha256": expected.library_sha256,
                "ferrum_profile": CUDA_PHILOX_NORMAL_BF16_NUMERICAL_PROFILE_ID,
                "ferrum_build_identity": CUDA_PHILOX_BUILD_IDENTITY_SHA256,
                "scope": "explicit cases on this deployed LibTorch/device; no global or CPU parity claim",
            })
        );
        Ok(())
    })();
    // A numerical mismatch still closes the real plan after executor/output
    // owners leave the closure. Native nonterminal failures retain their normal
    // fail-closed authority; they are not converted into a numerical mismatch.
    drop(compiled);
    close_resources(resources);
    drop(expected);
    ensure!(
        host.usage()?.committed_bytes() == 0 && host.usage()?.allocated_bytes() == 0,
        "Host charge remained after comparison cleanup"
    );
    result
}

#[test]
#[ignore = "exclusive CUDA plus explicit deployed LibTorch oracle configuration; exact bytes, not tolerance"]
fn managed_cuda_philox_matches_deployed_torch_bytes() {
    let mut oracle =
        native::Oracle::prepare(native::Config::from_test_environment().unwrap()).unwrap();
    let ordinal = usize::try_from(oracle.config.device).unwrap();
    let runtime = Arc::new(
        CudaDeviceRuntime::new_tensor_eager(CudaTensorRuntimeConfig {
            device: cuda_tensor_runtime_config(ordinal, id(format!("device.cuda.{ordinal}")))
                .unwrap(),
            usable_memory_bytes: USABLE_BYTES,
            bootstrap_memory_bytes: 1 << 30,
            stream_memory_bytes: 64 << 20,
            maximum_streams: 1,
            maximum_in_flight_per_stream: 1,
            maximum_commands_per_submission: 8,
            opaque_memory_profile: "experimental-unqualified-torch-oracle-test-v1".into(),
        })
        .unwrap(),
    );
    let device_geometry = CudaPhiloxDeviceGeometry::query(&runtime).unwrap();
    let registry = OperationRuntimeRegistry::new(
        vec![Box::new(philox_normal_bf16_contract().unwrap())],
        vec![cuda_vnext_philox_normal_bf16_provider(&runtime).unwrap()],
    )
    .unwrap();
    let descriptor = runtime.descriptor();
    let engine = EngineProviderDescriptor::new(
        id("provider.engine.cuda.tensor.torch_oracle"),
        ContractVersion::new(1, 0),
        descriptor.runtime_implementation_fingerprint.clone(),
        descriptor.id.clone(),
        descriptor.capabilities.clone(),
    )
    .unwrap();
    let catalog = registry
        .capability_catalog(descriptor.clone(), vec![engine])
        .unwrap();
    let round = u64::from(device_geometry.multiprocessors())
        .checked_mul(u64::from(
            device_geometry.max_threads_per_multiprocessor() / 256,
        ))
        .unwrap()
        .checked_mul(4 * 256)
        .unwrap();
    let base = native::Case {
        seed: 42,
        offset: 0,
        channels: 1,
        height: 1,
        width: 257,
        draws: 2,
    };
    // Cases exercise launch/tail and real latent layout boundaries, both seed
    // words, and state advance. Only fresh offset 0 targets initial pipeline
    // noise; offset 4 is an explicitly separate generator diagnostic.
    let cases = [
        ("block_tail", base),
        (
            "latent_nchw",
            native::Case {
                channels: 64,
                height: 64,
                width: 64,
                ..base
            },
        ),
        (
            "normal4_round_tail",
            native::Case {
                width: round + 1,
                ..base
            },
        ),
        (
            "high_seed_word",
            native::Case {
                seed: 42 ^ (1_u64 << 40),
                ..base
            },
        ),
        ("offset_diagnostic", native::Case { offset: 4, ..base }),
    ];
    let result = cases.into_iter().try_for_each(|(label, case)| {
        compare_case(&mut oracle, &runtime, &registry, &catalog, case, label)
    });
    drop((catalog, registry));
    drop(cuda_vnext_philox_normal_bf16_provider(&runtime).unwrap());
    drop(runtime.create_stream().unwrap());
    drop(runtime);
    result.unwrap();
}

#[test]
fn byte_comparison_detects_signed_zero_and_nan_payload_differences() {
    assert!(compare_bytes(&[0, 0], &[0, 0x80]).is_err());
    assert!(compare_bytes(&[0xc1, 0x7f], &[0xc2, 0x7f]).is_err());
    compare_bytes(&[0, 0x80, 0xc1, 0x7f], &[0, 0x80, 0xc1, 0x7f]).unwrap();
}
