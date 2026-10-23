//! Actual managed cuDNN rectangular cross-attention, separately qualified from
//! the legacy IEEE provider. No test passes merely because a GPU is absent.
use super::*;
use ferrum_kernels::backend::cuda::vnext_ops::{
    cuda_vnext_tensor_providers_with_cudnn_cross_attention,
    CUDA_CROSS_ATTENTION_FUSED_BF16_NUMERICAL_PROFILE_ID,
};
use ferrum_kernels::backend::cuda::{CudaCudnnConfig, CudnnSdpaPlanPolicy};
#[path = "cross_family.rs"]
mod cross_family;
#[path = "cross_oracle.rs"]
mod oracle;
use cross_family::{CrossConfig, CrossFamily};
use oracle::{Expected, Probe, BATCH, COLUMNS, KEYS, QUERIES};
const PROFILE: &str = CUDA_CROSS_ATTENTION_FUSED_BF16_NUMERICAL_PROFILE_ID;
const WORKSPACE_LIMIT: u64 = 64 << 20;

fn vendor_config() -> (CudaCudnnConfig, CudnnSdpaPlanPolicy) {
    let library_path = std::env::var_os("FERRUM_TEST_CUDNN_LIBRARY_PATH")
        .map(std::path::PathBuf::from)
        .expect(
            "set FERRUM_TEST_CUDNN_LIBRARY_PATH to an explicit absolute cuDNN 9.20 library path",
        );
    assert!(library_path.is_absolute());
    let dependency_libraries = std::env::var_os("FERRUM_TEST_CUDNN_DEPENDENCY_LIBRARIES")
        .map(|paths| std::env::split_paths(&paths).collect())
        .unwrap_or_default();
    (
        CudaCudnnConfig {
            library_path,
            dependency_libraries,
            opaque_memory_bytes: 64 << 20,
            workspace_limit_bytes: WORKSPACE_LIMIT,
            opaque_memory_profile: "experimental-cross-attention-test-v1".into(),
        },
        CudnnSdpaPlanPolicy {
            maximum_plans: 16,
            maximum_live_bindings: 64,
            opaque_memory_bytes: 128 << 20,
            workspace_limit_bytes: WORKSPACE_LIMIT,
        },
    )
}
fn bits(bytes: &ManagedHostBytes) -> Vec<u16> {
    bytes
        .bytes()
        .chunks_exact(2)
        .map(|b| u16::from_le_bytes([b[0], b[1]]))
        .collect()
}

fn all_valid_probe() -> Probe {
    let mut probe = Probe::new(false);
    probe.valid = [[true; KEYS]; BATCH];
    for (name, values) in [
        ("key_run_counts", vec![1; BATCH]),
        ("key_run_lengths", vec![KEYS as i64; BATCH]),
        ("key_run_valid", vec![1; BATCH]),
    ] {
        probe
            .config
            .attributes
            .insert(id(name), SemanticValue::Integers(values));
    }
    // Every physical row contributes. Distinct batch/head/key values expose a
    // mistaken Q-length KV stride or binding both batches to the first batch.
    // Q already varies across batch, query and head in the shared probe.
    probe.input[1].fill(0);
    for b in 0..BATCH {
        for k in 0..KEYS {
            for h in 0..oracle::HEADS {
                let start = (b * KEYS + k) * COLUMNS + h * oracle::DIM;
                probe.input[1][start] =
                    bf16::from_f32((k as f32 - 2.) / 4. + b as f32 / 8. + h as f32 / 16.).to_bits();
                for c in 0..oracle::DIM {
                    probe.input[2][start + c] = bf16::from_f32(
                        b as f32 * 2. + k as f32 / 2. - 1. + h as f32 / 4. + (c % 7) as f32 / 16.,
                    )
                    .to_bits();
                }
            }
        }
    }
    probe
}

fn run_case(name: &str, probe: Probe, expected: Expected) {
    let (vendor, sdpa) = vendor_config();
    let runtime = Arc::new(
        CudaDeviceRuntime::new_tensor_eager(CudaTensorRuntimeConfig {
            device: cuda_tensor_runtime_config(0, id("device.cuda.0")).unwrap(),
            usable_memory_bytes: USABLE_BYTES,
            bootstrap_memory_bytes: 1 << 30,
            stream_memory_bytes: 64 << 20,
            maximum_streams: 1,
            maximum_in_flight_per_stream: 1,
            maximum_commands_per_submission: 64,
            opaque_memory_profile: "experimental-cross-attention-test-v1".into(),
        })
        .unwrap(),
    );
    let registry = OperationRuntimeRegistry::new(
        vec![
            Box::new(conv2d_bf16_contract().unwrap()) as Box<dyn OperationContract>,
            Box::new(structured_attention_fused_bf16_contract().unwrap()),
            Box::new(cross_attention_fused_bf16_contract().unwrap()),
        ],
        cuda_vnext_tensor_providers_with_cudnn_cross_attention(&runtime, &vendor, sdpa, &[])
            .unwrap(),
    )
    .unwrap();
    let descriptor = runtime.descriptor();
    let engine = EngineProviderDescriptor::new(
        id("provider.engine.cuda.tensor.cross_attention.fixture"),
        ContractVersion::new(1, 0),
        descriptor.runtime_implementation_fingerprint.clone(),
        descriptor.id.clone(),
        descriptor.capabilities.clone(),
    )
    .unwrap();
    let catalog = registry
        .capability_catalog(descriptor.clone(), vec![engine])
        .unwrap();
    let family = TypedFamilyRegistration::new(CrossFamily)
        .prepare_with_profile(
            &serde_json::to_value(&probe.config).unwrap(),
            &NumericalProfileId::new(PROFILE).unwrap(),
        )
        .unwrap();
    let mut options = ProgramPlanCompileOptions::new(
        INPUTS
            .into_iter()
            .enumerate()
            .map(|(i, name)| (id(name), probe.config.spec(i == 0)))
            .collect(),
    )
    .unwrap();
    options.retain_completion_value(id(OUTPUT_ID));
    let compiled = ProgramPlanCompiler::compile(
        &family,
        &catalog,
        &policy(runtime.as_ref()),
        &registry.planning(),
        &options,
    )
    .unwrap();
    let executable = compiled.executable();
    let plan = executable.execution_plan();
    assert!(plan.payload().memory().static_allocations().is_empty());
    assert_eq!(plan.payload().nodes().len(), 1);
    let node = &plan.payload().nodes()[0];
    assert_eq!(
        node.operation_id().as_str(),
        CROSS_ATTENTION_FUSED_BF16_OPERATION_ID
    );
    let valid_lengths = probe
        .valid
        .map(|keys| keys.into_iter().filter(|&v| v).count());
    let batches = valid_lengths.iter().filter(|&&n| n != 0).count();
    let packed_rows = valid_lengths.into_iter().max().unwrap();
    // This fixture's BF16 row size is 256-byte aligned. The direct all-valid
    // path deliberately retains the same conservative pack-space admission.
    let packing_minimum = (2 * batches * packed_rows * COLUMNS * 2) as u64 + 512;
    let scratch = node
        .provider_resources()
        .scratch()
        .unwrap()
        .fixed_bytes()
        .unwrap();
    assert!(scratch >= packing_minimum && scratch <= packing_minimum + WORKSPACE_LIMIT);
    assert!(node.scratch_resource().is_some());
    eprintln!("cross {name}: profile={PROFILE}, provider={}, prepared={:?}, admitted scratch={scratch} (packing={packing_minimum}); native opaque limits are declarations, not measurements",
        node.provider_implementation_fingerprint(),node.provider_resources().prepared_implementation_fingerprint());
    let resources = no_static(runtime.clone(), plan);
    let input_bytes = (BATCH * QUERIES + 2 * BATCH * KEYS) * COLUMNS * 2;
    let output_bytes = BATCH * QUERIES * COLUMNS * 2;
    let host = HostMemoryAccount::new(
        HostMemoryLimits::new(
            input_bytes + output_bytes,
            HostMemoryClassLimits {
                input_preprocess_bytes: input_bytes,
                output_bytes,
                transfer_staging_bytes: 0,
                weight_backing_bytes: 0,
            },
        )
        .unwrap(),
    );
    let mut executor = PreparedTensorExecutor::new(
        executable.clone(),
        registry.bind_plan(executable).unwrap(),
        resources.clone(),
        host.clone(),
        TensorExecutorConfig::default(),
    )
    .unwrap();
    let sanitized = probe.sanitized();
    let mut previous = None;
    let mut retained = None;
    for (iteration, input) in [&probe.input, &sanitized].into_iter().enumerate() {
        let tensors = INPUTS
            .into_iter()
            .enumerate()
            .map(|(i, name)| {
                super::super::normalization::managed_input(
                    &host,
                    name,
                    probe.config.spec(i == 0).dimensions,
                    &input[i],
                )
            })
            .collect();
        let outputs = executor
            .execute(
                id(format!("run.cuda.cross.{name}.{iteration}")),
                id(format!("request.cuda.cross.{name}.{iteration}")),
                tensors,
            )
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(
            outputs[0].tensor().dimensions(),
            [probe.config.query_rows, probe.config.columns]
        );
        assert!(host.owns(outputs[0].bytes()));
        let actual = bits(outputs[0].bytes());
        let maximum = expected.check(&actual).unwrap();
        eprintln!("cross {name} execution {iteration}: {} elements, max_abs_error={maximum}, tolerance={}+{}*abs(F64 reference)",actual.len(),oracle::ABSOLUTE,oracle::RELATIVE);
        if let Some(previous) = &previous {
            assert_eq!(
                &actual, previous,
                "masked/empty payload must not affect outputs"
            );
        }
        previous = Some(actual);
        let output = outputs[0].bytes().clone();
        drop(outputs);
        assert_eq!(host.usage().unwrap().committed_bytes(), output_bytes);
        if iteration == 1 {
            retained = Some(output);
        } else {
            drop(output);
        }
    }
    drop((executor, compiled, registry));
    close_resources(resources);
    let retained = retained.unwrap();
    assert_eq!(host.usage().unwrap().committed_bytes(), output_bytes);
    expected.check(&bits(&retained)).unwrap();
    drop(retained);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    assert_eq!(host.usage().unwrap().allocated_bytes(), 0);
    // Successful retirement releases dynamic bindings/module permit; this is
    // not native error-injection or a claim that CUDA's process RSS is zero.
    drop(
        cuda_vnext_tensor_providers_with_cudnn_cross_attention(&runtime, &vendor, sdpa, &[])
            .unwrap(),
    );
}

#[test]
#[ignore = "requires exclusive CUDA and explicit FERRUM_TEST_CUDNN_LIBRARY_PATH for cuDNN 9.20"]
fn managed_cuda_cross_attention_rectangular_batches_holes_and_empty_retire() {
    let nonempty = Probe::new(false);
    let mixed = Probe::new(true);
    let all_valid = all_valid_probe();
    let expected_nonempty = Expected::prepare(&nonempty);
    let expected_mixed = Expected::prepare(&mixed);
    let expected_all_valid = Expected::prepare(&all_valid);
    run_case("both_nonempty", nonempty, expected_nonempty);
    run_case("mixed_empty", mixed, expected_mixed);
    run_case("all_valid_direct", all_valid, expected_all_valid);
}

#[test]
fn cross_fixture_prepares_exact_lengths_masks_and_independent_numeric_comparison() {
    for probe in [Probe::new(false), Probe::new(true), all_valid_probe()] {
        let geometry = probe.config.geometry().unwrap();
        assert_eq!(
            (
                geometry.batch(),
                geometry.query_length(),
                geometry.key_value_length(),
                geometry.heads()
            ),
            (2, 3, 5, 2)
        );
        for b in 0..BATCH {
            for k in 0..KEYS {
                assert_eq!(geometry.key_is_valid(b, k as u32), Some(probe.valid[b][k]));
            }
        }
        TypedFamilyRegistration::new(CrossFamily)
            .prepare_with_profile(
                &serde_json::to_value(&probe.config).unwrap(),
                &NumericalProfileId::new(PROFILE).unwrap(),
            )
            .unwrap();
        let expected = Expected::prepare(&probe);
        let rounded: Vec<_> = expected
            .values
            .iter()
            .map(|&x| bf16::from_f64(x).to_bits())
            .collect();
        expected.check(&rounded).unwrap();
        let mut corrupted = rounded;
        corrupted[0] = 0x7f80;
        assert!(expected.check(&corrupted).is_err());
        corrupted[0] = bf16::from_f32(8.).to_bits();
        assert!(expected.check(&corrupted).is_err());
    }

    // Counterfactual outputs establish that the all-valid fixture distinguishes
    // both the batch pointer and the rectangular KV sequence length. They are
    // independent F64 calculations, never copied from the candidate provider.
    let expected = Expected::prepare(&all_valid_probe());
    let mut swapped_batches: Vec<_> = expected
        .values
        .iter()
        .map(|&x| bf16::from_f64(x).to_bits())
        .collect();
    swapped_batches.rotate_left(QUERIES * COLUMNS);
    assert!(expected.check(&swapped_batches).is_err());

    let mut truncated_keys = all_valid_probe();
    for keys in &mut truncated_keys.valid {
        keys[QUERIES..].fill(false);
    }
    let truncated: Vec<_> = Expected::prepare(&truncated_keys)
        .values
        .iter()
        .map(|&x| bf16::from_f64(x).to_bits())
        .collect();
    assert!(expected.check(&truncated).is_err());
}
