//! Real Metal execution through compiled fixed-tensor plans and managed memory.
#![cfg(all(feature = "metal", target_os = "macos"))]

use ferrum_execution::{PreparedTensorExecutor, TensorExecutorConfig};
use ferrum_kernels::backend::metal::{
    vnext_ops::tensor::{
        metal_tensor_providers, metal_tensor_runtime_config, MetalTensorOperation,
    },
    vnext_runtime::{MetalDeviceRuntime, MetalTensorRuntimeConfig},
};
use ferrum_runtime::*;
use half::bf16;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, OnceLock};

#[path = "vnext_tensor_metal_runtime/attention.rs"]
mod attention;
#[path = "vnext_tensor_metal_runtime/family.rs"]
mod family;
#[path = "vnext_tensor_metal_runtime/linear.rs"]
mod linear;
#[path = "vnext_tensor_metal_runtime/primitives.rs"]
mod primitives;
#[path = "vnext_tensor_metal_runtime/qwen_attention.rs"]
mod qwen_attention;
#[path = "vnext_tensor_metal_runtime/qwen_graph.rs"]
mod qwen_graph;
#[path = "vnext_tensor_metal_runtime/qwen_qk.rs"]
mod qwen_qk;
#[path = "vnext_tensor_metal_runtime/scatter.rs"]
mod scatter;
#[path = "vnext_tensor_metal_runtime/spatial.rs"]
mod spatial;
#[path = "vnext_tensor_metal_runtime/transforms.rs"]
mod transforms;
#[path = "vnext_tensor_metal_runtime/transport.rs"]
mod transport;
use family::{Config, Family, TensorPolicy};

const OUTPUT: &str = "value.metal.tensor.output";
const CAPACITY: u64 = 64 << 20;

fn id<T: TryFrom<String>>(value: impl Into<String>) -> T
where
    T::Error: std::fmt::Debug,
{
    T::try_from(value.into()).unwrap()
}
fn invalid(reason: impl Into<String>) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: reason.into(),
    }
}
fn input_id(index: usize) -> ProgramValueId {
    id(format!("value.metal.tensor.input.{index}"))
}
fn rounds(x: f32) -> f32 {
    bf16::from_f32(x).to_f32()
}

// Independent CPU arithmetic; exact BF16 comparisons for algebraic operations,
// and one BF16 ULP for direct transcendental results. GELU propagates the
// documented FP32 tanh bound because cancellation near -1 amplifies its error.
// Non-finite classes are compared separately.
fn expected(kind: MetalTensorOperation, x: f32, p: f32, u: f32) -> u16 {
    use MetalTensorOperation::*;
    bf16::from_f32(match kind {
        Silu => x / (1.0 + libm::expf(-x)),
        SiluMul => rounds(x / (1.0 + libm::expf(-x))) * p,
        Tanh => libm::tanhf(x),
        ModulationScale => x * rounds(1.0 + p),
        GatedResidual => x + rounds(p * u),
        GeluTanh => (0.5 * x) * (1.0 + libm::tanhf(0.797_884_6 * (x + 0.044_715 * ((x * x) * x)))),
        PlainResidualAdd => x + p,
        _ => unreachable!("other operations have their own oracle"),
    })
    .to_bits()
}

fn gelu_interval(x: f32) -> (f32, f32) {
    // Apple MSL 4.1 section 8.4, table 8.1 permits <= 5 FP32 ULP for tanh:
    // https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf
    // Use an independent f64 reference rounded to f32; six adjacent values
    // conservatively cover both the documented error and reference rounding.
    // Propagate through the contract's staged arithmetic, without rewriting
    // GELU into a stable sigmoid (which would change the operation contract).
    let inner = 0.797_884_6 * (x + 0.044_715 * ((x * x) * x));
    let t = libm::tanh(inner as f64) as f32;
    let (mut lo, mut hi) = (t, t);
    for _ in 0..6 {
        lo = lo.next_down();
        hi = hi.next_up();
    }
    let a = rounds((0.5 * x) * (1.0 + lo.max(-1.0)));
    let b = rounds((0.5 * x) * (1.0 + hi.min(1.0)));
    (a.min(b), a.max(b))
}

fn inputs(config: &Config) -> Vec<Vec<f32>> {
    (0..config.input_count())
        .map(|port| {
            let count = config.dimensions(port).iter().product::<u64>() as usize;
            (0..count)
                .map(|i| {
                    if port == 0 {
                        match i {
                            0 => return -0.0,
                            1 => return 0.0,
                            2 => return f32::INFINITY,
                            3 => return f32::NEG_INFINITY,
                            4 => return f32::NAN,
                            _ => (),
                        }
                    }
                    rounds(((i * 113 + port * 61) % 1021) as f32 / 64.0 - 8.0)
                })
                .collect()
        })
        .collect()
}

fn upload(
    host: &HostMemoryAccount,
    config: &Config,
    values: &[Vec<f32>],
) -> Vec<ManagedHostTensor> {
    values
        .iter()
        .enumerate()
        .map(|(port, values)| {
            let mut bytes = host
                .allocate_zeroed(HostMemoryClass::InputPreprocess, values.len() * 2)
                .unwrap();
            for (dst, value) in bytes.as_mut().chunks_exact_mut(2).zip(values) {
                dst.copy_from_slice(&bf16::from_f32(*value).to_bits().to_le_bytes());
            }
            ManagedHostTensor::new(
                input_id(port),
                ResolvedTensorSpec::new(
                    config.dimensions(port),
                    ElementType::Bf16,
                    ResolvedTensorLayout::Contiguous,
                )
                .unwrap(),
                bytes.freeze(),
            )
            .unwrap()
        })
        .collect()
}

fn verify(config: &Config, values: &[Vec<f32>], bytes: &[u8]) {
    assert_eq!(bytes.len(), (config.rows * config.width * 2) as usize);
    for (i, bytes) in bytes.chunks_exact(2).enumerate() {
        let parameter_index = if config.row_broadcast {
            i % config.width as usize
        } else {
            i
        };
        let want = expected(
            config.kind(),
            values[0][i],
            values.get(1).map_or(0.0, |p| p[parameter_index]),
            values.get(2).map_or(0.0, |u| u[i]),
        );
        let got = u16::from_le_bytes([bytes[0], bytes[1]]);
        if config.kind() == MetalTensorOperation::GeluTanh
            && values[0][i].is_finite()
            && values[0][i] != 0.0
        {
            let (lo, hi) = gelu_interval(values[0][i]);
            let actual = bf16::from_bits(got).to_f32();
            assert!(
                actual >= lo && actual <= hi,
                "GELU element {i}: {actual} outside propagated Metal tanh bound [{lo}, {hi}]"
            );
            continue;
        }
        if bf16::from_bits(want).is_nan() {
            assert!(
                bf16::from_bits(got).is_nan(),
                "{:?} element {i}",
                config.kind()
            );
        } else if !bf16::from_bits(want).is_finite() || want & 0x7fff == 0 {
            assert_eq!(got, want, "{:?} special element {i}", config.kind());
        } else {
            let tolerance = match config.kind() {
                MetalTensorOperation::ModulationScale
                | MetalTensorOperation::GatedResidual
                | MetalTensorOperation::PlainResidualAdd => 0,
                _ => 1,
            };
            assert!(
                got.abs_diff(want) <= tolerance,
                "{:?} row={} element {i}: {got:#06x} != {want:#06x}",
                config.kind(),
                config.row_broadcast
            );
        }
    }
}

#[test]
fn managed_metal_tensor_operations_match_independent_arithmetic_and_retire_resources() {
    let runtime = Arc::new(
        MetalDeviceRuntime::new_tensor_eager(MetalTensorRuntimeConfig {
            device: metal_tensor_runtime_config(id("device.metal.0")).unwrap(),
            usable_memory_bytes: CAPACITY,
            bootstrap_memory_bytes: 16 << 20,
            stream_memory_bytes: 1 << 20,
            maximum_streams: 2,
            maximum_in_flight_per_stream: 1,
            maximum_commands_per_submission: qwen_graph::maximum_command_count(),
            opaque_memory_profile: "metal.pointwise.test.declared.v1".into(),
        })
        .unwrap(),
    );
    let selected: Vec<_> = MetalTensorOperation::POINTWISE
        .into_iter()
        .chain([
            MetalTensorOperation::DenseLinear,
            MetalTensorOperation::BlockLinear,
        ])
        .chain(transforms::OPERATIONS)
        .collect();
    let registry = OperationRuntimeRegistry::new(
        selected
            .iter()
            .copied()
            .map(|kind| Box::new(kind.contract().unwrap()) as Box<dyn OperationContract>)
            .collect(),
        metal_tensor_providers(&runtime, &selected).unwrap(),
    )
    .unwrap();
    assert!(metal_tensor_providers(&runtime, &MetalTensorOperation::POINTWISE).is_err());
    let engine = EngineProviderDescriptor::new(
        id("provider.engine.metal.tensor.test"),
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
    let host = runtime
        .tensor_host_account(
            HostMemoryLimits::new(
                3 << 20,
                HostMemoryClassLimits {
                    input_preprocess_bytes: 1 << 20,
                    // Retain both executions of the nine-output Qwen branch,
                    // including page-rounded no-copy backing for every output.
                    output_bytes: 2 << 20,
                    transfer_staging_bytes: 0,
                    weight_backing_bytes: 0,
                },
            )
            .unwrap(),
        )
        .unwrap();
    let policy = TensorPolicy {
        capacity_bytes: CAPACITY,
        storage_profiles: runtime
            .descriptor()
            .dynamic_storage_profiles
            .iter()
            .copied()
            .collect(),
    };
    for kind in MetalTensorOperation::POINTWISE {
        let modes: &[bool] = if matches!(
            kind,
            MetalTensorOperation::ModulationScale | MetalTensorOperation::GatedResidual
        ) {
            &[false, true]
        } else {
            &[false]
        };
        for &row_broadcast in modes {
            for (rows, width) in [(3, 17), (2, 257)] {
                let config = Config::new(kind, rows, width, row_broadcast);
                let prepared = TypedFamilyRegistration::new(Family)
                    .prepare_with_profile(
                        &serde_json::to_value(&config).unwrap(),
                        &id(kind.profile()),
                    )
                    .unwrap();
                let mut options = ProgramPlanCompileOptions::new(config.input_specs()).unwrap();
                options.retain_completion_value(id(OUTPUT));
                let compiled = ProgramPlanCompiler::compile(
                    &prepared,
                    &catalog,
                    &policy,
                    &registry.planning(),
                    &options,
                )
                .unwrap();
                let executable = compiled.executable();
                let resources = match executable
                    .execution_plan()
                    .provision_static(runtime.clone(), id("request.metal.pointwise.provision"))
                    .unwrap()
                    .into_provisioning()
                {
                    StaticProvisioning::NoStatic(provisioned) => provisioned.into_plan_runtime(),
                    _ => panic!("pointwise fixture has no weights"),
                };
                let mut executor = PreparedTensorExecutor::new(
                    executable.clone(),
                    registry.bind_plan(executable).unwrap(),
                    resources.clone(),
                    host.clone(),
                    TensorExecutorConfig::default(),
                )
                .unwrap();
                let values = inputs(&config);
                // Both executions must retire the one-flight capacity correctly.
                for repeat in 0..2 {
                    let output = executor
                        .execute(
                            id(format!("run.metal.pointwise.{repeat}")),
                            id(format!("request.metal.pointwise.{repeat}")),
                            upload(&host, &config, &values),
                        )
                        .unwrap();
                    assert_eq!(output.len(), 1);
                    verify(&config, &values, output[0].bytes().bytes());
                    drop(output);
                    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
                }
                drop((executor, compiled));
                assert!(matches!(
                    PlanRuntimeResources::close(resources),
                    Ok(PlanRuntimeCloseOutcome::Closed(_))
                ));
            }
        }
    }
    concurrent_executors(&runtime, &registry, &catalog, &policy, &host);
    linear::verify(&runtime, &registry, &catalog, &policy, &host);
    transforms::verify(&runtime, &registry, &catalog, &host);
    attention::verify(&runtime, &registry, &catalog, &host);
    transport::verify(&runtime, &registry, &catalog, &host);
    scatter::verify(&runtime, &registry, &catalog, &host);
    spatial::verify(&runtime, &registry, &catalog, &host);
    primitives::verify(&runtime, &registry, &catalog, &host);
    qwen_qk::verify(&runtime, &registry, &catalog, &host);
    qwen_attention::verify(&runtime, &registry, &catalog, &host);
    qwen_graph::verify(&runtime, &registry, &catalog);
    drop(registry);
    // Pipeline ownership and queue reservations are reusable after retirement.
    drop(metal_tensor_providers(&runtime, &MetalTensorOperation::POINTWISE).unwrap());
    let first = runtime.create_stream().unwrap();
    let second = runtime.create_stream().unwrap();
    assert!(runtime.create_stream().is_err());
    drop((first, second));
}

fn concurrent_executors(
    runtime: &Arc<MetalDeviceRuntime>,
    registry: &OperationRuntimeRegistry<MetalDeviceRuntime>,
    catalog: &CapabilityCatalog,
    policy: &TensorPolicy,
    host: &HostMemoryAccount,
) {
    let kind = MetalTensorOperation::GatedResidual;
    let config = Config::new(kind, 7, 257, true);
    let prepared = TypedFamilyRegistration::new(Family)
        .prepare_with_profile(&serde_json::to_value(&config).unwrap(), &id(kind.profile()))
        .unwrap();
    let mut options = ProgramPlanCompileOptions::new(config.input_specs()).unwrap();
    options.retain_completion_value(id(OUTPUT));
    let compiled =
        ProgramPlanCompiler::compile(&prepared, catalog, policy, &registry.planning(), &options)
            .unwrap();
    let executable = compiled.executable();
    let mut executors = Vec::new();
    let mut roots = Vec::new();
    // Each fixed-tensor root admits one job. Independent roots share the same
    // physical device account; they do not increase its total capacity ceiling.
    for worker in 0..2 {
        let resources = match executable
            .execution_plan()
            .provision_static(
                runtime.clone(),
                id(format!("request.metal.concurrent.provision.{worker}")),
            )
            .unwrap()
            .into_provisioning()
        {
            StaticProvisioning::NoStatic(provisioned) => provisioned.into_plan_runtime(),
            _ => panic!("no static weights"),
        };
        executors.push(
            PreparedTensorExecutor::new(
                executable.clone(),
                registry.bind_plan(executable).unwrap(),
                resources.clone(),
                host.clone(),
                TensorExecutorConfig::default(),
            )
            .unwrap(),
        );
        roots.push(resources);
    }
    let barrier = std::sync::Barrier::new(executors.len());
    std::thread::scope(|scope| {
        for (worker, mut executor) in executors.into_iter().enumerate() {
            let config = &config;
            let barrier = &barrier;
            scope.spawn(move || {
                barrier.wait();
                let values = inputs(config);
                for repeat in 0..3 {
                    let input = upload(host, config, &values);
                    let output = executor
                        .execute(
                            id(format!("run.metal.concurrent.{worker}.{repeat}")),
                            id(format!("request.metal.concurrent.{worker}.{repeat}")),
                            input,
                        )
                        .unwrap();
                    verify(config, &values, output[0].bytes().bytes());
                }
            });
        }
    });
    drop(compiled);
    for resources in roots {
        assert!(matches!(
            PlanRuntimeResources::close(resources),
            Ok(PlanRuntimeCloseOutcome::Closed(_))
        ));
    }
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}
