//! Concrete model/backend entrypoints for the generic execution assertions.
use ferrum_execution::driver::ProgramExecutor;
use ferrum_execution::test_support::{sampling as assertions, token_execution as token_assertions};
use ferrum_kernels::backend::cpu::vnext_ops::CpuVNextComposition;
use ferrum_runtime::*;
use ferrum_types::{
    AttentionExecutionPolicy, Device, EngineConfig, ReusableExecutionPreparationMode,
    SamplingReadbackMode,
};
use std::sync::Arc;

pub(in crate::executor::driver_contract_tests) mod fixture;
mod lifecycle;
mod startup;
mod token_execution;
use fixture::{Fixture, FixtureOptions};

fn cpu(
    mode: SamplingReadbackMode,
) -> Fixture<ferrum_kernels::backend::cpu::vnext_runtime::CpuDeviceRuntime> {
    cpu_with_nonfinite(mode, None)
}

fn cpu_with_nonfinite(
    mode: SamplingReadbackMode,
    nonfinite: Option<usize>,
) -> Fixture<ferrum_kernels::backend::cpu::vnext_runtime::CpuDeviceRuntime> {
    cpu_with_precision(mode, nonfinite, ElementType::F32)
}

fn cpu_with_precision(
    mode: SamplingReadbackMode,
    nonfinite: Option<usize>,
    logits_type: ElementType,
) -> Fixture<ferrum_kernels::backend::cpu::vnext_runtime::CpuDeviceRuntime> {
    let (runtime, registry, materializers, _, catalog) = CpuVNextComposition::create(
        DeviceId::new("device.fixture.sampling-readback").unwrap(),
        256 << 20,
    )
    .unwrap()
    .into_parts()
    .unwrap();
    Fixture::new(
        mode,
        Device::CPU,
        runtime,
        registry,
        materializers,
        catalog,
        FixtureOptions {
            logits_type,
            nonfinite_token: nonfinite,
            ..FixtureOptions::default()
        },
    )
}

async fn compare<R: DeviceRuntime>(uniform: Fixture<R>, participant: Fixture<R>) {
    assertions::assert_sampling_modes_equivalent(&uniform.executor, &participant.executor).await;
}
async fn compare_no_finite<R: DeviceRuntime>(uniform: Fixture<R>, participant: Fixture<R>) {
    assertions::assert_no_finite_failure_is_request_local(&uniform.executor, &participant.executor)
        .await;
}

#[tokio::test]
async fn lora_token_admission_rejects_before_any_native_submission() {
    let fixture = cpu(SamplingReadbackMode::Uniform);
    assertions::lora_token_admission_rejects_before_any_native_submission(&fixture.executor).await;
}

#[tokio::test]
async fn lora_rejected_admission_cannot_rebind_live_decode_incarnation() {
    let fixture = cpu(SamplingReadbackMode::Uniform);
    assertions::lora_rejected_admission_cannot_rebind_live_decode_incarnation(&fixture.executor)
        .await;
}

#[tokio::test]
async fn sampling_readback_cpu_terminal_preserves_mixed_outputs_masks_order_and_bytes() {
    compare(
        cpu(SamplingReadbackMode::Uniform),
        cpu(SamplingReadbackMode::Participant),
    )
    .await;
}

#[tokio::test]
async fn sampling_readback_cpu_f16_terminal_preserves_mixed_outputs_signed_zero_and_order() {
    compare(
        cpu_with_precision(SamplingReadbackMode::Uniform, None, ElementType::F16),
        cpu_with_precision(SamplingReadbackMode::Participant, None, ElementType::F16),
    )
    .await;
}

#[tokio::test]
async fn sampling_readback_cpu_terminal_keeps_nonfinite_failure_request_local() {
    compare_no_finite(
        cpu_with_nonfinite(SamplingReadbackMode::Uniform, Some(3)),
        cpu_with_nonfinite(SamplingReadbackMode::Participant, Some(3)),
    )
    .await;
}

#[tokio::test]
async fn sampling_readback_cpu_f16_terminal_keeps_nonfinite_failure_request_local() {
    compare_no_finite(
        cpu_with_precision(SamplingReadbackMode::Uniform, Some(3), ElementType::F16),
        cpu_with_precision(SamplingReadbackMode::Participant, Some(3), ElementType::F16),
    )
    .await;
}

#[cfg(feature = "cuda")]
#[tokio::test]
#[ignore = "requires exclusive CUDA and real F32 language providers"]
async fn sampling_readback_cuda_terminal_preserves_mixed_outputs_masks_order_and_bytes() {
    use ferrum_kernels::backend::cuda::vnext_ops::CudaVNextComposition;
    let make = |mode, nonfinite| {
        let (runtime, registry, materializers, catalog) = CudaVNextComposition::create(
            0,
            DeviceId::new("device.fixture.sampling-readback").unwrap(),
            AttentionExecutionPolicy::Portable,
        )
        .unwrap()
        .into_parts();
        Fixture::new(
            mode,
            Device::CUDA(0),
            runtime,
            registry,
            materializers,
            catalog,
            FixtureOptions {
                logits_type: ElementType::F32,
                nonfinite_token: nonfinite,
                ..FixtureOptions::default()
            },
        )
    };
    compare(
        make(SamplingReadbackMode::Uniform, None),
        make(SamplingReadbackMode::Participant, None),
    )
    .await;
    compare_no_finite(
        make(SamplingReadbackMode::Uniform, Some(3)),
        make(SamplingReadbackMode::Participant, Some(3)),
    )
    .await;
}

#[cfg(feature = "cuda")]
#[tokio::test]
#[ignore = "requires exclusive CUDA and real F16 language providers"]
async fn sampling_readback_cuda_f16_terminal_preserves_mixed_outputs_and_failure_scope() {
    use ferrum_kernels::backend::cuda::vnext_ops::CudaVNextComposition;
    let make = |mode, nonfinite| {
        let (runtime, registry, materializers, catalog) = CudaVNextComposition::create(
            0,
            DeviceId::new("device.fixture.sampling-readback-f16").unwrap(),
            AttentionExecutionPolicy::Portable,
        )
        .unwrap()
        .into_parts();
        Fixture::new(
            mode,
            Device::CUDA(0),
            runtime,
            registry,
            materializers,
            catalog,
            FixtureOptions {
                logits_type: ElementType::F16,
                nonfinite_token: nonfinite,
                ..FixtureOptions::default()
            },
        )
    };
    compare(
        make(SamplingReadbackMode::Uniform, None),
        make(SamplingReadbackMode::Participant, None),
    )
    .await;
    compare_no_finite(
        make(SamplingReadbackMode::Uniform, Some(3)),
        make(SamplingReadbackMode::Participant, Some(3)),
    )
    .await;
}
