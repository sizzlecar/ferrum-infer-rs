use super::*;
use crate::image_generation::p_h3::expansion::visual_assets::tests::fixture::{
    limits, MergedExpansionFixture,
};
use ferrum_kernels::backend::reference::{
    reference_vnext_operation_registry, ReferenceDeviceRuntime, ReferenceVNextComposition,
};
use ferrum_models::qwen_image21::image_input::ManagedRgba8Image;

fn host() -> HostMemoryAccount {
    HostMemoryAccount::with_allocation_alignment(
        HostMemoryLimits::new(
            32 << 20,
            HostMemoryClassLimits {
                input_preprocess_bytes: 16 << 20,
                output_bytes: 4 << 20,
                transfer_staging_bytes: 4 << 20,
                weight_backing_bytes: 4 << 20,
            },
        )
        .unwrap(),
        64,
    )
    .unwrap()
}
fn composition() -> (
    RuntimeComposition<ReferenceDeviceRuntime>,
    Arc<ReferenceDeviceRuntime>,
) {
    let reference = ReferenceVNextComposition::create_with_capacity(
        DeviceId::new("device.reference.upsampler.visual.producer").unwrap(),
        32 << 20,
    )
    .unwrap();
    let runtime = reference.runtime().clone();
    let composition = RuntimeComposition::new(
        runtime.clone(),
        reference_vnext_operation_registry(&runtime).unwrap(),
        WeightMaterializerRegistry::identity_only().unwrap(),
        reference.catalog().clone(),
    );
    (composition, runtime)
}
fn options() -> ExpansionVisionOptions {
    ExpansionVisionOptions {
        limits: limits(),
        profile: NumericalProfileId::new(Qwen3VLVisionHfFamily::REFERENCE_PROFILE_ID).unwrap(),
        device_capacity_bytes: 32 << 20,
        static_policy: StaticInitializationPolicy::new(1 << 16, 16).unwrap(),
        executor: TensorExecutorConfig::default(),
    }
}
fn id<T: TryFrom<String>>(s: impl Into<String>) -> T
where
    T::Error: fmt::Debug,
{
    T::try_from(s.into()).unwrap()
}
fn rgba(
    account: &HostMemoryAccount,
    fixture: &MergedExpansionFixture,
    width: u32,
    height: u32,
) -> ManagedRgba8Image {
    let mut bytes = account
        .allocate_zeroed(
            HostMemoryClass::InputPreprocess,
            width as usize * height as usize * 4,
        )
        .unwrap();
    for (i, pixel) in bytes.as_mut().chunks_exact_mut(4).enumerate() {
        pixel.copy_from_slice(&[
            (i * 31 % 256) as u8,
            (i * 13 + 23) as u8,
            (i * 7 + 101) as u8,
            0,
        ]);
    }
    ManagedRgba8Image::from_rgba8(
        account,
        bytes.freeze(),
        width,
        height,
        fixture.assets().processor().image_limits(),
    )
    .unwrap()
}

#[test]
fn actual_stored_hf_producer_uses_one_actor_returns_original_phase_after_close_and_refunds() {
    let fixture = MergedExpansionFixture::new();
    let (composition, runtime) = composition();
    let weak = Arc::downgrade(&runtime);
    let prepared =
        PreparedExpansionVision::compile_assets(&composition, fixture.assets(), 3, 2, options())
            .unwrap();
    let host = host();
    let before_compile = runtime.snapshot();
    let mut foreign_profile = options();
    foreign_profile.profile =
        NumericalProfileId::new(Qwen3VLVisionHfFamily::CUDA_PROFILE_ID).unwrap();
    assert!(PreparedExpansionVision::compile_assets(
        &composition,
        fixture.assets(),
        3,
        2,
        foreign_profile
    )
    .is_err());
    let mut foreign_limits = options();
    foreign_limits.limits.max_stored_bytes -= 1;
    assert!(PreparedExpansionVision::compile_assets(
        &composition,
        fixture.assets(),
        3,
        2,
        foreign_limits
    )
    .is_err());
    assert_eq!(runtime.snapshot().allocations, before_compile.allocations);
    let requirements = prepared.host_requirements(&host).unwrap();
    assert!(requirements.startup_device_bytes > 0 && requirements.budget.weights > 0);
    let mut budget = requirements.budget;
    let rgba_bytes = host.backing_bytes(3 * 2 * 4).unwrap();
    budget.input += rgba_bytes;
    let phase = host.reserve_scope(&budget.claims()).unwrap();
    let account = phase.account();
    let original = rgba(&account, &fixture, 3, 2);
    let mut actor = TensorSerialActor::new(runtime.clone()).unwrap();
    let request: RequestIdentity = id("request.upsampler.visual.actual");
    let completed = prepared
        .execute_on_actor(
            phase,
            &original,
            &mut actor,
            id("run.upsampler.visual.actual"),
            request.clone(),
            id("transaction.upsampler.visual.actual"),
        )
        .unwrap();
    assert_eq!(completed.request(), &request);
    assert_eq!(completed.output().tensor().element_type(), ElementType::F32);
    assert_eq!(completed.output().tensor().dimensions(), [2, 2]);
    assert_eq!(completed.output().bytes().class(), HostMemoryClass::Output);
    assert!(completed
        .output()
        .bytes()
        .bytes()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .all(f32::is_finite));
    assert!(completed
        .output()
        .bytes()
        .bytes()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .any(|value| value.abs() > 1e-6));
    assert!(!actor.is_failed());
    // Forward may leave source pixels borrowed; the product handoff drops them
    // before allocating F16 feature/table Input from the returned same phase.
    drop(original);
    let (output, phase, receipt) = completed.into_parts();
    assert!(receipt.released_static_resources() > 0);
    assert_eq!(runtime.snapshot().live_allocations, 0);
    assert!(account.owns(output.bytes()));
    let retained = output.bytes().backing_bytes();
    phase.close().unwrap();
    assert_eq!(host.usage().unwrap().committed_bytes(), retained);
    assert!(account
        .allocate_zeroed(HostMemoryClass::InputPreprocess, 1)
        .is_err());
    drop(output);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    drop(requirements);
    drop(prepared);
    drop(actor);
    drop(composition);
    drop(runtime);
    assert!(weak.upgrade().is_none());
}

#[test]
fn original_image_mismatch_rejects_before_native_and_returns_the_same_phase() {
    let fixture = MergedExpansionFixture::new();
    let (composition, runtime) = composition();
    let prepared =
        PreparedExpansionVision::compile_assets(&composition, fixture.assets(), 3, 2, options())
            .unwrap();
    let host = host();
    let requirements = prepared.host_requirements(&host).unwrap();
    let mut budget = requirements.budget;
    budget.input += host.backing_bytes(2 * 3 * 4).unwrap();
    let phase = host.reserve_scope(&budget.claims()).unwrap();
    let account = phase.account();
    let wrong = rgba(&account, &fixture, 2, 3);
    let mut actor = TensorSerialActor::new(runtime.clone()).unwrap();
    let before = runtime.snapshot();
    let failed = prepared
        .execute_on_actor(
            phase,
            &wrong,
            &mut actor,
            id("run.visual.wrong"),
            id("request.visual.wrong"),
            id("transaction.visual.wrong"),
        )
        .err()
        .expect("mismatched original dimensions must reject");
    let ExpansionVisionFailure::Preflight { source, phase } = failed else {
        panic!("shape mismatch must remain before native initialization")
    };
    assert!(source.to_string().contains("dimensions"));
    assert_eq!(runtime.snapshot().allocations, before.allocations);
    assert_eq!(runtime.snapshot().submissions, before.submissions);
    assert!(!actor.is_failed());
    drop(wrong);
    phase.close().unwrap();
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}
