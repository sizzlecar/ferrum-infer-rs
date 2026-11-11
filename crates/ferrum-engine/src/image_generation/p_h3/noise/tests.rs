use super::*;
fn host() -> HostMemoryAccount {
    HostMemoryAccount::with_allocation_alignment(
        HostMemoryLimits::new(
            8192,
            HostMemoryClassLimits {
                input_preprocess_bytes: 4096,
                output_bytes: 4096,
                transfer_staging_bytes: 0,
                weight_backing_bytes: 0,
            },
        )
        .unwrap(),
        64,
    )
    .unwrap()
}
fn spec(shape: &[u64]) -> ResolvedTensorSpec {
    ResolvedTensorSpec::new(
        shape.to_vec(),
        ElementType::F32,
        ResolvedTensorLayout::Contiguous,
    )
    .unwrap()
}
fn id(name: &str) -> ProgramValueId {
    ProgramValueId::new(name).unwrap()
}
fn layout() -> JointNoiseLayout {
    JointNoiseLayout::new(&spec(&[1, 2, 1, 2, 3]), &spec(&[1, 3, 2, 2])).unwrap()
}
fn output(host: &HostMemoryAccount) -> ManagedHostBytes {
    let mut buffer = host.allocate_zeroed(HostMemoryClass::Output, 96).unwrap();
    for (i, slot) in buffer.as_mut().chunks_exact_mut(4).enumerate() {
        // Distinct non-BF16 values exercise lossless storage and the boundary.
        slot.copy_from_slice(&(1.000123_f32 + i as f32 * 0.25).to_le_bytes());
    }
    buffer.freeze()
}
#[test]
fn joint_readback_splits_into_independent_charged_f32_sample_owners() {
    let host = host();
    let layout = layout();
    let output = output(&host);
    let scope = host
        .reserve_scope(&[(
            HostMemoryClass::InputPreprocess,
            layout.input_bytes(&host).unwrap(),
        )])
        .unwrap();
    let (image, audio) = layout
        .copy_samples(
            &scope.account(),
            &layout.config.output_tensor().unwrap(),
            &output,
            id("sample.image"),
            id("sample.audio"),
        )
        .unwrap();
    assert_eq!(image.tensor().dimensions(), [1, 2, 1, 2, 3]);
    assert_eq!(audio.tensor().dimensions(), [1, 3, 2, 2]);
    assert_eq!(image.bytes().bytes(), &output.bytes()[..48]);
    assert_eq!(audio.bytes().bytes(), &output.bytes()[48..]);
    let image_backing = image.bytes().backing_bytes();
    let audio_backing = audio.bytes().backing_bytes();
    assert_eq!(
        layout.input_bytes(&host).unwrap(),
        image_backing + audio_backing
    );
    assert_eq!(
        host.usage().unwrap().allocated_bytes(),
        output.backing_bytes() + image_backing + audio_backing
    );
    drop(output);
    assert_eq!(
        host.usage().unwrap().allocated_bytes(),
        image_backing + audio_backing
    );
    drop(image);
    assert_eq!(host.usage().unwrap().allocated_bytes(), audio_backing);
    drop(audio);
    drop(scope);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}
#[test]
fn late_audio_admission_failure_refunds_image_and_keeps_readback_for_retry() {
    let host = host();
    let layout = layout();
    let output = output(&host);
    let bytes = layout.input_bytes(&host).unwrap();
    let short = host
        .reserve_scope(&[(HostMemoryClass::InputPreprocess, bytes - 1)])
        .unwrap();
    assert!(layout
        .copy_samples(
            &short.account(),
            &layout.config.output_tensor().unwrap(),
            &output,
            id("sample.image"),
            id("sample.audio")
        )
        .is_err());
    assert_eq!(
        host.usage()
            .unwrap()
            .class_allocated_bytes(HostMemoryClass::InputPreprocess),
        0
    );
    assert_eq!(output.bytes().len(), 96);
    drop(short);
    let exact = host
        .reserve_scope(&[(HostMemoryClass::InputPreprocess, bytes)])
        .unwrap();
    let pair = layout
        .copy_samples(
            &exact.account(),
            &layout.config.output_tensor().unwrap(),
            &output,
            id("sample.image"),
            id("sample.audio"),
        )
        .unwrap();
    drop(pair);
    drop(exact);
    drop(output);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}
#[test]
fn joint_noise_rejects_wrong_shapes_foreign_owners_and_native_index_overflow() {
    let host = host();
    let layout = layout();
    let foreign_host = super::tests::host();
    let output = output(&foreign_host);
    assert!(layout
        .copy_samples(
            &host,
            &layout.config.output_tensor().unwrap(),
            &output,
            id("sample.image"),
            id("sample.audio")
        )
        .is_err());
    assert!(JointNoiseLayout::new(&spec(&[1, 2, 2, 2, 3]), &spec(&[1, 3, 2, 2])).is_err());
    assert!(JointNoiseLayout::new(&spec(&[1, 1, 1, 1 << 29]), &spec(&[1, 3, 2, 2])).is_err());
    assert_eq!(host.usage().unwrap().allocated_bytes(), 0);
}
#[test]
fn native_state_preserves_full_seed_in_actual_managed_input_without_global_rng() {
    let host = host();
    let policy = PH3ImageInitialNoisePolicy::NativeCudaPhiloxJointImageAudioV1;
    let a = managed_state(&host, policy.state(u64::MAX).unwrap()).unwrap();
    let b = managed_state(&host, policy.state(u64::MAX).unwrap()).unwrap();
    assert_eq!(a.bytes().bytes(), b.bytes().bytes());
    assert_eq!(&a.bytes().bytes()[..8], &u64::MAX.to_le_bytes());
    assert_eq!(&a.bytes().bytes()[8..], &0_u64.to_le_bytes());
    assert_eq!(
        host.usage().unwrap().allocated_bytes(),
        a.bytes().backing_bytes() + b.bytes().backing_bytes()
    );
    drop((a, b));
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}

fn cpu_layout() -> JointNoiseLayout {
    JointNoiseLayout::new(&spec(&[1, 2, 1, 4, 4]), &spec(&[1, 4, 2, 2])).unwrap()
}
fn cpu_pair(
    layout: &JointNoiseLayout,
    persistent: &HostMemoryAccount,
    work: &HostMemoryAccount,
    seed: u64,
) -> (ManagedHostTensor, ManagedHostTensor) {
    layout
        .cpu_samples(
            persistent,
            work,
            seed,
            id("sample.image"),
            id("sample.audio"),
        )
        .unwrap()
}
fn pair_bytes(pair: &(ManagedHostTensor, ManagedHostTensor)) -> [Vec<u8>; 2] {
    [
        pair.0.bytes().bytes().to_vec(),
        pair.1.bytes().bytes().to_vec(),
    ]
}

#[test]
fn cpu_initial_samples_keep_two_live_request_owners_isolated_across_cancel_and_refill() {
    // This exercises real Host ownership and production CPU preparation;
    // it does not synthesize a DiT B2 execution or a successful service reply.
    let host = host();
    let layout = cpu_layout();
    let persistent_bytes = layout.input_bytes(&host).unwrap();
    let work_bytes = layout.cpu_work_bytes(&host).unwrap();
    assert_eq!(
        work_bytes,
        host.backing_bytes(cpu_mt19937::SCRATCH_BYTES).unwrap()
    );
    let work = host
        .reserve_scope(&[(HostMemoryClass::InputPreprocess, work_bytes)])
        .unwrap();
    let reference_scope = host
        .reserve_scope(&[(HostMemoryClass::InputPreprocess, persistent_bytes)])
        .unwrap();
    let reference = cpu_pair(&layout, &reference_scope.account(), &work.account(), 43);
    let expected = pair_bytes(&reference);
    drop((reference, reference_scope));
    let a_scope = host
        .reserve_scope(&[(HostMemoryClass::InputPreprocess, persistent_bytes)])
        .unwrap();
    let b_scope = host
        .reserve_scope(&[(HostMemoryClass::InputPreprocess, persistent_bytes)])
        .unwrap();
    let a = cpu_pair(&layout, &a_scope.account(), &work.account(), 42);
    let b = cpu_pair(&layout, &b_scope.account(), &work.account(), 43);
    assert_eq!(
        pair_bytes(&b),
        expected,
        "arrival order must not consume another request's RNG"
    );
    assert_ne!(pair_bytes(&a), expected);
    assert!(host.owns(a.0.bytes()) && host.owns(b.1.bytes()));
    assert_eq!(
        host.usage().unwrap().allocated_bytes(),
        2 * persistent_bytes,
        "scratch retires after successful draws"
    );
    drop((a, a_scope)); // Cancel one initialized request while B remains alive.
    let c_scope = host
        .reserve_scope(&[(HostMemoryClass::InputPreprocess, persistent_bytes)])
        .unwrap();
    let c = cpu_pair(&layout, &c_scope.account(), &work.account(), 99);
    assert_eq!(
        pair_bytes(&b),
        expected,
        "refill must not mutate the survivor"
    );
    assert_ne!(pair_bytes(&c), expected);
    drop((b, b_scope, c, c_scope, work));
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    assert_eq!(host.usage().unwrap().allocated_bytes(), 0);
}

#[test]
fn cpu_initial_admission_refunds_partial_allocations_and_rejects_foreign_or_small_streams() {
    let host = host();
    let layout = cpu_layout();
    let persistent_bytes = layout.input_bytes(&host).unwrap();
    let work_bytes = layout.cpu_work_bytes(&host).unwrap();
    for (persistent_limit, work_limit) in [
        (persistent_bytes - 1, work_bytes),
        (persistent_bytes, work_bytes - 1),
    ] {
        let persistent = host
            .reserve_scope(&[(HostMemoryClass::InputPreprocess, persistent_limit)])
            .unwrap();
        let work = host
            .reserve_scope(&[(HostMemoryClass::InputPreprocess, work_limit)])
            .unwrap();
        assert!(layout
            .cpu_samples(
                &persistent.account(),
                &work.account(),
                42,
                id("image"),
                id("audio")
            )
            .is_err());
        assert_eq!(
            host.usage().unwrap().allocated_bytes(),
            0,
            "no half-initialized pair or scratch may escape"
        );
        drop((persistent, work));
        assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    }
    let foreign = super::tests::host();
    assert!(layout
        .cpu_samples(&host, &foreign, 42, id("image"), id("audio"))
        .is_err());
    assert_eq!(host.usage().unwrap().allocated_bytes(), 0);
    assert_eq!(foreign.usage().unwrap().allocated_bytes(), 0);
    assert!(super::tests::layout().validate_cpu().is_err());
    assert!(PH3ImageInitialNoisePolicy::CpuMt19937ImageAudioV1
        .state(42)
        .is_err());
    assert_eq!(
        PH3ImageInitialNoisePolicy::default(),
        PH3ImageInitialNoisePolicy::CpuMt19937ImageAudioV1
    );
}

#[test]
fn pipeline_defaults_to_cpu_initial_noise_and_keeps_explicit_native_policy() {
    let profile = NumericalProfileId::new("fixture.h3.edit_noise.native").unwrap();
    let mut options = super::super::PH3ImagePipelineOptions::comfy_f32(profile.clone(), 1).unwrap();
    assert_eq!(options.initial_noise, PH3ImageInitialNoisePolicy::default());
    assert_eq!(
        options.initial_noise,
        PH3ImageInitialNoisePolicy::CpuMt19937ImageAudioV1
    );
    assert_eq!(options.noise_profile, profile);
    assert_eq!(
        options.edit_noise,
        super::super::PH3ImageEditNoisePolicy::default()
    );
    options.initial_noise = PH3ImageInitialNoisePolicy::NativeCudaPhiloxJointImageAudioV1;
    assert_eq!(
        options.initial_noise.state(u64::MAX).unwrap(),
        PhiloxSeedOffset::fresh(u64::MAX)
    );
    assert_eq!(options.noise_profile, profile);
    assert_eq!(
        options.edit_noise,
        super::super::PH3ImageEditNoisePolicy::default()
    );
}
