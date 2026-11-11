use super::*;
use crate::p_h3_image::text_conditioning::tests::{config as conditioning_config, prompt};
use ferrum_execution::{
    component::{
        CompiledTensorComponent, ComponentInitializationConfig, ComponentWeightSource,
        TensorComponentPolicy,
    },
    TensorExecutorConfig,
};
use ferrum_kernels::backend::reference::ReferenceVNextComposition;
use ferrum_quantization::SafetensorsArchive;
use safetensors::{
    tensor::{serialize_to_file, TensorView},
    Dtype,
};

fn id<T: TryFrom<String>>(value: impl Into<String>) -> T
where
    T::Error: std::fmt::Debug,
{
    T::try_from(value.into()).unwrap()
}
fn account(bytes: usize, alignment: usize) -> HostMemoryAccount {
    HostMemoryAccount::with_allocation_alignment(
        HostMemoryLimits::new(
            bytes,
            HostMemoryClassLimits {
                input_preprocess_bytes: bytes,
                output_bytes: bytes,
                transfer_staging_bytes: bytes,
                weight_backing_bytes: bytes,
            },
        )
        .unwrap(),
        alignment,
    )
    .unwrap()
}
fn connector_archive(
    c: &PH3ImageTextConditioningConfig,
) -> (tempfile::TempDir, SafetensorsArchive) {
    let directory = tempfile::tempdir().unwrap();
    let connector = c.connector_config().unwrap();
    let input = connector.input_dim() as usize;
    let output = connector.out_channels() as usize;
    let data = [
        ("input_mean", vec![input], vec![0.25_f32; input]),
        ("input_std", vec![input], vec![2.0; input]),
        ("output_mean", vec![output], vec![0.125; output]),
        ("output_std", vec![output], vec![0.5; output]),
        ("sink_embedding", vec![output], vec![-0.75; output]),
        (
            "proj.weight",
            vec![output, input],
            (0..output * input)
                .map(|i| if i % 257 == i / input { 0.25 } else { 0.0 })
                .collect(),
        ),
    ]
    .into_iter()
    .map(|(name, shape, values)| {
        (
            format!("text_connector.{name}"),
            shape,
            values
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect::<Vec<_>>(),
        )
    })
    .collect::<Vec<_>>();
    serialize_to_file(
        data.iter().map(|(name, shape, bytes)| {
            (
                name.as_str(),
                TensorView::new(Dtype::F32, shape.clone(), bytes).unwrap(),
            )
        }),
        &None,
        &directory.path().join("model.safetensors"),
    )
    .unwrap();
    let archive = SafetensorsArchive::open(directory.path()).unwrap();
    (directory, archive)
}
fn connector_limits(c: &PH3ImageTextConditioningConfig) -> PH3ImageTextConnectorProjectionLimits {
    PH3ImageTextConnectorProjectionLimits {
        max_projection_bytes: c.projection_config().unwrap().projection_bytes().unwrap(),
    }
}
fn tap_buffers(
    host: &HostMemoryAccount,
    c: &PH3ImageTextConditioningConfig,
) -> Vec<ManagedHostBytes> {
    let text = c.encoder().text();
    let elements = (text.layout().rows() * text.architecture().hidden_size()) as usize;
    // These are actual managed Host tap owners for the budget test, not a
    // Reference implementation or a numerical claim for quantized TE.
    text.hidden_state_indices()
        .iter()
        .map(|&tap| {
            let mut buffer = host
                .allocate_zeroed(HostMemoryClass::Output, elements * 4)
                .unwrap();
            for (i, bytes) in buffer.as_mut().chunks_exact_mut(4).enumerate() {
                bytes.copy_from_slice(&(tap as f32 * 0.125 + (i % 7) as f32 * 0.25).to_le_bytes());
            }
            buffer.freeze()
        })
        .collect()
}
fn tap_views<'a>(
    c: &PH3ImageTextConditioningConfig,
    spec: &'a ResolvedTensorSpec,
    buffers: &'a [ManagedHostBytes],
) -> Vec<PH3ImageF32HiddenState<'a>> {
    c.encoder()
        .text()
        .hidden_state_indices()
        .iter()
        .zip(buffers)
        .map(|(&tap, bytes)| PH3ImageF32HiddenState::new(tap as u32, spec, bytes.bytes()))
        .collect()
}
fn tap_spec(c: &PH3ImageTextConditioningConfig) -> ResolvedTensorSpec {
    ResolvedTensorSpec::new(
        vec![
            c.encoder().text().layout().rows(),
            c.encoder().text().architecture().hidden_size(),
        ],
        ElementType::F32,
        ResolvedTensorLayout::Contiguous,
    )
    .unwrap()
}
fn assert_phase(host: &HostMemoryAccount, p: PH3ImageTextConditioningHostPhaseBudget) {
    let usage = host.usage().unwrap();
    assert_eq!(
        usage.class_allocated_bytes(HostMemoryClass::InputPreprocess),
        p.input_bytes
    );
    assert_eq!(
        usage.class_allocated_bytes(HostMemoryClass::Output),
        p.output_bytes
    );
    assert_eq!(usage.allocated_bytes(), p.total_bytes);
}

#[test]
fn query_uses_actual_per_backing_padding_without_allocating_or_granting_capacity() {
    let p = prompt();
    let c = conditioning_config(p.layout().clone());
    let logical_tap = (c.encoder().text().layout().rows()
        * c.encoder().text().architecture().hidden_size()
        * 4) as usize;
    for alignment in [1, 64, 4096] {
        let host = account(1 << 20, alignment);
        let report = c
            .host_requirements(&host, PH3ImageConditioningPrecision::F32)
            .unwrap();
        assert_eq!(host.usage().unwrap().committed_bytes(), 0);
        let input = p.managed_inputs(&c, &host).unwrap();
        let taps = tap_buffers(&host, &c);
        assert_phase(&host, report.phases[0]);
        assert_eq!(
            report.phases[0].input_bytes,
            input
                .iter()
                .map(|t| t.bytes().backing_bytes())
                .sum::<usize>()
        );
        assert_eq!(
            report.phases[0].output_bytes,
            taps.iter()
                .map(ManagedHostBytes::backing_bytes)
                .sum::<usize>()
        );
        if alignment != 1 {
            // Five independent readbacks have five separately padded owners.
            assert!(
                report.phases[0].output_bytes
                    > host.backing_bytes(logical_tap * taps.len()).unwrap()
            );
        }
        drop((input, taps));
        assert_eq!(host.usage().unwrap().committed_bytes(), 0);
        assert_eq!(
            report.scope_claims().iter().map(|(_, n)| n).sum::<usize>(),
            report.single_scope_bytes
        );
        assert!(report.single_scope_bytes >= report.live_peak_bytes);
    }
    // A metadata query does not silently grant capacity or cache an account.
    let tiny = account(1, 4096);
    let report = c
        .host_requirements(&tiny, PH3ImageConditioningPrecision::F32)
        .unwrap();
    assert!(tiny.check_scope_capacity(&report.scope_claims()).is_err());
    assert_eq!(tiny.usage().unwrap().committed_bytes(), 0);
    fn static_copy<T: Copy + Send + Sync + 'static>(_: &T) {}
    static_copy(&report);
}

#[test]
fn real_managed_standardization_stored_projection_and_finish_obey_all_four_phases() {
    let prompt = prompt();
    let c = conditioning_config(prompt.layout().clone());
    let (_directory, archive) = connector_archive(&c);
    let connector = c.connector_config().unwrap();
    let checkpoint = PH3ImageTextConnectorCheckpoint::new_with_layout(
        &connector,
        &archive,
        connector_limits(&c),
        PH3ImageTextConnectorCheckpointLayout::ComfyMergedTextEncoder,
    )
    .unwrap();
    let adapter = PH3ImageTextConnectorManagedAdapter::new(
        &connector,
        c.encoder().text().layout().batch(),
        c.encoder().text().layout().sequence_length() as usize,
    )
    .unwrap()
    .with_output_affine(crate::p_h3_image::PH3ImageTextConnectorAffineArithmetic::FusedAddcmulF32);
    let composition =
        ReferenceVNextComposition::create(id("device.reference.text_host_budget")).unwrap();
    let projection = c.projection_config().unwrap();
    let family = TypedFamilyRegistration::new(
        PH3ImageTextConnectorProjectionFamily::new(connector_limits(&c)).unwrap(),
    )
    .prepare_with_profile(
        &serde_json::to_value(&projection).unwrap(),
        &NumericalProfileId::new(PH3ImageTextConnectorProjectionFamily::REFERENCE_PROFILE_ID)
            .unwrap(),
    )
    .unwrap();
    let descriptor = composition.runtime().descriptor();
    let component = CompiledTensorComponent::compile(
        family,
        composition.registry(),
        composition.catalog(),
        &TensorComponentPolicy::new(
            descriptor.total_memory_bytes,
            descriptor
                .dynamic_storage_profiles
                .iter()
                .copied()
                .collect(),
        )
        .unwrap(),
        composition.weight_materializers(),
        &ProgramPlanCompileOptions::new(BTreeMap::from([(
            id(PH3ImageTextConnectorProjectionFamily::INPUT_ID),
            projection
                .input_spec(c.encoder().text().layout().rows())
                .unwrap(),
        )]))
        .unwrap(),
    )
    .unwrap();
    let host = account(1 << 20, 4096);
    let mut session = component
        .initialize(
            composition.runtime().clone(),
            host.clone(),
            ComponentWeightSource::Stored(&archive),
            ComponentInitializationConfig {
                run_id: id("run.text_host.init"),
                request_id: id("request.text_host.init"),
                transaction_id: id("transaction.text_host.init"),
                static_policy: StaticInitializationPolicy::new(4096, 8).unwrap(),
                executor: TensorExecutorConfig::default(),
            },
        )
        .unwrap();
    for precision in [
        PH3ImageConditioningPrecision::F32,
        PH3ImageConditioningPrecision::Bf16,
    ] {
        let report = c.host_requirements(&host, precision).unwrap();
        let scope = host.reserve_scope(&report.scope_claims()).unwrap();
        let scoped = scope.account();
        session.set_output_host_account(scoped.clone()).unwrap();
        let inputs = prompt.managed_inputs(&c, &scoped).unwrap();
        let taps = tap_buffers(&scoped, &c);
        assert_phase(&host, report.phases[0]);
        drop(inputs);
        let spec = tap_spec(&c);
        let normalized = adapter
            .managed_standardized_f32_input(&scoped, &tap_views(&c, &spec, &taps), &checkpoint)
            .unwrap();
        assert_phase(&host, report.phases[1]);
        drop(taps);
        let projected = session
            .execute(
                id(format!("run.text_host.{precision:?}")),
                id(format!("request.text_host.{precision:?}")),
                vec![normalized],
            )
            .unwrap();
        assert_eq!(
            host.usage()
                .unwrap()
                .class_allocated_bytes(HostMemoryClass::InputPreprocess),
            0
        );
        assert_eq!(
            projected[0].bytes().backing_bytes(),
            report.phases[2].output_bytes
        );
        let conditioning = adapter
            .managed_conditioning(&scoped, &projected[0], &checkpoint, precision)
            .unwrap();
        assert_phase(&host, report.phases[3]);
        drop(projected);
        assert_eq!(
            conditioning.bytes().backing_bytes(),
            report.retained_conditioning_bytes
        );
        session.set_output_host_account(host.clone()).unwrap();
        scope.close().unwrap();
        assert_eq!(
            host.usage().unwrap().committed_bytes(),
            report.retained_conditioning_bytes
        );
        assert!(host.owns(conditioning.bytes()));
        assert_eq!(
            conditioning.tensor().element_type(),
            match precision {
                PH3ImageConditioningPrecision::F32 => ElementType::F32,
                PH3ImageConditioningPrecision::Bf16 => ElementType::Bf16,
            }
        );
        drop(conditioning);
        assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    }
    session
        .close()
        .unwrap_or_else(|e| panic!("projection close: {:?}", e.failure()));
    assert_eq!(composition.runtime().snapshot().live_allocations, 0);
}

#[test]
fn one_byte_short_scopes_reject_real_normalization_and_atomic_tap_readback_admission() {
    let prompt = prompt();
    let c = conditioning_config(prompt.layout().clone());
    let (_directory, archive) = connector_archive(&c);
    let connector = c.connector_config().unwrap();
    let checkpoint = PH3ImageTextConnectorCheckpoint::new_with_layout(
        &connector,
        &archive,
        connector_limits(&c),
        PH3ImageTextConnectorCheckpointLayout::ComfyMergedTextEncoder,
    )
    .unwrap();
    let adapter = PH3ImageTextConnectorManagedAdapter::new(
        &connector,
        c.encoder().text().layout().batch(),
        c.encoder().text().layout().sequence_length() as usize,
    )
    .unwrap();
    let host = account(1 << 20, 4096);
    let report = c
        .host_requirements(&host, PH3ImageConditioningPrecision::F32)
        .unwrap();
    let scope = host
        .reserve_scope(&[
            (
                HostMemoryClass::InputPreprocess,
                report.phases[1].input_bytes - 1,
            ),
            (HostMemoryClass::Output, report.output_peak_bytes),
        ])
        .unwrap();
    let scoped = scope.account();
    let taps = tap_buffers(&scoped, &c);
    let spec = tap_spec(&c);
    let error = adapter
        .managed_standardized_f32_input(&scoped, &tap_views(&c, &spec, &taps), &checkpoint)
        .err()
        .unwrap();
    assert!(error.chain().any(|e| matches!(
        e.downcast_ref::<HostMemoryError>(),
        Some(HostMemoryError::ScopeCapacity {
            class: HostMemoryClass::InputPreprocess,
            ..
        })
    )));
    assert_eq!(
        host.usage()
            .unwrap()
            .class_allocated_bytes(HostMemoryClass::InputPreprocess),
        0
    );
    assert_eq!(
        host.usage()
            .unwrap()
            .class_allocated_bytes(HostMemoryClass::Output),
        report.phases[1].output_bytes
    );
    drop(taps);
    scope.close().unwrap();
    let scope = host
        .reserve_scope(&[(HostMemoryClass::Output, report.phases[0].output_bytes - 1)])
        .unwrap();
    let logical = (c.encoder().text().layout().rows()
        * c.encoder().text().architecture().hidden_size()
        * 4) as usize;
    // This is the same all-output reserve_many admission used by the actual
    // executor. It fails atomically before any readback or device submission.
    let demand =
        vec![(HostMemoryClass::Output, logical); c.encoder().text().hidden_state_indices().len()];
    assert!(matches!(
        scope.account().reserve_many(&demand),
        Err(HostMemoryError::ScopeCapacity {
            class: HostMemoryClass::Output,
            ..
        })
    ));
    assert_eq!(host.usage().unwrap().allocated_bytes(), 0);
    scope.close().unwrap();
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn padding_sum_overflow_is_rejected_without_allocating() {
    let p = prompt();
    let c = conditioning_config(p.layout().clone());
    let host = account(1, 1usize << (usize::BITS - 2));
    assert!(c
        .host_requirements(&host, PH3ImageConditioningPrecision::F32)
        .is_err());
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}
