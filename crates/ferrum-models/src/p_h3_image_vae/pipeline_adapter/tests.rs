use super::*;
#[cfg(feature = "cuda")]
use crate::p_h3_image_vae::decoder::tests as decoder_fixture;
use ferrum_execution::{
    component::{
        CompiledTensorComponent, ComponentInitializationConfig, ComponentWeightSource,
        TensorComponentPolicy,
    },
    TensorExecutorConfig,
};
use ferrum_kernels::backend::reference::{ReferenceDeviceRuntime, ReferenceVNextComposition};
use safetensors::Dtype;
mod encoder_host_requirements;
mod host_requirements;

fn id<T: TryFrom<String>>(s: impl Into<String>) -> T
where
    T::Error: std::fmt::Debug,
{
    T::try_from(s.into()).unwrap()
}
fn limits() -> PH3ImageVaeAdapterLimits {
    super::super::test_support::limits()
}
fn metadata() -> Vec<u8> {
    super::super::test_support::metadata()
}
fn fixture() -> (tempfile::TempDir, Arc<ComfyCanonicalWeightSource>, u64, u64) {
    super::super::test_support::TinyPH3ImageVaeFixture::new()
        .unwrap()
        .into_parts()
}
fn host() -> HostMemoryAccount {
    HostMemoryAccount::with_allocation_alignment(
        HostMemoryLimits::new(
            128 << 20,
            HostMemoryClassLimits {
                input_preprocess_bytes: 16 << 20,
                output_bytes: 48 << 20,
                transfer_staging_bytes: 32 << 20,
                weight_backing_bytes: 32 << 20,
            },
        )
        .unwrap(),
        64,
    )
    .unwrap()
}
fn cached<T: Send + Sync + 'static>(_: &T) {}
fn component<'a>(family: &'a PreparedModelFamily, name: &str) -> &'a WeightComponentSpec {
    family
        .weights()
        .stored_schema()
        .unwrap()
        .components
        .iter()
        .find(|c| c.external_names == [name])
        .unwrap()
}

#[test]
fn opaque_prepared_sources_keep_exact_temporal_int8_and_promotion_boundaries_without_cached_host_scopes(
) {
    let (_dir, source, encoder_bytes, decoder_bytes) = fixture();
    let adapter = PH3ImageVaeAdapter::new(source.clone(), &metadata(), limits()).unwrap();
    assert_eq!(adapter.latent_channels(), 24);
    assert_eq!(adapter.spatial_scale(), 16);
    assert_eq!(adapter.encoder_stored_bytes(), encoder_bytes);
    assert_eq!(adapter.decoder_stored_bytes(), decoder_bytes);
    let encoder = adapter
        .prepare_encoder(
            1,
            32,
            32,
            PH3ImageVaeSpatialPolicy::WholeImage,
            PH3ImageVaeEncoderPreset::ReferenceF32,
        )
        .unwrap();
    let larger = adapter
        .prepare_encoder(
            2,
            48,
            48,
            PH3ImageVaeSpatialPolicy::Tiled {
                tile_pixels: 32,
                minimum_overlap_pixels: 16,
                max_tiles: 4,
            },
            PH3ImageVaeEncoderPreset::CudaF32Fma,
        )
        .unwrap();
    let decoder = adapter
        .prepare_decoder(
            1,
            32,
            32,
            PH3ImageVaeSpatialPolicy::WholeImage,
            PH3ImageVaeDecoderPreset::F32A8FullAttention,
        )
        .unwrap();
    cached(&encoder);
    cached(&decoder);
    cached(&adapter);
    assert_eq!(
        encoder.family().weights().stored_schema().unwrap(),
        larger.family().weights().stored_schema().unwrap()
    );
    assert_ne!(
        encoder.family().numerical_profile().id,
        larger.family().numerical_profile().id
    );
    assert_eq!(
        encoder.pixel_input_spec().unwrap().dimensions(),
        [1, 3, 32, 32]
    );
    assert_eq!(
        larger.normalized_latent_spec().unwrap().dimensions(),
        [2, 24, 3, 3]
    );
    assert_eq!(
        decoder.pixel_output_spec().unwrap().dimensions(),
        [1, 3, 32, 32]
    );
    assert_eq!(decoder.pixel_encoding(), ImagePixelEncoding::F32UnitRgbRne);
    let (rows, frequencies, patches) = decoder.tile_layout_specs().unwrap();
    assert_eq!(rows.dimensions(), [4, 24]);
    assert_eq!(frequencies.dimensions(), [9, 48]);
    assert_eq!(patches.dimensions(), [9, 768]);
    let parent = host();
    let c = component(encoder.family(), "encoder.conv_in.weight");
    // Plenty of capacity exists at the parent. Only this injected grant is
    // too small; falling back to an unscoped parent would incorrectly succeed.
    let small = parent
        .reserve_scope(&[(HostMemoryClass::WeightBacking, 64)])
        .unwrap();
    let small_source = encoder.weight_source(small.account());
    assert!(small_source.component(c).is_err());
    assert_eq!(parent.usage().unwrap().allocated_bytes(), 0);
    drop(small_source);
    small.close().unwrap();
    let grant = parent
        .reserve_scope(&[(
            HostMemoryClass::WeightBacking,
            parent
                .backing_bytes(c.physical_bytes().unwrap() as usize)
                .unwrap(),
        )])
        .unwrap();
    let actual_source = encoder.weight_source(grant.account());
    let payload = actual_source.component(c).unwrap();
    let original = source.archive().tensor("encoder.conv_in.weight").unwrap();
    let shape = original.shape();
    let plane = (shape[3] * shape[4] * 4) as usize;
    for pair in 0..(shape[0] * shape[1]) as usize {
        let start = (pair * shape[2] as usize + shape[2] as usize - 1) * plane;
        assert_eq!(
            &payload.bytes()[pair * plane..(pair + 1) * plane],
            &original.bytes()[start..start + plane]
        );
    }
    assert!(payload.retained_host_memory().is_some());
    grant.close().unwrap();
    assert_eq!(payload.bytes().len(), c.physical_bytes().unwrap() as usize);
    drop(payload);
    drop(actual_source);
    assert_eq!(
        parent.usage().unwrap().committed_bytes(),
        0,
        "cached Prepared must retain no per-installation grant"
    );
    let decoder_source = decoder.weight_source(parent.clone());
    let raw = component(
        decoder.family(),
        "decoder.transformer_blocks.0.attn.to_qkv.weight",
    );
    let payload = decoder_source.component(raw).unwrap();
    assert_eq!(payload.element_type(), ElementType::I8);
    assert_eq!(
        payload.bytes().as_ptr(),
        source
            .archive()
            .tensor(&raw.external_names[0])
            .unwrap()
            .bytes()
            .as_ptr()
    );
    drop(payload);
    let norm = component(decoder.family(), "decoder.norm_out.weight");
    let payload = decoder_source.component(norm).unwrap();
    let original = source.archive().tensor("decoder.norm_out.weight").unwrap();
    for (x, y) in original
        .bytes()
        .chunks_exact(2)
        .zip(payload.bytes().chunks_exact(4))
    {
        let expected = half::bf16::from_bits(u16::from_le_bytes(x.try_into().unwrap())).to_f32();
        assert_eq!(y, expected.to_le_bytes());
    }
    assert!(parent.usage().unwrap().allocated_bytes() > 0);
    drop(payload);
    drop(decoder_source);
    assert_eq!(parent.usage().unwrap().committed_bytes(), 0);
    // The source owner and prepared metadata survive adapter destruction.
    drop(adapter);
    drop(source);
    assert_eq!(
        encoder
            .weight_source(parent.clone())
            .component(c)
            .unwrap()
            .bytes()
            .len(),
        c.physical_bytes().unwrap() as usize
    );
    assert_eq!(parent.usage().unwrap().committed_bytes(), 0);
}

fn initialize(
    backend: &ReferenceVNextComposition,
    prepared: &PreparedPH3ImageVaeEncoder,
    host: &HostMemoryAccount,
) -> TensorComponentSession<ReferenceDeviceRuntime> {
    let component = compile_reference(backend, prepared);
    let source = prepared.weight_source(host.clone());
    component
        .initialize(
            backend.runtime().clone(),
            host.clone(),
            ComponentWeightSource::Stored(&source),
            initialization(),
        )
        .unwrap_or_else(|e| panic!("public opaque VAE initialization: {e:?}"))
}
fn compile_reference(
    backend: &ReferenceVNextComposition,
    prepared: &PreparedPH3ImageVaeEncoder,
) -> CompiledTensorComponent<ReferenceDeviceRuntime> {
    let m = WeightMaterializerRegistry::identity_only().unwrap();
    let catalog = m.augment_catalog(backend.catalog().clone()).unwrap();
    let d = backend.runtime().descriptor();
    CompiledTensorComponent::compile(
        prepared.family().clone(),
        backend.registry(),
        &catalog,
        &TensorComponentPolicy::new(
            d.total_memory_bytes,
            d.dynamic_storage_profiles.iter().copied().collect(),
        )
        .unwrap(),
        &m,
        prepared.compile_options(),
    )
    .unwrap()
}
fn initialization() -> ComponentInitializationConfig {
    ComponentInitializationConfig {
        request_id: id("request.public_vae.init"),
        run_id: id("run.public_vae.init"),
        transaction_id: id("transaction.public_vae.init"),
        static_policy: StaticInitializationPolicy::new(16 << 20, 512).unwrap(),
        executor: TensorExecutorConfig::default(),
    }
}
fn input(
    host: &HostMemoryAccount,
    spec: ResolvedTensorSpec,
    values: &[f32],
    name: &str,
) -> ManagedHostTensor {
    let mut bytes = host
        .allocate_zeroed(HostMemoryClass::InputPreprocess, values.len() * 4)
        .unwrap();
    for (out, &value) in bytes.as_mut().chunks_exact_mut(4).zip(values) {
        out.copy_from_slice(&value.to_le_bytes());
    }
    ManagedHostTensor::new(id(name), spec, bytes.freeze()).unwrap()
}
#[test]
fn actual_stored_reference_public_consumers_interleave_one_actor_and_keep_normalized_output_owners()
{
    let (_dir, source, _, _) = fixture();
    let adapter = PH3ImageVaeAdapter::new(source, &metadata(), limits()).unwrap();
    let prepared = adapter
        .prepare_encoder(
            1,
            32,
            32,
            PH3ImageVaeSpatialPolicy::WholeImage,
            PH3ImageVaeEncoderPreset::ReferenceF32,
        )
        .unwrap();
    let backend = ReferenceVNextComposition::create(id("device.reference.public_vae")).unwrap();
    let host = host();
    let mut left = initialize(&backend, &prepared, &host);
    let mut right = initialize(&backend, &prepared, &host);
    let mut actor = TensorSerialActor::new(backend.runtime().clone()).unwrap();
    let unit: Vec<_> = (0..3 * 32 * 32)
        .map(|i| {
            if i == 0 {
                2_f32.powi(-26)
            } else {
                (i % 67) as f32 / 66.
            }
        })
        .collect();
    let signed: Vec<_> = unit
        .iter()
        .map(|&v| {
            let s = v * 2.;
            s - 1.
        })
        .collect();
    let noise: Vec<_> = (0..24 * 4)
        .map(|i| (i % 17) as f32 / 16. - 0.5 + 0.000123)
        .collect();
    let x = input(
        &host,
        prepared.pixel_input_spec().unwrap(),
        &unit,
        "value.public_vae.unit",
    );
    let y = input(
        &host,
        prepared.pixel_input_spec().unwrap(),
        &signed,
        "value.public_vae.signed",
    );
    let noise = input(
        &host,
        prepared.posterior_noise_spec().unwrap(),
        &noise,
        "value.public_vae.noise",
    );
    let mut a = prepared
        .bind(
            &mut left,
            &x,
            PH3ImageVaePixelDomain::UnitRgb,
            &noise,
            id("request.public_vae.a"),
            host.clone(),
            host.clone(),
        )
        .unwrap();
    let mut b = prepared
        .bind(
            &mut right,
            &y,
            PH3ImageVaePixelDomain::SignedRgb,
            &noise,
            id("request.public_vae.b"),
            host.clone(),
            host.clone(),
        )
        .unwrap();
    let before = backend.runtime().snapshot().commands;
    assert!(a.advance(&mut actor, id("run.public_vae.a")).unwrap());
    let middle = backend.runtime().snapshot().commands;
    assert!(middle > before);
    // Both consumers remain alive. The actor belongs to neither request and
    // can run another component between their advance/finish boundaries.
    assert!(b.advance(&mut actor, id("run.public_vae.b")).unwrap());
    assert!(backend.runtime().snapshot().commands > middle);
    let a_result = a.finish(id("value.public_vae.a.normalized")).unwrap();
    let b_result = b.finish(id("value.public_vae.b.normalized")).unwrap();
    assert_eq!(a_result.bytes().bytes(), b_result.bytes().bytes());
    assert_eq!(
        a_result.tensor(),
        &prepared.normalized_latent_spec().unwrap()
    );
    drop((a, b, actor));
    for session in [left, right] {
        assert!(matches!(
            session
                .close()
                .unwrap_or_else(|e| panic!("public VAE close: {:?}", e.failure())),
            PlanRuntimeCloseOutcome::Closed(_)
        ));
    }
    assert_eq!(backend.runtime().snapshot().live_allocations, 0);
    assert!(host.owns(a_result.bytes()));
    assert!(host.owns(b_result.bytes()));
    assert_eq!(a_result.bytes().bytes(), b_result.bytes().bytes());
    drop((a_result, b_result, x, y, noise));
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn actual_static_encoder_groups_admit_exact_source_peak_and_decoder_source_costs_match_real_allocations(
) {
    let (_dir, source, _, _) = fixture();
    let adapter = PH3ImageVaeAdapter::new(source.clone(), &metadata(), limits()).unwrap();
    let prepared = adapter
        .prepare_encoder(
            1,
            32,
            32,
            PH3ImageVaeSpatialPolicy::WholeImage,
            PH3ImageVaeEncoderPreset::ReferenceF32,
        )
        .unwrap();
    let backend =
        ReferenceVNextComposition::create(id("device.reference.vae_source_groups")).unwrap();
    let compiled = compile_reference(&backend, &prepared);
    let plan = compiled.executable().execution_plan();
    let parent = host();
    // Independent header-only inventory of actual packed source backings.
    // The executed core identity map below proves why these groups each have
    // one source payload; this is not an assumed fixed group size.
    let converted = prepared
        .family()
        .stored_weight_schema()
        .unwrap()
        .components
        .iter()
        .filter_map(|c| {
            let t = source.archive().tensor(&c.external_names[0]).unwrap();
            (t.shape().len() == 5 && t.shape()[2] > 1).then(|| {
                (
                    c,
                    parent
                        .backing_bytes(c.physical_bytes().unwrap() as usize)
                        .unwrap(),
                )
            })
        })
        .collect::<Vec<_>>();
    let (largest, expected) = converted.iter().max_by_key(|(_, n)| *n).copied().unwrap();
    let weights = plan.payload().execution_weights().stored().unwrap();
    for (component, sources) in weights.component_sources() {
        assert_eq!(sources, &[component.clone()]);
    }
    assert_eq!(
        prepared.maximum_group_backing_bytes(plan, &parent).unwrap(),
        expected
    );
    assert_eq!(
        parent.usage().unwrap().allocated_bytes(),
        0,
        "budget query must not materialize source weights"
    );
    let low = parent
        .reserve_scope(&[(HostMemoryClass::WeightBacking, expected - 1)])
        .unwrap();
    let limited = prepared.weight_source(low.account());
    assert!(limited.component(largest).is_err());
    assert_eq!(parent.usage().unwrap().allocated_bytes(), 0);
    drop(limited);
    low.close().unwrap();
    let scope = parent
        .reserve_scope(&[
            (HostMemoryClass::WeightBacking, expected),
            (HostMemoryClass::TransferStaging, 16 << 20),
        ])
        .unwrap();
    let scoped = scope.account();
    let exact = prepared.weight_source(scoped.clone());
    let session = compiled
        .initialize(
            backend.runtime().clone(),
            scoped.clone(),
            ComponentWeightSource::Stored(&exact),
            initialization(),
        )
        .unwrap_or_else(|e| {
            panic!("exact static-group backing must initialize real Stored encoder: {e:?}")
        });
    assert_eq!(
        scoped.usage().unwrap().allocated_bytes(),
        0,
        "source payloads and transfer staging retire after initialization"
    );
    assert!(matches!(
        session
            .close()
            .unwrap_or_else(|e| panic!("static group close: {:?}", e.failure())),
        PlanRuntimeCloseOutcome::Closed(_)
    ));
    drop((exact, scoped));
    scope.close().unwrap();
    assert_eq!(parent.usage().unwrap().committed_bytes(), 0);
    assert_eq!(backend.runtime().snapshot().live_allocations, 0);
    let other = adapter
        .prepare_encoder(
            2,
            32,
            32,
            PH3ImageVaeSpatialPolicy::WholeImage,
            PH3ImageVaeEncoderPreset::ReferenceF32,
        )
        .unwrap();
    let other_plan = compile_reference(&backend, &other);
    assert_eq!(
        adapter
            .maximum_encoder_group_backing_bytes(
                other.family(),
                other_plan.executable().execution_plan(),
                &parent
            )
            .unwrap(),
        expected
    );
    assert!(adapter
        .maximum_encoder_group_backing_bytes(
            prepared.family(),
            other_plan.executable().execution_plan(),
            &parent
        )
        .is_err());
    let source_view = adapter.encoder_weight_source(parent.clone());
    let source_payload = source_view.component(largest).unwrap();
    assert_eq!(parent.usage().unwrap().allocated_bytes(), expected);
    drop(source_payload);
    assert_eq!(parent.usage().unwrap().allocated_bytes(), 0);
    assert!(
        prepared
            .maximum_group_backing_bytes(other_plan.executable().execution_plan(), &parent)
            .is_err(),
        "another geometry's actual compiled plan cannot be substituted"
    );
    // Decoder cannot execute Reference raw W8. Instead verify every original
    // source allocation against the header-only prediction, and qualify its
    // actual compiled groups in the separate ignored CUDA fixture below.
    let decoder = adapter
        .prepare_decoder(
            1,
            32,
            32,
            PH3ImageVaeSpatialPolicy::WholeImage,
            PH3ImageVaeDecoderPreset::F32A8FullAttention,
        )
        .unwrap();
    let bound = decoder.weight_source(parent.clone());
    for c in &decoder.family().stored_weight_schema().unwrap().components {
        let original = source.archive().tensor(&c.external_names[0]).unwrap();
        let expected = if original.dtype() == Dtype::BF16 {
            parent
                .backing_bytes(c.physical_bytes().unwrap() as usize)
                .unwrap()
        } else {
            0
        };
        let metadata = decoder::checkpoint_source::temporary_backing_bytes(&source, c).unwrap();
        assert_eq!(
            if metadata == 0 {
                0
            } else {
                parent.backing_bytes(metadata as usize).unwrap()
            },
            expected
        );
        let payload = bound.component(c).unwrap();
        assert_eq!(
            parent.usage().unwrap().allocated_bytes(),
            expected,
            "actual source temporary for {}",
            c.id
        );
        drop(payload);
        assert_eq!(parent.usage().unwrap().allocated_bytes(), 0);
    }
    drop(bound);
    assert_eq!(parent.usage().unwrap().committed_bytes(), 0);
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires CUDA; actual complete decoder compilation/static initialization under its source-group grant"]
fn actual_cuda_decoder_static_groups_initialize_with_the_predicted_promotion_peak() {
    let _guard = decoder_fixture::cuda::GPU
        .lock()
        .unwrap_or_else(|p| p.into_inner());
    let execution = decoder_fixture::cuda::Execution::new();
    let (_dir, source, _, _) = fixture();
    let adapter = PH3ImageVaeAdapter::new(source.clone(), &metadata(), limits()).unwrap();
    let prepared = adapter
        .prepare_decoder(
            1,
            32,
            32,
            PH3ImageVaeSpatialPolicy::WholeImage,
            PH3ImageVaeDecoderPreset::F32A8FullAttention,
        )
        .unwrap();
    let parent = host();
    let d = execution.runtime.descriptor();
    let compiled = CompiledTensorComponent::compile(
        prepared.family().clone(),
        &execution.registry,
        &execution.catalog,
        &TensorComponentPolicy::new(
            d.total_memory_bytes,
            d.dynamic_storage_profiles.iter().copied().collect(),
        )
        .unwrap(),
        &execution.materializers,
        prepared.compile_options(),
    )
    .unwrap();
    let peak = prepared
        .maximum_group_backing_bytes(compiled.executable().execution_plan(), &parent)
        .unwrap();
    let expected = prepared
        .family()
        .stored_weight_schema()
        .unwrap()
        .components
        .iter()
        .filter(|c| {
            source
                .archive()
                .tensor(&c.external_names[0])
                .unwrap()
                .dtype()
                == Dtype::BF16
        })
        .map(|c| {
            parent
                .backing_bytes(c.physical_bytes().unwrap() as usize)
                .unwrap()
        })
        .max()
        .unwrap();
    assert_eq!(peak, expected);
    let scope = parent
        .reserve_scope(&[
            (HostMemoryClass::WeightBacking, peak),
            (HostMemoryClass::TransferStaging, 16 << 20),
        ])
        .unwrap();
    let scoped = scope.account();
    let bound = prepared.weight_source(scoped.clone());
    let session = compiled
        .initialize(
            execution.runtime.clone(),
            scoped.clone(),
            ComponentWeightSource::Stored(&bound),
            initialization(),
        )
        .unwrap_or_else(|e| panic!("CUDA decoder source group budget: {e:?}"));
    assert_eq!(scoped.usage().unwrap().allocated_bytes(), 0);
    assert!(matches!(
        session
            .close()
            .unwrap_or_else(|e| panic!("CUDA source group close: {:?}", e.failure())),
        PlanRuntimeCloseOutcome::Closed(_)
    ));
    drop((bound, scoped));
    scope.close().unwrap();
    assert_eq!(parent.usage().unwrap().committed_bytes(), 0);
}

mod tile_batch;
