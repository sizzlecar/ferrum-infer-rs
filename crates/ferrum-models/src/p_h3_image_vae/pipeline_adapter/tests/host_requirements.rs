use super::*;
use crate::p_h3_image_vae::tiling::PH3ImageVaeCanvasDomain;

fn aligned(alignment: usize) -> HostMemoryAccount {
    HostMemoryAccount::with_allocation_alignment(host().limits(), alignment).unwrap()
}
fn normalized(p: &PreparedPH3ImageVaeDecoder, host: &HostMemoryAccount) -> ManagedHostTensor {
    let tensor = p.normalized_latent_spec().unwrap();
    let mut bytes = host
        .allocate_zeroed(
            HostMemoryClass::InputPreprocess,
            tensor.minimum_storage_bytes().unwrap() as usize,
        )
        .unwrap();
    for (i, b) in bytes.as_mut().chunks_exact_mut(4).enumerate() {
        b.copy_from_slice(&((i % 17) as f32 / 32. - 0.25).to_le_bytes());
    }
    ManagedHostTensor::new(id("value.host_requirement.sampler"), tensor, bytes.freeze()).unwrap()
}

/// Execute the consumer's real Host boundaries (crop/denormalize, rotary,
/// unpatchify and transactional Canvas), using explicit zero patch values.
/// There is no Reference W8 provider and no decoder math/CPU backend claim.
fn host_decoder_boundaries(
    prepared: &PreparedPH3ImageVaeDecoder,
    host: &HostMemoryAccount,
    output_capacity: usize,
) -> Result<(ManagedHostTensor, ResolvedTensorSpec, ManagedHostBytes)> {
    let requirements = prepared.host_requirements(host)?;
    let sample = normalized(prepared, host);
    let scope = host.reserve_scope(&[
        (
            HostMemoryClass::InputPreprocess,
            requirements.input_peak_bytes,
        ),
        (HostMemoryClass::Output, output_capacity),
    ])?;
    let scoped = scope.account();
    let layout = PH3ImageVaeImageLayout::new(
        prepared.validated.decoder.clone(),
        prepared.config.batch() as usize,
        prepared.config.latent_height() as usize,
        prepared.config.latent_width() as usize,
    )?;
    let (cosine, sine) = layout.managed_rotary(
        &scoped,
        id(PH3ImageVaeDecoderFamily::COSINE_ID),
        id(PH3ImageVaeDecoderFamily::SINE_ID),
    )?;
    assert_eq!(
        cosine.bytes().backing_bytes() + sine.bytes().backing_bytes(),
        requirements.cached_rotary_input_bytes
    );
    let statistics =
        PH3ImageVaeLatentStatistics::new(&prepared.validated.decoder, &prepared.validated.source)?;
    let mut canvas = PH3ImageVaeCanvas::new(
        prepared.plan.clone(),
        scoped.clone(),
        prepared.config.batch() as usize,
        3,
    )?;
    for index in 0..prepared.plan.tile_count() {
        let rows = prepared.plan.managed_tile_rows(
            &scoped,
            &sample,
            &statistics,
            index,
            1,
            id(PH3ImageVaeDecoderFamily::INPUT_ID),
        )?;
        assert_eq!(rows.bytes().backing_bytes(), requirements.work_input_bytes);
        let patch_spec = layout.patch_output_spec()?;
        let patch = scoped
            .allocate_zeroed(
                HostMemoryClass::Output,
                patch_spec.minimum_storage_bytes()? as usize,
            )?
            .freeze();
        // A successful execution retires tile Input before unpatchify. The
        // explicitly provided patch owner models only that completed boundary.
        drop(rows);
        let (pixels_spec, pixels) = layout.unpatchify(&scoped, &patch_spec, &patch)?;
        drop(patch);
        let previous = canvas.completed_tiles();
        let source_pointer = pixels.bytes().as_ptr();
        if let Err(error) = canvas.push(index, &pixels_spec, &pixels) {
            assert_eq!(canvas.completed_tiles(), previous);
            assert_eq!(pixels.bytes().as_ptr(), source_pointer);
            return Err(error);
        }
    }
    let (tensor, pixels) = canvas.finish(PH3ImageVaeCanvasDomain::UnitRgb)?;
    drop((cosine, sine));
    scope.close()?;
    assert_eq!(
        pixels.backing_bytes(),
        requirements.retained_pixel_output_bytes
    );
    assert_eq!(
        host.usage()?.committed_bytes(),
        sample.bytes().backing_bytes() + requirements.retained_pixel_output_bytes
    );
    Ok((sample, tensor, pixels))
}

#[test]
fn exact_decoder_requirements_cover_real_host_paths_and_one_byte_less_rejects() {
    let (_directory, source, _, _) = fixture();
    let adapter = PH3ImageVaeAdapter::new(source, &metadata(), limits()).unwrap();
    let tiled = |size| PH3ImageVaeSpatialPolicy::Tiled {
        tile_pixels: size,
        minimum_overlap_pixels: 16,
        max_tiles: 16,
    };
    for (batch, width, height, policy) in [
        (1, 32, 32, PH3ImageVaeSpatialPolicy::WholeImage),
        (2, 48, 48, tiled(32)),
        // Unequal 32/16 overlaps along the width exercise old/new strip and
        // left-tail sizes instead of adding unrelated per-axis maxima.
        (1, 96, 80, tiled(48)),
    ] {
        let prepared = adapter
            .prepare_decoder(
                batch,
                width,
                height,
                policy,
                PH3ImageVaeDecoderPreset::F32A8FullAttention,
            )
            .unwrap();
        for alignment in [1, 4096] {
            let host = aligned(alignment);
            let report = prepared.host_requirements(&host).unwrap();
            assert_eq!(host.usage().unwrap().committed_bytes(), 0);
            assert_eq!(
                report.scope_claims().iter().map(|(_, n)| n).sum::<usize>(),
                report.single_scope_bytes
            );
            assert!(report.single_scope_bytes >= report.live_peak_bytes);
            let (sample, tensor, pixels) =
                host_decoder_boundaries(&prepared, &host, report.output_peak_bytes).unwrap();
            assert_eq!(tensor, prepared.pixel_output_spec().unwrap());
            let positions = width * height;
            for (i, bytes) in pixels.bytes().chunks_exact(4).enumerate() {
                let channel = (i / positions) % 3;
                assert_eq!(
                    f32::from_le_bytes(bytes.try_into().unwrap()).to_bits(),
                    [0.485_f32, 0.456, 0.406][channel].to_bits()
                );
            }
            drop((sample, pixels));
            assert_eq!(host.usage().unwrap().committed_bytes(), 0);
            let error = host_decoder_boundaries(&prepared, &host, report.output_peak_bytes - 1)
                .err()
                .unwrap();
            assert!(
                error.chain().any(|e| matches!(
                    e.downcast_ref::<HostMemoryError>(),
                    Some(HostMemoryError::ScopeCapacity {
                        class: HostMemoryClass::Output,
                        ..
                    })
                )),
                "{error:#}"
            );
            assert_eq!(host.usage().unwrap().committed_bytes(), 0);
        }
    }
}

#[test]
fn decoder_query_keeps_borrowed_state_outside_new_grants_and_rejects_padding_overflow() {
    let (_directory, source, _, _) = fixture();
    let adapter = PH3ImageVaeAdapter::new(source, &metadata(), limits()).unwrap();
    let prepared = adapter
        .prepare_decoder(
            1,
            32,
            32,
            PH3ImageVaeSpatialPolicy::WholeImage,
            PH3ImageVaeDecoderPreset::F32A8FullAttention,
        )
        .unwrap();
    let host = aligned(64);
    let before = prepared.host_requirements(&host).unwrap();
    let sample = normalized(&prepared, &host);
    assert_eq!(prepared.host_requirements(&host).unwrap(), before);
    assert_eq!(sample.tensor(), &before.borrowed_normalized_latents);
    assert_eq!(
        before.input_peak_bytes,
        before.cached_rotary_input_bytes + before.work_input_bytes
    );
    assert_eq!(
        before.output_peak_bytes,
        before.canvas_output_bytes + before.work_output_peak_bytes
    );
    // Existing compatibility helpers include borrowed state/conservative
    // unrelated phases. The new query exposes the actual consumer grant.
    assert!(prepared.input_peak(&host).unwrap() > before.input_peak_bytes);
    assert!(prepared.output_peak(&host).unwrap() > before.output_peak_bytes);
    drop(sample);
    let huge_padding = aligned(1usize << (usize::BITS - 2));
    assert!(prepared.host_requirements(&huge_padding).is_err());
    assert_eq!(huge_padding.usage().unwrap().committed_bytes(), 0);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a CUDA runtime and the original W8A8 decoder providers"]
fn actual_stored_cuda_tile_consumer_runs_inside_queried_decoder_scope_and_keeps_final_owner() {
    use crate::p_h3_image_vae::decoder::tests::cuda::{Execution, GPU};
    let _gpu = GPU.lock().unwrap_or_else(|p| p.into_inner());
    let (_directory, source, _, _) = fixture();
    let adapter = PH3ImageVaeAdapter::new(source, &metadata(), limits()).unwrap();
    let prepared = adapter
        .prepare_decoder(
            2,
            48,
            48,
            PH3ImageVaeSpatialPolicy::Tiled {
                tile_pixels: 32,
                minimum_overlap_pixels: 16,
                max_tiles: 4,
            },
            PH3ImageVaeDecoderPreset::F32A8FullAttention,
        )
        .unwrap();
    let f = Execution::new();
    let runtime_owner = Arc::downgrade(&f.runtime);
    let d = f.runtime.descriptor();
    let component = CompiledTensorComponent::compile(
        prepared.family().clone(),
        &f.registry,
        &f.catalog,
        &TensorComponentPolicy::new(
            d.total_memory_bytes,
            d.dynamic_storage_profiles.iter().copied().collect(),
        )
        .unwrap(),
        &f.materializers,
        prepared.compile_options(),
    )
    .unwrap();
    let host = aligned(4096);
    let weights = prepared.weight_source(host.clone());
    let mut session = component
        .initialize(
            f.runtime.clone(),
            host.clone(),
            ComponentWeightSource::Stored(&weights),
            ComponentInitializationConfig {
                run_id: id("run.vae_host.init"),
                request_id: id("request.vae_host.init"),
                transaction_id: id("transaction.vae_host.init"),
                static_policy: StaticInitializationPolicy::new(32 << 20, 1024).unwrap(),
                executor: TensorExecutorConfig::default(),
            },
        )
        .unwrap();
    drop(weights);
    let sample = normalized(&prepared, &host);
    let report = prepared.host_requirements(&host).unwrap();
    let scope = host.reserve_scope(&report.scope_claims()).unwrap();
    session.set_output_host_account(scope.account()).unwrap();
    let mut actor = TensorSerialActor::new(f.runtime.clone()).unwrap();
    let mut decoder = prepared
        .bind(
            &mut session,
            &sample,
            id("request.vae_host.decode"),
            scope.account(),
            scope.account(),
        )
        .unwrap();
    loop {
        let index = decoder.completed_tiles();
        if decoder
            .advance(&mut actor, id(format!("run.vae_host.tile{index}")))
            .unwrap()
        {
            break;
        }
    }
    let pixels = decoder.finish().unwrap();
    assert_eq!(pixels.tensor(), &prepared.pixel_output_spec().unwrap());
    session.set_output_host_account(host.clone()).unwrap();
    scope.close().unwrap();
    assert_eq!(
        host.usage()
            .unwrap()
            .class_allocated_bytes(HostMemoryClass::Output),
        report.retained_pixel_output_bytes
    );
    assert_eq!(
        host.usage()
            .unwrap()
            .class_allocated_bytes(HostMemoryClass::InputPreprocess),
        sample.bytes().backing_bytes()
    );
    drop(actor);
    assert!(matches!(
        session
            .close()
            .unwrap_or_else(|e| panic!("VAE close: {:?}", e.failure())),
        PlanRuntimeCloseOutcome::Closed(_)
    ));
    drop(component);
    drop(f);
    assert!(
        runtime_owner.upgrade().is_none(),
        "CUDA owners must retire while the final Host output remains live"
    );
    assert!(host.owns(pixels.bytes()));
    drop((sample, pixels));
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}
