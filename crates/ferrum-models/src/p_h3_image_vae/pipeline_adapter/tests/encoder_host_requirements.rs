use super::*;

fn aligned() -> HostMemoryAccount {
    HostMemoryAccount::with_allocation_alignment(host().limits(), 4096).unwrap()
}
fn pressure(error: &anyhow::Error, class: HostMemoryClass) {
    assert!(
        error
            .chain()
            .any(|e| matches!(e.downcast_ref::<HostMemoryError>(),
        Some(HostMemoryError::ScopeCapacity {class:actual,..}) if *actual==class)),
        "{error:#}"
    );
}

#[test]
fn actual_full_reference_encoder_uses_queried_scope_and_retries_completed_host_boundaries() {
    let (_directory, source, _, _) = fixture();
    let adapter = PH3ImageVaeAdapter::new(source, &metadata(), limits()).unwrap();
    let backend =
        ReferenceVNextComposition::create(id("device.reference.encoder_host_requirements"))
            .unwrap();
    let tiled = PH3ImageVaeSpatialPolicy::Tiled {
        tile_pixels: 32,
        minimum_overlap_pixels: 16,
        max_tiles: 4,
    };
    for (batch, width, height, policy, short) in [
        (1, 32, 32, PH3ImageVaeSpatialPolicy::WholeImage, false),
        (1, 32, 32, PH3ImageVaeSpatialPolicy::WholeImage, true),
        (1, 48, 32, tiled, false),
        (2, 48, 48, tiled, false),
    ] {
        let prepared = adapter
            .prepare_encoder(
                batch,
                width,
                height,
                policy,
                PH3ImageVaeEncoderPreset::ReferenceF32,
            )
            .unwrap();
        let host = aligned();
        let report = prepared.host_requirements(&host).unwrap();
        assert_eq!(host.usage().unwrap().committed_bytes(), 0);
        let mut session = initialize(&backend, &prepared, &host);
        let pixel_spec = prepared.pixel_input_spec().unwrap();
        let noise_spec = prepared.posterior_noise_spec().unwrap();
        let values = (0..pixel_spec.minimum_storage_bytes().unwrap() as usize / 4)
            .map(|i| (i % 127) as f32 / 126.)
            .collect::<Vec<_>>();
        let noises = (0..noise_spec.minimum_storage_bytes().unwrap() as usize / 4)
            .map(|i| (i % 19) as f32 / 16. - 0.5)
            .collect::<Vec<_>>();
        let pixels = input(&host, pixel_spec, &values, "value.encoder_host.original");
        let noise = input(&host, noise_spec, &noises, "value.encoder_host.noise");
        let borrowed = pixels.bytes().backing_bytes() + noise.bytes().backing_bytes();
        let scope = host
            .reserve_scope(&[
                (HostMemoryClass::InputPreprocess, report.input_peak_bytes),
                (
                    HostMemoryClass::Output,
                    report.output_peak_bytes - usize::from(short),
                ),
            ])
            .unwrap();
        session.set_output_host_account(scope.account()).unwrap();
        let mut actor = TensorSerialActor::new(backend.runtime().clone()).unwrap();
        let mut consumer = prepared
            .bind(
                &mut session,
                &pixels,
                PH3ImageVaePixelDomain::UnitRgb,
                &noise,
                id("request.encoder_host"),
                scope.account(),
                scope.account(),
            )
            .unwrap();
        if short {
            let before = backend.runtime().snapshot().commands;
            let result = consumer.advance(&mut actor, id("run.encoder_host.short"));
            let Err(PH3ImageVaePipelineError::Preparation(error)) = result else {
                panic!("expected completed-encoder stitch admission failure: {result:?}");
            };
            pressure(&error, HostMemoryClass::Output);
            assert_eq!(consumer.completed_tiles(), 0);
            assert!(backend.runtime().snapshot().commands > before);
            assert_eq!(
                host.usage()
                    .unwrap()
                    .class_allocated_bytes(HostMemoryClass::Output),
                report.moment_canvas_output_bytes + report.tile_readback_output_bytes
            );
            let completed = backend.runtime().snapshot().commands;
            let Err(PH3ImageVaePipelineError::Preparation(error)) =
                consumer.advance(&mut actor, id("run.encoder_host.retry_short"))
            else {
                panic!("retained readback must retry only Host stitch admission");
            };
            pressure(&error, HostMemoryClass::Output);
            assert_eq!(backend.runtime().snapshot().commands, completed);
            drop(consumer);
            session.set_output_host_account(host.clone()).unwrap();
            scope.close().unwrap();
            session
                .close()
                .unwrap_or_else(|e| panic!("encoder close: {:?}", e.failure()));
            drop(actor);
            assert_eq!(host.usage().unwrap().committed_bytes(), borrowed);
            drop((pixels, noise));
            assert_eq!(host.usage().unwrap().committed_bytes(), 0);
            assert_eq!(backend.runtime().snapshot().live_allocations, 0);
            continue;
        }
        loop {
            let tile = consumer.completed_tiles();
            if consumer
                .advance(&mut actor, id(format!("run.encoder_host.tile{tile}")))
                .unwrap()
            {
                break;
            }
        }
        assert_eq!(
            host.usage()
                .unwrap()
                .class_allocated_bytes(HostMemoryClass::Output),
            report.moment_canvas_output_bytes + report.raw_neighbour_output_peak_bytes
        );
        // An ordinary Input allocation spends existing work credit. finish
        // drops raw neighbours first, retains full moments on rejection, and
        // retries without repeating any complete CNN execution.
        let blocker = scope
            .account()
            .allocate_zeroed(
                HostMemoryClass::InputPreprocess,
                prepared
                    .tile_input_spec()
                    .unwrap()
                    .minimum_storage_bytes()
                    .unwrap() as usize,
            )
            .unwrap();
        let before = backend.runtime().snapshot().commands;
        let error = consumer
            .finish(id("value.encoder_host.posterior"))
            .err()
            .unwrap();
        pressure(&error, HostMemoryClass::InputPreprocess);
        assert_eq!(
            host.usage()
                .unwrap()
                .class_allocated_bytes(HostMemoryClass::Output),
            report.moment_canvas_output_bytes
        );
        let error = consumer
            .finish(id("value.encoder_host.posterior_retry"))
            .err()
            .unwrap();
        pressure(&error, HostMemoryClass::InputPreprocess);
        assert_eq!(backend.runtime().snapshot().commands, before);
        drop(blocker);
        let latent = consumer
            .finish(id("value.encoder_host.posterior_ready"))
            .unwrap();
        assert_eq!(backend.runtime().snapshot().commands, before);
        assert_eq!(latent.tensor(), &prepared.normalized_latent_spec().unwrap());
        assert_eq!(
            latent.bytes().backing_bytes(),
            report.retained_latent_input_bytes
        );
        assert!(latent
            .bytes()
            .bytes()
            .chunks_exact(4)
            .all(|b| f32::from_le_bytes(b.try_into().unwrap()).is_finite()));
        drop(consumer);
        session.set_output_host_account(host.clone()).unwrap();
        scope.close().unwrap();
        session
            .close()
            .unwrap_or_else(|e| panic!("encoder close: {:?}", e.failure()));
        drop(actor);
        assert_eq!(backend.runtime().snapshot().live_allocations, 0);
        assert_eq!(
            host.usage()
                .unwrap()
                .class_allocated_bytes(HostMemoryClass::Output),
            0
        );
        assert_eq!(
            host.usage().unwrap().committed_bytes(),
            borrowed + report.retained_latent_input_bytes
        );
        assert_eq!(
            pixels.bytes().bytes(),
            values
                .iter()
                .flat_map(|x| x.to_le_bytes())
                .collect::<Vec<_>>()
        );
        assert_eq!(
            noise.bytes().bytes(),
            noises
                .iter()
                .flat_map(|x| x.to_le_bytes())
                .collect::<Vec<_>>()
        );
        drop((latent, pixels, noise));
        assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    }
}

#[test]
fn encoder_query_is_geometry_metadata_with_borrowed_owners_separate_and_checked_padding() {
    let (_directory, source, _, _) = fixture();
    let adapter = PH3ImageVaeAdapter::new(source, &metadata(), limits()).unwrap();
    let host = aligned();
    for (width, height, policy) in [
        (32, 32, PH3ImageVaeSpatialPolicy::WholeImage),
        (
            48,
            32,
            PH3ImageVaeSpatialPolicy::Tiled {
                tile_pixels: 32,
                minimum_overlap_pixels: 16,
                max_tiles: 4,
            },
        ),
    ] {
        let prepared = adapter
            .prepare_encoder(
                1,
                width,
                height,
                policy,
                PH3ImageVaeEncoderPreset::ReferenceF32,
            )
            .unwrap();
        let report = prepared.host_requirements(&host).unwrap();
        assert_eq!(report.borrowed_pixels, prepared.pixel_input_spec().unwrap());
        assert_eq!(
            report.borrowed_posterior_noise,
            prepared.posterior_noise_spec().unwrap()
        );
        assert_eq!(
            report.retained_latent_input_bytes,
            host.backing_bytes(
                prepared
                    .normalized_latent_spec()
                    .unwrap()
                    .minimum_storage_bytes()
                    .unwrap() as usize
            )
            .unwrap()
        );
        assert_eq!(
            report.scope_claims().iter().map(|(_, n)| n).sum::<usize>(),
            report.single_scope_bytes
        );
        assert!(report.single_scope_bytes >= report.live_peak_bytes);
        assert!(
            prepared.output_peak(&host).unwrap() > report.output_peak_bytes,
            "one raw tile row does not imply a nonexistent additional upper row"
        );
        let huge_padding = HostMemoryAccount::with_allocation_alignment(
            host.limits(),
            1usize << (usize::BITS - 2),
        )
        .unwrap();
        assert!(prepared.host_requirements(&huge_padding).is_err());
        assert_eq!(huge_padding.usage().unwrap().committed_bytes(), 0);
    }
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}
