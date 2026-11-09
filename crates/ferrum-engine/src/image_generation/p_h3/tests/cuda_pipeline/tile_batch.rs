//! The product decoder batches across rows and attaches only the final real tail, sharing one
//! installation and preserving Cold retirement. This tiny Disabled-route wiring test complements
//! model-level tests of nonconstant tile isolation and Host retry.
use super::*;
use ferrum_models::p_h3_image_vae::PH3ImageVaeTileBatch;

#[test]
#[ignore = "requires CUDA, pinned native set and explicit cuDNN9 paths; actual product decoder cross-row4/3 tail transition"]
fn cuda_h3_decoder_tile_batches_cross_rows_and_retire_cold_shared_weights() -> Result<()> {
    let fixture = CompleteFixture::new()?;
    let mut baseline = None;
    for maximum in [1, 4] {
        let mut configured = config()?;
        configured.pipeline.maximum_width = 96;
        configured.pipeline.maximum_height = 64;
        configured.pipeline.residency = ImageResidencyPolicy::Cold;
        configured.pipeline.executor.device_timing_mode = ferrum_runtime::DeviceTimingMode::Kernel;
        configured.options.vae_spatial_policy = PH3ImageVaeSpatialPolicy::Tiled {
            tile_pixels: 32,
            minimum_overlap_pixels: 16,
            max_tiles: 32,
        };
        configured.options.vae_decoder_tile_batch = PH3ImageVaeTileBatch::new(maximum)?;
        let (mut pipeline, composition) =
            configured.create_pipeline_with_receipt(fixture.assets()?)?;
        assert_eq!(composition.vae_decoder_tile_batch.maximum_tiles(), maximum);
        // The service's image capacity remains B1 while a decoder invocation
        // has physical batch4. No device/account identity is substituted.
        assert_eq!(pipeline.execution.config.maximum_batch_size, 1);
        let host = pipeline.host_account().clone();
        let runtime = Arc::downgrade(pipeline.runtime());
        let mut input = request(17);
        input.width = 96;
        input.height = 64;
        let output = pipeline
            .generate(input)
            .unwrap_or_else(|e| panic!("product tile batch{maximum}: {e}"));
        assert_eq!(output.rgba.tensor().dimensions(), &[1, 3, 64, 96]);
        assert_eq!(
            output.receipt.decoder_batch_policy,
            ImageDecoderBatchPolicy::SingleImage
        );
        let decoder = output
            .receipt
            .stages
            .iter()
            .filter(|s| {
                s.execution.plan.stage == ImageStage::Decoder
                    && s.execution.completed_executions > 0
            })
            .collect::<Vec<_>>();
        let batches = decoder
            .iter()
            .map(|s| s.execution.plan.batch_size)
            .collect::<Vec<_>>();
        assert_eq!(batches, if maximum == 1 { vec![1] } else { vec![4, 3] });
        assert_eq!(
            decoder
                .iter()
                .map(|s| s.execution.completed_executions)
                .collect::<Vec<_>>(),
            if maximum == 1 { vec![15] } else { vec![3, 1] }
        );
        for (index, stage) in decoder.iter().enumerate() {
            assert_eq!(
                stage.execution.device_profiles.len(),
                stage.execution.completed_executions
            );
            assert_eq!(stage.execution.weights_reused, index > 0);
            assert!(stage.close.is_some());
        }
        assert!(
            !pipeline
                .execution
                .resident_stages
                .has_weights(ImageStage::Decoder),
            "Cold must release the shared decoder installation after the last real tail"
        );
        if maximum == 4 {
            assert!(output.receipt.stages.iter().any(|s| {
                s.execution.plan.stage == ImageStage::Decoder
                    && matches!(
                        s.retirement,
                        ImageStageRetirement::SharedWeightsClosed { .. }
                    )
            }));
        }
        let pixels = f32_pixels(&output);
        assert!(pixels.iter().all(|value| value.is_finite()));
        if let Some(expected) = &baseline {
            let expected: &Vec<f32> = expected;
            assert_eq!(pixels.len(), expected.len());
            // Predeclared fixture-local tolerance for changed decoder physical
            // batch geometry, matching the raw decoder's CUDA comparison.
            for (&actual, &expected) in pixels.iter().zip(expected) {
                assert!(
                    (actual - expected).abs() <= 1e-4 * (1.0 + expected.abs()),
                    "tile batch differs: serial={expected}, grouped={actual}"
                );
            }
        } else {
            baseline = Some(pixels);
        }
        pipeline
            .shutdown()
            .unwrap_or_else(|e| panic!("actual shutdown: {e}"));
        drop(pipeline);
        assert!(runtime.upgrade().is_none());
        assert!(host.usage()?.class_allocated_bytes(HostMemoryClass::Output) > 0);
        drop(output);
        assert_eq!(host.usage()?.committed_bytes(), 0);
    }
    Ok(())
}
