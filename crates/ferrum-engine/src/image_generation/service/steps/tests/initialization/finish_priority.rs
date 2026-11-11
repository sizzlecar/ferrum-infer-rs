//! Real completed DiT/VAE output versus an original later initializer.
use super::*;
use crate::image_generation::{
    host_budget, ImageStage, ImageStagePlanRole, PreparedImageStage, QwenImage21T2iLayout,
    QwenImage21TransformerConfig, QwenImage21VaeDecoderFamily,
};
use ferrum_models::image_pixels::{ImagePixelEncoding, OwnedImagePixels};
use ferrum_runtime::{ElementType, ResolvedTensorLayout, ResolvedTensorSpec};

pub(super) fn host_pressure_bytes(pipeline: &QwenImage21Pipeline<Runtime>) -> usize {
    let host = pipeline.host_account();
    let decoder = pipeline
        .execution
        .sources
        .assets
        .decoder_batch(1, 2, 2)
        .unwrap();
    let original = host
        .backing_bytes(
            decoder
                .output_tensor()
                .unwrap()
                .minimum_storage_bytes()
                .unwrap() as usize,
        )
        .unwrap();
    let config = QwenImage21TransformerConfig::new(
        pipeline.execution.sources.assets.transformer.clone(),
        QwenImage21T2iLayout::new(2, 2, 2, vec![1]).unwrap(),
    )
    .unwrap();
    let dit = pipeline.prepare_denoiser(&config, 1).unwrap();
    let step_output =
        host_budget::sum_ports(host, host_budget::ports(&dit).unwrap().outputs().iter()).unwrap();
    let f32_pixels = host.backing_bytes(3 * 32 * 32 * 4).unwrap();
    assert!(f32_pixels > step_output);
    // Keep the real B1 step runnable, but leave less than the actual F32
    // completed canvas. The original initializer owns and retires this credit.
    host.limits().class_bytes(HostMemoryClass::Output) - original - (step_output + f32_pixels) / 2
}

pub(super) fn finish_with_f32_pixels(
    inner: &mut Preconditioned,
    request: &mut ImageDenoisingRequest,
    batch: &mut ImageDenoisingBatch,
) -> Result<Option<ImageGenerationOutput>, ImageBatchError> {
    let host = inner.host_account().clone();
    let spec = ResolvedTensorSpec::new(
        vec![1, 3, 32, 32],
        ElementType::F32,
        ResolvedTensorLayout::Contiguous,
    )
    .unwrap();
    let mut pixels = match host.allocate_zeroed(HostMemoryClass::Output, 3 * 32 * 32 * 4) {
        Ok(pixels) => pixels,
        Err(error) if error.is_temporary_capacity() => return Ok(None),
        Err(error) => return Err(ImageBatchError::Rejected(error.to_string())),
    };
    let mut output = inner.finish_request(request, batch)?;
    for (source, target) in output.rgba.bytes().bytes()[..3 * 32 * 32 * 2]
        .chunks_exact(2)
        .zip(pixels.as_mut().chunks_exact_mut(4))
    {
        let value = half::bf16::from_bits(u16::from_le_bytes(source.try_into().unwrap())).to_f32();
        let clamped = half::bf16::from_f32(value.clamp(-1.0, 1.0));
        let half = half::bf16::from_f32(clamped.to_f32() * 0.5);
        let unit = half::bf16::from_f32(half.to_f32() + 0.5).to_f32();
        target.copy_from_slice(&unit.to_le_bytes());
    }
    output.rgba = OwnedImagePixels::new(
        &host,
        spec,
        pixels.freeze(),
        ImagePixelEncoding::F32UnitRgbRne,
    )
    .unwrap();
    Ok(Some(output))
}

pub(super) fn native_pressure_bytes(pipeline: &QwenImage21Pipeline<Runtime>) -> u64 {
    let config = QwenImage21TransformerConfig::new(
        pipeline.execution.sources.assets.transformer.clone(),
        QwenImage21T2iLayout::new(2, 2, 2, vec![1]).unwrap(),
    )
    .unwrap();
    let dit = pipeline.prepare_denoiser(&config, 1).unwrap();
    let decoder = pipeline
        .execution
        .sources
        .assets
        .decoder_batch(1, 2, 2)
        .unwrap();
    let family = pipeline
        .prepare_family(
            ImageStage::Decoder,
            ImageStagePlanRole::WholeStage,
            1,
            QwenImage21VaeDecoderFamily::new().unwrap(),
            &decoder,
            &pipeline.profiles.decoder,
        )
        .unwrap();
    let decode = PreparedImageStage::compile(
        ImageStage::Decoder,
        1,
        family,
        &decoder.compile_options().unwrap(),
        pipeline,
    )
    .unwrap();
    let demand = |stage: &PreparedImageStage<Runtime>| {
        let memory = stage.executable().execution_plan().payload().memory();
        memory.static_bytes() + memory.minimum_runnable_request_bytes()
    };
    let available = ((demand(&dit) + demand(&decode)) / 2 / 256) * 256;
    assert!(demand(&dit) <= available && available < demand(&decode));
    pipeline.execution.config.device_capacity_bytes - available
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn completed_output_is_delivered_before_later_ordinary_initialization() {
    let (service, boundary, trace, runtime, control) = start(Mode::Pending, true);
    let _release = ReleaseOnDrop(boundary.clone());
    let _phase_release = ReleaseInitializationOnDrop(control.clone());
    let mut first = service.try_submit(request(0)).unwrap();
    boundary.wait(1).await;
    control.state.lock().unwrap().completed_output_probe = first.inner.receiver.take();
    let later = service.try_submit(request(1)).unwrap();
    boundary.release(1);
    boundary.wait(2).await;
    assert!(control.state.lock().unwrap().begun);
    boundary.release(2);
    let later = tokio::time::timeout(Duration::from_secs(15), later.wait())
        .await
        .expect("later initialization resumes after delivery")
        .unwrap();
    let first_output = control
        .state
        .lock()
        .unwrap()
        .observed_completed_output
        .take()
        .expect("initializer consumed the already delivered original output");
    assert_eq!([first_output.receipt.seed, later.receipt.seed], [0, 1]);
    assert!(first_output
        .receipt
        .stages
        .iter()
        .any(|s| s.execution.plan.stage == ImageStage::Decoder));
    check(
        &trace.lock().unwrap(),
        &[
            fixture_trajectory(&[0], &[0]),
            fixture_trajectory(&[1], &[0]),
        ],
    );
    drop((first, first_output, later));
    finish(&service, &runtime).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn real_native_decoder_wait_still_advances_original_initializer_to_retirement() {
    completion_wait(Mode::FinishNativePressure).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn real_host_output_wait_still_advances_original_initializer_to_retirement() {
    completion_wait(Mode::FinishHostPressure).await;
}

async fn completion_wait(mode: Mode) {
    let (service, boundary, trace, runtime, control) = start(mode, true);
    let _release = ReleaseOnDrop(boundary.clone());
    let _phase_release = ReleaseInitializationOnDrop(control.clone());
    let first = service.try_submit(request(0)).unwrap();
    boundary.wait(1).await;
    let later = service.try_submit(request(1)).unwrap();
    boundary.release(1);
    boundary.wait(2).await;
    assert!(control.state.lock().unwrap().phase_live);
    boundary.release(2);
    let outputs = tokio::time::timeout(Duration::from_secs(15), async {
        [first.wait().await.unwrap(), later.wait().await.unwrap()]
    })
    .await
    .expect("real completion wait cannot strand its original initializer");
    let state = control.state.lock().unwrap();
    assert!(state.terminal && !state.phase_live);
    match mode {
        Mode::FinishNativePressure => assert!(state.native_finish_waited),
        Mode::FinishHostPressure => assert!(state.host_finish_waited),
        _ => unreachable!(),
    }
    drop(state);
    check(
        &trace.lock().unwrap(),
        &[
            fixture_trajectory(&[0], &[0]),
            fixture_trajectory(&[1], &[0]),
        ],
    );
    assert!(outputs
        .iter()
        .all(|output| output
            .receipt
            .stages
            .iter()
            .any(|s| s.execution.plan.stage == ImageStage::Decoder)));
    drop(outputs);
    finish(&service, &runtime).await;
}
