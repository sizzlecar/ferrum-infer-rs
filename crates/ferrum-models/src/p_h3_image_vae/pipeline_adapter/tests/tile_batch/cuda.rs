//! Real CUDA Stored decoder: one original installation, physical shape
//! transitions, shared-weight receipts, and public grouped-consumer execution.
use super::*;
use crate::p_h3_image_vae::decoder::tests::cuda::{Execution, GPU};
use ferrum_execution::component::ComponentInitializationReceipt;
use ferrum_kernels::backend::cuda::vnext_runtime::CudaDeviceRuntime;
use std::sync::atomic::{AtomicUsize, Ordering};

fn initialization() -> ComponentInitializationConfig {
    let mut config = super::initialization();
    config.executor.device_timing_mode = DeviceTimingMode::Kernel;
    config
}

struct CountSource<'a> {
    inner: PH3ImageVaeWeightSource<'a>,
    calls: AtomicUsize,
}
impl WeightComponentSource for CountSource<'_> {
    fn component<'s>(
        &'s self,
        c: &WeightComponentSpec,
    ) -> std::result::Result<WeightComponentPayload<'s>, VNextError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.inner.component(c)
    }
}
fn compile(
    f: &Execution,
    p: &PreparedPH3ImageVaeDecoder,
) -> CompiledTensorComponent<CudaDeviceRuntime> {
    let d = f.runtime.descriptor();
    CompiledTensorComponent::compile(
        p.family().clone(),
        &f.registry,
        &f.catalog,
        &TensorComponentPolicy::new(
            d.total_memory_bytes,
            d.dynamic_storage_profiles.iter().copied().collect(),
        )
        .unwrap(),
        &f.materializers,
        p.compile_options(),
    )
    .unwrap()
}
fn close(session: TensorComponentSession<CudaDeviceRuntime>) {
    assert!(matches!(
        session
            .close()
            .unwrap_or_else(|e| panic!("tile shape retirement: {:?}", e.failure())),
        PlanRuntimeCloseOutcome::Closed(_)
    ));
}
fn run(
    f: &Execution,
    adapter: &PH3ImageVaeAdapter,
    maximum: usize,
    perturb: bool,
    pressure: bool,
) -> Vec<f32> {
    let parent = host();
    let p = adapter
        .prepare_decoder_with_tile_batch(
            1,
            96,
            64,
            PH3ImageVaeSpatialPolicy::Tiled {
                tile_pixels: 32,
                minimum_overlap_pixels: 16,
                max_tiles: 32,
            },
            PH3ImageVaeDecoderPreset::F32A8FullAttention,
            PH3ImageVaeTileBatch::new(maximum).unwrap(),
        )
        .unwrap();
    let recipes = p
        .shapes()
        .iter()
        .map(|s| (s.tile_count(), compile(f, s)))
        .collect::<Vec<_>>();
    let source = CountSource {
        inner: p.shapes()[0].weight_source(parent.clone()),
        calls: AtomicUsize::new(0),
    };
    let (first, weights) = recipes[0]
        .1
        .initialize_shared_weights(
            f.runtime.clone(),
            parent.clone(),
            ComponentWeightSource::Stored(&source),
            initialization(),
        )
        .unwrap_or_else(|e| panic!("original tile installation: {e:?}"));
    let initial_calls = source.calls.load(Ordering::SeqCst);
    assert!(initial_calls > 0);
    let evidence = weights.evidence().clone();
    let mut session = Some(first);
    let mut active = recipes[0].0;
    let spec = p.normalized_latent_spec().unwrap();
    let n = spec.minimum_storage_bytes().unwrap() as usize / 4;
    let values = (0..n)
        .map(|i| {
            ((i * 7 + i / 24) % 59) as f32 / 32. - 0.75
                + 0.000123
                + if perturb && i % 24 == 23 { 0.75 } else { 0. }
        })
        .collect::<Vec<_>>();
    let sample = input(&parent, spec, &values, "value.grouped.cuda.sample");
    let requirements = p.host_requirements(&parent).unwrap();
    let scope = parent.reserve_scope(&requirements.scope_claims()).unwrap();
    let h = scope.account();
    let mut consumer = p
        .bind(&sample, id("request.grouped.cuda"), h.clone(), h.clone())
        .unwrap();
    let mut actor = ferrum_execution::TensorSerialActor::new(f.runtime.clone()).unwrap();
    if recipes.len() > 1 {
        let mut wrong = recipes[1]
            .1
            .initialize_with_shared_weights(
                f.runtime.clone(),
                parent.clone(),
                &weights,
                initialization(),
            )
            .unwrap_or_else(|e| panic!("wrong-shape attach: {e:?}"));
        wrong.set_output_host_account(h.clone()).unwrap();
        let before = parent.usage().unwrap().allocated_bytes();
        assert!(matches!(
            consumer.advance_with_session(&mut wrong, &mut actor, id("run.grouped.wrong_shape")),
            Err(PH3ImageVaePipelineError::Preparation(_))
        ));
        assert_eq!(consumer.executed_groups(), 0);
        assert!(!consumer.execution_failed());
        assert!(!consumer.has_pending_output());
        assert_eq!(parent.usage().unwrap().allocated_bytes(), before);
        assert!(wrong.take_execution_profile().is_none());
        close(wrong);
    }
    let mut blocker = pressure.then(|| h.allocate_zeroed(HostMemoryClass::Output, 1).unwrap());
    let mut retries = 0;
    let mut attachments = 0;
    let mut submitted_groups = 0;
    while let Some(group) = consumer.next_group().unwrap() {
        if active != group.tile_count() {
            assert!(!consumer.has_pending_output());
            close(session.take().unwrap());
            let compiled = &recipes
                .iter()
                .find(|(count, _)| *count == group.tile_count())
                .unwrap()
                .1;
            let next = compiled
                .initialize_with_shared_weights(
                    f.runtime.clone(),
                    parent.clone(),
                    &weights,
                    initialization(),
                )
                .unwrap_or_else(|e| panic!("real tail attach: {e:?}"));
            match next.initialization_receipt() {
                ComponentInitializationReceipt::SharedWeights { evidence: got } => {
                    assert_eq!(got, &evidence)
                }
                other => panic!("tail installed new weights: {other:?}"),
            }
            session = Some(next);
            active = group.tile_count();
            attachments += 1;
        }
        let current = session.as_mut().unwrap();
        current.set_output_host_account(h.clone()).unwrap();
        match consumer.advance_with_session(
            current,
            &mut actor,
            id(format!("run.grouped.cuda.{submitted_groups}.{retries}")),
        ) {
            Ok(_) => {
                submitted_groups += 1;
            }
            Err(PH3ImageVaePipelineError::Preparation(e)) => {
                assert!(
                    pressure && blocker.is_some(),
                    "unexpected preparation: {e:#}"
                );
                assert!(
                    e.chain().any(|e| matches!(
                        e.downcast_ref::<HostMemoryError>(),
                        Some(HostMemoryError::ScopeCapacity {
                            class: HostMemoryClass::Output,
                            ..
                        })
                    )),
                    "{e:#}"
                );
                assert!(consumer.has_pending_output());
                assert!(!consumer.execution_failed());
                let ran = consumer.executed_groups();
                let tiles = consumer.executed_tiles();
                assert!(
                    current.take_execution_profile().is_some(),
                    "pressure follows real completed native work"
                );
                drop(blocker.take());
                consumer
                    .advance_with_session(current, &mut actor, id("run.grouped.cuda.retry"))
                    .unwrap();
                assert_eq!(consumer.executed_groups(), ran);
                assert_eq!(consumer.executed_tiles(), tiles);
                assert!(current.take_execution_profile().is_none());
                retries += 1;
                submitted_groups += 1;
            }
            Err(e) => panic!("CUDA tile group execution failed: {e:?}"),
        }
    }
    assert_eq!(consumer.completed_tiles(), 15);
    assert_eq!(consumer.executed_tiles(), 15);
    assert_eq!(consumer.executed_groups(), submitted_groups);
    assert_eq!(submitted_groups, if maximum == 1 { 15 } else { 6 });
    if maximum == 4 {
        assert_eq!(attachments, 5);
    } else {
        assert_eq!(attachments, 0);
    }
    if pressure {
        assert_eq!(retries, 1);
    }
    assert_eq!(source.calls.load(Ordering::SeqCst), initial_calls);
    let pixels = consumer.finish().unwrap();
    assert_eq!(pixels.tensor().dimensions(), [1, 3, 64, 96]);
    close(session.take().unwrap());
    drop(actor);
    assert!(matches!(
        weights
            .close()
            .unwrap_or_else(|e| panic!("shared VAE weights close: {:?}", e.failure())),
        SharedComponentWeightsCloseOutcome::Closed(_)
    ));
    drop((source, sample, blocker));
    scope.close().unwrap();
    assert_eq!(
        parent.usage().unwrap().committed_bytes(),
        pixels.bytes().backing_bytes()
    );
    let out = pixels
        .bytes()
        .bytes()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    drop(pixels);
    assert_eq!(parent.usage().unwrap().committed_bytes(), 0);
    out
}
#[test]
#[ignore = "requires CUDA; complete raw-W8 decoder with actual4/1 tail sessions and one original shared installation"]
fn actual_cuda_spatial_tile_batches_share_one_installation_and_match_ordered_single_tiles() {
    let _guard = GPU.lock().unwrap_or_else(|p| p.into_inner());
    let f = Execution::new();
    let weak = Arc::downgrade(&f.runtime);
    let (_dir, source, _, _) = fixture();
    let adapter = PH3ImageVaeAdapter::new(source, &metadata(), limits()).unwrap();
    let serial = run(&f, &adapter, 1, false, false);
    let grouped = run(&f, &adapter, 4, false, true);
    let mut maximum = 0.0_f32;
    for (&a, &b) in serial.iter().zip(&grouped) {
        let delta = (a - b).abs();
        maximum = maximum.max(delta);
        assert!(
            a.is_finite() && b.is_finite() && delta <= 1e-4 * (1. + a.abs()),
            "serial={a}, grouped={b}, delta={delta}"
        );
    }
    eprintln!("spatial4/1 vs serial1 maximum normalized F32 pixel delta={maximum}");
    // Only the bottom-right normalized latent position changes (each channel
    // has24 spatial values). No other physical lane in the first row may see it.
    let perturbed = run(&f, &adapter, 4, true, false);
    assert!(grouped
        .iter()
        .zip(&perturbed)
        .any(|(a, b)| a.to_bits() != b.to_bits()));
    for channel in 0..3 {
        for y in 0..16 {
            let start = (channel * 64 + y) * 96;
            assert_eq!(
                &grouped[start..start + 96],
                &perturbed[start..start + 96],
                "unrelated top-row tiles must remain isolated"
            );
        }
    }
    drop(f);
    assert!(
        weak.upgrade().is_none(),
        "closed sessions/weights/actor must not retain native owner"
    );
}
