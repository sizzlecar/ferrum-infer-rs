//! Real Reference visual producer and original Host grants. No CUDA or
//! full-model numerical claim is made by these diagnostic boundary tests.
use super::*;
use crate::image_generation::continuous_trace::{
    ImageContinuousTraceHandle, ImageContinuousTraceLimits,
};
use crate::image_generation::p_h3::{
    expansion::MergedExpansionFixture, tests::fixture::CompleteFixture, PH3ImagePipelineOptions,
};
use crate::image_generation::ImageGenerationRequest;
use ferrum_kernels::backend::reference::{
    reference_vnext_operation_registry, ReferenceDeviceRuntime, ReferenceVNextComposition,
};
use ferrum_output::png::{write_rgba8_png, PngDimensions, PngLimits};
use serde_json::{json, Value};

pub(in crate::image_generation::p_h3::continuous::expansion) fn pipeline(
) -> PH3ImagePipeline<ReferenceDeviceRuntime> {
    let composition = ReferenceVNextComposition::create_with_capacity(
        DeviceId::new("device.reference.visual.phase").unwrap(),
        32 << 20,
    )
    .unwrap();
    let runtime = composition.runtime().clone();
    let host = HostMemoryAccount::with_allocation_alignment(
        HostMemoryLimits::new(
            32 << 20,
            HostMemoryClassLimits {
                input_preprocess_bytes: 16 << 20,
                output_bytes: 4 << 20,
                transfer_staging_bytes: 4 << 20,
                weight_backing_bytes: 8 << 20,
            },
        )
        .unwrap(),
        64,
    )
    .unwrap();
    PH3ImagePipeline::new(
        runtime.clone(),
        host,
        reference_vnext_operation_registry(&runtime).unwrap(),
        composition.catalog().clone(),
        WeightMaterializerRegistry::identity_only().unwrap(),
        CompleteFixture::new().unwrap().assets().unwrap(),
        ImagePipelineConfig {
            device_capacity_bytes: 32 << 20,
            maximum_user_tokens: 512,
            maximum_width: 32,
            maximum_height: 32,
            maximum_batch_size: 2,
            weight_arena_target_bytes: None,
            decoder_batch_policy: Default::default(),
            edit_prefix_policy: Default::default(),
            plan_cache: Default::default(),
            residency: Default::default(),
            dynamic_residency: Default::default(),
            static_initialization: StaticInitializationPolicy::new(1 << 16, 16).unwrap(),
            executor: Default::default(),
        },
        PH3ImagePipelineOptions::comfy_f32(
            NumericalProfileId::new("fixture.visual.phase.noise").unwrap(),
            16,
        )
        .unwrap(),
    )
    .unwrap()
}

fn events(handle: &ImageContinuousTraceHandle) -> Vec<Value> {
    let mut bytes = Vec::new();
    let status = handle.write_jsonl(&mut bytes).unwrap();
    assert!(!status.truncated && status.dropped == 0);
    bytes
        .split(|&byte| byte == b'\n')
        .filter(|line| !line.is_empty())
        .map(|line| serde_json::from_slice::<Value>(line).unwrap())
        .filter(|record| record["event"]["kind"] != "header" && record.get("event").is_some())
        .collect()
}

#[test]
fn prompt_expansion_vision_phase_covers_actual_binding_and_original_grant_retirement() {
    // Compare real submissions/allocations with diagnostics disabled, enabled,
    // and enabled without an original service correlation (direct run).
    let mut baseline = None;
    for (tracing, correlated) in [(false, true), (true, true), (true, false)] {
        let (counts, captured) = exercise(tracing, correlated, false, false);
        if let Some(baseline) = baseline {
            assert_eq!(counts, baseline, "tracing must not submit additional work");
        } else {
            baseline = Some(counts);
        }
        assert_eq!(captured, tracing && correlated);
    }
}

#[test]
fn prompt_expansion_vision_rejections_finish_failed_and_return_the_same_scope() {
    for after_native in [false, true] {
        exercise(true, true, true, after_native);
    }
}

#[test]
fn two_actual_visual_producers_bind_both_picture_inputs_before_causal_installation() {
    let fixture = MergedExpansionFixture::new();
    let mut pipeline = pipeline();
    let host = pipeline.host_account().clone();
    let runtime = pipeline.runtime().clone();
    let images =
        [(3, 2, [37, 91, 203, 255]), (2, 3, [217, 42, 13, 255])].map(|(width, height, pixel)| {
            let mut png = Vec::new();
            write_rgba8_png(
                &mut png,
                PngDimensions::new(width, height).unwrap(),
                PngLimits::default(),
                |writer| writer.write_all(&pixel.repeat((width * height) as usize)),
            )
            .unwrap();
            let mut bytes = host
                .allocate_zeroed(HostMemoryClass::InputPreprocess, png.len())
                .unwrap();
            bytes.as_mut().copy_from_slice(&png);
            bytes.freeze()
        });
    let request = ImageEditRequest {
        generation: ImageGenerationRequest {
            prompt: "cat".into(),
            width: 32,
            height: 32,
            seed: 42,
            n: 1,
        },
        image: images[0].clone(),
        references: vec![images[1].clone()],
    };
    let visual = PreparedVisual::prepare(
        &fixture.expansion,
        &pipeline.expansion_composition(),
        &host,
        &pipeline.execution.config,
        &request,
    )
    .unwrap();
    let mut engine = ferrum_types::EngineConfig::default();
    engine.backend.device = ferrum_types::Device::CPU;
    let mut prepared = fixture
        .expansion
        .prepare_edits(
            "cat",
            (32, 32),
            &visual.grids(),
            visual.plan().config().architecture().merge_size(),
            &MergedExpansionFixture::native_options(engine),
        )
        .unwrap();
    let report = visual.requirements(&host).unwrap();
    // Execute visual forwards and the causal input bridge, without installing
    // the causal model on this Reference-only visual composition.
    let mut budget = report.budget;
    budget.input += visual.decoded_backing_bytes(&host).unwrap()
        + prepared.retained_visual_input_bytes(&host).unwrap();
    let phase = host.reserve_scope(&budget.claims()).unwrap();
    let original = phase.account();
    let result = pipeline
        .bind_visual_input(&fixture.expansion, &mut prepared, &visual, phase, 7, None)
        .unwrap();
    let VisualInputBinding::Bound { phase, output } = result else {
        panic!("both images must bind")
    };
    assert_eq!(output.len(), 2);
    assert_ne!(output[0].bytes().bytes(), output[1].bytes().bytes());
    assert!(output.iter().all(|o| original.owns(o.bytes())));
    assert!(prepared.retained_causal_inputs().is_some());
    assert_eq!(
        runtime.snapshot().live_allocations,
        0,
        "both visual owners retired before causal installation"
    );
    drop((output, prepared, visual, request, images, report, original));
    phase.close().unwrap();
    pipeline
        .shutdown()
        .unwrap_or_else(|e| panic!("shutdown: {e}"));
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}

fn exercise(
    tracing: bool,
    correlated: bool,
    rejected: bool,
    after_native: bool,
) -> ((u64, u64), bool) {
    let fixture = MergedExpansionFixture::new();
    let mut pipeline = pipeline();
    let host = pipeline.host_account().clone();
    let runtime = pipeline.runtime().clone();
    let weak = Arc::downgrade(&runtime);
    let handle = tracing.then(|| {
        pipeline
            .enable_continuous_trace(
                ImageContinuousTraceLimits {
                    maximum_bytes: 32768,
                },
                &json!({"fixture":"actual_reference_visual_binding"}),
            )
            .unwrap()
    });
    let mut png = Vec::new();
    write_rgba8_png(
        &mut png,
        PngDimensions::new(3, 2).unwrap(),
        PngLimits::default(),
        |writer| writer.write_all(&[37, 91, 203, 255].repeat(6)),
    )
    .unwrap();
    let mut input = host
        .allocate_zeroed(HostMemoryClass::InputPreprocess, png.len())
        .unwrap();
    input.as_mut().copy_from_slice(&png);
    let request = ImageEditRequest {
        references: Vec::new(),
        generation: ImageGenerationRequest {
            prompt: "cat".into(),
            width: 32,
            height: 32,
            seed: 42,
            n: 1,
        },
        image: input.freeze(),
    };
    let mut visual = PreparedVisual::prepare(
        &fixture.expansion,
        &pipeline.expansion_composition(),
        &host,
        &pipeline.execution.config,
        &request,
    )
    .unwrap();
    let mut engine = ferrum_types::EngineConfig::default();
    engine.backend.device = ferrum_types::Device::CPU;
    let mut prepared = fixture
        .expansion
        .prepare_edit(
            "cat",
            (3, 2),
            visual.plan().config().grid(),
            fixture
                .expansion
                .visual_assets()
                .unwrap()
                .architecture()
                .merge_size(),
            &MergedExpansionFixture::native_options(engine),
        )
        .unwrap();
    let requirements = visual.requirements(&host).unwrap();
    let mut budget = requirements.budget;
    // Decode and the existing F32->F16 causal bridge share this original scope.
    // The broad explicit fixture allowance is not a production peak claim.
    budget.input += 1 << 20;
    let phase = host.reserve_scope(&budget.claims()).unwrap();
    let original = phase.account();
    if rejected {
        if after_native {
            // The same actual visual forward retires, then the bridge rejects
            // an already-bound preparation on this second call below.
        } else {
            visual.dimensions = (2, 3);
        }
    }
    if rejected && after_native {
        let first = pipeline
            .bind_visual_input(&fixture.expansion, &mut prepared, &visual, phase, 0, None)
            .unwrap();
        let VisualInputBinding::Bound { phase, output } = first else {
            panic!("first actual binding")
        };
        drop(output);
        return finish_second(
            pipeline,
            fixture,
            prepared,
            visual,
            requirements,
            request,
            phase,
            original,
            host,
            runtime,
            weak,
            handle,
            correlated,
        );
    }
    let correlation = correlated.then(|| ferrum_types::ImageRequestCorrelation::new(None));
    let result = pipeline
        .bind_visual_input(
            &fixture.expansion,
            &mut prepared,
            &visual,
            phase,
            1,
            correlation,
        )
        .unwrap();
    finish(
        pipeline,
        prepared,
        visual,
        requirements,
        request,
        result,
        original,
        host,
        runtime,
        weak,
        handle,
        correlation,
        rejected,
    )
}

#[allow(clippy::too_many_arguments)]
fn finish_second(
    mut pipeline: PH3ImagePipeline<ReferenceDeviceRuntime>,
    fixture: MergedExpansionFixture,
    mut prepared: PreparedTextExpansion,
    visual: PreparedVisual<ReferenceDeviceRuntime>,
    requirements: ExpansionVisionHostRequirements,
    request: ImageEditRequest,
    phase: HostMemoryScope,
    original: HostMemoryAccount,
    host: HostMemoryAccount,
    runtime: Arc<ReferenceDeviceRuntime>,
    weak: std::sync::Weak<ReferenceDeviceRuntime>,
    handle: Option<ImageContinuousTraceHandle>,
    correlated: bool,
) -> ((u64, u64), bool) {
    let correlation = correlated.then(|| ferrum_types::ImageRequestCorrelation::new(None));
    let result = pipeline
        .bind_visual_input(
            &fixture.expansion,
            &mut prepared,
            &visual,
            phase,
            1,
            correlation,
        )
        .unwrap();
    finish(
        pipeline,
        prepared,
        visual,
        requirements,
        request,
        result,
        original,
        host,
        runtime,
        weak,
        handle,
        correlation,
        true,
    )
}

#[allow(clippy::too_many_arguments)]
fn finish(
    mut pipeline: PH3ImagePipeline<ReferenceDeviceRuntime>,
    prepared: PreparedTextExpansion,
    visual: PreparedVisual<ReferenceDeviceRuntime>,
    requirements: ExpansionVisionHostRequirements,
    request: ImageEditRequest,
    result: VisualInputBinding,
    original: HostMemoryAccount,
    host: HostMemoryAccount,
    runtime: Arc<ReferenceDeviceRuntime>,
    weak: std::sync::Weak<ReferenceDeviceRuntime>,
    handle: Option<ImageContinuousTraceHandle>,
    correlation: Option<ferrum_types::ImageRequestCorrelation>,
    rejected: bool,
) -> ((u64, u64), bool) {
    let phase = match result {
        VisualInputBinding::Bound { phase, output } => {
            assert!(!rejected);
            drop(output);
            phase
        }
        VisualInputBinding::Rejected { phase, diagnostic } => {
            assert!(rejected);
            assert!(!diagnostic.is_empty());
            phase
        }
    };
    let probe = original
        .allocate_zeroed(HostMemoryClass::InputPreprocess, 1)
        .unwrap()
        .freeze();
    assert!(phase.account().owns(&probe));
    drop(probe);
    let snapshot = runtime.snapshot();
    assert_eq!(
        snapshot.live_allocations, 0,
        "native visual owner has retired before span return"
    );
    let captured = handle.as_ref().map(events).unwrap_or_default();
    if let (Some(_), Some(correlation)) = (&handle, correlation) {
        assert_eq!(captured.len(), 2);
        assert_eq!(captured[0]["event"]["phase"], "prompt_expansion_vision");
        assert_eq!(
            captured[0]["event"]["request_id"],
            correlation.request_id.to_string()
        );
        assert!(captured[0]["event"]["request"].is_null());
        assert_eq!(
            captured[1]["event"]["start_sequence"],
            captured[0]["sequence"]
        );
        assert_eq!(
            captured[1]["event"]["outcome"],
            if rejected { "failed" } else { "completed" }
        );
        assert!(
            captured[1]["event_ns"].as_u64().unwrap() >= captured[0]["event_ns"].as_u64().unwrap()
        );
    } else {
        assert!(captured.is_empty());
    }
    drop((prepared, visual, requirements, request));
    phase.close().unwrap();
    assert!(original
        .allocate_zeroed(HostMemoryClass::InputPreprocess, 1)
        .is_err());
    pipeline.shutdown().unwrap();
    drop((pipeline, handle, runtime));
    assert!(weak.upgrade().is_none());
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    (
        (snapshot.allocations, snapshot.submissions),
        !captured.is_empty(),
    )
}
