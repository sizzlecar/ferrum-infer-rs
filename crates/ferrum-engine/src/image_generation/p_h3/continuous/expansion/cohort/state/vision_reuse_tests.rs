//! Exercise real visual forwards and the production cohort/cancellation
//! transitions. No causal or full-model numerical execution is claimed here.
use super::*;
use crate::image_generation::p_h3::{
    expansion::MergedExpansionFixture, PH3ImagePromptExpansionVisionReuse as Policy,
};
use crate::image_generation::{ImageEditRequest, ImageGenerationRequest};
use ferrum_output::png::{write_rgba8_png, PngDimensions, PngLimits};

fn image(host: &HostMemoryAccount, color: [u8; 4], prompt: &str) -> ImageEditRequest {
    let mut png = Vec::new();
    write_rgba8_png(
        &mut png,
        PngDimensions::new(3, 2).unwrap(),
        PngLimits::default(),
        |writer| writer.write_all(&color.repeat(6)),
    )
    .unwrap();
    let mut bytes = host
        .allocate_zeroed(HostMemoryClass::InputPreprocess, png.len())
        .unwrap();
    bytes.as_mut().copy_from_slice(&png);
    ImageEditRequest {
        references: Vec::new(),
        generation: ImageGenerationRequest {
            prompt: prompt.into(),
            width: 32,
            height: 32,
            seed: 42,
            n: 1,
        },
        image: bytes.freeze(),
    }
}

#[test]
fn cohort_exact_image_reuses_only_retired_output_and_survives_producer_cancellation() {
    for (policy, different_image, cancel_producer) in [
        (Policy::Off, false, false),
        (Policy::ExactImageWithinCohort, false, false),
        (Policy::ExactImageWithinCohort, true, false),
        (Policy::ExactImageWithinCohort, false, true),
    ] {
        let mut fixture = MergedExpansionFixture::new();
        assert_eq!(fixture.expansion.vision_reuse(), Policy::Off);
        Arc::get_mut(&mut fixture.expansion)
            .unwrap()
            .configure_vision_reuse(policy);
        assert_eq!(
            fixture.expansion.trace_identity().unwrap()["vision_reuse"],
            serde_json::to_value(policy).unwrap()
        );
        let mut pipeline = super::super::super::vision::tests::pipeline();
        let host = pipeline.host_account().clone();
        let runtime = pipeline.runtime().clone();
        let requests = [
            image(&host, [37, 91, 203, 255], "cat"),
            image(
                &host,
                if different_image {
                    [211, 23, 49, 255]
                } else {
                    [37, 91, 203, 255]
                },
                "dog dog",
            ),
        ];
        assert!(!requests[0].image.same_region(&requests[1].image));
        let visuals = requests
            .iter()
            .map(|request| {
                PreparedVisual::prepare(
                    &fixture.expansion,
                    &pipeline.expansion_composition(),
                    &host,
                    &pipeline.execution.config,
                    request,
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        let mut engine = ferrum_types::EngineConfig::default();
        engine.backend.device = ferrum_types::Device::CPU;
        let options = MergedExpansionFixture::native_options(engine);
        let preparations = requests
            .iter()
            .zip(&visuals)
            .map(|(request, visual)| {
                Some(
                    fixture
                        .expansion
                        .prepare_edit(
                            &request.generation.prompt,
                            (3, 2),
                            visual.plan().config().grid(),
                            fixture
                                .expansion
                                .visual_assets()
                                .unwrap()
                                .architecture()
                                .merge_size(),
                            &options,
                        )
                        .unwrap(),
                )
            })
            .collect::<Vec<_>>();
        let requirements = visuals
            .iter()
            .map(|v| v.requirements(&host).unwrap())
            .collect::<Vec<_>>();
        let mut budget = requirements[0].budget;
        // Only Input is broadened for two independent token/rotary bridges.
        // Output remains exactly the original single-producer class grant.
        budget.input += 1 << 20;
        budget.weights += requirements[1].materializer_additional_bytes();
        let phase = host.reserve_scope(&budget.claims()).unwrap();
        let members = requests
            .iter()
            .enumerate()
            .map(|(index, request)| {
                let correlation = ferrum_types::ImageRequestCorrelation::new(None);
                Member {
                    request_id: correlation.request_id,
                    correlation: Some(correlation),
                    identity: IngressIdentity::new(ImageRequestRef::Edit(request)),
                    job: index as u64,
                    output_credit: 0,
                    ingress_backing_bytes: 0,
                    prompt_trace: None,
                }
            })
            .collect::<Vec<_>>();
        let ids = members.iter().map(|m| m.request_id).collect::<Vec<_>>();
        let mut views = requests
            .iter()
            .zip(&ids)
            .map(|(request, &request_id)| ImageInitializationCohortRequest {
                request_id,
                request: ImageRequestRef::Edit(request),
                cancelled: false,
            })
            .collect::<Vec<_>>();
        let mut cohort = Cohort {
            pipeline: pipeline.execution.service_preparation_identity.clone(),
            assets: fixture.expansion.clone(),
            members,
            receipts: vec![],
            stage: CohortPhase::Visuals,
            phase: Some(phase),
            preparations,
            visuals: visuals.into_iter().map(Some).collect(),
            next_visual: 0,
            retained_visual: None,
            native: None,
            active_indices: vec![],
            outcomes: (0..2).map(|_| None).collect(),
            unvalidated_outcomes: vec![],
            retired_at: None,
            inputs: vec![],
            _requirements: None,
        };
        let wake = futures::task::noop_waker();
        assert!(
            pipeline
                .advance_cohort(&mut cohort, &views, &wake)
                .unwrap()
                .made_progress
        );
        let first = runtime.snapshot();
        assert!(first.submissions > 0);
        assert_eq!(
            first.live_allocations, 0,
            "retained source is fully retired"
        );
        let first_inputs = cohort.preparations[0]
            .as_ref()
            .unwrap()
            .retained_causal_inputs()
            .unwrap();
        let first_inputs_weak = Arc::downgrade(&first_inputs);
        drop(first_inputs);
        let before_output = host
            .usage()
            .unwrap()
            .class_allocated_bytes(HostMemoryClass::Output);
        if policy == Policy::Off {
            assert_eq!(before_output, 0);
        } else {
            assert!(before_output > 0 && before_output <= budget.output);
        }
        if policy == Policy::ExactImageWithinCohort && !different_image && !cancel_producer {
            let foreign = MergedExpansionFixture::new();
            let other = PreparedVisual::prepare(
                &foreign.expansion,
                &pipeline.expansion_composition(),
                &host,
                &pipeline.execution.config,
                &requests[1],
            )
            .unwrap();
            assert!(
                !cohort
                    .retained_visual
                    .as_ref()
                    .unwrap()
                    .matches(&foreign.expansion, &other),
                "byte-equal images do not cross immutable asset domains"
            );
        }
        views[0].cancelled = cancel_producer;
        assert!(
            pipeline
                .advance_cohort(&mut cohort, &views, &wake)
                .unwrap()
                .made_progress
        );
        let second = runtime.snapshot();
        let expected_hit = policy != Policy::Off && !different_image;
        assert_eq!(second.submissions == first.submissions, expected_hit);
        assert_eq!(second.live_allocations, 0);
        assert_eq!(
            host.usage()
                .unwrap()
                .class_allocated_bytes(HostMemoryClass::Output),
            before_output,
            "a miss retires the old Output before admitting another, never two physical outputs"
        );
        let survivor = cohort.preparations[1].as_ref().unwrap();
        let receipt = survivor.vision_binding().unwrap();
        assert_eq!(receipt.reused, expected_hit);
        assert_eq!(receipt.producer_job, if expected_hit { 0 } else { 1 });
        assert_eq!(
            receipt.producer_request_id,
            Some(ids[if expected_hit { 0 } else { 1 }])
        );
        assert!(
            !first_inputs_weak.ptr_eq(&Arc::downgrade(&survivor.retained_causal_inputs().unwrap())),
            "each request owns a separately bound feature/mask/rotary set"
        );
        if cancel_producer {
            assert!(cohort.preparations[0].is_none());
            assert!(
                first_inputs_weak.upgrade().is_none(),
                "cancelled member inputs are actually released"
            );
            assert!(matches!(
                cohort.outcomes[0],
                Some(ExpansionParticipantResult::Cancelled)
            ));
            assert!(
                receipt.reused,
                "cancelled producer need not complete a causal sequence"
            );
        }
        // Cancel both before causal acquisition: production install must drop
        // the saved Output even when no original member remains alive.
        for view in &mut views {
            view.cancelled = true;
        }
        pipeline.cancel_pre_native_members(&mut cohort, &views);
        pipeline.install_cohort(&mut cohort).unwrap();
        assert!(cohort.retained_visual.is_none());
        assert_eq!(
            host.usage()
                .unwrap()
                .class_allocated_bytes(HostMemoryClass::Output),
            0
        );
        assert!(cohort.stage == CohortPhase::Done);
        drop((cohort, requirements, requests));
        pipeline.shutdown().unwrap();
        drop((pipeline, runtime, fixture));
        assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    }
}
