//! Execute the real four-port Stored F32 visual component. Text inputs are
//! independently materialized; these tests do not execute quantized TE/DiT.
use super::*;
use crate::image_generation::p_h3::{tests::fixture::CompleteFixture, PH3ImagePipelineOptions};
use crate::image_generation::{
    ImageDynamicResidencyPolicy, ImagePipelineConfig, ImageResidencyPolicy, ImageStageAttachment,
};
use ferrum_kernels::backend::reference::{
    reference_vnext_operation_registry, ReferenceDeviceRuntime, ReferenceVNextComposition,
};
use ferrum_models::qwen_image21::image_input::ImageInputLimits;
use ferrum_output::png::{write_rgba8_png, PngDimensions, PngLimits};

fn pipeline() -> PH3ImagePipeline<ReferenceDeviceRuntime> {
    let composition = ReferenceVNextComposition::create_with_capacity(
        DeviceId::new("device.reference.main_vision_reuse").unwrap(),
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
            residency: ImageResidencyPolicy::Adaptive,
            dynamic_residency: ImageDynamicResidencyPolicy::Release,
            static_initialization: StaticInitializationPolicy::new(1 << 16, 16).unwrap(),
            executor: Default::default(),
        },
        PH3ImagePipelineOptions::comfy_f32(
            NumericalProfileId::new("fixture.main_vision.noise").unwrap(),
            16,
        )
        .unwrap(),
    )
    .unwrap()
}

fn prepared(
    pipeline: &PH3ImagePipeline<ReferenceDeviceRuntime>,
    prompt: &str,
) -> (
    PreparedVision,
    PreparedImageStage<ReferenceDeviceRuntime>,
    PH3ImageTextConditioningConfig,
    usize,
) {
    let assets = &pipeline.execution.sources.assets;
    let mut recipe = PreparedVision::new(
        assets,
        prompt,
        4,
        4,
        Qwen3VLVisionComfyPixelPolicy::new(16, 16).unwrap(),
    )
    .unwrap();
    // Same F32 graph and original source, explicitly select the Reference backend.
    recipe.family =
        TypedFamilyRegistration::new(Qwen3VLVisionComfyFamily::new(assets.limits.vision).unwrap())
            .prepare_with_profile(
                &serde_json::to_value(recipe.pixels.config()).unwrap(),
                &NumericalProfileId::new(Qwen3VLVisionComfyFamily::REFERENCE_PROFILE_ID).unwrap(),
            )
            .unwrap();
    let stage = pipeline
        .compile_recipe(
            ImageStage::VisionEncoder,
            ImageStageAttachment::MayInitializeStored,
            1,
            recipe.family.clone(),
            &recipe.options,
        )
        .unwrap();
    let text = PH3ImageTextConditioningConfig::from_comfy_checkpoint_with_connector(
        assets.text.clone(),
        recipe.prompt.text().layout().clone(),
        &assets.sources.text_encoder,
        &assets.connector,
    )
    .unwrap()
    .with_visual_inputs(recipe.prompt.visual_rows().clone())
    .unwrap();
    let bytes = crate::image_generation::host_budget::ports(&stage)
        .unwrap()
        .outputs()
        .iter()
        .map(|port| {
            pipeline
                .host_account()
                .backing_bytes(port.tensor().minimum_storage_bytes().unwrap() as usize)
                .unwrap()
        })
        .sum();
    (recipe, stage, text, bytes)
}

fn png(host: &HostMemoryAccount, pixel: [u8; 4]) -> ManagedHostBytes {
    let mut png = Vec::new();
    write_rgba8_png(
        &mut png,
        PngDimensions::new(4, 4).unwrap(),
        PngLimits::default(),
        |writer| writer.write_all(&pixel.repeat(16)),
    )
    .unwrap();
    let mut bytes = host
        .allocate_zeroed(HostMemoryClass::InputPreprocess, png.len())
        .unwrap();
    bytes.as_mut().copy_from_slice(&png);
    bytes.freeze()
}
fn image(host: &HostMemoryAccount, encoded: &ManagedHostBytes) -> ManagedRgba8Image {
    ManagedRgba8Image::decode_png(host, encoded, ImageInputLimits::default()).unwrap()
}
fn compare(left: &[ManagedHostTensor], right: &[ManagedHostTensor]) {
    assert_eq!(left.len(), right.len());
    for (left, right) in left.iter().zip(right) {
        assert_eq!(left.value_id(), right.value_id());
        assert_eq!(left.tensor(), right.tensor());
        assert_eq!(left.bytes().bytes(), right.bytes().bytes());
    }
}

fn request_work(host: &HostMemoryAccount) -> HostMemoryScope {
    host.reserve_scope(&[
        (HostMemoryClass::InputPreprocess, 8 << 20),
        (HostMemoryClass::Output, 0),
        (HostMemoryClass::TransferStaging, 2 << 20),
        (HostMemoryClass::WeightBacking, 4 << 20),
    ])
    .unwrap()
}

#[test]
fn main_vision_reuse_real_outputs_survive_producer_release_and_keep_prompt_inputs_independent() {
    for same_image in [true, false] {
        let mut pipeline = pipeline();
        let host = pipeline.host_account().clone();
        let runtime = pipeline.runtime().clone();
        let domain = pipeline.execution.service_preparation_identity.clone();
        let (first, first_stage, first_text, capacity) = prepared(&pipeline, "cat");
        let (second, second_stage, second_text, second_capacity) = prepared(&pipeline, "dog dog");
        assert_eq!(capacity, second_capacity);
        assert_ne!(
            first_text.encoder().text().layout().sequence_length(),
            second_text.encoder().text().layout().sequence_length()
        );
        let encoded = png(&host, [31, 91, 203, 255]);
        let other = if same_image {
            encoded.clone()
        } else {
            png(&host, [193, 41, 11, 255])
        };
        let first_image = image(&host, &encoded);
        let second_image = image(&host, &other);
        let mut receipts = Vec::new();
        let second_id = RequestIdentity::new("request.main_vision.survivor").unwrap();
        let (golden, receipt) = pipeline
            .execute_vision_inputs(
                &second,
                &second_stage,
                &second_image,
                &second_text,
                &second_id,
                20,
                &mut receipts,
                None,
            )
            .unwrap_or_else(|e| panic!("golden visual: {e}"));
        assert!(receipt.is_none());
        let cohort = CohortMainVisionReuse::new(&host, domain.clone(), capacity, 21).unwrap();
        assert!(cohort.try_admit(&host, &domain).unwrap());
        let producer = MainVisionReuseBinding {
            cohort: cohort.clone(),
            encoded,
        };
        let survivor = MainVisionReuseBinding {
            cohort: cohort.clone(),
            encoded: other,
        };
        let producer_id = RequestIdentity::new("request.main_vision.cancelled_producer").unwrap();
        let work = request_work(&host);
        pipeline.execution.execution_host = Some(work.account());
        let (first_inputs, receipt) = pipeline
            .execute_vision_inputs(
                &first,
                &first_stage,
                &first_image,
                &first_text,
                &producer_id,
                21,
                &mut receipts,
                Some(&producer),
            )
            .unwrap_or_else(|e| panic!("producer visual: {e}"));
        let receipt = receipt.unwrap();
        assert!(!receipt.reused);
        assert_eq!(receipt.producer_job, 21);
        assert_eq!(
            host.usage()
                .unwrap()
                .class_allocated_bytes(HostMemoryClass::Output),
            capacity
        );
        // The producer's request-side owners go away. The surviving binding
        // alone must keep the originally charged output and its grant alive.
        drop((first_inputs, producer, first_image, cohort));
        pipeline.execution.execution_host = None;
        work.close().unwrap();
        // The producer work grant had no Output allowance and is now closed.
        // Its cached readbacks can only survive in their original shared grant.
        let work = request_work(&host);
        pipeline.execution.execution_host = Some(work.account());
        let before = runtime.snapshot().commands;
        let (actual, receipt) = pipeline
            .execute_vision_inputs(
                &second,
                &second_stage,
                &second_image,
                &second_text,
                &second_id,
                22,
                &mut receipts,
                Some(&survivor),
            )
            .unwrap_or_else(|e| panic!("survivor visual: {e}"));
        let receipt = receipt.unwrap();
        assert_eq!(receipt.reused, same_image);
        assert_eq!(receipt.producer_job, if same_image { 21 } else { 22 });
        if same_image {
            assert_eq!(runtime.snapshot().commands, before);
        } else {
            assert!(runtime.snapshot().commands > before);
        }
        compare(&actual, &golden);
        assert_eq!(
            host.usage()
                .unwrap()
                .class_allocated_bytes(HostMemoryClass::Output),
            capacity
        );
        assert!(survivor.lookup(&Arc::new(()), &second).is_err());
        drop((actual, golden, survivor, second_image));
        pipeline.execution.execution_host = None;
        work.close().unwrap();
        assert_eq!(
            host.usage()
                .unwrap()
                .class_allocated_bytes(HostMemoryClass::Output),
            0
        );
        pipeline.shutdown().unwrap();
        drop((pipeline, first, second, first_stage, second_stage));
        assert_eq!(runtime.snapshot().live_allocations, 0);
        assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    }
}

#[test]
fn main_vision_reuse_requires_real_output_credit_and_refunds_cancelled_pending_binding() {
    let host = HostMemoryAccount::new(
        HostMemoryLimits::new(
            64,
            HostMemoryClassLimits {
                input_preprocess_bytes: 0,
                output_bytes: 64,
                transfer_staging_bytes: 0,
                weight_backing_bytes: 0,
            },
        )
        .unwrap(),
    );
    let domain = Arc::new(());
    assert!(CohortMainVisionReuse::new(&host, domain.clone(), 65, 0).is_err());
    let blocked = host
        .reserve_scope(&[(HostMemoryClass::Output, 64)])
        .unwrap();
    let cohort = CohortMainVisionReuse::new(&host, domain.clone(), 64, 0).unwrap();
    assert!(!cohort.try_admit(&host, &domain).unwrap());
    assert_eq!(host.usage().unwrap().committed_bytes(), 64);
    assert!(cohort.output_account().is_err());
    drop(blocked);
    assert!(cohort.try_admit(&host, &domain).unwrap());
    let survivor = cohort.clone();
    drop(cohort);
    assert_eq!(host.usage().unwrap().committed_bytes(), 64);
    drop(survivor);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn main_vision_reuse_retained_producer_ingress_cannot_create_self_owned_capacity_wait() {
    let mut pipeline = pipeline();
    let host = pipeline.host_account().clone();
    let runtime = pipeline.runtime().clone();
    let domain = pipeline.execution.service_preparation_identity.clone();
    let (recipe, stage, text, capacity) = prepared(&pipeline, "cat");
    let source = png(&host, [31, 91, 203, 255]);
    let current = png(&host, [31, 91, 203, 255]);
    assert!(!source.same_region(&current));
    assert_eq!(source.bytes(), current.bytes());
    let source_bytes = source.backing_bytes();
    let current_bytes = current.backing_bytes();
    let cohort = CohortMainVisionReuse::new(&host, domain.clone(), capacity, 1).unwrap();
    assert!(cohort.try_admit(&host, &domain).unwrap());
    let producer = MainVisionReuseBinding {
        cohort: cohort.clone(),
        encoded: source,
    };
    let survivor = MainVisionReuseBinding {
        cohort: cohort.clone(),
        encoded: current,
    };
    let pixels = image(&host, &producer.encoded);
    let work = request_work(&host);
    pipeline.execution.execution_host = Some(work.account());
    let (inputs, _) = pipeline
        .execute_vision_inputs(
            &recipe,
            &stage,
            &pixels,
            &text,
            &RequestIdentity::new("request.main_vision.capacity_producer").unwrap(),
            1,
            &mut vec![],
            Some(&producer),
        )
        .unwrap_or_else(|e| panic!("capacity producer: {e}"));
    drop((inputs, pixels));
    pipeline.execution.execution_host = None;
    work.close().unwrap();
    let limit = host.limits().class_bytes(HostMemoryClass::InputPreprocess);
    let before_fix_peak = [
        (HostMemoryClass::InputPreprocess, limit),
        (HostMemoryClass::Output, capacity),
    ];
    // A clone of the exact original region is already included in ingress.
    producer
        .check_request_capacity(&host, &before_fix_peak, 0)
        .unwrap();
    drop((producer, cohort));
    assert_eq!(
        host.usage()
            .unwrap()
            .class_allocated_bytes(HostMemoryClass::InputPreprocess),
        source_bytes + current_bytes
    );
    let original_claim = [(HostMemoryClass::InputPreprocess, limit - current_bytes)];
    host.check_scope_capacity(&before_fix_peak).unwrap();
    assert!(
        host.check_scope_admission(&original_claim)
            .unwrap_err()
            .is_temporary_capacity(),
        "original admission would wait on the survivor's own retained producer PNG"
    );
    let failure = survivor
        .check_request_capacity(&host, &before_fix_peak, 0)
        .unwrap_err();
    assert!(!failure
        .downcast_ref::<HostMemoryError>()
        .unwrap()
        .is_temporary_capacity());
    let fit = limit - current_bytes - source_bytes;
    let fit_peak = [
        (HostMemoryClass::InputPreprocess, fit + current_bytes),
        (HostMemoryClass::Output, capacity),
    ];
    survivor
        .check_request_capacity(&host, &fit_peak, 0)
        .unwrap();
    let blocker = host
        .reserve_scope(&[(HostMemoryClass::InputPreprocess, 64)])
        .unwrap();
    assert!(host
        .check_scope_admission(&[(HostMemoryClass::InputPreprocess, fit)])
        .unwrap_err()
        .is_temporary_capacity());
    drop(blocker);
    let request = host
        .reserve_scope(&[(HostMemoryClass::InputPreprocess, fit)])
        .unwrap();
    let bytes = request
        .account()
        .allocate_zeroed(HostMemoryClass::InputPreprocess, 16)
        .unwrap();
    drop((bytes, request, survivor));
    pipeline.shutdown().unwrap();
    drop((pipeline, recipe, stage));
    assert_eq!(runtime.snapshot().live_allocations, 0);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}
