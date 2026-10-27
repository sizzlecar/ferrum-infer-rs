use super::*;
use crate::qwen_image21::image_input::tests::{host, limits, managed};
use serde_json::json;

fn config() -> serde_json::Value {
    json!({"processor_class":"Qwen3VLProcessor", "image_processor": {
        "image_processor_type":"Qwen2VLImageProcessorFast", "data_format":"channels_first",
        "do_convert_rgb":true,"do_normalize":true,"do_rescale":true,"do_resize":true,
        "image_mean":[0.,0.,0.],"image_std":[1.,1.,1.],"rescale_factor":1.,
        "patch_size":1,"merge_size":2,"temporal_patch_size":2,"resample":3,
        "size":{"shortest_edge":1,"longest_edge":1048576}
    }})
}

fn processor() -> QwenImage21ImageProcessor {
    QwenImage21ImageProcessor::from_json(&serde_json::to_vec(&config()).unwrap()).unwrap()
}

fn values(tensor: &ManagedHostTensor) -> Vec<f32> {
    tensor
        .bytes()
        .bytes()
        .chunks_exact(2)
        .map(|b| bf16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32())
        .collect()
}

#[test]
fn rectangular_patch_rows_merge_axes_and_repeat_temporal_channels() {
    let host = host(4096);
    let mut rgba = Vec::new();
    for index in 0..8u8 {
        rgba.extend_from_slice(&[index, 10 + index, 20 + index, 255]);
    }
    let image =
        ManagedRgba8Image::from_rgba8(&host, managed(&host, &rgba), 4, 2, limits()).unwrap();
    let prepared = processor().prepare_resized(&host, &image).unwrap();
    assert_eq!(prepared.grid_thw(), [1, 2, 4]);
    assert_eq!(prepared.vision_patches().tensor().dimensions(), &[8, 6]);
    assert_eq!(prepared.vae_rgba().tensor().dimensions(), &[1, 4, 2, 4]);
    let patches = values(prepared.vision_patches());
    // Processor's merge-block-major order, not ordinary raster rows.
    for (row, pixel) in [0u8, 1, 4, 5, 2, 3, 6, 7].into_iter().enumerate() {
        assert_eq!(
            &patches[row * 6..row * 6 + 6],
            &[
                f32::from(pixel),
                f32::from(pixel),
                f32::from(10 + pixel),
                f32::from(10 + pixel),
                f32::from(20 + pixel),
                f32::from(20 + pixel),
            ]
        );
    }
    assert!(host.owns(prepared.vision_patches().bytes()));
    assert!(host.owns(prepared.vae_rgba().bytes()));
    assert_eq!(host.usage().unwrap().allocated_bytes(), 32 + 96 + 64);
    let candidate = prepared.clone();
    assert!(candidate
        .vision_patches()
        .bytes()
        .same_region(prepared.vision_patches().bytes()));
    assert!(candidate
        .vae_rgba()
        .bytes()
        .same_region(prepared.vae_rgba().bytes()));
    assert_eq!(host.usage().unwrap().allocated_bytes(), 32 + 96 + 64);
    drop(prepared);
    assert_eq!(host.usage().unwrap().committed_bytes(), 32 + 96 + 64);
    drop(candidate);
    assert_eq!(host.usage().unwrap().committed_bytes(), 32);
    assert_eq!(image.bytes().bytes(), rgba);
    drop(image);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn white_alpha_and_rgba_vae_have_independent_normalization_boundaries() {
    let host = host(4096);
    let rgba = [
        0, 255, 128, 0, 0, 0, 0, 128, 255, 255, 255, 255, 0, 255, 128, 255,
    ];
    let image =
        ManagedRgba8Image::from_rgba8(&host, managed(&host, &rgba), 2, 2, limits()).unwrap();
    let output = processor().prepare_resized(&host, &image).unwrap();
    let patches = values(output.vision_patches());
    assert_eq!(&patches[..6], &[255.; 6]);
    assert_eq!(&patches[6..12], &[127.; 6]);
    let vae = output
        .vae_rgba()
        .bytes()
        .bytes()
        .chunks_exact(2)
        .map(|x| u16::from_le_bytes([x[0], x[1]]))
        .collect::<Vec<_>>();
    // NCHW; invisible source RGB is deliberately retained for the VAE.
    assert_eq!(
        vae,
        [
            0xbf80, 0xbf80, 0x3f80, 0xbf80, 0x3f80, 0xbf80, 0x3f80, 0x3f80, 0x3b81, 0xbf80, 0x3f80,
            0x3b81, 0xbf80, 0x3b81, 0x3f80, 0x3f80
        ]
    );
    let mut metadata = config();
    metadata["image_processor"]["rescale_factor"] = json!(1.0 / 255.0);
    metadata["image_processor"]["image_mean"] = json!([0.5, 0.5, 0.5]);
    metadata["image_processor"]["image_std"] = json!([0.5, 0.5, 0.5]);
    let processor =
        QwenImage21ImageProcessor::from_json(&serde_json::to_vec(&metadata).unwrap()).unwrap();
    assert_eq!(processor.fused_mean, [127.5; 3]);
    assert_eq!(processor.fused_std, [127.5; 3]);
    let output = processor.prepare_resized(&host, &image).unwrap();
    assert_eq!(&values(output.vision_patches())[..6], &[1.; 6]);
}

#[test]
fn invalid_metadata_geometry_and_atomic_capacity_do_not_publish_partial_tensors() {
    for (field, value) in [
        ("do_resize", json!(false)),
        ("resample", json!(2)),
        ("patch_size", json!(0)),
        ("merge_size", json!(0)),
        ("temporal_patch_size", json!(0)),
        ("image_std", json!([1., 0., 1.])),
        ("rescale_factor", json!(0.)),
    ] {
        let mut bad = config();
        bad["image_processor"][field] = value;
        assert!(QwenImage21ImageProcessor::from_json(&serde_json::to_vec(&bad).unwrap()).is_err());
    }
    let wire = serde_json::to_string(&config())
        .unwrap()
        .replace("\"merge_size\":2", "\"merge_size\":2,\"merge_size\":2");
    assert!(QwenImage21ImageProcessor::from_json(wire.as_bytes()).is_err());
    // Source 16 + patches 48 + VAE 32. Either output alone fits; both do not.
    let account = host(95);
    let image =
        ManagedRgba8Image::from_rgba8(&account, managed(&account, &[0; 16]), 2, 2, limits())
            .unwrap();
    assert!(matches!(
        processor().prepare_resized(&account, &image),
        Err(ImageInputError::Host(HostMemoryError::Capacity { .. }))
    ));
    assert_eq!(account.usage().unwrap().allocated_bytes(), 16);
    assert_eq!(account.usage().unwrap().reserved_bytes(), 0);
    let foreign = host(4096);
    assert!(processor().prepare_resized(&foreign, &image).is_err());
    assert_eq!(foreign.usage().unwrap().committed_bytes(), 0);
    let odd = ManagedRgba8Image::from_rgba8(&foreign, managed(&foreign, &[0; 12]), 3, 1, limits())
        .unwrap();
    assert!(processor().prepare_resized(&foreign, &odd).is_err());
}

#[test]
fn target_geometry_is_area_based_axis_aligned_and_not_output_resolution() {
    assert_eq!(
        QwenImage21ImageProcessor::condition_dimensions(1024, 1024).unwrap(),
        (1024, 1024)
    );
    assert_eq!(
        QwenImage21ImageProcessor::condition_dimensions(2048, 2048).unwrap(),
        (1024, 1024)
    );
    assert_eq!(
        QwenImage21ImageProcessor::condition_dimensions(2048, 512).unwrap(),
        (2048, 512)
    );
    assert!(QwenImage21ImageProcessor::condition_dimensions(0, 1).is_err());
    assert!(QwenImage21ImageProcessor::condition_dimensions(201, 1).is_err());
}

#[test]
fn retained_backing_prediction_matches_preparation_with_resize_and_alignment() {
    let processor = processor();
    for (width, height, alignment) in [(32, 32, 1), (64, 16, 4096)] {
        let host = HostMemoryAccount::with_allocation_alignment(
            host(64 * 1024 * 1024).limits(),
            alignment,
        )
        .unwrap();
        let predicted = processor
            .prepared_backing_bytes(&host, width, height)
            .unwrap();
        assert_eq!(host.usage().unwrap().committed_bytes(), 0);
        let source = host
            .allocate_zeroed(
                HostMemoryClass::InputPreprocess,
                width as usize * height as usize * 4,
            )
            .unwrap()
            .freeze();
        let image = ManagedRgba8Image::from_rgba8(&host, source, width, height, limits()).unwrap();
        let before = host.usage().unwrap().committed_bytes();
        let output = processor.prepare(&host, &image, limits()).unwrap();
        assert_eq!(host.usage().unwrap().allocated_bytes() - before, predicted);
        assert_eq!(host.usage().unwrap().committed_bytes() - before, predicted);
        assert_eq!(
            (output.width(), output.height()),
            QwenImage21ImageProcessor::condition_dimensions(width, height).unwrap()
        );
        let logical = output.vision_patches().bytes().len() + output.vae_rgba().bytes().len();
        if alignment > 1 {
            assert!(
                predicted > logical,
                "physical accounting includes alignment padding"
            );
        } else {
            assert_eq!(predicted, logical);
        }
        drop(output);
        assert_eq!(host.usage().unwrap().committed_bytes(), before);
    }
}

#[test]
fn retained_backing_prediction_is_metadata_only_and_preserves_geometry_rejections() {
    let host = host(1);
    let processor = processor();
    assert!(processor.prepared_backing_bytes(&host, 32, 32).unwrap() > host.limits().total_bytes());
    for (width, height) in [(0, 32), (201, 1)] {
        assert!(processor
            .prepared_backing_bytes(&host, width, height)
            .is_err());
    }
    let mut metadata = config();
    metadata["image_processor"]["size"]["longest_edge"] = json!(1024);
    let bounded =
        QwenImage21ImageProcessor::from_json(&serde_json::to_vec(&metadata).unwrap()).unwrap();
    assert!(matches!(
        bounded.prepared_backing_bytes(&host, 32, 32),
        Err(ImageInputError::Unsupported(
            "processor requires an additional bicubic resize"
        ))
    ));
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}
