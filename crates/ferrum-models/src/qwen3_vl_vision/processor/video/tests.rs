use super::super::tests::{account, architecture, limits, metadata};
use super::*;

fn processor() -> Qwen3VLVisionVideoProcessor {
    let mut raw = metadata();
    raw.as_object_mut().unwrap().remove("image_processor_type");
    raw["video_processor_type"] = "Qwen3VLVideoProcessor".into();
    Qwen3VLVisionVideoProcessor::from_source_bytes(&serde_json::to_vec(&raw).unwrap(), limits())
        .unwrap()
}

#[test]
fn video_pairs_keep_selected_frames_rgb_patch_order_and_repeat_only_the_final_frame() {
    let plan = processor()
        .prepare_video(architecture(), 8, 4, 7, &[0, 3, 6])
        .unwrap();
    assert_eq!(plan.config().grids()[0].temporal(), 2);
    assert_eq!(plan.resized_dimensions(), (8, 4));
    let host = account();
    let mut bytes = host
        .allocate_zeroed(HostMemoryClass::InputPreprocess, 3 * 7 * 4 * 8)
        .unwrap();
    let pixel = |c, t, y, x| ((c * 53 + t * 17 + y * 8 + x) % 256) as u8;
    for c in 0..3 {
        for t in 0..7 {
            for y in 0..4 {
                for x in 0..8 {
                    bytes.as_mut()[((c * 7 + t) * 4 + y) * 8 + x] = pixel(c, t, y, x);
                }
            }
        }
    }
    let pixels = ManagedHostTensor::new(
        ProgramValueId::new("test.video.rgb").unwrap(),
        plan.input.clone(),
        bytes.freeze(),
    )
    .unwrap();
    let before = host.usage().unwrap().committed_bytes();
    let output = plan.managed_patches(&host, &pixels).unwrap();
    let mut expected = Vec::new();
    // Independent nested permutation of the two original temporal groups.
    for pair in [[0, 3], [6, 6]] {
        for block_x in 0..2 {
            for merge_y in 0..2 {
                for merge_x in 0..2 {
                    for c in 0..3 {
                        for t in pair {
                            for dy in 0..2 {
                                for dx in 0..2 {
                                    let value = pixel(
                                        c,
                                        t,
                                        merge_y * 2 + dy,
                                        block_x * 4 + merge_x * 2 + dx,
                                    );
                                    expected.extend_from_slice(
                                        &bf16::from_f32((f32::from(value) - 127.5) / 127.5)
                                            .to_bits()
                                            .to_le_bytes(),
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    assert_eq!(output.tensor().dimensions(), [16, 24]);
    assert_eq!(output.bytes().bytes(), expected);
    assert!(plan.managed_patches(&account(), &pixels).is_err());
    drop(output);
    assert_eq!(host.usage().unwrap().committed_bytes(), before);
    drop(pixels);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn video_resize_uses_temporal_area_and_patch_capacity_is_checked_after_padding() {
    assert_eq!(resize(8, 8, 3, 4, 16, 128).unwrap(), (4, 4));
    assert_eq!(resize(8, 8, 5, 4, 16, 256).unwrap(), (8, 8));
    assert_eq!(resize(2, 1, 2, 4, 16, 256).unwrap(), (8, 4));
    let p = processor();
    assert!(p.prepare_video(architecture(), 8, 4, 7, &[0, 7]).is_err());
    assert!(p.prepare_video(architecture(), 8, 4, 7, &[3, 2]).is_err());
    let single = p.prepare_video(architecture(), 8, 4, 1, &[0]).unwrap();
    assert_eq!(single.config().grids()[0].temporal(), 1);
    let mut small = p;
    small.limits.maximum_patch_bytes = 16 * 24 * 2 - 1;
    assert!(small
        .prepare_video(architecture(), 8, 4, 7, &[0, 3, 6])
        .is_err());
}
