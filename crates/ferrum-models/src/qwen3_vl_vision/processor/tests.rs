use super::*;
pub(super) fn limits() -> Qwen3VLVisionProcessorLimits {
    Qwen3VLVisionProcessorLimits {
        maximum_metadata_bytes: 4096,
        image: ImageInputLimits::new(4096, 256, 32, 32, 4096).unwrap(),
        maximum_patch_rows: 64,
        maximum_patch_bytes: 4096,
    }
}
pub(super) fn metadata() -> serde_json::Value {
    serde_json::json!({"processor_class":"Qwen3VLProcessor","image_processor_type":"Qwen2VLImageProcessorFast",
        "patch_size":2,"temporal_patch_size":2,"merge_size":2,
        "image_mean":[0.5,0.5,0.5],"image_std":[0.5,0.5,0.5],
        "size":{"shortest_edge":16,"longest_edge":256}})
}
pub(super) fn architecture() -> Qwen3VLVisionArchitecture {
    Qwen3VLVisionArchitecture::from_hf_value(&serde_json::json!({"model_type":"qwen3_vl",
        "architectures":["Qwen3VLForConditionalGeneration"],"vision_config":{
        "model_type":"qwen3_vl","dtype":"bfloat16","hidden_act":"gelu_pytorch_tanh",
        "depth":3,"hidden_size":8,"intermediate_size":12,"num_heads":2,
        "num_position_embeddings":16,"out_hidden_size":10,"deepstack_visual_indexes":[0,1,2],
        "in_channels":3,"patch_size":2,"temporal_patch_size":2,"spatial_merge_size":2}}))
    .unwrap()
}
pub(super) fn account() -> HostMemoryAccount {
    let bytes = 1 << 18;
    HostMemoryAccount::new(
        HostMemoryLimits::new(
            bytes,
            HostMemoryClassLimits {
                input_preprocess_bytes: bytes,
                transfer_staging_bytes: bytes,
                output_bytes: bytes,
                weight_backing_bytes: bytes,
            },
        )
        .unwrap(),
    )
}
fn image(host: &HostMemoryAccount, w: u32, h: u32, constant: bool) -> ManagedRgba8Image {
    let mut bytes = host
        .allocate_zeroed(HostMemoryClass::InputPreprocess, (w * h * 4) as usize)
        .unwrap();
    for (i, pixel) in bytes.as_mut().chunks_exact_mut(4).enumerate() {
        pixel.copy_from_slice(&if constant {
            [31, 127, 241, 0]
        } else {
            [i as u8, 255 - i as u8, (i * 3) as u8, 0]
        });
    }
    ManagedRgba8Image::from_rgba8(host, bytes.freeze(), w, h, limits().image).unwrap()
}
#[test]
fn original_fast_processor_preserves_deepstack_and_exact_patch_channel_temporal_order() {
    let p = Qwen3VLVisionImageProcessor::from_source_bytes(
        &serde_json::to_vec(&metadata()).unwrap(),
        limits(),
    )
    .unwrap();
    let plan = p.prepare_image(architecture(), 8, 4).unwrap();
    assert_eq!(plan.config().architecture().deepstack_layers(), [0, 1, 2]);
    let host = account();
    let input = image(&host, 8, 4, false);
    let output = plan.managed_patches(&host, &input).unwrap();
    assert_eq!(output.tensor().dimensions(), [8, 24]);
    let mut expected = Vec::new();
    // Independent nested processor permutation, not the flattened row formula.
    for block_x in 0..2 {
        for merge_y in 0..2 {
            for merge_x in 0..2 {
                for channel in 0..3 {
                    for _frame in 0..2 {
                        for dy in 0..2 {
                            for dx in 0..2 {
                                let x = block_x * 4 + merge_x * 2 + dx;
                                let y = merge_y * 2 + dy;
                                let pixel = input.bytes().bytes()[(y * 8 + x) * 4 + channel];
                                expected.extend_from_slice(
                                    &bf16::from_f32((f32::from(pixel) - 127.5) / 127.5)
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
    assert_eq!(output.bytes().bytes(), expected);
    assert!(plan.managed_patches(&account(), &input).is_err());
    drop((input, output));
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}
#[test]
fn processor_resize_preserves_constant_rgb_and_rejects_changed_policy_or_capacity() {
    let raw = metadata();
    let p = Qwen3VLVisionImageProcessor::from_source_bytes(
        &serde_json::to_vec(&raw).unwrap(),
        limits(),
    )
    .unwrap();
    let plan = p.prepare_image(architecture(), 5, 3).unwrap();
    assert_eq!(plan.resized_dimensions(), (4, 4));
    let host = account();
    let input = image(&host, 5, 3, true);
    let output = plan.managed_patches(&host, &input).unwrap();
    for row in output.bytes().bytes().chunks_exact(24 * 2) {
        for (i, bytes) in row.chunks_exact(2).enumerate() {
            let pixel = [31u8, 127, 241][i / 8];
            assert_eq!(
                u16::from_le_bytes(bytes.try_into().unwrap()),
                bf16::from_f32((f32::from(pixel) - 127.5) / 127.5).to_bits()
            );
        }
    }
    let mut bad = raw.clone();
    bad["do_resize"] = false.into();
    assert!(Qwen3VLVisionImageProcessor::from_source_bytes(
        &serde_json::to_vec(&bad).unwrap(),
        limits()
    )
    .is_err());
    let mut small = limits();
    small.maximum_patch_bytes = 1;
    let p =
        Qwen3VLVisionImageProcessor::from_source_bytes(&serde_json::to_vec(&raw).unwrap(), small)
            .unwrap();
    assert!(p.prepare_image(architecture(), 8, 4).is_err());
}
