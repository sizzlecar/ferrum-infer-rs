use super::*;
use crate::qwen_image21::image_input::ImageInputLimits;

fn architecture() -> Qwen3VLVisionArchitecture {
    Qwen3VLVisionArchitecture::from_hf_value(&serde_json::json!({
        "architectures": ["Qwen3VLForConditionalGeneration"], "model_type": "qwen3_vl",
        "vision_config": {"model_type":"qwen3_vl_vision", "dtype":"bfloat16",
            "hidden_act":"gelu_pytorch_tanh", "depth":3, "hidden_size":8,
            "intermediate_size":12, "num_heads":2, "num_position_embeddings":16,
            "out_hidden_size":256, "deepstack_visual_indexes":[0,1,2],
            "in_channels":3, "patch_size":2, "temporal_patch_size":2,
            "spatial_merge_size":2,
            "rope_parameters":{"rope_type":"axial","rope_theta":10000.0}}
    }))
    .unwrap()
}
fn host() -> HostMemoryAccount {
    HostMemoryAccount::new(
        HostMemoryLimits::new(
            1 << 20,
            HostMemoryClassLimits {
                input_preprocess_bytes: 1 << 20,
                output_bytes: 1 << 20,
                transfer_staging_bytes: 1 << 20,
                weight_backing_bytes: 1 << 20,
            },
        )
        .unwrap(),
    )
}
fn image(host: &HostMemoryAccount, w: u32, h: u32, alpha: u8) -> ManagedRgba8Image {
    let mut bytes = host
        .allocate_zeroed(HostMemoryClass::InputPreprocess, (w * h * 4) as usize)
        .unwrap();
    for (i, px) in bytes.as_mut().chunks_exact_mut(4).enumerate() {
        px.copy_from_slice(&[
            (i * 47 % 256) as u8,
            (i * 13 + 151) as u8,
            (i * 79 + 23) as u8,
            alpha,
        ]);
    }
    ManagedRgba8Image::from_rgba8(host, bytes.freeze(), w, h, ImageInputLimits::default()).unwrap()
}

#[test]
fn resize_uses_ties_even_and_original_min_max_factor_rounding() {
    let plan = |w, h, min, max| {
        Qwen3VLVisionComfyImagePlan::new(
            architecture(),
            w,
            h,
            Qwen3VLVisionComfyPixelPolicy::new(min, max).unwrap(),
        )
        .unwrap()
    };
    // 1.5 rounds up, 2.5 down. Both are exact half-factor ties.
    assert_eq!(plan(6, 10, 1, 1024).resized_dimensions(), (8, 8));
    assert_eq!(plan(1, 2, 16, 1024).resized_dimensions(), (4, 8));
    assert_eq!(plan(12, 20, 1, 64).resized_dimensions(), (4, 8));
    assert!(Qwen3VLVisionComfyPixelPolicy::new(0, 1024).is_err());
    assert!(Qwen3VLVisionComfyPixelPolicy::new(1024, 1023).is_err());
    assert!(Qwen3VLVisionComfyImagePlan::new(
        architecture(),
        0,
        8,
        Qwen3VLVisionComfyPixelPolicy::default()
    )
    .is_err());
}

/// Independent four-neighbour F64 image oracle, in planar H/W order. The
/// production path emits directly into patches with F32 products and sums.
fn oracle(image: &ManagedRgba8Image, out_w: usize, out_h: usize) -> Vec<f64> {
    let w = image.width() as usize;
    let h = image.height() as usize;
    let mut expected = vec![0.; 3 * out_h * out_w];
    for c in 0..3 {
        for y in 0..out_h {
            for x in 0..out_w {
                let py = ((y as f64 + 0.5) * h as f64 / out_h as f64 - 0.5).max(0.);
                let px = ((x as f64 + 0.5) * w as f64 / out_w as f64 - 0.5).max(0.);
                let dy = py - py.floor();
                let dx = px - px.floor();
                let mut rgb = 0.;
                for (iy, wy) in [(0, 1. - dy), (1, dy)] {
                    for (ix, wx) in [(0, 1. - dx), (1, dx)] {
                        let sy = (py as usize + iy).min(h - 1);
                        let sx = (px as usize + ix).min(w - 1);
                        let input = image.bytes().bytes()[(sy * w + sx) * 4 + c];
                        rgb += f64::from(input) / 255. * wy * wx;
                    }
                }
                expected[(c * out_h + y) * out_w + x] = (rgb - 0.5) / 0.5;
            }
        }
    }
    expected
}

#[test]
fn patches_match_independent_resize_oracle_merge_order_and_repeat_frames_ignoring_alpha() {
    let host = host();
    for (w, h) in [(8, 8), (7, 5), (1, 2)] {
        let source = image(&host, w, h, 0);
        let opaque = image(&host, w, h, 255);
        let plan = Qwen3VLVisionComfyImagePlan::new(
            architecture(),
            w,
            h,
            Qwen3VLVisionComfyPixelPolicy::new(16, 1024).unwrap(),
        )
        .unwrap();
        let patches = plan.managed_patches(&host, &source).unwrap();
        let other = plan.managed_patches(&host, &opaque).unwrap();
        assert_eq!(patches.bytes().bytes(), other.bytes().bytes());
        let (ow, oh) = plan.resized_dimensions();
        let expected = oracle(&source, ow, oh);
        let mut index = 0;
        // Explicit original transpose: block_y, block_x, merge_y, merge_x,
        // channel, temporal, patch_y, patch_x. No production row formula.
        for by in 0..oh / 4 {
            for bx in 0..ow / 4 {
                for my in 0..2 {
                    for mx in 0..2 {
                        for c in 0..3 {
                            for _t in 0..2 {
                                for py in 0..2 {
                                    for px in 0..2 {
                                        let sy = by * 4 + my * 2 + py;
                                        let sx = bx * 4 + mx * 2 + px;
                                        let e = expected[(c * oh + sy) * ow + sx];
                                        let actual = f32::from_le_bytes(
                                            patches.bytes().bytes()[index * 4..index * 4 + 4]
                                                .try_into()
                                                .unwrap(),
                                        );
                                        assert!(
                                            (f64::from(actual) - e).abs() < 4e-7,
                                            "source={w}x{h}, column={index}, {actual} != {e}"
                                        );
                                        index += 1;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        assert_eq!(index * 4, patches.bytes().len());
        assert_eq!(
            patches.tensor().dimensions(),
            &[plan.config().rows().unwrap(), 24]
        );
    }
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn exact_original_host_scope_is_required_and_failed_preprocess_keeps_input_retryable() {
    let host = host();
    let source = image(&host, 7, 5, 7);
    let original = source.bytes().bytes().to_vec();
    let plan = Qwen3VLVisionComfyImagePlan::new(
        architecture(),
        7,
        5,
        Qwen3VLVisionComfyPixelPolicy::new(16, 1024).unwrap(),
    )
    .unwrap();
    let baseline = host.usage().unwrap();
    let bytes = plan.input_backing_bytes(&host).unwrap();
    let short = host
        .reserve_scope(&[(HostMemoryClass::InputPreprocess, bytes - 1)])
        .unwrap();
    assert!(plan.managed_patches(&short.account(), &source).is_err());
    drop(short);
    assert_eq!(host.usage().unwrap(), baseline);
    let exact = host
        .reserve_scope(&[(HostMemoryClass::InputPreprocess, bytes)])
        .unwrap();
    let patches = plan.managed_patches(&exact.account(), &source).unwrap();
    assert_eq!(
        host.usage().unwrap().allocated_bytes(),
        baseline.allocated_bytes() + bytes
    );
    assert!(plan.managed_patches(&self::host(), &source).is_err());
    assert!(plan.managed_patches(&host, &image(&host, 8, 4, 7)).is_err());
    drop(exact);
    assert_eq!(
        host.usage().unwrap().allocated_bytes(),
        baseline.allocated_bytes() + bytes
    );
    assert_eq!(source.bytes().bytes(), original);
    drop(patches);
    assert_eq!(host.usage().unwrap(), baseline);
    drop(source);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}
