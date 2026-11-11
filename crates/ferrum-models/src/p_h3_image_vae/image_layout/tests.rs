use super::*;
use ferrum_runtime::{HostMemoryClassLimits, HostMemoryLimits};
use serde_json::json;

fn layout(batch: usize, height: usize, width: usize) -> PH3ImageVaeImageLayout {
    let config = PH3ImageVaeDecoderConfig::from_hf_json(
        &serde_json::to_vec(&json!({
            "_class_name":"AutoencoderKLPH3Image", "latent_channels":2,"out_channels":3,
            "spatial_downsample_factors":[2],"decoder_num_layers":2,"decoder_num_attention_heads":2,
            "decoder_attention_head_dim":16,"decoder_num_register_tokens":4,"decoder_ffn_mult":4,
            "decoder_rope_dim_ratio":0.75,"decoder_rope_theta":100,"decoder_norm_eps":0.00001
        }))
        .unwrap(),
    )
    .unwrap();
    PH3ImageVaeImageLayout::new(config, batch, height, width).unwrap()
}
fn host(bytes: usize) -> HostMemoryAccount {
    HostMemoryAccount::new(
        HostMemoryLimits::new(
            bytes,
            HostMemoryClassLimits {
                input_preprocess_bytes: bytes,
                output_bytes: bytes,
                transfer_staging_bytes: 0,
                weight_backing_bytes: 0,
            },
        )
        .unwrap(),
    )
}
fn id(name: &str) -> ProgramValueId {
    ProgramValueId::new(name).unwrap()
}
fn bytes(host: &HostMemoryAccount, class: HostMemoryClass, values: &[f32]) -> ManagedHostBytes {
    let mut b = host.allocate_zeroed(class, values.len() * 4).unwrap();
    for (destination, value) in b.as_mut().chunks_exact_mut(4).zip(values) {
        destination.copy_from_slice(&value.to_le_bytes());
    }
    b.freeze()
}
fn values(b: &ManagedHostBytes) -> Vec<f32> {
    b.bytes()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

#[test]
fn latent_transpose_preserves_request_channel_order_and_every_f32_bit() {
    let layout = layout(2, 1, 3);
    let host = host(512);
    let source = [
        1.0,
        2.0,
        -0.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
        11.0,
        f32::from_bits(0x7fa12345),
    ];
    let input = ManagedHostTensor::new(
        id("value.latents"),
        layout.latent_spec().unwrap(),
        bytes(&host, HostMemoryClass::InputPreprocess, &source),
    )
    .unwrap();
    let packed = layout
        .pack_denormalized_latents(&host, &input, id("value.rows"))
        .unwrap();
    let expected = [
        source[0], source[3], source[1], source[4], source[2], source[5], source[6], source[9],
        source[7], source[10], source[8], source[11],
    ];
    assert_eq!(
        values(packed.bytes())
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        expected.map(f32::to_bits)
    );
    let other = super::tests::host(512);
    assert!(layout
        .pack_denormalized_latents(&other, &input, id("value.rows"))
        .is_err());
    assert_eq!(other.usage().unwrap().committed_bytes(), 0);
    drop((input, packed));
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn rotary_matches_independent_trig_with_fixed_image_time_and_identity_suffix() {
    let layout = layout(2, 2, 3);
    let host = host(4096);
    let (cos, sin) = layout
        .managed_rotary(&host, id("value.cos"), id("value.sin"))
        .unwrap();
    let c = values(cos.bytes());
    let s = values(sin.bytes());
    assert_eq!(cos.tensor().dimensions(), [22, 12]);
    // Independent double-precision angles; this tolerance covers the declared
    // F32 angle pipeline, not an unmeasured Torch kernel equivalence claim.
    for row in 0..6 {
        let y = if row < 3 { -0.5 } else { 0.5 };
        let x = [-2.0 / 3.0, 0.0, 2.0 / 3.0][row % 3];
        for (axis, coordinate) in [-6.0 / 7.0, y, x].into_iter().enumerate() {
            for (f, inv) in [1.0, 0.1].into_iter().enumerate() {
                let angle = std::f64::consts::TAU * coordinate * inv;
                for batch in 0..2 {
                    for half in [0, 6] {
                        let index = (batch * 11 + row) * 12 + axis * 2 + f + half;
                        assert!((f64::from(c[index]) - libm::cos(angle)).abs() < 7e-7);
                        assert!((f64::from(s[index]) - libm::sin(angle)).abs() < 7e-7);
                    }
                }
            }
        }
    }
    for batch in 0..2 {
        for row in 6..11 {
            assert!(c[(batch * 11 + row) * 12..(batch * 11 + row + 1) * 12]
                .iter()
                .all(|v| v.to_bits() == 1.0_f32.to_bits()));
            assert!(s[(batch * 11 + row) * 12..(batch * 11 + row + 1) * 12]
                .iter()
                .all(|v| v.to_bits() == 0.0_f32.to_bits()));
        }
    }
    drop((cos, sin));
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn rotary_second_allocation_failure_and_invalid_geometry_retain_nothing() {
    let layout = layout(2, 2, 3);
    let bytes = extent(&layout.frequency_spec().unwrap()).unwrap();
    let host = host(bytes * 2 - 1);
    assert!(layout
        .managed_rotary(&host, id("value.cos"), id("value.sin"))
        .is_err());
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    assert!(layout
        .managed_rotary(&host, id("value.same"), id("value.same"))
        .is_err());
    assert!(PH3ImageVaeImageLayout::new(layout.config.clone(), 0, 1, 1).is_err());
    assert!(PH3ImageVaeImageLayout::new(layout.config, usize::MAX, 1, 1).is_err());
}

#[test]
fn unpatchify_preserves_color_and_request_order_and_discards_poisoned_suffix() {
    let layout = layout(2, 1, 2);
    let host = host(4096);
    let mut source = vec![f32::NAN; 2 * 7 * 12];
    for b in 0..2 {
        // Each patch stores channel-major local 2x2 pixels, as in the source
        // unpatchify view. Distinct second request detects batch mixing.
        for p in 0..2 {
            for c in 0..3 {
                let base = (b * 1000 + p * 100 + c * 10) as f32;
                source[(b * 7 + p) * 12 + c * 4..(b * 7 + p) * 12 + c * 4 + 4].copy_from_slice(&[
                    base + 1.0,
                    base + 2.0,
                    base + 3.0,
                    base + 4.0,
                ]);
            }
        }
    }
    let input = bytes(&host, HostMemoryClass::Output, &source);
    let (spec, output) = layout
        .unpatchify(&host, &layout.patch_output_spec().unwrap(), &input)
        .unwrap();
    assert_eq!(spec.dimensions(), [2, 3, 2, 4]);
    let v = values(&output);
    for b in 0..2 {
        for c in 0..3 {
            let base = (b * 1000 + c * 10) as f32;
            assert_eq!(
                &v[(b * 3 + c) * 8..(b * 3 + c + 1) * 8],
                &[
                    base + 1.0,
                    base + 2.0,
                    base + 101.0,
                    base + 102.0,
                    base + 3.0,
                    base + 4.0,
                    base + 103.0,
                    base + 104.0
                ]
            );
        }
    }
    let before = host.usage().unwrap().committed_bytes();
    let invalid = bytes(
        &host,
        HostMemoryClass::Output,
        &vec![f32::NAN; source.len()],
    );
    let invalid_size = host.usage().unwrap().committed_bytes() - before;
    assert!(layout
        .unpatchify(&host, &layout.patch_output_spec().unwrap(), &invalid)
        .is_err());
    assert_eq!(
        host.usage().unwrap().committed_bytes(),
        before + invalid_size
    );
    drop((invalid, input, output));
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn rgb_finalization_has_channel_specific_affine_clamp_and_keeps_output_charge() {
    let layout = layout(1, 1, 1);
    let host = host(4096);
    let mut source = vec![f32::NAN; 6 * 12];
    source[..12].copy_from_slice(&[
        -10.0, 0.0, 1.0, 10.0, -10.0, 0.0, 1.0, 10.0, -10.0, 0.0, 1.0, 10.0,
    ]);
    let input = bytes(&host, HostMemoryClass::Output, &source);
    let input_charge = host.usage().unwrap().committed_bytes();
    let (spec, out) = layout
        .unpatchify_rgb(&host, &layout.patch_output_spec().unwrap(), &input)
        .unwrap();
    assert_eq!(spec.dimensions(), [1, 3, 2, 2]);
    let v = values(&out);
    for (c, (mean, std)) in [(0.485_f32, 0.229_f32), (0.456, 0.224), (0.406, 0.225)]
        .into_iter()
        .enumerate()
    {
        assert_eq!(v[c * 4], 0.0);
        assert_eq!(v[c * 4 + 1], mean);
        assert_eq!(v[c * 4 + 2], std + mean);
        assert_eq!(v[c * 4 + 3], 1.0);
    }
    let output_charge = host.usage().unwrap().committed_bytes() - input_charge;
    drop(input);
    assert_eq!(host.usage().unwrap().committed_bytes(), output_charge);
    drop(out);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}
