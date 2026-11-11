use super::*;

fn host() -> HostMemoryAccount {
    HostMemoryAccount::with_allocation_alignment(
        HostMemoryLimits::new(
            32 << 20,
            HostMemoryClassLimits {
                input_preprocess_bytes: 32 << 20,
                output_bytes: 0,
                transfer_staging_bytes: 0,
                weight_backing_bytes: 0,
            },
        )
        .unwrap(),
        64,
    )
    .unwrap()
}
fn rgba(host: &HostMemoryAccount, width: u32, height: u32, fill: [u8; 4]) -> ManagedRgba8Image {
    let mut bytes = host
        .allocate_zeroed(
            HostMemoryClass::InputPreprocess,
            width as usize * height as usize * 4,
        )
        .unwrap();
    for pixel in bytes.as_mut().chunks_exact_mut(4) {
        pixel.copy_from_slice(&fill);
    }
    ManagedRgba8Image::from_rgba8(
        host,
        bytes.freeze(),
        width,
        height,
        ImageInputLimits::default(),
    )
    .unwrap()
}
fn plan(w: u32, h: u32, gw: u32, gh: u32, cap: u64) -> PH3ImageEditPixelPlan {
    PH3ImageEditPixelPlan::new(
        w,
        h,
        gw,
        gh,
        PH3ImageEditPixelPolicy::new(cap).unwrap(),
        ImageInputLimits::default(),
    )
    .unwrap()
}
fn assert_channels(tensor: &ManagedHostTensor, expected: [u8; 3]) {
    assert_eq!(tensor.tensor().element_type(), ElementType::F32);
    let plane = tensor.bytes().len() / 3;
    for (channel, bytes) in tensor.bytes().bytes().chunks_exact(plane).enumerate() {
        let expected = f32::from(expected[channel]) / 255.0;
        for value in bytes.chunks_exact(4) {
            assert_eq!(
                f32::from_le_bytes(value.try_into().unwrap()).to_bits(),
                expected.to_bits()
            );
        }
    }
}

#[test]
fn outer_caps_are_independent_and_canvas_rounding_is_ties_even_without_upscale_to_area() {
    let p = plan(1920, 1080, 512, 512, 1_000_000);
    assert_eq!(p.vision_dimensions(), (1344, 736));
    assert_eq!(p.vae_dimensions(), (672, 384));
    let reference = p.reference_image(16).unwrap();
    assert_eq!(
        (reference.latent_width(), reference.latent_height()),
        (42, 24)
    );
    assert!(p.reference_image(5).is_err());
    let larger_target = plan(1920, 1080, 2048, 2048, 1_000_000);
    assert_eq!(larger_target.vision_dimensions(), p.vision_dimensions());
    assert_eq!(larger_target.vae_dimensions(), (1920, 1088));
    let ties = plan(48, 80, 512, 512, 1_000_000);
    assert_eq!(ties.vision_dimensions(), (64, 64));
    assert_eq!(ties.vae_dimensions(), (64, 64));
    assert_eq!(
        PH3ImageEditPixelPolicy::default().vision_maximum_pixels(),
        1_000_000
    );
    assert!(PH3ImageEditPixelPolicy::new(0).is_err());
    assert!(PH3ImageEditPixelPolicy::new((1 << 53) + 1).is_err());
    assert!(PH3ImageEditPixelPlan::new(
        32,
        32,
        u32::MAX,
        u32::MAX,
        PH3ImageEditPixelPolicy::default(),
        ImageInputLimits::default()
    )
    .is_err());
}

#[test]
fn transparent_rgb_survives_real_resize_and_f32_unit_conversion_at_exact_host_boundary() {
    let h = host();
    let image = rgba(&h, 64, 64, [17, 129, 233, 0]);
    let original = image.bytes().bytes().to_vec();
    let p = plan(64, 64, 32, 32, 1024);
    let r = p.host_requirements(&h, &image).unwrap();
    assert_eq!(
        r.input_peak_bytes,
        p.vision
            .input_peak_bytes(&h)
            .unwrap()
            .max(r.vision_new_input_bytes + r.vae_pixel_input_bytes)
    );
    let short = h
        .reserve_scope(&[(HostMemoryClass::InputPreprocess, r.input_peak_bytes - 1)])
        .unwrap();
    assert!(p.prepare(&short.account(), &image).is_err());
    assert_eq!(h.usage().unwrap().allocated_bytes(), r.decoded_input_bytes);
    drop(short);
    let exact = h
        .reserve_scope(&[(HostMemoryClass::InputPreprocess, r.input_peak_bytes)])
        .unwrap();
    let pixels = p.prepare(&exact.account(), &image).unwrap();
    assert_eq!(pixels.vision.width(), 32);
    assert_eq!(pixels.vision.height(), 32);
    assert!(pixels
        .vision
        .bytes()
        .bytes()
        .chunks_exact(4)
        .all(|v| v == [17, 129, 233, 255]));
    assert_channels(&pixels.vae, [17, 129, 233]);
    let value = f32::from(129u8) / 255.0;
    assert_ne!(
        value.to_bits() & 0xffff,
        0,
        "F32 input must retain its low bits"
    );
    assert_eq!(
        pixels.retained_owner_bytes().unwrap(),
        r.retained_input_bytes
    );
    assert_eq!(
        h.usage().unwrap().allocated_bytes(),
        r.decoded_input_bytes + r.retained_input_bytes
    );
    assert_eq!(image.bytes().bytes(), original);
    drop(exact);
    assert_eq!(
        h.usage().unwrap().committed_bytes(),
        r.decoded_input_bytes + r.retained_input_bytes
    );
    drop((pixels, image));
    assert_eq!(h.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn identity_keeps_actual_decoded_parent_owner_and_never_allocates_an_rgba_copy() {
    let h = host();
    let logical = 32 * 32 * 4;
    let mut parent = h
        .allocate_zeroed(HostMemoryClass::InputPreprocess, logical + 128)
        .unwrap();
    for pixel in parent.as_mut()[64..64 + logical].chunks_exact_mut(4) {
        pixel.copy_from_slice(&[7, 129, 251, 0]);
    }
    let image = ManagedRgba8Image::from_rgba8(
        &h,
        parent.freeze().slice(64..64 + logical).unwrap(),
        32,
        32,
        ImageInputLimits::default(),
    )
    .unwrap();
    let p = plan(32, 32, 64, 64, 1_000_000);
    let r = p.host_requirements(&h, &image).unwrap();
    assert_eq!(r.vision_new_input_bytes, 0);
    assert_eq!(r.vision_backing_bytes, image.bytes().backing_bytes());
    assert!(r.vision_backing_bytes > h.backing_bytes(logical).unwrap());
    assert_eq!(r.input_peak_bytes, r.vae_pixel_input_bytes);
    let output = p.prepare(&h, &image).unwrap();
    assert!(output.vision.bytes().same_region(image.bytes()));
    assert_eq!(output.vision.bytes().bytes()[3], 0);
    assert_channels(&output.vae, [7, 129, 251]);
    assert_eq!(
        output.retained_owner_bytes().unwrap(),
        r.vision_backing_bytes + r.vae_pixel_input_bytes
    );
    drop(image);
    assert_eq!(
        h.usage().unwrap().committed_bytes(),
        output.retained_owner_bytes().unwrap()
    );
    drop(output);
    assert_eq!(h.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn distinct_routes_read_the_original_source_and_reject_foreign_or_changed_geometry() {
    let h = host();
    let mut image = rgba(&h, 96, 64, [0, 0, 0, 0]);
    // A second immutable owner with nonconstant hidden RGB exercises both
    // independent original-source resizes, including two-axis intermediates.
    let mut backing = h
        .allocate_zeroed(HostMemoryClass::InputPreprocess, image.bytes().len())
        .unwrap();
    for (i, pixel) in backing.as_mut().chunks_exact_mut(4).enumerate() {
        pixel.copy_from_slice(&[
            (i * 7 % 256) as u8,
            (i * 13 % 256) as u8,
            (i * 19 % 256) as u8,
            0,
        ]);
    }
    image =
        ManagedRgba8Image::from_rgba8(&h, backing.freeze(), 96, 64, ImageInputLimits::default())
            .unwrap();
    let p = plan(96, 64, 32, 32, 2048);
    assert_eq!(p.vision_dimensions(), (64, 32));
    assert_eq!(p.vae_dimensions(), (32, 32));
    let r = p.host_requirements(&h, &image).unwrap();
    let scope = h
        .reserve_scope(&[(HostMemoryClass::InputPreprocess, r.input_peak_bytes)])
        .unwrap();
    let output = p.prepare(&scope.account(), &image).unwrap();
    assert_eq!((output.vision.width(), output.vision.height()), (64, 32));
    assert_eq!(output.vae.tensor().dimensions(), [1, 3, 32, 32]);
    assert_eq!(
        output.retained_owner_bytes().unwrap(),
        r.retained_input_bytes
    );
    let direct = p.vae.resize_rgb_ignoring_alpha(&h, &image).unwrap();
    let plane = 32 * 32;
    for (index, pixel) in direct.bytes().bytes().chunks_exact(4).enumerate() {
        for channel in 0..3 {
            let offset = (channel * plane + index) * 4;
            assert_eq!(
                &output.vae.bytes().bytes()[offset..offset + 4],
                &(f32::from(pixel[channel]) / 255.0).to_le_bytes()
            );
        }
    }
    assert!(p.prepare(&host(), &image).is_err());
    assert!(plan(64, 96, 32, 32, 2048).prepare(&h, &image).is_err());
}
