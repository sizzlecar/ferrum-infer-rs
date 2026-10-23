use super::*;
use crate::qwen_image21::image_input::tests::{host, limits, managed};

#[test]
fn identity_preserves_hidden_rgb_and_owner_without_allocating() {
    let account = host(8);
    let image = ManagedRgba8Image::from_rgba8(
        &account,
        managed(&account, &[17, 31, 99, 0, 200, 80, 40, 128]),
        2,
        1,
        limits(),
    )
    .unwrap();
    let resized = lanczos(&account, &image, 2, 1, limits()).unwrap();
    assert_eq!(resized.bytes().bytes(), image.bytes().bytes());
    assert_eq!(account.usage().unwrap().allocated_bytes(), 8);
    drop(image);
    assert_eq!(resized.bytes().bytes()[0], 17);
    drop(resized);
    assert_eq!(account.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn premultiplied_constant_and_symmetric_average_have_hand_computed_pixels() {
    let account = host(65536);
    let one = ManagedRgba8Image::from_rgba8(
        &account,
        managed(&account, &[200, 80, 40, 128]),
        1,
        1,
        limits(),
    )
    .unwrap();
    let grown = lanczos(&account, &one, 3, 2, limits()).unwrap();
    // Bounded integer alpha multiplication: [100,40,20]/128, then the
    // unpremultiply floor gives [199,79,39]. Both spatial passes retain it.
    assert!(grown
        .bytes()
        .bytes()
        .chunks_exact(4)
        .all(|x| x == [199, 79, 39, 128]));
    let pair = ManagedRgba8Image::from_rgba8(
        &account,
        managed(&account, &[0, 20, 100, 255, 1, 30, 200, 255]),
        2,
        1,
        limits(),
    )
    .unwrap();
    let reduced = lanczos(&account, &pair, 1, 1, limits()).unwrap();
    assert_eq!(reduced.bytes().bytes(), [1, 25, 150, 255]);
    // Vertical-only uses the same symmetric kernel with independent strides.
    let vertical = ManagedRgba8Image::from_rgba8(
        &account,
        managed(&account, pair.bytes().bytes()),
        1,
        2,
        limits(),
    )
    .unwrap();
    assert_eq!(
        lanczos(&account, &vertical, 1, 1, limits())
            .unwrap()
            .bytes()
            .bytes(),
        [1, 25, 150, 255]
    );
}

#[test]
fn two_axis_order_and_capacity_failure_keep_source_and_release_temporaries() {
    let account = host(65536);
    for (iw, ih, ow, oh) in [(2, 8, 3, 2), (8, 2, 2, 3)] {
        let bytes = [40, 80, 120, 255].repeat(iw * ih);
        let source = ManagedRgba8Image::from_rgba8(
            &account,
            managed(&account, &bytes),
            iw as u32,
            ih as u32,
            limits(),
        )
        .unwrap();
        let resized = lanczos(&account, &source, ow, oh, limits()).unwrap();
        assert_eq!(
            resized.bytes().bytes(),
            [40, 80, 120, 255].repeat((ow * oh) as usize)
        );
        assert_eq!(
            account.usage().unwrap().allocated_bytes(),
            bytes.len() + (ow * oh * 4) as usize
        );
    }
    assert_eq!(account.usage().unwrap().committed_bytes(), 0);
    let small = host(8);
    let source =
        ManagedRgba8Image::from_rgba8(&small, managed(&small, &[1, 2, 3, 255]), 1, 1, limits())
            .unwrap();
    assert!(matches!(
        lanczos(&small, &source, 2, 2, limits()),
        Err(ImageInputError::Host(HostMemoryError::Capacity { .. }))
    ));
    assert_eq!(source.bytes().bytes(), [1, 2, 3, 255]);
    assert_eq!(small.usage().unwrap().committed_bytes(), 4);
    assert!(lanczos(&account, &source, 2, 2, limits()).is_err());
}
