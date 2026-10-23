use super::*;
use ferrum_runtime::{HostMemoryClassLimits, HostMemoryLimits};

pub(super) fn host(bytes: usize) -> HostMemoryAccount {
    HostMemoryAccount::new(
        HostMemoryLimits::new(
            bytes,
            HostMemoryClassLimits {
                input_preprocess_bytes: bytes,
                transfer_staging_bytes: 0,
                output_bytes: bytes,
                weight_backing_bytes: 0,
            },
        )
        .unwrap(),
    )
}

pub(super) fn limits() -> ImageInputLimits {
    ImageInputLimits::new(1 << 20, 1 << 20, 4096, 4096, 1 << 20).unwrap()
}

pub(super) fn managed(host: &HostMemoryAccount, data: &[u8]) -> ManagedHostBytes {
    let mut buffer = host
        .allocate_zeroed(HostMemoryClass::InputPreprocess, data.len())
        .unwrap();
    buffer.as_mut().copy_from_slice(data);
    buffer.freeze()
}

fn png_bytes(width: u32, height: u32, color: ::png::ColorType, data: &[u8]) -> Vec<u8> {
    let mut bytes = Vec::new();
    {
        let mut encoder = ::png::Encoder::new(&mut bytes, width, height);
        encoder.set_color(color);
        encoder.set_depth(::png::BitDepth::Eight);
        let mut writer = encoder.write_header().unwrap();
        writer.write_image_data(data).unwrap();
    }
    bytes
}

#[test]
fn png_rgb_and_rgba_have_exact_pixels_and_retained_parent_charges() {
    for (color, input, expected, alpha) in [
        (
            ::png::ColorType::Rgb,
            vec![1, 17, 255, 0, 3, 91],
            vec![1, 17, 255, 255, 0, 3, 91, 255],
            false,
        ),
        (
            ::png::ColorType::Rgba,
            vec![1, 17, 255, 0, 0, 3, 91, 128],
            vec![1, 17, 255, 0, 0, 3, 91, 128],
            true,
        ),
    ] {
        let host = host(2 << 20);
        let png = managed(&host, &png_bytes(2, 1, color, &input));
        let image = ManagedRgba8Image::decode_png(&host, &png, limits()).unwrap();
        assert_eq!(
            (image.width(), image.height(), image.source_has_alpha()),
            (2, 1, alpha)
        );
        assert_eq!(image.bytes().bytes(), expected);
        assert_eq!(host.usage().unwrap().allocated_bytes(), png.len() + 8);
        let clone = image.clone();
        drop(image);
        assert_eq!(clone.bytes().bytes(), expected);
        drop(clone);
        assert_eq!(host.usage().unwrap().allocated_bytes(), png.len());
        drop(png);
        assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    }
}

#[test]
fn png_capacity_and_foreign_owner_fail_without_publishing_an_image() {
    let wire = png_bytes(2, 1, ::png::ColorType::Rgb, &[1, 2, 3, 4, 5, 6]);
    let account = host(wire.len() + 7);
    let png = managed(&account, &wire);
    assert!(matches!(
        ManagedRgba8Image::decode_png(&account, &png, limits()),
        Err(ImageInputError::Host(HostMemoryError::Capacity { .. }))
    ));
    assert_eq!(account.usage().unwrap().allocated_bytes(), wire.len());
    let other = host(2 << 20);
    assert!(matches!(
        ManagedRgba8Image::decode_png(&other, &png, limits()),
        Err(ImageInputError::Invalid(_))
    ));
    assert_eq!(other.usage().unwrap().committed_bytes(), 0);
    let output = other
        .allocate_zeroed(HostMemoryClass::Output, 8)
        .unwrap()
        .freeze();
    assert!(ManagedRgba8Image::from_rgba8(&other, output, 2, 1, limits()).is_err());
    assert_eq!(other.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn unsupported_png_and_corrupt_or_trailing_data_release_private_backings() {
    let account = host(2 << 20);
    let mut valid = png_bytes(2, 1, ::png::ColorType::Rgba, &[1, 2, 3, 4, 5, 6, 7, 8]);
    // CRC corruption in IHDR must be rejected by the independent codec.
    valid[29] ^= 1;
    let mut trailing = png_bytes(1, 1, ::png::ColorType::Rgb, &[1, 2, 3]);
    trailing.push(0);
    let gray = png_bytes(1, 1, ::png::ColorType::Grayscale, &[17]);
    for bytes in [valid, trailing, gray, b"not a PNG".to_vec()] {
        let wire = managed(&account, &bytes);
        assert!(ManagedRgba8Image::decode_png(&account, &wire, limits()).is_err());
        assert_eq!(account.usage().unwrap().allocated_bytes(), wire.len());
        drop(wire);
        assert_eq!(account.usage().unwrap().committed_bytes(), 0);
    }
}

#[test]
fn png_admission_rejects_encoded_pixel_and_axis_bounds_before_output() {
    let account = host(2 << 20);
    let wire = managed(&account, &png_bytes(2, 2, ::png::ColorType::Rgb, &[0; 12]));
    for bounds in [
        ImageInputLimits::new(wire.len() - 1, 4, 2, 2, 1 << 20).unwrap(),
        ImageInputLimits::new(wire.len(), 3, 2, 2, 1 << 20).unwrap(),
        ImageInputLimits::new(wire.len(), 4, 1, 2, 1 << 20).unwrap(),
    ] {
        assert!(matches!(
            ManagedRgba8Image::decode_png(&account, &wire, bounds),
            Err(ImageInputError::Limit(_))
        ));
        assert_eq!(account.usage().unwrap().allocated_bytes(), wire.len());
    }
    assert!(ImageInputLimits::new(0, 1, 1, 1, 1).is_err());
    assert!(ImageInputLimits::new(1, usize::MAX, 1, 1, 1).is_err());
    assert!(ImageInputLimits::new(1, 1, 1, 1, 0).is_err());
    assert!(
        ManagedRgba8Image::from_rgba8(&account, managed(&account, &[0; 7]), 2, 1, limits())
            .is_err()
    );
    assert_eq!(account.usage().unwrap().allocated_bytes(), wire.len());
}
