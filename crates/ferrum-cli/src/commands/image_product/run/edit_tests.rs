use super::*;
use ferrum_runtime::{HostMemoryAccount, HostMemoryClassLimits, HostMemoryLimits};
use std::fs;

fn host(bytes: usize) -> HostMemoryAccount {
    HostMemoryAccount::new(
        HostMemoryLimits::new(
            bytes,
            HostMemoryClassLimits {
                input_preprocess_bytes: bytes,
                transfer_staging_bytes: 0,
                weight_backing_bytes: 0,
                output_bytes: 0,
            },
        )
        .unwrap(),
    )
}

#[test]
fn edit_cli_input_is_read_into_one_admitted_backing_and_retires() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("input.png");
    let wire = b"\x89PNG\r\n\x1a\ntransport-fixture";
    fs::write(&path, wire).unwrap();
    let (file, length) = open_image(&path).unwrap();
    let account = host(wire.len());
    let bytes = read_image(&account, file, length).unwrap();
    assert_eq!(bytes.bytes(), wire);
    assert!(account.owns(&bytes));
    assert_eq!(account.usage().unwrap().allocated_bytes(), wire.len());
    drop(bytes);
    assert_eq!(account.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn edit_cli_reads_own_two_k_png_and_preserves_encoded_file_boundaries() {
    use ferrum_models::qwen_image21::image_input::{ImageInputLimits, ManagedRgba8Image};
    use ferrum_output::png::{write_rgba8_png, PngDimensions, PngLimits};
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("output.png");
    let dimensions = PngDimensions::new(2048, 2048).unwrap();
    let row = [1, 17, 255, 128].repeat(2048);
    let receipt = write_rgba8_png(
        &mut fs::File::create(&path).unwrap(),
        dimensions,
        PngLimits::default(),
        |writer| {
            for _ in 0..2048 {
                writer.write_all(&row)?;
            }
            Ok(())
        },
    )
    .unwrap();
    let (file, length) = open_image(&path).unwrap();
    assert_eq!(length as u64, receipt.encoded_bytes());
    assert!(length > dimensions.rgba_bytes());
    let account = host(length + dimensions.rgba_bytes());
    let encoded = read_image(&account, file, length).unwrap();
    let decoded =
        ManagedRgba8Image::decode_png(&account, &encoded, ImageInputLimits::default()).unwrap();
    assert_eq!((decoded.width(), decoded.height()), (2048, 2048));
    for actual in decoded.bytes().bytes().chunks_exact(row.len()) {
        assert_eq!(actual, row);
    }
    drop((decoded, encoded));
    assert_eq!(account.usage().unwrap().committed_bytes(), 0);

    // File admission checks extent before allocating or treating bytes as PNG.
    let maximum = ImageInputLimits::default().maximum_encoded_bytes();
    let sparse = dir.path().join("extent.png");
    let file = fs::File::create(&sparse).unwrap();
    file.set_len(maximum as u64).unwrap();
    assert_eq!(open_image(&sparse).unwrap().1, maximum);
    file.set_len(maximum as u64 + 1).unwrap();
    let error = open_image(&sparse).unwrap_err();
    assert!(error.to_string().contains(&maximum.to_string()));
}

#[test]
fn edit_cli_rejects_invalid_extent_capacity_and_file_changes_without_leaking() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("input.png");
    assert!(open_image(dir.path()).is_err());
    fs::write(&path, b"").unwrap();
    assert!(open_image(&path).is_err());
    let wire = b"\x89PNG\r\n\x1a\n1234";
    fs::write(&path, wire).unwrap();
    let (file, length) = open_image(&path).unwrap();
    let account = host(wire.len() - 1);
    assert!(read_image(&account, file, length).is_err());
    assert_eq!(account.usage().unwrap().committed_bytes(), 0);
    let account = host(1024);
    for altered in [
        b"\x89PNG\r\n\x1a\n".as_slice(),
        b"\x89PNG\r\n\x1a\n12345".as_slice(),
        b"not a png...".as_slice(),
    ] {
        fs::write(&path, wire).unwrap();
        let (file, length) = open_image(&path).unwrap();
        fs::write(&path, altered).unwrap();
        assert!(read_image(&account, file, length).is_err());
        assert_eq!(account.usage().unwrap().committed_bytes(), 0);
    }
}
