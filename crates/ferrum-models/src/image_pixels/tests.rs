use super::*;
use ferrum_runtime::{HostMemoryClassLimits, HostMemoryLimits};
fn host() -> HostMemoryAccount {
    HostMemoryAccount::new(
        HostMemoryLimits::new(
            4096,
            HostMemoryClassLimits {
                input_preprocess_bytes: 2048,
                output_bytes: 2048,
                transfer_staging_bytes: 0,
                weight_backing_bytes: 0,
            },
        )
        .unwrap(),
    )
}
fn spec() -> ResolvedTensorSpec {
    ResolvedTensorSpec::new(
        vec![2, 3, 1, 2],
        ElementType::F32,
        ResolvedTensorLayout::Contiguous,
    )
    .unwrap()
}
fn bytes(host: &HostMemoryAccount, class: HostMemoryClass, values: &[f32]) -> ManagedHostBytes {
    let mut b = host.allocate_zeroed(class, values.len() * 4).unwrap();
    for (bytes, value) in b.as_mut().chunks_exact_mut(4).zip(values) {
        bytes.copy_from_slice(&value.to_le_bytes());
    }
    b.freeze()
}
#[test]
fn unit_rgb_streams_both_requests_with_opaque_alpha_without_allocating_a_frame() {
    let host = host();
    let data = bytes(
        &host,
        HostMemoryClass::Output,
        &[0.0, 1.0, 0.5, 0.0, 1.0, 0.5, 1.0, 0.0, 0.0, 1.0, 0.5, 0.0],
    );
    let before = host.usage().unwrap().committed_bytes();
    let pixels =
        ManagedImagePixels::new(&host, &spec(), &data, ImagePixelEncoding::F32UnitRgbRne).unwrap();
    assert_eq!(pixels.rgba_dimensions(), [2, 4, 1, 2]);
    assert_eq!(pixels.frame_bytes(), 8);
    let mut first = Vec::new();
    let mut second = Vec::new();
    pixels
        .write_frame_rgba8(0, &mut [0; 3], &mut first)
        .unwrap();
    pixels
        .write_frame_rgba8(1, &mut [0; 1], &mut second)
        .unwrap();
    assert_eq!(first, [0, 128, 255, 255, 255, 0, 128, 255]);
    assert_eq!(second, [255, 0, 128, 255, 0, 255, 0, 255]);
    assert_eq!(host.usage().unwrap().committed_bytes(), before);
    let mut writer = vec![42];
    assert!(pixels
        .write_frame_rgba8(2, &mut [0; 4], &mut writer)
        .is_err());
    assert!(pixels.write_frame_rgba8(0, &mut [], &mut writer).is_err());
    assert_eq!(writer, [42]);
    drop(data);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}
#[test]
fn encoding_is_explicit_and_rejects_wrong_domain_owner_class_or_late_nonfinite() {
    let account = host();
    for invalid in [f32::NAN, f32::INFINITY, -f32::INFINITY, -0.01, 1.01] {
        let mut values = [0.0; 12];
        values[11] = invalid;
        let data = bytes(&account, HostMemoryClass::Output, &values);
        assert!(ManagedImagePixels::new(
            &account,
            &spec(),
            &data,
            ImagePixelEncoding::F32UnitRgbRne
        )
        .is_err());
    }
    let data = bytes(&account, HostMemoryClass::Output, &[0.0; 12]);
    assert!(ManagedImagePixels::new(
        &account,
        &spec(),
        &data,
        ImagePixelEncoding::Bf16SignedRgbaStaged
    )
    .is_err());
    assert!(
        ManagedImagePixels::new(&host(), &spec(), &data, ImagePixelEncoding::F32UnitRgbRne)
            .is_err()
    );
    let input = bytes(&account, HostMemoryClass::InputPreprocess, &[0.0; 12]);
    assert!(
        ManagedImagePixels::new(&account, &spec(), &input, ImagePixelEncoding::F32UnitRgbRne)
            .is_err()
    );
    let wrong = ResolvedTensorSpec::new(
        vec![1, 3, 1, 2],
        ElementType::F32,
        ResolvedTensorLayout::Contiguous,
    )
    .unwrap();
    assert!(
        ManagedImagePixels::new(&account, &wrong, &data, ImagePixelEncoding::F32UnitRgbRne)
            .is_err()
    );
    drop((data, input));
    assert_eq!(account.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn owned_artifact_keeps_original_scope_charge_and_encoding_after_producer_closes() {
    let account = host();
    let scope = account
        .reserve_scope(&[(HostMemoryClass::Output, 48)])
        .unwrap();
    let backing = bytes(&scope.account(), HostMemoryClass::Output, &[0.5; 12]);
    let artifact = OwnedImagePixels::new(
        &account,
        spec(),
        backing.clone(),
        ImagePixelEncoding::F32UnitRgbRne,
    )
    .unwrap();
    assert!(artifact.bytes().same_region(&backing));
    let retained = artifact.clone();
    drop((scope, backing, artifact));
    assert_eq!(account.usage().unwrap().committed_bytes(), 48);
    assert_eq!(retained.encoding(), ImagePixelEncoding::F32UnitRgbRne);
    let mut rgba = Vec::new();
    retained
        .view(&account)
        .unwrap()
        .write_frame_rgba8(1, &mut [0; 3], &mut rgba)
        .unwrap();
    assert_eq!(rgba, [128, 128, 128, 255, 128, 128, 128, 255]);
    assert!(retained.view(&host()).is_err());
    drop(retained);
    assert_eq!(account.usage().unwrap().committed_bytes(), 0);
}
