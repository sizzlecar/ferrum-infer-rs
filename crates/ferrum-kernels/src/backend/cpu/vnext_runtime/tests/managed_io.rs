use super::*;
use ferrum_runtime::{HostMemoryAccount, HostMemoryClass, HostMemoryClassLimits, HostMemoryLimits};

fn host() -> HostMemoryAccount {
    HostMemoryAccount::with_allocation_alignment(
        HostMemoryLimits::new(
            128,
            HostMemoryClassLimits {
                input_preprocess_bytes: 64,
                transfer_staging_bytes: 0,
                output_bytes: 64,
                weight_backing_bytes: 0,
            },
        )
        .unwrap(),
        8,
    )
    .unwrap()
}

#[test]
fn managed_upload_keeps_original_sliced_owner_without_a_second_cpu_allocation() {
    let runtime = runtime(8);
    let destination = buffer(&runtime, &[0; 8]);
    let host = host();
    let mut input = host
        .allocate_zeroed(HostMemoryClass::InputPreprocess, 16)
        .unwrap();
    input
        .as_mut()
        .copy_from_slice(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
    let original = input.freeze();
    let slice = original.slice(4..12).unwrap();
    drop(original);
    let command = runtime
        .encode_managed_upload(
            slice,
            HostTransferLayout::new(ElementType::U8, 8).unwrap(),
            &destination,
            0,
        )
        .unwrap()
        .unwrap();
    assert_eq!(
        runtime.resident_bytes(),
        8,
        "no room for an additional legacy upload staging buffer"
    );
    assert_eq!(
        host.usage().unwrap().allocated_bytes(),
        host.backing_bytes(16).unwrap()
    );
    let mut stream = runtime.create_stream().unwrap();
    let fence = runtime
        .execute_entries(
            &mut stream,
            entries(vec![command]),
            false,
            DeviceTimingMode::Off,
        )
        .unwrap();
    assert!(matches!(
        runtime.wait_fence(&fence).unwrap().terminal(),
        DeviceTerminal::Succeeded
    ));
    assert_eq!(read(&destination), [4, 5, 6, 7, 8, 9, 10, 11]);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    drop(destination);
    assert_eq!(runtime.resident_bytes(), 0);
}

#[test]
fn managed_readback_writes_only_the_declared_piece_into_the_original_output() {
    let runtime = runtime(8);
    let source = buffer(&runtime, &[10, 11, 12, 13, 14, 15, 16, 17]);
    let host = host();
    let mut output = host.allocate_zeroed(HostMemoryClass::Output, 6).unwrap();
    output.as_mut().fill(0xaa);
    let pointer = output.as_ref().as_ptr();
    let mut stream = runtime.create_stream().unwrap();
    assert!(runtime
        .readback_managed_into(
            &mut stream,
            &source,
            CopyRegion::new(2, 1, 3).unwrap(),
            HostTransferLayout::new(ElementType::U8, 3).unwrap(),
            &mut output
        )
        .unwrap());
    assert_eq!(output.as_ref(), [0xaa, 12, 13, 14, 0xaa, 0xaa]);
    assert_eq!(output.as_ref().as_ptr(), pointer);
    assert_eq!(
        runtime.peak_resident_bytes(),
        8,
        "managed output does not reserve legacy readback storage"
    );
    assert_eq!(
        host.usage().unwrap().allocated_bytes(),
        host.backing_bytes(6).unwrap()
    );
    for (region, layout) in [
        (
            CopyRegion::new(0, 4, 3).unwrap(),
            HostTransferLayout::new(ElementType::U8, 3).unwrap(),
        ),
        (
            CopyRegion::new(7, 0, 3).unwrap(),
            HostTransferLayout::new(ElementType::U8, 3).unwrap(),
        ),
        (
            CopyRegion::new(0, 0, 3).unwrap(),
            HostTransferLayout::new(ElementType::U8, 4).unwrap(),
        ),
        (
            CopyRegion::new(0, 0, 4).unwrap(),
            HostTransferLayout::new(ElementType::F32, 1).unwrap(),
        ),
    ] {
        assert!(runtime
            .readback_managed_into(&mut stream, &source, region, layout, &mut output)
            .is_err());
        assert_eq!(output.as_ref(), [0xaa, 12, 13, 14, 0xaa, 0xaa]);
    }
    drop(output);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn managed_upload_rejects_foreign_and_invalid_ranges_before_mutation_and_refunds() {
    let runtime = runtime(8);
    let other = super::runtime(8);
    let destination = buffer(&runtime, &[1; 8]);
    let foreign = buffer(&other, &[2; 8]);
    let host = host();
    for (destination, layout, offset) in [
        (
            &foreign,
            HostTransferLayout::new(ElementType::U8, 4).unwrap(),
            0,
        ),
        (
            &destination,
            HostTransferLayout::new(ElementType::U8, 4).unwrap(),
            6,
        ),
        (
            &destination,
            HostTransferLayout::new(ElementType::U8, 3).unwrap(),
            0,
        ),
        (
            &destination,
            HostTransferLayout::new(ElementType::F32, 1).unwrap(),
            0,
        ),
    ] {
        let bytes = host
            .allocate_zeroed(HostMemoryClass::InputPreprocess, 4)
            .unwrap()
            .freeze();
        assert!(runtime
            .encode_managed_upload(bytes, layout, destination, offset)
            .is_err());
        assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    }
    assert_eq!(read(&destination), [1; 8]);
    assert_eq!(read(&foreign), [2; 8]);
}
