//! Native Metal storage fixture. This exercises actual blits and private GPU
//! allocations; it does not fabricate a vNext model-step or restore receipt.

use metal::{
    Buffer, BufferRef, CommandQueueRef, DeviceRef, MTLCommandBufferStatus, MTLResourceOptions,
};

pub(super) fn copy(
    queue: &CommandQueueRef,
    source: &BufferRef,
    source_offset: u64,
    destination: &BufferRef,
    destination_offset: u64,
    bytes: u64,
) {
    assert!(bytes > 0);
    assert!(source_offset.checked_add(bytes).unwrap() <= source.length());
    assert!(destination_offset.checked_add(bytes).unwrap() <= destination.length());
    let command = queue.new_command_buffer();
    let blit = command.new_blit_command_encoder();
    blit.copy_from_buffer(
        source,
        source_offset,
        destination,
        destination_offset,
        bytes,
    );
    blit.end_encoding();
    command.commit();
    command.wait_until_completed();
    assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
}

pub(super) fn private_filled(
    device: &DeviceRef,
    queue: &CommandQueueRef,
    bytes: u64,
    fill: u8,
) -> Buffer {
    let initial = vec![fill; usize::try_from(bytes).unwrap()];
    let staging = device.new_buffer_with_data(
        initial.as_ptr().cast(),
        bytes,
        MTLResourceOptions::StorageModeShared,
    );
    let result = device.new_buffer(bytes, MTLResourceOptions::StorageModePrivate);
    copy(queue, &staging, 0, &result, 0, bytes);
    result
}

pub(super) fn read_bytes(
    device: &DeviceRef,
    queue: &CommandQueueRef,
    source: &BufferRef,
) -> Vec<u8> {
    let staging = device.new_buffer(source.length(), MTLResourceOptions::StorageModeShared);
    copy(queue, source, 0, &staging, 0, source.length());
    // SAFETY: the native copy completed, and only this fixture owns staging.
    unsafe {
        std::slice::from_raw_parts(
            staging.contents().cast::<u8>(),
            usize::try_from(staging.length()).unwrap(),
        )
        .to_vec()
    }
}

pub(super) fn assert_bits(label: &str, actual: &[u8], expected: &[u8]) {
    assert_eq!(actual.len(), expected.len(), "{label}: byte length");
    if let Some(index) = actual.iter().zip(expected).position(|(a, b)| a != b) {
        panic!(
            "{label}: first differing byte {index}: {} != {}",
            actual[index], expected[index]
        );
    }
}

pub(super) fn assert_output_bits(label: &str, actual: &[u16], expected: &[u16]) {
    assert_eq!(actual.len(), expected.len(), "{label}: output length");
    if let Some(index) = actual.iter().zip(expected).position(|(a, b)| a != b) {
        panic!(
            "{label}: first differing F16 element {index}: {:#06x} != {:#06x}",
            actual[index], expected[index]
        );
    }
    assert!(!actual.is_empty(), "{label}: output is empty");
    assert!(
        actual
            .iter()
            .all(|&bits| half::f16::from_bits(bits).is_finite()),
        "{label}: non-finite output"
    );
    assert!(
        actual
            .iter()
            .any(|&bits| half::f16::from_bits(bits).to_f32().abs() > 1.0e-4),
        "{label}: degenerate output"
    );
}
