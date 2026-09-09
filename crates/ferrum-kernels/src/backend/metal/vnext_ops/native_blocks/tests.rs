use super::*;
use crate::gguf_blocks::fixtures::{oracle_blocks, FORMATS};
use metal::{MTLCommandBufferStatus, MTLResourceOptions, MTLSize};

#[test]
fn native_block_decoding_matches_cpu_on_real_metal() {
    let device = Device::system_default().expect("native block conformance requires Metal");
    let pipelines = MetalNativeBlockPipelines::new(&device).unwrap();
    let queue = device.new_command_queue();
    for format in FORMATS {
        let bytes = oracle_blocks(format);
        let count = (bytes.len() / format.block_bytes() * format.block_values()) as u32;
        let mut expected = vec![0.0_f32; count as usize];
        format.decode(&bytes, &mut expected).unwrap();
        let mut padded = vec![0xcc_u8; 16];
        padded.extend_from_slice(&bytes);
        let input = device.new_buffer_with_data(
            padded.as_ptr().cast(),
            padded.len() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        // Both buffers use nonzero aligned byte offsets. Canary tail checks
        // the guarded dispatch beyond the final complete native block.
        let mut initial = vec![-12345.0_f32; count as usize + 8];
        let output = device.new_buffer_with_data(
            initial.as_ptr().cast(),
            std::mem::size_of_val(initial.as_slice()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let command = queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipelines.decode);
        encoder.set_buffer(0, Some(&input), 16);
        encoder.set_buffer(1, Some(&output), 16);
        bind_native_block(encoder, format, 2);
        encoder.set_bytes(3, 4, &count as *const _ as *const c_void);
        encoder.dispatch_threads(
            MTLSize::new(u64::from(count) + 17, 1, 1),
            MTLSize::new(64, 1, 1),
        );
        encoder.end_encoding();
        command.commit();
        command.wait_until_completed();
        assert_eq!(
            command.status(),
            MTLCommandBufferStatus::Completed,
            "{format:?}"
        );
        // SAFETY: Shared allocation has the requested f32-aligned size and
        // the command is complete; the buffer outlives this borrowed slice.
        let actual =
            unsafe { std::slice::from_raw_parts(output.contents().cast::<f32>(), initial.len()) };
        initial[4..4 + count as usize].copy_from_slice(&expected);
        for (index, (actual, expected)) in actual.iter().zip(&initial).enumerate() {
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "{format:?}[{index}] {actual} != {expected}"
            );
        }
    }
}
