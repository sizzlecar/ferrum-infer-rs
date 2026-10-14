//! Independent TEST native ABI; the product V1/V2 dispatch stays unchanged.
use super::*;
use crate::backend::cuda::vllm_paged_attn::dispatch_vnext_addressed_paged_attention_fixed_masked_batch_raw;
use std::ffi::c_void;

#[link(name = "ferrum_causal_gqa2_test", kind = "static")]
unsafe extern "C" {
    fn ferrum_test_vnext_vllm_paged_attention_v2_f16_h256_b16_gqa2_indexed_v1(
        out: *mut c_void,
        exp_sums: *mut f32,
        max_logits: *mut f32,
        tmp_out: *mut c_void,
        query: *const c_void,
        num_kv_heads: i32,
        scale: f32,
        block_addresses: *const u64,
        seq_lens: *const i32,
        row_map: *const i32,
        num_seqs: i32,
        original_num_seqs: i32,
        num_heads: i32,
        max_num_blocks_per_seq: i32,
        q_stride: i32,
        kv_block_stride: i32,
        kv_head_stride: i32,
        max_seq_len: i32,
        stream: *mut c_void,
    ) -> i32;
}

#[allow(clippy::too_many_arguments)]
pub(super) fn enqueue(
    batch: &FixedBatchDecode,
    stream: &Arc<CudaStream>,
    functions: &CausalAttentionFunctions,
    launches: &[CausalAttentionLaunch],
    binding: u64,
    binding_layout: BindingLayout,
    shape: CudaCausalAttentionShape,
    layout: ScratchLayout,
    scratch: u64,
    query_norm: u64,
    key_norm: u64,
    output_gate: bool,
) -> Result<(), CudaDeviceRuntimeError> {
    assert_eq!(shape.head_dim, 256);
    assert_eq!(shape.query_heads, 24);
    assert_eq!(shape.key_value_heads, 4);
    let partition = layout.vllm.expect("validated fixed attention scratch");
    let original_rows = i32::try_from(launches.len()).unwrap();
    batch.enqueue_with_attention(
        stream,
        functions,
        launches,
        binding,
        binding_layout,
        shape,
        layout,
        scratch,
        query_norm,
        key_norm,
        output_gate,
        |group, lengths, rows, expected| {
            assert_eq!(group.sequences, original_rows);
            let out = scratch_pointer(scratch, layout.context)?;
            let query = scratch_pointer(scratch, layout.query)?;
            let addresses = scratch_pointer(binding, BINDING_CONTROL_BYTES)?;
            let exp_sums = scratch_pointer(scratch, partition.exp_sums)?;
            let max_logits = scratch_pointer(scratch, partition.max_logits)?;
            let tmp = scratch_pointer(scratch, partition.temporary_output)?;
            if expected == VnextAddressedPagedAttentionKernel::V1 {
                // Exactly the production V1 call, including the all-inactive
                // mask when every current owner selects V2.
                unsafe {
                    dispatch_vnext_addressed_paged_attention_fixed_masked_batch_raw(
                        stream,
                        out,
                        query,
                        addresses,
                        lengths,
                        rows,
                        group.maximum_sequence,
                        Some(exp_sums),
                        Some(max_logits),
                        Some(tmp),
                        group.sequences,
                        original_rows,
                        shape.query_heads,
                        shape.key_value_heads,
                        shape.head_dim,
                        group.table_stride,
                        expected,
                    )
                }
                .map(|_| ())
                .map_err(|error| CudaDeviceRuntimeError::contract(error.to_string()))
            } else {
                assert_eq!(expected, VnextAddressedPagedAttentionKernel::V2);
                assert_eq!(group.maximum_sequence, 2048);
                assert!(group.table_stride >= 128);
                let kv_head_stride = shape.head_dim * 16;
                // The fixture owns all admitted pages/scratch until both
                // graphs are destroyed. The shared bundle supplies freshly
                // gathered identity-or--1 maps on this same stream.
                let status = unsafe {
                    ferrum_test_vnext_vllm_paged_attention_v2_f16_h256_b16_gqa2_indexed_v1(
                        out as *mut c_void,
                        exp_sums as *mut f32,
                        max_logits as *mut f32,
                        tmp as *mut c_void,
                        query as *const c_void,
                        shape.key_value_heads,
                        1.0 / (shape.head_dim as f32).sqrt(),
                        addresses as *const u64,
                        lengths as *const i32,
                        rows as *const i32,
                        group.sequences,
                        original_rows,
                        shape.query_heads,
                        group.table_stride,
                        shape.query_heads * shape.head_dim,
                        shape.key_value_heads * kv_head_stride,
                        kv_head_stride,
                        group.maximum_sequence as i32,
                        stream.cu_stream() as *mut c_void,
                    )
                };
                if status != 0 {
                    return Err(CudaDeviceRuntimeError::contract(format!(
                        "GQA2 test native launch returned CUDA status {status}"
                    )));
                }
                Ok(())
            }
        },
    )
}

#[test]
#[ignore = "requires CUDA and the FERRUM_TEST_CAUSAL_GQA2 native archive"]
fn gqa2_v2_preserves_fixed_bits_state_and_fresh_replay_on_cuda() {
    captured_capacity_replay_with_gqa2(
        &[(256, 8, true, 2048, None), (256, 32, true, 2048, None)],
        true,
    );
}

#[test]
#[ignore = "requires CUDA and the FERRUM_TEST_CAUSAL_GQA2 native archive"]
fn gqa2_v2_m8_m32_sixteen_layer_inclusive_paired_on_cuda() {
    sixteen_layer_inclusive_paired(true);
}
