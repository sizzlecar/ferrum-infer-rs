use super::*;

fn shape(head_dim: u64) -> CausalAttentionShape {
    let mut shape =
        CausalAttentionShape::from_attributes(&super::super::tests::attributes(false)).unwrap();
    shape.query_heads = 4;
    shape.key_value_heads = 2;
    shape.head_dim = head_dim;
    shape.hidden_size = 4 * head_dim;
    shape.query_features = 4 * head_dim;
    shape.query_projection_features = 4 * head_dim;
    shape.kv_features = 2 * head_dim;
    shape.rope_dim = head_dim;
    shape.rope_frequency_denominator = head_dim;
    shape.rope_pair_offset = head_dim / 2;
    shape.maximum_context_tokens = 2048;
    shape.attention_scale = 1.0 / (head_dim as f32).sqrt();
    shape
}

#[test]
fn packed_decode_batches_preserve_v1_v2_and_require_a_shared_run() {
    use CausalAttentionKernelPath::*;
    assert!(eligible([VllmAddressedDecodeV2; 4], true));
    assert!(eligible([VllmAddressedDecodeV1; 8], true));
    assert!(eligible(
        [
            VllmAddressedDecodeV1,
            VllmAddressedDecodeV1,
            VllmAddressedDecodeV2
        ],
        true
    ));
    assert!(eligible(
        [
            VllmAddressedDecodeV1,
            VllmAddressedDecodeV2,
            VllmAddressedDecodeV2,
            VllmAddressedDecodeV1
        ],
        true
    ));
    assert!(!eligible(
        [
            VllmAddressedDecodeV1,
            VllmAddressedDecodeV2,
            VllmAddressedDecodeV1
        ],
        true
    ));
    let mixed = [
        VllmAddressedDecodeV1,
        VllmAddressedDecodeV1,
        VllmAddressedDecodeV2,
        VllmAddressedDecodeV1,
    ];
    assert_eq!(
        group_ranges(&mixed).collect::<Vec<_>>(),
        vec![0..2, 2..3, 3..4]
    );
    assert!(!eligible([VllmAddressedDecodeV2; 4], false));
    assert!(!eligible([VllmAddressedDecodeV2], true));
    assert!(!eligible([], true));
    for other in [
        VllmAddressedDecodeV1,
        VllmAddressedFallback,
        TokenMajorFallback,
        VllmAddressedVarlen,
        VllmAddressedVarlenTiled,
    ] {
        assert!(!eligible([VllmAddressedDecodeV2, other], true));
    }
}

#[test]
fn table_stride_accounts_for_control_padding_and_rejects_invalid_extents() {
    let shape = shape(128);
    let binding = BindingLayout::new(shape, 4).unwrap();
    let batch = BatchDecode::dimensions(4, binding, 1536, 4, 128).unwrap();
    assert_eq!(batch.table_stride as u64 * 8, binding.slot_bytes);
    // The slot includes control words, but each owner's used table remains
    // wholly before its next owner. The external ABI uses this value as stride.
    assert!(24 + 1536 / 16 * 8 <= binding.slot_bytes);
    assert!(BatchDecode::dimensions(4, binding, 512, 4, 128).is_ok());
    assert!(BatchDecode::dimensions(4, binding, 0, 4, 128).is_err());
    assert!(BatchDecode::dimensions(4, binding, 4096, 4, 128).is_err());
    assert!(BatchDecode::dimensions(3, binding, 1536, 4, 128).is_err());
    let unaligned = BindingLayout {
        slot_bytes: binding.slot_bytes + 1,
        required_bytes: (binding.slot_bytes + 1) * 4,
    };
    assert!(BatchDecode::dimensions(4, unaligned, 1536, 4, 128).is_err());
    let large = BindingLayout {
        slot_bytes: binding.slot_bytes,
        required_bytes: binding.slot_bytes * 65_536,
    };
    assert!(BatchDecode::dimensions(65_536, large, 1536, 4, 128).is_err());
    assert!(BatchDecode::dimensions(4, binding, 1536, u64::MAX, 256).is_err());
}

#[test]
fn scratch_accounts_for_every_owner_without_aliasing_partition_arrays() {
    for heads in [128, 256] {
        let shape = shape(heads);
        for owners in [1, 4, 8] {
            let layout = ScratchLayout::for_participants(
                shape,
                owners as u64,
                owners,
                CausalProjection::F16,
                AttentionExecutionPolicy::NativeAdaptive,
            )
            .unwrap();
            assert_eq!(
                layout.required_bytes,
                shape.scratch_bytes_per_token().unwrap() * owners as u64
                    + shape.vllm_scratch_bytes().unwrap() * owners as u64
            );
            if let Some(vllm) = layout.vllm {
                let rows = owners as u64 * shape.query_heads * 4;
                assert!(vllm.exp_sums + rows * 4 <= vllm.max_logits);
                assert!(vllm.max_logits + rows * 4 <= vllm.temporary_output);
                assert!(vllm.temporary_output + rows * heads * 2 <= vllm.sequence_lengths);
                assert!(vllm.sequence_lengths + owners as u64 * 4 <= layout.required_bytes);
            }
        }
    }
    assert!(ScratchLayout::for_participants(
        shape(128),
        3,
        4,
        CausalProjection::F16,
        AttentionExecutionPolicy::NativeAdaptive
    )
    .is_err());
    assert!(ScratchLayout::for_participants(
        shape(128),
        u64::MAX,
        4,
        CausalProjection::F16,
        AttentionExecutionPolicy::NativeAdaptive
    )
    .is_err());
}

#[cfg(feature = "vllm-paged-attn-v2")]
#[test]
fn batched_launch_rejects_reordered_query_rows_and_single_owner_scratch() {
    let shape = shape(256);
    let binding = BindingLayout::new(shape, 2).unwrap();
    let layout = ScratchLayout::for_participants(
        shape,
        2,
        2,
        CausalProjection::F16,
        AttentionExecutionPolicy::NativeAdaptive,
    )
    .unwrap();
    let mut launches = [513_u64, 1025]
        .into_iter()
        .enumerate()
        .map(|(index, length)| {
            let offset = |base, width| layout.token_offset(base, index as u64, width).unwrap();
            CausalAttentionLaunch {
                input_region: 0,
                output_region: 1,
                binding_offset: binding.binding_offset(index).unwrap(),
                packed_token_start: index as u64,
                packed_query_raw: offset(layout.query_raw, shape.query_projection_features),
                packed_key_raw: offset(layout.key_raw, shape.kv_features),
                packed_value_raw: offset(layout.value_raw, shape.kv_features),
                packed_query: offset(layout.query, shape.query_features),
                packed_context: offset(layout.context, shape.query_features),
                tokens: 1,
                tokens_i32: 1,
                sequence_tokens: length,
                sequence_tokens_i32: length as i32,
                table_entries_i32: shape.table_entries(length).unwrap() as i32,
                replay_topology: CausalAttentionReplayTopology::new(
                    shape,
                    CausalAttentionKernelPath::VllmAddressedDecodeV2,
                    length,
                )
                .unwrap(),
                path: CausalAttentionKernelPath::VllmAddressedDecodeV2,
            }
        })
        .collect::<Vec<_>>();
    let batch = BatchDecode::for_launches(&launches, true, binding, shape, layout)
        .unwrap()
        .unwrap();
    assert_eq!(batch.maximum_sequence(), 1536);
    assert!(
        BatchDecode::for_launches(&launches, false, binding, shape, layout)
            .unwrap()
            .is_none()
    );
    let single = ScratchLayout::new(
        shape,
        2,
        CausalProjection::F16,
        AttentionExecutionPolicy::NativeAdaptive,
    )
    .unwrap();
    assert!(BatchDecode::for_launches(&launches, true, binding, shape, single).is_err());
    launches[1].packed_query = launches[0].packed_query;
    assert!(BatchDecode::for_launches(&launches, true, binding, shape, layout).is_err());
}

#[cfg(feature = "vllm-paged-attn-v2")]
mod gpu;

#[cfg(feature = "vllm-paged-attn-v2")]
mod timing;
