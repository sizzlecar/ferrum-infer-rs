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
                assert_eq!(vllm.sequence_lengths + owners as u64 * 4, vllm.row_indices);
                assert!(vllm.row_indices + owners as u64 * 4 <= layout.required_bytes);
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

#[test]
fn packed_prepare_gate_geometry_respects_native_index_bounds() {
    let shape = shape(256);
    let binding = BindingLayout::new(shape, 4).unwrap();
    let work = PackedDecodeWork::new(4, binding, shape, true).unwrap();
    let prepare = work.prepare().unwrap();
    assert_eq!(prepare.token_grid, 1);
    assert_eq!(prepare.participant_grid, 4);
    assert_eq!(prepare.binding_slot_bytes, binding.slot_bytes);
    assert_eq!(work.gate_tokens(), Some(4));
    assert!(PackedDecodeWork::new(
        65_535,
        BindingLayout::new(shape, 65_535).unwrap(),
        shape,
        true
    )
    .unwrap()
    .prepare()
    .is_some());
    let over_grid_z = PackedDecodeWork::new(
        65_536,
        BindingLayout::new(shape, 65_536).unwrap(),
        shape,
        true,
    )
    .unwrap();
    assert!(over_grid_z.prepare().is_none());
    assert_eq!(over_grid_z.gate_tokens(), Some(65_536));
    assert!(PackedDecodeWork::new(3, binding, shape, true).is_none());

    let last_complete_block = (i32::MAX as u64 / 256) * 256;
    assert_eq!(packed_gate_tokens(1, last_complete_block, 1), Some(1));
    // A partial last block would make native signed idx wrap before its guard.
    assert_eq!(packed_gate_tokens(1, last_complete_block + 1, 1), None);
    assert_eq!(packed_gate_tokens(2, 128, (i32::MAX as u64) / 2), Some(2));
    assert_eq!(packed_gate_tokens(2, 128, (i32::MAX as u64) / 2 + 1), None);
    assert_eq!(packed_gate_tokens(u64::MAX, 2, 1), None);
    assert_eq!(packed_gate_tokens(2, 1, u64::MAX), None);
    assert_eq!(packed_gate_tokens(0, 128, 256), None);
}

#[test]
fn packed_decode_dispatch_count_tracks_gate_fallback_and_unchanged_attention_groups() {
    use CausalAttentionKernelPath::{VllmAddressedDecodeV1 as V1, VllmAddressedDecodeV2 as V2};
    let paths = [V1, V1, V2, V1];
    let mut shape = shape(256);
    shape.output_gate = true;
    shape.query_projection_features = 2 * shape.query_features;
    let binding = BindingLayout::new(shape, paths.len()).unwrap();
    let work = PackedDecodeWork::new(paths.len(), binding, shape, true);
    assert_eq!(physical_dispatch_count(paths, true, false, true, None), 19);
    assert_eq!(physical_dispatch_count(paths, true, false, true, work), 13);
    assert_eq!(physical_dispatch_count(paths, false, false, true, work), 12);
    assert_eq!(physical_dispatch_count(paths, true, true, true, work), 14);

    // Request/Step KV can alias across participants: no parallel prepare proof.
    // Dense scratch rows still permit one gate, independently of KV ownership.
    let work = PackedDecodeWork::new(paths.len(), binding, shape, false).unwrap();
    assert!(work.prepare().is_none());
    assert_eq!(work.gate_tokens(), Some(4));
    assert_eq!(
        physical_dispatch_count(paths, true, false, true, Some(work)),
        16
    );
    assert_eq!(
        physical_dispatch_count(paths, false, false, true, Some(work)),
        15
    );

    // Each row's projection indexing fits, but four rows together do not.
    shape.query_projection_features = (i32::MAX as u64) / 2;
    let work = PackedDecodeWork::new(paths.len(), binding, shape, true).unwrap();
    assert!(work.gate_tokens().is_none());
    assert_eq!(
        physical_dispatch_count(paths, true, false, true, Some(work)),
        16
    );
    // The mode cannot enable a formerly ineligible attention batch.
    assert_eq!(
        physical_dispatch_count([V1, V2, V1], true, false, true, Some(work)),
        physical_dispatch_count([V1, V2, V1], true, false, true, None),
    );
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
    let batch = BatchDecode::for_launches(
        &launches,
        true,
        binding,
        shape,
        layout,
        false,
        true,
        CausalDecodeAttentionGrouping::Contiguous,
    )
    .unwrap()
    .unwrap();
    assert_eq!(batch.maximum_sequence(), 1536);
    assert!(batch.packed_work().is_none());
    let packed = BatchDecode::for_launches(
        &launches,
        true,
        binding,
        shape,
        layout,
        true,
        true,
        CausalDecodeAttentionGrouping::Contiguous,
    )
    .unwrap()
    .unwrap();
    assert_eq!(packed.maximum_sequence(), batch.maximum_sequence());
    assert!(packed.packed_work().is_some());
    let shared_kv = BatchDecode::for_launches(
        &launches,
        true,
        binding,
        shape,
        layout,
        true,
        false,
        CausalDecodeAttentionGrouping::Contiguous,
    )
    .unwrap()
    .unwrap();
    assert!(shared_kv.packed_work().unwrap().prepare().is_none());
    assert_eq!(shared_kv.packed_work().unwrap().gate_tokens(), Some(2));
    let mut bad_raw = launches.clone();
    bad_raw[1].packed_key_raw = bad_raw[0].packed_key_raw;
    assert!(BatchDecode::for_launches(
        &bad_raw,
        true,
        binding,
        shape,
        layout,
        true,
        true,
        CausalDecodeAttentionGrouping::Contiguous
    )
    .is_err());
    assert!(BatchDecode::for_launches(
        &launches,
        false,
        binding,
        shape,
        layout,
        true,
        true,
        CausalDecodeAttentionGrouping::Contiguous
    )
    .unwrap()
    .is_none());
    let single = ScratchLayout::new(
        shape,
        2,
        CausalProjection::F16,
        AttentionExecutionPolicy::NativeAdaptive,
    )
    .unwrap();
    assert!(BatchDecode::for_launches(
        &launches,
        true,
        binding,
        shape,
        single,
        true,
        true,
        CausalDecodeAttentionGrouping::Contiguous
    )
    .is_err());
    launches[1].packed_query = launches[0].packed_query;
    assert!(BatchDecode::for_launches(
        &launches,
        true,
        binding,
        shape,
        layout,
        true,
        true,
        CausalDecodeAttentionGrouping::Contiguous
    )
    .is_err());
}

#[test]
fn packed_prepare_gate_replay_identity_distinguishes_effective_fallbacks() {
    let mut shape = shape(256);
    let binding = BindingLayout::new(shape, 4).unwrap();
    let sequence_owned = PackedDecodeWork::new(4, binding, shape, true);
    let possibly_shared = PackedDecodeWork::new(4, binding, shape, false);
    let key = |work, output_gate| {
        bind_prepare_gate_replay(
            CudaCommandReplayKeyBuilder::new("same-provider", "causal"),
            work,
            output_gate,
        )
        .finish()
        .bind_runtime_payload("causal", std::iter::empty(), &[])
    };
    // Identical provider/address payloads must not reuse a parallel-prepare
    // graph for legal shared KV that requires the serial prepare path.
    assert_ne!(key(sequence_owned, true), key(possibly_shared, true));
    assert_ne!(key(possibly_shared, true), key(None, true));

    shape.query_projection_features = i32::MAX as u64;
    let prepare_only = PackedDecodeWork::new(4, binding, shape, true);
    let both_fallback = PackedDecodeWork::new(4, binding, shape, false);
    assert_ne!(key(sequence_owned, true), key(prepare_only, true));
    // When gate is absent its unused eligibility must not invent topology.
    assert_eq!(key(sequence_owned, false), key(prepare_only, false));
    assert_eq!(key(both_fallback, true), key(None, true));
}

#[test]
fn indexed_row_map_is_stable_unique_and_rejects_non_decode_or_native_overflow() {
    use CausalAttentionKernelPath::{VllmAddressedDecodeV1 as V1, VllmAddressedDecodeV2 as V2};
    let paths = [V2, V1, V2, V1, V1, V2];
    let map = indexed_row_map(&paths).unwrap();
    assert_eq!(map, [1, 3, 4, 0, 2, 5]);
    let mut unique = map.clone();
    unique.sort_unstable();
    assert_eq!(unique, (0..paths.len() as i32).collect::<Vec<_>>());
    assert_eq!(
        group_ranges(&paths).collect::<Vec<_>>(),
        [0..1, 1..2, 2..3, 3..5, 5..6]
    );
    assert_eq!(indexed_row_map(&[V1; 3]).unwrap(), [0, 1, 2]);
    assert_eq!(indexed_row_map(&[V2; 3]).unwrap(), [0, 1, 2]);
    assert!(indexed_row_map(&[]).is_err());
    assert!(indexed_row_map(&[V1, CausalAttentionKernelPath::VllmAddressedFallback]).is_err());
    assert!(indexed_row_map(&vec![V1; 65_536]).is_err());
}

#[cfg(feature = "cuda-indexed-causal-attention")]
#[test]
fn indexed_groups_preserve_original_ranges_and_exact_native_dispatch_geometry() {
    let shape = shape(256);
    let binding = BindingLayout::new(shape, 4).unwrap();
    let layout = ScratchLayout::for_participants(
        shape,
        4,
        4,
        CausalProjection::F16,
        AttentionExecutionPolicy::NativeAdaptive,
    )
    .unwrap();
    let launches = build_launches(shape, binding, layout, &[513, 512, 1025, 16]);
    let make = |launches: &[CausalAttentionLaunch], grouping| {
        BatchDecode::for_launches(launches, true, binding, shape, layout, true, true, grouping)
    };
    // Existing contiguous optimization stays ineligible for alternating paths.
    assert!(make(&launches, CausalDecodeAttentionGrouping::Contiguous)
        .unwrap()
        .is_none());
    let batch = make(&launches, CausalDecodeAttentionGrouping::Indexed)
        .unwrap()
        .unwrap();
    assert_eq!(batch.row_map.as_deref(), Some(&[1, 3, 0, 2][..]));
    assert_eq!(batch.v1_count, 2);
    assert_eq!(
        batch
            .groups
            .iter()
            .map(|(r, g)| (r.clone(), g.sequences, g.maximum_sequence))
            .collect::<Vec<_>>(),
        [(0..2, 2, 512), (2..4, 2, 1536)]
    );
    assert_eq!(batch.attention_dispatches(&launches), 3); // V1 plus V2 main/reduce.
    assert_eq!(batch.dispatch_count(&launches, false, false), 11);
    assert_eq!(batch.dispatch_count(&launches, true, false), 12);
    let mut invalid = launches.clone();
    invalid[1].packed_query = invalid[0].packed_query;
    assert!(make(&invalid, CausalDecodeAttentionGrouping::Indexed).is_err());
    let mut invalid = launches.clone();
    invalid[0].sequence_tokens_i32 = 512;
    assert!(make(&invalid, CausalDecodeAttentionGrouping::Indexed).is_err());
    assert!(BatchDecode::for_launches(
        &launches,
        false,
        binding,
        shape,
        layout,
        true,
        true,
        CausalDecodeAttentionGrouping::Indexed
    )
    .unwrap()
    .is_none());
}

#[cfg(all(
    feature = "vllm-paged-attn-v2",
    not(feature = "cuda-indexed-causal-attention")
))]
#[test]
fn indexed_groups_reject_old_native_build_without_changing_contiguous() {
    let shape = shape(128);
    let binding = BindingLayout::new(shape, 2).unwrap();
    let layout = ScratchLayout::for_participants(
        shape,
        2,
        2,
        CausalProjection::F16,
        AttentionExecutionPolicy::NativeAdaptive,
    )
    .unwrap();
    let launches = build_launches(shape, binding, layout, &[16, 32]);
    assert!(BatchDecode::for_launches(
        &launches,
        true,
        binding,
        shape,
        layout,
        true,
        true,
        CausalDecodeAttentionGrouping::Contiguous
    )
    .unwrap()
    .is_some());
    assert!(BatchDecode::for_launches(
        &launches,
        true,
        binding,
        shape,
        layout,
        true,
        true,
        CausalDecodeAttentionGrouping::Indexed
    )
    .is_err());
}

#[cfg(feature = "cuda-indexed-causal-attention")]
#[test]
fn indexed_replay_binds_stable_map_path_boundary_and_partition_envelope() {
    let shape = shape(128);
    let binding = BindingLayout::new(shape, 4).unwrap();
    let layout = ScratchLayout::for_participants(
        shape,
        4,
        4,
        CausalProjection::F16,
        AttentionExecutionPolicy::NativeAdaptive,
    )
    .unwrap();
    let key = |lengths: &[u64], grouping| {
        let launches = build_launches(shape, binding, layout, lengths);
        let batch = BatchDecode::for_launches(
            &launches, true, binding, shape, layout, true, true, grouping,
        )
        .unwrap()
        .unwrap();
        batch
            .bind_replay(CudaCommandReplayKeyBuilder::new("same-provider", "causal"))
            .finish()
            .bind_runtime_payload("causal", std::iter::empty(), &[])
    };
    let mode = CausalDecodeAttentionGrouping::Indexed;
    let base = key(&[16, 32, 1023, 16], mode);
    assert_eq!(base, key(&[17, 33, 1024, 17], mode)); // Fresh data, same geometry.
    assert_ne!(base, key(&[16, 1023, 32, 16], mode)); // Same counts, different row map.
    assert_ne!(
        key(&[512, 32, 1023, 16], mode),
        key(&[513, 32, 1023, 16], mode)
    );
    assert_ne!(base, key(&[16, 32, 1025, 16], mode)); // V2 partition envelope grows.
    assert_ne!(
        base,
        key(
            &[16, 32, 1023, 16],
            CausalDecodeAttentionGrouping::Contiguous
        )
    );
}

#[cfg(feature = "vllm-paged-attn-v2")]
mod gpu;

#[cfg(feature = "vllm-paged-attn-v2")]
mod packed_prepare_gate_gpu;

#[cfg(feature = "vllm-paged-attn-v2")]
fn build_launches(
    shape: CausalAttentionShape,
    binding: BindingLayout,
    layout: ScratchLayout,
    lengths: &[u64],
) -> Vec<CausalAttentionLaunch> {
    lengths
        .iter()
        .enumerate()
        .map(|(index, &length)| {
            let path = CausalAttentionKernelPath::select(
                AttentionExecutionPolicy::NativeAdaptive,
                shape,
                1,
                length,
            )
            .unwrap();
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
                replay_topology: CausalAttentionReplayTopology::new(shape, path, length).unwrap(),
                path,
            }
        })
        .collect()
}

#[test]
#[cfg(feature = "cuda-indexed-causal-attention")]
fn paired_query_heads_selects_only_supported_fixed_execution_shapes() {
    use ferrum_types::CausalDecodeKvReuse::{None, PairedQueryHeads};
    let grouping = CausalDecodeAttentionGrouping::FixedExecutionCapacity;
    let supported = shape(256);
    let eligible =
        |mode, grouping, shape| fixed::paired_query_heads_eligible(mode, grouping, shape);
    assert_eq!(
        eligible(PairedQueryHeads, grouping, supported),
        cfg!(feature = "cuda-causal-gqa2")
    );
    assert!(!eligible(None, grouping, supported));
    assert!(!eligible(
        PairedQueryHeads,
        CausalDecodeAttentionGrouping::FixedCapacity,
        supported
    ));
    assert!(!eligible(PairedQueryHeads, grouping, shape(128)));
    for length in [512, 2049] {
        let mut unsupported = supported;
        unsupported.maximum_context_tokens = length;
        assert!(!eligible(PairedQueryHeads, grouping, unsupported));
    }
    let mut unsupported = supported;
    unsupported.query_heads = 6;
    assert!(!eligible(PairedQueryHeads, grouping, unsupported));
    unsupported = supported;
    unsupported.int8_kv = true;
    assert!(!eligible(PairedQueryHeads, grouping, unsupported));
    // Product M1 deliberately retains the existing per-participant attention.
    let binding = BindingLayout::new(supported, 1).unwrap();
    let layout = ScratchLayout::for_participants(
        supported,
        1,
        1,
        CausalProjection::F16,
        AttentionExecutionPolicy::NativeAdaptive,
    )
    .unwrap();
    let launches = build_launches(supported, binding, layout, &[1024]);
    assert!(PreparedBatchDecode::for_launches(
        &launches,
        true,
        binding,
        supported,
        layout,
        true,
        true,
        grouping,
        PairedQueryHeads
    )
    .unwrap()
    .is_none());
}
