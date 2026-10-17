//! Fixed-capacity Indexed replay. Each original row has one slot in
//! both maps; fresh device lengths choose the configured adaptive V1/V2 or
//! unified paired V2 path. No CPU row permutation or frontier enters the closure.
use super::*;
use crate::backend::cuda::vllm_paged_attn::dispatch_vnext_addressed_paged_attention_fixed_masked_batch_raw;

#[derive(Clone, Debug)]
pub(in super::super) struct FixedBatchDecode {
    participants: i32,
    packed_work: Option<PackedDecodeWork>,
    v1: BatchGeometry,
    v2: Option<BatchGeometry>,
    paired_query_heads: bool,
    unified_v2: bool,
}

impl FixedBatchDecode {
    pub(super) fn packed_work(&self) -> Option<PackedDecodeWork> {
        self.packed_work
    }

    pub(super) fn dispatch_count(
        &self,
        launches: &[CausalAttentionLaunch],
        output_gate: bool,
        post_norm: bool,
    ) -> u64 {
        6 + u64::from(post_norm)
            + if self
                .packed_work
                .and_then(PackedDecodeWork::prepare)
                .is_some()
            {
                1
            } else {
                launches.len() as u64
            }
            + 1
            + if self.unified_v2 {
                2
            } else if self.v2.is_some() {
                3
            } else {
                1
            }
            + if !output_gate {
                0
            } else if self
                .packed_work
                .and_then(PackedDecodeWork::gate_tokens)
                .is_some()
            {
                1
            } else {
                launches.len() as u64
            }
    }

    pub(super) fn for_launches(
        launches: &[CausalAttentionLaunch],
        packed: bool,
        binding: BindingLayout,
        shape: CausalAttentionShape,
        layout: ScratchLayout,
        packed_prepare_gate: bool,
        independent_sequence_kv: bool,
    ) -> Result<Option<Self>, String> {
        if shape.int8_kv {
            return Ok(None);
        }
        // Reuse the existing canonical packed-row, path, native-index, owner
        // extent and scratch validation. Only its packed preparation metadata
        // survives; its CPU map, V1 count and row capacities are discarded.
        let Some(validated) = BatchDecode::for_launches(
            launches,
            packed,
            binding,
            shape,
            layout,
            packed_prepare_gate,
            independent_sequence_kv,
            CausalDecodeAttentionGrouping::Indexed,
        )?
        else {
            return Ok(None);
        };
        let partition = layout.vllm.ok_or("fixed causal scratch absent")?;
        let participants = launches.len() as u64;
        let masks_end = partition
            .row_indices
            .checked_add(participants.checked_mul(8).ok_or("fixed maps overflow")?)
            .ok_or("fixed map endpoint overflows")?;
        let reservation_end = partition
            .sequence_lengths
            .checked_add(
                participants
                    .checked_mul(SCRATCH_ALIGNMENT)
                    .ok_or("fixed length reservation overflows")?,
            )
            .ok_or("fixed length reservation endpoint overflows")?;
        if masks_end > reservation_end || reservation_end > layout.required_bytes {
            return Err("fixed maps escape the admitted length reservation".into());
        }
        let dimensions = |capacity| {
            BatchDecode::dimensions(
                launches.len(),
                binding,
                capacity,
                shape.query_heads,
                shape.head_dim,
            )
        };
        Ok(Some(Self {
            paired_query_heads: false,
            unified_v2: false,
            participants: validated.participants,
            packed_work: validated.packed_work,
            v1: dimensions(VLLM_PARTITION_TOKENS.min(shape.maximum_context_tokens))?,
            v2: (shape.maximum_context_tokens > VLLM_PARTITION_TOKENS)
                .then(|| dimensions(shape.maximum_context_tokens))
                .transpose()?,
        }))
    }

    #[cfg(test)]
    pub(super) fn uses_paired_query_heads(&self) -> bool {
        self.paired_query_heads
    }

    #[cfg(test)]
    pub(super) fn uses_unified_v2(&self) -> bool {
        self.unified_v2
    }

    pub(super) fn with_kv_reuse(
        mut self,
        mode: ferrum_types::CausalDecodeKvReuse,
        grouping: CausalDecodeAttentionGrouping,
        shape: CausalAttentionShape,
    ) -> Self {
        self.paired_query_heads = paired_query_heads_eligible(mode, grouping, shape);
        self.unified_v2 = self.paired_query_heads
            && mode == ferrum_types::CausalDecodeKvReuse::PairedQueryHeadsUnifiedV2;
        self
    }

    pub(super) fn bind_replay(
        &self,
        mut key: CudaCommandReplayKeyBuilder,
    ) -> CudaCommandReplayKeyBuilder {
        key = key
            .bytes(b"ferrum.cuda.causal-fixed-capacity.v1")
            .i32(self.participants)
            .boolean(self.v2.is_some())
            .boolean(self.paired_query_heads)
            .boolean(self.unified_v2);
        for group in std::iter::once(&self.v1).chain(self.v2.iter()) {
            key = key
                .i32(group.sequences)
                .i32(group.table_stride)
                .u64(group.maximum_sequence);
        }
        key
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn enqueue(
        &self,
        stream: &CudaStream,
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
        #[cfg(feature = "cuda-causal-gqa2")]
        if self.unified_v2 {
            let group = self.v2.ok_or_else(|| {
                CudaDeviceRuntimeError::contract("unified causal V2 capacity absent")
            })?;
            let partition = layout.vllm.ok_or_else(|| {
                CudaDeviceRuntimeError::contract("unified causal V2 scratch absent")
            })?;
            return self.enqueue_with_gathered_attention(
                stream, functions, launches, binding, binding_layout, shape, layout,
                scratch, query_norm, key_norm, output_gate,
                |lengths, rows, unused_rows| {
                    let mut gather = stream.launch_builder(&functions.gather_decode_lengths_unified_v2);
                    gather.arg(&binding).arg(&binding_layout.slot_bytes).arg(&lengths)
                        .arg(&rows).arg(&unused_rows).arg(&self.participants);
                    unsafe { gather.launch(LaunchConfig::for_num_elems(self.participants as u32)) }
                        .map_err(|e| CudaDeviceRuntimeError::driver("causal unified V2 decode gather", e))?;
                    // The fresh gather supplies i for every positive length,
                    // including short rows, and -1 for inactive slots. Current
                    // bindings retain the admitted pages and capacity scratch.
                    unsafe {
                        crate::backend::cuda::vllm_paged_attn::dispatch_vnext_addressed_paged_attention_gqa2_unified_batch_raw(
                            stream, scratch_pointer(scratch, layout.context)?,
                            scratch_pointer(scratch, layout.query)?,
                            scratch_pointer(binding, BINDING_CONTROL_BYTES)?,
                            lengths, rows, group.maximum_sequence,
                            Some(scratch_pointer(scratch, partition.exp_sums)?),
                            Some(scratch_pointer(scratch, partition.max_logits)?),
                            Some(scratch_pointer(scratch, partition.temporary_output)?),
                            group.sequences, self.participants, shape.query_heads,
                            shape.key_value_heads, shape.head_dim, group.table_stride,
                        )
                    }.map(|_| ()).map_err(|e| CudaDeviceRuntimeError::contract(e.to_string()))
                },
            );
        }
        self.enqueue_with_attention(
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
                let partition = layout.vllm.ok_or_else(|| {
                    CudaDeviceRuntimeError::contract("fixed causal partition scratch absent")
                })?;
                #[cfg(feature = "cuda-causal-gqa2")]
                if self.paired_query_heads && expected == VnextAddressedPagedAttentionKernel::V2 {
                    return unsafe {
                        crate::backend::cuda::vllm_paged_attn::dispatch_vnext_addressed_paged_attention_gqa2_fixed_masked_batch_raw(
                            stream,
                            scratch_pointer(scratch, layout.context)?,
                            scratch_pointer(scratch, layout.query)?,
                            scratch_pointer(binding, BINDING_CONTROL_BYTES)?,
                            lengths, rows, group.maximum_sequence,
                            Some(scratch_pointer(scratch, partition.exp_sums)?),
                            Some(scratch_pointer(scratch, partition.max_logits)?),
                            Some(scratch_pointer(scratch, partition.temporary_output)?),
                            group.sequences, self.participants, shape.query_heads,
                            shape.key_value_heads, shape.head_dim, group.table_stride, expected,
                        )
                    }.map(|_| ()).map_err(|error| CudaDeviceRuntimeError::contract(error.to_string()));
                }
                // The same-stream gather establishes identity-or--1 masks.
                // Current bindings retain every original KV range/frontier.
                unsafe {
                    dispatch_vnext_addressed_paged_attention_fixed_masked_batch_raw(
                        stream,
                        scratch_pointer(scratch, layout.context)?,
                        scratch_pointer(scratch, layout.query)?,
                        scratch_pointer(binding, BINDING_CONTROL_BYTES)?,
                        lengths,
                        rows,
                        group.maximum_sequence,
                        Some(scratch_pointer(scratch, partition.exp_sums)?),
                        Some(scratch_pointer(scratch, partition.max_logits)?),
                        Some(scratch_pointer(scratch, partition.temporary_output)?),
                        group.sequences,
                        self.participants,
                        shape.query_heads,
                        shape.key_value_heads,
                        shape.head_dim,
                        group.table_stride,
                        expected,
                    )
                }
                .map(|_| ())
                .map_err(|error| CudaDeviceRuntimeError::contract(error.to_string()))
            },
        )
    }

    // Static dispatch keeps the normal path unchanged. The test-only caller can
    // substitute a V2 launcher while sharing this entire preparation/gate bundle.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn enqueue_with_attention<F>(
        &self,
        stream: &CudaStream,
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
        mut attention: F,
    ) -> Result<(), CudaDeviceRuntimeError>
    where
        F: FnMut(
            BatchGeometry,
            u64,
            u64,
            VnextAddressedPagedAttentionKernel,
        ) -> Result<(), CudaDeviceRuntimeError>,
    {
        self.enqueue_with_gathered_attention(
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
            |lengths, v1_rows, v2_rows| {
                let mut gather =
                    stream.launch_builder(&functions.gather_decode_lengths_fixed_capacity);
                gather
                    .arg(&binding)
                    .arg(&binding_layout.slot_bytes)
                    .arg(&lengths)
                    .arg(&v1_rows)
                    .arg(&v2_rows)
                    .arg(&self.participants);
                unsafe { gather.launch(LaunchConfig::for_num_elems(self.participants as u32)) }
                    .map_err(|error| {
                        CudaDeviceRuntimeError::driver("causal fixed-capacity decode gather", error)
                    })?;
                for (group, rows, expected) in
                    std::iter::once((&self.v1, v1_rows, VnextAddressedPagedAttentionKernel::V1))
                        .chain(
                            self.v2.as_ref().map(|group| {
                                (group, v2_rows, VnextAddressedPagedAttentionKernel::V2)
                            }),
                        )
                {
                    attention(*group, lengths, rows, expected)?;
                }
                Ok(())
            },
        )
    }

    // Shared preparation/gate sequence for the adaptive and unified product
    // paths, including each path's device-side fresh-map construction.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn enqueue_with_gathered_attention<F>(
        &self,
        stream: &CudaStream,
        functions: &CausalAttentionFunctions,
        launches: &[CausalAttentionLaunch],
        binding: u64,
        _binding_layout: BindingLayout,
        shape: CudaCausalAttentionShape,
        layout: ScratchLayout,
        scratch: u64,
        query_norm: u64,
        key_norm: u64,
        output_gate: bool,
        mut gathered_attention: F,
    ) -> Result<(), CudaDeviceRuntimeError>
    where
        F: FnMut(u64, u64, u64) -> Result<(), CudaDeviceRuntimeError>,
    {
        if launches.len() != self.participants as usize {
            return Err(CudaDeviceRuntimeError::contract(
                "fixed causal participant count changed",
            ));
        }
        if let Some(prepare) = self.packed_work.and_then(PackedDecodeWork::prepare) {
            launch_prepare(
                stream,
                &functions.prepare,
                scratch_pointer(scratch, layout.query_raw)?,
                scratch_pointer(scratch, layout.key_raw)?,
                scratch_pointer(scratch, layout.value_raw)?,
                query_norm,
                key_norm,
                scratch_pointer(scratch, layout.query)?,
                binding,
                scratch_pointer(binding, BINDING_CONTROL_BYTES)?,
                launches[0],
                shape,
                1,
                Some(prepare),
            )?;
        } else {
            for launch in launches {
                let control = scratch_pointer(binding, launch.binding_offset)?;
                launch_prepare(
                    stream,
                    &functions.prepare,
                    scratch_pointer(scratch, launch.packed_query_raw)?,
                    scratch_pointer(scratch, launch.packed_key_raw)?,
                    scratch_pointer(scratch, launch.packed_value_raw)?,
                    query_norm,
                    key_norm,
                    scratch_pointer(scratch, launch.packed_query)?,
                    control,
                    scratch_pointer(control, BINDING_CONTROL_BYTES)?,
                    *launch,
                    shape,
                    1,
                    None,
                )?;
            }
        }
        let partition = layout.vllm.ok_or_else(|| {
            CudaDeviceRuntimeError::contract("fixed causal partition scratch absent")
        })?;
        let lengths = scratch_pointer(scratch, partition.sequence_lengths)?;
        let v1_rows = scratch_pointer(scratch, partition.row_indices)?;
        let v2_rows = scratch_pointer(v1_rows, self.participants as u64 * 4)?;
        gathered_attention(lengths, v1_rows, v2_rows)?;
        if output_gate {
            if let Some(tokens) = self.packed_work.and_then(PackedDecodeWork::gate_tokens) {
                launch_attention_gate(
                    stream,
                    &functions.attention_gate,
                    scratch_pointer(scratch, layout.context)?,
                    scratch_pointer(scratch, layout.query_raw)?,
                    tokens,
                    shape,
                )?;
            } else {
                for launch in launches {
                    launch_attention_gate(
                        stream,
                        &functions.attention_gate,
                        scratch_pointer(scratch, launch.packed_context)?,
                        scratch_pointer(scratch, launch.packed_query_raw)?,
                        launch.tokens,
                        shape,
                    )?;
                }
            }
        }
        Ok(())
    }
}

// Qualification is per provider shape; original V1/V2 remains the fallback.
pub(super) fn paired_query_heads_eligible(
    mode: ferrum_types::CausalDecodeKvReuse,
    grouping: CausalDecodeAttentionGrouping,
    shape: CausalAttentionShape,
) -> bool {
    cfg!(feature = "cuda-causal-gqa2")
        && matches!(
            mode,
            ferrum_types::CausalDecodeKvReuse::PairedQueryHeads
                | ferrum_types::CausalDecodeKvReuse::PairedQueryHeadsUnifiedV2
        )
        && grouping == CausalDecodeAttentionGrouping::FixedExecutionCapacity
        && !shape.int8_kv
        && shape.head_dim == 256
        && shape.key_value_heads > 0
        && shape.query_heads % shape.key_value_heads == 0
        && (shape.query_heads / shape.key_value_heads) % 2 == 0
        && (513..=2048).contains(&shape.maximum_context_tokens)
}
