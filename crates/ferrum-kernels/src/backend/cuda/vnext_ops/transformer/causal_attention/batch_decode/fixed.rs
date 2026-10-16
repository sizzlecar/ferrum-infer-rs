//! Fixed-capacity Indexed replay. Each original row has one slot in
//! both maps; fresh device lengths choose exactly one of its existing V1/V2
//! algorithms. No CPU row permutation or sequence frontier enters the closure.
use super::*;
use crate::backend::cuda::vllm_paged_attn::dispatch_vnext_addressed_paged_attention_fixed_masked_batch_raw;

#[derive(Clone, Debug)]
pub(super) struct FixedBatchDecode {
    participants: i32,
    packed_work: Option<PackedDecodeWork>,
    v1: BatchGeometry,
    v2: Option<BatchGeometry>,
    paired_query_heads: bool,
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
            + if self.v2.is_some() { 3 } else { 1 }
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

    pub(super) fn with_kv_reuse(
        mut self,
        mode: ferrum_types::CausalDecodeKvReuse,
        grouping: CausalDecodeAttentionGrouping,
        shape: CausalAttentionShape,
    ) -> Self {
        self.paired_query_heads = paired_query_heads_eligible(mode, grouping, shape);
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
            .boolean(self.paired_query_heads);
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

    // Shared preparation/gate sequence. Production supplies the unchanged
    // adaptive gather and launches above; diagnostics may replace that entire
    // middle transaction, including its device-side fresh-map construction.
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
        && mode == ferrum_types::CausalDecodeKvReuse::PairedQueryHeads
        && grouping == CausalDecodeAttentionGrouping::FixedExecutionCapacity
        && !shape.int8_kv
        && shape.head_dim == 256
        && shape.key_value_heads > 0
        && shape.query_heads % shape.key_value_heads == 0
        && (shape.query_heads / shape.key_value_heads) % 2 == 0
        && (513..=2048).contains(&shape.maximum_context_tokens)
}
