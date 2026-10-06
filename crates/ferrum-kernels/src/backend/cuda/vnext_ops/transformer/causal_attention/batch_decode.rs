//! Batch contiguous rows of the same existing addressed V1 or V2 math.
//! Mixed paths retain their selected algorithm and original row order. Physical KV references remain in
//! the admitted program binding; the gather reads its current lengths on every
//! replay. No host frontier, pointer, or sequence length is cached here.
use super::*;

pub(super) fn eligible(
    paths: impl IntoIterator<Item = CausalAttentionKernelPath>,
    packed: bool,
) -> bool {
    if !packed {
        return false;
    }
    let mut previous = None;
    let mut adjacent_pair = false;
    for path in paths {
        if !matches!(
            path,
            CausalAttentionKernelPath::VllmAddressedDecodeV1
                | CausalAttentionKernelPath::VllmAddressedDecodeV2
        ) {
            return false;
        }
        adjacent_pair |= previous == Some(path);
        previous = Some(path);
    }
    adjacent_pair
}

/// Maximal contiguous runs keep the native ABI's dense query/output rows and
/// binding stride. No request reordering, additional KV copy, or V1→V2 switch.
pub(super) fn group_ranges(
    paths: &[CausalAttentionKernelPath],
) -> impl Iterator<Item = std::ops::Range<usize>> + '_ {
    let mut start = 0;
    std::iter::from_fn(move || {
        if start == paths.len() {
            return None;
        }
        let begin = start;
        start += 1;
        while start < paths.len() && paths[start] == paths[begin] {
            start += 1;
        }
        Some(begin..start)
    })
}

#[derive(Clone, Debug)]
pub(super) struct BatchDecode {
    groups: Vec<(std::ops::Range<usize>, BatchGeometry)>,
    participants: i32,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct BatchGeometry {
    sequences: i32,
    table_stride: i32,
    maximum_sequence: u64,
}

impl BatchDecode {
    #[cfg(test)]
    fn maximum_sequence(&self) -> u64 {
        self.groups
            .iter()
            .map(|(_, group)| group.maximum_sequence)
            .max()
            .unwrap_or(0)
    }

    pub(super) fn dimensions(
        participants: usize,
        binding: BindingLayout,
        maximum_sequence: u64,
        heads: u64,
        head_dim: u64,
    ) -> Result<BatchGeometry, String> {
        let sequences = i32::try_from(participants)
            .map_err(|_| "causal attention batch exceeds i32 sequences")?;
        if sequences < 1 || sequences > 65_535 {
            return Err("causal attention batch exceeds native grid-y bounds".into());
        }
        if maximum_sequence == 0
            || maximum_sequence > (i32::MAX - 511) as u64
            || binding.slot_bytes % POINTER_BYTES != 0
        {
            return Err("causal attention batched decode envelope is invalid".into());
        }
        let table_bytes = maximum_sequence
            .div_ceil(VLLM_BLOCK_TOKENS)
            .checked_mul(POINTER_BYTES)
            .and_then(|bytes| bytes.checked_add(BINDING_CONTROL_BYTES))
            .ok_or("causal attention batch table extent overflows")?;
        if table_bytes > binding.slot_bytes
            || binding.slot_bytes.checked_mul(participants as u64) != Some(binding.required_bytes)
        {
            return Err("causal attention batched table escapes admitted binding slots".into());
        }
        let table_stride = i32::try_from(binding.slot_bytes / POINTER_BYTES)
            .map_err(|_| "causal attention table stride exceeds i32")?;
        let rows = (participants as u64)
            .checked_mul(heads)
            .and_then(|n| n.checked_mul(maximum_sequence.div_ceil(VLLM_PARTITION_TOKENS)))
            .and_then(|n| n.checked_mul(head_dim))
            .ok_or("causal attention batched partition indexing overflows")?;
        if rows > i32::MAX as u64 || sequences.checked_mul(table_stride).is_none() {
            return Err("causal attention batched native indexing exceeds i32".into());
        }
        Ok(BatchGeometry {
            sequences,
            table_stride,
            maximum_sequence,
        })
    }

    pub(super) fn for_launches(
        launches: &[CausalAttentionLaunch],
        packed: bool,
        binding: BindingLayout,
        shape: CausalAttentionShape,
        layout: ScratchLayout,
    ) -> Result<Option<Self>, String> {
        if !eligible(launches.iter().map(|launch| launch.path), packed) {
            return Ok(None);
        }
        if !layout
            .vllm
            .is_some_and(|scratch| scratch.participants >= launches.len() as u64)
        {
            return Err("causal attention batch lacks admitted partition scratch".into());
        }
        for (index, launch) in launches.iter().enumerate() {
            if launch.replay_topology
                != CausalAttentionReplayTopology::new(shape, launch.path, launch.sequence_tokens)?
                || launch.tokens != 1
                || launch.packed_token_start != index as u64
                || launch.binding_offset != binding.binding_offset(index)?
                || launch.packed_query
                    != layout.token_offset(layout.query, index as u64, shape.query_features)?
                || launch.packed_context
                    != layout.token_offset(layout.context, index as u64, shape.query_features)?
            {
                return Err(
                    "causal attention batch rows are not canonical packed decode rows".into(),
                );
            }
        }
        if binding.slot_bytes.checked_mul(launches.len() as u64) != Some(binding.required_bytes) {
            return Err("causal attention batch owner extent changed".into());
        }
        let paths = launches
            .iter()
            .map(|launch| launch.path)
            .collect::<Vec<_>>();
        let groups = group_ranges(&paths)
            .map(|range| {
                let maximum_sequence = launches[range.clone()]
                    .iter()
                    .map(|launch| launch.replay_topology.envelope().sequence_capacity_tokens)
                    .max()
                    .ok_or("causal attention batch group is empty")?;
                let local_binding = BindingLayout {
                    slot_bytes: binding.slot_bytes,
                    required_bytes: binding
                        .slot_bytes
                        .checked_mul(range.len() as u64)
                        .ok_or("causal attention batch group binding overflows")?,
                };
                let dimensions = Self::dimensions(
                    range.len(),
                    local_binding,
                    maximum_sequence,
                    shape.query_heads,
                    shape.head_dim,
                )?;
                Ok((range, dimensions))
            })
            .collect::<Result<Vec<_>, String>>()?;
        Ok(Some(Self {
            groups,
            participants: i32::try_from(launches.len())
                .map_err(|_| "causal batch owner count exceeds i32")?,
        }))
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
        #[cfg(feature = "vllm-paged-attn-v2")]
        {
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
            let partition = layout
                .vllm
                .ok_or_else(|| CudaDeviceRuntimeError::contract("batch scratch absent"))?;
            let lengths = scratch_pointer(scratch, partition.sequence_lengths)?;
            gather_lengths(
                stream,
                &functions.gather_decode_lengths,
                binding,
                binding_layout.slot_bytes,
                lengths,
                self.participants,
            )?;
            for (range, group) in &self.groups {
                let first = launches[range.start];
                let group_control = scratch_pointer(binding, first.binding_offset)?;
                let group_lengths = scratch_pointer(lengths, (range.start as u64) * 4)?;
                let actual = unsafe {
                crate::backend::cuda::vllm_paged_attn::dispatch_vnext_addressed_paged_attention_batch_raw(
                    stream, scratch_pointer(scratch, first.packed_context)?, scratch_pointer(scratch, first.packed_query)?,
                    scratch_pointer(group_control, BINDING_CONTROL_BYTES)?, group_lengths, group.maximum_sequence,
                    Some(scratch_pointer(scratch, partition.exp_sums)?),
                    Some(scratch_pointer(scratch, partition.max_logits)?),
                    Some(scratch_pointer(scratch, partition.temporary_output)?),
                    group.sequences, shape.query_heads, shape.key_value_heads, shape.head_dim, group.table_stride)
            }.map_err(|error| CudaDeviceRuntimeError::contract(error.to_string()))?;
                let expected = match first.path {
                    CausalAttentionKernelPath::VllmAddressedDecodeV1 => {
                        VnextAddressedPagedAttentionKernel::V1
                    }
                    CausalAttentionKernelPath::VllmAddressedDecodeV2 => {
                        VnextAddressedPagedAttentionKernel::V2
                    }
                    _ => {
                        return Err(CudaDeviceRuntimeError::contract(
                            "batch contains a non-decode path",
                        ))
                    }
                };
                if actual != expected {
                    return Err(CudaDeviceRuntimeError::contract(
                        "batched attention changed native kernel",
                    ));
                }
            }
            if output_gate {
                for launch in launches {
                    launch_attention_gate(
                        stream,
                        &functions.attention_gate,
                        scratch_pointer(scratch, launch.packed_context)?,
                        scratch_pointer(scratch, launch.packed_query_raw)?,
                        *launch,
                        shape,
                    )?;
                }
            }
            Ok(())
        }
        #[cfg(not(feature = "vllm-paged-attn-v2"))]
        {
            let _ = (
                &self.groups,
                self.participants,
                stream,
                &functions.gather_decode_lengths,
                launches,
                binding,
                binding_layout,
                shape,
                layout,
                scratch,
                query_norm,
                key_norm,
                output_gate,
            );
            Err(CudaDeviceRuntimeError::contract(
                "batched addressed attention requires vllm-paged-attn-v2",
            ))
        }
    }
}

#[cfg(feature = "vllm-paged-attn-v2")]
pub(super) fn gather_lengths(
    stream: &CudaStream,
    function: &CudaFunction,
    binding: u64,
    slot_bytes: u64,
    lengths: u64,
    sequences: i32,
) -> Result<(), CudaDeviceRuntimeError> {
    let mut launch = stream.launch_builder(function);
    launch
        .arg(&binding)
        .arg(&slot_bytes)
        .arg(&lengths)
        .arg(&sequences);
    unsafe { launch.launch(LaunchConfig::for_num_elems(sequences as u32)) }
        .map_err(|error| CudaDeviceRuntimeError::driver("causal decode length gather", error))?;
    Ok(())
}

#[cfg(test)]
mod tests;
