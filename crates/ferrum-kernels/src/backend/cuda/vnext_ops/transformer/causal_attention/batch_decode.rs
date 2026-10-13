//! Batch rows of the same existing addressed V1 or V2 math.
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
    packed_work: Option<PackedDecodeWork>,
    // Group-local ordinal -> original packed row, without moving query or KV.
    row_map: Option<Vec<i32>>,
    v1_count: i32,
}

fn indexed_row_map(paths: &[CausalAttentionKernelPath]) -> Result<Vec<i32>, String> {
    if paths.is_empty() || paths.len() > 65_535 {
        return Err("indexed causal rows exceed native grid bounds".into());
    }
    let mut rows = Vec::with_capacity(paths.len());
    for selected in [
        CausalAttentionKernelPath::VllmAddressedDecodeV1,
        CausalAttentionKernelPath::VllmAddressedDecodeV2,
    ] {
        for (index, path) in paths.iter().enumerate() {
            if *path == selected {
                rows.push(i32::try_from(index).map_err(|_| "causal row index exceeds i32")?);
            }
        }
    }
    if rows.len() != paths.len() {
        return Err("indexed causal group contains a non-decode path".into());
    }
    Ok(rows)
}

#[derive(Clone, Copy, Debug)]
pub(super) struct PackedDecodeWork {
    prepare: Option<PackedPrepareLaunch>,
    gate_tokens: Option<u64>,
}

impl PackedDecodeWork {
    fn new(
        participants: usize,
        binding: BindingLayout,
        shape: CausalAttentionShape,
        independent_sequence_kv: bool,
    ) -> Option<Self> {
        if participants == 0
            || binding.slot_bytes == 0
            || binding.slot_bytes.checked_mul(participants as u64) != Some(binding.required_bytes)
        {
            return None;
        }
        Some(Self {
            // Only actual participant-owned Sequence views prove independent
            // mutable KV claims. Request/Step views can alias across rows.
            // Preparation's grid-z bound does not constrain the flat gate.
            prepare: (independent_sequence_kv && participants <= 65_535).then_some(
                PackedPrepareLaunch {
                    token_grid: 1,
                    participant_grid: participants as u32,
                    binding_slot_bytes: binding.slot_bytes,
                },
            ),
            gate_tokens: packed_gate_tokens(
                participants as u64,
                shape.query_features,
                shape.query_projection_features,
            ),
        })
    }

    pub(super) fn prepare(self) -> Option<PackedPrepareLaunch> {
        self.prepare
    }

    pub(super) fn gate_tokens(self) -> Option<u64> {
        self.gate_tokens
    }
}

pub(super) fn bind_prepare_gate_replay(
    key: CudaCommandReplayKeyBuilder,
    work: Option<PackedDecodeWork>,
    output_gate: bool,
) -> CudaCommandReplayKeyBuilder {
    // Provider mode alone is insufficient: legal KV ownership and native
    // index limits can select different captured kernels under the same mode.
    key.boolean(work.and_then(PackedDecodeWork::prepare).is_some())
        .u64(if output_gate {
            work.and_then(PackedDecodeWork::gate_tokens).unwrap_or(0)
        } else {
            0
        })
}

fn packed_gate_tokens(tokens: u64, query_features: u64, projection_features: u64) -> Option<u64> {
    if tokens == 0 || query_features == 0 || projection_features == 0 {
        return None;
    }
    let elements = tokens.checked_mul(query_features)?;
    let launched_threads = elements
        .div_ceil(u64::from(THREADS_PER_BLOCK))
        .checked_mul(u64::from(THREADS_PER_BLOCK))?;
    let projection_elements = tokens.checked_mul(projection_features)?;
    // The native gate uses signed i32 for idx, total and gate_idx. Include
    // padded threads in the bound, even though valid threads alone may fit.
    (tokens <= i32::MAX as u64
        && launched_threads <= i32::MAX as u64
        && projection_elements <= i32::MAX as u64)
        .then_some(tokens)
}

#[derive(Clone, Copy, Debug)]
pub(super) struct BatchGeometry {
    sequences: i32,
    table_stride: i32,
    maximum_sequence: u64,
}

impl BatchDecode {
    pub(super) fn packed_work(&self) -> Option<PackedDecodeWork> {
        self.packed_work
    }

    pub(super) fn attention_dispatches(&self, launches: &[CausalAttentionLaunch]) -> u64 {
        self.groups
            .iter()
            .map(|(range, _)| {
                let row = self
                    .row_map
                    .as_ref()
                    .map_or(range.start, |map| map[range.start] as usize);
                launches[row].path.attention_dispatch_count()
            })
            .sum()
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
            + self.attention_dispatches(launches)
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

    pub(super) fn bind_replay(
        &self,
        mut key: CudaCommandReplayKeyBuilder,
    ) -> CudaCommandReplayKeyBuilder {
        key = key
            .boolean(self.row_map.is_some())
            .i32(self.participants)
            .i32(self.v1_count);
        if let Some(map) = &self.row_map {
            for &row in map {
                key = key.i32(row);
            }
        }
        key = key.u64(self.groups.len() as u64);
        for (range, geometry) in &self.groups {
            key = key
                .u64(range.start as u64)
                .u64(range.end as u64)
                .i32(geometry.sequences)
                .i32(geometry.table_stride)
                .u64(geometry.maximum_sequence);
        }
        key
    }

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
        packed_prepare_gate: bool,
        independent_sequence_kv: bool,
        grouping: CausalDecodeAttentionGrouping,
    ) -> Result<Option<Self>, String> {
        let indexed = grouping == CausalDecodeAttentionGrouping::Indexed;
        if indexed && !cfg!(feature = "cuda-indexed-causal-attention") {
            return Err(
                "indexed causal attention requires the cuda-indexed-causal-attention build feature"
                    .into(),
            );
        }
        let can_batch = if indexed {
            packed
                && launches.len() > 1
                && launches.iter().all(|launch| {
                    matches!(
                        launch.path,
                        CausalAttentionKernelPath::VllmAddressedDecodeV1
                            | CausalAttentionKernelPath::VllmAddressedDecodeV2
                    )
                })
        } else {
            eligible(launches.iter().map(|launch| launch.path), packed)
        };
        if !can_batch {
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
            if indexed
                && (launch.sequence_tokens_i32 <= 0
                    || launch.sequence_tokens != launch.sequence_tokens_i32 as u64
                    || (launch.sequence_tokens <= VLLM_PARTITION_TOKENS)
                        != (launch.path == CausalAttentionKernelPath::VllmAddressedDecodeV1))
            {
                return Err("indexed causal row length disagrees with its native path".into());
            }
            if packed_prepare_gate
                && (launch.tokens_i32 != 1
                    || launch.packed_query_raw
                        != layout.token_offset(
                            layout.query_raw,
                            index as u64,
                            shape.query_projection_features,
                        )?
                    || launch.packed_key_raw
                        != layout.token_offset(layout.key_raw, index as u64, shape.kv_features)?
                    || launch.packed_value_raw
                        != layout.token_offset(
                            layout.value_raw,
                            index as u64,
                            shape.kv_features,
                        )?)
            {
                return Err("packed causal preparation rows differ from scratch geometry".into());
            }
        }
        if binding.slot_bytes.checked_mul(launches.len() as u64) != Some(binding.required_bytes) {
            return Err("causal attention batch owner extent changed".into());
        }
        let paths = launches
            .iter()
            .map(|launch| launch.path)
            .collect::<Vec<_>>();
        let row_map = indexed.then(|| indexed_row_map(&paths)).transpose()?;
        if indexed {
            // Indexed query/output/table strides address original packed rows,
            // while partition scratch below addresses only the selected group.
            let rows = launches.len() as u64;
            if rows
                .checked_mul(shape.query_heads)
                .and_then(|n| n.checked_mul(shape.head_dim))
                .is_none_or(|n| n > i32::MAX as u64)
                || rows
                    .checked_mul(binding.slot_bytes / POINTER_BYTES)
                    .is_none_or(|n| n > i32::MAX as u64)
            {
                return Err("indexed causal original-row indexing exceeds i32".into());
            }
        }
        let v1_count = if indexed {
            i32::try_from(
                paths
                    .iter()
                    .filter(|&&path| path == CausalAttentionKernelPath::VllmAddressedDecodeV1)
                    .count(),
            )
            .map_err(|_| "causal V1 count exceeds i32")?
        } else {
            0
        };
        let ranges = if indexed {
            [0..v1_count as usize, v1_count as usize..paths.len()]
                .into_iter()
                .filter(|range| !range.is_empty())
                .collect::<Vec<_>>()
        } else {
            group_ranges(&paths).collect()
        };
        let groups = ranges
            .into_iter()
            .map(|range| {
                let maximum_sequence = range
                    .clone()
                    .map(|index| row_map.as_ref().map_or(index, |map| map[index] as usize))
                    .map(|index| {
                        launches[index]
                            .replay_topology
                            .envelope()
                            .sequence_capacity_tokens
                    })
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
            row_map,
            v1_count,
            participants: i32::try_from(launches.len())
                .map_err(|_| "causal batch owner count exceeds i32")?,
            packed_work: packed_prepare_gate
                .then(|| {
                    PackedDecodeWork::new(launches.len(), binding, shape, independent_sequence_kv)
                })
                .flatten(),
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
            let partition = layout
                .vllm
                .ok_or_else(|| CudaDeviceRuntimeError::contract("batch scratch absent"))?;
            let lengths = scratch_pointer(scratch, partition.sequence_lengths)?;
            let row_indices = scratch_pointer(scratch, partition.row_indices)?;
            if self.row_map.is_some() {
                gather_lengths_indexed(
                    stream,
                    &functions.gather_decode_lengths_indexed,
                    binding,
                    binding_layout.slot_bytes,
                    lengths,
                    row_indices,
                    self.participants,
                    self.v1_count,
                )?;
            } else {
                gather_lengths(
                    stream,
                    &functions.gather_decode_lengths,
                    binding,
                    binding_layout.slot_bytes,
                    lengths,
                    self.participants,
                )?;
            }
            for (range, group) in &self.groups {
                let original = self
                    .row_map
                    .as_ref()
                    .map_or(range.start, |map| map[range.start] as usize);
                let first = launches[original];
                let group_control = scratch_pointer(binding, first.binding_offset)?;
                let group_lengths = scratch_pointer(lengths, (range.start as u64) * 4)?;
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
                let actual = unsafe {
                    if self.row_map.is_some() {
                        #[cfg(feature = "cuda-indexed-causal-attention")]
                        {
                        crate::backend::cuda::vllm_paged_attn::dispatch_vnext_addressed_paged_attention_indexed_batch_raw(
                            stream, scratch_pointer(scratch, layout.context)?, scratch_pointer(scratch, layout.query)?,
                            scratch_pointer(binding, BINDING_CONTROL_BYTES)?, lengths,
                            scratch_pointer(row_indices, (range.start as u64) * 4)?, group.maximum_sequence,
                            Some(scratch_pointer(scratch, partition.exp_sums)?),
                            Some(scratch_pointer(scratch, partition.max_logits)?),
                            Some(scratch_pointer(scratch, partition.temporary_output)?),
                            group.sequences, self.participants, shape.query_heads, shape.key_value_heads, shape.head_dim,
                            group.table_stride, expected)
                        }
                        #[cfg(not(feature = "cuda-indexed-causal-attention"))]
                        {
                            return Err(CudaDeviceRuntimeError::contract(
                                "indexed causal attention requires the cuda-indexed-causal-attention build feature",
                            ));
                        }
                    } else {
                        crate::backend::cuda::vllm_paged_attn::dispatch_vnext_addressed_paged_attention_batch_raw(
                            stream, scratch_pointer(scratch, first.packed_context)?, scratch_pointer(scratch, first.packed_query)?,
                            scratch_pointer(group_control, BINDING_CONTROL_BYTES)?, group_lengths, group.maximum_sequence,
                            Some(scratch_pointer(scratch, partition.exp_sums)?),
                            Some(scratch_pointer(scratch, partition.max_logits)?),
                            Some(scratch_pointer(scratch, partition.temporary_output)?),
                            group.sequences, shape.query_heads, shape.key_value_heads, shape.head_dim, group.table_stride)
                    }
                }.map_err(|error| CudaDeviceRuntimeError::contract(error.to_string()))?;
                if actual != expected {
                    return Err(CudaDeviceRuntimeError::contract(
                        "batched attention changed native kernel",
                    ));
                }
            }
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
        #[cfg(not(feature = "vllm-paged-attn-v2"))]
        {
            let _ = (
                &self.groups,
                self.participants,
                stream,
                &functions.gather_decode_lengths,
                &functions.gather_decode_lengths_indexed,
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
#[allow(clippy::too_many_arguments)]
pub(super) fn gather_lengths_indexed(
    stream: &CudaStream,
    function: &CudaFunction,
    binding: u64,
    slot_bytes: u64,
    lengths: u64,
    row_indices: u64,
    sequences: i32,
    v1_count: i32,
) -> Result<(), CudaDeviceRuntimeError> {
    let mut launch = stream.launch_builder(function);
    launch
        .arg(&binding)
        .arg(&slot_bytes)
        .arg(&lengths)
        .arg(&row_indices)
        .arg(&sequences)
        .arg(&v1_count);
    unsafe { launch.launch(LaunchConfig::for_num_elems(sequences as u32)) }.map_err(|error| {
        CudaDeviceRuntimeError::driver("causal indexed decode length gather", error)
    })?;
    Ok(())
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

#[cfg(feature = "cuda-indexed-causal-attention")]
pub(super) mod fixed;

pub(super) enum PreparedBatchDecode {
    Grouped(BatchDecode),
    #[cfg(feature = "cuda-indexed-causal-attention")]
    Fixed(fixed::FixedBatchDecode),
}

impl PreparedBatchDecode {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn for_launches(
        launches: &[CausalAttentionLaunch],
        packed: bool,
        binding: BindingLayout,
        shape: CausalAttentionShape,
        layout: ScratchLayout,
        packed_prepare_gate: bool,
        independent_sequence_kv: bool,
        grouping: CausalDecodeAttentionGrouping,
    ) -> Result<Option<Self>, String> {
        if matches!(
            grouping,
            CausalDecodeAttentionGrouping::FixedCapacity
                | CausalDecodeAttentionGrouping::FixedExecutionCapacity
        ) {
            #[cfg(feature = "cuda-indexed-causal-attention")]
            return fixed::FixedBatchDecode::for_launches(
                launches,
                packed,
                binding,
                shape,
                layout,
                packed_prepare_gate,
                independent_sequence_kv,
            )
            .map(|batch| batch.map(Self::Fixed));
            #[cfg(not(feature = "cuda-indexed-causal-attention"))]
            return Err("fixed causal replay requires compiled indexed attention".into());
        }
        BatchDecode::for_launches(
            launches,
            packed,
            binding,
            shape,
            layout,
            packed_prepare_gate,
            independent_sequence_kv,
            grouping,
        )
        .map(|batch| batch.map(Self::Grouped))
    }

    pub(super) fn is_fixed(&self) -> bool {
        match self {
            Self::Grouped(_) => false,
            #[cfg(feature = "cuda-indexed-causal-attention")]
            Self::Fixed(_) => true,
        }
    }

    pub(super) fn packed_work(&self) -> Option<PackedDecodeWork> {
        match self {
            Self::Grouped(batch) => batch.packed_work(),
            #[cfg(feature = "cuda-indexed-causal-attention")]
            Self::Fixed(batch) => batch.packed_work(),
        }
    }

    pub(super) fn dispatch_count(
        &self,
        launches: &[CausalAttentionLaunch],
        output_gate: bool,
        post_norm: bool,
    ) -> u64 {
        match self {
            Self::Grouped(batch) => batch.dispatch_count(launches, output_gate, post_norm),
            #[cfg(feature = "cuda-indexed-causal-attention")]
            Self::Fixed(batch) => batch.dispatch_count(launches, output_gate, post_norm),
        }
    }

    pub(super) fn bind_replay(
        &self,
        key: CudaCommandReplayKeyBuilder,
    ) -> CudaCommandReplayKeyBuilder {
        match self {
            Self::Grouped(batch) => batch.bind_replay(key),
            #[cfg(feature = "cuda-indexed-causal-attention")]
            Self::Fixed(batch) => batch.bind_replay(key),
        }
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
        match self {
            Self::Grouped(batch) => batch.enqueue(
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
            ),
            #[cfg(feature = "cuda-indexed-causal-attention")]
            Self::Fixed(batch) => batch.enqueue(
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
            ),
        }
    }
}
