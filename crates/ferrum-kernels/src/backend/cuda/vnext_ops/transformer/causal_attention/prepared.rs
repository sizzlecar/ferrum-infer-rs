//! The actual physical preparation, shared by command encoding and read-only
//! replay evidence. This owns metadata and retained regions, never an execution
//! permit; all original shape, alias, workspace, and page checks still run.
use super::*;

pub(super) struct Prepared {
    pub shape: CausalAttentionShape,
    pub projection: CausalProjection,
    pub total_tokens: u64,
    pub layout: ScratchLayout,
    pub binding_layout: BindingLayout,
    pub cuda: CudaCausalAttentionShape,
    pub shared: SharedRegions,
    pub packed: Option<PackedCausalAttentionLaunch>,
    pub compute_regions: Vec<CudaBufferRegion>,
    pub binding_regions: Vec<CudaBufferRegion>,
    pub compute_fence_dependencies: Vec<CudaBufferRegion>,
    pub host_storage: Vec<Box<[u8]>>,
    pub launches: Vec<CausalAttentionLaunch>,
    pub bindings: Vec<CausalAttentionBinding>,
    pub participant_count: u32,
}

pub(super) fn prepare(
    attention_policy: AttentionExecutionPolicy,
    semantics: CausalAttentionSemantics,
    precision: CausalPrecision,
    operation_id: &str,
    #[cfg(feature = "vllm-marlin")] projection_runtime: MarlinProjectionRuntime,
    invocation: &BatchedOperationInvocation<'_, CudaDeviceBuffer>,
) -> Result<Prepared, String> {
    if invocation.participants().is_empty() || invocation.operation().id.as_str() != operation_id {
        return Err("CUDA causal attention received another or empty operation".to_owned());
    }
    super::super::gguf_f16_projection::validate_invocation(invocation)?;
    let first = &invocation.participants()[0];
    let shape = CausalAttentionShape::from_attributes_for(first.attributes(), semantics)?;
    let projection = CausalProjection::from_values(
        first.bindings(),
        #[cfg(feature = "vllm-marlin")]
        projection_runtime,
    )?;
    validate_signature(first, shape, semantics, precision)?;
    for participant in &invocation.participants()[1..] {
        if CausalAttentionShape::from_attributes_for(participant.attributes(), semantics)? != shape
        {
            return Err("CUDA causal attention participant attributes disagree".to_owned());
        }
        #[cfg(feature = "vllm-marlin")]
        if CausalProjection::from_values(participant.bindings(), projection_runtime)?.replay_tag()
            != projection.replay_tag()
        {
            return Err(
                "CUDA causal attention participants use different projection ABIs".to_owned(),
            );
        }
        validate_signature(participant, shape, semantics, precision)?;
    }

    let total_tokens = invocation.work_shape().immediate_tokens();
    let layout = ScratchLayout::for_participants(
        shape,
        total_tokens,
        invocation.participants().len(),
        projection,
        attention_policy,
    )?;
    let binding_layout = BindingLayout::new(shape, invocation.participants().len())?;
    let cuda = shape.cuda_shape()?;
    let token_ranges = invocation.participant_token_ranges();
    if token_ranges.len() != invocation.participants().len() {
        return Err("CUDA causal attention participant ranges are incomplete".to_owned());
    }
    let input_packed =
        super::super::token_binding_is_packed(invocation, ResolvedValueRole::Input, 0)?;
    let output_packed =
        super::super::token_binding_is_packed(invocation, ResolvedValueRole::Output, 0)?;

    let mut compute_regions = Vec::new();
    let shared = SharedRegions {
        input_norm: push_shared_weight(&mut compute_regions, invocation, 1)?,
        query_weight: push_shared_projection_weight(
            &mut compute_regions,
            invocation,
            2,
            &[shape.query_projection_features, shape.hidden_size],
        )?,
        key_weight: push_shared_projection_weight(
            &mut compute_regions,
            invocation,
            3,
            &[shape.kv_features, shape.hidden_size],
        )?,
        value_weight: push_shared_projection_weight(
            &mut compute_regions,
            invocation,
            4,
            &[shape.kv_features, shape.hidden_size],
        )?,
        output_weight: push_shared_projection_weight(
            &mut compute_regions,
            invocation,
            5,
            &[shape.hidden_size, shape.query_features],
        )?,
        query_norm: push_shared_weight(&mut compute_regions, invocation, 6)?,
        key_norm: push_shared_weight(&mut compute_regions, invocation, 7)?,
        post_attention_norm: semantics
            .has_post_attention_norm()
            .then(|| push_shared_weight(&mut compute_regions, invocation, 9))
            .transpose()?,
        scratch: {
            let index = compute_regions.len();
            compute_regions.push(super::super::shared_scratch_region(
                invocation,
                layout.required_bytes,
            )?);
            index
        },
        binding: {
            let index = compute_regions.len();
            compute_regions.push(super::super::shared_binding_region(
                invocation,
                binding_layout.required_bytes,
            )?);
            index
        },
    };

    let packed = if input_packed && output_packed && invocation.participants().len() > 1 {
        let input_region = compute_regions.len();
        compute_regions.push(super::super::shared_token_region(
            invocation,
            ResolvedValueRole::Input,
            0,
            precision.hidden(),
            total_tokens,
        )?);
        let output_region = compute_regions.len();
        compute_regions.push(super::super::shared_token_region(
            invocation,
            ResolvedValueRole::Output,
            0,
            precision.hidden(),
            total_tokens,
        )?);
        Some(PackedCausalAttentionLaunch {
            input_region,
            output_region,
            tokens: total_tokens,
            tokens_i32: checked_i32(total_tokens, "packed causal attention token count")?,
        })
    } else {
        None
    };

    let mut binding_regions = vec![compute_regions[shared.binding].clone()];
    let mut compute_fence_dependencies = Vec::new();
    let mut host_storage = Vec::with_capacity(invocation.participants().len());
    let mut launches = Vec::with_capacity(invocation.participants().len());
    let mut bindings = Vec::with_capacity(invocation.participants().len());
    for (participant_index, (participant, token_range)) in invocation
        .participants()
        .iter()
        .zip(token_ranges)
        .enumerate()
    {
        let tokens = token_range.immediate_tokens();
        let source = token_range.source_token_range();
        let packed_range = token_range.immediate_token_range();
        if source.end > token_range.full_input_tokens()
            || token_range.full_input_tokens() > shape.maximum_context_tokens
        {
            return Err("causal attention token range exceeds its admitted context".to_owned());
        }
        let input_region = if let Some(packed) = packed {
            packed.input_region
        } else {
            let input_region = compute_regions.len();
            compute_regions.push(contiguous_token_region(
                participant,
                binding(participant.bindings(), ResolvedValueRole::Input, 0)?,
                precision.hidden(),
                if input_packed {
                    packed_range.start
                } else {
                    source.start
                },
                tokens,
            )?);
            input_region
        };
        let output_region = if let Some(packed) = packed {
            packed.output_region
        } else {
            let output_region = compute_regions.len();
            compute_regions.push(contiguous_token_region(
                participant,
                binding(participant.bindings(), ResolvedValueRole::Output, 0)?,
                precision.hidden(),
                if output_packed {
                    packed_range.start
                } else {
                    source.start
                },
                tokens,
            )?);
            output_region
        };

        let first_page_region = binding_regions.len();
        let state = binding(participant.bindings(), ResolvedValueRole::Input, 8)?;
        let pages = paged_state_regions(
            participant,
            state,
            shape.physical_state_bytes_for_source_frontier(
                source.end,
                token_range.full_input_tokens(),
            )?,
            shape.kv_element_type(),
        )?;
        let scale_pages = scale_state_regions(participant, shape, source.end)?;
        let page_count = u64::try_from(pages.len())
            .map_err(|_| "causal attention page count exceeds u64".to_owned())?;
        if page_count > shape.maximum_pages()? {
            return Err("causal attention page table exceeds its admitted maximum".to_owned());
        }
        let table_entries = shape.table_entries(source.end)?;
        if table_entries > shape.table_entries(shape.maximum_context_tokens)? {
            return Err("causal attention address table exceeds its admitted maximum".to_owned());
        }
        let tokens_i32 = checked_i32(tokens, "causal attention participant token count")?;
        let position_start = checked_i32(source.start, "causal attention source position")?;
        let sequence_tokens_i32 = checked_i32(source.end, "causal attention sequence token count")?;
        let table_entries_i32 =
            checked_i32(table_entries, "causal attention address-table entry count")?;
        let path = CausalAttentionKernelPath::select(attention_policy, shape, tokens, source.end)?;
        let replay_topology = CausalAttentionReplayTopology::new(shape, path, source.end)?;
        let host_binding = host_storage.len();
        host_storage.push(binding_payload(
            shape.kv_layout()?,
            table_entries_i32,
            position_start,
            tokens_i32,
            sequence_tokens_i32,
            checked_i32(packed_range.start, "causal attention packed token start")?,
            &pages,
            &scale_pages,
        )?);
        let payload_page_count = pages.len();
        let page_count = payload_page_count + scale_pages.len();
        compute_fence_dependencies.extend(pages.iter().cloned());
        compute_fence_dependencies.extend(scale_pages.iter().cloned());
        binding_regions.extend(pages);
        binding_regions.extend(scale_pages);
        let binding_offset = binding_layout.binding_offset(participant_index)?;
        bindings.push(CausalAttentionBinding {
            first_page_region,
            page_count,
            payload_page_count,
            payload_element_type: shape.kv_element_type(),
            host_binding,
            binding_offset,
        });
        launches.push(CausalAttentionLaunch {
            input_region,
            output_region,
            binding_offset,
            packed_token_start: packed_range.start,
            packed_query_raw: layout.token_offset(
                layout.query_raw,
                packed_range.start,
                shape.query_projection_features,
            )?,
            packed_key_raw: layout.token_offset(
                layout.key_raw,
                packed_range.start,
                shape.kv_features,
            )?,
            packed_value_raw: layout.token_offset(
                layout.value_raw,
                packed_range.start,
                shape.kv_features,
            )?,
            packed_query: layout.token_offset(
                layout.query,
                packed_range.start,
                shape.query_features,
            )?,
            packed_context: layout.token_offset(
                layout.context,
                packed_range.start,
                shape.query_features,
            )?,
            tokens,
            tokens_i32,
            sequence_tokens: source.end,
            sequence_tokens_i32,
            table_entries_i32,
            replay_topology,
            path,
        });
    }
    if packed.is_some() {
        validate_packed_token_ranges(
            launches
                .iter()
                .map(|launch| (launch.packed_token_start, launch.tokens)),
            total_tokens,
        )?;
    }

    let participant_count = u32::try_from(invocation.participants().len())
        .map_err(|_| "CUDA causal attention participant count exceeds u32".to_owned())?;
    Ok(Prepared {
        shape,
        projection,
        total_tokens,
        layout,
        binding_layout,
        cuda,
        shared,
        packed,
        compute_regions,
        binding_regions,
        compute_fence_dependencies,
        host_storage,
        launches,
        bindings,
        participant_count,
    })
}
