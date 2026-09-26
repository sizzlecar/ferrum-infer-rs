//! The same checked physical preparation is consumed by eager encoding and
//! by an opt-in replay query for the current invocation, never the capture wave.
use super::*;

pub(super) struct PreparedAttention {
    pub(super) shape: AttentionShape,
    pub(super) projection: AttentionProjection,
    pub(super) total_tokens: u64,
    pub(super) layout: ScratchLayout,
    pub(super) binding_layout: StateBindingLayout,
    pub(super) cuda_shape: CudaAttentionShape,
    pub(super) use_packed: bool,
    pub(super) compute_regions: Vec<CudaBufferRegion>,
    pub(super) shared: SharedRegions,
    pub(super) binding_regions: Vec<CudaBufferRegion>,
    pub(super) binding_host_storage: Vec<Box<[u8]>>,
    pub(super) state_bindings: Vec<AttentionStateBinding>,
    pub(super) compute_fence_dependencies: Vec<CudaBufferRegion>,
    pub(super) host_storage: Vec<Box<[u8]>>,
    pub(super) launches: Vec<AttentionLaunch>,
    pub(super) participant_token_counts: Vec<u64>,
}

pub(super) fn prepare(
    invocation: &BatchedOperationInvocation<'_, CudaDeviceBuffer>,
    precision: AttentionPrecision,
    execution_capabilities: GatedDeltaExecutionCapabilities,
    #[cfg(feature = "vllm-marlin")] projection_runtime: MarlinProjectionRuntime,
) -> Result<PreparedAttention, String> {
    if invocation.participants().is_empty()
        || invocation.operation().id.as_str() != precision.operation()
    {
        return Err("CUDA recurrent attention received another or empty operation".to_owned());
    }
    super::super::gguf_f16_projection::validate_invocation(invocation)?;
    let first = &invocation.participants()[0];
    let shape = AttentionShape::from_attributes(first.attributes())?;
    let projection = AttentionProjection::from_values(
        first.bindings(),
        precision,
        #[cfg(feature = "vllm-marlin")]
        projection_runtime,
    )?;
    validate_signature(first, shape, precision)?;
    for participant in &invocation.participants()[1..] {
        if AttentionShape::from_attributes(participant.attributes())? != shape {
            return Err("CUDA recurrent attention participant attributes disagree".to_owned());
        }
        validate_signature(participant, shape, precision)?;
    }

    let total_tokens = invocation.work_shape().immediate_tokens();
    let participant_count_usize = invocation.participants().len();
    let participant_count_u64 = u64::try_from(participant_count_usize)
        .map_err(|_| "CUDA recurrent attention participant count exceeds u64".to_owned())?;
    shape.validate_launch_extents(total_tokens)?;
    let layout = ScratchLayout::new(shape, total_tokens, participant_count_usize, projection)?;
    let binding_layout = StateBindingLayout::new(participant_count_usize)?;
    let cuda_shape = shape.cuda_shape()?;
    let token_ranges = invocation.participant_token_ranges();
    if token_ranges.len() != invocation.participants().len() {
        return Err("CUDA recurrent attention participant ranges are incomplete".to_owned());
    }
    let input_packed =
        super::super::token_binding_is_packed(&invocation, ResolvedValueRole::Input, 0)?;
    let output_packed =
        super::super::token_binding_is_packed(&invocation, ResolvedValueRole::Output, 0)?;
    let use_packed = participant_count_usize > 1 && input_packed && output_packed;

    let mut compute_regions = Vec::new();
    let shared = SharedRegions {
        input_norm: push_shared_weight(&mut compute_regions, &invocation, 1, ElementType::F16)?,
        qkvzba: push_shared_projection_weight(
            &mut compute_regions,
            &invocation,
            2,
            &[shape.qkvzba_features, shape.hidden_size],
        )?,
        conv: push_shared_weight(&mut compute_regions, &invocation, 3, ElementType::F16)?,
        a_log: push_shared_weight(&mut compute_regions, &invocation, 4, ElementType::F32)?,
        dt_bias: push_shared_weight(&mut compute_regions, &invocation, 5, ElementType::F32)?,
        norm: push_shared_weight(&mut compute_regions, &invocation, 6, ElementType::F32)?,
        output: push_shared_projection_weight(
            &mut compute_regions,
            &invocation,
            7,
            &[shape.hidden_size, shape.value_features],
        )?,
        scratch: {
            let index = compute_regions.len();
            compute_regions.push(shared_scratch_region(&invocation, layout.required_bytes)?);
            index
        },
        binding: {
            let index = compute_regions.len();
            compute_regions.push(super::super::shared_binding_region(
                &invocation,
                binding_layout.required_bytes,
            )?);
            index
        },
    };
    let packed_regions = if use_packed {
        let input_region = compute_regions.len();
        compute_regions.push(super::super::shared_token_region(
            &invocation,
            ResolvedValueRole::Input,
            0,
            precision.hidden(),
            total_tokens,
        )?);
        let output_region = compute_regions.len();
        compute_regions.push(super::super::shared_token_region(
            &invocation,
            ResolvedValueRole::Output,
            0,
            precision.hidden(),
            total_tokens,
        )?);
        Some((input_region, output_region))
    } else {
        None
    };
    let mut binding_regions = vec![compute_regions[shared.binding].clone()];
    let mut binding_host_storage = Vec::with_capacity(invocation.participants().len());
    let mut state_bindings = Vec::with_capacity(invocation.participants().len());
    let mut compute_fence_dependencies =
        Vec::with_capacity(invocation.participants().len().saturating_mul(2));
    let mut host_storage = Vec::with_capacity(if use_packed {
        2
    } else {
        participant_count_usize
    });
    let mut launches = Vec::with_capacity(if use_packed {
        1
    } else {
        participant_count_usize
    });
    let mut participant_token_counts = Vec::with_capacity(participant_count_usize);
    let mut packed_token_cursor = 0_u64;
    let mut packed_execution_form = None;
    for (participant_index, (participant, token_range)) in invocation
        .participants()
        .iter()
        .zip(token_ranges)
        .enumerate()
    {
        let tokens = token_range.immediate_tokens();
        shape.validate_launch_extents(tokens)?;
        if use_packed {
            let packed = token_range.immediate_token_range();
            let expected_end = packed_token_cursor
                .checked_add(tokens)
                .ok_or_else(|| "CUDA packed recurrent token range overflows".to_owned())?;
            if packed.start != packed_token_cursor || packed.end != expected_end {
                return Err(
                    "CUDA packed recurrent attention token ranges are not canonical".to_owned(),
                );
            }
            packed_token_cursor = expected_end;
        }
        participant_token_counts.push(tokens);
        let conv_state = contiguous_region(
            participant,
            binding(participant.bindings(), ResolvedValueRole::Input, 8)?,
            ElementType::F16,
        )?;
        let delta_state = contiguous_region(
            participant,
            binding(participant.bindings(), ResolvedValueRole::Input, 9)?,
            ElementType::F32,
        )?;
        let first_state_region = binding_regions.len();
        binding_regions.push(conv_state.clone());
        binding_regions.push(delta_state.clone());
        compute_fence_dependencies.push(conv_state.clone());
        compute_fence_dependencies.push(delta_state.clone());
        let binding_offset = binding_layout.offset(participant_index)?;
        let host_binding = binding_host_storage.len();
        binding_host_storage.push(state_binding_payload(&conv_state, &delta_state));
        state_bindings.push(AttentionStateBinding {
            first_state_region,
            host_binding,
            binding_offset,
            conv_state_bytes: conv_state.length_bytes(),
            delta_state_bytes: delta_state.length_bytes(),
        });
        let tokens_i32 = checked_i32(tokens, "attention participant token count")?;
        let execution_form = execution_capabilities
            .select(tokens, GatedDeltaExecutionPreference::RecurrentScan)
            .map_err(|error| error.to_string())?;
        if let GatedDeltaExecutionForm::ChunkedScan(plan) = execution_form {
            return Err(format!(
                "CUDA gated-delta provider selected an uninstalled {} form for {} tokens",
                execution_form.as_str(),
                plan.token_count()
            ));
        }
        if use_packed {
            if packed_execution_form
                .replace(execution_form)
                .is_some_and(|previous| previous != execution_form)
            {
                return Err(
                    "CUDA packed recurrent attention participants selected different execution forms"
                        .to_owned(),
                );
            }
        } else {
            let source = token_range.source_token_range();
            let packed = token_range.immediate_token_range();
            let input_region = compute_regions.len();
            compute_regions.push(contiguous_token_region(
                participant,
                binding(participant.bindings(), ResolvedValueRole::Input, 0)?,
                precision.hidden(),
                if input_packed {
                    packed.start
                } else {
                    source.start
                },
                tokens,
            )?);
            let output_region = compute_regions.len();
            compute_regions.push(contiguous_token_region(
                participant,
                binding(participant.bindings(), ResolvedValueRole::Output, 0)?,
                precision.hidden(),
                if output_packed {
                    packed.start
                } else {
                    source.start
                },
                tokens,
            )?);
            let host_control = host_storage.len();
            host_storage.push(sequence_control(&[tokens])?);
            launches.push(AttentionLaunch {
                input_region,
                output_region,
                state_binding_offset: binding_offset,
                host_control,
                host_token_seq_indices: None,
                execution_form,
                batch_i32: 1,
                tokens,
                tokens_i32,
            });
        }
    }
    if let Some((input_region, output_region)) = packed_regions {
        if packed_token_cursor != total_tokens {
            return Err(
                "CUDA packed recurrent attention token ranges do not cover the wave".to_owned(),
            );
        }
        let host_control = host_storage.len();
        host_storage.push(sequence_control(&participant_token_counts)?);
        let host_token_seq_indices = host_storage.len();
        host_storage.push(token_sequence_indices(&participant_token_counts)?);
        launches.push(AttentionLaunch {
            input_region,
            output_region,
            state_binding_offset: 0,
            host_control,
            host_token_seq_indices: Some(host_token_seq_indices),
            execution_form: packed_execution_form.ok_or_else(|| {
                "CUDA packed recurrent attention has no execution form".to_owned()
            })?,
            batch_i32: checked_i32(participant_count_u64, "attention packed participant count")?,
            tokens: total_tokens,
            tokens_i32: checked_i32(total_tokens, "attention packed token count")?,
        });
    }

    Ok(PreparedAttention {
        shape,
        projection,
        total_tokens,
        layout,
        binding_layout,
        cuda_shape,
        use_packed,
        compute_regions,
        shared,
        binding_regions,
        binding_host_storage,
        state_bindings,
        compute_fence_dependencies,
        host_storage,
        launches,
        participant_token_counts,
    })
}
