//! Eager candidate state output for the existing recurrent scan arithmetic.
//!
//! State pointers occupy admitted Scratch, never a reusable Binding workspace.
//! Inputs remain read-only; each output has exactly the N-1 and N tail states.

use super::*;

#[derive(Clone)]
pub(super) struct Functions {
    pub(super) convolution: CudaFunction,
    pub(super) delta: CudaFunction,
    pub(super) tiled_delta: CudaFunction,
}

impl Functions {
    pub(super) fn load(
        linear: &Arc<cudarc::driver::CudaModule>,
        delta: &Arc<cudarc::driver::CudaModule>,
    ) -> Result<Self, CudaDeviceRuntimeError> {
        Ok(Self {
            convolution: load_function(
                linear,
                "vnext_gated_delta_candidate_conv_states_f16",
                "candidate convolution states",
            )?,
            delta: load_function(
                delta,
                "vnext_gated_delta_candidate_scan_f32",
                "candidate delta states",
            )?,
            tiled_delta: load_function(
                delta,
                "vnext_gated_delta_candidate_scan_tiled16_f32",
                "candidate tiled delta states",
            )?,
        })
    }
}

pub(super) fn storage_bindings(
    candidate: bool,
) -> Vec<ferrum_runtime::ProviderStorageBindingRequirement> {
    let mut bindings = contiguous_bindings(10);
    if candidate {
        bindings.extend((1..=2).map(|ordinal| {
            ferrum_runtime::ProviderStorageBindingRequirement::new(
                ResolvedValueRole::Output,
                ordinal,
                DynamicStorageRequirement::contiguous(),
            )
        }));
    }
    bindings
}

pub(super) fn candidate_effects(
    descriptor: OperationProviderDescriptor,
    storage_profile: DynamicStorageProfile,
) -> OperationProviderDescriptor {
    descriptor.with_candidate_state_effects(
        ferrum_runtime::ProviderCandidateStateEffects::PrefixCausal {
            ports: vec![
                ferrum_runtime::ProviderCandidateStatePort::TailTwo {
                    input_ordinal: 8,
                    output_ordinal: 1,
                    storage_profile,
                },
                ferrum_runtime::ProviderCandidateStatePort::TailTwo {
                    input_ordinal: 9,
                    output_ordinal: 2,
                    storage_profile,
                },
            ],
        },
    )
}

pub(super) fn scratch_bytes_per_sequence(bytes: u64, candidate: bool) -> Result<u64, VNextError> {
    bytes
        .checked_add(if candidate {
            2 * STATE_BINDING_SLOT_BYTES
        } else {
            0
        })
        .ok_or_else(|| invalid_plan("candidate control scratch overflows".to_owned()))
}

pub(super) fn validate_outputs(
    participant: &OperationInvocation<'_, CudaDeviceBuffer>,
    shape: AttentionShape,
) -> Result<(), String> {
    for (ordinal, dimensions, dtype) in [
        (
            1,
            vec![2, shape.qkv_features, shape.conv_state_width],
            ElementType::F16,
        ),
        (
            2,
            vec![
                2,
                shape.value_heads,
                shape.value_head_dim,
                shape.key_head_dim,
            ],
            ElementType::F32,
        ),
    ] {
        let output = binding(participant.bindings(), ResolvedValueRole::Output, ordinal)?;
        if output.tensor().dimensions() != dimensions.as_slice() || !contiguous(output, dtype) {
            return Err(
                "candidate scan output must contain exactly two contiguous tail states".into(),
            );
        }
    }
    Ok(())
}

/// Check the entire selected wave, not just aliases within one participant.
pub(super) fn validate_output_aliases(
    computation: &[CudaBufferRegion],
    states: &[CudaBufferRegion],
    participants: usize,
) -> Result<(), String> {
    let split = participants
        .checked_mul(2)
        .ok_or("candidate state count overflows")?;
    if states.len()
        != split
            .checked_mul(2)
            .ok_or("candidate state count overflows")?
    {
        return Err("candidate state region set is incomplete".into());
    }
    let (originals, outputs) = states.split_at(split);
    let overlaps = |left: &CudaBufferRegion, right: &CudaBufferRegion| -> Result<bool, String> {
        let a = left.device_ptr();
        let b = right.device_ptr();
        let a_end = a
            .checked_add(left.length_bytes())
            .ok_or("candidate state address overflows")?;
        let b_end = b
            .checked_add(right.length_bytes())
            .ok_or("candidate state address overflows")?;
        Ok(a < b_end && b < a_end)
    };
    for (index, output) in outputs.iter().enumerate() {
        for other in computation.iter().chain(originals).chain(&outputs[..index]) {
            if overlaps(output, other)? {
                return Err("candidate state output overlaps a read/input/output region in the selected wave".into());
            }
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn launch_convolution(
    stream: &CudaStream,
    function: &CudaFunction,
    qkvzba: u64,
    original_bindings: u64,
    candidate_bindings: u64,
    cu_seqlens: u64,
    batch: i32,
    shape: AttentionShape,
) -> Result<(), CudaDeviceRuntimeError> {
    let width = checked_i32(shape.qkvzba_features, "candidate QKVZBA width")
        .map_err(CudaDeviceRuntimeError::contract)?;
    let channels = checked_i32(shape.qkv_features, "candidate convolution channels")
        .map_err(CudaDeviceRuntimeError::contract)?;
    let state_width = checked_i32(shape.conv_state_width, "candidate convolution width")
        .map_err(CudaDeviceRuntimeError::contract)?;
    let elements =
        shape
            .conv_state_elements()
            .map_err(CudaDeviceRuntimeError::contract)?
            .checked_mul(u64::try_from(batch).map_err(|_| {
                CudaDeviceRuntimeError::contract("candidate batch must be positive")
            })?)
            .and_then(|n| n.checked_mul(2))
            .ok_or_else(|| {
                CudaDeviceRuntimeError::contract("candidate convolution extent overflows")
            })?;
    let mut builder = stream.launch_builder(function);
    builder
        .arg(&qkvzba)
        .arg(&original_bindings)
        .arg(&candidate_bindings)
        .arg(&cu_seqlens)
        .arg(&batch)
        .arg(&width)
        .arg(&channels)
        .arg(&state_width);
    unsafe {
        builder.launch(LaunchConfig {
            grid_dim: (
                checked_grid(elements, THREADS_PER_BLOCK, "candidate convolution states")?,
                1,
                1,
            ),
            block_dim: (THREADS_PER_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        })
    }
    .map(|_| ())
    .map_err(|error| CudaDeviceRuntimeError::driver("candidate convolution states", error))
}
