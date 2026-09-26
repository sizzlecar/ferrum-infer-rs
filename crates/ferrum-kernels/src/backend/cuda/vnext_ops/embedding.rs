//! Dense lookup's retained physical preparation shared by actual and replay.
use super::*;

pub(super) struct DensePrepared {
    pub(super) regions: Vec<CudaBufferRegion>,
    pub(super) launches: Vec<EmbeddingLaunch>,
    pub(super) participants: u32,
    pub(super) tokens: u64,
}

pub(super) fn prepare_dense(
    invocation: &BatchedOperationInvocation<'_, CudaDeviceBuffer>,
) -> Result<DensePrepared, String> {
    if invocation.operation().id.as_str() != TOKEN_EMBEDDING_OPERATION_ID
        || invocation.participants().is_empty()
    {
        return Err("CUDA token embedding received another or empty operation".to_owned());
    }

    let token_ranges = invocation.participant_token_ranges();
    if token_ranges.len() != invocation.participants().len() {
        return Err("CUDA token embedding participant ranges are incomplete".to_owned());
    }
    let input_packed =
        transformer::token_binding_is_packed(invocation, ResolvedValueRole::Input, 0)?;
    let mut regions = Vec::with_capacity(invocation.participants().len() * 3);
    let mut launches = Vec::with_capacity(invocation.participants().len());
    for (participant, token_range) in invocation.participants().iter().zip(token_ranges) {
        let token_ids = binding(participant.bindings(), ResolvedValueRole::Input, 0)?;
        let table = binding(participant.bindings(), ResolvedValueRole::Input, 1)?;
        let output = binding(participant.bindings(), ResolvedValueRole::Output, 0)?;
        let hidden_size = unsigned_attribute(participant.attributes(), "hidden_size")?;
        let vocabulary_size = unsigned_attribute(participant.attributes(), "vocab_size")?;
        validate_signature(
            token_ids,
            table,
            output,
            vocabulary_size,
            hidden_size,
            ElementType::F16,
        )?;
        let source_range = token_range.source_token_range();
        let packed_range = token_range.immediate_token_range();
        let token_count = token_range.immediate_tokens();
        let grid_x = hidden_size
            .div_ceil(THREADS_PER_BLOCK as u64)
            .try_into()
            .map_err(|_| "embedding launch grid exceeds u32".to_owned())?;

        let first_region = regions.len();
        regions.push(contiguous_region(participant, table, ElementType::F16)?);
        regions.push(contiguous_token_region(
            participant,
            token_ids,
            ElementType::U32,
            if input_packed {
                packed_range.start
            } else {
                source_range.start
            },
            token_count,
        )?);
        regions.push(contiguous_token_region(
            participant,
            output,
            ElementType::F16,
            packed_range.start,
            token_count,
        )?);
        launches.push(EmbeddingLaunch {
            first_region,
            token_count,
            vocabulary_size: vocabulary_size
                .try_into()
                .map_err(|_| "embedding vocabulary size exceeds u32".to_owned())?,
            hidden_size: hidden_size
                .try_into()
                .map_err(|_| "embedding hidden size exceeds i32".to_owned())?,
            grid_x,
        });
    }

    Ok(DensePrepared {
        regions,
        launches,
        participants: u32::try_from(invocation.participants().len())
            .map_err(|_| "embedding participant count exceeds u32")?,
        tokens: invocation.work_shape().immediate_tokens(),
    })
}

#[derive(Clone, Copy)]
pub(super) struct DensePlan {
    pub(super) config: LaunchConfig,
    pub(super) parameters: [u32; 3],
}

pub(super) fn dense_plan(
    vocabulary: u64,
    hidden: u64,
    count: u64,
) -> Result<DensePlan, CudaDeviceRuntimeError> {
    let invalid = || {
        CudaDeviceRuntimeError::contract("dense embedding launch dimensions exceed installed ABI")
    };
    if vocabulary == 0 || hidden == 0 || count == 0 || count > MAXIMUM_TOKENS_PER_LAUNCH {
        return Err(invalid());
    }
    let vocabulary = u32::try_from(vocabulary).map_err(|_| invalid())?;
    let hidden = i32::try_from(hidden).map_err(|_| invalid())? as u32;
    let count = u32::try_from(count).map_err(|_| invalid())?;
    Ok(DensePlan {
        config: LaunchConfig {
            grid_dim: (hidden.div_ceil(THREADS_PER_BLOCK), count, 1),
            block_dim: (THREADS_PER_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        },
        parameters: [count, hidden, vocabulary],
    })
}

pub(super) fn selected_dense(
    leaves: impl IntoIterator<Item = (u64, u64, u64)>,
    tokens: u64,
    capture: ferrum_types::SloStructuredCostCapture,
) -> Option<ferrum_interfaces::execution_cost::SelectedCommandCostEvidenceV1> {
    use ferrum_interfaces::execution_cost::{
        KernelNumericWorkV1, KernelReplayGeometryV1, SelectedAlgorithmClassV1,
    };
    use sha2::{Digest, Sha256};
    use std::sync::OnceLock;
    let mut builder = super::super::vnext_runtime::selected_cost::builder(capture, tokens)?;
    static NUMERICAL: OnceLock<[u8; 32]> = OnceLock::new();
    let algorithm = SelectedAlgorithmClassV1::new(
        EMBEDDING_FUNCTION_NAME,
        1,
        *NUMERICAL.get_or_init(|| Sha256::digest(crate::ptx::EMBEDDING_LOOKUP.as_bytes()).into()),
        Sha256::digest(b"cuda.embedding.dense_f16.row_major.guard_invalid_ids.v1").into(),
    )
    .ok()?;
    let mut seen = 0_u64;
    for (vocabulary, hidden, mut count) in leaves {
        if count == 0 {
            return None;
        }
        seen = seen.checked_add(count)?;
        if seen > tokens {
            return None;
        }
        while count > 0 {
            let chunk = count.min(MAXIMUM_TOKENS_PER_LAUNCH);
            let selected = dense_plan(vocabulary, hidden, chunk).ok()?;
            let fixed = selected.parameters.map(u64::from);
            let c = selected.config;
            builder
                .kernel_with_replay_geometry(
                    algorithm,
                    KernelNumericWorkV1 {
                        logical_units: chunk.checked_mul(hidden)?,
                        padded_units: u64::from(c.grid_dim.0)
                            .checked_mul(u64::from(c.block_dim.0))?
                            .checked_mul(chunk)?,
                        inner_units_per_logical_unit: 1,
                        grid: [c.grid_dim.0, c.grid_dim.1, c.grid_dim.2],
                        scratch_bytes: 0,
                        staged_weight_bytes: 0,
                    },
                    KernelReplayGeometryV1 {
                        block: [c.block_dim.0, c.block_dim.1, c.block_dim.2],
                        dynamic_shared_bytes: 0,
                        fixed_parameters: &fixed,
                    },
                )
                .ok()?;
            count -= chunk;
        }
    }
    if seen != tokens {
        return None;
    }
    builder.finish().ok()
}

#[cfg(test)]
mod tests;
