//! Original native lookup preparation, retained once by actual encoding and
//! reused for a current replay evidence query. This does not upload or execute.
use super::*;
pub(in crate::backend::cuda::vnext_ops) struct Prepared {
    pub(in crate::backend::cuda::vnext_ops) part: weights::MatrixPart,
    pub(in crate::backend::cuda::vnext_ops) regions: Vec<CudaBufferRegion>,
    pub(in crate::backend::cuda::vnext_ops) scratch: Option<usize>,
    pub(in crate::backend::cuda::vnext_ops) launches: Vec<(usize, usize, u64)>,
    pub(in crate::backend::cuda::vnext_ops) key: CudaCommandReplayKeyBuilder,
    pub(in crate::backend::cuda::vnext_ops) chunk_limit: u64,
    pub(in crate::backend::cuda::vnext_ops) participants: u32,
    pub(in crate::backend::cuda::vnext_ops) tokens: u64,
    pub(in crate::backend::cuda::vnext_ops) dispatches: u64,
}
impl Prepared {
    pub(in crate::backend::cuda::vnext_ops) fn selected(
        &self,
        precision: TokenPrecision,
        capture: ferrum_types::SloStructuredCostCapture,
    ) -> Option<ferrum_interfaces::execution_cost::SelectedCommandCostEvidenceV1> {
        super::super::native_blocks::embedding::selected(
            &self.part,
            self.launches.iter().map(|&(_, _, count)| count),
            self.tokens,
            precision.element(),
            self.scratch
                .map_or(0, |index| self.regions[index].length_bytes()),
            capture,
        )
    }
}
pub(in crate::backend::cuda::vnext_ops) fn prepare(
    fingerprint: &str,
    precision: TokenPrecision,
    invocation: &BatchedOperationInvocation<'_, CudaDeviceBuffer>,
) -> Result<Prepared, String> {
    transformer::ensure_invocation(invocation, precision.embedding_operation())?;
    let first = &invocation.participants()[0];
    let hidden = unsigned_attribute(first.attributes(), "hidden_size")?;
    let vocabulary = unsigned_attribute(first.attributes(), "vocab_size")?;
    let weight = retain_shared_weight(invocation, &[vocabulary, hidden])?;
    // Vocabulary partitions require ID routing before lookup; this kernel
    // consumes one complete table and must not silently address the first part.
    let [part] = weight.parts.as_slice() else {
        return Err("CUDA native embedding requires one complete vocabulary table".into());
    };
    let chunk_limit = embedding_chunk_limit(part)?;
    let input_packed =
        transformer::token_binding_is_packed(invocation, ResolvedValueRole::Input, 0)?;
    let output_packed =
        transformer::token_binding_is_packed(invocation, ResolvedValueRole::Output, 0)?;
    let mut key = matrix_key(fingerprint, "vnext_native_embedding", &weight);
    let mut regions = weight.regions;
    let scratch = super::native_blocks::hadamard::retain_workspace(invocation, &mut regions)?;
    let mut launches = Vec::new();
    for (participant, range) in invocation
        .participants()
        .iter()
        .zip(invocation.participant_token_ranges())
    {
        let input = binding(participant.bindings(), ResolvedValueRole::Input, 0)?;
        let table = binding(participant.bindings(), ResolvedValueRole::Input, 1)?;
        let output = binding(participant.bindings(), ResolvedValueRole::Output, 0)?;
        if unsigned_attribute(participant.attributes(), "hidden_size")? != hidden
            || unsigned_attribute(participant.attributes(), "vocab_size")? != vocabulary
        {
            return Err("CUDA native embedding participant dimensions disagree".into());
        }
        validate_signature(
            input,
            table,
            output,
            vocabulary,
            hidden,
            precision.element(),
        )?;
        let count = range.immediate_tokens();
        if count == 0 {
            return Err("CUDA native embedding cannot launch an empty token span".into());
        }
        let source = range.source_token_range();
        let packed = range.immediate_token_range();
        let input_index = regions.len();
        regions.push(contiguous_token_region(
            participant,
            input,
            ElementType::U32,
            if input_packed {
                packed.start
            } else {
                source.start
            },
            count,
        )?);
        let output_index = regions.len();
        regions.push(contiguous_token_region(
            participant,
            output,
            precision.element(),
            if output_packed {
                packed.start
            } else {
                source.start
            },
            count,
        )?);
        launches.push((input_index, output_index, count));
        key = key
            .u64(input_index as u64)
            .u64(output_index as u64)
            .u64(count);
    }
    let participants =
        u32::try_from(launches.len()).map_err(|_| "too many embedding participants")?;
    let tokens = invocation.work_shape().immediate_tokens();
    let dispatches = embedding_dispatches(part, launches.iter().map(|&(_, _, count)| count))?;
    let part = part.clone();
    Ok(Prepared {
        part,
        regions,
        scratch,
        launches,
        key,
        chunk_limit,
        participants,
        tokens,
        dispatches,
    })
}
