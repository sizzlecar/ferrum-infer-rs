//! Physical payload demands of the existing fresh-text refinement consumer.
//! Work buffers and request-persistent cache destinations have separate grants.
use super::*;
use anyhow::{ensure, Context, Result};

/// One new request-owned Input allocation, admitted before device submission.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PH3ImageRefinementCacheHostRequirement {
    pub request: RequestIdentity,
    pub tensor: ResolvedTensorSpec,
    pub input_bytes: usize,
}
impl PH3ImageRefinementCacheHostRequirement {
    pub fn scope_claim(&self) -> (HostMemoryClass, usize) {
        (HostMemoryClass::InputPreprocess, self.input_bytes)
    }
}

/// Pure metadata over the actual prepared RefineOnly geometry. No account or
/// grant is retained, and no payload is allocated by this query.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PH3ImageRefinementHostRequirements {
    /// Required shape of the already-owned F32 connector input. Its caller
    /// retains its real backing charge (possibly a larger sliced owner).
    /// Neither this payload nor an optional caller-side F32 concatenation is
    /// included in the new work grant below.
    pub borrowed_conditioning: ResolvedTensorSpec,
    pub work_input: ResolvedTensorSpec,
    pub work_output: ResolvedTensorSpec,
    pub work_input_bytes: usize,
    pub work_output_bytes: usize,
    pub work_scope_bytes: usize,
    pub persistent_cache_by_request: Vec<PH3ImageRefinementCacheHostRequirement>,
    pub persistent_cache_bytes: usize,
    /// New caches are preallocated, so they coexist with BF16 work Input and
    /// Output. Already-owned connector/other caller payloads are excluded.
    pub new_live_peak_bytes: usize,
}
impl PH3ImageRefinementHostRequirements {
    /// Request cache credit is deliberately absent: each destination consumes
    /// its own persistent request scope and must not be granted twice.
    pub fn work_scope_claims(&self) -> [(HostMemoryClass, usize); 2] {
        [
            (HostMemoryClass::InputPreprocess, self.work_input_bytes),
            (HostMemoryClass::Output, self.work_output_bytes),
        ]
    }
}

pub(in crate::p_h3_image) fn query(
    config: &PH3ImageDiTConfig,
    identities: &[RequestIdentity],
    family: &PreparedModelFamily,
    options: &ProgramPlanCompileOptions,
    host: &HostMemoryAccount,
) -> Result<PH3ImageRefinementHostRequirements> {
    ensure!(
        config.phase == PH3ImageDiTPhase::RefineOnly,
        "refinement Host requirements need the fresh-text refinement phase"
    );
    ensure!(
        identities.len() == config.requests.len()
            && !identities.is_empty()
            && identities.iter().collect::<BTreeSet<_>>().len() == identities.len(),
        "refinement cache identities differ from distinct prepared requests"
    );
    let input_id = ProgramValueId::new(PH3ImageDiTFamily::RAW_TEXT_ID)?;
    let output_id = ProgramValueId::new(PH3ImageDiTFamily::REFINED_TEXT_ID)?;
    let program = family.program();
    ensure!(
        program.inputs() == [input_id.clone()]
            && program.outputs() == [output_id.clone()]
            && family.numerical_profile().id.as_str() == PH3ImageDiTFamily::cuda_profile_id(config),
        "refinement Host requirements differ from the prepared input/output program"
    );
    let inputs = config.input_tensor_specs()?;
    let input = options
        .tensor_specs()
        .get(&input_id)
        .context("missing refinement input shape")?;
    let output = options
        .tensor_specs()
        .get(&output_id)
        .context("missing refinement output shape")?;
    ensure!(
        inputs.get(&input_id) == Some(input)
            && input.element_type == ElementType::Bf16
            && input.layout == ResolvedTensorLayout::Contiguous
            && output.dimensions == [config.text_rows()?, config.hidden]
            && output.element_type == ElementType::Bf16
            && output.layout == ResolvedTensorLayout::Contiguous,
        "refinement Host requirements differ from actual BF16 work/cache shapes"
    );
    let resolve = |spec: &ProgramTensorSpec| -> Result<ResolvedTensorSpec> {
        Ok(ResolvedTensorSpec::new(
            spec.dimensions.clone(),
            spec.element_type,
            spec.layout.clone(),
        )?)
    };
    let work_input = resolve(input)?;
    let work_output = resolve(output)?;
    let work_input_bytes = physical(host, &work_input)?;
    let work_output_bytes = physical(host, &work_output)?;
    let borrowed_conditioning = ResolvedTensorSpec::new(
        input.dimensions.clone(),
        ElementType::F32,
        input.layout.clone(),
    )?;
    let mut persistent_cache_by_request = Vec::new();
    persistent_cache_by_request
        .try_reserve_exact(identities.len())
        .context("refinement Host cache metadata allocation failed")?;
    let mut persistent_cache_bytes = 0usize;
    for (request, shape) in identities.iter().zip(&config.requests) {
        // Same per-request allocation shape as PreparedRefinedTextCaches.
        // Each cache gets independent backing padding, not a batched slice.
        let tensor = ResolvedTensorSpec::new(
            vec![shape.text_rows, output.dimensions[1]],
            output.element_type,
            output.layout.clone(),
        )?;
        let input_bytes = physical(host, &tensor)?;
        persistent_cache_bytes = persistent_cache_bytes
            .checked_add(input_bytes)
            .context("refinement persistent cache backing overflows")?;
        persistent_cache_by_request.push(PH3ImageRefinementCacheHostRequirement {
            request: request.clone(),
            tensor,
            input_bytes,
        });
    }
    let work_scope_bytes = work_input_bytes
        .checked_add(work_output_bytes)
        .context("refinement work scope backing overflows")?;
    let new_live_peak_bytes = work_scope_bytes
        .checked_add(persistent_cache_bytes)
        .context("refinement simultaneous backing overflows")?;
    Ok(PH3ImageRefinementHostRequirements {
        borrowed_conditioning,
        work_input,
        work_output,
        work_input_bytes,
        work_output_bytes,
        work_scope_bytes,
        persistent_cache_by_request,
        persistent_cache_bytes,
        new_live_peak_bytes,
    })
}
fn physical(host: &HostMemoryAccount, tensor: &ResolvedTensorSpec) -> Result<usize> {
    Ok(host.backing_bytes(usize::try_from(tensor.minimum_storage_bytes()?)?)?)
}

#[cfg(test)]
mod tests;
