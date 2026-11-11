pub use super::super::dit_forward::{
    PH3ImageRefinementCacheHostRequirement, PH3ImageRefinementHostRequirements,
};
use super::*;

impl PreparedPH3ImageDenoiser {
    /// Only fresh refinement creates these work buffers and new per-request
    /// cache owners. Static backing/staging and already-owned connector or
    /// sample payloads are separate caller/common-stage responsibilities.
    pub fn refinement_host_requirements(
        &self,
        host: &HostMemoryAccount,
    ) -> Result<PH3ImageRefinementHostRequirements> {
        ensure!(
            self.phase == PH3ImageDenoiserPhase::RefineFreshText,
            "only fresh-text refinement has refinement Host requirements"
        );
        dit_forward::query_refinement_host_requirements(
            &self.config,
            &self.identities,
            &self.family,
            &self.options,
            host,
        )
    }
}

#[cfg(test)]
mod tests;
