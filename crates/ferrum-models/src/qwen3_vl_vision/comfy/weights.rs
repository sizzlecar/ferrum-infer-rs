//! Immutable validated original-format weights; each invocation binds the
//! caller's current Host authority for its declared BF16-to-F32 promotions.
use super::*;
use ferrum_quantization::comfy_convrot::ComfyCanonicalWeightSource;
use std::sync::Arc;

pub struct Qwen3VLVisionComfyWeightAdapter {
    source: Arc<ComfyCanonicalWeightSource>,
    identity: ExecutionWeightPlan,
    index: BTreeMap<WeightId, usize>,
}

impl Qwen3VLVisionComfyWeightAdapter {
    pub fn new(
        config: &Qwen3VLVisionComfyConfig,
        source: Arc<ComfyCanonicalWeightSource>,
        limits: Qwen3VLVisionComfyLimits,
    ) -> Result<Self> {
        Qwen3VLVisionComfyCheckpoint::new(config, source.archive(), limits)?;
        let family = TypedFamilyRegistration::new(Qwen3VLVisionComfyFamily::new(limits)?)
            .prepare_with_profile(
                &serde_json::to_value(config)?,
                &NumericalProfileId::new(Qwen3VLVisionComfyFamily::REFERENCE_PROFILE_ID)
                    .map_err(invalid)?,
            )?;
        let identity = ExecutionWeightPlan::identity(&family)?;
        let index = identity
            .schema()
            .components
            .iter()
            .enumerate()
            .map(|(i, c)| (c.id.clone(), i))
            .collect();
        Ok(Self {
            source,
            identity,
            index,
        })
    }
    pub fn source(&self) -> &Arc<ComfyCanonicalWeightSource> {
        &self.source
    }
    pub fn stored_bytes(&self) -> Result<u64> {
        self.identity
            .schema()
            .components
            .iter()
            .try_fold(0u64, |sum, c| {
                sum.checked_add(c.physical_bytes()?)
                    .context("visual weight extent overflows")
            })
    }
    pub fn weight_source<'a>(
        &'a self,
        host: &'a HostMemoryAccount,
    ) -> Qwen3VLVisionComfyWeightSource<'a> {
        Qwen3VLVisionComfyWeightSource {
            schema: self.identity.schema(),
            index: &self.index,
            dense: ManagedDenseArchiveSource::new(self.source.archive(), host),
        }
    }
    pub fn component_backing_bytes(
        &self,
        component: &WeightComponentSpec,
        host: &HostMemoryAccount,
    ) -> Result<usize> {
        let i = self
            .index
            .get(&component.id)
            .context("foreign visual weight component")?;
        ensure!(
            &self.identity.schema().components[*i] == component,
            "visual component differs from validated original contract"
        );
        Ok(host.backing_bytes(usize::try_from(component.physical_bytes()?)?)?)
    }
    /// The same original component groups coexist until static upload has
    /// copied them into its separate TransferStaging grant. No account or
    /// payload is retained by this metadata-only query.
    pub fn maximum_group_backing_bytes(
        &self,
        family: &PreparedModelFamily,
        plan: &ExecutionPlan,
        host: &HostMemoryAccount,
    ) -> Result<usize> {
        let payload = plan.payload();
        ensure!(
            family.family_id().as_str() == Qwen3VLVisionComfyFamily::FAMILY_ID
                && payload.family_id() == family.family_id()
                && payload.prepared_family_fingerprint() == family.fingerprint()?
                && payload.program_fingerprint() == family.program().fingerprint()?
                && family.stored_weight_schema()? == self.identity.schema()
                && payload.execution_weights().stored()? == &self.identity,
            "visual source requires its exact checked family, Stored schema and identity mapping"
        );
        let placed = payload
            .nodes()
            .iter()
            .flat_map(|n| n.values())
            .filter(|v| v.usage() == BufferUsage::Weights)
            .flat_map(|v| v.storage().components())
            .filter_map(|c| c.component_id().cloned())
            .collect::<BTreeSet<_>>();
        ensure!(
            placed.iter().all(|id| self.index.contains_key(id)),
            "visual static placement contains a foreign component"
        );
        let mut groups = BTreeMap::<Vec<WeightId>, usize>::new();
        for component in &self.identity.schema().components {
            if !placed.contains(&component.id) {
                continue;
            }
            let sources = self
                .identity
                .component_sources()
                .get(&component.id)
                .context("visual component has no original source mapping")?;
            let group = groups.entry(sources.clone()).or_default();
            *group = group
                .checked_add(self.component_backing_bytes(component, host)?)
                .context("visual static group Host demand overflows")?;
        }
        Ok(groups.into_values().max().unwrap_or(0))
    }
}

#[cfg(test)]
mod tests;
