use super::*;
use ferrum_quantization::comfy_convrot::ComfyCanonicalWeightSource;
use std::sync::Arc;

/// Owns the same original-format source, not a dense F32 mirror. Runtime
/// installation borrows the caller's admitted account through the old source.
pub struct Qwen3VLVisionHfWeightAdapter {
    inner: Qwen3VLVisionComfyWeightAdapter,
    identity: ExecutionWeightPlan,
}
impl Qwen3VLVisionHfWeightAdapter {
    pub fn new(
        c: &Qwen3VLVisionHfConfig,
        source: Arc<ComfyCanonicalWeightSource>,
        limits: Qwen3VLVisionHfLimits,
    ) -> Result<Self> {
        c.check()?;
        let inner = Qwen3VLVisionComfyWeightAdapter::new(&c.base, source, limits)?;
        let family = TypedFamilyRegistration::new(Qwen3VLVisionHfFamily::new(limits)?)
            .prepare_with_profile(
                &serde_json::to_value(c)?,
                &NumericalProfileId::new(Qwen3VLVisionHfFamily::REFERENCE_PROFILE_ID)
                    .map_err(invalid)?,
            )?;
        Ok(Self {
            inner,
            identity: ExecutionWeightPlan::identity(&family)?,
        })
    }
    pub fn source(&self) -> &Arc<ComfyCanonicalWeightSource> {
        self.inner.source()
    }
    pub fn stored_bytes(&self) -> Result<u64> {
        self.inner.stored_bytes()
    }
    pub fn weight_source<'a>(
        &'a self,
        host: &'a HostMemoryAccount,
    ) -> Qwen3VLVisionComfyWeightSource<'a> {
        self.inner.weight_source(host)
    }
    pub fn component_backing_bytes(
        &self,
        c: &WeightComponentSpec,
        host: &HostMemoryAccount,
    ) -> Result<usize> {
        self.inner.component_backing_bytes(c, host)
    }
    pub fn maximum_group_backing_bytes(
        &self,
        family: &PreparedModelFamily,
        plan: &ExecutionPlan,
        host: &HostMemoryAccount,
    ) -> Result<usize> {
        let p = plan.payload();
        ensure!(
            family.family_id().as_str() == Qwen3VLVisionHfFamily::FAMILY_ID
                && p.family_id() == family.family_id()
                && p.prepared_family_fingerprint() == family.fingerprint()?
                && p.program_fingerprint() == family.program().fingerprint()?
                && family.stored_weight_schema()? == self.identity.schema()
                && p.execution_weights().stored()? == &self.identity,
            "HF visual source requires its exact family/schema/program/identity mapping"
        );
        let placed = p
            .nodes()
            .iter()
            .flat_map(|n| n.values())
            .filter(|v| v.usage() == BufferUsage::Weights)
            .flat_map(|v| v.storage().components())
            .filter_map(|c| c.component_id().cloned())
            .collect::<BTreeSet<_>>();
        ensure!(
            placed.iter().all(|id| self
                .identity
                .schema()
                .components
                .iter()
                .any(|c| &c.id == id)),
            "foreign HF visual static component"
        );
        let mut groups = BTreeMap::<Vec<WeightId>, usize>::new();
        for c in &self.identity.schema().components {
            if !placed.contains(&c.id) {
                continue;
            }
            let sources = self
                .identity
                .component_sources()
                .get(&c.id)
                .context("HF visual component lacks source mapping")?;
            let n = groups.entry(sources.clone()).or_default();
            *n = n
                .checked_add(self.component_backing_bytes(c, host)?)
                .context("HF visual source group physical capacity overflow")?;
        }
        Ok(groups.into_values().max().unwrap_or(0))
    }
}
