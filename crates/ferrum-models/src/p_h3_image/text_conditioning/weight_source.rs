//! Checked H3 text weights over the caller's current Host authorization. Only
//! gamma vectors used by declared F32 head-RMS operations may promote BF16;
//! quantized matrices and F8 scale carriers remain original borrowed storage.
use super::*;
use ferrum_quantization::ManagedDenseArchiveSource;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

pub struct PH3ImageTextWeightAdapter {
    source: Arc<ComfyCanonicalWeightSource>,
    identity: ExecutionWeightPlan,
    index: BTreeMap<WeightId, usize>,
    promotions: BTreeSet<WeightId>,
}
pub struct PH3ImageTextWeightSource<'a> {
    adapter: &'a PH3ImageTextWeightAdapter,
    host: &'a HostMemoryAccount,
    dense: ManagedDenseArchiveSource<'a>,
}
impl PH3ImageTextWeightAdapter {
    pub fn new(
        config: &PH3ImageTextConditioningConfig,
        source: Arc<ComfyCanonicalWeightSource>,
        limits: PH3ImageTextConditioningLimits,
    ) -> Result<Self> {
        PH3ImageTextConditioningCheckpoint::new(config, &source, limits)?;
        let prepared = TypedFamilyRegistration::new(PH3ImageTextConditioningFamily::new(limits)?)
            .prepare_with_profile(
            &serde_json::to_value(config)?,
            &NumericalProfileId::new(PH3ImageTextConditioningFamily::cuda_profile_id(config))
                .map_err(invalid)?,
        )?;
        let identity = ExecutionWeightPlan::identity(&prepared)?;
        let schema = identity.schema();
        // Derive the exact norm whitelist from typed operation weight uses,
        // rather than duplicating Qwen parameter-name rules in this adapter.
        let references = prepared
            .program()
            .weights()
            .iter()
            .map(|w| (&w.value_id, &w.weight_id))
            .collect::<BTreeMap<_, _>>();
        let mut norms = BTreeSet::new();
        for node in prepared
            .program()
            .blocks()
            .iter()
            .flat_map(|b| &b.nodes)
            .filter(|n| n.operation_id.as_str() == HEAD_RMS_NORM_F32_OPERATION_ID)
        {
            let gamma = node
                .inputs
                .get(1)
                .context("declared F32 RMS operation lacks gamma")?;
            let id = references
                .get(gamma)
                .context("declared norm gamma is not a stored tensor")?;
            let components = schema.physical_component_refs(id)?;
            ensure!(
                components.len() == 1,
                "norm gamma requires one original dense component"
            );
            let c = components[0];
            ensure!(
                c.dimensions.len() == 1
                    && c.encoding
                        == (WeightEncoding::Dense {
                            element_type: ElementType::F32
                        }),
                "norm gamma requires original vector-to-F32 contract"
            );
            norms.insert(c.id.clone());
        }
        let mut promotions = BTreeSet::new();
        let mut index = BTreeMap::new();
        for (position, c) in schema.components.iter().enumerate() {
            let [name] = c.external_names.as_slice() else {
                anyhow::bail!("H3 text source requires one original tensor per component")
            };
            let raw = source.archive().tensor(name)?;
            ensure!(
                raw.shape() == c.dimensions,
                "H3 text original component shape differs: {name}"
            );
            let WeightEncoding::Dense { element_type } = c.encoding else {
                anyhow::bail!(
                    "H3 text execution components must declare exact dense physical ports"
                )
            };
            if norms.contains(&c.id) {
                ensure!(
                    raw.element_type() == Some(ElementType::Bf16),
                    "declared H3 text norm must retain original BF16 storage"
                );
                promotions.insert(c.id.clone());
            } else if element_type == ElementType::U8 && name.ends_with(".weight_s_rel") {
                ensure!(
                    raw.dtype() == safetensors::Dtype::F8_E4M3,
                    "relative scale carrier must retain original F8 bytes"
                );
            } else {
                ensure!(
                    raw.element_type() == Some(element_type),
                    "non-norm H3 text component may not transcode: {name}"
                );
            }
            ensure!(
                index.insert(c.id.clone(), position).is_none(),
                "duplicate H3 text physical component identity"
            );
        }
        let metadata = serde_json::to_vec(&identity)?
            .len()
            .checked_add(
                schema
                    .components
                    .len()
                    .checked_mul(
                        std::mem::size_of::<(WeightId, usize)>() + std::mem::size_of::<WeightId>(),
                    )
                    .context("text source metadata overflow")?,
            )
            .context("text source metadata overflow")?;
        ensure!(
            metadata <= limits.max_metadata_bytes,
            "typed text source metadata exceeds declared capacity"
        );
        Ok(Self {
            source,
            identity,
            index,
            promotions,
        })
    }
    pub fn weight_source<'a>(
        &'a self,
        host: &'a HostMemoryAccount,
    ) -> PH3ImageTextWeightSource<'a> {
        PH3ImageTextWeightSource {
            adapter: self,
            host,
            dense: ManagedDenseArchiveSource::new(self.source.archive(), host),
        }
    }
    fn check_component(&self, c: &WeightComponentSpec) -> Result<bool> {
        let position = self
            .index
            .get(&c.id)
            .context("component is absent from the verified H3 text schema")?;
        ensure!(
            &self.identity.schema().components[*position] == c,
            "component differs from the verified H3 text physical contract"
        );
        Ok(self.promotions.contains(&c.id))
    }
    /// Metadata only. Every managed component rounds separately for the actual
    /// Host allocator's padding; borrowed canonical storage costs zero here.
    pub fn component_backing_bytes(
        &self,
        c: &WeightComponentSpec,
        host: &HostMemoryAccount,
    ) -> Result<usize> {
        if self.check_component(c)? {
            Ok(host.backing_bytes(usize::try_from(c.physical_bytes()?)?)?)
        } else {
            Ok(0)
        }
    }
    /// Same grouping as runtime static initialization: placed components are
    /// grouped by their exact source-ID vector; group payloads coexist until
    /// copied into separately admitted TransferStaging. This is not a claim.
    pub fn maximum_group_backing_bytes(
        &self,
        family: &PreparedModelFamily,
        plan: &ExecutionPlan,
        host: &HostMemoryAccount,
    ) -> Result<usize> {
        let payload = plan.payload();
        ensure!(
            family.family_id().as_str() == PH3ImageTextConditioningFamily::FAMILY_ID
                && payload.family_id() == family.family_id()
                && payload.prepared_family_fingerprint() == family.fingerprint()?
                && payload.program_fingerprint() == family.program().fingerprint()?,
            "Host source demand belongs to another model family"
        );
        ensure!(
            family.stored_weight_schema()? == self.identity.schema(),
            "prepared text family changes the validated immutable weight contract"
        );
        self.check_identity(payload.execution_weights().stored()?)?;
        let placed = payload
            .nodes()
            .iter()
            .flat_map(|n| n.values())
            .filter(|v| v.usage() == BufferUsage::Weights)
            .flat_map(|v| v.storage().components())
            .filter_map(|c| c.component_id().cloned())
            .collect::<BTreeSet<_>>();
        self.group_peak(&placed, host)
    }
    fn check_identity(&self, plan: &ExecutionWeightPlan) -> Result<()> {
        ensure!(plan==&self.identity,"H3 text Host demand supports only this exact identity materializer/schema/source map, without required transforms or quality substitution");
        Ok(())
    }
    fn group_peak(&self, placed: &BTreeSet<WeightId>, host: &HostMemoryAccount) -> Result<usize> {
        ensure!(
            placed.iter().all(|id| self.index.contains_key(id)),
            "static placement contains a foreign H3 text component"
        );
        let mut groups = BTreeMap::<Vec<WeightId>, usize>::new();
        for c in &self.identity.schema().components {
            if !placed.contains(&c.id) {
                continue;
            }
            let ids = self
                .identity
                .component_sources()
                .get(&c.id)
                .context("static component has no identity source mapping")?;
            let group = groups.entry(ids.clone()).or_default();
            *group = group
                .checked_add(self.component_backing_bytes(c, host)?)
                .context("static text promotion group Host extent overflow")?;
        }
        Ok(groups.into_values().max().unwrap_or(0))
    }
}
impl WeightComponentSource for PH3ImageTextWeightSource<'_> {
    fn component<'s>(
        &'s self,
        c: &WeightComponentSpec,
    ) -> std::result::Result<WeightComponentPayload<'s>, VNextError> {
        if self
            .adapter
            .check_component(c)
            .map_err(|e| invalid(e.to_string()))?
        {
            self.dense.component(c)
        } else {
            self.adapter.source.component(c)
        }
    }
}
impl PH3ImageTextWeightSource<'_> {
    pub fn component_backing_bytes(&self, c: &WeightComponentSpec) -> Result<usize> {
        self.adapter.component_backing_bytes(c, self.host)
    }
}
#[cfg(test)]
mod tests;
