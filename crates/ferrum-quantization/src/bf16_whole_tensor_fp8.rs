//! Explicit cold-path BF16 matrix conversion into E4M3FN and one F32 scale per matrix.
//!
//! Ordered logical-weight groups share an amax and numerical scale, with a
//! distinct physical scale component for each execution matrix. Only declared
//! FP8 dense consumers authorize conversion.
//! Source mappings remain retained mmap ranges; converted payloads use the
//! caller's Host account. This does not account for mmap residency or establish
//! deployment/model quality from the small numerical conversion fixtures.

use ferrum_runtime::*;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::num::NonZeroU64;

mod conversion;
mod host_budget;
mod host_cache;
mod lookup;
mod plan;
mod quality;
mod source;
mod timing;
pub use source::ExactBf16TensorSource;
/// Backward-compatible spelling; the source now accepts any tensor archive.
pub type ExactBf16SafetensorsSource<'a> = ExactBf16TensorSource<'a>;
#[cfg(test)]
mod tests;

pub const BF16_WHOLE_TENSOR_FP8_MATERIALIZER_ID: &str =
    "weight-materializer.cpu.bf16-whole-tensor-e4m3fn";
pub const BF16_WHOLE_TENSOR_FP8_LAYOUT_ID: &str =
    "weight-layout.execution.bf16-whole-tensor-e4m3fn.mixed.v1";
pub const BF16_WHOLE_TENSOR_FP8_NUMERICAL_PROFILE: &str =
    "cpu.bf16.e4m3fn.shared-amax.static-reciprocal.finite.v1";
const VERSION: ContractVersion = ContractVersion::new(1, 0);

/// A finite-domain, approximate materializer. `groups` contains ordered logical
/// WeightIds, not checkpoint-name patterns. Group order is canonicalized; order
/// within each group is retained in provenance and implementation identity.
///
/// Scale is RN32(amax/448), then quantization uses RN32(x*RN32(1/scale)) and
/// saturating E4M3FN round-to-nearest-even. All-zero groups use scale 1. Nonfinite
/// inputs, or nonzero groups whose scale/reciprocal is not finite positive, fail.
/// This static reciprocal path deliberately differs from activation division
/// and its scale floor. Selection still passes the runtime's quality verifier.
pub struct Bf16WholeTensorFp8Materializer {
    descriptor: WeightMaterializerDescriptor,
    host: HostMemoryAccount,
    groups: Vec<Vec<WeightId>>,
    host_cache: Option<host_cache::HostCache>,
}

impl Bf16WholeTensorFp8Materializer {
    pub fn new(host: HostMemoryAccount, groups: Vec<Vec<WeightId>>) -> Result<Self, VNextError> {
        Self::new_with_id(
            WeightMaterializerId::new(BF16_WHOLE_TENSOR_FP8_MATERIALIZER_ID)?,
            host,
            groups,
        )
    }

    /// Distinct composition-owned identities allow independently configured
    /// stages to coexist in one registry; unknown family weights still fail.
    pub fn new_with_id(
        id: WeightMaterializerId,
        host: HostMemoryAccount,
        mut groups: Vec<Vec<WeightId>>,
    ) -> Result<Self, VNextError> {
        if groups.is_empty() || groups.len() > 4096 {
            return Err(invalid("requires 1..=4096 explicit conversion groups"));
        }
        let mut seen = BTreeSet::new();
        for group in &groups {
            if group.is_empty() || group.len() > 32 {
                return Err(invalid("each ordered group requires 1..=32 weights"));
            }
            for weight in group {
                if !seen.insert(weight.clone()) {
                    return Err(invalid("logical weight occurs in multiple group positions"));
                }
            }
        }
        groups.sort();
        let policy = serde_json::to_vec(&groups).map_err(|e| invalid(e.to_string()))?;
        let descriptor = WeightMaterializerDescriptor::new(
            id,
            VERSION,
            fingerprint(&[
                BF16_WHOLE_TENSOR_FP8_NUMERICAL_PROFILE.as_bytes(),
                &policy,
                include_bytes!("bf16_whole_tensor_fp8.rs"),
                include_bytes!("bf16_whole_tensor_fp8/plan.rs"),
                include_bytes!("bf16_whole_tensor_fp8/conversion.rs"),
                include_bytes!("bf16_whole_tensor_fp8/host_budget.rs"),
                include_bytes!("bf16_whole_tensor_fp8/lookup.rs"),
                include_bytes!("bf16_whole_tensor_fp8/timing.rs"),
                include_bytes!("bf16_whole_tensor_fp8/host_cache.rs"),
                include_bytes!("bf16_whole_tensor_fp8/quality.rs"),
            ]),
            WeightMaterializationFidelity::Approximate,
            BTreeSet::new(),
        )?
        .with_approximate_quality_contract(quality::contract()?)?;
        Ok(Self {
            descriptor,
            host,
            groups,
            host_cache: None,
        })
    }

    /// Retain successful converted groups in this materializer's bounded LRU.
    /// Clones supplied to uploads keep the original WeightBacking charge after
    /// eviction. A group larger than this cap converts normally without caching.
    /// Source mappings remain independently retained, immutable mmap owners.
    pub fn with_host_cache(mut self, maximum_bytes: NonZeroU64) -> Result<Self, VNextError> {
        if maximum_bytes.get()
            > self
                .host
                .limits()
                .class_bytes(HostMemoryClass::WeightBacking) as u64
            || maximum_bytes.get() > self.host.limits().total_bytes() as u64
        {
            return Err(invalid(
                "Host conversion cache exceeds its WeightBacking or total allowance",
            ));
        }
        self.host_cache = Some(host_cache::HostCache::new(
            maximum_bytes,
            format!(
                "{}:{}",
                self.descriptor.id(),
                self.descriptor.implementation_fingerprint()
            ),
        ));
        Ok(self)
    }

    /// Metadata-only bytes for all complete converted groups. Original borrowed
    /// components, mmap RSS and concurrent transfer/temporary allocations are
    /// excluded; the composition must separately budget those live owners.
    pub fn converted_cache_bytes(&self, family: &PreparedModelFamily) -> Result<u64, VNextError> {
        Ok(plan::prepare(family, &self.groups)?.converted_bytes)
    }

    /// Small fixed conversion evidence bound to this implementation and the
    /// live schemas. It is not an approval of full-model image quality.
    pub fn selection(
        &self,
        family: &PreparedModelFamily,
    ) -> Result<WeightMaterializerSelection, VNextError> {
        let prepared = plan::prepare(family, &self.groups)?;
        WeightMaterializerSelection::numeric_quality_artifact(
            self.descriptor.id().clone(),
            quality::artifact(
                &self.descriptor,
                family.stored_weight_schema()?,
                &prepared.schema,
            )?,
        )
    }

    /// Peak logical bytes of converted payload plus temporary lookup for one
    /// complete initializer group. Separate allocation padding is excluded;
    /// use `maximum_group_accounted_bytes` for Host capacity admission.
    pub fn maximum_group_backing_bytes(
        &self,
        family: &PreparedModelFamily,
    ) -> Result<u64, VNextError> {
        Ok(plan::prepare(family, &self.groups)?.maximum_group_bytes)
    }

    /// Peak charged bytes for a complete conversion group on this materializer's
    /// injected Host account. Payload and temporary lookup are distinct claims,
    /// so each allocation's alignment padding is included independently.
    /// Retained cache entries, transfer staging and mmap residency are excluded.
    pub fn maximum_group_accounted_bytes(
        &self,
        family: &PreparedModelFamily,
    ) -> Result<u64, VNextError> {
        plan::prepare(family, &self.groups)?.maximum_group_accounted_bytes(&self.host)
    }

    /// Required physical execution payload bytes, including every matrix's
    /// physical scale scalar. This is a metadata-only lower bound: placement alignment, dynamic
    /// resources and runtime/library reservations are not included.
    pub fn execution_weight_bytes(&self, family: &PreparedModelFamily) -> Result<u64, VNextError> {
        plan::prepare(family, &self.groups)?.execution_weight_bytes()
    }
}

impl WeightMaterializer for Bf16WholeTensorFp8Materializer {
    fn descriptor(&self) -> &WeightMaterializerDescriptor {
        &self.descriptor
    }
    fn execution_schema(
        &self,
        family: &PreparedModelFamily,
        _device: &DeviceDescriptor,
    ) -> Result<WeightSchema, VNextError> {
        Ok(plan::prepare(family, &self.groups)?.schema)
    }
    fn component_sources(
        &self,
        family: &PreparedModelFamily,
        schema: &WeightSchema,
    ) -> Result<BTreeMap<WeightId, Vec<WeightId>>, VNextError> {
        let prepared = plan::prepare(family, &self.groups)?;
        if &prepared.schema != schema {
            return Err(invalid("execution schema differs"));
        }
        Ok(prepared.sources)
    }
    fn static_weight_transforms(
        &self,
        family: &PreparedModelFamily,
        schema: &WeightSchema,
    ) -> Result<Vec<StaticWeightTransformPlan>, VNextError> {
        self.component_sources(family, schema)?;
        Ok(Vec::new())
    }
    fn materialize_component<'source>(
        &self,
        source: &'source dyn WeightComponentSource,
        sources: &[&WeightComponentSpec],
        output: &WeightComponentSpec,
    ) -> Result<WeightComponentPayload<'source>, VNextError> {
        Ok(self
            .materialize_components(source, sources, &[output])?
            .remove(0))
    }
    fn materialize_components<'source>(
        &self,
        source: &'source dyn WeightComponentSource,
        sources: &[&WeightComponentSpec],
        outputs: &[&WeightComponentSpec],
    ) -> Result<Vec<WeightComponentPayload<'source>>, VNextError> {
        self.materialize_components_with_host(source, sources, outputs, &self.host)
    }
    fn materialize_components_with_host<'source>(
        &self,
        source: &'source dyn WeightComponentSource,
        sources: &[&WeightComponentSpec],
        outputs: &[&WeightComponentSpec],
        host: &HostMemoryAccount,
    ) -> Result<Vec<WeightComponentPayload<'source>>, VNextError> {
        if !self.host.same_parent(host) {
            return Err(invalid("invocation Host account has a different parent"));
        }
        match &self.host_cache {
            Some(cache) => {
                conversion::materialize_with_cache(host, source, sources, outputs, Some(cache))
            }
            None => conversion::materialize(host, source, sources, outputs),
        }
    }
    fn additional_host_backing_bytes(
        &self,
        family: &PreparedModelFamily,
        source: Option<&dyn WeightComponentSource>,
    ) -> Result<usize, VNextError> {
        host_budget::additional(
            &self.host,
            &plan::prepare(family, &self.groups)?,
            self.host_cache.as_ref(),
            source,
        )
    }
    fn prepare_host_backing(
        &self,
        family: &PreparedModelFamily,
        source: Option<&dyn WeightComponentSource>,
    ) -> Result<HostBackingPreparation, VNextError> {
        host_budget::prepare(
            &self.host,
            &plan::prepare(family, &self.groups)?,
            self.host_cache.as_ref(),
            source,
        )
    }
}

fn fingerprint(parts: &[&[u8]]) -> String {
    let mut hash = Sha256::new();
    for part in parts {
        hash.update((part.len() as u64).to_le_bytes());
        hash.update(part);
    }
    format!("{:x}", hash.finalize())
}
fn invalid(reason: impl Into<String>) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: format!("BF16 whole-tensor FP8: {}", reason.into()),
    }
}
