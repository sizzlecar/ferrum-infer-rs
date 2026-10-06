//! Explicit cold-path GGUF projection conversion. Operation consumers authorize
//! the mathematical change; model/tensor names never do. This materializer does
//! not authorize serving quality or manufacture device cost evidence.

use ferrum_interfaces::vnext::*;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

mod conversion;
mod plan;
mod quality;
#[cfg(test)]
mod tests;
pub use plan::GgufRnFragmentInventoryV1;

pub const GGUF_RN_FRAGMENT_MATERIALIZER_ID: &str =
    "weight-materializer.cuda.gguf-rn-f16-projections-fragment";
pub const GGUF_RN_FRAGMENT_FORMAT_ID: &str =
    "weight-format.execution.cuda.gguf-rn-f16-projections-fragment-mixed";
pub const GGUF_RN_FRAGMENT_LAYOUT_ID: &str =
    "weight-layout.execution.cuda.gguf-rn-f16-projections-fragment-mixed";
pub const GGUF_RN_FRAGMENT_CAPABILITY_ID: &str =
    "capability.weight-materializer.cuda.gguf-rn-f16-projections-fragment";
const VERSION: ContractVersion = ContractVersion::new(1, 0);

pub fn gguf_rn_fragment_materializer() -> Result<Box<dyn WeightMaterializer>, VNextError> {
    Ok(Box::new(GgufRnFragmentMaterializer::new()?))
}

/// An explicit operation contract, not a family-name test. The full plan checks
/// every use, the source schema and all physical leaves before selection.
pub fn requests_gguf_rn_fragment_materialization(family: &PreparedModelFamily) -> bool {
    family
        .program()
        .blocks()
        .iter()
        .flat_map(|block| &block.nodes)
        .any(|node| {
            node.inputs.iter().enumerate().any(|(ordinal, _)| {
                u32::try_from(ordinal)
                    .ok()
                    .and_then(|ordinal| gguf_rn_f16_fragment_role_v1(&node.operation_id, ordinal))
                    .is_some()
            })
        })
}

pub fn gguf_rn_fragment_materializer_selection(
    family: &PreparedModelFamily,
) -> Result<WeightMaterializerSelection, VNextError> {
    let materializer = GgufRnFragmentMaterializer::new()?;
    let prepared = plan::prepare(family)?;
    let artifact = quality::artifact(
        materializer.descriptor(),
        family.weight_schema(),
        &prepared.schema,
    )?;
    WeightMaterializerSelection::numeric_quality_artifact(
        materializer.descriptor.id().clone(),
        artifact,
    )
}

pub fn gguf_rn_fragment_inventory(
    family: &PreparedModelFamily,
) -> Result<GgufRnFragmentInventoryV1, VNextError> {
    Ok(plan::prepare(family)?.inventory)
}

struct GgufRnFragmentMaterializer {
    descriptor: WeightMaterializerDescriptor,
}
impl GgufRnFragmentMaterializer {
    fn new() -> Result<Self, VNextError> {
        let descriptor = WeightMaterializerDescriptor::new(
            WeightMaterializerId::new(GGUF_RN_FRAGMENT_MATERIALIZER_ID)?,
            VERSION,
            fingerprint(&[
                include_bytes!("gguf_rn_fragment_materializer.rs"),
                include_bytes!("gguf_rn_fragment_materializer/plan.rs"),
                include_bytes!("gguf_rn_fragment_materializer/conversion.rs"),
                include_bytes!("gguf_rn_fragment_materializer/quality.rs"),
                include_bytes!("gguf_rn_fragment.rs"),
                include_bytes!("gguf_f16_projection_materializer.rs"),
                include_bytes!("gguf_f16_projection_materializer/plan.rs"),
                include_bytes!("gguf_f16_projection_materializer/conversion.rs"),
                include_bytes!("gguf_f16_projection_materializer/quality.rs"),
                include_bytes!("gguf_blocks/mod.rs"),
                include_bytes!("gguf_blocks/block_decode.rs"),
            ]),
            WeightMaterializationFidelity::Approximate,
            BTreeSet::from([CapabilityId::new(GGUF_RN_FRAGMENT_CAPABILITY_ID)?]),
        )?
        .with_approximate_quality_contract(quality::contract()?)?;
        Ok(Self { descriptor })
    }
}
impl WeightMaterializer for GgufRnFragmentMaterializer {
    fn descriptor(&self) -> &WeightMaterializerDescriptor {
        &self.descriptor
    }
    fn execution_schema(
        &self,
        family: &PreparedModelFamily,
        _device: &DeviceDescriptor,
    ) -> Result<WeightSchema, VNextError> {
        Ok(plan::prepare(family)?.schema)
    }
    fn component_sources(
        &self,
        family: &PreparedModelFamily,
        execution_schema: &WeightSchema,
    ) -> Result<BTreeMap<WeightId, Vec<WeightId>>, VNextError> {
        let prepared = plan::prepare(family)?;
        if &prepared.schema != execution_schema {
            return Err(invalid(
                "GGUF RN-F16 execution schema differs from its checked consumer plan",
            ));
        }
        Ok(prepared.sources)
    }
    fn static_weight_transforms(
        &self,
        family: &PreparedModelFamily,
        execution_schema: &WeightSchema,
    ) -> Result<Vec<StaticWeightTransformPlan>, VNextError> {
        // The trusted-plan adoption boundary invokes this existing callback
        // again with the current family. Equal source schemas do not authorize
        // a converted plan for strict or otherwise different consumers.
        let prepared = plan::prepare(family)?;
        if &prepared.schema != execution_schema {
            return Err(invalid(
                "GGUF RN-F16 trusted plan differs from the current typed consumer plan",
            ));
        }
        // Conversion remains a cold host materialization; no additional device
        // transform or new numerical approval is granted by this revalidation.
        Ok(Vec::new())
    }
    fn materialize_component<'source>(
        &self,
        source: &'source dyn WeightComponentSource,
        source_components: &[&WeightComponentSpec],
        execution_component: &WeightComponentSpec,
    ) -> Result<WeightComponentPayload<'source>, VNextError> {
        let mut payloads =
            conversion::materialize_group(source, source_components, &[execution_component])?;
        Ok(payloads.remove(0))
    }
    fn materialize_components<'source>(
        &self,
        source: &'source dyn WeightComponentSource,
        source_components: &[&WeightComponentSpec],
        execution_components: &[&WeightComponentSpec],
    ) -> Result<Vec<WeightComponentPayload<'source>>, VNextError> {
        conversion::materialize_group(source, source_components, execution_components)
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
        reason: reason.into(),
    }
}
