use std::collections::BTreeSet;

use serde::{Deserialize, Deserializer, Serialize};

use super::{
    canonical_fingerprint, invalid_plan, CapabilityCatalog, ExecutionWeightPlan, ModelFamilyId,
    PreparedModelFamily, QuantizationFormatId, ResourceId, StaticWeightTransformPlan,
    TrustedExecutionWeightPlan, VNextError, WeightFormatId, WeightSchema,
};
use crate::{
    ModelSemanticMetadata, ModelWeights, WeightComponentPayload, WeightComponentSource,
    WeightComponentSpec,
};

#[cfg(test)]
mod tests;

/// Whether a plan consumes stored model weights. Absence carries no source,
/// materializer, schema or artifact identity.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", content = "plan", rename_all = "snake_case")]
pub enum ExecutionWeights {
    None,
    Stored(ExecutionWeightPlan),
}

#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum ExecutionWeightsWire {
    None {},
    Stored { plan: ExecutionWeightPlan },
}

impl<'de> Deserialize<'de> for ExecutionWeights {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Ok(match ExecutionWeightsWire::deserialize(deserializer)? {
            ExecutionWeightsWire::None {} => Self::None,
            ExecutionWeightsWire::Stored { plan } => Self::Stored(plan),
        })
    }
}

fn validate_absence(family: &PreparedModelFamily) -> Result<(), VNextError> {
    if !matches!(family.weights(), ModelWeights::None)
        || !matches!(family.metadata(), ModelSemanticMetadata::TensorComponent)
        || !family.program().weights().is_empty()
    {
        return Err(invalid_plan(
            "absent execution weights require a weightless tensor component",
        ));
    }
    Ok(())
}

impl ExecutionWeights {
    pub(crate) fn for_family(family: &PreparedModelFamily) -> Result<Self, VNextError> {
        match family.weights() {
            ModelWeights::None => {
                validate_absence(family)?;
                Ok(Self::None)
            }
            ModelWeights::Stored(_) => ExecutionWeightPlan::identity(family).map(Self::Stored),
        }
    }

    pub fn is_none(&self) -> bool {
        matches!(self, Self::None)
    }

    pub fn schema(&self) -> Option<&WeightSchema> {
        match self {
            Self::None => None,
            Self::Stored(plan) => Some(plan.schema()),
        }
    }

    pub fn stored(&self) -> Result<&ExecutionWeightPlan, VNextError> {
        match self {
            Self::None => Err(invalid_plan("operation requires stored execution weights")),
            Self::Stored(plan) => Ok(plan),
        }
    }

    pub fn weight_format(&self) -> Option<&WeightFormatId> {
        self.schema().map(|schema| &schema.format_id)
    }

    pub fn quantization_formats(&self) -> BTreeSet<QuantizationFormatId> {
        self.schema()
            .map(WeightSchema::quantization_formats)
            .unwrap_or_default()
    }

    pub fn fingerprint(&self) -> Result<String, VNextError> {
        canonical_fingerprint(self, "fingerprint execution weight presence")
    }

    pub fn static_weight_transforms(&self) -> &[StaticWeightTransformPlan] {
        match self {
            Self::None => &[],
            Self::Stored(plan) => plan.static_weight_transforms(),
        }
    }

    pub fn maximum_static_weight_transform_scratch_bytes(&self) -> Result<u64, VNextError> {
        match self {
            Self::None => Ok(0),
            Self::Stored(plan) => plan.maximum_static_weight_transform_scratch_bytes(),
        }
    }

    pub fn static_weight_transform_scratch_resource_id(
        &self,
    ) -> Result<Option<ResourceId>, VNextError> {
        match self {
            Self::None => Ok(None),
            Self::Stored(plan) => plan.static_weight_transform_scratch_resource_id(),
        }
    }

    pub(super) fn validate_structure(&self, family_id: &ModelFamilyId) -> Result<(), VNextError> {
        match self {
            Self::None => Ok(()),
            Self::Stored(plan) => plan.validate_structure(family_id),
        }
    }

    pub(crate) fn validate_against_family(
        &self,
        family: &PreparedModelFamily,
    ) -> Result<(), VNextError> {
        match self {
            Self::None => validate_absence(family),
            Self::Stored(plan) => {
                family.stored_weight_schema()?;
                plan.validate_against_family(family)
            }
        }
    }
}

/// Process-local authority for weight presence. The None proof binds the full
/// prepared family and can only be created by core. A wire declaration cannot
/// construct this type or a materializer authority.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrustedExecutionWeights {
    authority: ExecutionWeightAuthority,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum ExecutionWeightAuthority {
    None { prepared_family_fingerprint: String },
    Stored(TrustedExecutionWeightPlan),
}

impl From<TrustedExecutionWeightPlan> for TrustedExecutionWeights {
    fn from(plan: TrustedExecutionWeightPlan) -> Self {
        Self {
            authority: ExecutionWeightAuthority::Stored(plan),
        }
    }
}

impl TrustedExecutionWeights {
    pub(crate) fn for_family(family: &PreparedModelFamily) -> Result<Self, VNextError> {
        match family.weights() {
            ModelWeights::None => {
                validate_absence(family)?;
                Ok(Self {
                    authority: ExecutionWeightAuthority::None {
                        prepared_family_fingerprint: family.fingerprint()?,
                    },
                })
            }
            ModelWeights::Stored(_) => TrustedExecutionWeightPlan::identity(family).map(Self::from),
        }
    }

    pub fn declaration(&self) -> ExecutionWeights {
        match &self.authority {
            ExecutionWeightAuthority::None { .. } => ExecutionWeights::None,
            ExecutionWeightAuthority::Stored(plan) => ExecutionWeights::Stored(plan.plan().clone()),
        }
    }

    pub fn schema(&self) -> Option<&WeightSchema> {
        match &self.authority {
            ExecutionWeightAuthority::None { .. } => None,
            ExecutionWeightAuthority::Stored(plan) => Some(plan.plan().schema()),
        }
    }

    fn stored(&self) -> Result<&TrustedExecutionWeightPlan, VNextError> {
        match &self.authority {
            ExecutionWeightAuthority::None { .. } => Err(invalid_plan(
                "weight materialization requires stored weight authority",
            )),
            ExecutionWeightAuthority::Stored(plan) => Ok(plan),
        }
    }

    pub(crate) fn validate_against_family(
        &self,
        family: &PreparedModelFamily,
    ) -> Result<(), VNextError> {
        match &self.authority {
            ExecutionWeightAuthority::None {
                prepared_family_fingerprint,
            } => {
                validate_absence(family)?;
                if prepared_family_fingerprint != &family.fingerprint()? {
                    return Err(invalid_plan(
                        "absent weight authority belongs to a different prepared family",
                    ));
                }
                Ok(())
            }
            ExecutionWeightAuthority::Stored(plan) => {
                family.stored_weight_schema()?;
                plan.plan().validate_against_family(family)
            }
        }
    }

    pub(crate) fn validate_against_catalog(
        &self,
        family: &PreparedModelFamily,
        catalog: &CapabilityCatalog,
    ) -> Result<(), VNextError> {
        self.validate_against_family(family)?;
        match &self.authority {
            ExecutionWeightAuthority::None { .. } => Ok(()),
            ExecutionWeightAuthority::Stored(plan) => {
                plan.validate_against_catalog(family, catalog)
            }
        }
    }

    pub(crate) fn materialize_components<'source>(
        &self,
        family: &PreparedModelFamily,
        source: &'source dyn WeightComponentSource,
        components: &[&WeightComponentSpec],
        host: Option<&crate::HostMemoryAccount>,
    ) -> Result<Vec<WeightComponentPayload<'source>>, VNextError> {
        self.validate_against_family(family)?;
        self.stored()?
            .materialize_components(family, source, components, host)
    }

    pub(crate) fn additional_host_backing_bytes(
        &self,
        family: &PreparedModelFamily,
        source: Option<&dyn WeightComponentSource>,
    ) -> Result<usize, VNextError> {
        self.validate_against_family(family)?;
        match &self.authority {
            ExecutionWeightAuthority::None { .. } => Ok(0),
            ExecutionWeightAuthority::Stored(plan) => {
                plan.additional_host_backing_bytes(family, source)
            }
        }
    }

    pub(crate) fn prepare_host_backing(
        &self,
        family: &PreparedModelFamily,
        source: Option<&dyn WeightComponentSource>,
    ) -> Result<crate::HostBackingPreparation, VNextError> {
        self.validate_against_family(family)?;
        match &self.authority {
            ExecutionWeightAuthority::None { .. } => Ok(crate::HostBackingPreparation::new(0)),
            ExecutionWeightAuthority::Stored(plan) => plan.prepare_host_backing(family, source),
        }
    }

    pub(crate) fn static_weight_transform_for_components(
        &self,
        components: &[&WeightComponentSpec],
    ) -> Result<Option<&StaticWeightTransformPlan>, VNextError> {
        self.stored()?
            .static_weight_transform_for_components(components)
    }
}
