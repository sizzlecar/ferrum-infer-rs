use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use super::{
    canonical_fingerprint, invalid_plan, is_canonical_sha256, CapabilityCatalog, CapabilityId,
    ContractVersion, Deserialize, DeviceDescriptor, ModelFamilyId, PreparedModelFamily, Serialize,
    VNextError, WeightMaterializerId, WeightSchema,
};

pub const IDENTITY_WEIGHT_MATERIALIZER_ID: &str = "weight-materializer.identity";
const IDENTITY_MATERIALIZER_VERSION: ContractVersion = ContractVersion::new(1, 0);
pub const MAX_WEIGHT_MATERIALIZERS: usize = 64;

#[derive(Serialize)]
struct IdentityMaterializerFingerprint<'a> {
    id: &'a str,
    version: ContractVersion,
    contract: &'a str,
}

fn identity_materializer_fingerprint() -> Result<String, VNextError> {
    canonical_fingerprint(
        &IdentityMaterializerFingerprint {
            id: IDENTITY_WEIGHT_MATERIALIZER_ID,
            version: IDENTITY_MATERIALIZER_VERSION,
            contract: "execution-weight-plan.identity.v1",
        },
        "fingerprint identity weight materializer",
    )
}

/// Serializable capability identity for one trusted weight transformation.
///
/// The descriptor advertises availability, but cannot construct an execution
/// schema. Only the process-local [`WeightMaterializerRegistry`] retains the
/// implementation object that may produce a trusted plan.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WeightMaterializerDescriptor {
    id: WeightMaterializerId,
    version: ContractVersion,
    implementation_fingerprint: String,
    required_capabilities: BTreeSet<CapabilityId>,
}

impl WeightMaterializerDescriptor {
    pub fn new(
        id: WeightMaterializerId,
        version: ContractVersion,
        implementation_fingerprint: impl Into<String>,
        required_capabilities: BTreeSet<CapabilityId>,
    ) -> Result<Self, VNextError> {
        let descriptor = Self {
            id,
            version,
            implementation_fingerprint: implementation_fingerprint.into(),
            required_capabilities,
        };
        descriptor.validate_structure()?;
        Ok(descriptor)
    }

    pub(crate) fn identity() -> Result<Self, VNextError> {
        Self::new(
            WeightMaterializerId::new(IDENTITY_WEIGHT_MATERIALIZER_ID)?,
            IDENTITY_MATERIALIZER_VERSION,
            identity_materializer_fingerprint()?,
            BTreeSet::new(),
        )
    }

    pub fn id(&self) -> &WeightMaterializerId {
        &self.id
    }

    pub const fn version(&self) -> ContractVersion {
        self.version
    }

    pub fn implementation_fingerprint(&self) -> &str {
        &self.implementation_fingerprint
    }

    pub fn required_capabilities(&self) -> &BTreeSet<CapabilityId> {
        &self.required_capabilities
    }

    pub fn fingerprint(&self) -> Result<String, VNextError> {
        canonical_fingerprint(self, "fingerprint weight materializer descriptor")
    }

    pub(crate) fn validate_for_device(&self, device: &DeviceDescriptor) -> Result<(), VNextError> {
        self.validate_structure()?;
        if !self.required_capabilities.is_subset(&device.capabilities) {
            return Err(invalid_plan(format!(
                "weight materializer `{}` requires capabilities absent from device `{}`",
                self.id, device.id
            )));
        }
        Ok(())
    }

    fn validate_structure(&self) -> Result<(), VNextError> {
        if self.version.major == 0 || !is_canonical_sha256(&self.implementation_fingerprint) {
            return Err(invalid_plan(format!(
                "weight materializer descriptor `{}` has invalid version or implementation identity",
                self.id
            )));
        }
        Ok(())
    }
}

/// Planning half of a physical weight transformation.
///
/// Implementations are invoked once while compiling an immutable plan. They
/// must derive the complete execution schema from the prepared family and
/// device descriptor without allocating device memory or reading request
/// state.
pub trait WeightMaterializerPlanner: Send + Sync {
    fn descriptor(&self) -> &WeightMaterializerDescriptor;

    fn execution_schema(
        &self,
        family: &PreparedModelFamily,
        device: &DeviceDescriptor,
    ) -> Result<WeightSchema, VNextError>;
}

struct IdentityWeightMaterializer {
    descriptor: WeightMaterializerDescriptor,
}

impl IdentityWeightMaterializer {
    fn new() -> Result<Self, VNextError> {
        Ok(Self {
            descriptor: WeightMaterializerDescriptor::identity()?,
        })
    }
}

impl WeightMaterializerPlanner for IdentityWeightMaterializer {
    fn descriptor(&self) -> &WeightMaterializerDescriptor {
        &self.descriptor
    }

    fn execution_schema(
        &self,
        family: &PreparedModelFamily,
        _device: &DeviceDescriptor,
    ) -> Result<WeightSchema, VNextError> {
        Ok(family.weight_schema().clone())
    }
}

/// Process-local registry retaining the exact implementations authorized to
/// transform checkpoint schemas. It is deliberately neither serializable nor
/// reconstructible from a capability catalog.
pub struct WeightMaterializerRegistry {
    materializers: BTreeMap<WeightMaterializerId, Arc<dyn WeightMaterializerPlanner>>,
}

impl WeightMaterializerRegistry {
    pub fn new(materializers: Vec<Box<dyn WeightMaterializerPlanner>>) -> Result<Self, VNextError> {
        if materializers.len() >= MAX_WEIGHT_MATERIALIZERS {
            return Err(invalid_plan(format!(
                "weight materializer registry exceeds {} non-identity entries",
                MAX_WEIGHT_MATERIALIZERS - 1
            )));
        }
        let identity: Arc<dyn WeightMaterializerPlanner> =
            Arc::new(IdentityWeightMaterializer::new()?);
        let mut entries = BTreeMap::from([(identity.descriptor().id().clone(), identity)]);
        for materializer in materializers {
            materializer.descriptor().validate_structure()?;
            let id = materializer.descriptor().id().clone();
            if entries
                .insert(id.clone(), Arc::from(materializer))
                .is_some()
            {
                return Err(invalid_plan(format!(
                    "duplicate weight materializer `{id}`"
                )));
            }
        }
        Ok(Self {
            materializers: entries,
        })
    }

    pub fn identity_only() -> Result<Self, VNextError> {
        Self::new(Vec::new())
    }

    /// Extends a capability catalog with descriptors from these exact
    /// process-local implementation objects.
    pub fn augment_catalog(
        &self,
        catalog: CapabilityCatalog,
    ) -> Result<CapabilityCatalog, VNextError> {
        catalog.with_weight_materializer_descriptors(self.descriptors())
    }

    /// Invokes the selected trusted implementation and returns a witness that
    /// cannot be forged by deserializing an [`ExecutionWeightPlan`].
    pub fn select(
        &self,
        family: &PreparedModelFamily,
        catalog: &CapabilityCatalog,
        materializer_id: &WeightMaterializerId,
    ) -> Result<TrustedExecutionWeightPlan, VNextError> {
        let materializer = self.materializers.get(materializer_id).ok_or_else(|| {
            invalid_plan(format!(
                "weight materializer `{materializer_id}` is not registered"
            ))
        })?;
        let descriptor = materializer.descriptor();
        let catalog_descriptor = catalog.weight_materializer(materializer_id)?;
        if descriptor != catalog_descriptor {
            return Err(invalid_plan(format!(
                "weight materializer `{materializer_id}` differs from its capability catalog descriptor"
            )));
        }
        descriptor.validate_for_device(catalog.device())?;
        let mut schema = materializer.execution_schema(family, catalog.device())?;
        schema.normalize();
        let plan = ExecutionWeightPlan::from_materializer(family, descriptor, schema)?;
        Ok(TrustedExecutionWeightPlan {
            plan,
            descriptor: descriptor.clone(),
        })
    }

    pub fn descriptors(&self) -> BTreeMap<WeightMaterializerId, WeightMaterializerDescriptor> {
        self.materializers
            .iter()
            .map(|(id, materializer)| (id.clone(), materializer.descriptor().clone()))
            .collect()
    }
}

/// Non-serializable proof that a process-local registry implementation
/// produced and validated this physical execution schema.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrustedExecutionWeightPlan {
    plan: ExecutionWeightPlan,
    descriptor: WeightMaterializerDescriptor,
}

impl TrustedExecutionWeightPlan {
    pub(crate) fn identity(family: &PreparedModelFamily) -> Result<Self, VNextError> {
        let descriptor = WeightMaterializerDescriptor::identity()?;
        Ok(Self {
            plan: ExecutionWeightPlan::from_materializer(
                family,
                &descriptor,
                family.weight_schema().clone(),
            )?,
            descriptor,
        })
    }

    pub fn plan(&self) -> &ExecutionWeightPlan {
        &self.plan
    }

    pub(crate) fn validate_against_catalog(
        &self,
        family: &PreparedModelFamily,
        catalog: &CapabilityCatalog,
    ) -> Result<(), VNextError> {
        let catalog_descriptor = catalog.weight_materializer(self.descriptor.id())?;
        if &self.descriptor != catalog_descriptor {
            return Err(invalid_plan(format!(
                "weight materializer `{}` differs from its capability catalog descriptor",
                self.descriptor.id()
            )));
        }
        self.plan
            .validate_against_materializer(family, &self.descriptor)
    }
}

/// Trusted physical weight contract selected for one immutable execution plan.
///
/// The prepared family remains the source/checkpoint contract. This plan owns
/// the physical schema consumed by providers, static memory planning, and
/// initialization. Keeping those contracts separate allows a backend to
/// select a prepared layout without changing model semantics or hiding an
/// allocation behind a provider.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutionWeightPlan {
    source_schema_fingerprint: String,
    materializer_id: WeightMaterializerId,
    materializer_version: ContractVersion,
    materializer_implementation_fingerprint: String,
    schema: WeightSchema,
}

impl ExecutionWeightPlan {
    pub fn identity(family: &PreparedModelFamily) -> Result<Self, VNextError> {
        Self::from_materializer(
            family,
            &WeightMaterializerDescriptor::identity()?,
            family.weight_schema().clone(),
        )
    }

    fn from_materializer(
        family: &PreparedModelFamily,
        descriptor: &WeightMaterializerDescriptor,
        schema: WeightSchema,
    ) -> Result<Self, VNextError> {
        let plan = Self {
            source_schema_fingerprint: family.weight_schema().fingerprint()?,
            materializer_id: descriptor.id().clone(),
            materializer_version: descriptor.version(),
            materializer_implementation_fingerprint: descriptor
                .implementation_fingerprint()
                .to_owned(),
            schema,
        };
        plan.validate_against_materializer(family, descriptor)?;
        Ok(plan)
    }

    pub fn source_schema_fingerprint(&self) -> &str {
        &self.source_schema_fingerprint
    }

    pub fn materializer_id(&self) -> &WeightMaterializerId {
        &self.materializer_id
    }

    pub const fn materializer_version(&self) -> ContractVersion {
        self.materializer_version
    }

    pub fn materializer_implementation_fingerprint(&self) -> &str {
        &self.materializer_implementation_fingerprint
    }

    pub fn schema(&self) -> &WeightSchema {
        &self.schema
    }

    pub fn fingerprint(&self) -> Result<String, VNextError> {
        canonical_fingerprint(self, "fingerprint execution weight plan")
    }

    pub(super) fn validate_structure(&self, family_id: &ModelFamilyId) -> Result<(), VNextError> {
        if !is_canonical_sha256(&self.source_schema_fingerprint)
            || !is_canonical_sha256(&self.materializer_implementation_fingerprint)
            || self.materializer_version.major == 0
        {
            return Err(VNextError::InvalidExecutionPlan {
                reason: "execution weight plan provenance is invalid".to_owned(),
            });
        }
        self.schema.validate(family_id)
    }

    pub(crate) fn validate_against_family(
        &self,
        family: &PreparedModelFamily,
    ) -> Result<(), VNextError> {
        self.validate_structure(family.family_id())?;
        if self.source_schema_fingerprint != family.weight_schema().fingerprint()? {
            return Err(invalid_plan(
                "execution weight plan source schema differs from its prepared family",
            ));
        }
        let source_tensors = family
            .weight_schema()
            .tensors
            .iter()
            .map(|tensor| (&tensor.id, tensor))
            .collect::<BTreeMap<_, _>>();
        let execution_tensors = self
            .schema
            .tensors
            .iter()
            .map(|tensor| (&tensor.id, tensor))
            .collect::<BTreeMap<_, _>>();
        if source_tensors.len() != execution_tensors.len()
            || source_tensors.iter().any(|(id, source)| {
                execution_tensors.get(id).is_none_or(|execution| {
                    source.dimensions != execution.dimensions
                        || source.logical_element_type != execution.logical_element_type
                        || source.required != execution.required
                })
            })
        {
            return Err(invalid_plan(
                "execution weight schema changes the prepared family's logical tensor contract",
            ));
        }
        Ok(())
    }

    fn validate_against_materializer(
        &self,
        family: &PreparedModelFamily,
        descriptor: &WeightMaterializerDescriptor,
    ) -> Result<(), VNextError> {
        self.validate_against_family(family)?;
        if &self.materializer_id != descriptor.id()
            || self.materializer_version != descriptor.version()
            || self.materializer_implementation_fingerprint
                != descriptor.implementation_fingerprint()
        {
            return Err(invalid_plan(
                "execution weight plan differs from its trusted materializer descriptor",
            ));
        }
        Ok(())
    }
}
