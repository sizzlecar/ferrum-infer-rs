use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::sync::Arc;

use crate::vnext::{
    CanonicalRational, WeightComponentPayload, WeightComponentSource, WeightComponentSpec,
};

use super::{
    canonical_fingerprint, invalid_plan, is_canonical_sha256, CapabilityCatalog, CapabilityId,
    ContractVersion, Deserialize, DeviceDescriptor, ModelFamilyId, PreparedModelFamily, Serialize,
    VNextError, WeightId, WeightMaterializerId, WeightSchema,
};

pub const IDENTITY_WEIGHT_MATERIALIZER_ID: &str = "weight-materializer.identity";
const IDENTITY_MATERIALIZER_VERSION: ContractVersion = ContractVersion::new(2, 0);
pub const MAX_WEIGHT_MATERIALIZERS: usize = 64;

/// Whether a physical weight transformation preserves source values.
///
/// Kernel availability is not authorization to change model precision.
/// Approximate materializers may be registered for capability discovery, but
/// require a separate numerical-quality approval path before plan selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WeightMaterializationFidelity {
    Exact,
    Approximate,
}

/// Checked-in numerical policy for an approximate materializer.
///
/// This is not an approval record. The public compiler remains exact-only;
/// M3 must add a crate-owned verifier that consumes real numeric artifact
/// bytes before an approximate materializer can be selected.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ApproximateWeightQualityContract {
    execution_contract_fingerprint: String,
    quality_vector_digest: String,
    required_case_count: u32,
    relative_l2_max: CanonicalRational,
    nan_count_max: u64,
    inf_count_max: u64,
}

impl ApproximateWeightQualityContract {
    pub fn new(
        execution_contract_fingerprint: impl Into<String>,
        quality_vector_digest: impl Into<String>,
        required_case_count: u32,
        relative_l2_max: CanonicalRational,
        nan_count_max: u64,
        inf_count_max: u64,
    ) -> Result<Self, VNextError> {
        let contract = Self {
            execution_contract_fingerprint: execution_contract_fingerprint.into(),
            quality_vector_digest: quality_vector_digest.into(),
            required_case_count,
            relative_l2_max,
            nan_count_max,
            inf_count_max,
        };
        contract.validate()?;
        Ok(contract)
    }

    pub fn execution_contract_fingerprint(&self) -> &str {
        &self.execution_contract_fingerprint
    }

    pub fn quality_vector_digest(&self) -> &str {
        &self.quality_vector_digest
    }

    pub const fn required_case_count(&self) -> u32 {
        self.required_case_count
    }

    pub const fn relative_l2_max(&self) -> CanonicalRational {
        self.relative_l2_max
    }

    pub const fn nan_count_max(&self) -> u64 {
        self.nan_count_max
    }

    pub const fn inf_count_max(&self) -> u64 {
        self.inf_count_max
    }

    fn validate(&self) -> Result<(), VNextError> {
        if !is_canonical_sha256(&self.execution_contract_fingerprint)
            || !is_canonical_sha256(&self.quality_vector_digest)
            || self.required_case_count == 0
            || self.relative_l2_max.numerator() <= 0
        {
            return Err(invalid_plan(
                "approximate weight quality contract has invalid digests, case count, or threshold",
            ));
        }
        Ok(())
    }
}

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
            contract: "execution-weight-plan.identity.v2",
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
    fidelity: WeightMaterializationFidelity,
    required_capabilities: BTreeSet<CapabilityId>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    approximate_quality_contract: Option<ApproximateWeightQualityContract>,
}

impl WeightMaterializerDescriptor {
    pub fn new(
        id: WeightMaterializerId,
        version: ContractVersion,
        implementation_fingerprint: impl Into<String>,
        fidelity: WeightMaterializationFidelity,
        required_capabilities: BTreeSet<CapabilityId>,
    ) -> Result<Self, VNextError> {
        let descriptor = Self {
            id,
            version,
            implementation_fingerprint: implementation_fingerprint.into(),
            fidelity,
            required_capabilities,
            approximate_quality_contract: None,
        };
        descriptor.validate_structure()?;
        Ok(descriptor)
    }

    pub(crate) fn identity() -> Result<Self, VNextError> {
        Self::new(
            WeightMaterializerId::new(IDENTITY_WEIGHT_MATERIALIZER_ID)?,
            IDENTITY_MATERIALIZER_VERSION,
            identity_materializer_fingerprint()?,
            WeightMaterializationFidelity::Exact,
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

    pub const fn fidelity(&self) -> WeightMaterializationFidelity {
        self.fidelity
    }

    pub fn required_capabilities(&self) -> &BTreeSet<CapabilityId> {
        &self.required_capabilities
    }

    pub fn with_approximate_quality_contract(
        mut self,
        contract: ApproximateWeightQualityContract,
    ) -> Result<Self, VNextError> {
        if self.fidelity != WeightMaterializationFidelity::Approximate {
            return Err(invalid_plan(format!(
                "exact weight materializer `{}` cannot carry an approximate quality contract",
                self.id
            )));
        }
        contract.validate()?;
        self.approximate_quality_contract = Some(contract);
        Ok(self)
    }

    pub fn approximate_quality_contract(&self) -> Option<&ApproximateWeightQualityContract> {
        self.approximate_quality_contract.as_ref()
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
        if let Some(contract) = &self.approximate_quality_contract {
            contract.validate()?;
            if self.fidelity != WeightMaterializationFidelity::Approximate {
                return Err(invalid_plan(format!(
                    "exact weight materializer `{}` cannot carry an approximate quality contract",
                    self.id
                )));
            }
        }
        Ok(())
    }
}

/// Trusted authority for one physical weight transformation.
///
/// Implementations are invoked once while compiling an immutable plan. They
/// must derive the complete execution schema from the prepared family and
/// device descriptor without allocating device memory or reading request
/// state. During static initialization the same retained implementation turns
/// source components into the exact execution components recorded in that
/// plan; no provider or weight source may bypass this authority.
pub trait WeightMaterializer: Send + Sync {
    fn descriptor(&self) -> &WeightMaterializerDescriptor;

    fn execution_schema(
        &self,
        family: &PreparedModelFamily,
        device: &DeviceDescriptor,
    ) -> Result<WeightSchema, VNextError>;

    /// Ordered source-component identities for every execution component.
    ///
    /// The default only permits a same-id mapping. Materializers that create
    /// derived component identities or combine multiple source components must
    /// declare that provenance explicitly.
    fn component_sources(
        &self,
        family: &PreparedModelFamily,
        execution_schema: &WeightSchema,
    ) -> Result<BTreeMap<WeightId, Vec<WeightId>>, VNextError> {
        identity_component_sources(family, execution_schema)
    }

    /// Materialize one execution component on the cold initialization path.
    ///
    /// `source_components` is resolved from the immutable plan's ordered source
    /// map. Implementations may borrow source bytes for identity layouts or
    /// return owned bytes for repacked/quantized layouts.
    fn materialize_component<'source>(
        &self,
        source: &'source dyn WeightComponentSource,
        source_components: &[&WeightComponentSpec],
        execution_component: &WeightComponentSpec,
    ) -> Result<WeightComponentPayload<'source>, VNextError>;

    /// Materialize execution components that share one exact ordered source set.
    ///
    /// The default preserves existing one-component implementations. Expensive
    /// transforms may override this method to decode, quantize, or repack their
    /// common source once and return one payload per requested component in the
    /// same order.
    fn materialize_components<'source>(
        &self,
        source: &'source dyn WeightComponentSource,
        source_components: &[&WeightComponentSpec],
        execution_components: &[&WeightComponentSpec],
    ) -> Result<Vec<WeightComponentPayload<'source>>, VNextError> {
        execution_components
            .iter()
            .map(|component| self.materialize_component(source, source_components, component))
            .collect()
    }
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

impl WeightMaterializer for IdentityWeightMaterializer {
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

    fn materialize_component<'source>(
        &self,
        source: &'source dyn WeightComponentSource,
        source_components: &[&WeightComponentSpec],
        execution_component: &WeightComponentSpec,
    ) -> Result<WeightComponentPayload<'source>, VNextError> {
        let [source_component] = source_components else {
            return Err(invalid_plan(
                "identity weight materializer requires exactly one source component",
            ));
        };
        if *source_component != execution_component {
            return Err(invalid_plan(format!(
                "identity weight materializer cannot transform component `{}`",
                execution_component.id
            )));
        }
        source.component(source_component)
    }
}

fn identity_component_sources(
    family: &PreparedModelFamily,
    execution_schema: &WeightSchema,
) -> Result<BTreeMap<WeightId, Vec<WeightId>>, VNextError> {
    let source_ids = family
        .weight_schema()
        .components
        .iter()
        .map(|component| component.id.clone())
        .collect::<BTreeSet<_>>();
    execution_schema
        .components
        .iter()
        .map(|component| {
            if !source_ids.contains(&component.id) {
                return Err(invalid_plan(format!(
                    "weight materializer must declare sources for derived component `{}`",
                    component.id
                )));
            }
            Ok((component.id.clone(), vec![component.id.clone()]))
        })
        .collect()
}

/// Process-local registry retaining the exact implementations authorized to
/// transform checkpoint schemas. It is deliberately neither serializable nor
/// reconstructible from a capability catalog.
pub struct WeightMaterializerRegistry {
    materializers: BTreeMap<WeightMaterializerId, Arc<dyn WeightMaterializer>>,
}

impl WeightMaterializerRegistry {
    pub fn new(materializers: Vec<Box<dyn WeightMaterializer>>) -> Result<Self, VNextError> {
        if materializers.len() >= MAX_WEIGHT_MATERIALIZERS {
            return Err(invalid_plan(format!(
                "weight materializer registry exceeds {} non-identity entries",
                MAX_WEIGHT_MATERIALIZERS - 1
            )));
        }
        let identity: Arc<dyn WeightMaterializer> = Arc::new(IdentityWeightMaterializer::new()?);
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
    /// Selects a materializer that is value-preserving by contract.
    ///
    /// Approximate transforms require a future typed quality approval carrying
    /// checked-in numerical-tolerance evidence. Keeping that authority out of
    /// this entrypoint prevents a build feature or backend capability from
    /// silently changing checkpoint precision.
    pub fn select_exact(
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
        if descriptor.fidelity() != WeightMaterializationFidelity::Exact {
            return Err(VNextError::WeightMaterializerQualityApprovalRequired {
                materializer_id: materializer_id.to_string(),
            });
        }
        descriptor.validate_for_device(catalog.device())?;
        let mut schema = materializer.execution_schema(family, catalog.device())?;
        schema.normalize();
        let component_sources = materializer.component_sources(family, &schema)?;
        let plan =
            ExecutionWeightPlan::from_materializer(family, descriptor, schema, component_sources)?;
        Ok(TrustedExecutionWeightPlan {
            plan,
            descriptor: descriptor.clone(),
            materializer: Arc::clone(materializer),
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
#[derive(Clone)]
pub struct TrustedExecutionWeightPlan {
    plan: ExecutionWeightPlan,
    descriptor: WeightMaterializerDescriptor,
    materializer: Arc<dyn WeightMaterializer>,
}

impl fmt::Debug for TrustedExecutionWeightPlan {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("TrustedExecutionWeightPlan")
            .field("plan", &self.plan)
            .field("descriptor", &self.descriptor)
            .finish_non_exhaustive()
    }
}

impl PartialEq for TrustedExecutionWeightPlan {
    fn eq(&self, other: &Self) -> bool {
        self.plan == other.plan && self.descriptor == other.descriptor
    }
}

impl Eq for TrustedExecutionWeightPlan {}

impl TrustedExecutionWeightPlan {
    pub(crate) fn identity(family: &PreparedModelFamily) -> Result<Self, VNextError> {
        let materializer: Arc<dyn WeightMaterializer> =
            Arc::new(IdentityWeightMaterializer::new()?);
        let descriptor = materializer.descriptor().clone();
        let schema = family.weight_schema().clone();
        let component_sources = materializer.component_sources(family, &schema)?;
        Ok(Self {
            plan: ExecutionWeightPlan::from_materializer(
                family,
                &descriptor,
                schema,
                component_sources,
            )?,
            descriptor,
            materializer,
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
        self.validate_runtime_authority()?;
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

    pub(crate) fn materialize_components<'source>(
        &self,
        family: &PreparedModelFamily,
        source: &'source dyn WeightComponentSource,
        execution_components: &[&WeightComponentSpec],
    ) -> Result<Vec<WeightComponentPayload<'source>>, VNextError> {
        self.validate_runtime_authority()?;
        let Some(first_execution_component) = execution_components.first() else {
            return Err(invalid_plan(
                "weight materializer received an empty execution component group",
            ));
        };
        let mut planned_components = Vec::with_capacity(execution_components.len());
        for execution_component in execution_components {
            let planned_component_index = self
                .plan
                .schema
                .components
                .binary_search_by(|component| component.id.cmp(&execution_component.id))
                .map_err(|_| {
                    invalid_plan(format!(
                        "execution component `{}` is absent from the trusted weight plan",
                        execution_component.id
                    ))
                })?;
            let planned_component = &self.plan.schema.components[planned_component_index];
            if planned_component != *execution_component {
                return Err(invalid_plan(format!(
                    "execution component `{}` differs from the trusted weight plan",
                    execution_component.id
                )));
            }
            planned_components.push(planned_component);
        }
        let source_ids = self
            .plan
            .component_sources
            .get(&first_execution_component.id)
            .ok_or_else(|| {
                invalid_plan(format!(
                    "execution component `{}` has no source mapping",
                    first_execution_component.id
                ))
            })?;
        if execution_components
            .iter()
            .skip(1)
            .any(|component| self.plan.component_sources.get(&component.id) != Some(source_ids))
        {
            return Err(invalid_plan(
                "grouped execution components do not share one ordered source mapping",
            ));
        }
        let source_components = source_ids
            .iter()
            .map(|source_id| {
                let source_component_index = family
                    .weight_schema()
                    .components
                    .binary_search_by(|component| component.id.cmp(source_id))
                    .map_err(|_| {
                        invalid_plan(format!(
                            "execution component `{}` references unknown source component `{source_id}`",
                            first_execution_component.id
                        ))
                    })?;
                Ok(&family.weight_schema().components[source_component_index])
            })
            .collect::<Result<Vec<_>, _>>()?;
        let payloads = self.materializer.materialize_components(
            source,
            &source_components,
            &planned_components,
        )?;
        if payloads.len() != planned_components.len() {
            return Err(invalid_plan(format!(
                "weight materializer `{}` returned {} payloads for {} execution components",
                self.descriptor.id,
                payloads.len(),
                planned_components.len()
            )));
        }
        for (payload, execution_component) in payloads.iter().zip(&planned_components) {
            if payload.component_id() != &execution_component.id
                || payload.external_names() != execution_component.external_names.as_slice()
                || payload.dimensions() != execution_component.dimensions.as_slice()
                || payload.element_type() != execution_component.physical_element_type()
                || u64::try_from(payload.bytes().len()).ok()
                    != Some(execution_component.physical_bytes()?)
            {
                return Err(invalid_plan(format!(
                    "weight materializer `{}` returned invalid or reordered payload for execution component `{}`",
                    self.descriptor.id, execution_component.id
                )));
            }
        }
        Ok(payloads)
    }

    fn validate_runtime_authority(&self) -> Result<(), VNextError> {
        if self.materializer.descriptor() != &self.descriptor {
            return Err(invalid_plan(format!(
                "weight materializer `{}` runtime authority differs from its trusted descriptor",
                self.descriptor.id
            )));
        }
        Ok(())
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
    component_sources: BTreeMap<WeightId, Vec<WeightId>>,
    schema: WeightSchema,
}

impl ExecutionWeightPlan {
    pub fn identity(family: &PreparedModelFamily) -> Result<Self, VNextError> {
        let descriptor = WeightMaterializerDescriptor::identity()?;
        let schema = family.weight_schema().clone();
        let component_sources = identity_component_sources(family, &schema)?;
        Self::from_materializer(family, &descriptor, schema, component_sources)
    }

    fn from_materializer(
        family: &PreparedModelFamily,
        descriptor: &WeightMaterializerDescriptor,
        schema: WeightSchema,
        component_sources: BTreeMap<WeightId, Vec<WeightId>>,
    ) -> Result<Self, VNextError> {
        let plan = Self {
            source_schema_fingerprint: family.weight_schema().fingerprint()?,
            materializer_id: descriptor.id().clone(),
            materializer_version: descriptor.version(),
            materializer_implementation_fingerprint: descriptor
                .implementation_fingerprint()
                .to_owned(),
            component_sources,
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

    pub fn component_sources(&self) -> &BTreeMap<WeightId, Vec<WeightId>> {
        &self.component_sources
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
        self.schema.validate(family_id)?;
        let execution_component_ids = self
            .schema
            .components
            .iter()
            .map(|component| component.id.clone())
            .collect::<BTreeSet<_>>();
        let mapped_component_ids = self
            .component_sources
            .keys()
            .cloned()
            .collect::<BTreeSet<_>>();
        if execution_component_ids != mapped_component_ids
            || self.component_sources.values().any(|source_ids| {
                source_ids.is_empty()
                    || source_ids.iter().collect::<BTreeSet<_>>().len() != source_ids.len()
            })
        {
            return Err(invalid_plan(
                "execution weight component source map is incomplete or contains duplicate sources",
            ));
        }
        Ok(())
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
        let source_components = family
            .weight_schema()
            .components
            .iter()
            .map(|component| (&component.id, component))
            .collect::<BTreeMap<_, _>>();
        let mut referenced_source_components = BTreeSet::new();
        for (execution_component_id, source_ids) in &self.component_sources {
            for source_id in source_ids {
                if !source_components.contains_key(source_id) {
                    return Err(invalid_plan(format!(
                        "execution component `{execution_component_id}` references unknown source component `{source_id}`"
                    )));
                }
                referenced_source_components.insert(source_id.clone());
            }
        }
        if let Some(component) = source_components.values().find(|component| {
            component.required && !referenced_source_components.contains(&component.id)
        }) {
            return Err(invalid_plan(format!(
                "required source component `{}` is not represented in the execution weight plan",
                component.id
            )));
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
