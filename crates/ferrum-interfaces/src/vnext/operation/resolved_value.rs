use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Deserializer, Serialize};

use super::super::{
    BufferUsage, ProgramValueId, ResolvedWeightBinding, ResourceId, VNextError, WeightId,
};
use super::foundation::invalid_operation;
use super::{
    AliasPolicy, DynamicStorageRequirement, ElementType, ResolvedTensorSpec, TensorAccess,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResolvedValueRole {
    Input,
    Output,
}

/// Provider-accepted physical profiles for one exact operation binding slot.
/// Role and ordinal are contract identities, not model-specific names.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ProviderStorageBindingRequirement {
    role: ResolvedValueRole,
    ordinal: u32,
    storage: DynamicStorageRequirement,
}

impl ProviderStorageBindingRequirement {
    pub fn new(role: ResolvedValueRole, ordinal: u32, storage: DynamicStorageRequirement) -> Self {
        Self {
            role,
            ordinal,
            storage,
        }
    }

    pub const fn role(&self) -> ResolvedValueRole {
        self.role
    }

    pub const fn ordinal(&self) -> u32 {
        self.ordinal
    }

    pub fn storage(&self) -> &DynamicStorageRequirement {
        &self.storage
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ProviderStorageBindingRequirementWire {
    role: ResolvedValueRole,
    ordinal: u32,
    storage: DynamicStorageRequirement,
}

impl<'de> Deserialize<'de> for ProviderStorageBindingRequirement {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = ProviderStorageBindingRequirementWire::deserialize(deserializer)?;
        Ok(Self::new(wire.role, wire.ordinal, wire.storage))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ResolvedStorageComponent {
    component_id: Option<WeightId>,
    resource_id: ResourceId,
    offset_bytes: u64,
    length_bytes: u64,
    element_type: ElementType,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ResolvedStorageComponentWire {
    component_id: Option<WeightId>,
    resource_id: ResourceId,
    offset_bytes: u64,
    length_bytes: u64,
    element_type: ElementType,
}

impl<'de> Deserialize<'de> for ResolvedStorageComponent {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = ResolvedStorageComponentWire::deserialize(deserializer)?;
        Self::new(
            wire.component_id,
            wire.resource_id,
            wire.offset_bytes,
            wire.length_bytes,
            wire.element_type,
        )
        .map_err(serde::de::Error::custom)
    }
}

impl ResolvedStorageComponent {
    pub fn new(
        component_id: Option<WeightId>,
        resource_id: ResourceId,
        offset_bytes: u64,
        length_bytes: u64,
        element_type: ElementType,
    ) -> Result<Self, VNextError> {
        if length_bytes == 0
            || offset_bytes.checked_add(length_bytes).is_none()
            || offset_bytes % element_type.size_bytes() != 0
            || length_bytes % element_type.size_bytes() != 0
        {
            return Err(invalid_operation(
                "resolved storage component is empty or overflows u64",
            ));
        }
        Ok(Self {
            component_id,
            resource_id,
            offset_bytes,
            length_bytes,
            element_type,
        })
    }

    pub fn component_id(&self) -> Option<&WeightId> {
        self.component_id.as_ref()
    }

    pub fn resource_id(&self) -> &ResourceId {
        &self.resource_id
    }

    pub const fn offset_bytes(&self) -> u64 {
        self.offset_bytes
    }

    pub const fn length_bytes(&self) -> u64 {
        self.length_bytes
    }

    pub const fn element_type(&self) -> ElementType {
        self.element_type
    }
}

/// Physical resources backing one semantic value. A logical quantized weight
/// can bind packed values, scales, zero-points, and indices without pretending
/// they are one dense allocation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ResolvedValueStorage {
    components: Vec<ResolvedStorageComponent>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ResolvedValueStorageWire {
    components: Vec<ResolvedStorageComponent>,
}

impl<'de> Deserialize<'de> for ResolvedValueStorage {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = ResolvedValueStorageWire::deserialize(deserializer)?;
        Self::new(wire.components).map_err(serde::de::Error::custom)
    }
}

impl ResolvedValueStorage {
    pub fn single(
        resource_id: ResourceId,
        offset_bytes: u64,
        length_bytes: u64,
        element_type: ElementType,
    ) -> Result<Self, VNextError> {
        Self::new(vec![ResolvedStorageComponent::new(
            None,
            resource_id,
            offset_bytes,
            length_bytes,
            element_type,
        )?])
    }

    pub fn composite(components: Vec<ResolvedStorageComponent>) -> Result<Self, VNextError> {
        if components
            .iter()
            .any(|component| component.component_id.is_none())
        {
            return Err(invalid_operation(
                "composite value storage requires a physical component identity",
            ));
        }
        Self::new(components)
    }

    fn new(mut components: Vec<ResolvedStorageComponent>) -> Result<Self, VNextError> {
        if components.is_empty() {
            return Err(invalid_operation("resolved value storage is empty"));
        }
        if components.len() > 1
            && components
                .iter()
                .any(|component| component.component_id.is_none())
        {
            return Err(invalid_operation(
                "multi-component value storage requires physical component identities",
            ));
        }
        components.sort_by(|left, right| {
            left.component_id
                .cmp(&right.component_id)
                .then(left.resource_id.cmp(&right.resource_id))
                .then(left.offset_bytes.cmp(&right.offset_bytes))
        });
        let mut component_ids = BTreeSet::new();
        for (index, component) in components.iter().enumerate() {
            if component.length_bytes == 0
                || component
                    .offset_bytes
                    .checked_add(component.length_bytes)
                    .is_none()
                || component
                    .component_id
                    .as_ref()
                    .is_some_and(|component_id| !component_ids.insert(component_id.clone()))
            {
                return Err(invalid_operation(
                    "resolved value storage has invalid or duplicate components",
                ));
            }
            if components[..index].iter().any(|previous| {
                previous.resource_id == component.resource_id
                    && previous.offset_bytes
                        < component
                            .offset_bytes
                            .saturating_add(component.length_bytes)
                    && component.offset_bytes
                        < previous.offset_bytes.saturating_add(previous.length_bytes)
            }) {
                return Err(invalid_operation(
                    "resolved value storage components overlap in one resource",
                ));
            }
        }
        Ok(Self { components })
    }

    pub fn components(&self) -> &[ResolvedStorageComponent] {
        &self.components
    }

    pub fn resource_ids(&self) -> BTreeSet<&ResourceId> {
        self.components
            .iter()
            .map(|component| &component.resource_id)
            .collect()
    }

    pub fn total_physical_bytes(&self) -> Result<u64, VNextError> {
        self.components.iter().try_fold(0_u64, |total, component| {
            total
                .checked_add(component.length_bytes)
                .ok_or_else(|| invalid_operation("resolved storage byte count overflows u64"))
        })
    }
}

/// Value/resource binding shared by the execution plan and provider
/// invocation. Keeping one representation prevents a lossy translation at the
/// runtime boundary.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ResolvedValueBinding {
    value_id: ProgramValueId,
    role: ResolvedValueRole,
    ordinal: u32,
    tensor: ResolvedTensorSpec,
    access: TensorAccess,
    alias: AliasPolicy,
    usage: BufferUsage,
    weight: Option<ResolvedWeightBinding>,
    storage: ResolvedValueStorage,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ResolvedValueBindingWire {
    value_id: ProgramValueId,
    role: ResolvedValueRole,
    ordinal: u32,
    tensor: ResolvedTensorSpec,
    access: TensorAccess,
    alias: AliasPolicy,
    usage: BufferUsage,
    weight: Option<ResolvedWeightBinding>,
    storage: ResolvedValueStorage,
}

impl<'de> Deserialize<'de> for ResolvedValueBinding {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = ResolvedValueBindingWire::deserialize(deserializer)?;
        Self::new(
            wire.value_id,
            wire.role,
            wire.ordinal,
            wire.tensor,
            wire.access,
            wire.alias,
            wire.usage,
            wire.weight,
            wire.storage,
        )
        .map_err(serde::de::Error::custom)
    }
}

impl ResolvedValueBinding {
    pub fn new(
        value_id: ProgramValueId,
        role: ResolvedValueRole,
        ordinal: u32,
        tensor: ResolvedTensorSpec,
        access: TensorAccess,
        alias: AliasPolicy,
        usage: BufferUsage,
        weight: Option<ResolvedWeightBinding>,
        storage: ResolvedValueStorage,
    ) -> Result<Self, VNextError> {
        if (role == ResolvedValueRole::Input
            && !matches!(access, TensorAccess::Read | TensorAccess::ReadWrite))
            || (role == ResolvedValueRole::Output
                && !matches!(access, TensorAccess::Write | TensorAccess::ReadWrite))
            || (role == ResolvedValueRole::Input && !matches!(alias, AliasPolicy::NoAlias))
        {
            return Err(invalid_operation(
                "resolved value role, access, and alias policy are inconsistent",
            ));
        }
        if usage != BufferUsage::Weights && storage.components.len() != 1 {
            return Err(invalid_operation(
                "only a weight value may use composite physical storage",
            ));
        }
        if storage.components.len() == 1
            && storage.components[0].component_id.is_none()
            && storage.components[0].element_type != tensor.element_type()
        {
            return Err(invalid_operation(
                "single-resource value dtype differs from its logical tensor dtype",
            ));
        }
        if usage != BufferUsage::Weights
            && storage.components[0].length_bytes < tensor.minimum_storage_bytes()?
        {
            return Err(invalid_operation(
                "resolved value storage is smaller than its tensor span",
            ));
        }
        match (usage, weight.as_ref()) {
            (BufferUsage::Weights, Some(weight)) => {
                weight.validate_logical(tensor.dimensions(), tensor.element_type())?;
                validate_resolved_weight_storage(weight, &storage)?;
            }
            (BufferUsage::Weights, None) => {
                return Err(invalid_operation(
                    "weight value lacks its resolved physical layout contract",
                ));
            }
            (_, Some(_)) => {
                return Err(invalid_operation(
                    "non-weight value carries a resolved weight layout contract",
                ));
            }
            (_, None) => {}
        }
        Ok(Self {
            value_id,
            role,
            ordinal,
            tensor,
            access,
            alias,
            usage,
            weight,
            storage,
        })
    }

    pub fn value_id(&self) -> &ProgramValueId {
        &self.value_id
    }

    pub fn role(&self) -> ResolvedValueRole {
        self.role
    }

    pub fn ordinal(&self) -> u32 {
        self.ordinal
    }

    pub fn tensor(&self) -> &ResolvedTensorSpec {
        &self.tensor
    }

    pub fn access(&self) -> TensorAccess {
        self.access
    }

    pub fn alias(&self) -> &AliasPolicy {
        &self.alias
    }

    pub const fn usage(&self) -> BufferUsage {
        self.usage
    }

    pub fn weight(&self) -> Option<&ResolvedWeightBinding> {
        self.weight.as_ref()
    }

    pub fn storage(&self) -> &ResolvedValueStorage {
        &self.storage
    }
}

fn validate_resolved_weight_storage(
    weight: &ResolvedWeightBinding,
    storage: &ResolvedValueStorage,
) -> Result<(), VNextError> {
    let expected = weight
        .components()
        .iter()
        .map(|component| (component.component_id(), component))
        .collect::<BTreeMap<_, _>>();
    if storage.components().len() != expected.len() {
        return Err(invalid_operation(
            "resolved weight storage component count differs from its layout contract",
        ));
    }
    let mut seen = BTreeSet::new();
    for stored in storage.components() {
        let component_id = stored.component_id().ok_or_else(|| {
            invalid_operation("resolved weight storage component lacks its physical identity")
        })?;
        let component = expected.get(component_id).ok_or_else(|| {
            invalid_operation(format!(
                "resolved weight storage contains unknown component `{component_id}`"
            ))
        })?;
        if !seen.insert(component_id)
            || stored.length_bytes() != component.physical_bytes()?
            || stored.element_type() != component.physical_element_type()
        {
            return Err(invalid_operation(format!(
                "resolved weight storage component `{component_id}` differs from its layout contract"
            )));
        }
    }
    Ok(())
}
