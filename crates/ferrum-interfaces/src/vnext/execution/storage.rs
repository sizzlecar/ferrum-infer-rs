use super::{
    canonical_fingerprint, invalid_plan, is_canonical_sha256, quantize_storage_bytes,
    validate_active_sequence_ceiling, AllocationKind, AllocationLifetime, BTreeSet,
    BlockedTensorPadding, BufferUsage, ContractVersion, Deserialize, Deserializer,
    DynamicResourceDemand, DynamicResourceShape, DynamicStorageProfile, ElementType, NodeId,
    ResolvedTensorLayout, ResourceId, ResourceWorkShape, Serialize, StateInitialization,
    VNextError,
};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(transparent)]
pub struct DynamicBackingPoolId(String);

impl DynamicBackingPoolId {
    pub(super) fn from_compatibility(key: &PoolCompatibilityKey) -> Result<Self, VNextError> {
        Ok(Self(format!(
            "dynamic-pool/sha256/{}",
            canonical_fingerprint(key, "fingerprint dynamic pool compatibility")?
        )))
    }

    pub(super) fn validate(&self) -> Result<(), VNextError> {
        let Some(hash) = self.0.strip_prefix("dynamic-pool/sha256/") else {
            return Err(invalid_plan(
                "dynamic backing pool id has an invalid prefix",
            ));
        };
        if !is_canonical_sha256(hash) {
            return Err(invalid_plan("dynamic backing pool id has an invalid hash"));
        }
        Ok(())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl<'de> Deserialize<'de> for DynamicBackingPoolId {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let id = Self(String::deserialize(deserializer)?);
        id.validate().map_err(serde::de::Error::custom)?;
        Ok(id)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DynamicStorageContract {
    pub(super) profile: DynamicStorageProfile,
    pub(super) logical_layout_fingerprint: String,
}

impl DynamicStorageContract {
    pub(super) fn new(
        profile: DynamicStorageProfile,
        logical_layout_fingerprint: String,
    ) -> Result<Self, VNextError> {
        if !is_canonical_sha256(&logical_layout_fingerprint) {
            return Err(invalid_plan(
                "dynamic storage logical layout fingerprint is invalid",
            ));
        }
        Ok(Self {
            profile,
            logical_layout_fingerprint,
        })
    }

    pub const fn profile(&self) -> DynamicStorageProfile {
        self.profile
    }

    pub fn logical_layout_fingerprint(&self) -> &str {
        &self.logical_layout_fingerprint
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct DynamicStorageContractWire {
    pub(super) profile: DynamicStorageProfile,
    pub(super) logical_layout_fingerprint: String,
}

impl<'de> Deserialize<'de> for DynamicStorageContract {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = DynamicStorageContractWire::deserialize(deserializer)?;
        Self::new(wire.profile, wire.logical_layout_fingerprint).map_err(serde::de::Error::custom)
    }
}

#[derive(Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum TensorStorageLayoutClass<'a> {
    Contiguous,
    Strided {
        byte_strides: &'a [u64],
    },
    Blocked {
        block: &'a [u64],
        axis_order: &'a [u32],
        padding: BlockedStoragePaddingClass,
    },
}

#[derive(Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum BlockedStoragePaddingClass {
    Exact,
    ZeroFill,
}

#[derive(Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum WorkspaceStorageLayoutClass {
    OpaqueBytesV1,
}

pub(super) fn tensor_storage_layout_fingerprint(
    layout: &ResolvedTensorLayout,
) -> Result<String, VNextError> {
    let class = match layout {
        ResolvedTensorLayout::Contiguous => TensorStorageLayoutClass::Contiguous,
        ResolvedTensorLayout::Strided { byte_strides } => {
            TensorStorageLayoutClass::Strided { byte_strides }
        }
        ResolvedTensorLayout::Blocked {
            block,
            axis_order,
            padding,
        } => TensorStorageLayoutClass::Blocked {
            block,
            axis_order,
            padding: match padding {
                BlockedTensorPadding::Exact => BlockedStoragePaddingClass::Exact,
                BlockedTensorPadding::ZeroFill { .. } => BlockedStoragePaddingClass::ZeroFill,
            },
        },
    };
    canonical_fingerprint(&class, "fingerprint tensor storage layout class")
}

pub(super) fn workspace_storage_layout_fingerprint() -> Result<String, VNextError> {
    canonical_fingerprint(
        &WorkspaceStorageLayoutClass::OpaqueBytesV1,
        "fingerprint workspace storage layout class",
    )
}

pub(super) fn static_contiguous_storage_profile() -> Result<DynamicStorageProfile, VNextError> {
    DynamicStorageProfile::new(
        super::DynamicStorageAllocator::LinearArena,
        super::DynamicStorageView::Contiguous,
    )
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct PoolCompatibilityKey {
    pub(super) version: ContractVersion,
    pub(super) profile: DynamicStorageProfile,
    pub(super) usage: BufferUsage,
    pub(super) element_type: ElementType,
    pub(super) logical_layout_fingerprint: String,
    pub(super) alignment_bytes: u64,
}

impl PoolCompatibilityKey {
    pub(super) fn new(
        storage: &DynamicStorageContract,
        usage: BufferUsage,
        element_type: ElementType,
        alignment_bytes: u64,
    ) -> Result<Self, VNextError> {
        if alignment_bytes == 0 || !alignment_bytes.is_power_of_two() {
            return Err(invalid_plan(
                "dynamic pool compatibility alignment is invalid",
            ));
        }
        let key = Self {
            version: ContractVersion::new(1, 0),
            profile: storage.profile,
            usage,
            element_type,
            logical_layout_fingerprint: storage.logical_layout_fingerprint.clone(),
            alignment_bytes,
        };
        key.validate()?;
        Ok(key)
    }

    pub(super) fn validate(&self) -> Result<(), VNextError> {
        if self.version != ContractVersion::new(1, 0)
            || !is_canonical_sha256(&self.logical_layout_fingerprint)
            || self.alignment_bytes == 0
            || !self.alignment_bytes.is_power_of_two()
        {
            return Err(invalid_plan("dynamic pool compatibility key is invalid"));
        }
        Ok(())
    }

    pub const fn profile(&self) -> DynamicStorageProfile {
        self.profile
    }

    pub const fn usage(&self) -> BufferUsage {
        self.usage
    }

    pub const fn element_type(&self) -> ElementType {
        self.element_type
    }

    pub fn logical_layout_fingerprint(&self) -> &str {
        &self.logical_layout_fingerprint
    }

    pub const fn alignment_bytes(&self) -> u64 {
        self.alignment_bytes
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct PoolCompatibilityKeyWire {
    pub(super) version: ContractVersion,
    pub(super) profile: DynamicStorageProfile,
    pub(super) usage: BufferUsage,
    pub(super) element_type: ElementType,
    pub(super) logical_layout_fingerprint: String,
    pub(super) alignment_bytes: u64,
}

impl<'de> Deserialize<'de> for PoolCompatibilityKey {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = PoolCompatibilityKeyWire::deserialize(deserializer)?;
        let key = Self {
            version: wire.version,
            profile: wire.profile,
            usage: wire.usage,
            element_type: wire.element_type,
            logical_layout_fingerprint: wire.logical_layout_fingerprint,
            alignment_bytes: wire.alignment_bytes,
        };
        key.validate().map_err(serde::de::Error::custom)?;
        Ok(key)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DynamicPoolProvisioningMode {
    DemandDrivenElastic,
}

/// Typed bounds for elastic residency. `minimum_resident_bytes` is the amount
/// required to make one request runnable, not an initial reservation. Pools
/// may grow on demand up to `maximum_resident_bytes`; the process-wide device
/// account remains the authority when several pools compete for that memory.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DynamicPoolProvisioningPolicy {
    pub(super) mode: DynamicPoolProvisioningMode,
    pub(super) minimum_resident_bytes: u64,
    pub(super) maximum_resident_bytes: u64,
}

impl DynamicPoolProvisioningPolicy {
    pub(super) fn demand_driven(
        minimum_resident_bytes: u64,
        maximum_resident_bytes: u64,
    ) -> Result<Self, VNextError> {
        let policy = Self {
            mode: DynamicPoolProvisioningMode::DemandDrivenElastic,
            minimum_resident_bytes,
            maximum_resident_bytes,
        };
        policy.validate()?;
        Ok(policy)
    }

    pub(super) fn validate(&self) -> Result<(), VNextError> {
        if self.minimum_resident_bytes == 0
            || self.maximum_resident_bytes < self.minimum_resident_bytes
        {
            return Err(invalid_plan("dynamic pool provisioning bounds are invalid"));
        }
        Ok(())
    }

    pub const fn mode(&self) -> DynamicPoolProvisioningMode {
        self.mode
    }

    pub const fn minimum_resident_bytes(&self) -> u64 {
        self.minimum_resident_bytes
    }

    pub const fn maximum_resident_bytes(&self) -> u64 {
        self.maximum_resident_bytes
    }
}

/// One self-contained physical-compatibility class for demand-driven backing.
/// Membership, runnable minima, completion-order reuse evidence, and elastic
/// bounds are canonical plan data, so a runtime does not have to rediscover
/// pool structure by scanning unrelated descriptors.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DynamicBackingPoolSpec {
    pub(super) pool_id: DynamicBackingPoolId,
    pub(super) compatibility: PoolCompatibilityKey,
    pub(super) resource_ids: Vec<ResourceId>,
    pub(super) minimum_request_bytes: u64,
    pub(super) minimum_sequence_bytes: u64,
    pub(super) minimum_step_bytes: u64,
    pub(super) minimum_invocation_peak_bytes: u64,
    pub(super) step_resource_slots: Vec<StepResourceSlot>,
    pub(super) theoretical_ceiling_bytes: CanonicalU128,
    pub(super) reusable_workspace_ceiling_bytes: u64,
    pub(super) provisioning: DynamicPoolProvisioningPolicy,
    pub(super) invocation_liveness_mode: InvocationLivenessMode,
    pub(super) invocation_liveness: Vec<InvocationResourceLiveness>,
}

impl DynamicBackingPoolSpec {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn from_core(
        compatibility: PoolCompatibilityKey,
        resource_ids: Vec<ResourceId>,
        minimum_request_bytes: u64,
        minimum_sequence_bytes: u64,
        minimum_step_bytes: u64,
        minimum_invocation_peak_bytes: u64,
        step_resource_slots: Vec<StepResourceSlot>,
        theoretical_ceiling_bytes: u128,
        reusable_workspace_ceiling_bytes: u64,
        dynamic_capacity_bytes: u64,
        invocation_liveness_mode: InvocationLivenessMode,
        invocation_liveness: Vec<InvocationResourceLiveness>,
    ) -> Result<Self, VNextError> {
        compatibility.validate()?;
        let pool_id = DynamicBackingPoolId::from_compatibility(&compatibility)?;
        let minimum_resident_bytes = minimum_request_bytes
            .checked_add(minimum_sequence_bytes)
            .and_then(|bytes| bytes.checked_add(minimum_step_bytes))
            .and_then(|bytes| bytes.checked_add(minimum_invocation_peak_bytes))
            .ok_or_else(|| invalid_plan("dynamic pool runnable minimum overflows u64"))?;
        let combined_ceiling_bytes = theoretical_ceiling_bytes
            .checked_add(u128::from(reusable_workspace_ceiling_bytes))
            .ok_or_else(|| invalid_plan("dynamic pool combined ceiling overflows u128"))?;
        let maximum_resident_bytes =
            u64::try_from(combined_ceiling_bytes.min(u128::from(dynamic_capacity_bytes)))
                .map_err(|_| invalid_plan("dynamic pool resident ceiling exceeds u64"))?;
        let spec = Self {
            pool_id,
            compatibility,
            resource_ids,
            minimum_request_bytes,
            minimum_sequence_bytes,
            minimum_step_bytes,
            minimum_invocation_peak_bytes,
            step_resource_slots,
            theoretical_ceiling_bytes: CanonicalU128::new(theoretical_ceiling_bytes),
            reusable_workspace_ceiling_bytes,
            provisioning: DynamicPoolProvisioningPolicy::demand_driven(
                minimum_resident_bytes,
                maximum_resident_bytes,
            )?,
            invocation_liveness_mode,
            invocation_liveness,
        };
        spec.validate_local()?;
        Ok(spec)
    }

    pub(super) fn validate_local(&self) -> Result<(), VNextError> {
        self.pool_id.validate()?;
        self.compatibility.validate()?;
        self.provisioning.validate()?;
        for slot in &self.step_resource_slots {
            slot.validate()?;
        }
        let minimum_resident_bytes = self
            .minimum_request_bytes
            .checked_add(self.minimum_sequence_bytes)
            .and_then(|bytes| bytes.checked_add(self.minimum_step_bytes))
            .and_then(|bytes| bytes.checked_add(self.minimum_invocation_peak_bytes))
            .ok_or_else(|| invalid_plan("dynamic pool runnable minimum overflows u64"))?;
        if self.pool_id != DynamicBackingPoolId::from_compatibility(&self.compatibility)?
            || self.resource_ids.is_empty()
            || self.resource_ids.windows(2).any(|pair| pair[0] >= pair[1])
            || minimum_resident_bytes != self.provisioning.minimum_resident_bytes
            || u128::from(self.provisioning.maximum_resident_bytes)
                > self
                    .theoretical_ceiling_bytes
                    .get()
                    .checked_add(u128::from(self.reusable_workspace_ceiling_bytes))
                    .ok_or_else(|| invalid_plan("dynamic pool combined ceiling overflows u128"))?
            || self
                .step_resource_slots
                .windows(2)
                .any(|pair| pair[0].resource_ids >= pair[1].resource_ids)
            || self
                .step_resource_slots
                .iter()
                .flat_map(|slot| slot.resource_ids.iter())
                .collect::<BTreeSet<_>>()
                .len()
                != self
                    .step_resource_slots
                    .iter()
                    .map(|slot| slot.resource_ids.len())
                    .sum::<usize>()
        {
            return Err(invalid_plan(
                "dynamic backing pool identity, membership, or bounds are invalid",
            ));
        }
        match self.invocation_liveness_mode {
            InvocationLivenessMode::NoInvocationResources => {
                if self.minimum_invocation_peak_bytes != 0 || !self.invocation_liveness.is_empty() {
                    return Err(invalid_plan(
                        "non-invocation pool carries invocation liveness evidence",
                    ));
                }
            }
            InvocationLivenessMode::TotalOrderReuse
            | InvocationLivenessMode::ConservativeConcurrent => {
                if self.minimum_invocation_peak_bytes == 0
                    || self.invocation_liveness.is_empty()
                    || self
                        .invocation_liveness
                        .windows(2)
                        .any(|pair| pair[0].node_id >= pair[1].node_id)
                {
                    return Err(invalid_plan(
                        "invocation pool liveness evidence is empty or non-canonical",
                    ));
                }
            }
        }
        Ok(())
    }

    pub fn pool_id(&self) -> &DynamicBackingPoolId {
        &self.pool_id
    }

    pub fn compatibility(&self) -> &PoolCompatibilityKey {
        &self.compatibility
    }

    pub fn resource_ids(&self) -> &[ResourceId] {
        &self.resource_ids
    }

    pub const fn minimum_request_bytes(&self) -> u64 {
        self.minimum_request_bytes
    }

    pub const fn minimum_sequence_bytes(&self) -> u64 {
        self.minimum_sequence_bytes
    }

    pub const fn minimum_step_bytes(&self) -> u64 {
        self.minimum_step_bytes
    }

    pub const fn minimum_invocation_peak_bytes(&self) -> u64 {
        self.minimum_invocation_peak_bytes
    }

    pub fn step_resource_slots(&self) -> &[StepResourceSlot] {
        &self.step_resource_slots
    }

    pub fn theoretical_ceiling_bytes(&self) -> u128 {
        self.theoretical_ceiling_bytes.get()
    }

    pub const fn reusable_workspace_ceiling_bytes(&self) -> u64 {
        self.reusable_workspace_ceiling_bytes
    }

    pub fn provisioning(&self) -> &DynamicPoolProvisioningPolicy {
        &self.provisioning
    }

    pub const fn invocation_liveness_mode(&self) -> InvocationLivenessMode {
        self.invocation_liveness_mode
    }

    pub fn invocation_liveness(&self) -> &[InvocationResourceLiveness] {
        &self.invocation_liveness
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct DynamicBackingPoolSpecWire {
    pub(super) pool_id: DynamicBackingPoolId,
    pub(super) compatibility: PoolCompatibilityKey,
    pub(super) resource_ids: Vec<ResourceId>,
    pub(super) minimum_request_bytes: u64,
    pub(super) minimum_sequence_bytes: u64,
    pub(super) minimum_step_bytes: u64,
    pub(super) minimum_invocation_peak_bytes: u64,
    pub(super) step_resource_slots: Vec<StepResourceSlot>,
    pub(super) theoretical_ceiling_bytes: CanonicalU128,
    pub(super) reusable_workspace_ceiling_bytes: u64,
    pub(super) provisioning: DynamicPoolProvisioningPolicy,
    pub(super) invocation_liveness_mode: InvocationLivenessMode,
    pub(super) invocation_liveness: Vec<InvocationResourceLiveness>,
}

impl<'de> Deserialize<'de> for DynamicBackingPoolSpec {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = DynamicBackingPoolSpecWire::deserialize(deserializer)?;
        let spec = Self {
            pool_id: wire.pool_id,
            compatibility: wire.compatibility,
            resource_ids: wire.resource_ids,
            minimum_request_bytes: wire.minimum_request_bytes,
            minimum_sequence_bytes: wire.minimum_sequence_bytes,
            minimum_step_bytes: wire.minimum_step_bytes,
            minimum_invocation_peak_bytes: wire.minimum_invocation_peak_bytes,
            step_resource_slots: wire.step_resource_slots,
            theoretical_ceiling_bytes: wire.theoretical_ceiling_bytes,
            reusable_workspace_ceiling_bytes: wire.reusable_workspace_ceiling_bytes,
            provisioning: wire.provisioning,
            invocation_liveness_mode: wire.invocation_liveness_mode,
            invocation_liveness: wire.invocation_liveness,
        };
        spec.validate_local().map_err(serde::de::Error::custom)?;
        Ok(spec)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DynamicResourceDescriptor {
    pub(super) base_resource_id: ResourceId,
    pub(super) demand: DynamicResourceDemand,
    pub(super) alignment_bytes: u64,
    pub(super) usage: BufferUsage,
    pub(super) element_type: ElementType,
    pub(super) lifetime: AllocationLifetime,
    pub(super) kind: AllocationKind,
    pub(super) storage: DynamicStorageContract,
    pub(super) pool_id: DynamicBackingPoolId,
    pub(super) initialization: StateInitialization,
    /// Protocol-only ceiling used for checked evidence. No API may iterate,
    /// reserve, allocate, or claim this many instances.
    pub(super) theoretical_maximum_instances: u32,
}

impl DynamicResourceDescriptor {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        base_resource_id: ResourceId,
        demand: DynamicResourceDemand,
        alignment_bytes: u64,
        usage: BufferUsage,
        element_type: ElementType,
        lifetime: AllocationLifetime,
        kind: AllocationKind,
        storage: DynamicStorageContract,
        initialization: StateInitialization,
        theoretical_maximum_instances: u32,
    ) -> Result<Self, VNextError> {
        validate_active_sequence_ceiling(theoretical_maximum_instances)?;
        if alignment_bytes == 0
            || !alignment_bytes.is_power_of_two()
            || lifetime == AllocationLifetime::Plan
        {
            return Err(invalid_plan(
                "dynamic resource descriptor has invalid alignment or static lifetime",
            ));
        }
        let kind_valid = match &kind {
            AllocationKind::Scratch { .. } => {
                lifetime == AllocationLifetime::Invocation
                    && usage == BufferUsage::Scratch
                    && element_type == ElementType::U8
            }
            AllocationKind::Binding { .. } => {
                lifetime == AllocationLifetime::Invocation
                    && usage == BufferUsage::Binding
                    && element_type == ElementType::U8
            }
            AllocationKind::Persistent { .. } => {
                matches!(
                    lifetime,
                    AllocationLifetime::Request
                        | AllocationLifetime::Sequence
                        | AllocationLifetime::Step
                ) && usage == BufferUsage::Persistent
                    && element_type == ElementType::U8
            }
            AllocationKind::Value => usage != BufferUsage::Weights,
        };
        if !kind_valid {
            return Err(invalid_plan(
                "dynamic resource kind, lifetime, usage, or element type is inconsistent",
            ));
        }
        if initialization == StateInitialization::Zero
            && (kind != AllocationKind::Value
                || usage != BufferUsage::State
                || lifetime != AllocationLifetime::Sequence)
        {
            return Err(invalid_plan(
                "zero initialization requires semantic Sequence state backing",
            ));
        }
        demand.validate()?;
        let pool_id = DynamicBackingPoolId::from_compatibility(&PoolCompatibilityKey::new(
            &storage,
            usage,
            element_type,
            alignment_bytes,
        )?)?;
        let descriptor = Self {
            base_resource_id,
            demand,
            alignment_bytes,
            usage,
            element_type,
            lifetime,
            kind,
            storage,
            pool_id,
            initialization,
            theoretical_maximum_instances,
        };
        descriptor.evaluate_request_bytes_for_shape(descriptor.demand.minimum_shape())?;
        descriptor
            .evaluate_request_bytes_for_shape(descriptor.demand.theoretical_maximum_shape())?;
        Ok(descriptor)
    }

    pub fn base_resource_id(&self) -> &ResourceId {
        &self.base_resource_id
    }

    pub fn demand(&self) -> &DynamicResourceDemand {
        &self.demand
    }

    pub const fn theoretical_maximum_instances(&self) -> u32 {
        self.theoretical_maximum_instances
    }

    pub fn evaluate_logical_request_bytes(
        &self,
        work: &ResourceWorkShape,
    ) -> Result<u64, VNextError> {
        self.demand.evaluate_bytes(work)
    }

    pub(crate) fn evaluate_logical_request_bytes_for_shape(
        &self,
        shape: DynamicResourceShape,
    ) -> Result<u64, VNextError> {
        self.demand.evaluate_shape_bytes(shape)
    }

    pub fn physical_allocation_quantum_bytes(&self) -> u64 {
        match self.storage.profile().allocator() {
            super::DynamicStorageAllocator::LinearArena => self.alignment_bytes,
            super::DynamicStorageAllocator::FixedBlockArena { block_bytes } => {
                block_bytes.max(self.alignment_bytes)
            }
        }
    }

    /// Exact physical claim for one logical shape. The semantic demand stays
    /// unchanged in the plan; allocator geometry is applied only at this
    /// boundary so admission and backing cannot under-count fixed blocks.
    pub fn evaluate_request_bytes(&self, work: &ResourceWorkShape) -> Result<u64, VNextError> {
        self.evaluate_request_bytes_for_shape(work.immediate_shape())
    }

    pub fn evaluate_fit_request_bytes(&self, work: &ResourceWorkShape) -> Result<u64, VNextError> {
        self.evaluate_request_bytes_for_shape(work.fit_shape())
    }

    pub(crate) fn evaluate_request_bytes_for_shape(
        &self,
        shape: DynamicResourceShape,
    ) -> Result<u64, VNextError> {
        quantize_storage_bytes(
            self.evaluate_logical_request_bytes_for_shape(shape)?,
            self.alignment_bytes,
            self.storage.profile(),
        )
    }

    pub fn minimum_request_bytes(&self) -> Result<u64, VNextError> {
        self.evaluate_request_bytes_for_shape(self.demand.minimum_shape())
    }

    pub fn theoretical_maximum_request_bytes(&self) -> Result<u64, VNextError> {
        self.evaluate_request_bytes_for_shape(self.demand.theoretical_maximum_shape())
    }

    pub const fn alignment_bytes(&self) -> u64 {
        self.alignment_bytes
    }

    pub const fn usage(&self) -> BufferUsage {
        self.usage
    }

    pub const fn element_type(&self) -> ElementType {
        self.element_type
    }

    pub const fn lifetime(&self) -> AllocationLifetime {
        self.lifetime
    }

    pub fn kind(&self) -> &AllocationKind {
        &self.kind
    }

    pub fn storage(&self) -> &DynamicStorageContract {
        &self.storage
    }

    pub fn pool_id(&self) -> &DynamicBackingPoolId {
        &self.pool_id
    }

    pub const fn initialization(&self) -> StateInitialization {
        self.initialization
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct DynamicResourceDescriptorWire {
    pub(super) base_resource_id: ResourceId,
    pub(super) demand: DynamicResourceDemand,
    pub(super) alignment_bytes: u64,
    pub(super) usage: BufferUsage,
    pub(super) element_type: ElementType,
    pub(super) lifetime: AllocationLifetime,
    pub(super) kind: AllocationKind,
    pub(super) storage: DynamicStorageContract,
    pub(super) pool_id: DynamicBackingPoolId,
    pub(super) initialization: StateInitialization,
    pub(super) theoretical_maximum_instances: u32,
}

impl<'de> Deserialize<'de> for DynamicResourceDescriptor {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = DynamicResourceDescriptorWire::deserialize(deserializer)?;
        let descriptor = Self::new(
            wire.base_resource_id,
            wire.demand,
            wire.alignment_bytes,
            wire.usage,
            wire.element_type,
            wire.lifetime,
            wire.kind,
            wire.storage,
            wire.initialization,
            wire.theoretical_maximum_instances,
        )
        .map_err(serde::de::Error::custom)?;
        if descriptor.pool_id != wire.pool_id {
            return Err(serde::de::Error::custom(
                "dynamic resource pool id is not core-derived from compatibility",
            ));
        }
        Ok(descriptor)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(transparent)]
pub(super) struct CanonicalU128(String);

impl CanonicalU128 {
    pub(super) fn new(value: u128) -> Self {
        Self(value.to_string())
    }

    pub(super) fn get(&self) -> u128 {
        self.0
            .parse()
            .expect("canonical u128 is validated at construction or deserialization")
    }
}

impl<'de> Deserialize<'de> for CanonicalU128 {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        let parsed = value.parse::<u128>().map_err(serde::de::Error::custom)?;
        if parsed.to_string() != value {
            return Err(serde::de::Error::custom(
                "u128 evidence must be a canonical unsigned decimal string",
            ));
        }
        Ok(Self(value))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InvocationLivenessMode {
    NoInvocationResources,
    /// Every invocation row in this pool is ordered by a transitive node
    /// completion dependency, so one runnable request needs only the maximum
    /// row size rather than their sum.
    TotalOrderReuse,
    /// The plan cannot prove that every invocation row completes before the
    /// next one starts. The runnable minimum therefore sums member resources.
    ConservativeConcurrent,
}

/// A set of Step-scoped logical resources that may project onto one physical
/// extent. Multi-resource slots are emitted only when plan dependencies prove
/// that every member's final user completes before the next member starts.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StepResourceSlot {
    pub(super) kind: StepResourceSlotKind,
    pub(super) resource_ids: Vec<ResourceId>,
}

impl StepResourceSlot {
    pub(super) fn dedicated(resource_id: ResourceId) -> Self {
        Self {
            kind: StepResourceSlotKind::Dedicated,
            resource_ids: vec![resource_id],
        }
    }

    pub(super) fn ordered_single_fence_wave(
        mut resource_ids: Vec<ResourceId>,
    ) -> Result<Self, VNextError> {
        resource_ids.sort();
        if resource_ids.len() < 2 || resource_ids.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(invalid_plan(
                "ordered single-fence step slot requires at least two unique resources",
            ));
        }
        Ok(Self {
            kind: StepResourceSlotKind::OrderedSingleFenceStepWave,
            resource_ids,
        })
    }

    pub(super) fn validate(&self) -> Result<(), VNextError> {
        if self.resource_ids.is_empty()
            || self.resource_ids.windows(2).any(|pair| pair[0] >= pair[1])
            || match self.kind {
                StepResourceSlotKind::Dedicated => self.resource_ids.len() != 1,
                StepResourceSlotKind::OrderedSingleFenceStepWave => self.resource_ids.len() < 2,
            }
        {
            return Err(invalid_plan(
                "step resource slot kind or members are invalid",
            ));
        }
        Ok(())
    }

    pub const fn kind(&self) -> StepResourceSlotKind {
        self.kind
    }

    pub fn resource_ids(&self) -> &[ResourceId] {
        &self.resource_ids
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StepResourceSlotKind {
    Dedicated,
    /// Every member is an internal activation and the runtime must submit the
    /// canonical plan order as one ordered command batch with one terminal
    /// fence before it may consume this reuse proof.
    OrderedSingleFenceStepWave,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InvocationResourceLiveness {
    pub(super) node_id: NodeId,
    pub(super) resource_ids: Vec<ResourceId>,
}

impl InvocationResourceLiveness {
    pub fn node_id(&self) -> &NodeId {
        &self.node_id
    }

    pub fn resource_ids(&self) -> &[ResourceId] {
        &self.resource_ids
    }
}
