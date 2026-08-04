use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use super::super::{
    BatchWorkShape, ClaimedSubmissionWaveBacking, ContractVersion, DeviceDescriptor,
    DeviceReusableAddressScope, DeviceReusableExecutionTopologyFingerprint, DeviceRuntime,
    EncodedDeviceOperation, EncodedReusableExecutionBindings, ExecutablePlanView,
    LogicalBackingSliceAuthority, MemoryPlan, NodeId, OperationId, PlanHash, PlanId, ProviderId,
    ProviderWorkspaceRequirement, SemanticValue, VNextError,
};
use super::foundation::{canonical_sha256, invalid_operation};
use super::invocation::PreparedOperationDispatchBinding;
use super::resolved_value::resource_uses_packed_batch_coordinates;
use super::{
    AttributeId, BatchedOperationInvocation, CapabilityCatalog, EngineProviderDescriptor,
    OperationContract, OperationDescriptor, OperationFailure, OperationProviderDescriptor,
    ResolvedValueBinding, ResolvedValueRole,
};

/// Exact semantic input presented to a selected provider's resource estimator.
/// The core creates this request only after provider selection and verifies the
/// raw estimate against the same independently computed fingerprint. Global
/// admission ceilings are deliberately absent: the provider describes one
/// actual invocation and the scheduler decides how many invocations to admit.
pub struct OperationResourceEstimateRequest<'a> {
    node_id: &'a NodeId,
    operation: &'a OperationDescriptor,
    values: &'a [ResolvedValueBinding],
    attributes: &'a BTreeMap<AttributeId, SemanticValue>,
    input_fingerprint: &'a str,
}

/// Lightweight provider view used to bind dynamic compute topology into a
/// reusable program identity before catalog lookup.
///
/// It deliberately exposes no buffers, request identity, or submission
/// authority. Providers may derive only an opaque fixed-size topology
/// fingerprint from immutable plan semantics, typed reusable-address
/// authority, and the current batch work shape.
pub struct ReusableExecutionTopologyRequest<'a> {
    node_id: &'a NodeId,
    operation_id: &'a OperationId,
    attributes: &'a BTreeMap<AttributeId, SemanticValue>,
    bindings: &'a [ResolvedValueBinding],
    scratch_resource: Option<&'a super::super::ResourceId>,
    binding_resource: Option<&'a super::super::ResourceId>,
    persistent_resource: Option<&'a super::super::ResourceId>,
    memory: &'a MemoryPlan,
    work_shape: &'a BatchWorkShape,
    claimed_backing: &'a ClaimedSubmissionWaveBacking,
    step_backing: &'a [LogicalBackingSliceAuthority],
}

/// How one resolved value address enters a resident reusable executable.
/// Direct captures require lane-stable address authority. Program-bound
/// values are instead materialized into the provider's typed binding slot
/// before every replay and therefore may remain request- or sequence-owned.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReusableExecutionValueAddress {
    Captured {
        role: ResolvedValueRole,
        ordinal: u32,
    },
    ProgramBinding {
        role: ResolvedValueRole,
        ordinal: u32,
    },
}

impl ReusableExecutionValueAddress {
    pub const fn captured(role: ResolvedValueRole, ordinal: u32) -> Self {
        Self::Captured { role, ordinal }
    }

    pub const fn program_binding(role: ResolvedValueRole, ordinal: u32) -> Self {
        Self::ProgramBinding { role, ordinal }
    }

    const fn identity(self) -> (ResolvedValueRole, u32) {
        match self {
            Self::Captured { role, ordinal } | Self::ProgramBinding { role, ordinal } => {
                (role, ordinal)
            }
        }
    }
}

/// Provider workspace addresses captured by a resident executable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReusableExecutionWorkspaceAddress {
    Scratch,
    Binding,
    Persistent,
}

impl<'a> ReusableExecutionTopologyRequest<'a> {
    pub(super) fn new(
        node_id: &'a NodeId,
        operation_id: &'a OperationId,
        attributes: &'a BTreeMap<AttributeId, SemanticValue>,
        bindings: &'a [ResolvedValueBinding],
        scratch_resource: Option<&'a super::super::ResourceId>,
        binding_resource: Option<&'a super::super::ResourceId>,
        persistent_resource: Option<&'a super::super::ResourceId>,
        memory: &'a MemoryPlan,
        work_shape: &'a BatchWorkShape,
        claimed_backing: &'a ClaimedSubmissionWaveBacking,
        step_backing: &'a [LogicalBackingSliceAuthority],
    ) -> Result<Self, VNextError> {
        if work_shape.participants().is_empty() {
            return Err(invalid_operation(
                "reusable execution topology request has no participants",
            ));
        }
        Ok(Self {
            node_id,
            operation_id,
            attributes,
            bindings,
            scratch_resource,
            binding_resource,
            persistent_resource,
            memory,
            work_shape,
            claimed_backing,
            step_backing,
        })
    }

    pub fn node_id(&self) -> &NodeId {
        self.node_id
    }

    pub fn operation_id(&self) -> &OperationId {
        self.operation_id
    }

    pub fn attributes(&self) -> &BTreeMap<AttributeId, SemanticValue> {
        self.attributes
    }

    pub fn bindings(&self) -> &[ResolvedValueBinding] {
        self.bindings
    }

    pub fn work_shape(&self) -> &BatchWorkShape {
        self.work_shape
    }

    /// Returns whether one resolved value uses the packed token coordinates of
    /// the physical submission wave. Providers must use the same coordinate
    /// authority here that runtime invocation construction uses when selecting
    /// captured buffer regions.
    pub fn binding_uses_packed_batch_coordinates(
        &self,
        role: ResolvedValueRole,
        ordinal: u32,
    ) -> Result<bool, VNextError> {
        let binding = self
            .bindings
            .iter()
            .find(|binding| binding.role() == role && binding.ordinal() == ordinal)
            .ok_or_else(|| {
                invalid_operation("reusable topology requested an unknown value binding")
            })?;
        let [component] = binding.storage().components() else {
            return Err(invalid_operation(
                "reusable topology coordinate ownership requires one resource component",
            ));
        };
        resource_uses_packed_batch_coordinates(self.memory, component.resource_id())
    }

    /// Resolves one complete provider address contract. Every value binding
    /// must appear exactly once, preventing a provider from gaining replay by
    /// silently omitting a dynamic operand. Program-bound values are legal
    /// only when the resident executable captures a lane-stable binding slot.
    pub fn reusable_address_scope(
        &self,
        values: &[ReusableExecutionValueAddress],
        workspaces: &[ReusableExecutionWorkspaceAddress],
    ) -> Result<Option<DeviceReusableAddressScope>, VNextError> {
        if values.len() != self.bindings.len()
            || values.iter().enumerate().any(|(index, value)| {
                values[..index]
                    .iter()
                    .any(|prior| prior.identity() == value.identity())
            })
            || self.bindings.iter().any(|binding| {
                values
                    .iter()
                    .filter(|value| value.identity() == (binding.role(), binding.ordinal()))
                    .count()
                    != 1
            })
            || workspaces.iter().enumerate().any(|(index, workspace)| {
                workspaces[..index].iter().any(|prior| prior == workspace)
            })
        {
            return Err(invalid_operation(
                "reusable topology address contract does not cover every value exactly once",
            ));
        }

        let has_program_bound_values = values
            .iter()
            .any(|value| matches!(value, ReusableExecutionValueAddress::ProgramBinding { .. }));
        if has_program_bound_values
            && !workspaces.contains(&ReusableExecutionWorkspaceAddress::Binding)
        {
            return Err(invalid_operation(
                "program-bound reusable values require a captured binding workspace",
            ));
        }

        let mut aggregate = DeviceReusableAddressScope::Plan;
        for value in values {
            let ReusableExecutionValueAddress::Captured { role, ordinal } = value else {
                continue;
            };
            let Some(scope) = self.binding_reusable_address_scope(*role, *ordinal)? else {
                return Ok(None);
            };
            aggregate = merge_reusable_address_scope(aggregate, scope)?;
        }
        for workspace in workspaces {
            let scope = match workspace {
                ReusableExecutionWorkspaceAddress::Scratch => {
                    self.scratch_reusable_address_scope()?
                }
                ReusableExecutionWorkspaceAddress::Binding => {
                    self.binding_workspace_reusable_address_scope()?
                }
                ReusableExecutionWorkspaceAddress::Persistent => {
                    self.persistent_workspace_reusable_address_scope()?
                }
            };
            let Some(scope) = scope else {
                return Ok(None);
            };
            aggregate = merge_reusable_address_scope(aggregate, scope)?;
        }
        Ok(Some(aggregate))
    }

    /// Returns the reusable address authority shared by every physical
    /// component of one resolved value. `None` means at least one component is
    /// submission-scoped and the backend must exclude commands that capture it
    /// from resident reusable segments.
    pub fn binding_reusable_address_scope(
        &self,
        role: ResolvedValueRole,
        ordinal: u32,
    ) -> Result<Option<DeviceReusableAddressScope>, VNextError> {
        let binding = self
            .bindings
            .iter()
            .find(|binding| binding.role() == role && binding.ordinal() == ordinal)
            .ok_or_else(|| {
                invalid_operation("reusable topology requested an unknown value binding")
            })?;
        let mut aggregate = DeviceReusableAddressScope::Plan;
        for component in binding.storage().components() {
            let Some(component_scope) =
                self.resource_reusable_address_scope(component.resource_id())?
            else {
                return Ok(None);
            };
            aggregate = merge_reusable_address_scope(aggregate, component_scope)?;
        }
        Ok(Some(aggregate))
    }

    /// Returns the address authority for scratch captured by the provider.
    /// `None` means the scratch address is scoped to this submission.
    pub fn scratch_reusable_address_scope(
        &self,
    ) -> Result<Option<DeviceReusableAddressScope>, VNextError> {
        self.workspace_reusable_address_scope(self.scratch_resource, "scratch")
    }

    /// Returns the address authority for reusable-program binding workspace.
    /// `None` means the binding workspace address is scoped to this submission.
    pub fn binding_workspace_reusable_address_scope(
        &self,
    ) -> Result<Option<DeviceReusableAddressScope>, VNextError> {
        self.workspace_reusable_address_scope(self.binding_resource, "binding")
    }

    /// Returns the address authority for persistent workspace captured by the
    /// provider. `None` means the address is scoped to this submission.
    pub fn persistent_workspace_reusable_address_scope(
        &self,
    ) -> Result<Option<DeviceReusableAddressScope>, VNextError> {
        self.workspace_reusable_address_scope(self.persistent_resource, "persistent")
    }

    fn workspace_reusable_address_scope(
        &self,
        resource_id: Option<&super::super::ResourceId>,
        workspace: &str,
    ) -> Result<Option<DeviceReusableAddressScope>, VNextError> {
        let resource_id = resource_id.ok_or_else(|| {
            invalid_operation(format!(
                "reusable topology requested absent provider {workspace} workspace"
            ))
        })?;
        self.resource_reusable_address_scope(resource_id)
    }

    fn resource_reusable_address_scope(
        &self,
        resource_id: &super::super::ResourceId,
    ) -> Result<Option<DeviceReusableAddressScope>, VNextError> {
        if self
            .memory
            .static_allocations()
            .binary_search_by(|allocation| allocation.resource_id().cmp(resource_id))
            .is_ok()
        {
            return Ok(Some(DeviceReusableAddressScope::Plan));
        }

        let mut resource_scope = None;
        for backing_slices in [self.claimed_backing.backing_slices(), self.step_backing] {
            let authority_start =
                backing_slices.partition_point(|authority| authority.resource_id() < resource_id);
            let authority_end = authority_start
                + backing_slices[authority_start..]
                    .partition_point(|authority| authority.resource_id() == resource_id);
            for authority in &backing_slices[authority_start..authority_end] {
                let Some(authority_scope) = authority.reusable_address_scope() else {
                    return Ok(None);
                };
                resource_scope = Some(merge_reusable_address_scope(
                    resource_scope.unwrap_or(DeviceReusableAddressScope::Plan),
                    authority_scope,
                )?);
            }
        }
        if resource_scope.is_some() {
            return Ok(resource_scope);
        }
        if self
            .memory
            .dynamic_descriptors()
            .binary_search_by(|descriptor| descriptor.base_resource_id().cmp(resource_id))
            .is_ok()
        {
            return Ok(None);
        }
        Err(invalid_operation(
            "reusable topology references an unknown memory resource",
        ))
    }
}

fn merge_reusable_address_scope(
    left: DeviceReusableAddressScope,
    right: DeviceReusableAddressScope,
) -> Result<DeviceReusableAddressScope, VNextError> {
    match (left, right) {
        (DeviceReusableAddressScope::Plan, scope) | (scope, DeviceReusableAddressScope::Plan) => {
            Ok(scope)
        }
        (
            DeviceReusableAddressScope::ExecutionLane(left),
            DeviceReusableAddressScope::ExecutionLane(right),
        ) if left == right => Ok(DeviceReusableAddressScope::ExecutionLane(left)),
        _ => Err(invalid_operation(
            "reusable topology value spans different execution lanes",
        )),
    }
}

impl<'a> OperationResourceEstimateRequest<'a> {
    pub(crate) fn new(
        node_id: &'a NodeId,
        operation: &'a OperationDescriptor,
        values: &'a [ResolvedValueBinding],
        attributes: &'a BTreeMap<AttributeId, SemanticValue>,
        input_fingerprint: &'a str,
    ) -> Result<Self, VNextError> {
        operation.validate()?;
        operation.validate_attributes(attributes)?;
        operation.validate_resolved_bindings(values)?;
        if !canonical_sha256(input_fingerprint) {
            return Err(invalid_operation(
                "resource estimator request has invalid input fingerprint",
            ));
        }
        Ok(Self {
            node_id,
            operation,
            values,
            attributes,
            input_fingerprint,
        })
    }

    pub fn node_id(&self) -> &NodeId {
        self.node_id
    }

    pub fn operation(&self) -> &OperationDescriptor {
        self.operation
    }

    pub fn values(&self) -> &[ResolvedValueBinding] {
        self.values
    }

    pub fn attributes(&self) -> &BTreeMap<AttributeId, SemanticValue> {
        self.attributes
    }

    pub fn input_fingerprint(&self) -> &str {
        self.input_fingerprint
    }
}

/// Untrusted raw output from one registered provider implementation. Identity
/// and input claims remain explicit so the core can reject a buggy or
/// malicious implementation before creating a trusted plan resource record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OperationResourceEstimate {
    estimator_id: String,
    estimator_version: ContractVersion,
    estimator_implementation_fingerprint: String,
    claimed_input_fingerprint: String,
    value_alignment_bytes: u64,
    scratch: Option<ProviderWorkspaceRequirement>,
    binding: Option<ProviderWorkspaceRequirement>,
    persistent: Option<ProviderWorkspaceRequirement>,
}

impl OperationResourceEstimate {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        estimator_id: impl Into<String>,
        estimator_version: ContractVersion,
        estimator_implementation_fingerprint: impl Into<String>,
        claimed_input_fingerprint: impl Into<String>,
        value_alignment_bytes: u64,
        scratch: Option<ProviderWorkspaceRequirement>,
        persistent: Option<ProviderWorkspaceRequirement>,
    ) -> Self {
        Self {
            estimator_id: estimator_id.into(),
            estimator_version,
            estimator_implementation_fingerprint: estimator_implementation_fingerprint.into(),
            claimed_input_fingerprint: claimed_input_fingerprint.into(),
            value_alignment_bytes,
            scratch,
            binding: None,
            persistent,
        }
    }

    pub fn with_binding(mut self, binding: ProviderWorkspaceRequirement) -> Self {
        self.binding = Some(binding);
        self
    }

    pub fn estimator_id(&self) -> &str {
        &self.estimator_id
    }

    pub const fn estimator_version(&self) -> ContractVersion {
        self.estimator_version
    }

    pub fn estimator_implementation_fingerprint(&self) -> &str {
        &self.estimator_implementation_fingerprint
    }

    pub fn claimed_input_fingerprint(&self) -> &str {
        &self.claimed_input_fingerprint
    }

    pub const fn value_alignment_bytes(&self) -> u64 {
        self.value_alignment_bytes
    }

    pub fn scratch(&self) -> Option<&ProviderWorkspaceRequirement> {
        self.scratch.as_ref()
    }

    pub fn binding(&self) -> Option<&ProviderWorkspaceRequirement> {
        self.binding.as_ref()
    }

    pub fn persistent(&self) -> Option<&ProviderWorkspaceRequirement> {
        self.persistent.as_ref()
    }
}

/// Runtime-independent planning half of an operation provider. This remains
/// object-safe so planning can invoke the real implementation without
/// inventing a device runtime type.
pub trait OperationResourceEstimator: Send + Sync {
    fn descriptor(&self) -> &OperationProviderDescriptor;

    fn estimate_resources(
        &self,
        request: OperationResourceEstimateRequest<'_>,
    ) -> Result<OperationResourceEstimate, VNextError>;
}

/// Typed implementation registry used at the planning trust boundary. The
/// core requires exactly one matching contract and estimator; missing or
/// duplicate registrations fail closed before an executable plan is built.
pub trait OperationPlanningRegistry: Send + Sync {
    fn contracts_for(&self, operation_id: &OperationId) -> Vec<&dyn OperationContract>;

    fn estimators_for(&self, provider_id: &ProviderId) -> Vec<&dyn OperationResourceEstimator>;
}

/// Process-local authority for the composition root that supplied the exact
/// contract and provider objects used during planning. It deliberately has no
/// wire representation and is never part of a deterministic plan hash.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct OperationRegistryAuthority(u64);

impl OperationRegistryAuthority {
    fn mint() -> Result<Self, VNextError> {
        static NEXT_AUTHORITY: AtomicU64 = AtomicU64::new(1);
        let id = NEXT_AUTHORITY
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                current.checked_add(1)
            })
            .map_err(|_| invalid_operation("operation registry authority space exhausted"))?;
        Ok(Self(id))
    }
}

/// Planning view issued only by a concrete runtime registry. Holding this
/// view proves that node resolution used the same composition root that can
/// later bind the selected runtime provider.
pub struct OperationPlanningHandle<'registry> {
    registry: &'registry dyn OperationPlanningRegistry,
    authority: OperationRegistryAuthority,
}

impl OperationPlanningHandle<'_> {
    pub(crate) fn authority(&self) -> &OperationRegistryAuthority {
        &self.authority
    }
}

impl OperationPlanningRegistry for OperationPlanningHandle<'_> {
    fn contracts_for(&self, operation_id: &OperationId) -> Vec<&dyn OperationContract> {
        self.registry.contracts_for(operation_id)
    }

    fn estimators_for(&self, provider_id: &ProviderId) -> Vec<&dyn OperationResourceEstimator> {
        self.registry.estimators_for(provider_id)
    }
}

/// A provider declaration for the compute topology captured by a resident
/// reusable program.
///
/// `Static` means every captured choice and address is already bound by the
/// immutable plan and lane identity. `Dynamic` contributes an opaque
/// provider-owned fingerprint. `EagerBoundary` means this node lacks reusable
/// address authority and must remain outside resident segments. It does not
/// veto reusable segments owned by other nodes in the same wave.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReusableExecutionTopology {
    Static,
    Dynamic(DeviceReusableExecutionTopologyFingerprint),
    EagerBoundary,
}

/// A compile-time provider contract for one concrete runtime buffer type. The
/// kernel method consumes only a dispatch-created invocation.
pub trait OperationProvider<R: DeviceRuntime>: OperationResourceEstimator {
    /// Publishes the provider-private compute topology that must match a
    /// resident reusable program. Static topology and an eager boundary are
    /// intentionally distinct states: a provider may never silently turn a
    /// submission-scoped address into resident state, while one eager node must
    /// not disable safe resident segments elsewhere in the wave.
    ///
    /// This declaration is intentionally required. A new provider cannot
    /// silently inherit a static topology after adding shape-dependent kernel
    /// selection. Providers must query the reusable address scope of every
    /// value binding and workspace their encoded command actually captures;
    /// an unqueried operand is not covered by this contract.
    fn reusable_execution_topology(
        &self,
        request: ReusableExecutionTopologyRequest<'_>,
    ) -> Result<ReusableExecutionTopology, VNextError>;

    fn encode_selected(
        &self,
        invocation: BatchedOperationInvocation<'_, R::Buffer>,
    ) -> Result<EncodedDeviceOperation<R::Command>, OperationFailure>;

    /// Encodes only the request-varying boundaries around a compute segment
    /// that was already prepared as a reusable backend executable.
    ///
    /// The default deliberately reuses the exact provider implementation and
    /// discards its compute command. Providers may override this cold-selected
    /// boundary to avoid rebuilding static launch metadata.
    fn encode_reusable_execution_bindings(
        &self,
        invocation: BatchedOperationInvocation<'_, R::Buffer>,
    ) -> Result<EncodedReusableExecutionBindings<R::Command>, OperationFailure> {
        self.encode_selected(invocation)
            .map(EncodedReusableExecutionBindings::from_operation)
    }
}

/// Composition-root registry that owns the exact provider objects used for
/// both planning and runtime dispatch. A dispatch call receives only a bound
/// handle issued by this registry, never an arbitrary provider implementation.
pub struct OperationRuntimeRegistry<R>
where
    R: DeviceRuntime,
{
    authority: OperationRegistryAuthority,
    contracts: BTreeMap<OperationId, Box<dyn OperationContract>>,
    providers: BTreeMap<ProviderId, Arc<dyn OperationProvider<R>>>,
}

impl<R> OperationRuntimeRegistry<R>
where
    R: DeviceRuntime,
{
    pub fn new(
        contracts: Vec<Box<dyn OperationContract>>,
        providers: Vec<Box<dyn OperationProvider<R>>>,
    ) -> Result<Self, VNextError> {
        if contracts.is_empty() || providers.is_empty() {
            return Err(invalid_operation(
                "operation runtime registry requires contracts and providers",
            ));
        }
        let mut contract_map = BTreeMap::new();
        for contract in contracts {
            let descriptor = contract.descriptor();
            descriptor.validate()?;
            let operation_id = descriptor.id.clone();
            if contract_map
                .insert(operation_id.clone(), contract)
                .is_some()
            {
                return Err(invalid_operation(format!(
                    "operation runtime registry has duplicate contract `{operation_id}`"
                )));
            }
        }
        let mut provider_map: BTreeMap<ProviderId, Arc<dyn OperationProvider<R>>> = BTreeMap::new();
        for provider in providers {
            let descriptor = provider.descriptor();
            let contract = contract_map.get(descriptor.operation_id()).ok_or_else(|| {
                invalid_operation(format!(
                    "runtime provider `{}` has no registered operation contract",
                    descriptor.provider_id()
                ))
            })?;
            if descriptor.operation_fingerprint() != contract.descriptor().fingerprint()? {
                return Err(invalid_operation(format!(
                    "runtime provider `{}` differs from its registered operation contract",
                    descriptor.provider_id()
                )));
            }
            let provider_id = descriptor.provider_id().clone();
            if provider_map
                .insert(provider_id.clone(), Arc::from(provider))
                .is_some()
            {
                return Err(invalid_operation(format!(
                    "operation runtime registry has duplicate or byte-identical provider `{provider_id}`"
                )));
            }
        }
        Ok(Self {
            authority: OperationRegistryAuthority::mint()?,
            contracts: contract_map,
            providers: provider_map,
        })
    }

    /// Derives the planning catalog from the exact contract/provider objects
    /// retained for dispatch, preventing descriptor drift between two
    /// independently assembled registries.
    pub fn capability_catalog(
        &self,
        device: DeviceDescriptor,
        engine_providers: Vec<EngineProviderDescriptor>,
    ) -> Result<CapabilityCatalog, VNextError> {
        let operations = self
            .contracts
            .values()
            .map(|contract| contract.descriptor().clone())
            .collect::<Vec<_>>();
        let mut providers = self
            .contracts
            .keys()
            .cloned()
            .map(|operation_id| (operation_id, Vec::new()))
            .collect::<BTreeMap<_, _>>();
        for provider in self.providers.values() {
            providers
                .get_mut(provider.descriptor().operation_id())
                .ok_or_else(|| {
                    invalid_operation(
                        "runtime provider operation is absent while deriving its catalog",
                    )
                })?
                .push(provider.descriptor().clone());
        }
        CapabilityCatalog::new(device, operations, providers, engine_providers)
    }

    pub fn planning(&self) -> OperationPlanningHandle<'_> {
        OperationPlanningHandle {
            registry: self,
            authority: self.authority.clone(),
        }
    }

    pub fn bind<'registry>(
        &'registry self,
        resolved: &dyn ExecutablePlanView,
        node_id: &NodeId,
    ) -> Result<BoundOperationProvider<'registry, R>, VNextError> {
        let provider = self.selected_provider(resolved, node_id)?;
        let plan = resolved.execution_plan();
        let dispatch =
            PreparedOperationDispatchBinding::prepare(resolved, provider.descriptor(), node_id)?;
        Ok(BoundOperationProvider {
            provider: BoundOperationProviderSource::Borrowed(provider.as_ref()),
            plan_id: plan.payload().plan_id().clone(),
            plan_hash: plan.plan_hash().clone(),
            node_id: node_id.clone(),
            dispatch,
        })
    }

    /// Binds every selected provider once in immutable plan-node order.
    ///
    /// The returned handles own their provider objects, so execution can drop
    /// the composition registry and cannot re-enter provider lookup from the
    /// token loop.
    pub fn bind_plan(
        &self,
        resolved: &dyn ExecutablePlanView,
    ) -> Result<BoundOperationProviderSet<R>, VNextError> {
        let providers = resolved
            .execution_plan()
            .payload()
            .nodes()
            .iter()
            .map(|node| {
                let provider = self.selected_provider(resolved, node.id())?;
                let plan = resolved.execution_plan();
                let dispatch = PreparedOperationDispatchBinding::prepare(
                    resolved,
                    provider.descriptor(),
                    node.id(),
                )?;
                Ok(BoundOperationProvider {
                    provider: BoundOperationProviderSource::Owned(Arc::clone(provider)),
                    plan_id: plan.payload().plan_id().clone(),
                    plan_hash: plan.plan_hash().clone(),
                    node_id: node.id().clone(),
                    dispatch,
                })
            })
            .collect::<Result<Vec<BoundOperationProvider<'static, R>>, _>>()?;
        if providers.is_empty() {
            return Err(invalid_operation(
                "executable plan cannot bind an empty provider set",
            ));
        }
        Ok(BoundOperationProviderSet { providers })
    }

    fn selected_provider(
        &self,
        resolved: &dyn ExecutablePlanView,
        node_id: &NodeId,
    ) -> Result<&Arc<dyn OperationProvider<R>>, VNextError> {
        let plan = resolved.execution_plan();
        if plan.operation_registry_authority() != &self.authority {
            return Err(invalid_operation(
                "resolved plan belongs to a different operation runtime registry",
            ));
        }
        let node = plan
            .payload()
            .nodes()
            .iter()
            .find(|node| node.id() == node_id)
            .ok_or_else(|| invalid_operation(format!("plan has no node `{node_id}`")))?;
        let provider = self
            .providers
            .get(node.selection().selected_provider())
            .ok_or_else(|| {
                invalid_operation(format!(
                    "runtime registry has no selected provider `{}`",
                    node.selection().selected_provider()
                ))
            })?;
        let catalog_provider = resolved
            .capabilities()
            .providers_for(node.operation_id())?
            .iter()
            .find(|candidate| candidate.provider_id() == provider.descriptor().provider_id())
            .ok_or_else(|| invalid_operation("runtime provider is absent from resolved catalog"))?;
        if provider.descriptor() != catalog_provider
            || provider.descriptor().provider_id() != node.selection().selected_provider()
            || provider.descriptor().provider_implementation_fingerprint()
                != node.provider_implementation_fingerprint()
        {
            return Err(invalid_operation(
                "runtime provider is not the exact registry object selected by the resolved plan",
            ));
        }
        Ok(provider)
    }
}

impl<R> OperationPlanningRegistry for OperationRuntimeRegistry<R>
where
    R: DeviceRuntime,
{
    fn contracts_for(&self, operation_id: &OperationId) -> Vec<&dyn OperationContract> {
        self.contracts
            .get(operation_id)
            .map(|contract| vec![contract.as_ref()])
            .unwrap_or_default()
    }

    fn estimators_for(&self, provider_id: &ProviderId) -> Vec<&dyn OperationResourceEstimator> {
        self.providers
            .get(provider_id)
            .map(|provider| vec![provider.as_ref() as &dyn OperationResourceEstimator])
            .unwrap_or_default()
    }
}

enum BoundOperationProviderSource<'registry, R>
where
    R: DeviceRuntime,
{
    Borrowed(&'registry dyn OperationProvider<R>),
    Owned(Arc<dyn OperationProvider<R>>),
}

impl<R> BoundOperationProviderSource<'_, R>
where
    R: DeviceRuntime,
{
    fn provider(&self) -> &dyn OperationProvider<R> {
        match self {
            Self::Borrowed(provider) => *provider,
            Self::Owned(provider) => provider.as_ref(),
        }
    }
}

/// Unforgeable per-node provider authority. Its provider object and plan/node
/// binding are private. Normal bindings borrow the composition registry;
/// immutable plan bindings own the same selected provider object.
pub struct BoundOperationProvider<'registry, R>
where
    R: DeviceRuntime,
{
    provider: BoundOperationProviderSource<'registry, R>,
    plan_id: PlanId,
    plan_hash: PlanHash,
    node_id: NodeId,
    dispatch: PreparedOperationDispatchBinding,
}

impl<R> BoundOperationProvider<'_, R>
where
    R: DeviceRuntime,
{
    pub(super) fn provider(&self) -> &dyn OperationProvider<R> {
        self.provider.provider()
    }

    pub(super) fn validate_binding(
        &self,
        resolved: &dyn ExecutablePlanView,
        node_id: &NodeId,
    ) -> Result<(), VNextError> {
        let plan = resolved.execution_plan();
        if self.plan_id != *plan.payload().plan_id()
            || self.plan_hash != *plan.plan_hash()
            || &self.node_id != node_id
        {
            return Err(invalid_operation(
                "bound operation provider belongs to a different plan or node",
            ));
        }
        self.dispatch.node(resolved, node_id)?;
        Ok(())
    }

    pub(super) fn matches_plan_node(
        &self,
        plan_id: &PlanId,
        plan_hash: &PlanHash,
        node_id: &NodeId,
    ) -> bool {
        &self.plan_id == plan_id && &self.plan_hash == plan_hash && &self.node_id == node_id
    }

    pub(super) fn dispatch(&self) -> &PreparedOperationDispatchBinding {
        &self.dispatch
    }

    pub fn descriptor(&self) -> &OperationProviderDescriptor {
        self.provider().descriptor()
    }
}

/// Immutable provider selection for every node in one executable plan.
///
/// Construction is restricted to [`OperationRuntimeRegistry::bind_plan`],
/// which checks registry authority, catalog identity, provider fingerprint,
/// and node order before the runtime begins executing requests.
pub struct BoundOperationProviderSet<R>
where
    R: DeviceRuntime,
{
    providers: Vec<BoundOperationProvider<'static, R>>,
}

impl<R> BoundOperationProviderSet<R>
where
    R: DeviceRuntime,
{
    pub fn providers(&self) -> &[BoundOperationProvider<'static, R>] {
        &self.providers
    }

    pub fn len(&self) -> usize {
        self.providers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.providers.is_empty()
    }
}
