//! Numerical selector tests only. This view provides no live resource proof,
//! graph program or submission permission; real adapter/graph tests live at
//! the interface and product boundaries.
use super::*;
use ferrum_interfaces::vnext::{
    AliasPolicy, BufferUsage, DeviceReusableAddressScope, MemoryPlan, OperationCostWorkRow,
    OperationId, ProgramValueId, ResolvedTensorSpec, ResolvedValueStorage, ResourceId,
    ReusableExecutionWorkspaceAddress, TensorAccess,
};
use std::collections::BTreeMap;
use std::num::NonZeroU64;

struct View {
    operation: OperationId,
    attributes: BTreeMap<AttributeId, SemanticValue>,
    values: Vec<ResolvedValueBinding>,
    scratch: ResourceId,
    scopes: BTreeMap<ResourceId, DeviceReusableAddressScope>,
    rows: Vec<OperationCostWorkRow>,
    packed: bool,
}

impl View {
    fn new(input_count: u32) -> Self {
        let mut values = Vec::new();
        for (role, ordinal) in (0..input_count)
            .map(|ordinal| (ResolvedValueRole::Input, ordinal))
            .chain([(ResolvedValueRole::Output, 0)])
        {
            let name = format!("topology.{role:?}.{ordinal}");
            values.push(
                ResolvedValueBinding::new(
                    ProgramValueId::new(name.clone()).unwrap(),
                    role,
                    ordinal,
                    ResolvedTensorSpec::new(
                        vec![16, 8],
                        ElementType::F16,
                        ResolvedTensorLayout::Contiguous,
                    )
                    .unwrap(),
                    if role == ResolvedValueRole::Input {
                        TensorAccess::Read
                    } else {
                        TensorAccess::Write
                    },
                    AliasPolicy::NoAlias,
                    BufferUsage::Activations,
                    None,
                    ResolvedValueStorage::single(
                        ResourceId::new(name).unwrap(),
                        0,
                        256,
                        ElementType::F16,
                    )
                    .unwrap(),
                )
                .unwrap(),
            );
        }
        let scratch = ResourceId::new("topology.scratch").unwrap();
        let scopes = values
            .iter()
            .flat_map(|value| value.storage().components())
            .map(|part| (part.resource_id().clone(), DeviceReusableAddressScope::Plan))
            .chain([(scratch.clone(), DeviceReusableAddressScope::Plan)])
            .collect();
        Self {
            operation: OperationId::new("topology.numeric-test").unwrap(),
            attributes: BTreeMap::new(),
            values,
            scratch,
            scopes,
            rows: vec![row(2, 1), row(7, 3)],
            packed: true,
        }
    }
}

fn row(offset: u64, count: u64) -> OperationCostWorkRow {
    OperationCostWorkRow {
        offset,
        count: NonZeroU64::new(count).unwrap(),
        full_input_tokens: NonZeroU64::new(offset + count).unwrap(),
    }
}

impl ReusableExecutionTopologyView for View {
    fn operation_id(&self) -> &OperationId {
        &self.operation
    }
    fn attributes(&self) -> &BTreeMap<AttributeId, SemanticValue> {
        &self.attributes
    }
    fn bindings(&self) -> &[ResolvedValueBinding] {
        &self.values
    }
    fn memory_plan(&self) -> &MemoryPlan {
        panic!("this numerical selector consumes declared coordinates/scopes, not a fabricated memory plan")
    }
    fn participant_count(&self) -> usize {
        self.rows.len()
    }
    fn immediate_tokens(&self) -> u64 {
        self.rows.iter().map(|row| row.count.get()).sum()
    }
    fn token_row(&self, index: usize) -> Option<OperationCostWorkRow> {
        self.rows.get(index).copied()
    }
    fn workspace_resource(
        &self,
        workspace: ReusableExecutionWorkspaceAddress,
    ) -> Option<&ResourceId> {
        match workspace {
            ReusableExecutionWorkspaceAddress::Scratch => Some(&self.scratch),
            _ => None,
        }
    }
    fn resource_reusable_address_scope(
        &self,
        resource: &ResourceId,
    ) -> Result<Option<DeviceReusableAddressScope>, VNextError> {
        Ok(self.scopes.get(resource).copied())
    }
    fn binding_uses_packed_batch_coordinates(
        &self,
        _: ResolvedValueRole,
        _: u32,
    ) -> Result<bool, VNextError> {
        Ok(self.packed)
    }
}

fn topology(view: &View) -> ReusableExecutionTopology {
    reusable_token_topology(
        view,
        b"ferrum.cuda.last-token-linear.reusable-topology.v3\0",
    )
    .unwrap()
}

#[test]
fn token_topology_preserves_coordinate_authority_and_physical_row_order() {
    let mut view = View::new(2);
    let packed = topology(&view);
    assert!(matches!(packed, ReusableExecutionTopology::Dynamic(_)));
    view.rows[0] = row(20, 1);
    view.rows[1] = row(70, 3);
    assert_eq!(
        packed,
        topology(&view),
        "packed input excludes owner-local history addresses"
    );
    view.packed = false;
    let owner_local = topology(&view);
    view.rows[0] = row(21, 1);
    assert_ne!(
        owner_local,
        topology(&view),
        "owner-local input must bind its real source span"
    );
    view.packed = true;
    view.rows.reverse();
    assert_ne!(
        packed,
        topology(&view),
        "same totals do not make reordered physical rows equivalent"
    );
}

#[test]
fn token_topology_does_not_promote_an_unproved_value_address_to_static() {
    for index in 0..3 {
        let mut view = View::new(2);
        let resource = view.values[index].storage().components()[0]
            .resource_id()
            .clone();
        view.scopes.remove(&resource);
        assert_eq!(topology(&view), ReusableExecutionTopology::EagerBoundary);
    }
}

#[test]
fn ffn_topology_needs_every_value_and_the_real_scratch_scope() {
    use transformer::{static_contiguous_reusable_topology, CapturedProviderWorkspace};
    let mut view = View::new(3);
    let query = |view: &View| {
        static_contiguous_reusable_topology(view, 3, &[CapturedProviderWorkspace::Scratch])
    };
    assert_eq!(query(&view).unwrap(), ReusableExecutionTopology::Static);
    view.scopes.remove(&view.scratch);
    assert_eq!(
        query(&view).unwrap(),
        ReusableExecutionTopology::EagerBoundary
    );
    view.scopes
        .insert(view.scratch.clone(), DeviceReusableAddressScope::Plan);
    let output = view.values.last().unwrap().storage().components()[0]
        .resource_id()
        .clone();
    view.scopes.remove(&output);
    assert_eq!(
        query(&view).unwrap(),
        ReusableExecutionTopology::EagerBoundary
    );
    view.scopes.insert(output, DeviceReusableAddressScope::Plan);
    view.values.pop();
    assert!(
        query(&view).is_err(),
        "an incomplete binding contract is an error, not a graph topology"
    );
}

#[test]
fn rms_residual_topology_requires_both_inputs_and_output_address_evidence() {
    let query = |view: &View| transformer::static_contiguous_reusable_topology(view, 2, &[]);
    let mut view = View::new(2);
    assert_eq!(query(&view).unwrap(), ReusableExecutionTopology::Static);
    for value in &view.values {
        let resource = value.storage().components()[0].resource_id().clone();
        let original = view.scopes.remove(&resource).unwrap();
        assert_eq!(
            query(&view).unwrap(),
            ReusableExecutionTopology::EagerBoundary
        );
        view.scopes.insert(resource, original);
    }
    view.values.pop();
    assert!(
        query(&view).is_err(),
        "a missing output binding is not an eager fallback proof"
    );
}
