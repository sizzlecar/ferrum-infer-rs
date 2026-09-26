//! Common numerical topology inputs for actual and future selection.
//! Implementations expose no buffers, leases or permission to submit.
use super::foundation::invalid_operation;
use super::registry::merge_reusable_address_scope;
use super::*;
use crate::vnext::{DeviceReusableAddressScope, MemoryPlan, ResourceId, VNextError};
use std::collections::BTreeMap;

/// Provider topology selectors consume this same interface on the actual
/// admitted wave and on a fenced numerical resource projection.
pub trait ReusableExecutionTopologyView {
    fn operation_id(&self) -> &crate::vnext::OperationId;
    fn attributes(&self) -> &BTreeMap<AttributeId, SemanticValue>;
    fn bindings(&self) -> &[ResolvedValueBinding];
    fn memory_plan(&self) -> &MemoryPlan;
    fn participant_count(&self) -> usize;
    fn immediate_tokens(&self) -> u64;
    fn token_row(&self, index: usize) -> Option<OperationCostWorkRow>;
    fn workspace_resource(
        &self,
        workspace: ReusableExecutionWorkspaceAddress,
    ) -> Option<&ResourceId>;
    fn resource_reusable_address_scope(
        &self,
        resource: &ResourceId,
    ) -> Result<Option<DeviceReusableAddressScope>, VNextError>;

    fn binding_uses_packed_batch_coordinates(
        &self,
        role: ResolvedValueRole,
        ordinal: u32,
    ) -> Result<bool, VNextError> {
        let binding = self
            .bindings()
            .iter()
            .find(|b| b.role() == role && b.ordinal() == ordinal)
            .ok_or_else(|| invalid_operation("topology references unknown binding"))?;
        let [component] = binding.storage().components() else {
            return Err(invalid_operation(
                "topology coordinate ownership requires one resource component",
            ));
        };
        super::resolved_value::resource_uses_packed_batch_coordinates(
            self.memory_plan(),
            component.resource_id(),
        )
    }
    fn binding_reusable_address_scope(
        &self,
        role: ResolvedValueRole,
        ordinal: u32,
    ) -> Result<Option<DeviceReusableAddressScope>, VNextError> {
        let binding = self
            .bindings()
            .iter()
            .find(|b| b.role() == role && b.ordinal() == ordinal)
            .ok_or_else(|| invalid_operation("topology references unknown binding"))?;
        let mut aggregate = DeviceReusableAddressScope::Plan;
        for component in binding.storage().components() {
            let Some(scope) = self.resource_reusable_address_scope(component.resource_id())? else {
                return Ok(None);
            };
            aggregate = merge_reusable_address_scope(aggregate, scope)?;
        }
        Ok(Some(aggregate))
    }
    fn workspace_reusable_scope(
        &self,
        workspace: ReusableExecutionWorkspaceAddress,
    ) -> Result<Option<DeviceReusableAddressScope>, VNextError> {
        let resource = self
            .workspace_resource(workspace)
            .ok_or_else(|| invalid_operation("topology references absent workspace"))?;
        self.resource_reusable_address_scope(resource)
    }
    fn scratch_reusable_address_scope(
        &self,
    ) -> Result<Option<DeviceReusableAddressScope>, VNextError> {
        self.workspace_reusable_scope(ReusableExecutionWorkspaceAddress::Scratch)
    }
    fn binding_workspace_reusable_address_scope(
        &self,
    ) -> Result<Option<DeviceReusableAddressScope>, VNextError> {
        self.workspace_reusable_scope(ReusableExecutionWorkspaceAddress::Binding)
    }
    fn persistent_workspace_reusable_address_scope(
        &self,
    ) -> Result<Option<DeviceReusableAddressScope>, VNextError> {
        self.workspace_reusable_scope(ReusableExecutionWorkspaceAddress::Persistent)
    }
    fn reusable_address_scope(
        &self,
        values: &[ReusableExecutionValueAddress],
        workspaces: &[ReusableExecutionWorkspaceAddress],
    ) -> Result<Option<DeviceReusableAddressScope>, VNextError> {
        if values.len() != self.bindings().len()
            || values.iter().enumerate().any(|(index, value)| {
                values[..index]
                    .iter()
                    .any(|prior| prior.identity() == value.identity())
            })
            || self.bindings().iter().any(|binding| {
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
}
