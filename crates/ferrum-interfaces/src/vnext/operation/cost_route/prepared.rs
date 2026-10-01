//! Provider-owned immutable cost inputs, compiled for one validated plan node.
use super::*;
use std::{any::Any, sync::Arc};

/// Static semantics only: no request rows, resource residency or execution
/// authority can be obtained from this preparation request.
pub struct OperationCostPreparationRequest<'a> {
    node: &'a PlanNode,
}

impl<'a> OperationCostPreparationRequest<'a> {
    pub(in crate::vnext::operation) fn new(node: &'a PlanNode) -> Self {
        Self { node }
    }

    pub fn node_id(&self) -> &NodeId {
        self.node.id()
    }

    pub fn operation_id(&self) -> &OperationId {
        self.node.operation_id()
    }

    pub fn attributes(&self) -> &BTreeMap<AttributeId, SemanticValue> {
        self.node.attributes()
    }

    pub fn bindings(&self) -> &[ResolvedValueBinding] {
        self.node.values()
    }
}

/// One immutable, provider-typed numerical recipe per bound node. This value
/// is neither serialized nor accepted as a physical resource/route proof.
/// Providers retain only plan-derived metadata here; current work, residency
/// and selected launch quantities must still come from each cost query.
#[derive(Clone)]
pub struct PreparedOperationCostData(Arc<dyn Any + Send + Sync>);

impl PreparedOperationCostData {
    pub fn new<T: Any + Send + Sync>(value: T) -> Self {
        Self(Arc::new(value))
    }

    pub(super) fn get<T: Any>(&self) -> Option<&T> {
        self.0.downcast_ref()
    }
}
