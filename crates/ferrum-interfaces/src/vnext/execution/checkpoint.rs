use super::{
    invalid_plan, BTreeSet, BufferUsage, Deserialize, NodeId, PlanNode, ProgramValueId,
    ResolvedTensorLayout, ResolvedTensorSpec, ResolvedValueBinding, ResolvedValueRole, ResourceId,
    Serialize, TensorAccess, VNextError,
};
use crate::vnext::{CompletionReadbackRequest, HostTransferLayout};

/// Explicit semantic activations that must remain readable at the terminal
/// completion fence. The empty default preserves the normal execution plan.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CompletionRetentionSpec {
    values: BTreeSet<ProgramValueId>,
}

impl CompletionRetentionSpec {
    pub fn new(values: BTreeSet<ProgramValueId>) -> Self {
        Self { values }
    }

    pub fn insert(&mut self, value_id: ProgramValueId) -> bool {
        self.values.insert(value_id)
    }

    pub fn values(&self) -> &BTreeSet<ProgramValueId> {
        &self.values
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

/// Immutable plan evidence for one semantic activation retained until the
/// terminal completion fence. Callers never reconstruct this binding from raw
/// node and resource strings.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RetainedCompletionValue {
    value_id: ProgramValueId,
    producer_node_id: NodeId,
    output_ordinal: u32,
    resource_id: ResourceId,
    logical_offset_bytes: u64,
    tensor: ResolvedTensorSpec,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RetainedCompletionValueWire {
    value_id: ProgramValueId,
    producer_node_id: NodeId,
    output_ordinal: u32,
    resource_id: ResourceId,
    logical_offset_bytes: u64,
    tensor: ResolvedTensorSpec,
}

impl<'de> Deserialize<'de> for RetainedCompletionValue {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let wire = RetainedCompletionValueWire::deserialize(deserializer)?;
        Self::new(
            wire.value_id,
            wire.producer_node_id,
            wire.output_ordinal,
            wire.resource_id,
            wire.logical_offset_bytes,
            wire.tensor,
        )
        .map_err(serde::de::Error::custom)
    }
}

impl RetainedCompletionValue {
    fn new(
        value_id: ProgramValueId,
        producer_node_id: NodeId,
        output_ordinal: u32,
        resource_id: ResourceId,
        logical_offset_bytes: u64,
        tensor: ResolvedTensorSpec,
    ) -> Result<Self, VNextError> {
        let byte_len = tensor.minimum_storage_bytes()?;
        if logical_offset_bytes.checked_add(byte_len).is_none() {
            return Err(invalid_plan(
                "retained completion value range overflows u64",
            ));
        }
        Ok(Self {
            value_id,
            producer_node_id,
            output_ordinal,
            resource_id,
            logical_offset_bytes,
            tensor,
        })
    }

    pub fn value_id(&self) -> &ProgramValueId {
        &self.value_id
    }

    pub fn producer_node_id(&self) -> &NodeId {
        &self.producer_node_id
    }

    pub const fn output_ordinal(&self) -> u32 {
        self.output_ordinal
    }

    pub fn resource_id(&self) -> &ResourceId {
        &self.resource_id
    }

    pub const fn logical_offset_bytes(&self) -> u64 {
        self.logical_offset_bytes
    }

    pub fn tensor(&self) -> &ResolvedTensorSpec {
        &self.tensor
    }

    pub fn readback_request(
        &self,
        participant_index: u32,
        output_layout: HostTransferLayout,
    ) -> Result<CompletionReadbackRequest, VNextError> {
        if !matches!(self.tensor.layout(), ResolvedTensorLayout::Contiguous) {
            return Err(invalid_plan(
                "completion readback currently requires contiguous retained storage",
            ));
        }
        if output_layout.element_type() != self.tensor.element_type() {
            return Err(invalid_plan(
                "completion readback element type differs from retained activation",
            ));
        }
        if output_layout.byte_len()? > self.tensor.minimum_storage_bytes()? {
            return Err(invalid_plan(
                "completion readback exceeds retained activation capacity",
            ));
        }
        CompletionReadbackRequest::new(
            self.producer_node_id.clone(),
            participant_index,
            self.resource_id.clone(),
            self.logical_offset_bytes,
            output_layout,
        )
    }
}

pub(super) fn resolve_retained_completion_values(
    nodes: &[PlanNode],
    spec: &CompletionRetentionSpec,
) -> Result<Vec<RetainedCompletionValue>, VNextError> {
    let mut retained = Vec::with_capacity(spec.values.len());
    for value_id in &spec.values {
        let matches = nodes
            .iter()
            .enumerate()
            .flat_map(|(node_index, node)| {
                node.values()
                    .iter()
                    .map(move |binding| (node_index, node, binding))
            })
            .filter(|(_, _, binding)| {
                binding.value_id() == value_id && binding.role() == ResolvedValueRole::Output
            })
            .collect::<Vec<_>>();
        let [(producer_index, producer, binding)] = matches.as_slice() else {
            return Err(invalid_plan(format!(
                "completion retention value `{value_id}` must identify exactly one plan-node output"
            )));
        };
        if binding.usage() != BufferUsage::Activations {
            return Err(invalid_plan(format!(
                "completion retention value `{value_id}` is not an activation"
            )));
        }
        let [component] = binding.storage().components() else {
            return Err(invalid_plan(format!(
                "completion retention value `{value_id}` does not use one physical component"
            )));
        };
        let retained_end = component
            .offset_bytes()
            .checked_add(binding.tensor().minimum_storage_bytes()?)
            .ok_or_else(|| invalid_plan("retained completion range overflows u64"))?;
        reject_later_overlapping_write(
            nodes,
            *producer_index,
            binding,
            component.resource_id(),
            component.offset_bytes(),
            retained_end,
        )?;
        retained.push(RetainedCompletionValue::new(
            value_id.clone(),
            producer.id().clone(),
            binding.ordinal(),
            component.resource_id().clone(),
            component.offset_bytes(),
            binding.tensor().clone(),
        )?);
    }
    Ok(retained)
}

fn reject_later_overlapping_write(
    nodes: &[PlanNode],
    producer_index: usize,
    retained_binding: &ResolvedValueBinding,
    retained_resource: &ResourceId,
    retained_start: u64,
    retained_end: u64,
) -> Result<(), VNextError> {
    for (node_index, node) in nodes.iter().enumerate().skip(producer_index) {
        for binding in node.values() {
            let is_target = node_index == producer_index
                && binding.role() == retained_binding.role()
                && binding.ordinal() == retained_binding.ordinal()
                && binding.value_id() == retained_binding.value_id();
            if is_target
                || !matches!(
                    binding.access(),
                    TensorAccess::Write | TensorAccess::ReadWrite
                )
            {
                continue;
            }
            for component in binding.storage().components() {
                if component.resource_id() != retained_resource {
                    continue;
                }
                let write_end = component
                    .offset_bytes()
                    .checked_add(component.length_bytes())
                    .ok_or_else(|| invalid_plan("plan value write range overflows u64"))?;
                if component.offset_bytes() < retained_end && retained_start < write_end {
                    return Err(invalid_plan(format!(
                        "completion retention value `{}` is overwritten by node `{}` after its producer",
                        retained_binding.value_id(),
                        node.id()
                    )));
                }
            }
        }
    }
    Ok(())
}
