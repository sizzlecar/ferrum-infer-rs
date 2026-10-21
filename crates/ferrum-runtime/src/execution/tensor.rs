//! Exact tensor ports and owned input evidence. Descriptors describe bytes;
//! only a live plan root can bind them into admitted work evidence.

use std::{
    any::Any,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::invalid_plan;
use crate::{
    HostMemoryClass, ManagedHostBytes, NodeId, PlanNode, PreparedModelFamily, ProgramValueId,
    ResolvedTensorLayout, ResolvedTensorSpec, ResolvedValueRole, TrustedPlanRuntimeEvidence,
    VNextError,
};

/// One compiler-derived boundary, including its first input consumer or output
/// producer. A public descriptor never grants allocation or submission rights.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompiledTensorPort {
    value_id: ProgramValueId,
    tensor: ResolvedTensorSpec,
    node_id: NodeId,
    ordinal: u32,
}

impl CompiledTensorPort {
    pub fn value_id(&self) -> &ProgramValueId {
        &self.value_id
    }
    pub fn tensor(&self) -> &ResolvedTensorSpec {
        &self.tensor
    }
    pub fn node_id(&self) -> &NodeId {
        &self.node_id
    }
    pub const fn ordinal(&self) -> u32 {
        self.ordinal
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompiledTensorPorts {
    inputs: Vec<CompiledTensorPort>,
    outputs: Vec<CompiledTensorPort>,
}

impl CompiledTensorPorts {
    pub fn inputs(&self) -> &[CompiledTensorPort] {
        &self.inputs
    }
    pub fn outputs(&self) -> &[CompiledTensorPort] {
        &self.outputs
    }

    pub(super) fn derive(
        family: &PreparedModelFamily,
        nodes: &[PlanNode],
    ) -> Result<Self, VNextError> {
        let ports = |ids: &[ProgramValueId], role| -> Result<Vec<CompiledTensorPort>, VNextError> {
            ids.iter().map(|id| {
                let mut matches = nodes.iter().flat_map(|node| node.values().iter().map(move |binding| (node, binding)))
                    .filter(|(_, binding)| binding.value_id() == id && binding.role() == role);
                let (node, binding) = matches.next().ok_or_else(|| invalid_plan("compiled tensor port has no matching node binding"))?;
                if binding.tensor().layout() != &ResolvedTensorLayout::Contiguous
                    || binding.tensor().minimum_storage_bytes()? == 0
                    || matches.any(|(_, other)| other.tensor() != binding.tensor() || other.storage() != binding.storage()) {
                    return Err(invalid_plan("fixed tensor ports require consistent contiguous exact tensor bindings"));
                }
                Ok(CompiledTensorPort { value_id: id.clone(), tensor: binding.tensor().clone(), node_id: node.id().clone(), ordinal: binding.ordinal() })
            }).collect()
        };
        Ok(Self {
            inputs: ports(family.program().inputs(), ResolvedValueRole::Input)?,
            outputs: ports(family.program().outputs(), ResolvedValueRole::Output)?,
        })
    }
}

/// An immutable typed host input. Its exact shape is checked again against the
/// compiled port by the root. Construction does not grant device capacity.
#[derive(Debug, Clone)]
pub struct ManagedHostTensor {
    value_id: ProgramValueId,
    tensor: ResolvedTensorSpec,
    bytes: ManagedHostBytes,
}

impl ManagedHostTensor {
    pub fn new(
        value_id: ProgramValueId,
        tensor: ResolvedTensorSpec,
        bytes: ManagedHostBytes,
    ) -> Result<Self, VNextError> {
        if tensor.layout() != &ResolvedTensorLayout::Contiguous
            || tensor.minimum_storage_bytes()?
                != u64::try_from(bytes.len())
                    .map_err(|_| invalid_plan("host tensor byte length exceeds u64"))?
            || bytes.is_empty()
            || bytes.class() != HostMemoryClass::InputPreprocess
        {
            return Err(invalid_plan(
                "tensor input requires an exact nonempty contiguous InputPreprocess allocation",
            ));
        }
        Ok(Self {
            value_id,
            tensor,
            bytes,
        })
    }
    pub fn value_id(&self) -> &ProgramValueId {
        &self.value_id
    }
    pub fn tensor(&self) -> &ResolvedTensorSpec {
        &self.tensor
    }
    pub fn bytes(&self) -> &ManagedHostBytes {
        &self.bytes
    }
}

/// Core-created proof of real owned inputs and planned outputs for one live
/// root. The root guard and input allocations live at least as long as work.
#[derive(Clone, Serialize)]
pub struct TensorWorkEvidence {
    fingerprint: String,
    plan: TrustedPlanRuntimeEvidence,
    #[serde(skip)]
    root: Arc<dyn Any + Send + Sync>,
    #[serde(skip)]
    inputs: Vec<ManagedHostTensor>,
}

impl std::fmt::Debug for TensorWorkEvidence {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TensorWorkEvidence")
            .field("fingerprint", &self.fingerprint)
            .field("plan", &self.plan)
            .finish()
    }
}
impl PartialEq for TensorWorkEvidence {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.root, &other.root) && self.fingerprint == other.fingerprint
    }
}
impl Eq for TensorWorkEvidence {}

impl TensorWorkEvidence {
    pub(crate) fn bind(
        root: Arc<dyn Any + Send + Sync>,
        plan: TrustedPlanRuntimeEvidence,
        ports: &CompiledTensorPorts,
        mut inputs: Vec<ManagedHostTensor>,
    ) -> Result<Self, VNextError> {
        inputs.sort_by(|a, b| a.value_id.cmp(&b.value_id));
        if inputs.len() != ports.inputs.len()
            || inputs
                .windows(2)
                .any(|pair| pair[0].value_id == pair[1].value_id)
        {
            return Err(invalid_plan("tensor input set is incomplete or duplicated"));
        }
        let mut digest = Sha256::new();
        digest.update(b"ferrum.tensor-work.v1\0");
        digest.update(plan.plan_hash().as_str().as_bytes());
        for port in &ports.inputs {
            let input = inputs
                .iter()
                .find(|input| input.value_id == port.value_id)
                .ok_or_else(|| invalid_plan("tensor input set differs from compiled ports"))?;
            if input.tensor != port.tensor {
                return Err(invalid_plan(
                    "tensor input axes, dtype or layout differ from compiled port",
                ));
            }
            digest.update(port.value_id.as_str().as_bytes());
        }
        static NEXT_BINDING: AtomicU64 = AtomicU64::new(1);
        let binding_id = NEXT_BINDING
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
            .map_err(|_| invalid_plan("tensor binding identity space exhausted"))?;
        digest.update(binding_id.to_le_bytes());
        Ok(Self {
            fingerprint: format!("{:x}", digest.finalize()),
            plan,
            root,
            inputs,
        })
    }
    pub fn fingerprint(&self) -> &str {
        &self.fingerprint
    }
    pub fn inputs(&self) -> &[ManagedHostTensor] {
        &self.inputs
    }
    pub(crate) fn belongs_to(&self, root: &Arc<dyn Any + Send + Sync>) -> bool {
        Arc::ptr_eq(&self.root, root)
    }
}
