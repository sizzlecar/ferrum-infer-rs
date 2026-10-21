use std::{any::Any, sync::Arc};

use super::{invalid_resource, PlanRuntimeResources, TrustedPlanRuntimeBinding};
use crate::{
    CompiledTensorPorts, DeviceRuntime, ExecutionPlan, HostMemoryAccount, ManagedHostTensor,
    ResourceWorkShape, TensorWorkEvidence, VNextError,
};

/// Product composition injects the host parent once. Work submissions cannot
/// substitute a different account or pass arbitrary compiled port descriptors.
pub struct TensorExecutionBinding<R: DeviceRuntime> {
    runtime: TrustedPlanRuntimeBinding<R>,
    ports: CompiledTensorPorts,
    host: HostMemoryAccount,
}

impl<R: DeviceRuntime> PlanRuntimeResources<R> {
    pub fn bind_tensor_execution(
        self: &Arc<Self>,
        plan: &ExecutionPlan,
        host: HostMemoryAccount,
    ) -> Result<TensorExecutionBinding<R>, VNextError> {
        let runtime = self.trusted_runtime_binding()?;
        if runtime.plan_hash() != plan.plan_hash() || !runtime.work_limits().is_tensor() {
            return Err(invalid_resource(
                "tensor execution plan differs from its live resource root",
            ));
        }
        let ports =
            plan.payload().tensor_ports().cloned().ok_or_else(|| {
                invalid_resource("tensor execution requires compiled tensor ports")
            })?;
        Ok(TensorExecutionBinding {
            runtime,
            ports,
            host,
        })
    }
}

impl<R: DeviceRuntime> TensorExecutionBinding<R> {
    pub fn ports(&self) -> &CompiledTensorPorts {
        &self.ports
    }
    pub fn host_account(&self) -> &HostMemoryAccount {
        &self.host
    }
    pub fn runtime_binding(&self) -> &TrustedPlanRuntimeBinding<R> {
        &self.runtime
    }

    pub fn supports_managed_io(&self) -> bool {
        self.runtime.runtime().supports_managed_tensor_io()
    }

    /// Builds upload descriptors from the exact immutable owners sealed into
    /// this root's work. Dispatch repeats the identity/range check before encode.
    pub fn input_uploads(
        &self,
        work: &ResourceWorkShape,
    ) -> Result<Vec<crate::SubmissionWaveInputUpload>, VNextError> {
        let _lifecycle = self
            .runtime
            .resources
            .read_lifecycle("prepare tensor uploads")?;
        self.runtime.validate_work_root(work)?;
        let evidence = work
            .tensor_evidence()
            .ok_or_else(|| invalid_resource("tensor upload requires tensor evidence"))?;
        self.ports
            .inputs()
            .iter()
            .map(|port| {
                let input = evidence
                    .inputs()
                    .iter()
                    .find(|input| input.value_id() == port.value_id())
                    .ok_or_else(|| invalid_resource("bound tensor input port is missing"))?;
                let layout = crate::HostTransferLayout::new(
                    port.tensor().element_type(),
                    port.tensor().minimum_storage_bytes()?
                        / port.tensor().element_type().size_bytes(),
                )?;
                crate::SubmissionWaveInputUpload::managed(
                    port.node_id().clone(),
                    port.ordinal(),
                    layout,
                    input.bytes().clone(),
                )
            })
            .collect()
    }

    pub fn bind_inputs(
        &self,
        inputs: Vec<ManagedHostTensor>,
    ) -> Result<ResourceWorkShape, VNextError> {
        let _lifecycle = self
            .runtime
            .resources
            .read_lifecycle("bind tensor inputs")?;
        if inputs.iter().any(|input| !self.host.owns(input.bytes())) {
            return Err(invalid_resource(
                "tensor input belongs to a different host memory parent",
            ));
        }
        let root: Arc<dyn Any + Send + Sync> = self.runtime.resources.clone();
        let evidence =
            TensorWorkEvidence::bind(root, self.runtime.evidence(), &self.ports, inputs)?;
        Ok(ResourceWorkShape::from_tensor_evidence(evidence))
    }
}

impl<R: DeviceRuntime> TrustedPlanRuntimeBinding<R> {
    pub(super) fn validate_work_root(&self, work: &ResourceWorkShape) -> Result<(), VNextError> {
        if let Some(tensor) = work.tensor_evidence() {
            let root: Arc<dyn Any + Send + Sync> = self.resources.clone();
            if !self.work_limits().is_tensor() || !tensor.belongs_to(&root) {
                return Err(invalid_resource(
                    "tensor work belongs to another plan runtime root",
                ));
            }
        } else if self.work_limits().is_tensor() {
            return Err(invalid_resource(
                "token work cannot enter a fixed tensor plan",
            ));
        }
        Ok(())
    }
}
