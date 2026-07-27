use super::{
    CapabilityCatalog, DeviceDescriptor, ExecutionPlan, PlanBuildRequest, RuntimePolicy, VNextError,
};

/// Minimal trusted view consumed by resource admission and operation dispatch.
/// Product resolution may wrap this with tokenizer, sampling, source, and API
/// policy, but those concerns do not enter the device execution boundary.
pub trait ExecutablePlanView {
    fn execution_plan(&self) -> &ExecutionPlan;
    fn device(&self) -> &DeviceDescriptor;
    fn capabilities(&self) -> &CapabilityCatalog;
}

/// Owned executable produced directly by the planner for a concrete runtime
/// composition root. It is intentionally smaller than a product-level
/// `ResolvedModelPlan`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutablePlan {
    plan: ExecutionPlan,
    capabilities: CapabilityCatalog,
}

impl ExecutablePlan {
    pub fn new(plan: ExecutionPlan, capabilities: CapabilityCatalog) -> Result<Self, VNextError> {
        let catalog_fingerprint = capabilities.fingerprint()?;
        if plan.payload().device_id() != &capabilities.device().id
            || plan.payload().device_runtime_implementation_fingerprint()
                != capabilities.device().runtime_implementation_fingerprint
            || plan.payload().capability_catalog_fingerprint() != catalog_fingerprint
        {
            return Err(VNextError::InvalidExecutionPlan {
                reason: "executable plan differs from its device capability catalog".to_owned(),
            });
        }
        Ok(Self { plan, capabilities })
    }

    pub fn into_parts(self) -> (ExecutionPlan, CapabilityCatalog) {
        (self.plan, self.capabilities)
    }
}

impl ExecutablePlanView for ExecutablePlan {
    fn execution_plan(&self) -> &ExecutionPlan {
        &self.plan
    }

    fn device(&self) -> &DeviceDescriptor {
        self.capabilities.device()
    }

    fn capabilities(&self) -> &CapabilityCatalog {
        &self.capabilities
    }
}

/// Pure planner boundary. Execution consumes the immutable plan and performs
/// no capability/backend selection in the token loop.
pub trait ExecutionPlanner: Send + Sync {
    type Policy: RuntimePolicy;

    fn build_plan(
        &self,
        request: PlanBuildRequest<'_, Self::Policy>,
    ) -> Result<ExecutionPlan, VNextError>;
}
