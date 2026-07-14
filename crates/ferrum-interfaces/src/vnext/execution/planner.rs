use super::{ExecutionPlan, PlanBuildRequest, RuntimePolicy, VNextError};

/// Pure planner boundary. Execution consumes the immutable plan and performs
/// no capability/backend selection in the token loop.
pub trait ExecutionPlanner: Send + Sync {
    type Policy: RuntimePolicy;

    fn build_plan(
        &self,
        request: PlanBuildRequest<'_, Self::Policy>,
    ) -> Result<ExecutionPlan, VNextError>;
}
