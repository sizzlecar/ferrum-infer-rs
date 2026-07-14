use super::{
    invalid_plan, CapabilityCatalog, ExecutionPlan, PlanBuildRequest, PlanNodeResolution,
    PlanSchemaVersion, PreparedModelFamily, RuntimePolicy, UnvalidatedExecutionPlan, VNextError,
    EXECUTION_PLAN_SCHEMA,
};

impl UnvalidatedExecutionPlan {
    pub fn schema(&self) -> PlanSchemaVersion {
        self.payload.schema
    }

    pub fn revalidate<P: RuntimePolicy>(
        self,
        family: &PreparedModelFamily,
        capabilities: &CapabilityCatalog,
        policy: &P,
        node_resolutions: Vec<PlanNodeResolution>,
    ) -> Result<ExecutionPlan, VNextError> {
        if self.payload.schema != EXECUTION_PLAN_SCHEMA {
            return Err(VNextError::UnsupportedPlanSchema {
                expected_major: EXECUTION_PLAN_SCHEMA.major,
                expected_minor: EXECUTION_PLAN_SCHEMA.minor,
                actual_major: self.payload.schema.major,
                actual_minor: self.payload.schema.minor,
            });
        }
        let rebuilt = ExecutionPlan::build(PlanBuildRequest::new(
            family,
            capabilities,
            policy,
            node_resolutions,
        )?)?;
        let untrusted_payload =
            serde_json::to_value(&self.payload).map_err(|error| VNextError::Serialization {
                context: "serialize unvalidated execution plan payload",
                message: error.to_string(),
            })?;
        let rebuilt_payload =
            serde_json::to_value(&rebuilt.payload).map_err(|error| VNextError::Serialization {
                context: "serialize rebuilt execution plan payload",
                message: error.to_string(),
            })?;
        if untrusted_payload != rebuilt_payload {
            return Err(invalid_plan(
                "untrusted plan differs from a semantic rebuild against current dependencies",
            ));
        }
        if rebuilt.plan_hash != self.plan_hash {
            return Err(VNextError::PlanHashMismatch {
                expected: rebuilt.plan_hash.to_string(),
                actual: self.plan_hash.to_string(),
            });
        }
        Ok(rebuilt)
    }
}
