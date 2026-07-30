use super::{
    invalid_plan, CapabilityCatalog, CompletionRetentionSpec, Deserialize, ExecutionPlan,
    PlanBuildRequest, PlanNodeResolution, PlanSchemaVersion, PreparedModelFamily, RuntimePolicy,
    TrustedExecutionWeightPlan, UnvalidatedExecutionPlan, UnvalidatedExecutionPlanWire, VNextError,
    EXECUTION_PLAN_SCHEMA, MAX_EXECUTION_PLAN_WIRE_BYTES,
};

#[derive(Deserialize)]
struct ExecutionPlanSchemaEnvelope {
    payload: ExecutionPlanSchemaHeader,
}

#[derive(Deserialize)]
struct ExecutionPlanSchemaHeader {
    schema: PlanSchemaVersion,
}

pub(super) fn validate_execution_plan_wire_size(
    wire_size: usize,
    context: &'static str,
) -> Result<(), VNextError> {
    if wire_size > MAX_EXECUTION_PLAN_WIRE_BYTES {
        return Err(VNextError::Serialization {
            context,
            message: format!(
                "execution plan wire size {wire_size} exceeds limit {MAX_EXECUTION_PLAN_WIRE_BYTES}"
            ),
        });
    }
    Ok(())
}

impl ExecutionPlan {
    pub fn to_json(&self) -> Result<Vec<u8>, VNextError> {
        let bytes = serde_json::to_vec(self).map_err(|error| VNextError::Serialization {
            context: "serialize execution plan",
            message: error.to_string(),
        })?;
        validate_execution_plan_wire_size(bytes.len(), "serialize execution plan")?;
        Ok(bytes)
    }

    pub fn decode_untrusted(bytes: &[u8]) -> Result<UnvalidatedExecutionPlan, VNextError> {
        const CONTEXT: &str = "decode untrusted execution plan";
        validate_execution_plan_wire_size(bytes.len(), CONTEXT)?;
        let header =
            serde_json::from_slice::<ExecutionPlanSchemaEnvelope>(bytes).map_err(|error| {
                VNextError::Serialization {
                    context: CONTEXT,
                    message: error.to_string(),
                }
            })?;
        if header.payload.schema != EXECUTION_PLAN_SCHEMA {
            return Err(VNextError::UnsupportedPlanSchema {
                expected_major: EXECUTION_PLAN_SCHEMA.major,
                expected_minor: EXECUTION_PLAN_SCHEMA.minor,
                actual_major: header.payload.schema.major,
                actual_minor: header.payload.schema.minor,
            });
        }
        serde_json::from_slice::<UnvalidatedExecutionPlanWire>(bytes)
            .map(UnvalidatedExecutionPlan::from)
            .map_err(|error| VNextError::Serialization {
                context: CONTEXT,
                message: error.to_string(),
            })
    }

    pub fn from_json_validated<P: RuntimePolicy>(
        bytes: &[u8],
        family: &PreparedModelFamily,
        capabilities: &CapabilityCatalog,
        policy: &P,
        node_resolutions: Vec<PlanNodeResolution>,
    ) -> Result<Self, VNextError> {
        Self::decode_untrusted(bytes)?.revalidate(family, capabilities, policy, node_resolutions)
    }

    pub fn from_json_validated_with_completion_retention<P: RuntimePolicy>(
        bytes: &[u8],
        family: &PreparedModelFamily,
        capabilities: &CapabilityCatalog,
        policy: &P,
        node_resolutions: Vec<PlanNodeResolution>,
        completion_retention: CompletionRetentionSpec,
    ) -> Result<Self, VNextError> {
        Self::decode_untrusted(bytes)?.revalidate_with_completion_retention(
            family,
            capabilities,
            policy,
            node_resolutions,
            completion_retention,
        )
    }

    pub fn from_json_validated_with_execution_weights<P: RuntimePolicy>(
        bytes: &[u8],
        family: &PreparedModelFamily,
        capabilities: &CapabilityCatalog,
        policy: &P,
        node_resolutions: Vec<PlanNodeResolution>,
        completion_retention: CompletionRetentionSpec,
        execution_weights: TrustedExecutionWeightPlan,
    ) -> Result<Self, VNextError> {
        Self::decode_untrusted(bytes)?.revalidate_with_execution_weights(
            family,
            capabilities,
            policy,
            node_resolutions,
            completion_retention,
            execution_weights,
        )
    }

    pub fn validate_against<P: RuntimePolicy>(
        &self,
        family: &PreparedModelFamily,
        capabilities: &CapabilityCatalog,
        policy: &P,
        node_resolutions: &[PlanNodeResolution],
    ) -> Result<(), VNextError> {
        self.validate_against_with_completion_retention(
            family,
            capabilities,
            policy,
            node_resolutions,
            CompletionRetentionSpec::default(),
        )
    }

    pub fn validate_against_with_completion_retention<P: RuntimePolicy>(
        &self,
        family: &PreparedModelFamily,
        capabilities: &CapabilityCatalog,
        policy: &P,
        node_resolutions: &[PlanNodeResolution],
        completion_retention: CompletionRetentionSpec,
    ) -> Result<(), VNextError> {
        let rebuilt = ExecutionPlan::build(
            PlanBuildRequest::new(family, capabilities, policy, node_resolutions.to_vec())?
                .with_execution_weights(self.trusted_execution_weights.clone())?
                .with_completion_retention(completion_retention)?,
        )?;
        if rebuilt.operation_registry_authority != self.operation_registry_authority {
            return Err(invalid_plan(
                "execution plan belongs to a different operation runtime registry",
            ));
        }
        if &rebuilt != self {
            return Err(invalid_plan(
                "execution plan is not identical to its semantic rebuild",
            ));
        }
        Ok(())
    }
}

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
        self.revalidate_with_completion_retention(
            family,
            capabilities,
            policy,
            node_resolutions,
            CompletionRetentionSpec::default(),
        )
    }

    pub fn revalidate_with_completion_retention<P: RuntimePolicy>(
        self,
        family: &PreparedModelFamily,
        capabilities: &CapabilityCatalog,
        policy: &P,
        node_resolutions: Vec<PlanNodeResolution>,
        completion_retention: CompletionRetentionSpec,
    ) -> Result<ExecutionPlan, VNextError> {
        let execution_weights = TrustedExecutionWeightPlan::identity(family)?;
        self.revalidate_with_execution_weights(
            family,
            capabilities,
            policy,
            node_resolutions,
            completion_retention,
            execution_weights,
        )
    }

    pub fn revalidate_with_execution_weights<P: RuntimePolicy>(
        self,
        family: &PreparedModelFamily,
        capabilities: &CapabilityCatalog,
        policy: &P,
        node_resolutions: Vec<PlanNodeResolution>,
        completion_retention: CompletionRetentionSpec,
        execution_weights: TrustedExecutionWeightPlan,
    ) -> Result<ExecutionPlan, VNextError> {
        if self.payload.schema != EXECUTION_PLAN_SCHEMA {
            return Err(VNextError::UnsupportedPlanSchema {
                expected_major: EXECUTION_PLAN_SCHEMA.major,
                expected_minor: EXECUTION_PLAN_SCHEMA.minor,
                actual_major: self.payload.schema.major,
                actual_minor: self.payload.schema.minor,
            });
        }
        let rebuilt = ExecutionPlan::build(
            PlanBuildRequest::new(family, capabilities, policy, node_resolutions)?
                .with_execution_weights(execution_weights)?
                .with_completion_retention(completion_retention)?,
        )?;
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
