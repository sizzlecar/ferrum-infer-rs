//! Bind an executor to the immutable sources supplied to the product builder.
//!
//! This checks composition consistency, not device execution authorization or
//! the executor's complete runtime policy. Source-independent custom executors
//! retain the existing extension path.

use ferrum_execution::{model_executor::ExecutionPlanningModel, ModelExecutor};
use ferrum_models::vnext::{DefinedProductionModel, ProductionModelSourceBundle};
use ferrum_runtime::{ModelArtifactSourceRole, ResolvedModelPlan};
use ferrum_types::{EngineConfig, FerrumError, NumericalExecutionPolicy, Result};
use std::sync::Arc;

pub(super) fn validate(
    config: &EngineConfig,
    sources: Option<&Arc<ProductionModelSourceBundle>>,
    defined: Option<&Arc<DefinedProductionModel>>,
    executor: &dyn ModelExecutor,
) -> Result<()> {
    if let Some(defined) = defined {
        if !sources.is_some_and(|sources| Arc::ptr_eq(defined.sources(), sources)) {
            return Err(FerrumError::model(
                "defined product model and builder sources do not share one source lease",
            ));
        }
    }
    let Some(sources) = sources else {
        return Ok(());
    };
    let reported = executor.bound_resolved_sources().ok_or_else(|| {
        FerrumError::model("executor did not report its bound product model sources")
    })?;
    if reported != sources.resolved_sources() {
        return Err(FerrumError::model(
            "executor bound model sources differ from the builder's semantic, tokenizer, or weight source",
        ));
    }
    if executor.execution_planning_model() == ExecutionPlanningModel::ExecutorManaged {
        if let NumericalExecutionPolicy::Require(profile) = &config.numerical_execution {
            return Err(FerrumError::unsupported(format!(
                "numerical profile {profile} requires a registered vNext model package"
            )));
        }
    }
    let Some(plan) = executor.resolved_model_plan() else {
        if defined.is_some() {
            return Err(FerrumError::model(
                "executor for a defined product model did not expose its authoritative ResolvedModelPlan",
            ));
        }
        return Ok(());
    };
    validate_plan_sources(sources, plan)?;
    if let Some(defined) = defined {
        validate_defined_family(defined, plan)?;
    }
    Ok(())
}

fn validate_plan_sources(
    sources: &ProductionModelSourceBundle,
    plan: &ResolvedModelPlan,
) -> Result<()> {
    let parts = plan.parts();
    // A wrapper may override its source report. Its actual typed plan must
    // still bind the same roles, rather than relying on that report alone.
    if &parts.resolved_sources != sources.resolved_sources() {
        return Err(FerrumError::model(
            "executor resolved plan sources differ from the builder's product sources",
        ));
    }
    let config = sources
        .fingerprint(ModelArtifactSourceRole::Semantic, "config.json")
        .ok_or_else(|| FerrumError::model("product source lost its semantic config fingerprint"))?;
    if parts.config.source_file != "config.json" || parts.config.sha256 != config.sha256 {
        return Err(FerrumError::model(
            "executor resolved plan does not bind the frozen semantic config.json",
        ));
    }
    // Parse only the retained bytes. Do not resolve through the built-in
    // registry: a custom family may own its own trusted plan registration.
    let metadata = ferrum_models::vnext::external_metadata_id_from_bytes(
        sources.config_json(),
        "frozen semantic config.json",
    )?;
    if parts.external_metadata_id != metadata {
        return Err(FerrumError::model(format!(
            "executor resolved plan metadata {} differs from frozen semantic metadata {metadata}",
            parts.external_metadata_id,
        )));
    }
    Ok(())
}

fn validate_defined_family(
    defined: &DefinedProductionModel,
    plan: &ResolvedModelPlan,
) -> Result<()> {
    let parts = plan.parts();
    let expected = defined.prepare(parts.numerical_execution.selected_profile())?;
    if expected.family() != &parts.prepared_family {
        return Err(FerrumError::model(
            "executor prepared family differs from the builder's defined product model at the selected numerical profile",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests;
