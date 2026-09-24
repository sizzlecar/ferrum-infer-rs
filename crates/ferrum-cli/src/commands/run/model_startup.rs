//! Shared source and numerical preparation before either product engine build.
use super::*;
use std::sync::Arc;

pub(in crate::commands) struct PreparedProductSource {
    pub input: crate::source_resolver::ProductEngineInput,
    pub defined_model: Option<Arc<ferrum_models::vnext::DefinedProductionModel>>,
}

pub(in crate::commands) async fn prepare_product_source(
    model: &str,
    sources: &crate::source_resolver::ProductSourceArgs,
    config: &CliConfig,
    numerical_profile: Option<&ferrum_types::NumericalExecutionPolicy>,
    effective_runtime: &RuntimeConfigSnapshot,
) -> Result<PreparedProductSource> {
    let cache_dir = crate::source_resolver::hf_cache_dir(config);
    let resolved = crate::source_resolver::resolve_model_source_with_product_sources(
        model,
        &cache_dir,
        crate::source_resolver::DownloadPolicy::AutoDownload,
        None,
        sources,
    )
    .await?;
    let mut input = resolved.into_product_engine_input();
    input.engine_config.numerical_execution = config.resolve_numerical_execution(numerical_profile);
    apply_kv_dtype_override(
        &mut input.engine_config,
        crate::runtime_env::runtime_snapshot_value(effective_runtime, "FERRUM_KV_DTYPE"),
    )?;
    let defined_model = crate::source_resolver::define_registered_product_model(
        input.model_sources.as_ref(),
        &input.engine_config.numerical_execution,
        input.engine_config.kv_cache.dtype,
    )?;
    Ok(PreparedProductSource {
        input,
        defined_model,
    })
}
