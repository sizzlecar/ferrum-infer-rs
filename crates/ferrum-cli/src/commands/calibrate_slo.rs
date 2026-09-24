//! Real, bounded manual calibration. This is not a serving benchmark.
use crate::config::CliConfig;
use clap::{Args, ValueEnum};
use ferrum_types::{FerrumError, Result};
use std::path::PathBuf;

mod driver;
mod http_inputs;
mod inputs;
mod manifest;
mod paths;
mod reference;
mod report;
#[cfg(test)]
mod selected_tests;
mod sharegpt;
mod startup;
#[cfg(test)]
mod tests;
mod validation;

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum CalibrationStartupUsage {
    Run,
    Serve,
}

#[derive(Args)]
#[command(
    after_help = "Uses real PlanRuntime owners with explicit Observe/CompleteRequests and credited output. Schema 1 accepts already rendered inputs; schema 2 replays a pinned ShareGPT selection through product Chat conversion. Optional reference phases persist discovered identities before fresh singleton trials and publish from the original training cut. An explicit fixed reference policy creates independent requests; training and held-out source limits stay unchanged. The selected_whole_wave_v1 (profile6/source1), selected_independent_attention_v2 (profile7/source2), and selected_work_support_v1 (profile8/source3) validation sources instead run fresh training, freeze the fitted model, collect independent residual cohorts, and evaluate held-out requests against the sealed imported model; collect any reference separately. Work-support keeps family schema2 and excludes only output_budget_sum from statistical support; request authority and terminal categories remain intact. Model revision and family version are separate identities. Never relabel old profile/source headers. Defaults, min_samples, residual quantile and TTL are unchanged. Every request completes its own declared output policy. The report distinguishes retrospective held-out cost checks from pre-submission forecasts. It is not an SLO or serving-performance certificate. Existing output files are never overwritten."
)]
pub struct CalibrateSloCommand {
    /// Registered model alias, source directory or GGUF artifact.
    pub model: String,
    #[command(flatten)]
    pub product_sources: crate::source_resolver::ProductSourceArgs,
    /// Version 1 rendered-input or version 2 frozen-ShareGPT manifest with exact cohorts.
    #[arg(long)]
    pub manifest: PathBuf,
    /// Explicit Observe/CompleteRequests/credited SLO configuration.
    #[arg(long)]
    pub slo_config: PathBuf,
    #[arg(long, default_value = "auto")]
    pub backend: String,
    #[arg(long)]
    pub numerical_profile: Option<ferrum_types::NumericalExecutionPolicy>,
    #[arg(long)]
    pub kv_dtype: Option<String>,
    #[arg(long)]
    pub gpu_devices: Option<String>,
    #[arg(long, default_value_t = 0.9)]
    pub gpu_memory_utilization: f32,
    /// Fixed native runtime memory budget, shared by every calibration cohort.
    #[arg(long)]
    pub runtime_memory_budget_bytes: Option<std::num::NonZeroUsize>,
    /// Apply the target product's startup defaults; capacity stays fixed for all cases.
    #[arg(long, value_enum)]
    pub startup_usage: CalibrationStartupUsage,
    /// New bounded raw JSONL file (includes failures and actual row identities).
    #[arg(long)]
    pub observations: PathBuf,
    /// New JSON report, including the immutable validation cut and effective config.
    #[arg(long)]
    pub out: PathBuf,
    /// Also export original frozen ShareGPT Chat bodies to a new directory.
    /// Runs before normal collection; never applies the independent reference
    /// output policy. One body per source index, at most 256 MiB in total.
    #[arg(long)]
    pub export_http_inputs: Option<PathBuf>,
}

pub async fn execute(cmd: CalibrateSloCommand, config: CliConfig) -> Result<()> {
    let manifest = manifest::load(&cmd.manifest)?;
    if cmd.export_http_inputs.is_some() && manifest.sharegpt.is_none() {
        return Err(FerrumError::config(
            "--export-http-inputs requires schema 2 frozen ShareGPT Chat inputs",
        ));
    }
    paths::validate(paths::outputs(&cmd, &manifest))?;
    let mut artifacts = report::Artifacts::create(&cmd.observations, &cmd.out, &manifest)?;
    let prepared = startup::prepare(&cmd, &config, &manifest).await;
    let (mut session, provenance, inputs) = match prepared {
        Ok(value) => value,
        Err(error) => {
            artifacts.finish(None, None, Some(&error))?;
            return Err(error);
        }
    };
    let mut summary = report::Summary::default();
    let result = tokio::time::timeout(
        std::time::Duration::from_millis(manifest.protocol.total_timeout_ms.get()),
        async {
            if let Some(directory) = &cmd.export_http_inputs {
                let receipt = http_inputs::export(&inputs, &manifest, directory)?;
                artifacts
                    .record(&serde_json::json!({"kind":"http_inputs_export", "receipt":receipt}))?;
            }
            driver::collect(
                &mut session,
                &manifest,
                &inputs,
                &mut artifacts,
                &mut summary,
            )
            .await
        },
    )
    .await
    .map_err(|_| {
        FerrumError::resource_exhausted(
            "calibration deadline exceeded; no further wave will be submitted",
        )
    })
    .and_then(|value| value);
    // The session retains an in-flight owner across a dropped step waiter.
    // Shutdown is always attempted before returning an error or publishing a report.
    let shutdown = tokio::time::timeout(
        std::time::Duration::from_millis(manifest.protocol.shutdown_timeout_ms.get()),
        session.shutdown(),
    )
    .await
    .map_err(|_| {
        FerrumError::resource_exhausted(
            "calibration shutdown deadline exceeded; resource reconciliation is unconfirmed",
        )
    })
    .and_then(|value| value);
    let error = finish_outcome(result, shutdown, &mut summary);
    artifacts.finish(Some(provenance), Some(summary), error.as_ref())?;
    if let Some(error) = error {
        return Err(error);
    }
    println!("Calibration observations and held-out cost report written. Serving SLOs and future-route coverage are not certified by this report.");
    Ok(())
}

fn finish_outcome(
    collection: Result<()>,
    shutdown: Result<()>,
    summary: &mut report::Summary,
) -> Option<FerrumError> {
    summary.collection_error = collection.as_ref().err().map(ToString::to_string);
    summary.shutdown = Some(report::ShutdownEvidence {
        completed: shutdown.is_ok(),
        error: shutdown.as_ref().err().map(ToString::to_string),
    });
    match (collection, shutdown) {
        (Ok(()), Ok(())) => None,
        (Err(collection), Err(shutdown)) => Some(FerrumError::internal(format!(
            "calibration failed: {collection}; shutdown also failed, resource reconciliation is unconfirmed: {shutdown}"
        ))),
        (Err(error), Ok(())) | (Ok(()), Err(error)) => Some(error),
    }
}
