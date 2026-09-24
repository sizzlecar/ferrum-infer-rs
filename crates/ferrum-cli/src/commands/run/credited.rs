//! Direct CLI transport for already encoded, leased UTF-8 output.
use ferrum_interfaces::{
    engine::LlmInferenceEngine,
    output_credit::LeasedOutput,
    output_flow::{CreditedOutputSession, OutputCompletion, OutputProjectionContract},
    InferenceRequestContext,
};
use ferrum_types::{FerrumError, InferenceRequest, Result, TokenUsage};
use futures::StreamExt;
use std::io::Write;
use std::sync::Arc;

pub(super) fn validate_options(one_shot: bool, format: super::OutputFormat) -> Result<()> {
    if !one_shot || format != super::OutputFormat::Text {
        return Err(FerrumError::unsupported(
            "credited CLI output currently requires --prompt and --output-format text; interactive history and JSONL need separate output contracts",
        ));
    }
    Ok(())
}

/// Only the bounded terminal evidence serialized by `write_evidence` is
/// supported here. Lifecycle sinks and replay/resource bundles still need their
/// own credited contracts; silently accepting them would drop requested data.
pub(super) fn validate_observability(
    config: &crate::observability_product::ProductObservabilityConfig,
) -> Result<()> {
    use ferrum_types::ObservabilityProfileDetail as Detail;
    if config.synthetic_no_weight_enabled() {
        return Err(FerrumError::unsupported(
            "credited output requires real inference, not synthetic observability",
        ));
    }
    if config.request_dump_dir.is_some()
        || config.memory_profile_jsonl.is_some()
        || config.scheduler_trace_jsonl.is_some()
        || !matches!(
            config.profile_detail,
            Detail::Off | Detail::Basic | Detail::Latency | Detail::Kernel
        )
    {
        return Err(FerrumError::unsupported(
            "credited CLI output supports basic, latency, and kernel terminal profiles; resource/lifecycle sinks and replay/debug/verify/full bundles require separate output contracts",
        ));
    }
    config
        .core
        .validate()
        .map_err(FerrumError::invalid_parameter)
}

pub(super) struct CreditedRunStats {
    pub usage: TokenUsage,
    pub visible_frames: usize,
    pub elapsed: std::time::Duration,
}

/// This one-shot helper owns shutdown on success, admission error, and writer
/// failure. The completion is inspected under its lease and then released;
/// only fixed-size counters escape into the CLI's statistics line.
pub(super) async fn execute_one_shot(
    engine: &dyn LlmInferenceEngine,
    request: InferenceRequest,
    context: InferenceRequestContext,
    suppress_text: bool,
    observability: &crate::observability_product::ProductObservabilityConfig,
) -> Result<CreditedRunStats> {
    if suppress_text {
        execute_observed(
            engine,
            request,
            context,
            std::io::sink(),
            std::io::stderr(),
            Some(observability),
        )
        .await
    } else {
        execute_observed(
            engine,
            request,
            context,
            std::io::stdout(),
            std::io::stderr(),
            Some(observability),
        )
        .await
    }
}

async fn execute_observed<W, E>(
    engine: &dyn LlmInferenceEngine,
    mut request: InferenceRequest,
    context: InferenceRequestContext,
    output: W,
    errors: E,
    observability: Option<&crate::observability_product::ProductObservabilityConfig>,
) -> Result<CreditedRunStats>
where
    W: Write + Send + 'static,
    E: Write + Send + 'static,
{
    request.stream = true;
    if let Some(config) = observability {
        request.evidence_request.capture_engine_token_timing =
            config.profile_detail.captures_engine_token_timing();
        request.evidence_request.capture_prompt_token_ids = config.request_dump_dir.is_some();
    }
    let started = std::time::Instant::now();
    let result = async {
        let session = engine
            .infer_credited_stream(
                request,
                context,
                Arc::new(OutputProjectionContract::cli_text()),
            )
            .await?;
        let written = write_with(session, output, errors).await?;
        let elapsed = started.elapsed();
        let completion = Arc::new(written.completion);
        if let Some(config) = observability.filter(|config| config.enabled()) {
            let config = config.clone();
            let retained = completion.clone();
            tokio::task::spawn_blocking(move || write_evidence(&config, retained, elapsed))
                .await
                .map_err(|e| FerrumError::internal(e.to_string()))??;
        }
        match completion.payload() {
            OutputCompletion::Succeeded { usage, .. } => Ok(CreditedRunStats {
                usage: usage.clone(),
                visible_frames: written.visible_frames,
                elapsed,
            }),
            OutputCompletion::Failed(error) => Err(FerrumError::backend(error.message())),
        }
    }
    .await;
    let shutdown = engine.shutdown().await;
    match (result, shutdown) {
        (Ok(stats), Ok(())) => Ok(stats),
        (Ok(_), Err(error)) | (Err(error), _) => Err(error),
    }
}

pub(super) struct CreditedRunResult {
    pub completion: LeasedOutput<OutputCompletion>,
    pub visible_frames: usize,
}

/// One blocking writer owns the session. A frame's lease remains live through
/// write_all and flush; no intermediate output queue or payload copy is added.
/// Cancelling an async waiter cannot interrupt a blocking OS write. The held
/// frame remains charged until that write actually returns.
async fn write_with<W, E>(
    session: CreditedOutputSession,
    mut output: W,
    mut errors: E,
) -> Result<CreditedRunResult>
where
    W: Write + Send + 'static,
    E: Write + Send + 'static,
{
    let runtime = tokio::runtime::Handle::current();
    tokio::task::spawn_blocking(move || {
        let CreditedOutputSession {
            mut frames,
            completion,
        } = session;
        let mut visible_frames = 0usize;
        let mut terminal_seen = false;
        while let Some(frame) = runtime.block_on(frames.next()) {
            let terminal = frame.metadata().terminal;
            let wire = frame.wire();
            if wire.credit().events == 0 || wire.payload().capacity() > wire.credit().bytes {
                return Err(FerrumError::internal(
                    "credited CLI frame exceeds its leased wire capacity",
                ));
            }
            let bytes = wire.payload().as_slice();
            let target: &mut dyn Write = if terminal { &mut errors } else { &mut output };
            if !bytes.is_empty() {
                target
                    .write_all(bytes)
                    .and_then(|()| target.flush())
                    .map_err(|error| {
                        FerrumError::backend(format!("write credited output: {error}"))
                    })?;
                if !terminal {
                    visible_frames = visible_frames.checked_add(1).ok_or_else(|| {
                        FerrumError::internal("credited output frame count overflow")
                    })?;
                }
            }
            // Drop only after the underlying writer is finished with the bytes.
            drop(frame);
            if terminal {
                terminal_seen = true;
                break;
            }
        }
        if !terminal_seen {
            return Err(FerrumError::backend(
                "credited output stream closed without a terminal frame",
            ));
        }
        let completion = runtime.block_on(completion).map_err(|_| {
            FerrumError::backend("credited output owner closed without a completion")
        })?;
        Ok(CreditedRunResult {
            completion,
            visible_frames,
        })
    })
    .await
    .map_err(|error| FerrumError::internal(format!("credited output writer failed: {error}")))?
}

fn write_evidence(
    config: &crate::observability_product::ProductObservabilityConfig,
    completion: Arc<LeasedOutput<OutputCompletion>>,
    elapsed: std::time::Duration,
) -> Result<()> {
    use ferrum_interfaces::output_flow::{CreditedExecutionProfile, CreditedPromptEvidence};
    if let Some(path) = config.profile_jsonl.as_ref() {
        let record = CreditedExecutionProfile::new(
            completion.clone(),
            ferrum_types::ProfileEntrypoint::Run,
            config.model.clone(),
            "cli_text",
            config.profile_detail,
            u64::try_from(elapsed.as_micros()).unwrap_or(u64::MAX),
            crate::observability_product::runtime_preset_hash(config),
            Default::default(),
        )
        .map_err(|e| FerrumError::internal(e.to_string()))?;
        ferrum_bench_core::write_jsonl_owned_record(path, record)
            .map_err(|e| FerrumError::io(e.to_string()))?;
    }
    if let Some(root) = config
        .request_dump_dir
        .as_ref()
        .filter(|_| matches!(completion.payload(), OutputCompletion::Succeeded { .. }))
    {
        let dir = root.join(completion.request_id().to_string());
        std::fs::create_dir_all(&dir).map_err(|e| FerrumError::io(e.to_string()))?;
        ferrum_bench_core::write_json_owned_record(
            &dir.join("prompt_token_ids.json"),
            CreditedPromptEvidence {
                completion,
                model: config.model.clone(),
            },
        )
        .map_err(|e| FerrumError::io(e.to_string()))?;
    }
    Ok(())
}

#[cfg(test)]
async fn execute_with<W: Write + Send + 'static, E: Write + Send + 'static>(
    engine: &dyn LlmInferenceEngine,
    request: InferenceRequest,
    context: InferenceRequestContext,
    output: W,
    errors: E,
) -> Result<CreditedRunStats> {
    execute_observed(engine, request, context, output, errors, None).await
}
#[cfg(test)]
mod tests;
