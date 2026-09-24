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
) -> Result<CreditedRunStats> {
    if suppress_text {
        execute_with(engine, request, context, std::io::sink(), std::io::stderr()).await
    } else {
        execute_with(
            engine,
            request,
            context,
            std::io::stdout(),
            std::io::stderr(),
        )
        .await
    }
}

async fn execute_with<W, E>(
    engine: &dyn LlmInferenceEngine,
    mut request: InferenceRequest,
    context: InferenceRequestContext,
    output: W,
    errors: E,
) -> Result<CreditedRunStats>
where
    W: Write + Send + 'static,
    E: Write + Send + 'static,
{
    request.stream = true;
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
        match written.completion.payload() {
            OutputCompletion::Succeeded { usage, .. } => Ok(CreditedRunStats {
                usage: usage.clone(),
                visible_frames: written.visible_frames,
                elapsed: started.elapsed(),
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

#[cfg(test)]
mod tests;
