//! CLI utility functions

use ferrum_types::Result;
use std::io;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

/// Setup logging based on verbosity level.
///
/// `suppress_chat_template_warnings` is used by the interactive `run` UX:
/// template fallback diagnostics are useful in server logs, but they should
/// not be printed in the middle of a user-facing REPL turn.
pub fn setup_logging(
    verbose: bool,
    quiet: bool,
    suppress_chat_template_warnings: bool,
) -> Result<()> {
    let log_level = if quiet {
        tracing::Level::ERROR
    } else if verbose {
        tracing::Level::DEBUG
    } else {
        tracing::Level::WARN
    };

    // Build filter: suppress noisy warnings unless verbose. Default level
    // is WARN, but the model-load and weight-loader namespaces stay at
    // INFO so users see per-layer progress during the (10-60s) load.
    // Without this, `ferrum run /path/to/30b-moe` looks frozen between
    // "Loading..." and the first decode token.
    let filter = if verbose {
        EnvFilter::new(log_level.to_string())
    } else {
        EnvFilter::try_from_default_env().unwrap_or_else(|_| {
            let mut filter = format!(
                "{},\
                 tokenizers=error,\
                 ferrum_models=info,\
                 ferrum_quantization=info,\
                 ferrum_engine::registry=info",
                log_level
            );
            if suppress_chat_template_warnings {
                filter.push_str(",ferrum_server::chat_template=error");
            }
            EnvFilter::new(filter)
        })
    };

    tracing_subscriber::registry()
        .with(filter)
        .with(tracing_subscriber::fmt::layer().with_writer(io::stderr))
        .init();

    Ok(())
}
