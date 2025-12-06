//! CLI utility functions

use ferrum_types::Result;
use std::io;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

/// Setup logging based on verbosity level
pub fn setup_logging(verbose: bool, quiet: bool) -> Result<()> {
    let log_level = if quiet {
        tracing::Level::ERROR
    } else if verbose {
        tracing::Level::DEBUG
    } else {
        tracing::Level::WARN
    };

    // Build filter: suppress noisy metal executor warnings unless verbose
    let filter = if verbose {
        EnvFilter::new(log_level.to_string())
    } else {
        EnvFilter::try_from_default_env().unwrap_or_else(|_| {
            EnvFilter::new(format!(
                "{},ferrum_engine::metal::metal_executor=error",
                log_level
            ))
        })
    };

    tracing_subscriber::registry()
        .with(filter)
        .with(tracing_subscriber::fmt::layer().with_writer(io::stderr))
        .init();

    Ok(())
}
