//! Strict, model-independent ordinary linear LoRA import and managed composition.
//!
//! Importing inspects immutable mappings and metadata only. A model must bind
//! every target before payload authentication or materialization is permitted.
mod compose;
mod config;
mod import;

pub use compose::{LinearAdapterBinding, ManagedLinearAdapter, PreparedLinearAdapter};
pub use config::{LinearAdapterScaling, LinearAdapterSource};
pub use import::{ImportedLinearAdapter, LinearAdapterMatrixView, LowRankMatrix};

use ferrum_types::{FerrumError, Result};

fn invalid(reason: impl std::fmt::Display) -> FerrumError {
    FerrumError::model(format!("linear LoRA: {reason}"))
}

#[cfg(test)]
mod tests;
