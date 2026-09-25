//! Versioned live structured population capture. V1 remains unchanged.
mod prepared;
pub(in crate::continuous_engine::inner) use prepared::{
    PreparedRowBindingV2, PreparedStructuredFactsV2,
};
