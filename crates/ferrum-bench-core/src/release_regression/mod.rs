//! Change-based regression planning. A plan is not execution evidence or release approval.
mod impact;
mod selection;
mod types;

pub use impact::analyze_paths;
pub use selection::plan;
pub use types::*;
