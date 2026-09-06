//! Change-based regression planning. A plan is not execution evidence or release approval.
mod impact;
mod selection;
mod types;
mod version_change;

pub use impact::analyze_paths;
pub use selection::plan;
pub use types::*;
pub use version_change::coordinated_version_paths;
