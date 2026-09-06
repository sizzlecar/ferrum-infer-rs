//! Change-based regression planning. A plan is not execution evidence or release approval.
pub mod contracts;
pub mod dependency_change;
pub mod distribution;
mod impact;
pub mod model_basic;
pub mod model_schedule;
pub mod model_stop;
mod model_stop_evidence;
pub mod model_tasks;
pub mod model_tool;
pub mod numerics;
pub mod performance;
mod selection;
pub mod source_change;
pub mod submission;
mod types;
mod version_change;

pub use impact::analyze_paths;
pub use selection::plan;
pub use types::*;
pub use version_change::coordinated_version_paths;
