//! Shared, bounded resource evidence for local benchmark clients.
//!
//! Parsers retain their existing image-module export for compatibility. Process
//! traversal and cadence have no async runtime or GPU-library dependency.

pub use crate::images::resource_usage::{
    nvidia_memory_csv, proc_memory_bytes, proc_stat, GpuIdentity, ProcessIdentity, SampledMemory,
};
mod gpu;
mod procfs;
pub use gpu::GpuMemorySampler;
pub mod timing;
pub use procfs::ProcessSampler;
use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ProcessScope {
    Process,
    ProcessTree,
}

#[derive(Debug, Clone, Serialize)]
pub struct ProcessSamplingConfig {
    pub process_scope: ProcessScope,
    pub maximum_processes: usize,
    pub maximum_tree_nodes: usize,
    pub maximum_proc_input_bytes: usize,
}

impl Default for ProcessSamplingConfig {
    fn default() -> Self {
        Self {
            process_scope: ProcessScope::ProcessTree,
            maximum_processes: 4096,
            maximum_tree_nodes: 65_536,
            maximum_proc_input_bytes: 16 * 1024 * 1024,
        }
    }
}
