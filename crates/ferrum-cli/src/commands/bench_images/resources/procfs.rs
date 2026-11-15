use super::{ImageProcessScope, ImageResourceSamplingConfig};
use ferrum_bench_core::resource_usage::{ProcessIdentity, ProcessSamplingConfig, ProcessScope};
use std::path::PathBuf;

pub(super) struct ProcessSampler(ferrum_bench_core::resource_usage::ProcessSampler);

impl ProcessSampler {
    pub(super) fn new(root: PathBuf, pid: u32) -> Self {
        Self(ferrum_bench_core::resource_usage::ProcessSampler::new(
            root, pid,
        ))
    }
    pub(super) fn identity(&self) -> Option<ProcessIdentity> {
        self.0.identity()
    }
    pub(super) fn sample(&mut self, config: &ImageResourceSamplingConfig) -> Result<u64, String> {
        self.0.sample(&ProcessSamplingConfig {
            process_scope: match config.process_scope {
                ImageProcessScope::Process => ProcessScope::Process,
                ImageProcessScope::ProcessTree => ProcessScope::ProcessTree,
            },
            maximum_processes: config.maximum_processes,
            maximum_tree_nodes: config.maximum_tree_nodes,
            maximum_proc_input_bytes: config.maximum_proc_input_bytes,
        })
    }
}
