//! KV coordinator for managing KV cache handles

use ferrum_types::Result;

/// KV coordinator for managing cache handles
#[derive(Debug)]
pub struct KvCoordinator {
    // Coordinator implementation
}

impl KvCoordinator {
    pub fn new() -> Self {
        Self {}
    }
}

impl Default for KvCoordinator {
    fn default() -> Self {
        Self::new()
    }
}
