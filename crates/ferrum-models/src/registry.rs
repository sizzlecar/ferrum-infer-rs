//! Model registry

use ferrum_types::{Architecture, Result};
use std::collections::HashMap;

/// Model registry
#[derive(Debug, Clone, Default)]
pub struct DefaultModelRegistry {
    models: HashMap<String, Architecture>,
}

impl DefaultModelRegistry {
    pub fn new() -> Self {
        Self::default()
    }
}
