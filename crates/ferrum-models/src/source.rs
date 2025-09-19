//! Model source resolution

use ferrum_types::Result;

/// Model source resolver
#[derive(Debug, Clone, Default)]
pub struct DefaultModelSourceResolver;

impl DefaultModelSourceResolver {
    pub fn new() -> Self {
        Self::default()
    }
}

/// Weight format enumeration  
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightFormat {
    SafeTensors,
    GGUF,
    Pickle,
}
