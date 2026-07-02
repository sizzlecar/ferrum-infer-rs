//! Minimal in-memory native operator registry.

use std::collections::BTreeMap;
use std::path::PathBuf;

use ferrum_types::NativeOperatorBackend;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct NativeOperatorKey {
    pub operator: String,
    pub backend: NativeOperatorBackend,
}

impl NativeOperatorKey {
    pub fn new(operator: impl Into<String>, backend: NativeOperatorBackend) -> Self {
        Self {
            operator: operator.into(),
            backend,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeOperatorRegistration {
    pub manifest_path: PathBuf,
    pub artifact_path: PathBuf,
}

#[derive(Debug, Default, Clone)]
pub struct NativeOperatorRegistry {
    entries: BTreeMap<NativeOperatorKey, NativeOperatorRegistration>,
}

impl NativeOperatorRegistry {
    pub fn insert(
        &mut self,
        key: NativeOperatorKey,
        registration: NativeOperatorRegistration,
    ) -> Option<NativeOperatorRegistration> {
        self.entries.insert(key, registration)
    }

    pub fn get(&self, key: &NativeOperatorKey) -> Option<&NativeOperatorRegistration> {
        self.entries.get(key)
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}
