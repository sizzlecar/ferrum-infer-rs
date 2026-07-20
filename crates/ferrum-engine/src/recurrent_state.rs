//! In-memory recurrent-state manager used for lifecycle and integration tests.
//!
//! This manager owns metadata handles and capacity accounting only. Backend
//! implementations that carry actual CUDA/Metal tensors should provide their
//! own handle type and downcast payload.

use ferrum_interfaces::{
    RecurrentStateHandle, RecurrentStateHandleStats, RecurrentStateManager,
    RecurrentStateManagerStats, RecurrentStateSpec,
};
use ferrum_types::{FerrumError, RequestId, Result};
use parking_lot::Mutex;
use std::{
    any::Any,
    collections::HashMap,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc,
    },
    time::Instant,
};

#[derive(Debug, Clone)]
pub struct InMemoryRecurrentStateConfig {
    pub total_memory_bytes: usize,
    pub total_batch_slots: usize,
}

impl Default for InMemoryRecurrentStateConfig {
    fn default() -> Self {
        Self {
            total_memory_bytes: usize::MAX,
            total_batch_slots: usize::MAX,
        }
    }
}

#[derive(Debug, Clone)]
pub struct InMemoryRecurrentStateHandle {
    spec: RecurrentStateSpec,
    memory_bytes: usize,
    cache_id: String,
    valid: Arc<AtomicBool>,
    created_at: Instant,
}

impl InMemoryRecurrentStateHandle {
    fn new(spec: RecurrentStateSpec) -> Self {
        let memory_bytes = spec.estimated_memory_bytes();
        let cache_id = format!("recurrent-state-{}", spec.request_id);
        Self {
            spec,
            memory_bytes,
            cache_id,
            valid: Arc::new(AtomicBool::new(true)),
            created_at: Instant::now(),
        }
    }

    fn invalidate(&self) {
        self.valid.store(false, Ordering::Relaxed);
    }
}

impl RecurrentStateHandle for InMemoryRecurrentStateHandle {
    fn request_id(&self) -> RequestId {
        self.spec.request_id.clone()
    }

    fn device(&self) -> ferrum_types::Device {
        self.spec.device.clone()
    }

    fn num_layers(&self) -> usize {
        self.spec.num_layers
    }

    fn state_bytes(&self) -> usize {
        self.memory_bytes
    }

    fn clone_handle(&self) -> Result<Arc<dyn RecurrentStateHandle>> {
        Ok(Arc::new(self.clone()))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn stats(&self) -> RecurrentStateHandleStats {
        RecurrentStateHandleStats {
            memory_bytes: self.memory_bytes,
            state_tensors: self.spec.tensors.len(),
            batch_slots: self.spec.max_batch_slots,
            last_access: self.created_at,
        }
    }

    fn is_valid(&self) -> bool {
        self.valid.load(Ordering::Relaxed)
    }

    fn cache_id(&self) -> String {
        self.cache_id.clone()
    }
}

#[derive(Debug)]
pub struct InMemoryRecurrentStateManager {
    config: InMemoryRecurrentStateConfig,
    handles: Mutex<HashMap<RequestId, Arc<InMemoryRecurrentStateHandle>>>,
    allocation_count: AtomicU64,
    allocation_failures: AtomicU64,
}

impl InMemoryRecurrentStateManager {
    pub fn new(config: InMemoryRecurrentStateConfig) -> Self {
        Self {
            config,
            handles: Mutex::new(HashMap::new()),
            allocation_count: AtomicU64::new(0),
            allocation_failures: AtomicU64::new(0),
        }
    }

    fn used_memory_bytes_locked(
        handles: &HashMap<RequestId, Arc<InMemoryRecurrentStateHandle>>,
    ) -> usize {
        handles.values().map(|handle| handle.memory_bytes).sum()
    }

    fn used_batch_slots_locked(
        handles: &HashMap<RequestId, Arc<InMemoryRecurrentStateHandle>>,
    ) -> usize {
        handles
            .values()
            .map(|handle| handle.spec.max_batch_slots)
            .sum()
    }
}

#[async_trait::async_trait]
impl RecurrentStateManager for InMemoryRecurrentStateManager {
    async fn allocate(&self, spec: &RecurrentStateSpec) -> Result<Arc<dyn RecurrentStateHandle>> {
        let mut handles = self.handles.lock();
        if handles.contains_key(&spec.request_id) {
            self.allocation_failures.fetch_add(1, Ordering::Relaxed);
            return Err(FerrumError::already_exists(format!(
                "recurrent state already allocated for {}",
                spec.request_id
            )));
        }

        let projected_memory =
            Self::used_memory_bytes_locked(&handles).saturating_add(spec.estimated_memory_bytes());
        let projected_slots =
            Self::used_batch_slots_locked(&handles).saturating_add(spec.max_batch_slots);
        if projected_memory > self.config.total_memory_bytes
            || projected_slots > self.config.total_batch_slots
        {
            self.allocation_failures.fetch_add(1, Ordering::Relaxed);
            return Err(FerrumError::resource_exhausted(
                "insufficient recurrent-state capacity",
            ));
        }

        let handle = Arc::new(InMemoryRecurrentStateHandle::new(spec.clone()));
        handles.insert(spec.request_id.clone(), handle.clone());
        self.allocation_count.fetch_add(1, Ordering::Relaxed);
        Ok(handle)
    }

    async fn deallocate(&self, request_id: RequestId) -> Result<()> {
        if let Some(handle) = self.handles.lock().remove(&request_id) {
            handle.invalidate();
        }
        Ok(())
    }

    fn can_allocate(&self, spec: &RecurrentStateSpec) -> bool {
        let handles = self.handles.lock();
        if handles.contains_key(&spec.request_id) {
            return false;
        }
        Self::used_memory_bytes_locked(&handles).saturating_add(spec.estimated_memory_bytes())
            <= self.config.total_memory_bytes
            && Self::used_batch_slots_locked(&handles).saturating_add(spec.max_batch_slots)
                <= self.config.total_batch_slots
    }

    fn get_handle(&self, request_id: RequestId) -> Option<Arc<dyn RecurrentStateHandle>> {
        self.handles
            .lock()
            .get(&request_id)
            .map(|handle| handle.clone() as Arc<dyn RecurrentStateHandle>)
    }

    fn list_handles(&self) -> Vec<(RequestId, Arc<dyn RecurrentStateHandle>)> {
        self.handles
            .lock()
            .iter()
            .map(|(request_id, handle)| {
                (
                    request_id.clone(),
                    handle.clone() as Arc<dyn RecurrentStateHandle>,
                )
            })
            .collect()
    }

    fn stats(&self) -> RecurrentStateManagerStats {
        let handles = self.handles.lock();
        RecurrentStateManagerStats {
            total_memory_bytes: self.config.total_memory_bytes,
            used_memory_bytes: Self::used_memory_bytes_locked(&handles),
            active_states: handles.len(),
            active_state_tensors: handles
                .values()
                .map(|handle| handle.spec.tensors.len())
                .sum(),
            total_batch_slots: self.config.total_batch_slots,
            used_batch_slots: Self::used_batch_slots_locked(&handles),
            allocation_count: self.allocation_count.load(Ordering::Relaxed),
            allocation_failures: self.allocation_failures.load(Ordering::Relaxed),
            eviction_count: 0,
        }
    }

    async fn reset(&self) -> Result<()> {
        let mut handles = self.handles.lock();
        for handle in handles.values() {
            handle.invalidate();
        }
        handles.clear();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrum_interfaces::{RecurrentStateManager, RecurrentStateTensorSpec};
    use ferrum_types::{DataType, Device};

    fn spec(request_id: RequestId) -> RecurrentStateSpec {
        RecurrentStateSpec {
            request_id,
            num_layers: 2,
            tensors: vec![
                RecurrentStateTensorSpec::new(0, "delta_state", vec![8, 16], DataType::BF16),
                RecurrentStateTensorSpec::new(1, "delta_state", vec![8, 16], DataType::BF16),
            ],
            device: Device::CPU,
            max_batch_slots: 1,
        }
    }

    #[tokio::test]
    async fn in_memory_manager_allocates_and_deallocates_state() {
        let manager = InMemoryRecurrentStateManager::new(InMemoryRecurrentStateConfig {
            total_memory_bytes: 4096,
            total_batch_slots: 4,
        });
        let request_id = RequestId::new();
        let spec = spec(request_id.clone());

        let handle = manager.allocate(&spec).await.unwrap();

        assert_eq!(handle.request_id(), request_id);
        assert_eq!(handle.state_bytes(), 512);
        assert!(handle.is_valid());
        assert_eq!(manager.stats().active_states, 1);
        assert_eq!(manager.stats().used_memory_bytes, 512);

        manager.deallocate(request_id.clone()).await.unwrap();

        assert!(!handle.is_valid());
        assert!(manager.get_handle(request_id).is_none());
        assert_eq!(manager.stats().active_states, 0);
    }

    #[tokio::test]
    async fn in_memory_manager_rejects_duplicate_and_capacity_overcommit() {
        let manager = InMemoryRecurrentStateManager::new(InMemoryRecurrentStateConfig {
            total_memory_bytes: 512,
            total_batch_slots: 1,
        });
        let request_id = RequestId::new();
        let first = spec(request_id.clone());
        let second = spec(RequestId::new());

        manager.allocate(&first).await.unwrap();

        let duplicate = manager.allocate(&first).await.unwrap_err();
        assert!(matches!(duplicate, FerrumError::AlreadyExists { .. }));

        let overcommit = manager.allocate(&second).await.unwrap_err();
        assert!(matches!(overcommit, FerrumError::ResourceExhausted { .. }));
        assert_eq!(manager.stats().allocation_failures, 2);
    }

    #[tokio::test]
    async fn in_memory_manager_reset_invalidates_all_handles() {
        let manager = InMemoryRecurrentStateManager::new(InMemoryRecurrentStateConfig {
            total_memory_bytes: 4096,
            total_batch_slots: 4,
        });
        let handle_a = manager.allocate(&spec(RequestId::new())).await.unwrap();
        let handle_b = manager.allocate(&spec(RequestId::new())).await.unwrap();

        manager.reset().await.unwrap();

        assert!(!handle_a.is_valid());
        assert!(!handle_b.is_valid());
        assert!(manager.list_handles().is_empty());
    }
}
