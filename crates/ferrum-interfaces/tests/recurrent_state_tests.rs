use async_trait::async_trait;
use ferrum_interfaces::{
    RecurrentStateHandle, RecurrentStateHandleStats, RecurrentStateManager,
    RecurrentStateManagerStats, RecurrentStateSpec, RecurrentStateTensorSpec,
};
use ferrum_types::{DataType, Device, FerrumError, RequestId, Result};
use std::{
    any::Any,
    collections::HashMap,
    sync::{Arc, Mutex},
    time::Instant,
};

#[derive(Debug, Clone)]
struct MockRecurrentStateHandle {
    spec: RecurrentStateSpec,
    memory_bytes: usize,
    cache_id: String,
    valid: bool,
    last_access: Instant,
}

impl MockRecurrentStateHandle {
    fn new(spec: RecurrentStateSpec) -> Self {
        let memory_bytes = spec.estimated_memory_bytes();
        let cache_id = format!("mock-recurrent-state-{}", spec.request_id);
        Self {
            spec,
            memory_bytes,
            cache_id,
            valid: true,
            last_access: Instant::now(),
        }
    }
}

impl RecurrentStateHandle for MockRecurrentStateHandle {
    fn request_id(&self) -> RequestId {
        self.spec.request_id.clone()
    }

    fn device(&self) -> Device {
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
            last_access: self.last_access,
        }
    }

    fn is_valid(&self) -> bool {
        self.valid
    }

    fn cache_id(&self) -> String {
        self.cache_id.clone()
    }
}

#[derive(Debug)]
struct MockRecurrentStateManager {
    total_memory_bytes: usize,
    total_batch_slots: usize,
    handles: Mutex<HashMap<RequestId, Arc<MockRecurrentStateHandle>>>,
    allocation_count: Mutex<u64>,
    allocation_failures: Mutex<u64>,
}

impl MockRecurrentStateManager {
    fn new(total_memory_bytes: usize, total_batch_slots: usize) -> Self {
        Self {
            total_memory_bytes,
            total_batch_slots,
            handles: Mutex::new(HashMap::new()),
            allocation_count: Mutex::new(0),
            allocation_failures: Mutex::new(0),
        }
    }

    fn used_memory_bytes(&self) -> usize {
        self.handles
            .lock()
            .unwrap()
            .values()
            .map(|handle| handle.memory_bytes)
            .sum()
    }

    fn used_batch_slots(&self) -> usize {
        self.handles
            .lock()
            .unwrap()
            .values()
            .map(|handle| handle.spec.max_batch_slots)
            .sum()
    }
}

#[async_trait]
impl RecurrentStateManager for MockRecurrentStateManager {
    async fn allocate(&self, spec: &RecurrentStateSpec) -> Result<Arc<dyn RecurrentStateHandle>> {
        if !self.can_allocate(spec) {
            *self.allocation_failures.lock().unwrap() += 1;
            return Err(FerrumError::resource_exhausted(
                "insufficient recurrent-state capacity",
            ));
        }

        let handle = Arc::new(MockRecurrentStateHandle::new(spec.clone()));
        self.handles
            .lock()
            .unwrap()
            .insert(spec.request_id.clone(), handle.clone());
        *self.allocation_count.lock().unwrap() += 1;
        Ok(handle)
    }

    async fn deallocate(&self, request_id: RequestId) -> Result<()> {
        self.handles.lock().unwrap().remove(&request_id);
        Ok(())
    }

    fn can_allocate(&self, spec: &RecurrentStateSpec) -> bool {
        self.used_memory_bytes() + spec.estimated_memory_bytes() <= self.total_memory_bytes
            && self.used_batch_slots() + spec.max_batch_slots <= self.total_batch_slots
    }

    fn get_handle(&self, request_id: RequestId) -> Option<Arc<dyn RecurrentStateHandle>> {
        self.handles
            .lock()
            .unwrap()
            .get(&request_id)
            .map(|handle| handle.clone() as Arc<dyn RecurrentStateHandle>)
    }

    fn list_handles(&self) -> Vec<(RequestId, Arc<dyn RecurrentStateHandle>)> {
        self.handles
            .lock()
            .unwrap()
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
        let handles = self.handles.lock().unwrap();
        let used_memory_bytes = handles.values().map(|handle| handle.memory_bytes).sum();
        let active_state_tensors = handles
            .values()
            .map(|handle| handle.spec.tensors.len())
            .sum();
        let used_batch_slots = handles
            .values()
            .map(|handle| handle.spec.max_batch_slots)
            .sum();
        RecurrentStateManagerStats {
            total_memory_bytes: self.total_memory_bytes,
            used_memory_bytes,
            active_states: handles.len(),
            active_state_tensors,
            total_batch_slots: self.total_batch_slots,
            used_batch_slots,
            allocation_count: *self.allocation_count.lock().unwrap(),
            allocation_failures: *self.allocation_failures.lock().unwrap(),
            eviction_count: 0,
        }
    }

    async fn reset(&self) -> Result<()> {
        self.handles.lock().unwrap().clear();
        Ok(())
    }
}

fn recurrent_spec(request_id: RequestId) -> RecurrentStateSpec {
    RecurrentStateSpec {
        request_id,
        num_layers: 2,
        tensors: vec![
            RecurrentStateTensorSpec::new(0, "delta_state", vec![8, 16]),
            RecurrentStateTensorSpec::new(1, "delta_state", vec![8, 16]),
        ],
        dtype: DataType::BF16,
        device: Device::CPU,
        max_batch_slots: 1,
    }
}

#[test]
fn recurrent_state_spec_estimates_state_bytes() {
    let spec = recurrent_spec(RequestId::new());

    assert_eq!(spec.estimated_memory_bytes(), 2 * 8 * 16 * 2);
}

#[tokio::test]
async fn recurrent_state_manager_lifecycle_is_separate_from_kv() {
    let manager = MockRecurrentStateManager::new(4096, 4);
    let request_id = RequestId::new();
    let spec = recurrent_spec(request_id.clone());

    assert!(manager.can_allocate(&spec));
    let handle = manager.allocate(&spec).await.unwrap();

    assert_eq!(handle.request_id(), request_id);
    assert_eq!(handle.device(), Device::CPU);
    assert_eq!(handle.num_layers(), 2);
    assert_eq!(handle.state_bytes(), 512);
    assert!(handle.is_valid());
    assert!(handle.cache_id().starts_with("mock-recurrent-state-"));
    assert!(handle.as_any().is::<MockRecurrentStateHandle>());

    let cloned = handle.clone_handle().unwrap();
    assert_eq!(cloned.state_bytes(), handle.state_bytes());

    let fetched = manager.get_handle(request_id.clone()).unwrap();
    assert_eq!(fetched.state_bytes(), handle.state_bytes());
    assert_eq!(manager.list_handles().len(), 1);

    let stats = manager.stats();
    assert_eq!(stats.active_states, 1);
    assert_eq!(stats.active_state_tensors, 2);
    assert_eq!(stats.used_memory_bytes, 512);
    assert_eq!(stats.used_batch_slots, 1);
    assert_eq!(stats.allocation_count, 1);

    manager.deallocate(request_id.clone()).await.unwrap();
    assert!(manager.get_handle(request_id).is_none());
    assert_eq!(manager.stats().active_states, 0);
}

#[tokio::test]
async fn recurrent_state_manager_rejects_capacity_overcommit() {
    let manager = MockRecurrentStateManager::new(256, 4);
    let spec = recurrent_spec(RequestId::new());

    assert!(!manager.can_allocate(&spec));
    let err = manager.allocate(&spec).await.unwrap_err();
    assert!(matches!(err, FerrumError::ResourceExhausted { .. }));
    assert_eq!(manager.stats().allocation_failures, 1);
}

#[tokio::test]
async fn recurrent_state_manager_reset_drops_all_handles() {
    let manager = MockRecurrentStateManager::new(4096, 4);
    let spec_a = recurrent_spec(RequestId::new());
    let spec_b = recurrent_spec(RequestId::new());

    manager.allocate(&spec_a).await.unwrap();
    manager.allocate(&spec_b).await.unwrap();
    assert_eq!(manager.stats().active_states, 2);

    manager.reset().await.unwrap();
    assert_eq!(manager.stats().active_states, 0);
    assert!(manager.list_handles().is_empty());
}
