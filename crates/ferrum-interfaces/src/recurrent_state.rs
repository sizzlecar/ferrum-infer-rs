//! Recurrent-state cache contracts for state-space / hybrid models.
//!
//! Recurrent state is intentionally separate from KV cache. KV grows with
//! sequence length and is addressed through attention blocks; recurrent state is
//! compact per-layer state that can coexist with KV in hybrid models.

use ferrum_types::{DataType, Device, RequestId, Result};
use serde::{Deserialize, Serialize};
use std::{any::Any, sync::Arc, time::Instant};

/// A single recurrent-state tensor owned by a model layer.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecurrentStateTensorSpec {
    /// Layer index that owns this state tensor.
    pub layer_index: usize,
    /// Backend/model-local state name, for example `delta_state`.
    pub name: String,
    /// Tensor shape excluding any request/batch slot dimension.
    pub shape: Vec<usize>,
    /// Storage dtype for this state tensor.
    pub dtype: DataType,
}

impl RecurrentStateTensorSpec {
    pub fn new(
        layer_index: usize,
        name: impl Into<String>,
        shape: Vec<usize>,
        dtype: DataType,
    ) -> Self {
        Self {
            layer_index,
            name: name.into(),
            shape,
            dtype,
        }
    }

    pub fn checked_num_elements(&self) -> Option<usize> {
        self.shape
            .iter()
            .copied()
            .try_fold(1usize, usize::checked_mul)
    }

    pub fn num_elements(&self) -> usize {
        self.checked_num_elements().unwrap_or(usize::MAX)
    }
}

/// Allocation request for recurrent state.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecurrentStateSpec {
    /// Request ID this allocation is for.
    pub request_id: RequestId,
    /// Number of model layers that may carry recurrent state.
    pub num_layers: usize,
    /// Concrete state tensors to allocate.
    pub tensors: Vec<RecurrentStateTensorSpec>,
    /// Target device.
    pub device: Device,
    /// Number of request/batch slots reserved by this handle.
    pub max_batch_slots: usize,
}

impl RecurrentStateSpec {
    pub fn estimated_memory_bytes(&self) -> usize {
        let state_bytes_per_slot = self.tensors.iter().fold(0usize, |total, tensor| {
            total.saturating_add(
                tensor
                    .num_elements()
                    .saturating_mul(tensor.dtype.size_bytes()),
            )
        });
        state_bytes_per_slot.saturating_mul(self.max_batch_slots)
    }
}

/// Resume policy for state that has been preempted or evicted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecurrentStateResumePolicy {
    /// Rebuild state from request token history before the next decode.
    RecomputeOnResume,
    /// Save and restore backend-specific state bytes during preemption.
    SnapshotOnPreempt,
}

/// Statistics for one recurrent-state handle.
#[derive(Debug, Clone)]
pub struct RecurrentStateHandleStats {
    /// Total memory used by this handle.
    pub memory_bytes: usize,
    /// Number of state tensors represented by this handle.
    pub state_tensors: usize,
    /// Number of request/batch slots reserved by this handle.
    pub batch_slots: usize,
    /// Last access timestamp for eviction policies.
    pub last_access: Instant,
}

/// Recurrent-state cache handle.
pub trait RecurrentStateHandle: Send + Sync + std::fmt::Debug {
    /// Request ID this state belongs to.
    fn request_id(&self) -> RequestId;

    /// Device where the state resides.
    fn device(&self) -> Device;

    /// Number of model layers represented by this handle.
    fn num_layers(&self) -> usize;

    /// Approximate memory used by this state.
    fn state_bytes(&self) -> usize;

    /// Clone handle reference. Implementations should not deep-copy state.
    fn clone_handle(&self) -> Result<Arc<dyn RecurrentStateHandle>>;

    /// Downcast support for backend-specific handles.
    fn as_any(&self) -> &dyn Any;

    /// Handle statistics.
    fn stats(&self) -> RecurrentStateHandleStats;

    /// Check whether state is still valid and accessible.
    fn is_valid(&self) -> bool;

    /// Unique identifier for this cache instance.
    fn cache_id(&self) -> String;
}

/// Aggregate recurrent-state manager statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecurrentStateManagerStats {
    /// Total memory budget visible to this manager.
    pub total_memory_bytes: usize,
    /// Memory currently used by active recurrent states.
    pub used_memory_bytes: usize,
    /// Number of active state handles.
    pub active_states: usize,
    /// Number of active state tensors.
    pub active_state_tensors: usize,
    /// Total slots visible to this manager.
    pub total_batch_slots: usize,
    /// Slots currently allocated to active state handles.
    pub used_batch_slots: usize,
    /// Number of successful allocations.
    pub allocation_count: u64,
    /// Number of failed allocations.
    pub allocation_failures: u64,
    /// Number of evictions performed.
    pub eviction_count: u64,
}

/// Recurrent-state cache manager for allocation and lifecycle management.
#[async_trait::async_trait]
pub trait RecurrentStateManager: Send + Sync {
    /// Allocate recurrent state for one request.
    async fn allocate(&self, spec: &RecurrentStateSpec) -> Result<Arc<dyn RecurrentStateHandle>>;

    /// Deallocate recurrent state for one request.
    async fn deallocate(&self, request_id: RequestId) -> Result<()>;

    /// Check whether the manager can satisfy the allocation.
    fn can_allocate(&self, spec: &RecurrentStateSpec) -> bool;

    /// Get handle for an existing request.
    fn get_handle(&self, request_id: RequestId) -> Option<Arc<dyn RecurrentStateHandle>>;

    /// List all active recurrent-state handles.
    fn list_handles(&self) -> Vec<(RequestId, Arc<dyn RecurrentStateHandle>)>;

    /// Get aggregate manager statistics.
    fn stats(&self) -> RecurrentStateManagerStats;

    /// Drop all state owned by this manager.
    async fn reset(&self) -> Result<()>;
}
