use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::marker::PhantomData;
use std::mem::ManuallyDrop;
use std::num::NonZeroU64;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering};
use std::sync::{Arc, Mutex, OnceLock, RwLock, RwLockReadGuard, Weak};
use tokio::sync::watch;

use super::{
    defer_device_cleanup, deferred_device_cleanup_status, maintain_deferred_device_cleanups,
    new_deferred_device_cleanup_domain, retire_deferred_device_cleanup_domain, AdmissionDecision,
    AdmissionDeferred, AdmissionDemand, AdmissionFitPolicy, AdmissionPreflightDecision,
    AdmissionPressureAction, AdmissionRejected, AllocationKind, AllocationLifetime,
    BatchCapacityClaimDecision, BatchInvocationId, BatchStepId, BufferDescriptor, BufferRequest,
    BufferUsage, CapacityAvailabilityEpoch, CapacityClaimDecision, CapacityDomainId,
    CapacityDomainSpec, CapacityEntry, CapacityEpochs, CapacityUnits, CapacityVector,
    CapacityWaitCondition, CapacityWaitRecheck, DeferredDeviceCleanupDisposition,
    DeferredDeviceCleanupDomainId, DeferredDeviceCleanupMaintenanceReceipt,
    DeferredDeviceCleanupStatus, DeferredDeviceCleanupTask, DeviceCommandBatch, DeviceDescriptor,
    DeviceId, DeviceRuntime, DynamicBackingPoolId, DynamicBackingPoolSpec,
    DynamicResourceDescriptor, DynamicResourceShape, DynamicStorageAllocator,
    DynamicStorageProfile, DynamicStorageView, ElementType, ExecutionFrameId, ExecutionPlan,
    FailureDomain, FailureEnvelope, InvocationLivenessMode, LogicalAdmissionCoordinator,
    LogicalAdmissionCoordinatorId, LogicalAdmissionLease, LogicalBatchCapacityLease,
    LogicalCapacityLease, LogicalRequestLease, NodeId, PlanHash, PlanId, PlanNode,
    RequestAdmissionDecision, RequestAuthorityId, RequestIdentity, ResourceAllocation, ResourceId,
    ResourceWorkShape, RunId, SequenceAuthorityId, StateInitialization, StepResourceSlotKind,
    StreamState, TokenSpanWork, TransactionId, VNextError,
    MAX_DEFERRED_DEVICE_CLEANUP_MAINTENANCE_TASKS,
};

mod contracts;
pub use contracts::*;
mod capacity;
use capacity::*;
mod provisioning;
pub use provisioning::*;
mod allocation;
pub use allocation::*;
mod runtime_driver;
pub use runtime_driver::*;
mod ledger;
pub use ledger::*;
mod recovery;
pub use recovery::*;
mod backing_extent;
use backing_extent::{backing_segment_range, FreeExtentIndex};
pub use backing_extent::{BackingChunkIdentity, BackingSegment};
mod dynamic_pool;
pub use dynamic_pool::*;
mod dynamic_pool_maintenance;
pub use dynamic_pool_maintenance::*;
mod static_lease;
pub use static_lease::*;
mod work;
pub use work::*;
mod plan_runtime;
pub use plan_runtime::*;
mod sequence;
pub use sequence::*;
mod batch;
pub use batch::*;
mod invocation;
pub use invocation::*;
mod transaction;
pub use transaction::*;
mod static_initialization;
pub use static_initialization::*;

#[cfg(test)]
#[path = "resource/dynamic_pool_tests.rs"]
mod dynamic_pool_tests;

#[cfg(test)]
#[path = "resource/sequence_session_frame_tests.rs"]
mod sequence_session_frame_tests;
