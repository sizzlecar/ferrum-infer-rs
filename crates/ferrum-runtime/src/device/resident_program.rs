//! Current-use authority for a cold, complete resident decode binding program.
use super::DeviceBufferRetention;
use crate::{
    BatchWorkShape, DeviceResidentPlanDependencyBatch, EncodedResidentPlanDependency,
    SequencePhysicalBindingTable,
};
use std::sync::Arc;

/// An append-only physical-table transition checked by Core under current
/// resource authority. Creating this does not consume dependencies or enqueue.
pub struct DeviceResidentDecodeRefresh<'a, B> {
    work: &'a BatchWorkShape,
    previous: &'a [Arc<SequencePhysicalBindingTable<B>>],
    current: &'a [Arc<SequencePhysicalBindingTable<B>>],
}

impl<'a, B> DeviceResidentDecodeRefresh<'a, B> {
    pub(crate) fn new(
        work: &'a BatchWorkShape,
        previous: &'a [Arc<SequencePhysicalBindingTable<B>>],
        current: &'a [Arc<SequencePhysicalBindingTable<B>>],
    ) -> Self {
        Self {
            work,
            previous,
            current,
        }
    }
    pub fn work_shape(&self) -> &BatchWorkShape {
        self.work
    }
    pub fn previous_tables(&self) -> &[Arc<SequencePhysicalBindingTable<B>>] {
        self.previous
    }
    pub fn current_tables(&self) -> &[Arc<SequencePhysicalBindingTable<B>>] {
        self.current
    }
}

/// Core constructs this only after current resource/frame/window validation.
/// It cannot be cloned or manufactured by an operation provider.
pub struct DeviceResidentDecodeRequest<'a> {
    work: &'a BatchWorkShape,
    retention: DeviceBufferRetention,
    dependencies: Option<DeviceResidentPlanDependencyBatch>,
}

impl<'a> DeviceResidentDecodeRequest<'a> {
    pub(crate) fn new(
        work: &'a BatchWorkShape,
        retention: DeviceBufferRetention,
        dependencies: DeviceResidentPlanDependencyBatch,
    ) -> Self {
        Self {
            work,
            retention,
            dependencies: Some(dependencies),
        }
    }

    pub fn work_shape(&self) -> &BatchWorkShape {
        self.work
    }
    pub fn retention(&self) -> &DeviceBufferRetention {
        &self.retention
    }
    pub fn dependencies(&self) -> Option<&DeviceResidentPlanDependencyBatch> {
        self.dependencies.as_ref()
    }
    pub fn take_dependencies(&mut self) -> Option<DeviceResidentPlanDependencyBatch> {
        self.dependencies.take()
    }
}

/// The core retains the consumed fresh dependency authority and places both
/// commands before the existing resident replay and original completion fence.
pub struct DeviceEncodedResidentDecode<C> {
    dependencies: EncodedResidentPlanDependency<Option<C>>,
    bindings: C,
}

impl<C> DeviceEncodedResidentDecode<C> {
    pub fn new(dependencies: EncodedResidentPlanDependency<Option<C>>, bindings: C) -> Self {
        Self {
            dependencies,
            bindings,
        }
    }
    pub(crate) fn into_parts(self) -> (EncodedResidentPlanDependency<Option<C>>, C) {
        (self.dependencies, self.bindings)
    }
}
