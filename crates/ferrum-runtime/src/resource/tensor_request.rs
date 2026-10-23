//! Fixed request state initialized by a successful producer wave. Copies use
//! already admitted source/destination backing and the producer's one fence.
use super::*;
use crate::{
    CopyRegion, ProgramValueId, ResolvedStorageComponent, ResolvedTensorLayout, ResolvedTensorSpec,
    ResolvedValueRole, StateInitialization,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorRequestStateMapping {
    pub source_output: ProgramValueId,
    pub destination_state: ProgramValueId,
}

#[derive(Clone)]
struct StatePort {
    tensor: ResolvedTensorSpec,
    component: ResolvedStorageComponent,
}

const FRESH: u8 = 0;
const COPYING: u8 = 1;
const READY: u8 = 2;
const FAILED: u8 = 3;

struct Target<R: DeviceRuntime> {
    status: AtomicU8,
    ports: BTreeMap<ProgramValueId, StatePort>,
    session: Arc<SequenceSession<R>>,
}

/// No native addresses are exposed. This owns exactly one admitted request's
/// readonly fixed state; cloning physical storage or importing another plan's
/// allocation catalog is deliberately unsupported.
pub struct TensorRequestStateTarget<R: DeviceRuntime> {
    inner: Arc<Target<R>>,
}

impl<R: DeviceRuntime> TensorRequestStateTarget<R> {
    pub fn new(plan: &ExecutionPlan, session: Arc<SequenceSession<R>>) -> Result<Self, VNextError> {
        if !plan.payload().work_limits().is_tensor()
            || plan.payload().memory().maximum_active_sequences() != 1
            || session.resources().plan_evidence().plan_hash() != plan.plan_hash()
        {
            return Err(invalid_resource(
                "fixed request state differs from its admitted plan",
            ));
        }
        let mut ports = BTreeMap::<ProgramValueId, StatePort>::new();
        for node in plan.payload().nodes() {
            for binding in node
                .values()
                .iter()
                .filter(|v| v.usage() == BufferUsage::State)
            {
                let [component] = binding.storage().components() else {
                    return Err(invalid_resource(
                        "fixed request state requires one dense component",
                    ));
                };
                let descriptor = plan
                    .payload()
                    .memory()
                    .dynamic_descriptors()
                    .iter()
                    .find(|d| d.base_resource_id() == component.resource_id())
                    .ok_or_else(|| invalid_resource("request state descriptor is missing"))?;
                if binding.role() != ResolvedValueRole::Input
                    || binding.access() != TensorAccess::Read
                    || binding.tensor().layout() != &ResolvedTensorLayout::Contiguous
                    || descriptor.lifetime() != AllocationLifetime::Request
                    || descriptor.initialization() != StateInitialization::None
                    || !matches!(
                        descriptor.demand(),
                        crate::DynamicResourceDemand::Fixed { .. }
                    )
                    || component.offset_bytes() != 0
                    || component.length_bytes() != binding.tensor().minimum_storage_bytes()?
                {
                    return Err(invalid_resource("request initialization requires readonly contiguous fixed Request state with no implicit initializer"));
                }
                let port = StatePort {
                    tensor: binding.tensor().clone(),
                    component: component.clone(),
                };
                if let Some(prior) = ports.insert(binding.value_id().clone(), port.clone()) {
                    if prior.tensor != port.tensor || prior.component != port.component {
                        return Err(invalid_resource("request state has inconsistent bindings"));
                    }
                }
            }
        }
        if ports.is_empty() {
            return Err(invalid_resource(
                "persistent tensor request requires declared readonly state",
            ));
        }
        session
            .resources()
            .request_resources()
            .tensor_state_initialization_claimed
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| invalid_resource("request state already has an initialization owner"))?;
        Ok(Self {
            inner: Arc::new(Target {
                status: AtomicU8::new(FRESH),
                ports,
                session,
            }),
        })
    }

    pub fn ensure_ready(&self) -> Result<(), VNextError> {
        if self.inner.status.load(Ordering::Acquire) != READY {
            return Err(invalid_resource(
                "fixed request state has not completed successful initialization",
            ));
        }
        Ok(())
    }

    pub fn initialization_in_flight(&self) -> bool {
        self.inner.status.load(Ordering::Acquire) == COPYING
    }
}

struct StateCopy {
    node_index: usize,
    source: ResolvedStorageComponent,
    destination: StatePort,
}

pub(super) struct PreparedTensorRequestStateInitialization<R: DeviceRuntime> {
    copies: Vec<StateCopy>,
    // Physical segment retentions defend against a backend forgetting to retain
    // its borrowed views; the wave itself holds this through fence retirement.
    retentions: Mutex<Vec<DeviceBufferRetention>>,
    reservation: Option<PreparedSequenceStateTransfer<R>>,
    target: Arc<Target<R>>,
    settled: AtomicBool,
}

impl<R: DeviceRuntime> Drop for PreparedTensorRequestStateInitialization<R> {
    fn drop(&mut self) {
        if !self.settled.load(Ordering::Acquire) {
            self.target.status.store(FAILED, Ordering::Release);
        }
    }
}

impl<R: DeviceRuntime> PreparedStepSubmissionWave<R> {
    pub fn initialize_tensor_request_state(
        &mut self,
        source_plan: &ExecutionPlan,
        destination: &TensorRequestStateTarget<R>,
        mappings: &[TensorRequestStateMapping],
    ) -> Result<(), VNextError> {
        let target = &destination.inner;
        let destination_runtime = &target
            .session
            .resources()
            .request_resources()
            .plan
            .resources
            .runtime;
        if self.tensor_state_initialization.is_some()
            || self.purpose() != SubmissionWavePurpose::FullPlan
            || !Arc::ptr_eq(self.runtime(), destination_runtime)
            || self.step_resources().plan_evidence().plan_hash() != source_plan.plan_hash()
            || mappings.len() != target.ports.len()
            || mappings.is_empty()
        {
            return Err(invalid_resource(
                "request state copy requires complete ports, one source wave and the same runtime",
            ));
        }
        let ports = source_plan
            .payload()
            .tensor_ports()
            .ok_or_else(|| invalid_resource("producer has no tensor ports"))?;
        let mut source_ids = BTreeSet::new();
        let mut destination_ids = BTreeSet::new();
        let mut copies = Vec::with_capacity(mappings.len());
        for mapping in mappings {
            if !source_ids.insert(&mapping.source_output)
                || !destination_ids.insert(&mapping.destination_state)
            {
                return Err(invalid_resource(
                    "request state copy mappings are duplicated",
                ));
            }
            let output = ports
                .outputs()
                .iter()
                .find(|p| p.value_id() == &mapping.source_output)
                .ok_or_else(|| {
                    invalid_resource("state source is not a declared completion output")
                })?;
            let destination = target
                .ports
                .get(&mapping.destination_state)
                .ok_or_else(|| {
                    invalid_resource("state destination is not declared readonly Request state")
                })?;
            let node_index = self
                .nodes()
                .iter()
                .position(|n| n.node_id() == output.node_id())
                .ok_or_else(|| invalid_resource("state source producer is absent from wave"))?;
            let node = source_plan
                .payload()
                .nodes()
                .iter()
                .find(|n| n.id() == output.node_id())
                .ok_or_else(|| invalid_resource("state source producer is absent from plan"))?;
            let value = node
                .values()
                .iter()
                .find(|v| {
                    v.value_id() == output.value_id()
                        && v.role() == ResolvedValueRole::Output
                        && v.ordinal() == output.ordinal()
                })
                .ok_or_else(|| invalid_resource("state source output binding is missing"))?;
            let [source] = value.storage().components() else {
                return Err(invalid_resource(
                    "state source requires one dense physical component",
                ));
            };
            if output.tensor() != &destination.tensor
                || source.length_bytes() != destination.component.length_bytes()
                || !source_plan
                    .payload()
                    .terminal_output_resources()
                    .contains(source.resource_id())
            {
                return Err(invalid_resource(
                    "state copy shape/dtype/layout or completion retention differs",
                ));
            }
            copies.push(StateCopy {
                node_index,
                source: source.clone(),
                destination: destination.clone(),
            });
        }
        target
            .status
            .compare_exchange(FRESH, COPYING, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| invalid_resource("request state initialization is one-shot"))?;
        let reservation = target
            .session
            .resources()
            .backing_generation()
            .and_then(|generation| {
                target
                    .session
                    .try_prepare_state_transfer(SequenceStateTransferKind::RestoreWrite, generation)
            });
        let reservation = match reservation {
            Ok(SequenceStateTransferPreparation::Prepared(guard)) => guard,
            _ => {
                target.status.store(FAILED, Ordering::Release);
                return Err(invalid_resource(
                    "request state initialization target is busy or stale",
                ));
            }
        };
        self.tensor_state_initialization = Some(PreparedTensorRequestStateInitialization {
            copies,
            retentions: Mutex::new(Vec::new()),
            reservation: Some(reservation),
            target: target.clone(),
            settled: AtomicBool::new(false),
        });
        Ok(())
    }

    pub(crate) fn encode_tensor_state_initialization(
        &self,
        runtime: &R,
        commands: &mut DeviceCommandBatch<R::Command>,
    ) -> Result<(), VNextError> {
        let Some(init) = &self.tensor_state_initialization else {
            return Ok(());
        };
        let mut retained = init
            .retentions
            .lock()
            .map_err(|_| invalid_resource("state copy retention lock poisoned"))?;
        for copy in &init.copies {
            let source = self.backing_view(copy.node_index, copy.source.resource_id())?;
            let destination = init
                .target
                .session
                .resources()
                .request_resources()
                .backing_view(copy.destination.component.resource_id())?;
            let [src] = source.segment_bindings() else {
                return Err(invalid_resource(
                    "fixed state copy source must be a contiguous admitted extent",
                ));
            };
            let [dst] = destination.segment_bindings() else {
                return Err(invalid_resource(
                    "fixed state copy destination must be a contiguous admitted extent",
                ));
            };
            let bytes = copy.source.length_bytes();
            let src_offset = src
                .segment()
                .offset_bytes()
                .checked_add(copy.source.offset_bytes())
                .ok_or_else(|| invalid_resource("state copy source offset overflow"))?;
            let dst_offset = dst
                .segment()
                .offset_bytes()
                .checked_add(copy.destination.component.offset_bytes())
                .ok_or_else(|| invalid_resource("state copy destination offset overflow"))?;
            if copy
                .source
                .offset_bytes()
                .checked_add(bytes)
                .is_none_or(|n| n > source.size_bytes())
                || copy
                    .destination
                    .component
                    .offset_bytes()
                    .checked_add(bytes)
                    .is_none_or(|n| n > destination.size_bytes())
                || runtime.buffer_descriptor(src.buffer()) != *src.descriptor()
                || runtime.buffer_descriptor(dst.buffer()) != *dst.descriptor()
                || source.element_type() != copy.destination.tensor.element_type()
                || destination.element_type() != copy.destination.tensor.element_type()
            {
                return Err(invalid_resource(
                    "state copy exceeds exact backing or descriptor changed",
                ));
            }
            let region = CopyRegion::new(src_offset, dst_offset, bytes)?;
            region.validate_bounds(src.descriptor(), dst.descriptor())?;
            retained.push(src.retention().clone());
            retained.push(dst.retention().clone());
            let command = runtime
                .encode_tensor_state_copy(crate::DeviceTensorStateCopy::new(
                    src.buffer(),
                    dst.buffer(),
                    region,
                    src.retention(),
                    dst.retention(),
                ))
                .map_err(|e| invalid_resource(format!("fixed state copy encode failed: {e}")))?;
            commands.push_result_binding(command);
        }
        Ok(())
    }

    pub(crate) fn finish_tensor_state_initialization(
        &mut self,
        success: bool,
    ) -> Result<(), VNextError> {
        let Some(init) = &mut self.tensor_state_initialization else {
            return Ok(());
        };
        let guard = init
            .reservation
            .take()
            .ok_or_else(|| invalid_resource("state initialization already settled"))?;
        if success {
            guard.finish_fixed_tensor_initialization()?;
            init.target.status.store(READY, Ordering::Release);
        } else {
            init.target.status.store(FAILED, Ordering::Release);
            drop(guard);
        }
        init.settled.store(true, Ordering::Release);
        Ok(())
    }
}
