//! Once-per-physical-leaf MarkerV2 coefficient checks on admitted storage.
//!
//! The dynamic command remains outside capture. Compute must unconditionally
//! retain the returned Arc and read its device flag; a host Ready state means
//! the scan/event were queued, never that the coefficients passed the check.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, Weak};

use cudarc::driver::{CudaEvent, CudaStream};
use ferrum_interfaces::vnext::DeviceNativeOperationId;

use crate::backend::cuda::vnext_runtime::{
    CudaBufferRegion, CudaDependencyWork, CudaDeviceCommand, CudaDeviceRuntimeError,
    CudaPlanBackingIdentity,
};
use crate::native_ops::upstream_linear::{Arithmetic, DeviceSpan, PreparedUpstreamLinear};
use crate::native_ops::upstream_q6_f32_linear::PreparedQ6F32Linear;

const RETAINED_WEIGHT_VALIDATION_OPERATION: DeviceNativeOperationId =
    match DeviceNativeOperationId::new("upstream.retained_weight_validation") {
        Some(identity) => identity,
        None => panic!("retained weight validation operation identity must be portable"),
    };

enum ValidationPlan {
    Upstream(Arc<PreparedUpstreamLinear>),
    Q6F32(Arc<PreparedQ6F32Linear>),
}
impl ValidationPlan {
    unsafe fn check_weights(
        &self,
        weights: DeviceSpan,
        flag: DeviceSpan,
        stream: *mut std::ffi::c_void,
    ) -> Result<(), crate::native_ops::upstream_linear::Error> {
        match self {
            Self::Upstream(plan) => unsafe { plan.check_weights(weights, flag, stream) },
            Self::Q6F32(plan) => unsafe { plan.check_weights(weights, flag, stream) },
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ValidationIdentity {
    weight: CudaPlanBackingIdentity,
    implementation: String,
    native_operator: &'static str,
    algorithm: u32,
    format: u32,
    inputs: u32,
    outputs: u32,
    padded_outputs: u32,
    weight_bytes: u64,
}

/// Weak entries never keep an old model/plan alive. Each live validation owns
/// both core-retained ranges; they prohibit allocation recycling while an
/// eager command or reusable graph still depends on this generation.
#[derive(Default)]
pub(in crate::backend::cuda::vnext_ops) struct WeightValidationRegistry {
    flags: Mutex<BTreeMap<CudaPlanBackingIdentity, Weak<WeightValidation>>>,
}

impl WeightValidationRegistry {
    pub fn prepare(
        &self,
        native: Arc<PreparedUpstreamLinear>,
        implementation: &str,
        weights: CudaBufferRegion,
        flag: CudaBufferRegion,
    ) -> Result<Arc<WeightValidation>, CudaDeviceRuntimeError> {
        if native.arithmetic() != Arithmetic::MarkerV2 {
            return Err(CudaDeviceRuntimeError::contract(
                "weight validation requires MarkerV2",
            ));
        }
        let geometry = native.geometry();
        let identity = ValidationIdentity {
            weight: weights.plan_backing_identity()?,
            implementation: implementation.to_owned(),
            native_operator: native.operator(),
            algorithm: geometry.algorithm,
            format: geometry.request.format,
            inputs: geometry.request.inputs,
            outputs: geometry.request.outputs,
            padded_outputs: geometry.padded_outputs,
            weight_bytes: geometry.weight_bytes,
        };
        self.prepare_plan(ValidationPlan::Upstream(native), identity, weights, flag)
    }

    pub fn prepare_q6(
        &self,
        native: Arc<PreparedQ6F32Linear>,
        implementation: &str,
        weights: CudaBufferRegion,
        flag: CudaBufferRegion,
    ) -> Result<Arc<WeightValidation>, CudaDeviceRuntimeError> {
        let geometry = native.geometry();
        let identity = ValidationIdentity {
            weight: weights.plan_backing_identity()?,
            implementation: implementation.to_owned(),
            native_operator: native.operator(),
            algorithm: geometry.algorithm,
            format: geometry.request.format,
            inputs: geometry.request.inputs,
            outputs: geometry.request.outputs,
            padded_outputs: geometry.padded_outputs,
            weight_bytes: geometry.weight_bytes,
        };
        self.prepare_plan(ValidationPlan::Q6F32(native), identity, weights, flag)
    }

    fn prepare_plan(
        &self,
        native: ValidationPlan,
        identity: ValidationIdentity,
        weights: CudaBufferRegion,
        flag: CudaBufferRegion,
    ) -> Result<Arc<WeightValidation>, CudaDeviceRuntimeError> {
        let error = CudaDeviceRuntimeError::contract;
        let flag_identity = flag.plan_backing_identity()?;
        if identity.implementation.is_empty()
            || identity.weight.runtime_instance != flag_identity.runtime_instance
            || weights.device_ptr() % 4 != 0
            || weights.length_bytes() != identity.weight_bytes
            || flag.device_ptr() % 4 != 0
            || flag.length_bytes() != 4
        {
            return Err(error(
                "weight validation differs from its exact admitted MarkerV2 views",
            ));
        }
        let weight_end = weights
            .device_ptr()
            .checked_add(weights.length_bytes())
            .ok_or_else(|| error("weight validation range overflow"))?;
        let flag_end = flag
            .device_ptr()
            .checked_add(4)
            .ok_or_else(|| error("weight validation flag overflow"))?;
        if weights.device_ptr() < flag_end && flag.device_ptr() < weight_end {
            return Err(error("weight validation flag aliases immutable weights"));
        }
        let mut flags = self
            .flags
            .lock()
            .map_err(|_| error("weight validation registry poisoned"))?;
        flags.retain(|_, state| state.strong_count() != 0);
        if let Some(existing) = flags.get(&flag_identity).and_then(Weak::upgrade) {
            if existing.identity != identity {
                return Err(error(
                    "live weight flag is already bound to different coefficients or arithmetic",
                ));
            }
            return Ok(existing);
        }
        let state = Arc::new(WeightValidation {
            identity,
            native,
            weights,
            flag,
            publication: Mutex::new(ScanPublication::Pending),
        });
        flags.insert(flag_identity, Arc::downgrade(&state));
        Ok(state)
    }
}

pub(in crate::backend::cuda::vnext_ops) struct WeightValidation {
    identity: ValidationIdentity,
    native: ValidationPlan,
    weights: CudaBufferRegion,
    flag: CudaBufferRegion,
    publication: Mutex<ScanPublication<CudaEvent>>,
}

impl WeightValidation {
    pub fn flag_span(&self) -> DeviceSpan {
        DeviceSpan {
            address: self.flag.device_ptr(),
            bytes: self.flag.length_bytes(),
        }
    }

    /// The existing scan/publication/wait is authorized independently from
    /// request binding slots and runs before every consuming wave, including replay.
    pub fn retained_dependency(
        self: &Arc<Self>,
        invocation: &ferrum_interfaces::vnext::BatchedOperationInvocation<
            '_,
            crate::backend::cuda::vnext_runtime::CudaDeviceBuffer,
        >,
        input_ordinal: u32,
        component_id: &ferrum_interfaces::vnext::WeightId,
        persistent_offset_bytes: u64,
    ) -> Result<
        ferrum_interfaces::vnext::EncodedRetainedPlanDependency<CudaDeviceCommand>,
        CudaDeviceRuntimeError,
    > {
        let identity = format!(
            "{}:{}:{}:{}:{}:{}:{}:{}",
            self.identity.implementation,
            self.identity.native_operator,
            self.identity.algorithm,
            self.identity.format,
            self.identity.inputs,
            self.identity.outputs,
            self.identity.padded_outputs,
            self.identity.weight_bytes
        );
        let authority = invocation
            .retained_plan_dependency(ferrum_interfaces::vnext::RetainedPlanDependencySpec {
                input_ordinal,
                component_id,
                source_offset_bytes: 0,
                source_length_bytes: self.identity.weight_bytes,
                persistent_offset_bytes,
                persistent_length_bytes: 4,
                alignment_bytes: 4,
                validation_identity: &identity,
            })
            .map_err(|error| CudaDeviceRuntimeError::contract(error.to_string()))?;
        Ok(authority.encode(self.dynamic_command()?))
    }

    /// Always attach outside capture, including replay submissions. This safe
    /// command belongs in the independently authorized Plan dependency prefix;
    /// it must never be captured into a reusable compute segment.
    pub fn dynamic_command(self: &Arc<Self>) -> Result<CudaDeviceCommand, CudaDeviceRuntimeError> {
        let state = self.clone();
        CudaDeviceCommand::conditional_dependency(
            RETAINED_WEIGHT_VALIDATION_OPERATION,
            vec![self.weights.clone(), self.flag.clone()],
            move |stream, _regions| state.enqueue(stream),
        )
    }

    fn enqueue(&self, stream: &CudaStream) -> Result<CudaDependencyWork, CudaDeviceRuntimeError> {
        let mut publication = self.publication.lock().map_err(|_| {
            CudaDeviceRuntimeError::contract("weight validation publication poisoned")
        })?;
        publication.ensure(
            || {
                // Only this callback publishes the flag. On any enqueue or
                // event-record error, no later stream may assume it is ready.
                unsafe {
                    self.native.check_weights(
                        DeviceSpan {
                            address: self.weights.device_ptr(),
                            bytes: self.weights.length_bytes(),
                        },
                        self.flag_span(),
                        stream.cu_stream().cast(),
                    )
                }
                .map_err(|e| CudaDeviceRuntimeError::contract(e.to_string()))?;
                stream.record_event(None).map_err(|e| {
                    CudaDeviceRuntimeError::driver("weight validation event record", e)
                })
            },
            |event| {
                stream.wait(event).map_err(|e| {
                    CudaDeviceRuntimeError::driver("weight validation stream dependency", e)
                })
            },
            || CudaDeviceRuntimeError::contract("prior weight validation submission failed"),
        )
    }
}

/// The lock covers enqueue plus event publication, not device completion.
/// Failure is sticky: even an event-record failure may follow a partial scan.
enum ScanPublication<E> {
    Pending,
    Ready(E),
    Failed,
}

impl<E> ScanPublication<E> {
    fn ensure<Error>(
        &mut self,
        scan_and_record: impl FnOnce() -> Result<E, Error>,
        wait: impl FnOnce(&E) -> Result<(), Error>,
        previous_failure: impl FnOnce() -> Error,
    ) -> Result<CudaDependencyWork, Error> {
        let scanned = matches!(self, Self::Pending);
        if scanned {
            // Publish failure before any fallible or panicking enqueue. A
            // poisoned Mutex is also rejected by the caller, never recovered.
            *self = Self::Failed;
            *self = Self::Ready(scan_and_record()?);
        }
        match self {
            Self::Ready(event) => {
                if let Err(error) = wait(event) {
                    *self = Self::Failed;
                    return Err(error);
                }
                Ok(if scanned {
                    CudaDependencyWork::ScanAndWait
                } else {
                    CudaDependencyWork::Wait
                })
            }
            Self::Failed => Err(previous_failure()),
            Self::Pending => unreachable!(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{CudaDependencyWork, ScanPublication};
    use std::cell::Cell;

    #[test]
    fn upstream_weight_validation_publishes_once_but_orders_every_consumer() {
        let scans = Cell::new(0);
        let waits = Cell::new(0);
        let mut state = ScanPublication::Pending;
        for (index, consumer) in [0, 1, 0, 2].into_iter().enumerate() {
            let receipt = state
                .ensure(
                    || {
                        scans.set(scans.get() + 1);
                        Ok::<_, &str>(17)
                    },
                    |event| {
                        assert_eq!(*event, 17);
                        let _ = consumer;
                        waits.set(waits.get() + 1);
                        Ok(())
                    },
                    || "failed",
                )
                .unwrap();
            assert_eq!(
                receipt,
                if index == 0 {
                    CudaDependencyWork::ScanAndWait
                } else {
                    CudaDependencyWork::Wait
                }
            );
        }
        assert_eq!(scans.get(), 1);
        assert_eq!(waits.get(), 4);
    }

    #[test]
    fn upstream_weight_validation_never_publishes_failed_scan_record_or_wait() {
        for failure in ["scan", "record"] {
            let mut state = ScanPublication::<u32>::Pending;
            assert_eq!(
                state.ensure(|| Err(failure), |_| panic!("unrecorded event"), || "prior"),
                Err(failure)
            );
            assert_eq!(
                state.ensure(
                    || panic!("unsafe retry"),
                    |_| panic!("failed event"),
                    || "prior"
                ),
                Err("prior")
            );
        }
        let mut state = ScanPublication::Pending;
        assert_eq!(
            state.ensure(|| Ok(17), |_| Err("wait"), || "prior"),
            Err("wait")
        );
        assert_eq!(
            state.ensure(
                || panic!("unsafe retry"),
                |_| panic!("failed event"),
                || "prior"
            ),
            Err("prior")
        );
    }
}
