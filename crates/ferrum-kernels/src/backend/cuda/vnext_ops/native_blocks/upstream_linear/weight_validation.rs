//! Once-per-physical-leaf coefficient preparation on admitted storage.
//!
//! The dynamic command remains outside capture. Compute must unconditionally
//! retain the returned Arc and consume its declared destination; a host Ready
//! state means the scan/conversion and event were queued. MarkerV2 consumers
//! still read the device flag rather than assuming the coefficients passed.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, Weak};

use super::super::CudaNativeBlockKernels;
use cudarc::driver::{CudaEvent, CudaStream};
use ferrum_runtime::{DeviceNativeOperationId, ElementType};

use super::native_plan::PreparedProjectionNative;
use crate::backend::cuda::vnext_runtime::{
    CudaBufferRegion, CudaDependencyWork, CudaDeviceCommand, CudaDeviceRuntimeError,
    CudaPlanBackingIdentity,
};
use crate::native_ops::upstream_linear::{Arithmetic, DeviceSpan};
use crate::native_ops::upstream_q6_f32_linear::PreparedQ6F32Linear;

const RETAINED_WEIGHT_VALIDATION_OPERATION: DeviceNativeOperationId =
    match DeviceNativeOperationId::new("upstream.retained_weight_validation") {
        Some(identity) => identity,
        None => panic!("retained weight validation operation identity must be portable"),
    };

enum ValidationPlan {
    Upstream(Arc<PreparedProjectionNative>),
    Q6F32(Arc<PreparedQ6F32Linear>),
    Q8F32 {
        kernels: CudaNativeBlockKernels,
        inputs: u32,
        outputs: u32,
    },
}
fn source_alignment_valid(address: u64, byte_safe_q8: bool) -> bool {
    byte_safe_q8 || address % 4 == 0
}

impl ValidationPlan {
    fn prepare_destination(
        &self,
        weights: &CudaBufferRegion,
        destination: &CudaBufferRegion,
        stream: &CudaStream,
    ) -> Result<(), CudaDeviceRuntimeError> {
        let source = DeviceSpan {
            address: weights.device_ptr(),
            bytes: weights.length_bytes(),
        };
        let target = DeviceSpan {
            address: destination.device_ptr(),
            bytes: destination.length_bytes(),
        };
        match self {
            Self::Upstream(plan) => {
                unsafe { plan.check_weights(source, target, stream.cu_stream().cast()) }
                    .map_err(|e| CudaDeviceRuntimeError::contract(e.to_string()))
            }
            Self::Q6F32(plan) => {
                unsafe { plan.check_weights(source, target, stream.cu_stream().cast()) }
                    .map_err(|e| CudaDeviceRuntimeError::contract(e.to_string()))
            }
            Self::Q8F32 {
                kernels,
                inputs,
                outputs,
            } => kernels.predecode_q8_0_f32(stream, weights, destination, *inputs, *outputs),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ValidationIdentity {
    weight: CudaPlanBackingIdentity,
    implementation: String,
    native_operator: &'static str,
    native_boundary_abi: u32,
    algorithm: u32,
    format: u32,
    inputs: u32,
    outputs: u32,
    padded_outputs: u32,
    weight_bytes: u64,
    destination_bytes: u64,
}

impl ValidationIdentity {
    fn declaration_identity(&self) -> String {
        let mut identity = format!(
            "{}:{}:{}:{}:{}:{}:{}:{}",
            self.implementation,
            self.native_operator,
            self.algorithm,
            self.format,
            self.inputs,
            self.outputs,
            self.padded_outputs,
            self.weight_bytes,
        );
        if self.native_boundary_abi != 1 {
            identity.push_str(&format!(":boundary{}", self.native_boundary_abi));
        }
        if self.native_operator == "ferrum.q8_0.exact-f32-predecode.v1" {
            identity.push(':');
            identity.push_str(&self.destination_bytes.to_string());
        }
        identity
    }
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
        native: Arc<PreparedProjectionNative>,
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
            native_boundary_abi: native.boundary_abi(),
            algorithm: geometry.algorithm,
            format: geometry.request.format,
            inputs: geometry.request.inputs,
            outputs: geometry.request.outputs,
            padded_outputs: geometry.padded_outputs,
            weight_bytes: geometry.weight_bytes,
            destination_bytes: 4,
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
            native_boundary_abi: 1,
            algorithm: geometry.algorithm,
            format: geometry.request.format,
            inputs: geometry.request.inputs,
            outputs: geometry.request.outputs,
            padded_outputs: geometry.padded_outputs,
            weight_bytes: geometry.weight_bytes,
            destination_bytes: 4,
        };
        self.prepare_plan(ValidationPlan::Q6F32(native), identity, weights, flag)
    }

    pub fn prepare_q8_f32(
        &self,
        inputs: u32,
        outputs: u32,
        kernels: &CudaNativeBlockKernels,
        implementation: &str,
        weights: CudaBufferRegion,
        destination: CudaBufferRegion,
    ) -> Result<Arc<WeightValidation>, CudaDeviceRuntimeError> {
        kernels.validate_q8_0_f32_destination(&weights, &destination, inputs, outputs)?;
        if destination.element_type() != ElementType::U8 {
            return Err(CudaDeviceRuntimeError::contract(
                "Q8 F32 Plan destination requires declared opaque U8 storage",
            ));
        }
        let (weight_bytes, destination_bytes) =
            super::super::q8_predecode::q8_f32_bytes(inputs, outputs)?;
        let identity = ValidationIdentity {
            weight: weights.plan_backing_identity()?,
            implementation: implementation.to_owned(),
            native_operator: "ferrum.q8_0.exact-f32-predecode.v1",
            native_boundary_abi: 1,
            algorithm: 0,
            format: 8,
            inputs,
            outputs,
            padded_outputs: outputs,
            weight_bytes,
            destination_bytes,
        };
        self.prepare_plan(
            ValidationPlan::Q8F32 {
                kernels: kernels.clone(),
                inputs,
                outputs,
            },
            identity,
            weights,
            destination,
        )
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
            || !source_alignment_valid(
                weights.device_ptr(),
                matches!(&native, ValidationPlan::Q8F32 { .. }),
            )
            || weights.length_bytes() != identity.weight_bytes
            || flag.device_ptr() % 4 != 0
            || flag.length_bytes() != identity.destination_bytes
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
            .checked_add(identity.destination_bytes)
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

    pub fn destination_region(&self) -> &CudaBufferRegion {
        &self.flag
    }

    /// The existing scan/publication/wait is authorized independently from
    /// request binding slots and runs before every consuming wave, including replay.
    pub fn retained_dependency(
        self: &Arc<Self>,
        invocation: &ferrum_runtime::BatchedOperationInvocation<
            '_,
            crate::backend::cuda::vnext_runtime::CudaDeviceBuffer,
        >,
        input_ordinal: u32,
        component_id: &ferrum_runtime::WeightId,
        persistent_offset_bytes: u64,
    ) -> Result<
        ferrum_runtime::EncodedRetainedPlanDependency<CudaDeviceCommand>,
        CudaDeviceRuntimeError,
    > {
        let identity = self.identity.declaration_identity();
        let authority = invocation
            .retained_plan_dependency(ferrum_runtime::RetainedPlanDependencySpec {
                input_ordinal,
                component_id,
                source_offset_bytes: 0,
                source_length_bytes: self.identity.weight_bytes,
                persistent_offset_bytes,
                persistent_length_bytes: self.identity.destination_bytes,
                alignment_bytes: 4,
                validation_identity: &identity,
            })
            .map_err(|error| CudaDeviceRuntimeError::contract(error.to_string()))?;
        Ok(authority.encode(self.dynamic_command()?))
    }

    pub(in crate::backend::cuda::vnext_ops) fn segment_declaration(
        self: &Arc<Self>,
        input_ordinal: u32,
        component_id: &ferrum_runtime::WeightId,
        persistent_offset_bytes: u64,
    ) -> crate::backend::cuda::vnext_ops::transformer::segment_bindings::ValidationRecipe {
        use ferrum_runtime::SegmentBindingDependency;
        crate::backend::cuda::vnext_ops::transformer::segment_bindings::ValidationRecipe {
            state: self.clone(),
            declaration: SegmentBindingDependency {
                input_ordinal,
                component_id: component_id.clone(),
                source_offset_bytes: 0,
                source_length_bytes: self.identity.weight_bytes,
                persistent_offset_bytes,
                persistent_length_bytes: self.identity.destination_bytes,
                alignment_bytes: 4,
                validation_identity: self.identity.declaration_identity(),
            },
            weight_element_type: self.weights.element_type(),
        }
    }

    pub(in crate::backend::cuda::vnext_ops) fn validate_segment_owners(
        &self,
        weights: &CudaBufferRegion,
        flag: &CudaBufferRegion,
    ) -> Result<(), CudaDeviceRuntimeError> {
        if weights.plan_backing_identity()? != self.weights.plan_backing_identity()?
            || flag.plan_backing_identity()? != self.flag.plan_backing_identity()?
            || weights.length_bytes() != self.weights.length_bytes()
            || flag.length_bytes() != self.flag.length_bytes()
        {
            return Err(CudaDeviceRuntimeError::contract(
                "segment retained dependency differs from the cold Plan owners",
            ));
        }
        Ok(())
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
                self.native
                    .prepare_destination(&self.weights, &self.flag, stream)?;
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
    use super::{source_alignment_valid, CudaDependencyWork, ScanPublication};
    use std::cell::Cell;

    #[test]
    fn q8_f32_byte_safe_source_accepts_odd_offsets_without_relaxing_native_alignment() {
        for address in [1, 2, 3, 5, 7] {
            assert!(source_alignment_valid(address, true));
            assert!(!source_alignment_valid(address, false));
        }
        for address in [4, 8, 12] {
            assert!(source_alignment_valid(address, true));
            assert!(source_alignment_valid(address, false));
        }
    }

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
