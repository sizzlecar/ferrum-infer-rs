//! Recommended working-set selection, not a claim to observe system free RAM.
use super::*;

#[derive(Debug, Clone)]
pub struct MetalTensorRuntimeAutoConfig {
    pub device: MetalDeviceRuntimeConfig,
    pub utilization: f64,
    pub maximum_memory_bytes: Option<u64>,
    pub bootstrap_memory_bytes: u64,
    pub stream_memory_bytes: u64,
    pub maximum_streams: usize,
    pub maximum_in_flight_per_stream: usize,
    pub maximum_commands_per_submission: usize,
    pub opaque_memory_profile: String,
}

#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct MetalTensorMemoryObservation {
    pub recommended_working_set_bytes: u64,
    pub process_allocated_bytes: u64,
    pub utilization: f64,
    pub maximum_memory_bytes: Option<u64>,
    pub effective_ceiling_bytes: u64,
}
impl MetalTensorMemoryObservation {
    fn resolve(
        recommended: u64,
        allocated: u64,
        utilization: f64,
        maximum: Option<u64>,
    ) -> Result<Self, MetalDeviceRuntimeError> {
        if !utilization.is_finite() || utilization <= 0.0 || utilization > 1.0 || maximum == Some(0)
        {
            return Err(invalid(
                "Metal memory utilization must be in (0,1] and any explicit maximum positive",
            ));
        }
        let effective = ((recommended as f64 * utilization).floor() as u64)
            .min(recommended.saturating_sub(allocated))
            .min(maximum.unwrap_or(u64::MAX));
        if effective == 0 {
            return Err(invalid(
                "Metal working-set observation leaves no device capacity",
            ));
        }
        Ok(Self {
            recommended_working_set_bytes: recommended,
            process_allocated_bytes: allocated,
            utilization,
            maximum_memory_bytes: maximum,
            effective_ceiling_bytes: effective,
        })
    }
}
impl MetalTensorRuntimeAutoConfig {
    fn explicit(&self, capacity: u64) -> MetalTensorRuntimeConfig {
        MetalTensorRuntimeConfig {
            device: self.device.clone(),
            usable_memory_bytes: capacity,
            bootstrap_memory_bytes: self.bootstrap_memory_bytes,
            stream_memory_bytes: self.stream_memory_bytes,
            maximum_streams: self.maximum_streams,
            maximum_in_flight_per_stream: self.maximum_in_flight_per_stream,
            maximum_commands_per_submission: self.maximum_commands_per_submission,
            opaque_memory_profile: self.opaque_memory_profile.clone(),
        }
    }
    pub fn validate(&self) -> Result<(), MetalDeviceRuntimeError> {
        MetalTensorMemoryObservation::resolve(
            u64::MAX,
            0,
            self.utilization,
            self.maximum_memory_bytes,
        )?;
        self.explicit(self.maximum_memory_bytes.unwrap_or(u64::MAX))
            .validate()
    }
}
impl MetalDeviceRuntime {
    /// Query the exact device handle used to create this runtime. External
    /// process memory is not observable here; normal plan admission still applies.
    pub fn new_tensor_eager_auto(
        config: MetalTensorRuntimeAutoConfig,
    ) -> Result<(Self, MetalTensorMemoryObservation), MetalDeviceRuntimeError> {
        config.validate()?;
        let device =
            metal::Device::system_default().ok_or_else(|| invalid("no default Metal device"))?;
        let observation = MetalTensorMemoryObservation::resolve(
            device.recommended_max_working_set_size(),
            device.current_allocated_size(),
            config.utilization,
            config.maximum_memory_bytes,
        )?;
        let runtime = Self::new_tensor_eager_on_device(
            config.explicit(observation.effective_ceiling_bytes),
            device,
        )?;
        Ok((runtime, observation))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn working_set_ceiling_respects_existing_allocations_and_explicit_cap() {
        for (allocated, maximum, expected) in [
            (0, None, 750),
            (900, None, 100),
            (0, Some(600), 600),
            (900, Some(600), 100),
        ] {
            assert_eq!(
                MetalTensorMemoryObservation::resolve(1000, allocated, 0.75, maximum)
                    .unwrap()
                    .effective_ceiling_bytes,
                expected
            );
        }
        for utilization in [0.0, -0.1, 1.01, f64::NAN, f64::INFINITY] {
            assert!(MetalTensorMemoryObservation::resolve(1000, 0, utilization, None).is_err());
        }
        assert!(MetalTensorMemoryObservation::resolve(1000, 0, 0.75, Some(0)).is_err());
        for allocated in [1000, 1001] {
            assert!(MetalTensorMemoryObservation::resolve(1000, allocated, 1.0, None).is_err());
        }
    }
}
