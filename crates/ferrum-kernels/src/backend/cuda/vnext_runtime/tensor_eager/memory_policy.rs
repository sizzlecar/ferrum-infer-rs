//! A declared policy and an observed snapshot are different evidence.
//! The ceiling is conservative and is not measured peak device utilization.

use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};

pub const CUDA_TENSOR_AUTO_MEMORY_POLICY_VERSION: &str =
    "cuda.tensor.auto-memory.free-total-fraction.v1";

/// An exact finite binary64 fraction in (0, 1]. Conversion to byte capacity
/// rounds down using integer arithmetic, including at the u64 boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CudaMemoryUtilization(u64);

impl CudaMemoryUtilization {
    pub fn new(ratio: f64) -> Result<Self, String> {
        if !ratio.is_finite() || ratio <= 0.0 || ratio > 1.0 {
            return Err("CUDA memory utilization must be finite and in (0, 1]".into());
        }
        Ok(Self(ratio.to_bits()))
    }

    pub fn ratio(self) -> f64 {
        f64::from_bits(self.0)
    }

    fn floor_bytes(self, total: u64) -> u64 {
        let exponent = (self.0 >> 52) & 0x7ff;
        let fraction = self.0 & ((1_u64 << 52) - 1);
        let (significand, shift) = if exponent == 0 {
            (fraction, 1074)
        } else {
            ((1_u64 << 52) | fraction, 1075 - exponent)
        };
        // At most 64 + 53 significant bits; no intermediate exceeds u128.
        let product = u128::from(total) * u128::from(significand);
        if shift >= 128 {
            0
        } else {
            (product >> shift) as u64
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct AutoMemoryPolicy {
    utilization: CudaMemoryUtilization,
    maximum_bytes: Option<u64>,
}

impl AutoMemoryPolicy {
    pub(crate) fn new(
        utilization: CudaMemoryUtilization,
        maximum_bytes: Option<u64>,
    ) -> Result<Self, String> {
        if maximum_bytes == Some(0) {
            return Err("CUDA automatic memory override must be positive".into());
        }
        Ok(Self {
            utilization,
            maximum_bytes,
        })
    }

    /// Stable declaration only. Observed free memory does not change this
    /// identity after a process-wide capacity account has been opened.
    pub(crate) fn identity(self) -> String {
        format!(
            "{CUDA_TENSOR_AUTO_MEMORY_POLICY_VERSION}:ratio-bits={:016x}:maximum={:?}",
            self.utilization.0, self.maximum_bytes
        )
    }

    pub(crate) fn provisional_ceiling(self, total: u64) -> Result<u64, String> {
        let fraction = self.utilization.floor_bytes(total);
        let ceiling = self
            .maximum_bytes
            .map_or(fraction, |maximum| maximum.min(fraction));
        if total == 0 || ceiling == 0 {
            return Err("CUDA automatic memory policy leaves no byte capacity".into());
        }
        Ok(ceiling)
    }

    pub(crate) fn observe(
        self,
        total: u64,
        free: u64,
        opaque_bytes: u64,
    ) -> Result<CudaTensorMemoryObservation, String> {
        if free > total {
            return Err("CUDA memory snapshot reports free bytes above total bytes".into());
        }
        let ceiling = self.provisional_ceiling(total)?.min(free);
        if ceiling <= opaque_bytes {
            return Err("CUDA observed conservative total ceiling leaves no storage after declared opaque allowances".into());
        }
        Ok(CudaTensorMemoryObservation {
            policy_version: CUDA_TENSOR_AUTO_MEMORY_POLICY_VERSION,
            ceiling_semantics: "conservative_total_including_opaque",
            utilization: self.utilization.ratio(),
            total_bytes: total,
            free_bytes: free,
            utilization_ceiling_bytes: self.utilization.floor_bytes(total),
            maximum_memory_bytes: self.maximum_bytes,
            effective_ceiling_bytes: ceiling,
        })
    }
}

/// Snapshot taken on the runtime's sole CUDA context after bootstrap creation.
/// `effective_ceiling_bytes` is a conservative total account ceiling, including
/// all bootstrap/stream/library claims. No opaque estimate is added back to
/// observed free memory. Other processes may change availability afterwards.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize)]
pub struct CudaTensorMemoryObservation {
    policy_version: &'static str,
    ceiling_semantics: &'static str,
    utilization: f64,
    total_bytes: u64,
    free_bytes: u64,
    utilization_ceiling_bytes: u64,
    maximum_memory_bytes: Option<u64>,
    effective_ceiling_bytes: u64,
}

impl CudaTensorMemoryObservation {
    pub fn policy_version(&self) -> &'static str {
        self.policy_version
    }
    pub fn ceiling_semantics(&self) -> &'static str {
        self.ceiling_semantics
    }
    pub fn utilization(&self) -> f64 {
        self.utilization
    }
    pub fn total_bytes(&self) -> u64 {
        self.total_bytes
    }
    pub fn free_bytes(&self) -> u64 {
        self.free_bytes
    }
    pub fn utilization_ceiling_bytes(&self) -> u64 {
        self.utilization_ceiling_bytes
    }
    pub fn maximum_memory_bytes(&self) -> Option<u64> {
        self.maximum_memory_bytes
    }
    pub fn effective_ceiling_bytes(&self) -> u64 {
        self.effective_ceiling_bytes
    }
}

/// On every early return/unwind, destroy native objects before their admission
/// owner. A destructor panic cannot establish cleanup: retain the owner then.
pub(crate) struct RetainedNative<N, O> {
    native: Option<N>,
    owner: Option<O>,
}

impl<N, O> RetainedNative<N, O> {
    pub(crate) fn new(native: N, owner: O) -> Self {
        Self {
            native: Some(native),
            owner: Some(owner),
        }
    }
    pub(crate) fn native(&self) -> &N {
        self.native
            .as_ref()
            .expect("construction guard owns native")
    }
    pub(crate) fn owner_mut(&mut self) -> &mut O {
        self.owner.as_mut().expect("construction guard owns quota")
    }
    pub(crate) fn into_parts(mut self) -> (N, O) {
        (
            self.native.take().expect("construction guard owns native"),
            self.owner.take().expect("construction guard owns quota"),
        )
    }
    pub(crate) fn retain_owner(&mut self) {
        if let Some(owner) = self.owner.take() {
            std::mem::forget(owner);
        }
    }
}

impl<N, O> Drop for RetainedNative<N, O> {
    fn drop(&mut self) {
        if let Some(native) = self.native.take() {
            if let Err(payload) = catch_unwind(AssertUnwindSafe(|| drop(native))) {
                self.retain_owner();
                resume_unwind(payload);
            }
        }
        // `owner` drops after the native object's complete destructor.
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    #[test]
    fn utilization_rejects_invalid_values_and_floors_at_u64_boundary() {
        for ratio in [
            0.0,
            -0.0,
            -0.5,
            1.0001,
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
        ] {
            assert!(CudaMemoryUtilization::new(ratio).is_err());
        }
        assert_eq!(
            CudaMemoryUtilization::new(1.0)
                .unwrap()
                .floor_bytes(u64::MAX),
            u64::MAX
        );
        assert_eq!(
            CudaMemoryUtilization::new(0.5)
                .unwrap()
                .floor_bytes(u64::MAX),
            u64::MAX / 2
        );
        assert_eq!(
            CudaMemoryUtilization::new(f64::from_bits(1))
                .unwrap()
                .floor_bytes(u64::MAX),
            0
        );
        assert_eq!(
            CudaMemoryUtilization::new(f64::from_bits(1.0_f64.to_bits() - 1))
                .unwrap()
                .floor_bytes(u64::MAX),
            u64::MAX - 2048
        );
        assert_eq!(CudaMemoryUtilization::new(0.75).unwrap().floor_bytes(7), 5);
    }

    #[test]
    fn observation_is_minimum_of_free_fraction_and_positive_override() {
        let ratio = CudaMemoryUtilization::new(0.75).unwrap();
        assert!(AutoMemoryPolicy::new(ratio, Some(0)).is_err());
        let policy = AutoMemoryPolicy::new(ratio, None).unwrap();
        let first = policy.observe(1024, 900, 64).unwrap();
        let second = policy.observe(1024, 600, 64).unwrap();
        assert_eq!(first.effective_ceiling_bytes(), 768);
        assert_eq!(second.effective_ceiling_bytes(), 600);
        assert_eq!(first.policy_version(), second.policy_version());
        assert_eq!(first.ceiling_semantics(), second.ceiling_semantics());
        assert_ne!(first.free_bytes(), second.free_bytes());
        let capped = AutoMemoryPolicy::new(ratio, Some(512)).unwrap();
        assert_ne!(policy.identity(), capped.identity());
        assert_eq!(
            capped
                .observe(1024, 900, 64)
                .unwrap()
                .effective_ceiling_bytes(),
            512
        );
        assert_eq!(
            capped
                .observe(1024, 400, 64)
                .unwrap()
                .effective_ceiling_bytes(),
            400
        );
        for (total, free, opaque) in [
            (0, 0, 0),
            (1024, 1025, 0),
            (1024, 0, 0),
            (1024, 64, 64),
            (1024, 65, 66),
        ] {
            assert!(policy.observe(total, free, opaque).is_err());
        }
    }

    struct DropProbe {
        label: &'static str,
        log: Arc<Mutex<Vec<&'static str>>>,
        panic: bool,
    }
    impl Drop for DropProbe {
        fn drop(&mut self) {
            self.log.lock().unwrap().push(self.label);
            assert!(!self.panic, "injected native destruction failure");
        }
    }

    #[test]
    fn query_or_insufficient_capacity_errors_destroy_native_before_quota() {
        for query_error in [true, false] {
            let log = Arc::new(Mutex::new(Vec::new()));
            let result = (|| -> Result<(), String> {
                let _guard = RetainedNative::new(
                    DropProbe {
                        label: "native",
                        log: log.clone(),
                        panic: false,
                    },
                    DropProbe {
                        label: "quota",
                        log: log.clone(),
                        panic: false,
                    },
                );
                if query_error {
                    return Err("injected memory query failure".into());
                }
                AutoMemoryPolicy::new(CudaMemoryUtilization::new(1.0)?, None)?
                    .observe(128, 16, 16)?;
                Ok(())
            })();
            assert!(result.is_err());
            assert_eq!(*log.lock().unwrap(), ["native", "quota"]);
        }
    }

    #[test]
    fn uncertain_native_destruction_retains_quota() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let guard = RetainedNative::new(
            DropProbe {
                label: "native",
                log: log.clone(),
                panic: true,
            },
            DropProbe {
                label: "quota",
                log: log.clone(),
                panic: false,
            },
        );
        assert!(catch_unwind(AssertUnwindSafe(|| drop(guard))).is_err());
        assert_eq!(*log.lock().unwrap(), ["native"]);
    }
}
