//! A CPU payload ledger, separate from execution/GPU memory authority. Shared
//! immutable templates stay charged until their last observing owner releases
//! them, including owners queued after a resident graph has been evicted.
use super::*;
use std::sync::atomic::{AtomicUsize, Ordering};

pub const DEFAULT_OBSERVATION_TEMPLATE_BYTES: usize = 32 * 1024 * 1024;
pub const MAXIMUM_OBSERVATION_TEMPLATE_BYTES: usize = 256 * 1024 * 1024;

// Known CPU lease value and Arc strong/weak counters for the lease and its
// template. Providers may also conservatively include their Arc counters.
const RESERVATION_PAYLOAD_OVERHEAD: usize =
    std::mem::size_of::<TemplatePayloadLease>() + 4 * std::mem::size_of::<usize>();

#[derive(Debug)]
pub struct DeviceObservationTemplateBudget {
    maximum: usize,
    retained: AtomicUsize,
    peak: AtomicUsize,
}
impl DeviceObservationTemplateBudget {
    /// A passive numeric catalog shares the installed template ledger. This
    /// grants no template projection or execution permission.
    pub(crate) fn reserve_cost_catalog(
        self: &Arc<Self>,
        payload_upper_bound: usize,
    ) -> Result<DeviceCostCatalogPayloadLease, DeviceCostCatalogReservationError> {
        let required = payload_upper_bound
            .checked_add(RESERVATION_PAYLOAD_OVERHEAD)
            .ok_or(DeviceCostCatalogReservationError::Invalid)?;
        Ok(DeviceCostCatalogPayloadLease {
            reservation: self.reserve(payload_upper_bound).map_err(|_| {
                DeviceCostCatalogReservationError::Capacity {
                    required_exclusive_peak_bytes: required,
                }
            })?,
        })
    }
    pub fn new(maximum_bytes: usize) -> Result<Arc<Self>, VNextError> {
        if maximum_bytes == 0 || maximum_bytes > MAXIMUM_OBSERVATION_TEMPLATE_BYTES {
            return Err(VNextError::InvalidExecutionPlan {
                reason: "observation template budget must be within 1..=256 MiB".to_owned(),
            });
        }
        Ok(Arc::new(Self {
            maximum: maximum_bytes,
            retained: AtomicUsize::new(0),
            peak: AtomicUsize::new(0),
        }))
    }
    pub fn maximum_bytes(&self) -> usize {
        self.maximum
    }
    pub fn retained_payload_bytes(&self) -> usize {
        self.retained.load(Ordering::Acquire)
    }
    pub fn peak_retained_payload_bytes(&self) -> usize {
        self.peak.load(Ordering::Acquire)
    }
    /// Charges the whole declared CPU payload conservatively. Shared children
    /// may be charged by both their original owner and a composite template;
    /// this is deliberately not an exclusive-allocation or RSS claim.
    pub fn retain(
        self: &Arc<Self>,
        template: Arc<dyn DeviceObservationTemplate>,
    ) -> Result<RetainedDeviceObservationTemplate, StatisticalEvidenceUnknown> {
        let payload = template
            .retained_payload_bytes()
            .ok_or(StatisticalEvidenceUnknown::Capacity)?;
        self.reserve(payload)?.retain(template)
    }
    /// Reserve before copying CPU metadata. The reservation stays in the same
    /// ledger through construction and, on success, the template's lifetime.
    /// `payload_upper_bound` covers all owned allocations and the template value;
    /// allocator bookkeeping and process RSS are deliberately not represented.
    pub fn reserve(
        self: &Arc<Self>,
        payload_upper_bound: usize,
    ) -> Result<DeviceObservationTemplateReservation, StatisticalEvidenceUnknown> {
        self.reserve_with_diagnostic(payload_upper_bound, "template.reserve")
            .map_err(|failure| failure.error.expect("reservation preserves Capacity"))
    }

    pub fn reserve_with_diagnostic(
        self: &Arc<Self>,
        payload_upper_bound: usize,
        site: &'static str,
    ) -> Result<DeviceObservationTemplateReservation, DeviceObservationDiagnostic> {
        let failure = |required, current| {
            let mut failure = DeviceObservationDiagnostic::new(
                DeviceObservationFailureStage::Reserve,
                site,
                Some(StatisticalEvidenceUnknown::Capacity),
            );
            failure.budget = Some(DeviceObservationBudgetFailure {
                required,
                current,
                maximum: self.maximum,
            });
            failure
        };
        let payload = payload_upper_bound
            .checked_add(RESERVATION_PAYLOAD_OVERHEAD)
            .ok_or_else(|| failure(None, self.retained.load(Ordering::Acquire)))?;
        let prior = self
            .retained
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| {
                current
                    .checked_add(payload)
                    .filter(|next| *next <= self.maximum)
            })
            .map_err(|current| failure(Some(payload), current))?;
        self.peak.fetch_max(prior + payload, Ordering::AcqRel);
        Ok(DeviceObservationTemplateReservation {
            payload_upper_bound,
            lease: Arc::new(TemplatePayloadLease {
                budget: Arc::clone(self),
                payload,
            }),
        })
    }
}

#[derive(Debug)]
pub(crate) enum DeviceCostCatalogReservationError {
    Capacity {
        required_exclusive_peak_bytes: usize,
    },
    Invalid,
}

/// Unique construction lease, then owned by one immutable catalog root.
/// Cloning the root shares its payload and lease; it never releases the charge
/// while an older captured numeric root remains alive.
#[derive(Debug)]
pub(crate) struct DeviceCostCatalogPayloadLease {
    reservation: DeviceObservationTemplateReservation,
}
impl DeviceCostCatalogPayloadLease {
    pub(crate) fn grow(
        &mut self,
        additional: usize,
    ) -> Result<(), DeviceCostCatalogReservationError> {
        let next = self
            .reservation
            .payload_upper_bound
            .checked_add(additional)
            .ok_or(DeviceCostCatalogReservationError::Invalid)?;
        let lease = Arc::get_mut(&mut self.reservation.lease)
            .ok_or(DeviceCostCatalogReservationError::Invalid)?;
        let charged = lease
            .payload
            .checked_add(additional)
            .ok_or(DeviceCostCatalogReservationError::Invalid)?;
        let prior = lease
            .budget
            .retained
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| {
                current
                    .checked_add(additional)
                    .filter(|n| *n <= lease.budget.maximum)
            })
            .map_err(|_| DeviceCostCatalogReservationError::Capacity {
                required_exclusive_peak_bytes: charged,
            })?;
        lease
            .budget
            .peak
            .fetch_max(prior + additional, Ordering::AcqRel);
        lease.payload = charged;
        self.reservation.payload_upper_bound = next;
        Ok(())
    }

    pub(crate) fn release_temporary(&mut self, bytes: usize) {
        let lease = Arc::get_mut(&mut self.reservation.lease)
            .expect("passive catalog lease is never cloned");
        assert!(bytes <= self.reservation.payload_upper_bound);
        let prior = lease.budget.retained.fetch_sub(bytes, Ordering::AcqRel);
        debug_assert!(prior >= bytes);
        lease.payload -= bytes;
        self.reservation.payload_upper_bound -= bytes;
    }
}
/// A CPU-only construction reservation; dropping a failed build releases it.
pub struct DeviceObservationTemplateReservation {
    payload_upper_bound: usize,
    lease: Arc<TemplatePayloadLease>,
}
impl std::fmt::Debug for DeviceObservationTemplateReservation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceObservationTemplateReservation")
            .field("payload_upper_bound", &self.payload_upper_bound)
            .finish()
    }
}
impl DeviceObservationTemplateReservation {
    pub fn retain(
        self,
        template: Arc<dyn DeviceObservationTemplate>,
    ) -> Result<RetainedDeviceObservationTemplate, StatisticalEvidenceUnknown> {
        if template
            .retained_payload_bytes()
            .filter(|bytes| *bytes <= self.payload_upper_bound)
            .is_none()
        {
            return Err(StatisticalEvidenceUnknown::Capacity);
        }
        Ok(RetainedDeviceObservationTemplate {
            template,
            lease: self.lease,
        })
    }
}
struct TemplatePayloadLease {
    budget: Arc<DeviceObservationTemplateBudget>,
    payload: usize,
}
impl Drop for TemplatePayloadLease {
    fn drop(&mut self) {
        let prior = self
            .budget
            .retained
            .fetch_sub(self.payload, Ordering::AcqRel);
        debug_assert!(prior >= self.payload);
    }
}

#[derive(Clone)]
pub struct RetainedDeviceObservationTemplate {
    template: Arc<dyn DeviceObservationTemplate>,
    lease: Arc<TemplatePayloadLease>,
}
impl std::fmt::Debug for RetainedDeviceObservationTemplate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RetainedDeviceObservationTemplate")
            .field("payload", &self.lease.payload)
            .finish()
    }
}
impl RetainedDeviceObservationTemplate {
    pub fn packet(
        &self,
        input: FrozenObservationInput,
    ) -> Result<DeviceObservationPacket, StatisticalEvidenceUnknown> {
        let mut packet = DeviceObservationPacket::new(Arc::clone(&self.template), input)?;
        packet.retained = packet
            .retained
            .checked_add(RESERVATION_PAYLOAD_OVERHEAD)
            .ok_or(StatisticalEvidenceUnknown::Capacity)?;
        packet.retention = Some(self.clone());
        Ok(packet)
    }
    pub fn budget(&self) -> &Arc<DeviceObservationTemplateBudget> {
        &self.lease.budget
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    struct PureMetadata;
    impl DeviceObservationTemplate for PureMetadata {
        fn command_count(&self) -> usize {
            1
        }
        fn retained_payload_bytes(&self) -> Option<usize> {
            Some(128)
        }
        fn projection_retained_bytes_upper_bound(&self) -> Option<usize> {
            Some(0)
        }
        fn project(
            &self,
            _: &FrozenObservationInput,
        ) -> Result<Vec<Option<SelectedCommandCostEvidenceV1>>, StatisticalEvidenceUnknown>
        {
            panic!("budgeting must never resolve a template")
        }
    }
    #[test]
    fn observation_template_budget_retains_queue_owner_after_registry_eviction() {
        let budget = DeviceObservationTemplateBudget::new(1024).unwrap();
        let resident = budget.retain(Arc::new(PureMetadata)).unwrap();
        let charged = budget.retained_payload_bytes();
        let queued = resident.packet(FrozenObservationInput::command(1)).unwrap();
        let copied = queued.clone();
        assert_eq!(budget.retained_payload_bytes(), charged);
        drop(resident);
        drop(queued);
        assert_eq!(budget.retained_payload_bytes(), charged);
        drop(copied);
        assert_eq!(budget.retained_payload_bytes(), 0);
        assert_eq!(budget.peak_retained_payload_bytes(), charged);
    }
    #[test]
    fn observation_template_budget_failed_reservation_does_not_replace_or_leak_pool() {
        let bytes = 128 + RESERVATION_PAYLOAD_OVERHEAD;
        let budget = DeviceObservationTemplateBudget::new(bytes).unwrap();
        let first = budget.retain(Arc::new(PureMetadata)).unwrap();
        assert!(matches!(
            budget.retain(Arc::new(PureMetadata)),
            Err(StatisticalEvidenceUnknown::Capacity)
        ));
        assert_eq!(budget.retained_payload_bytes(), bytes);
        drop(first);
        assert!(budget.retain(Arc::new(PureMetadata)).is_ok());
        assert_eq!(budget.retained_payload_bytes(), 0);
        assert!(DeviceObservationTemplateBudget::new(0).is_err());
        assert!(
            DeviceObservationTemplateBudget::new(MAXIMUM_OBSERVATION_TEMPLATE_BYTES + 1).is_err()
        );
    }

    #[test]
    fn observation_template_budget_construction_and_queued_payload_share_one_limit() {
        let bytes = 128 + RESERVATION_PAYLOAD_OVERHEAD;
        let budget = DeviceObservationTemplateBudget::new(bytes).unwrap();
        let construction = budget.reserve(128).unwrap();
        assert!(budget.reserve(1).is_err());
        let template = construction.retain(Arc::new(PureMetadata)).unwrap();
        assert_eq!(budget.retained_payload_bytes(), bytes);
        let queued = template.packet(FrozenObservationInput::command(1)).unwrap();
        drop(template);
        assert!(budget.reserve(1).is_err());
        drop(queued);
        assert_eq!(budget.retained_payload_bytes(), 0);
        let too_small = budget.reserve(127).unwrap();
        assert!(too_small.retain(Arc::new(PureMetadata)).is_err());
        assert_eq!(budget.retained_payload_bytes(), 0);
        assert!(budget.reserve(usize::MAX).is_err());
        assert_eq!(budget.retained_payload_bytes(), 0);
    }

    #[test]
    fn observation_template_payload_is_dropped_before_its_budget_is_released() {
        struct CheckDrop(Arc<DeviceObservationTemplateBudget>);
        impl Drop for CheckDrop {
            fn drop(&mut self) {
                assert!(self.0.retained_payload_bytes() >= std::mem::size_of::<Self>());
            }
        }
        impl DeviceObservationTemplate for CheckDrop {
            fn command_count(&self) -> usize {
                1
            }
            fn retained_payload_bytes(&self) -> Option<usize> {
                Some(std::mem::size_of::<Self>())
            }
            fn projection_retained_bytes_upper_bound(&self) -> Option<usize> {
                Some(0)
            }
            fn project(
                &self,
                _: &FrozenObservationInput,
            ) -> Result<Vec<Option<SelectedCommandCostEvidenceV1>>, StatisticalEvidenceUnknown>
            {
                panic!("dropping CPU metadata cannot project")
            }
        }
        let budget = DeviceObservationTemplateBudget::new(1024).unwrap();
        let reservation = budget.reserve(std::mem::size_of::<CheckDrop>()).unwrap();
        let owner = reservation
            .retain(Arc::new(CheckDrop(Arc::clone(&budget))))
            .unwrap();
        let queued = owner.packet(FrozenObservationInput::command(1)).unwrap();
        drop(owner);
        drop(queued);
        assert_eq!(budget.retained_payload_bytes(), 0);
    }
}
