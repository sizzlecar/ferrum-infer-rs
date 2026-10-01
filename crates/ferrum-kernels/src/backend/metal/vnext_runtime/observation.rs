//! Pure CPU metadata for actual Metal core transfers. No encoder, buffer,
//! pipeline or runtime reference is retained by the observation worker.
use super::*;
use ferrum_interfaces::execution_cost::{
    SelectedCommandCostEvidenceV1, StatisticalEvidenceUnknown, StatisticalTransferKindV1,
};

/// The factory runs immediately on the original preparation path after a CPU
/// reservation. It is never retained by the template or queued to the worker.
pub(crate) fn prepare_template(
    budget: Option<&Arc<DeviceObservationTemplateBudget>>,
    payload_upper: Option<usize>,
    freeze: impl FnOnce() -> Option<Arc<dyn DeviceObservationTemplate>>,
) -> Option<RetainedDeviceObservationTemplate> {
    let reservation = budget?.reserve(payload_upper?).ok()?;
    reservation.retain(freeze()?).ok()
}

/// Vec growth is charged conservatively, including its minimum allocation.
/// The final reservation also checks the reported actual Vec capacities.
pub(crate) fn capacity_upper(elements: usize) -> Option<usize> {
    elements.checked_mul(2)?.checked_add(4)
}

#[derive(Clone, Copy)]
pub(super) struct CoreTransfer {
    pub kind: StatisticalTransferKindV1,
    pub bytes: u64,
    pub capture: ferrum_types::SloStructuredCostCapture,
}
impl DeviceObservationTemplate for CoreTransfer {
    fn command_count(&self) -> usize {
        1
    }
    fn retained_payload_bytes(&self) -> Option<usize> {
        std::mem::size_of::<Self>().checked_add(2 * std::mem::size_of::<usize>())
    }
    fn projection_retained_bytes_upper_bound(&self) -> Option<usize> {
        SelectedCommandCostEvidenceV1::maximum_working_payload_bytes(1)?
            .checked_add(std::mem::size_of::<Option<SelectedCommandCostEvidenceV1>>())
    }
    fn project(
        &self,
        input: &FrozenObservationInput,
    ) -> Result<Vec<Option<SelectedCommandCostEvidenceV1>>, StatisticalEvidenceUnknown> {
        if !input.participant_ranges().is_empty() || !input.source_ranges().is_empty() {
            return Err(StatisticalEvidenceUnknown::CommandMismatch);
        }
        Ok(vec![core_cost_route::transfer_evidence(
            self.kind,
            self.bytes,
            input.tokens(),
            self.capture,
        )])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metal_template_factory_is_skipped_until_capacity_is_reserved() {
        let calls = std::cell::Cell::new(0);
        let budget = DeviceObservationTemplateBudget::new(1024).unwrap();
        let make = || {
            calls.set(calls.get() + 1);
            Some(Arc::new(CoreTransfer {
                kind: StatisticalTransferKindV1::Fill,
                bytes: 32,
                capture: ferrum_types::SloStructuredCostCapture::HostSettledV1,
            }) as Arc<dyn DeviceObservationTemplate>)
        };
        assert!(prepare_template(Some(&budget), Some(2048), make).is_none());
        assert_eq!(calls.get(), 0);
        let owned = prepare_template(Some(&budget), Some(256), make).unwrap();
        assert_eq!(calls.get(), 1);
        assert!(budget.retained_payload_bytes() >= 256);
        let _ = owned.packet(FrozenObservationInput::command(2)).unwrap();
        assert_eq!(calls.get(), 1);
        drop(owned);
        assert_eq!(budget.retained_payload_bytes(), 0);
    }

    #[test]
    fn metal_core_observation_demand_defers_factory_and_retains_only_actual_packet() {
        let command = MetalDeviceCommand::transfer(
            1,
            "fixture.passive.fill",
            Vec::new(),
            Vec::new(),
            Box::new(|_, _, _| panic!("CPU observation never executes")),
        )
        .with_core_transfer(
            StatisticalTransferKindV1::Fill,
            257,
            ferrum_types::SloStructuredCostCapture::HostSettledV1,
        );
        assert!(command.observation_template.is_none());
        assert!(command.core_transfer.is_some());
        assert!(command
            .observation_packet(DeviceCostObservationDemand::NotRequired, || panic!(
                "no consumer must not get or create a pool"
            ))
            .is_none());
        let small = DeviceObservationTemplateBudget::new(1).unwrap();
        assert!(command
            .observation_packet(DeviceCostObservationDemand::Required, || Some(
                small.clone()
            ))
            .is_none());
        assert_eq!(small.retained_payload_bytes(), 0);
        let budget = DeviceObservationTemplateBudget::new(4096).unwrap();
        assert_eq!(budget.retained_payload_bytes(), 0);
        let packet = command
            .observation_packet(DeviceCostObservationDemand::Required, || {
                Some(budget.clone())
            })
            .unwrap();
        let charged = budget.retained_payload_bytes();
        assert!(charged > 0);
        let copied = packet.clone();
        let _ = format!("{command:?} {packet:?}");
        drop(command);
        drop(packet);
        assert_eq!(budget.retained_payload_bytes(), charged);
        drop(copied);
        assert_eq!(budget.retained_payload_bytes(), 0);
    }

    #[test]
    fn raw_core_transfer_uses_final_logical_work_and_preserves_exact_projection() {
        for capture in [
            ferrum_types::SloStructuredCostCapture::Disabled,
            ferrum_types::SloStructuredCostCapture::HostSettledV1,
        ] {
            for kind in [
                StatisticalTransferKindV1::HostToDevice,
                StatisticalTransferKindV1::DeviceToDevice,
                StatisticalTransferKindV1::Fill,
            ] {
                let template = CoreTransfer {
                    kind,
                    bytes: 257,
                    capture,
                };
                for tokens in [0, 1, 9] {
                    let projected = template
                        .project(&FrozenObservationInput::command(tokens))
                        .unwrap();
                    assert_eq!(
                        projected[0],
                        core_cost_route::transfer_evidence(kind, 257, tokens, capture)
                    );
                    let evidence = projected[0].as_ref().unwrap();
                    evidence.validate_command(tokens, 0, 1).unwrap();
                    assert!(
                        projected.capacity()
                            * std::mem::size_of::<Option<SelectedCommandCostEvidenceV1>>()
                            + evidence.retained_payload_bytes().unwrap()
                            <= template.projection_retained_bytes_upper_bound().unwrap()
                    );
                }
            }
        }
    }

    #[test]
    fn shared_host_readback_keeps_its_existing_missing_device_command_semantics() {
        let template = CoreTransfer {
            kind: StatisticalTransferKindV1::DeviceToHost,
            bytes: 17,
            capture: ferrum_types::SloStructuredCostCapture::HostSettledV1,
        };
        assert_eq!(
            template
                .project(&FrozenObservationInput::command(1))
                .unwrap(),
            vec![None]
        );
    }
}
