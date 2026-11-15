//! Image planner classification; the capacity proof belongs to the common worker.
pub use crate::iterative_service::capacity::{
    DeviceCapacityExceeded as ImageDeviceCapacityExceeded,
    DeviceCapacityReport as ImageDeviceCapacityReport,
};
/// Preserve the original planner error in the chain. Only this typed fresh-plan
/// capacity failure is a shrink signal: unsupported providers, native cache
/// limits and runtime admission/allocator failures are deliberately unchanged.
pub(super) fn classify_plan_error(source: anyhow::Error) -> anyhow::Error {
    let Some(ferrum_runtime::VNextError::PlanCapacityExceeded {
        required_bytes,
        usable_capacity_bytes,
    }) = source.downcast_ref::<ferrum_runtime::VNextError>()
    else {
        return source;
    };
    let report = ImageDeviceCapacityReport {
        complete_batch: false,
        required_bytes: *required_bytes,
        capacity_bytes: *usable_capacity_bytes,
        retained_static_bytes: 0,
        active_stage_peak_bytes: *required_bytes,
        retained_dynamic_allowance_bytes: 0,
        native_allowance_bytes: 0,
    };
    source.context(ImageDeviceCapacityExceeded { report })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrum_runtime::VNextError;

    #[test]
    fn only_typed_fresh_plan_capacity_is_classified_as_shrinkable() {
        let source = anyhow::Error::new(VNextError::PlanCapacityExceeded {
            required_bytes: 1025,
            usable_capacity_bytes: 1024,
        })
        .context("stage preparation");
        let classified = classify_plan_error(source);
        let proof = classified
            .downcast_ref::<ImageDeviceCapacityExceeded>()
            .unwrap();
        assert_eq!(proof.report.required_bytes, 1025);
        assert_eq!(proof.report.capacity_bytes, 1024);
        assert!(!proof.report.complete_batch);
        assert!(classified.downcast_ref::<VNextError>().is_some());
        for source in [
            anyhow::anyhow!("native plan cache capacity exceeded"),
            anyhow::Error::new(VNextError::InvalidExecutionPlan {
                reason: "device capacity unavailable after submission".into(),
            }),
        ] {
            assert!(classify_plan_error(source)
                .downcast_ref::<ImageDeviceCapacityExceeded>()
                .is_none());
        }
    }

    #[test]
    fn complete_capacity_proof_accepts_exact_budget_and_rejects_one_byte_short() {
        let mut report = ImageDeviceCapacityReport {
            complete_batch: true,
            required_bytes: 1024,
            capacity_bytes: 1024,
            retained_static_bytes: 512,
            active_stage_peak_bytes: 384,
            retained_dynamic_allowance_bytes: 0,
            native_allowance_bytes: 128,
        };
        assert!(report.validate().is_ok());
        report.capacity_bytes -= 1;
        assert_eq!(report.validate().unwrap_err().report, report);
    }
}
