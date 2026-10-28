//! Compiled image demand is planning evidence, not a device reservation.

use std::{error::Error, fmt};

use serde::Serialize;

/// The complete device peak for a prepared request batch under its selected
/// residency policy. Native workspace is already included in the compiled
/// stage peak; `native_allowance_bytes` covers separately reserved owners.
///
/// A fit against the configured ceiling does not grant capacity currently
/// held by another request or account. Runtime admission remains authoritative.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ImageDeviceCapacityReport {
    /// False when an individual stage already exceeds its planner ceiling.
    /// Such a lower bound proves failure, but never proves that a batch fits.
    pub complete_batch: bool,
    pub required_bytes: u64,
    pub capacity_bytes: u64,
    pub retained_static_bytes: u64,
    pub active_stage_peak_bytes: u64,
    pub retained_dynamic_allowance_bytes: u64,
    pub native_allowance_bytes: u64,
}

impl ImageDeviceCapacityReport {
    pub fn fits(&self) -> bool {
        self.required_bytes <= self.capacity_bytes
    }

    pub fn validate(self) -> Result<(), ImageDeviceCapacityExceeded> {
        if self.fits() {
            Ok(())
        } else {
            Err(ImageDeviceCapacityExceeded { report: self })
        }
    }
}

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

/// A proven failure against the configured total budget, before execution.
/// This is distinct from temporarily unavailable capacity in the live ledger,
/// provider planning errors, and failures that retain execution owners.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageDeviceCapacityExceeded {
    pub report: ImageDeviceCapacityReport,
}

impl fmt::Display for ImageDeviceCapacityExceeded {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let report = self.report;
        if !report.complete_batch {
            return write!(
                f,
                "image stage lower bound requires {} device bytes, exceeding usable capacity {}",
                report.required_bytes, report.capacity_bytes
            );
        }
        write!(f,
            "resident image batch needs {} device bytes (retained static {}, active stage peak {}, total idle dynamic allowance {}, native allowance {}), exceeding total ceiling {}",
            report.required_bytes, report.retained_static_bytes,
            report.active_stage_peak_bytes, report.retained_dynamic_allowance_bytes,
            report.native_allowance_bytes, report.capacity_bytes)
    }
}

impl Error for ImageDeviceCapacityExceeded {}

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
