//! Compiled iterative demand is planning evidence, not a device reservation.

use std::{error::Error, fmt};

use serde::Serialize;

/// The complete device peak for a prepared request batch under its selected
/// residency policy. Native workspace is already included in the compiled
/// stage peak; `native_allowance_bytes` covers separately reserved owners.
///
/// A fit against the configured ceiling does not grant capacity currently
/// held by another request or account. Runtime admission remains authoritative.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct DeviceCapacityReport {
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

impl DeviceCapacityReport {
    pub fn fits(&self) -> bool {
        self.required_bytes <= self.capacity_bytes
    }

    pub fn validate(self) -> Result<(), DeviceCapacityExceeded> {
        if self.fits() {
            Ok(())
        } else {
            Err(DeviceCapacityExceeded { report: self })
        }
    }
}

/// A proven failure against the configured total budget, before execution.
/// This is distinct from temporarily unavailable capacity in the live ledger,
/// provider planning errors, and failures that retain execution owners.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceCapacityExceeded {
    pub report: DeviceCapacityReport,
}

impl fmt::Display for DeviceCapacityExceeded {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let report = self.report;
        if !report.complete_batch {
            return write!(
                f,
                "stage lower bound requires {} device bytes, exceeding usable capacity {}",
                report.required_bytes, report.capacity_bytes
            );
        }
        write!(f,
            "resident batch needs {} device bytes (retained static {}, active stage peak {}, total idle dynamic allowance {}, native allowance {}), exceeding total ceiling {}",
            report.required_bytes, report.retained_static_bytes,
            report.active_stage_peak_bytes, report.retained_dynamic_allowance_bytes,
            report.native_allowance_bytes, report.capacity_bytes)
    }
}

impl Error for DeviceCapacityExceeded {}
