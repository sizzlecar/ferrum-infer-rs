//! Complete-cohort multi-owner capture. Children are declared before warmup;
//! source3/profile10 and the product catalog loader remain the authorities.
use super::*;
use ferrum_engine::continuous_engine::{
    CalibrationSession, StructuredCalibrationGroupOptionsV2, StructuredCapturePhase,
};
use serde::{Deserialize, Serialize};
mod config;
mod driver;
mod export;
mod prefix;
pub(in crate::commands::calibrate_slo) use prefix::validate_prefix_v5;
#[cfg(test)]
mod tests;
pub(in crate::commands::calibrate_slo) use config::GroupCaptureConfigV2;
pub(in crate::commands::calibrate_slo) use driver::{collect, finish};

#[derive(Serialize)]
pub(crate) struct GroupReportV2 {
    pub schema_version: u32,
    pub scope_note: &'static str,
    pub declared_limits: config::GroupLimitsV2,
    pub children: Vec<report::StructuredReportV2>,
    pub group_failure: Option<String>,
    pub exported_shared_profile: Option<ferrum_scheduler::implementations::continuous::cost_profile::StructuredProfileExportReceiptV11>,
    pub exported_prefix_profile: Option<ferrum_scheduler::implementations::continuous::cost_profile::StructuredProfileExportReceiptV12>,
    pub verified_catalog: Option<ferrum_types::SloCostProfileReceipt>,
    pub finalization_error: Option<String>,
}
impl GroupReportV2 {
    fn new(config: &GroupCaptureConfigV2) -> Self {
        Self {
            schema_version: 1,
            scope_note: "predeclared distinct owners share complete cohorts and original wall observations; every child retains original FIFO/outside/phase clocks; all children and the original catalog loader must succeed; this does not establish full-horizon coverage or serving SLO compliance",
            declared_limits: config.limits.clone(),
            children: config.children.iter().map(|c| report::StructuredReportV2::new(c.scope.clone())).collect(),
            group_failure: None,
            exported_shared_profile: None,
            exported_prefix_profile: None,
            verified_catalog: None,
            finalization_error: None,
        }
    }
}
