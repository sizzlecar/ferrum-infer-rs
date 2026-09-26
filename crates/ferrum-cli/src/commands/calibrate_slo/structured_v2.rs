//! Explicit complete-cohort live V2 capture, sharing the original executor and
//! credited output driver. Settings are frozen before the first actual request.
use super::*;
mod config;
pub(super) mod discovery;
mod driver;
pub(super) mod group;
mod report;
#[cfg(test)]
mod tests;
pub(super) use config::CaptureConfigV2;
pub(super) use driver::{collect, finish};
pub(super) use group::{GroupCaptureConfigV2, GroupReportV2};
pub(super) use report::StructuredReportV2;

#[cfg(test)]
mod discovery_tests;
