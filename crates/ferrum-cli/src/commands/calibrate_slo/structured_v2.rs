//! Explicit complete-cohort live V2 capture, sharing the original executor and
//! credited output driver. Settings are frozen before the first actual request.
use super::*;
mod config;
mod driver;
mod report;
#[cfg(test)]
mod tests;
pub(super) use config::CaptureConfigV2;
pub(super) use driver::{collect, finish};
pub(super) use report::StructuredReportV2;
