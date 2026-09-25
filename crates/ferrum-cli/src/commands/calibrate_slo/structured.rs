//! Explicit live three-phase calibration. No raw JSON can construct a live
//! receipt, and reaching a desired count never cancels an output owner.
use super::*;

mod config;
mod driver;
mod report;
#[cfg(test)]
mod tests;

pub(super) use config::{export_limits, CaptureConfig, Settings};
pub(super) use driver::{collect, finish};
pub(super) use report::StructuredReport;
