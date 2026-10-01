//! A declared input-only route population. Excluded attempts still consume
//! the original offer quota and require complete original execution evidence.
use serde::{Deserialize, Serialize};
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SloCalibrationRoutePopulationV1 {
    #[default]
    AllAttempts,
    WarmOrGraphDisabledV1,
    /// Original attempts remain in the quota, including privately proven
    /// pre-submit deferrals. Such attempts have no numerical timing sample.
    WarmOrGraphDisabledWithNoSubmissionV2,
}
impl SloCalibrationRoutePopulationV1 {
    pub const fn allows_no_submission(&self) -> bool {
        matches!(self, Self::WarmOrGraphDisabledWithNoSubmissionV2)
    }

    pub const fn is_all_attempts(&self) -> bool {
        matches!(self, Self::AllAttempts)
    }
}
