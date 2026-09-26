use super::*;
use ferrum_engine::continuous_engine::CalibrationAuditReadinessV2;
use std::num::NonZeroUsize;

/// Deliberately finite stage vocabulary, not a predicate DSL. No cost value is
/// an input. Every row belongs to the original fixed cohort in manifest order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub(super) enum AuditFrontierTriggerV2 {
    InitialPrefillReady,
    DecodeReady { generated_tokens: NonZeroUsize },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum StageMatch {
    Before,
    At,
    Passed,
    PopulationChanged,
}
impl AuditFrontierTriggerV2 {
    pub(super) fn match_stage(
        self,
        expected: usize,
        rows: impl ExactSizeIterator<Item = (usize, Option<(usize, usize)>)>,
    ) -> StageMatch {
        if expected == 0 || rows.len() != expected {
            return StageMatch::PopulationChanged;
        }
        let mut before = false;
        for (generated, prefill) in rows {
            match self {
                Self::InitialPrefillReady => {
                    if generated != 0 || !matches!(prefill, Some((0, total)) if total > 0) {
                        return StageMatch::Passed;
                    }
                }
                Self::DecodeReady { generated_tokens } => {
                    if generated > generated_tokens.get() {
                        return StageMatch::Passed;
                    }
                    before |= generated < generated_tokens.get() || prefill.is_some();
                }
            }
        }
        if before {
            StageMatch::Before
        } else {
            StageMatch::At
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum UntriggeredReason {
    StagePassedBeforeReadiness,
    OriginalCohortNoLongerComplete,
    CohortCompletedBeforeTrigger,
}

#[derive(Debug, Default, Serialize)]
pub(super) struct TriggerProgress {
    pub readiness_checks: usize,
    pub stage_seen: bool,
    pub last_readiness: Option<CalibrationAuditReadinessV2>,
    pub last_readiness_error: Option<String>,
    pub untriggered_reason: Option<UntriggeredReason>,
}
