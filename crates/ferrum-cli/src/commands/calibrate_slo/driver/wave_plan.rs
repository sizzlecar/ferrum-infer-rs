//! Calibration choices are a declared sequence, not a response to cost or timing.
//! This module owns no execution authority and never advances a choice because of a retry.
use super::*;
use ferrum_engine::continuous_engine::{
    CalibrationDecodeRoute, CalibrationWaveReport, HostRowStageV1, HostStageCompleteness,
    HostStageEvidenceV1, HostStageWork,
};
use ferrum_interfaces::execution_cost::ActualRowWork;
use ferrum_types::RequestId;
use serde::Serialize;
use std::num::NonZeroU32;

pub(super) struct Cursor<'a> {
    case: &'a manifest::Cohort,
    prefill_completed: u64,
    decode_completed: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub(super) struct Choice {
    prefill_completed: u64,
    decode_completed: u64,
    prefill_cycle_index: Option<usize>,
    decode_cycle_index: Option<usize>,
    pub prefill_chunk_tokens: NonZeroU32,
    pub decode_route: CalibrationDecodeRoute,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum Work {
    Prefill {
        offset: u32,
        count: u32,
        total_prompt_tokens: u32,
    },
    Decode {
        kv_tokens: u32,
    },
}

impl Work {
    fn actual(value: ActualRowWork) -> Result<Self> {
        match value {
            ActualRowWork::Prefill {
                offset,
                count,
                total_prompt_tokens,
            } if count > 0
                && offset
                    .checked_add(count)
                    .is_some_and(|end| end <= total_prompt_tokens) =>
            {
                Ok(Self::Prefill {
                    offset,
                    count,
                    total_prompt_tokens,
                })
            }
            ActualRowWork::Decode { kv_tokens } => Ok(Self::Decode { kv_tokens }),
            _ => Err(invalid("wave plan requires real prefill or decode work")),
        }
    }

    fn matches_host(self, value: HostStageWork) -> bool {
        matches!((self, value),
            (Self::Prefill { offset: a, count: b, total_prompt_tokens: c },
             HostStageWork::Prefill { offset: x, count: y, total_prompt_tokens: z }) if (a,b,c)==(x,y,z))
            || matches!((self, value), (Self::Decode { kv_tokens: a }, HostStageWork::Decode { kv_tokens: b }) if a==b)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct Row {
    request_id: RequestId,
    owner_incarnation: u64,
    work_generation: u64,
    work: Work,
}

impl Row {
    fn matches_host(&self, value: &HostRowStageV1) -> bool {
        self.request_id == value.request_id
            && self.owner_incarnation == value.owner_incarnation
            && self.work_generation == value.work_generation
            && self.work.matches_host(value.actual_work)
    }
}

#[derive(Debug, Serialize)]
pub(super) struct Attempt {
    pub choice: Choice,
    // At most the engine's existing 256 rows; no retained buffers or permits.
    rows: Vec<Row>,
}

struct ReportedRow {
    row: Row,
    full_logits: bool,
}

impl<'a> Cursor<'a> {
    pub fn new(case: &'a manifest::Cohort) -> Result<Option<Self>> {
        let Some(plan) = &case.wave_plan else {
            return Ok(None);
        };
        plan.validate()?;
        Ok(Some(Self {
            case,
            prefill_completed: 0,
            decode_completed: 0,
        }))
    }

    pub fn choice(&self) -> Choice {
        let plan = self
            .case
            .wave_plan
            .as_ref()
            .expect("validated at construction");
        let prefill_cycle_index = plan
            .prefill_chunks
            .as_ref()
            .map(|values| (self.prefill_completed % values.len() as u64) as usize);
        let decode_cycle_index = plan
            .decode_routes
            .as_ref()
            .map(|values| (self.decode_completed % values.len() as u64) as usize);
        Choice {
            prefill_completed: self.prefill_completed,
            decode_completed: self.decode_completed,
            prefill_cycle_index,
            decode_cycle_index,
            prefill_chunk_tokens: prefill_cycle_index.map_or(self.case.prefill_chunk_tokens, |i| {
                plan.prefill_chunks.as_ref().unwrap()[i]
            }),
            decode_route: decode_cycle_index.map_or(self.case.decode_route, |i| {
                plan.decode_routes.as_ref().unwrap()[i]
            }),
        }
    }

    pub fn begin(&self, rows: &[CalibrationWork]) -> Result<Attempt> {
        let rows = rows
            .iter()
            .map(|row| {
                Ok(Row {
                    request_id: row.frontier().request_id().clone(),
                    owner_incarnation: row.frontier().owner_incarnation().get(),
                    work_generation: row.frontier().work_generation().get(),
                    work: Work::actual(row.work())?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        self.begin_rows(rows)
    }

    fn begin_rows(&self, rows: Vec<Row>) -> Result<Attempt> {
        if rows.is_empty() || rows.len() > 256 || rows.iter().enumerate().any(|(i, row)| {
            row.owner_incarnation == 0 || row.work_generation == 0
                || rows[..i].iter().any(|other| row.request_id == other.request_id)
                || matches!(row.work, Work::Prefill { count, .. } if count > self.choice().prefill_chunk_tokens.get())
        }) {
            return Err(invalid("wave plan has invalid or duplicate work"));
        }
        Ok(Attempt {
            choice: self.choice(),
            rows,
        })
    }

    pub fn reconcile(&mut self, attempt: &Attempt, report: &CalibrationWaveReport) -> Result<bool> {
        let rows = report
            .ordered_work
            .participants()
            .iter()
            .map(|participant| {
                let row = participant.selection();
                Ok(ReportedRow {
                    row: Row {
                        request_id: row.request_id.clone(),
                        owner_incarnation: row.owner_incarnation.get(),
                        work_generation: row.work_generation.get(),
                        work: Work::actual(row.work)?,
                    },
                    full_logits: row
                        .decode_policy
                        .as_ref()
                        .is_some_and(|policy| policy.requires_full_logits()),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        self.apply(
            attempt,
            report.submission,
            report.error.is_some(),
            &rows,
            report.host_stages.as_deref(),
        )
    }

    fn apply(
        &mut self,
        attempt: &Attempt,
        submission: CalibrationSubmissionState,
        failed: bool,
        reported: &[ReportedRow],
        host: Option<&HostStageEvidenceV1>,
    ) -> Result<bool> {
        if attempt.choice != self.choice() {
            return Err(invalid(
                "wave plan cannot consume another or already consumed ordinal",
            ));
        }
        if failed {
            return Err(invalid("wave plan cannot advance a failed wave"));
        }
        match submission {
            CalibrationSubmissionState::NotSubmitted => return Ok(false),
            CalibrationSubmissionState::HostReconciled => {}
            _ => {
                return Err(invalid(
                    "wave plan requires conclusive host reconciliation; refusing replay",
                ))
            }
        }
        // HostReconciled alone is not an all-owner success receipt. Correlate
        // each actual commit, including terminal cleanup, with this exact work.
        // Cost Known, queue acceptance, fit eligibility and elapsed values do
        // not select the next option. Additional *successful* cleanup may be
        // ineligible as a cost sample, yet still represents completed work.
        let host = host.ok_or_else(|| invalid("wave plan has no actual host commit evidence"))?;
        let successful = |status| {
            matches!(
                status,
                HostStageCompleteness::CompleteSingleWave
                    | HostStageCompleteness::AdditionalOrUnknownWork
            )
        };
        if !successful(host.completeness)
            || host.finalized_at_ns.is_none()
            || reported.len() != attempt.rows.len()
            || host.rows.len() != attempt.rows.len()
            || reported.iter().enumerate().any(|(i, row)| {
                reported[..i]
                    .iter()
                    .any(|prior| prior.row.request_id == row.row.request_id)
            })
            || host.rows.iter().enumerate().any(|(i, row)| {
                host.rows[..i]
                    .iter()
                    .any(|prior| prior.request_id == row.request_id)
            })
        {
            return Err(invalid(
                "wave plan host result is incomplete or has unrelated rows",
            ));
        }
        for requested in &attempt.rows {
            let actual = reported
                .iter()
                .find(|row| row.row == *requested)
                .ok_or_else(|| {
                    invalid("wave plan actual owner/frontier/work differs from its attempt")
                })?;
            if matches!(requested.work, Work::Decode { .. })
                && attempt.choice.decode_route == CalibrationDecodeRoute::FullLogits
                && !actual.full_logits
            {
                return Err(invalid(
                    "wave plan FullLogits work was not the submitted policy",
                ));
            }
            let row = host
                .rows
                .iter()
                .find(|row| requested.matches_host(row))
                .ok_or_else(|| invalid("wave plan has no matching actual committed row"))?;
            if !successful(row.completeness)
                || row.token_committed_at_ns.is_none()
                || row.settled_at_ns.is_none()
                || row.terminal.as_ref().is_some_and(|terminal| {
                    !terminal.owner_matched
                        || terminal.output_failed
                        || terminal.physical_failed
                        || terminal.scheduler_failed
                        || !terminal.terminal_handoff_succeeded
                        || !terminal.request_slot_closed
                })
            {
                return Err(invalid("wave plan row did not complete successfully"));
            }
        }
        let prefill = attempt
            .rows
            .iter()
            .any(|row| matches!(row.work, Work::Prefill { .. }));
        let decode = attempt
            .rows
            .iter()
            .any(|row| matches!(row.work, Work::Decode { .. }));
        // Compute both before publishing either; even overflow cannot half-
        // advance a mixed wave. Counters are also bounded by the existing
        // protocol's maximum_wave_attempts, never a new retry allowance.
        let p = self
            .prefill_completed
            .checked_add(u64::from(prefill))
            .ok_or_else(|| invalid("prefill wave ordinal overflow"))?;
        let d = self
            .decode_completed
            .checked_add(u64::from(decode))
            .ok_or_else(|| invalid("decode wave ordinal overflow"))?;
        self.prefill_completed = p;
        self.decode_completed = d;
        Ok(true)
    }
}

fn invalid(message: &'static str) -> FerrumError {
    FerrumError::invalid_request(message)
}

#[cfg(test)]
mod tests;
