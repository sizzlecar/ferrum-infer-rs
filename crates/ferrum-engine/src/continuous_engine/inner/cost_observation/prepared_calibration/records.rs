use super::super::host_stages::ExportEvidence;
use super::*;
use crate::continuous_engine::inner::calibration::token_preparation::{
    CapturedPrefixReleaseV5, CapturedPrefixWaveV5, PrefixPreparedRowV5,
};

impl PreparedOwnerCalibration {
    fn allocate_offer(&mut self, preparation: bool, count: usize) -> Result<Offer> {
        let c = self
            .active
            .as_ref()
            .ok_or_else(|| error("offer outside original cohort"))?;
        if self.pending.is_some()
            || count == 0
            || count > c.slots.len()
            || c.admitted != c.slots.len()
            || preparation != self.preparing_prefix()
        {
            return Err(error(
                "offer differs from complete declared cohort/prefix state",
            ));
        }
        let ticket = self
            .offered
            .checked_add(1)
            .filter(|t| *t <= self.declaration.maximum_offered_waves as u64)
            .ok_or_else(|| error("original offered capacity exhausted"))?;
        self.ensure_block()?;
        let issued_not_before_ns = self.now()?;
        self.charge(
            count
                .checked_mul(size_of::<OfferedRow>())
                .ok_or_else(|| error("offer bytes overflow"))?,
        )?;
        let mut rows = Vec::new();
        rows.try_reserve_exact(count).map_err(error)?;
        Ok(Offer {
            ticket,
            issued_not_before_ns,
            preparation,
            rows,
        })
    }
    fn validate_offered_row(&self, row: &OfferedRow) -> Result<()> {
        let c = self
            .active
            .as_ref()
            .ok_or_else(|| error("offer outside cohort"))?;
        let s = c
            .slots
            .iter()
            .find(|s| s.id.as_ref() == Some(&row.id))
            .ok_or_else(|| error("offer references unadmitted owner"))?;
        if s.completed
            || s.owner != Some(row.owner)
            || s.generated != row.generated
            || row.generation == 0
        {
            return Err(error("offer changed the original owner/frontier"));
        }
        Ok(())
    }
    pub fn offer(&mut self, work: &[CalibrationWork]) -> Result<u64> {
        let r = (|| {
            let mut p = self.allocate_offer(false, work.len())?;
            for w in work {
                let f = w.frontier();
                let row = OfferedRow {
                    id: f.request_id().clone(),
                    owner: f.owner_incarnation().get(),
                    generation: f.work_generation().get(),
                    generated: f.generated_tokens() as u64,
                    work: w.work(),
                };
                self.validate_offered_row(&row)?;
                if p.rows.iter().any(|r| r.id == row.id) {
                    return Err(error("duplicate original offered owner"));
                }
                p.rows.push(row);
            }
            let ticket = p.ticket;
            self.pending = Some(p);
            self.offered = ticket;
            self.charge(0)?;
            Ok(ticket)
        })();
        self.checked(r)
    }
    pub fn offer_preparation(&mut self, rows: &[PrefixPreparedRowV5]) -> Result<u64> {
        let r = (|| {
            let mut p = self.allocate_offer(true, rows.len())?;
            for original in rows {
                let b = original.before();
                let work=match original.work() {
                    ferrum_scheduler::implementations::continuous::cost_model::structured_v2::windows::PreparedWorkV2::Prefill{offset,count,total_prompt_tokens}=>ActualRowWork::Prefill{offset,count,total_prompt_tokens},
                    ferrum_scheduler::implementations::continuous::cost_model::structured_v2::windows::PreparedWorkV2::Decode{kv_tokens}=>ActualRowWork::Decode{kv_tokens},
                };
                let row = OfferedRow {
                    id: b.request_id.clone(),
                    owner: b.owner_incarnation,
                    generation: b.work_generation,
                    generated: b.generated_tokens as u64,
                    work,
                };
                self.validate_offered_row(&row)?;
                if p.rows.iter().any(|r| r.id == row.id) {
                    return Err(error("duplicate original preparation owner"));
                }
                p.rows.push(row);
            }
            let ticket = p.ticket;
            self.pending = Some(p);
            self.offered = ticket;
            self.charge(0)?;
            let c = self.active.as_ref().unwrap();
            #[derive(Serialize)]
            struct Offered<'a> {
                kind: &'static str,
                offered: u64,
                phase: file::StructuredProfilePhaseV10,
                cohort: usize,
                rows: &'a [PrefixPreparedRowV5],
            }
            let v = serde_json::to_value(Offered {
                kind: "preparation_offered",
                offered: ticket,
                phase: phase(c.pass)?,
                cohort: c.ordinal,
                rows,
            })
            .map_err(error)?;
            self.preparation_event(v)?;
            Ok(ticket)
        })();
        self.checked(r)
    }
    fn original_stage_position(
        &self,
        stages: &HostStageEvidenceV1,
        queue: Option<HostStageQueueReceipt>,
    ) -> Result<u64> {
        let p = self
            .pending
            .as_ref()
            .ok_or_else(|| error("settlement has no original offer"))?;
        let q = queue.ok_or_else(|| error("original FIFO receipt missing"))?;
        let fifo = q
            .accepted_ordinal
            .ok_or_else(|| error("original FIFO did not accept call"))?;
        if q.disposition != HostStageQueueDisposition::Published
            || self.last_fifo.checked_add(1) != Some(fifo)
            || stages.call_id <= self.last_call
            || stages.rows.len() != p.rows.len()
            || stages.completeness != HostStageCompleteness::CompleteSingleWave
            || stages
                .prepare_started_at_ns
                .is_none_or(|t| t < p.issued_not_before_ns)
            || stages
                .finalized_at_ns
                .is_none_or(|t| t > self.deadline || t < p.issued_not_before_ns)
        {
            return Err(error("original settlement clock/FIFO/call/count differs"));
        }
        for row in &stages.rows {
            let original = p
                .rows
                .iter()
                .find(|r| r.id == row.request_id)
                .ok_or_else(|| error("settled unoffered row"))?;
            let work = match row.actual_work {
                HostStageWork::Prefill {
                    offset,
                    count,
                    total_prompt_tokens,
                } => ActualRowWork::Prefill {
                    offset,
                    count,
                    total_prompt_tokens,
                },
                HostStageWork::Decode { kv_tokens } => ActualRowWork::Decode { kv_tokens },
                _ => return Err(error("settled non-inference work")),
            };
            if row.owner_incarnation != original.owner
                || row.work_generation != original.generation
                || work != original.work
                || row.completeness != HostStageCompleteness::CompleteSingleWave
            {
                return Err(error("settlement changed original offered identity/work"));
            }
        }
        Ok(fifo)
    }
    pub fn complete_preparation(&mut self, captured: &CapturedPrefixWaveV5) -> Result<()> {
        let r = (|| {
            if !self.prefix_pending() {
                return Err(error(
                    "preparation completed without original preparation offer",
                ));
            }
            let e = captured.evidence();
            let p = self.pending.as_ref().unwrap();
            let c = self.active.as_ref().unwrap();
            #[derive(Serialize)]
            struct Completed<'a> {
                kind: &'static str,
                offered: u64,
                phase: file::StructuredProfilePhaseV10,
                cohort: usize,
                reconciled: bool,
                queue: Option<HostStageQueueReceipt>,
                host_stages: Option<ExportEvidence<'a>>,
                rows: &'a [PrefixRowEvidenceV1],
                failure: Option<&'a str>,
            }
            let value = serde_json::to_value(Completed {
                kind: "preparation_completed",
                offered: p.ticket,
                phase: phase(c.pass)?,
                cohort: c.ordinal,
                reconciled: e.submission == CalibrationSubmissionState::HostReconciled
                    && e.error.is_none(),
                queue: e.host_stage_queue,
                host_stages: e
                    .host_stages
                    .as_deref()
                    .map(|s| ExportEvidence::new(s, true)),
                rows: &e.rows,
                failure: e.error.as_deref().or(e.chain_error.as_deref()),
            })
            .map_err(error)?;
            // The original failed evidence reaches validation; it never becomes
            // an ordinary Completed or a numerical sample.
            self.preparation_event(value)?;
            if e.error.is_some()
                || e.chain_error.is_some()
                || e.submission != CalibrationSubmissionState::HostReconciled
            {
                return Err(error("preparation was not reconciled"));
            }
            let stages = e
                .host_stages
                .as_deref()
                .ok_or_else(|| error("preparation settlement missing"))?;
            let fifo = self.original_stage_position(stages, e.host_stage_queue)?;
            for row in &e.rows {
                let after = row
                    .after
                    .as_ref()
                    .ok_or_else(|| error("preparation terminated before release"))?;
                let slot = self
                    .active
                    .as_mut()
                    .unwrap()
                    .slots
                    .iter_mut()
                    .find(|s| s.id.as_ref() == Some(&row.before.request_id))
                    .ok_or_else(|| error("preparation completed unadmitted slot"))?;
                if slot.completed
                    || slot.generated != row.before.generated_tokens as u64
                    || slot.owner != Some(after.owner_incarnation)
                    || after.generated_tokens as u64 >= slot.maximum
                {
                    return Err(error("preparation actual frontier differs"));
                }
                slot.generated = after.generated_tokens as u64;
            }
            self.settled(stages.call_id, fifo)
        })();
        self.checked(r)
    }
    pub fn released(&mut self, captured: &CapturedPrefixReleaseV5) -> Result<()> {
        let r = (|| {
            self.now()?;
            if self.pending.is_some() {
                return Err(error("release crossed unresolved call"));
            }
            let e = captured.receipt();
            let c = self
                .active
                .as_ref()
                .ok_or_else(|| error("release outside cohort"))?;
            let index = c
                .slots
                .iter()
                .position(|s| s.id.as_ref() == Some(&e.frontier.request_id))
                .ok_or_else(|| error("release owner never admitted"))?;
            let s = &c.slots[index];
            if s.released
                || s.completed
                || s.owner != Some(e.frontier.owner_incarnation)
                || s.generated != e.frontier.generated_tokens as u64
                || e.through_fifo_ordinal > self.last_fifo
            {
                return Err(error("release differs from original owner/frontier"));
            }
            self.preparation_event(
                serde_json::json!({"kind":"preparation_released","phase":phase(c.pass)?,
                "cohort":c.ordinal,"slot":index,"receipt":e}),
            )?;
            self.active.as_mut().unwrap().slots[index].released = true;
            Ok(())
        })();
        self.checked(r)
    }
    fn validate_report_work(&self, report: &CalibrationWaveReport) -> Result<()> {
        let p = self
            .pending
            .as_ref()
            .ok_or_else(|| error("report has no original offer"))?;
        if report.ordered_work.participants().len() != p.rows.len() {
            return Err(error("report row count differs from offer"));
        }
        for selected in report.ordered_work.participants() {
            let s = selected.selection();
            let r = p
                .rows
                .iter()
                .find(|r| r.id == s.request_id)
                .ok_or_else(|| error("report contains unoffered owner"))?;
            if r.owner != s.owner_incarnation.get()
                || r.generation != s.work_generation.get()
                || r.work != s.work
            {
                return Err(error("report changed offered exact work"));
            }
        }
        Ok(())
    }
    pub fn ordinary(&mut self, report: &CalibrationWaveReport) -> Result<()> {
        if report.no_submission_proof().is_some() {
            return self.no_submission(report);
        }
        let r = (|| {
            if self.pending.as_ref().is_none_or(|p| p.preparation) || self.preparing_prefix() {
                return Err(error("ordinary sample crossed preparation"));
            }
            self.validate_report_work(report)?;
            if report.error.is_some()
                || report.submission != CalibrationSubmissionState::HostReconciled
            {
                return Err(error("ordinary call was not reconciled"));
            }
            let stages = report
                .host_stages
                .as_deref()
                .ok_or_else(|| error("ordinary host settlement missing"))?;
            let fifo = self.original_stage_position(stages, report.host_stage_queue)?;
            let ticket = self.pending.as_ref().unwrap().ticket;
            let issued = stages
                .prepare_started_at_ns
                .ok_or_else(|| error("original preparation clock missing"))?;
            let record = if let Some(route) =
                stages.route_evidence.as_ref().filter(|r| r.is_outside())
            {
                file::StructuredServiceRecordV7::OutsideDeclaredRoute {
                    wave: file::StructuredServiceOutsideRouteV7::from_diagnostic(
                        ticket,
                        issued,
                        fifo,
                        serde_json::to_value(route.outside_diagnostic(stages)).map_err(error)?,
                    )
                    .map_err(error)?,
                }
            } else {
                let independent = stages
                    .statistical_evidence
                    .as_ref()
                    .and_then(|s| s.independent_attention_v2())
                    .map(|s| s.to_wire_v2());
                let wave = file::StructuredServiceWaveV7::from_diagnostic(
                    ticket,
                    issued,
                    fifo,
                    serde_json::to_value(stages.structured_source_view()).map_err(error)?,
                    independent,
                )
                .map_err(error)?;
                let wave = if let Some(route) = &stages.route_evidence {
                    wave.with_prepared_route(
                        serde_json::to_value(route.eligible_diagnostic()).map_err(error)?,
                    )
                    .map_err(error)?
                } else {
                    wave
                };
                file::StructuredServiceRecordV7::Completed { wave }
            };
            self.push(&Record::Population(record))?;
            self.complete_ordinary_rows(stages, fifo)?;
            self.settled(stages.call_id, fifo)
        })();
        self.checked(r)
    }
    pub fn no_submission(&mut self, report: &CalibrationWaveReport) -> Result<()> {
        let r = (|| {
            self.validate_report_work(report)?;
            let p = self.pending.as_ref().unwrap();
            let proof = report
                .no_submission_proof()
                .ok_or_else(|| error("original no-submission proof missing"))?;
            if report.error.is_some()
                || report.submission != CalibrationSubmissionState::NotSubmitted
                || report.host_stages.is_some()
                || proof.call_id() <= self.last_call
                || self.last_fifo.checked_add(1) != Some(proof.fifo())
                || proof.issued_at_ns() < p.issued_not_before_ns
                || proof.observed_at_ns() > self.deadline
                || proof.participants().len() != p.rows.len()
            {
                return Err(error("original no-submission identity/clock/FIFO differs"));
            }
            for (index, original) in proof.participants().iter().enumerate() {
                let selected = report.ordered_work.participants()[index].selection();
                if original.input_index as usize != index
                    || original.request_id != selected.request_id
                    || original.owner_incarnation != selected.owner_incarnation.get()
                    || original.work_generation != selected.work_generation.get()
                {
                    return Err(error(
                        "no-submission proof changed the originally offered participants",
                    ));
                }
            }
            let ticket = p.ticket;
            let wire=proof.bind_source_position(ticket,
                ferrum_scheduler::implementations::continuous::cost_model::structured_v2::StructuredPhaseV2::Fit).map_err(error)?;
            let attempt =
                file::StructuredServiceNoSubmissionV7::from_original_record(&wire, ticket)
                    .map_err(error)?;
            if p.preparation {
                // Source8 explicitly settles an unsuccessful preparation call;
                // it consumes the original ticket and advances no frontier.
                let c = self.active.as_ref().unwrap();
                let record = Record::preparation_not_submitted(phase(c.pass)?, c.ordinal, attempt);
                self.push(&record)?;
            } else {
                self.push(&Record::Population(
                    file::StructuredServiceRecordV7::NotSubmitted { attempt },
                ))?;
            }
            self.settled(proof.call_id(), proof.fifo())
        })();
        self.checked(r)
    }
}
