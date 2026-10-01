use super::*;
impl PreparedOwnerCalibration {
    pub fn begin_cohort(&mut self, pass: usize, ordinal: usize) -> Result<()> {
        let r = (|| {
            self.now()?;
            if pass >= 3
                || self.active.is_some()
                || self.pending.is_some()
                || ordinal != self.ended[pass]
                || (0..pass).any(|p| self.ended[p] != self.declaration.cohort_plan.phases[p].len())
            {
                return Err(error("original cohort order differs"));
            }
            let c = self.declaration.cohort_plan.phases[pass]
                .get(ordinal)
                .ok_or_else(|| error("cohort is not declared"))?;
            let count = c.requests.len();
            let manifest = c.manifest_case;
            let repetition = c.repetition;
            self.charge(
                count
                    .checked_mul(size_of::<Slot>())
                    .ok_or_else(|| error("slot bytes overflow"))?,
            )?;
            let mut slots = Vec::new();
            slots.try_reserve_exact(count).map_err(error)?;
            for request in &self.declaration.cohort_plan.phases[pass][ordinal].requests {
                slots.push(Slot {
                    id: None,
                    owner: None,
                    generated: 0,
                    maximum: request.maximum_output,
                    released: false,
                    completed: false,
                });
            }
            self.active = Some(Cohort {
                pass,
                ordinal,
                admitted: 0,
                slots,
            });
            self.charge(0)?;
            self.cohort_event(
                serde_json::json!({"kind":"cohort_begin","phase":phase(pass)?,
                "cohort":ordinal,"manifest_case":manifest,"repetition":repetition}),
            )
        })();
        self.checked(r)
    }
    pub fn admitted(&mut self, sequence: &SequenceState) -> Result<usize> {
        let r = (|| {
            self.now()?;
            if self.pending.is_some() {
                return Err(error("admission crossed an issued call"));
            }
            let c = self
                .active
                .as_ref()
                .ok_or_else(|| error("admission outside cohort"))?;
            let slot = c
                .slots
                .get(c.admitted)
                .ok_or_else(|| error("cohort slots exhausted"))?;
            let frontier = sequence
                .cost_frontier
                .ok_or_else(|| error("admitted owner has no original frontier"))?;
            if slot.maximum != sequence.sampling_params.max_tokens as u64
                || !sequence.generated_tokens.is_empty()
                || sequence.prefill_tokens_processed != 0
                || sequence.model_kv.is_some()
                || sequence.credited_output.is_none()
                || c.slots
                    .iter()
                    .any(|s| s.id.as_ref() == Some(&sequence.request_id))
            {
                return Err(error(
                    "actual admitted owner differs from fresh declared slot",
                ));
            }
            let (pass, ordinal, index) = (c.pass, c.ordinal, c.admitted);
            self.cohort_event(
                serde_json::json!({"kind":"request_admitted","phase":phase(pass)?,
                "cohort":ordinal,"slot":index,"request_id":sequence.request_id,
                "maximum_output":sequence.sampling_params.max_tokens}),
            )?;
            let c = self.active.as_mut().unwrap();
            let slot = &mut c.slots[index];
            slot.id = Some(sequence.request_id.clone());
            slot.owner = Some(frontier.owner_incarnation.get());
            c.admitted += 1;
            self.charge(0)?;
            Ok(index)
        })();
        self.checked(r)
    }
    pub fn prefix_declaration_for_next_request(&self) -> Option<(u64, StructuredPrefixSlotV5)> {
        let c = self.active.as_ref()?;
        let prefix = self
            .declaration
            .prefix_plan
            .phases
            .get(c.pass)?
            .get(c.ordinal)?
            .as_ref()?;
        Some((
            prefix.release_generated,
            prefix.slots.get(c.admitted)?.clone(),
        ))
    }
    pub fn prefix_release_generated(&self) -> Option<u64> {
        let c = self.active.as_ref()?;
        self.declaration
            .prefix_plan
            .phases
            .get(c.pass)?
            .get(c.ordinal)?
            .as_ref()
            .map(|p| p.release_generated)
    }
    pub fn preparing_prefix(&self) -> bool {
        self.prefix_release_generated().is_some()
            && self
                .active
                .as_ref()
                .is_some_and(|c| c.slots.iter().any(|s| !s.released))
    }
    pub fn prefix_pending(&self) -> bool {
        self.pending.as_ref().is_some_and(|p| p.preparation)
    }
    pub fn prefix_release_order(&self) -> Vec<RequestId> {
        self.active.as_ref().map_or_else(Vec::new, |c| {
            c.slots.iter().filter_map(|s| s.id.clone()).collect()
        })
    }
    pub fn end_cohort(&mut self) -> Result<()> {
        let r = (|| {
            self.now()?;
            let c = self
                .active
                .as_ref()
                .ok_or_else(|| error("no original cohort to close"))?;
            if self.pending.is_some()
                || c.admitted != c.slots.len()
                || c.slots.iter().any(|s| !s.completed)
            {
                return Err(error("original cohort contains incomplete requests"));
            }
            let (pass, ordinal, count) = (c.pass, c.ordinal, c.slots.len());
            self.cohort_event(serde_json::json!({"kind":"cohort_end","phase":phase(pass)?,
                "cohort":ordinal,"admitted_count":count,"completed_count":count}))?;
            self.ended[pass] += 1;
            self.active = None;
            self.charge(0)
        })();
        self.checked(r)
    }
    pub(super) fn complete_ordinary_rows(
        &mut self,
        stages: &HostStageEvidenceV1,
        fifo: u64,
    ) -> Result<()> {
        let offer = self
            .pending
            .as_ref()
            .ok_or_else(|| error("ordinary completion lacks preoffer"))?;
        let c = self
            .active
            .as_ref()
            .ok_or_else(|| error("ordinary completion lacks cohort"))?;
        for row in &stages.rows {
            let slot = c
                .slots
                .iter()
                .find(|s| s.id.as_ref() == Some(&row.request_id))
                .ok_or_else(|| error("ordinary row was not admitted"))?;
            let original = offer
                .rows
                .iter()
                .find(|r| r.id == row.request_id)
                .ok_or_else(|| error("ordinary row was not offered"))?;
            let emits = match original.work {
                ActualRowWork::Decode { .. } => true,
                ActualRowWork::Prefill {
                    offset,
                    count,
                    total_prompt_tokens,
                } => offset.checked_add(count) == Some(total_prompt_tokens),
                _ => return Err(error("non-inference work in original offer")),
            };
            let next = original
                .generated
                .checked_add(u64::from(emits))
                .ok_or_else(|| error("generated count overflow"))?;
            if slot.completed
                || slot.owner != Some(row.owner_incarnation)
                || slot.generated != original.generated
                || row.work_generation != original.generation
                || next > slot.maximum
            {
                return Err(error(
                    "ordinary settlement changed the offered owner/frontier",
                ));
            }
            if let Some(terminal) = &row.terminal {
                // The source8 lifecycle also validates the actual terminal and
                // installed policy. Never manufacture EOS/Stop from candidates.
                if !emits
                    || terminal.generated_tokens != next
                    || !terminal.owner_matched
                    || !terminal.terminal_handoff_succeeded
                    || terminal.output_failed
                    || terminal.physical_failed
                    || terminal.scheduler_failed
                    || !matches!(
                        terminal.finish_reason,
                        ferrum_types::FinishReason::EOS
                            | ferrum_types::FinishReason::Stop
                            | ferrum_types::FinishReason::Length
                    )
                    || (terminal.finish_reason == ferrum_types::FinishReason::Length
                        && next != slot.maximum)
                {
                    return Err(error(
                        "ordinary terminal is not successful original completion",
                    ));
                }
            } else if next == slot.maximum {
                return Err(error("original Length terminal missing"));
            }
        }
        let (pass, cohort) = (c.pass, c.ordinal);
        // All rows were checked before any event/state update. Re-read only
        // the immutable original receipt; no temporary cloned terminal table.
        for row in &stages.rows {
            let original = self
                .pending
                .as_ref()
                .unwrap()
                .rows
                .iter()
                .find(|r| r.id == row.request_id)
                .unwrap();
            let emits = match original.work {
                ActualRowWork::Decode { .. } => true,
                ActualRowWork::Prefill {
                    offset,
                    count,
                    total_prompt_tokens,
                } => offset.checked_add(count) == Some(total_prompt_tokens),
                _ => return Err(error("non-inference work in original offer")),
            };
            let next = original
                .generated
                .checked_add(u64::from(emits))
                .ok_or_else(|| error("generated count overflow"))?;
            let slot = self
                .active
                .as_ref()
                .unwrap()
                .slots
                .iter()
                .position(|s| s.id.as_ref() == Some(&row.request_id))
                .unwrap();
            if let Some(terminal) = &row.terminal {
                let id = &row.request_id;
                let owner = row.owner_incarnation;
                self.cohort_event(serde_json::json!({"kind":"request_completed","request":{
                    "phase":phase(pass)?,"cohort":cohort,"slot":slot,"request_id":id,
                    "owner_incarnation":owner,"call_id":stages.call_id,"fifo":fifo,
                    "generated_tokens":next,"terminal":terminal}}))?;
                self.active.as_mut().unwrap().slots[slot].completed = true;
            }
            self.active.as_mut().unwrap().slots[slot].generated = next;
        }
        Ok(())
    }
}
