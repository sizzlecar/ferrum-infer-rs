use super::*;
use crate::implementations::continuous::cost_profile::structured_v10::service::StructuredNativePrefixAcquisitionPlanV1;
use ferrum_interfaces::execution_cost::host_history_cost_signature;
use ferrum_interfaces::output_flow::advance_committed_utf8_fragment;
mod native_acquisition;

struct Slot {
    id: Option<String>,
    maximum: u64,
    frontier: Option<Frontier>,
    prompt: Option<u64>,
    released: Option<Released>,
    first_measured: bool,
    native_restored: bool,
    native_restore_slot: Option<u64>,
    last_call: u64,
    last_fifo: u64,
}
struct Active {
    phase: usize,
    cohort: usize,
    slots: Vec<Slot>,
    native_capture: Option<NativeTransfer>,
}
struct Pending {
    offered: u64,
    phase: usize,
    cohort: usize,
    rows: Vec<Offered>,
}
pub(in super::super) struct Preparation {
    mode:
        crate::implementations::continuous::cost_profile::structured_v10::lifecycle::LifecycleMode,
    plan: StructuredPrefixPlanV5,
    native_acquisition: Option<StructuredNativePrefixAcquisitionPlanV1>,
    active: Option<Active>,
    pending: Option<Pending>,
    last_call: u64,
}
pub(in super::super) struct Progress<'a> {
    pub phase: usize,
    pub offered: &'a mut u64,
    pub maximum_offered: u64,
    pub last_fifo: &'a mut u64,
    pub last_finalized: &'a mut u64,
    pub earliest: u64,
    pub calls: &'a mut HashSet<u64>,
    pub total_rows: &'a mut usize,
    pub fingerprint: &'a ProfileFingerprint,
    pub source_opened_at_ns: u64,
    pub limits: &'a CostProfileLoadLimits,
    pub lifecycle: &'a mut super::super::super::lifecycle::Lifecycle,
}
#[track_caller]
fn fail() -> CostProfileError {
    let caller = std::panic::Location::caller();
    eprintln!(
        "source5 preparation trajectory check failed at {}:{}",
        caller.file(),
        caller.line(),
    );
    invalid("source5 preparation trajectory differs from original declaration")
}

#[track_caller]
fn fail_context(context: std::fmt::Arguments<'_>) -> CostProfileError {
    eprintln!("source5 preparation failure context: {context}");
    fail()
}

impl Preparation {
    pub(in super::super) fn not_submitted(
        &mut self,
        offered: u64,
        phase: usize,
        cohort: usize,
        participants: &[ferrum_interfaces::execution_cost::CallNoSubmissionParticipantV1],
    ) -> Result<(), CostProfileError> {
        let p = self.pending.as_ref().ok_or_else(|| fail())?;
        if p.offered != offered
            || p.phase != phase
            || p.cohort != cohort
            || p.rows.len() != participants.len()
        {
            return Err(fail());
        }
        for (index, row) in p.rows.iter().enumerate() {
            let actual = participants
                .iter()
                .find(|p| p.input_index as usize == index)
                .ok_or_else(|| fail())?;
            if actual.request_id.to_string() != row.before.request_id
                || actual.owner_incarnation != row.before.owner_incarnation
                || actual.work_generation != row.before.work_generation
            {
                return Err(fail());
            }
        }
        // The original proof establishes that no inference work or frontier
        // commit occurred. A retry is a new offered call, never a replacement.
        self.pending = None;
        Ok(())
    }
    pub(in super::super) fn outside_completed(
        &mut self,
        rows: &[crate::implementations::continuous::cost_profile::structured_v10::lifecycle::OriginalCohortRow<'_>],
    ) -> Result<(), CostProfileError> {
        self.ready()?;
        let active = self.active.as_mut().ok_or_else(|| fail())?;
        let prefixed = self.plan.phases[active.phase][active.cohort].is_some();
        for row in rows {
            let slot = active
                .slots
                .iter_mut()
                .find(|s| s.id.as_deref() == Some(row.request_id))
                .ok_or_else(|| fail())?;
            if prefixed {
                let released = slot.released.as_ref().ok_or_else(|| fail())?;
                if row.owner_incarnation != released.frontier.owner_incarnation
                    || row.policy != released.original_numeric_policy
                    || (!slot.first_measured
                        && (row.work_generation != released.frontier.work_generation
                            || row.frontier.generated_before != released.frontier.generated_tokens
                            || row.frontier.context_before != released.frontier.kv_tokens
                            || row.pending_decoded_utf8
                                != !released.frontier.pending_utf8.is_empty()))
                {
                    return Err(fail());
                }
            }
            slot.first_measured = true;
        }
        Ok(())
    }
    pub(in super::super) fn retained_heap_bytes(&self) -> Option<usize> {
        fn frontier(f: &Frontier) -> Option<usize> {
            f.request_id
                .capacity()
                .checked_add(f.pending_utf8.capacity())?
                .checked_add(f.model_cache_id.as_ref().map_or(0, String::capacity))
        }
        let mut n = self
            .native_acquisition
            .as_ref()
            .map_or(Some(0), |plan| plan.retained_heap_bytes())?;
        for phase in &self.plan.phases {
            n = n.checked_add(phase.capacity().checked_mul(std::mem::size_of::<Option<crate::implementations::continuous::cost_model::structured_v2::prefixes::StructuredPrefixCohortV5>>())?)?;
            for c in phase.iter().flatten() {
                n = n.checked_add(c.slots.capacity().checked_mul(std::mem::size_of::<crate::implementations::continuous::cost_model::structured_v2::prefixes::StructuredPrefixSlotV5>())?)?;
                for s in &c.slots {
                    n = n
                        .checked_add(
                            s.token_ids
                                .capacity()
                                .checked_mul(std::mem::size_of::<ferrum_types::TokenId>())?,
                        )?
                        .checked_add(
                            s.token_bytes
                                .capacity()
                                .checked_mul(std::mem::size_of::<Vec<u8>>())?,
                        )?;
                    for bytes in &s.token_bytes {
                        n = n.checked_add(bytes.capacity())?;
                    }
                }
            }
        }
        if let Some(a) = &self.active {
            if let Some(capture) = &a.native_capture {
                n = n
                    .checked_add(capture.plan_hash.capacity())?
                    .checked_add(capture.layout_fingerprint.capacity())?
                    .checked_add(capture.runtime_implementation_fingerprint.capacity())?
                    .checked_add(capture.device_id.capacity())?;
            }
            n = n.checked_add(
                a.slots
                    .capacity()
                    .checked_mul(std::mem::size_of::<Slot>())?,
            )?;
            for s in &a.slots {
                n = n.checked_add(s.id.as_ref().map_or(0, String::capacity))?;
                if let Some(f) = &s.frontier {
                    n = n.checked_add(frontier(f)?)?;
                }
                if let Some(r) = &s.released {
                    n = n.checked_add(frontier(&r.frontier)?)?;
                }
            }
        }
        if let Some(p) = &self.pending {
            n = n.checked_add(
                p.rows
                    .capacity()
                    .checked_mul(std::mem::size_of::<Offered>())?,
            )?;
            for r in &p.rows {
                n = n.checked_add(frontier(&r.before)?)?;
            }
        }
        Some(n)
    }
    pub fn new(plan: StructuredPrefixPlanV5) -> Self {
        Self::with_mode(plan, crate::implementations::continuous::cost_profile::structured_v10::lifecycle::LifecycleMode::LegacyLengthOnly)
    }
    pub(in super::super) fn with_mode(
        plan: StructuredPrefixPlanV5,
        mode: crate::implementations::continuous::cost_profile::structured_v10::lifecycle::LifecycleMode,
    ) -> Self {
        Self {
            mode,
            plan,
            native_acquisition: None,
            active: None,
            pending: None,
            last_call: 0,
        }
    }
    pub fn begin(
        &mut self,
        phase: usize,
        cohort: usize,
        requests: &CohortPlanV2,
    ) -> Result<(), CostProfileError> {
        if self.active.is_some() || self.pending.is_some() {
            return Err(fail());
        }
        let c = requests
            .phases
            .get(phase)
            .and_then(|v| v.get(cohort))
            .ok_or_else(|| fail())?;
        self.active = Some(Active {
            phase,
            cohort,
            native_capture: None,
            slots: c
                .requests
                .iter()
                .map(|r| Slot {
                    id: None,
                    maximum: r.maximum_output,
                    frontier: None,
                    prompt: None,
                    released: None,
                    first_measured: false,
                    native_restored: false,
                    native_restore_slot: None,
                    last_call: 0,
                    last_fifo: 0,
                })
                .collect(),
        });
        Ok(())
    }
    pub fn admit(&mut self, slot: usize, id: &str) -> Result<(), CostProfileError> {
        let slot = self
            .active
            .as_mut()
            .and_then(|a| a.slots.get_mut(slot))
            .ok_or_else(|| fail())?;
        if slot.id.is_some() {
            return Err(fail());
        }
        slot.id = Some(id.into());
        Ok(())
    }
    pub fn end(&mut self) -> Result<(), CostProfileError> {
        self.ready()?;
        if self.active.take().is_none() {
            return Err(fail());
        }
        Ok(())
    }
    pub fn idle(&self) -> bool {
        self.pending.is_none()
    }
    pub fn ready(&self) -> Result<(), CostProfileError> {
        if self.pending.is_some() {
            return Err(fail());
        }
        if let Some(a) = &self.active {
            if self.plan.phases[a.phase][a.cohort].is_some()
                && a.slots.iter().any(|s| s.released.is_none())
            {
                return Err(fail());
            }
        }
        Ok(())
    }
    pub fn observed_call(&mut self, call: u64) -> Result<(), CostProfileError> {
        if call <= self.last_call {
            return Err(fail());
        }
        self.last_call = call;
        Ok(())
    }
    pub fn handle(
        &mut self,
        value: serde_json::Value,
        p: &mut Progress<'_>,
    ) -> Result<bool, CostProfileError> {
        let record: PreparationRecord = serde_json::from_value(value)?;
        match record {
            PreparationRecord::NativePrefixRestored {
                phase,
                cohort,
                slot,
                before,
                after,
                input_tokens_sha256,
                capture,
                restore,
                captured_at_ns,
                acknowledged_at_ns,
                expires_at_ns,
                acknowledged,
            } => self.native_restored(
                phase,
                cohort,
                slot,
                before,
                after,
                input_tokens_sha256,
                capture,
                restore,
                captured_at_ns,
                acknowledged_at_ns,
                expires_at_ns,
                acknowledged,
                p,
            ),
            PreparationRecord::PreparationOffered {
                offered,
                phase,
                cohort,
                rows,
            } => {
                p.lifecycle.active(p.phase, cohort)?;
                let active = self.active.as_mut().ok_or_else(|| fail())?;
                let declared = self
                    .plan
                    .phases
                    .get(p.phase)
                    .and_then(|v| v.get(cohort))
                    .and_then(Option::as_ref)
                    .ok_or_else(|| fail())?;
                if phase.index() != p.phase
                    || active.phase != p.phase
                    || active.cohort != cohort
                    || self.pending.is_some()
                    || offered != p.offered.checked_add(1).ok_or_else(|| fail())?
                    || offered > p.maximum_offered
                    || rows.is_empty()
                    || rows.len() > 128
                    || active.slots.iter().any(|s| s.id.is_none())
                {
                    return Err(fail());
                }
                let mut ids = HashSet::new();
                for row in &rows {
                    let f = &row.before;
                    let slot = active
                        .slots
                        .iter_mut()
                        .find(|s| s.id.as_deref() == Some(f.request_id.as_str()))
                        .ok_or_else(|| fail())?;
                    if !ids.insert(&f.request_id)
                        || f.owner_incarnation == 0
                        || f.work_generation == 0
                        || f.request_id.len() > p.limits.max_source_field_bytes.get()
                        || f.pending_utf8.len() > 3
                        || f.generated_tokens >= declared.release_generated
                        || f.model_cache_id.as_ref().is_some_and(|id| {
                            id.is_empty() || id.len() > p.limits.max_source_field_bytes.get()
                        })
                        || slot.released.is_some()
                        || (self
                            .native_acquisition
                            .as_ref()
                            .and_then(|plan| plan.phases[p.phase].get(cohort))
                            .and_then(Option::as_ref)
                            .is_some()
                            && !slot.native_restored)
                    {
                        return Err(fail());
                    }
                    match &slot.frontier {
                        Some(previous) if previous == f => {}
                        None if f.generated_tokens == 0
                            && f.kv_tokens == 0
                            && f.model_cache_id.is_none()
                            && f.pending_utf8.is_empty()
                            && f.output_accepted_ordinal == 0 => {}
                        _ => return Err(fail()),
                    }
                    row.work.emits_token().map_err(numeric_error)?;
                    match row.work {
                        PreparedWorkV2::Prefill {
                            offset,
                            count: _,
                            total_prompt_tokens,
                        } => {
                            if f.generated_tokens != 0
                                || u64::from(offset) != f.kv_tokens
                                || slot
                                    .prompt
                                    .is_some_and(|n| n != u64::from(total_prompt_tokens))
                            {
                                return Err(fail());
                            }
                            slot.prompt = Some(u64::from(total_prompt_tokens));
                        }
                        PreparedWorkV2::Decode { kv_tokens } => {
                            if f.generated_tokens == 0
                                || u64::from(kv_tokens) != f.kv_tokens
                                || slot
                                    .prompt
                                    .and_then(|n| n.checked_add(f.generated_tokens - 1))
                                    != Some(f.kv_tokens)
                            {
                                return Err(fail());
                            }
                        }
                    }
                }
                *p.offered = offered;
                self.pending = Some(Pending {
                    offered,
                    phase: p.phase,
                    cohort,
                    rows,
                });
                Ok(false)
            }
            PreparationRecord::PreparationCompleted {
                offered,
                phase,
                cohort,
                reconciled,
                queue,
                host_stages,
                rows,
                failure,
            } => {
                let pending = self.pending.take().ok_or_else(|| fail())?;
                // The producer already retained the original reconciliation
                // or prefix-chain failure. Preserve it when rejecting this
                // record instead of replacing it with a generic trajectory error.
                if let Some(failure) = failure {
                    return Err(fail_context(format_args!(
                        "source5 preparation trajectory differs from original declaration: producer failure at phase={} cohort={cohort} offered={offered} reconciled={reconciled}: {failure}",
                        phase.index(),
                    )));
                }
                if !reconciled
                    || phase.index() != p.phase
                    || pending.phase != p.phase
                    || pending.cohort != cohort
                    || pending.offered != offered
                    || rows.len() != pending.rows.len()
                {
                    return Err(fail());
                }
                let queue = queue.ok_or_else(|| fail())?;
                let fifo = queue.accepted_ordinal.ok_or_else(|| fail())?;
                if queue.disposition != "published" || Some(fifo) != p.last_fifo.checked_add(1) {
                    return Err(fail_context(format_args!(
                        "source5 preparation trajectory differs from original declaration: completed FIFO at phase={} cohort={cohort} offered={offered}: disposition={} actual={fifo} previous={}",
                        phase.index(), queue.disposition, *p.last_fifo,
                    )));
                }
                let stages = host_stages.ok_or_else(|| fail())?;
                let observed = stages::validate(
                    p.fingerprint,
                    p.source_opened_at_ns,
                    &pending.rows,
                    &stages,
                    if self.native_acquisition.is_some() {
                        p.earliest.max(*p.last_finalized)
                    } else {
                        p.earliest
                    },
                )?;
                if observed < *p.last_finalized {
                    return Err(fail_context(format_args!(
                        "source5 preparation trajectory differs from original declaration: finalized clock at phase={} cohort={cohort} offered={offered} call={}: actual={observed} previous={}",
                        phase.index(), stages.call_id, *p.last_finalized,
                    )));
                }
                if !p.calls.insert(stages.call_id) {
                    return Err(fail_context(format_args!(
                        "source5 preparation trajectory differs from original declaration: duplicate completed call at phase={} cohort={cohort} offered={offered} call={}",
                        phase.index(), stages.call_id,
                    )));
                }
                self.observed_call(stages.call_id)?;
                *p.total_rows = p.total_rows.checked_add(rows.len()).ok_or_else(|| fail())?;
                if *p.total_rows > p.limits.max_total_shape_rows.get() {
                    return Err(CostProfileError::Limit("source5 preparation shape rows"));
                }
                let active = self.active.as_mut().ok_or_else(|| fail())?;
                let declaration = self.plan.phases[p.phase][cohort]
                    .as_ref()
                    .ok_or_else(|| fail())?;
                let mut ids = HashSet::new();
                for row in rows {
                    let before = &row.before;
                    let offered = pending
                        .rows
                        .iter()
                        .find(|o| o.before.request_id == before.request_id)
                        .ok_or_else(|| fail())?;
                    let slot_index = active
                        .slots
                        .iter()
                        .position(|s| s.id.as_deref() == Some(before.request_id.as_str()))
                        .ok_or_else(|| fail())?;
                    let slot = &mut active.slots[slot_index];
                    let after = row.after.ok_or_else(|| fail())?;
                    let emits = offered.work.emits_token().map_err(numeric_error)?;
                    let next_kv = match offered.work {
                        PreparedWorkV2::Prefill { offset, count, .. } => {
                            u64::from(offset).checked_add(u64::from(count))
                        }
                        PreparedWorkV2::Decode { kv_tokens } => u64::from(kv_tokens).checked_add(1),
                    }
                    .ok_or_else(|| fail())?;
                    if !ids.insert(before.request_id.clone())
                        || before != &offered.before
                        || after.request_id != before.request_id
                        || after.owner_incarnation != before.owner_incarnation
                        || Some(after.work_generation) != before.work_generation.checked_add(1)
                        || Some(after.generated_tokens)
                            != before.generated_tokens.checked_add(u64::from(emits))
                        || after.generated_tokens > declaration.release_generated
                        || after.generated_tokens >= slot.maximum
                        || after.kv_tokens != next_kv
                        || after.pending_utf8.len() > 3
                        || after.model_cache_id.as_ref().is_some_and(|id| {
                            id.is_empty() || id.len() > p.limits.max_source_field_bytes.get()
                        })
                        || before
                            .model_cache_id
                            .as_ref()
                            .is_some_and(|id| after.model_cache_id.as_ref() != Some(id))
                        || Some(after.output_accepted_ordinal)
                            != before.output_accepted_ordinal.checked_add(u64::from(emits))
                    {
                        return Err(fail());
                    }
                    if emits {
                        let commit = row.preparation_commit.ok_or_else(|| fail())?;
                        let token = before.generated_tokens as usize;
                        let declared = &declaration.slots[slot_index];
                        let bytes = declared.token_bytes.get(token).ok_or_else(|| fail())?;
                        let expected_pending =
                            advance_committed_utf8_fragment(&before.pending_utf8, bytes)
                                .map_err(|_| fail())?;
                        if commit.request_id != before.request_id
                            || commit.owner_incarnation != before.owner_incarnation
                            || commit.work_generation != before.work_generation
                            || commit.generated_before != before.generated_tokens
                            || commit.generated_after != after.generated_tokens
                            || declared.token_ids.get(token) != Some(&commit.committed_token)
                            || commit.pending_before != before.pending_utf8
                            || commit.pending_after != after.pending_utf8
                            || after.pending_utf8 != expected_pending
                        {
                            return Err(fail());
                        }
                    } else if row.preparation_commit.is_some()
                        || after.pending_utf8 != before.pending_utf8
                    {
                        return Err(fail());
                    }
                    p.lifecycle.preparation_row(
                        p.phase,
                        cohort,
                        &before.request_id,
                        before.owner_incarnation,
                        before.generated_tokens,
                        after.generated_tokens,
                    )?;
                    slot.frontier = Some(after);
                    slot.last_call = stages.call_id;
                    slot.last_fifo = fifo;
                }
                *p.last_fifo = fifo;
                *p.last_finalized = observed;
                Ok(true)
            }
            PreparationRecord::PreparationReleased {
                phase,
                cohort,
                slot,
                receipt,
            } => {
                p.lifecycle.active(p.phase, cohort)?;
                let active = self.active.as_mut().ok_or_else(|| fail())?;
                let declaration = self
                    .plan
                    .phases
                    .get(p.phase)
                    .and_then(|v| v.get(cohort))
                    .and_then(Option::as_ref)
                    .ok_or_else(|| fail())?;
                if phase.index() != p.phase
                    || active.phase != p.phase
                    || active.cohort != cohort
                    || self.pending.is_some()
                    || active.slots.iter().any(|slot| {
                        slot.frontier.as_ref().is_none_or(|frontier| {
                            frontier.generated_tokens != declaration.release_generated
                        })
                    })
                {
                    return Err(fail());
                }
                let actual = active.slots.get_mut(slot).ok_or_else(|| fail())?;
                let declared = declaration.slots.get(slot).ok_or_else(|| fail())?;
                let mut hash = Sha256::new();
                hash.update(b"ferrum.calibration.generated-prefix.v1\0");
                for token in &declared.token_ids {
                    hash.update(token.get().to_le_bytes());
                }
                let expected: [u8; 32] = hash.finalize().into();
                if actual.released.is_some()
                    || actual.last_call == 0
                    || actual.frontier.as_ref() != Some(&receipt.frontier)
                    || receipt.frontier.generated_tokens != declaration.release_generated
                    || receipt.frontier.pending_utf8
                        != declared.expected_pending().map_err(numeric_error)?
                    || receipt.generated_prefix_sha256 != expected
                    || receipt.through_call_id != actual.last_call
                    || receipt.through_fifo_ordinal != actual.last_fifo
                    || receipt.actor_applied_output_ordinal
                        != receipt.frontier.output_accepted_ordinal
                    || !self.mode.accepts_policy(receipt.original_numeric_policy)
                {
                    return Err(fail());
                }
                actual.released = Some(receipt);
                Ok(false)
            }
        }
    }

    pub fn prepared(&self, p: &Prepared) -> Result<(), CostProfileError> {
        self.ready()?;
        let active = self.active.as_ref().ok_or_else(|| fail())?;
        if self.plan.phases[active.phase][active.cohort].is_none() {
            return Ok(());
        }
        let mut hash = Sha256::new();
        hash_bytes(&mut hash, b"ferrum.canonical-wave.output.v1");
        hash.update(
            match p.recipe.device.product.as_str() {
                "full_logits" => 0u64,
                "greedy_token" => 1,
                _ => return Err(fail()),
            }
            .to_le_bytes(),
        );
        let numeric = p.exact.numeric_features.as_ref().ok_or_else(|| fail())?;
        if p.rows.len() != p.recipe.physical_host_rows.len() || p.rows.len() != numeric.rows.len() {
            return Err(fail());
        }
        for ((row, host), numeric) in p
            .rows
            .iter()
            .zip(&p.recipe.physical_host_rows)
            .zip(&numeric.rows)
        {
            let slot = active
                .slots
                .iter()
                .find(|s| s.id.as_deref() == Some(row.request_id.as_str()))
                .ok_or_else(|| fail())?;
            let released = slot.released.as_ref().ok_or_else(|| fail())?;
            if row.owner_incarnation != released.frontier.owner_incarnation
                || host.installed_policy != released.original_numeric_policy
                || (!slot.first_measured
                    && (row.work_generation != released.frontier.work_generation
                        || row.frontier.generated_before != released.frontier.generated_tokens
                        || row.frontier.context_before != released.frontier.kv_tokens
                        || host.pending_decoded_utf8 != !released.frontier.pending_utf8.is_empty()))
            {
                return Err(fail());
            }
            hash_bytes(
                &mut hash,
                &host_history_cost_signature(
                    released.original_policy_signature,
                    row.frontier.generated_before,
                ),
            );
            hash.update(u64::from(host.mask_upload_required).to_le_bytes());
            match row.frontier.work {
                PreparedWorkV2::Prefill { .. } => return Err(fail()),
                PreparedWorkV2::Decode { .. } => {
                    hash.update(2u64.to_le_bytes());
                    hash.update(
                        u64::from(host.decode_requires_full_logits.ok_or_else(|| fail())?)
                            .to_le_bytes(),
                    );
                    hash.update(numeric.repetition_tokens.to_le_bytes());
                    hash.update(
                        u64::from(host.repetition_penalty_bits.ok_or_else(|| fail())?)
                            .to_le_bytes(),
                    );
                }
            }
        }
        if <[u8; 32]>::from(hash.finalize()) != p.exact.exact.output_policy_signature {
            return Err(invalid(
                "source5 restored ordinary host policy differs from actual Prepared",
            ));
        }
        Ok(())
    }
    pub fn ordinary_completed(&mut self, p: &Prepared) -> Result<(), CostProfileError> {
        let active = self.active.as_mut().ok_or_else(|| fail())?;
        for row in &p.rows {
            let slot = active
                .slots
                .iter_mut()
                .find(|s| s.id.as_deref() == Some(row.request_id.as_str()))
                .ok_or_else(|| fail())?;
            slot.first_measured = true;
        }
        Ok(())
    }
}
fn hash_bytes(hash: &mut Sha256, value: &[u8]) {
    hash.update((value.len() as u64).to_le_bytes());
    hash.update(value);
}
