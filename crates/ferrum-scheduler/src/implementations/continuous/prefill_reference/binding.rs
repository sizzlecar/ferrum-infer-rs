use super::*;

#[derive(Debug, Clone, Copy)]
pub struct ReferenceChunkLimits {
    pub maximum_tokens: NonZeroU32,
    pub alignment: NonZeroU32,
    pub allow_final_short_chunk: bool,
    pub maximum_candidates: NonZeroUsize,
}
impl LoadedPrefillReference {
    /// A bounded subset of measured endpoints, intersected with backend rules.
    /// Merging reference segments changes physical cost, never the work unit.
    pub fn legal_chunks(
        &self,
        total: NonZeroU32,
        offset: u32,
        limits: ReferenceChunkLimits,
    ) -> Result<Vec<NonZeroU32>, ReferenceUnknown> {
        if limits.maximum_candidates.get() > 64 {
            return Err(ReferenceUnknown::PointBudget);
        }
        if let Some(definition) = &self.piecewise {
            if total < definition.spec.minimum_prompt_tokens
                || total > definition.spec.maximum_prompt_tokens
            {
                return Err(ReferenceUnknown::LengthNotCalibrated);
            }
            let mut counts = std::collections::BTreeSet::new();
            let remaining = total
                .get()
                .checked_sub(offset)
                .ok_or(ReferenceUnknown::InvalidProgress)?;
            if offset % limits.alignment.get() != 0 {
                return Err(ReferenceUnknown::NoLegalChunk);
            }
            for value in [
                limits.alignment.get(),
                self.protocol.granule_tokens.get(),
                limits.maximum_tokens.get(),
            ] {
                let count = value.min(remaining).min(limits.maximum_tokens.get());
                let count = count - count % limits.alignment.get();
                if let Some(count) = NonZeroU32::new(count) {
                    counts.insert(count);
                }
            }
            if limits.allow_final_short_chunk && remaining <= limits.maximum_tokens.get() {
                if let Some(tail) = NonZeroU32::new(remaining) {
                    counts.insert(tail);
                }
            }
            if counts.is_empty() {
                return Err(ReferenceUnknown::NoLegalChunk);
            }
            return Ok(counts
                .into_iter()
                .take(limits.maximum_candidates.get())
                .collect());
        }
        let curve = self.curve(total)?;
        curve
            .work_at(offset)
            .ok_or(ReferenceUnknown::MissingEndpoint)?;
        let mut chunks = Vec::with_capacity(limits.maximum_candidates.get());
        for point in &curve.points {
            let Some(count) = point
                .prompt_tokens
                .checked_sub(offset)
                .and_then(NonZeroU32::new)
            else {
                continue;
            };
            if count > limits.maximum_tokens {
                break;
            }
            if count.get() % limits.alignment.get() == 0
                || (point.prompt_tokens == total.get() && limits.allow_final_short_chunk)
            {
                chunks.push(count);
                if chunks.len() == limits.maximum_candidates.get() {
                    break;
                }
            }
        }
        if chunks.is_empty() {
            Err(ReferenceUnknown::NoLegalChunk)
        } else {
            Ok(chunks)
        }
    }

    /// The caller supplies a real original promise anchor and verified prefix
    /// receipt. All times use one stable monotonic domain for this binding's
    /// lifetime. An engine using a snapshot-local origin must translate both
    /// original anchors and checkpoints together; it must not bind again at
    /// snapshot.now. This creates scoring state, never execution authority.
    pub fn bind(
        self: &Arc<Self>,
        request_id: RequestId,
        incarnation: NonZeroU64,
        total: NonZeroU32,
        admitted_at_ns: u64,
        first_deadline_ns: u64,
        committed_prefix_at_admission: u32,
    ) -> Result<RequestPrefillReferenceBinding, ReferenceUnknown> {
        if admitted_at_ns >= first_deadline_ns || committed_prefix_at_admission >= total.get() {
            return Err(ReferenceUnknown::InvalidProgress);
        }
        let curve = self.curve(total)?;
        let baseline = curve
            .work_at(committed_prefix_at_admission)
            .ok_or(ReferenceUnknown::MissingEndpoint)?;
        Ok(RequestPrefillReferenceBinding {
            calibration: Arc::clone(self),
            curve,
            request_id,
            incarnation,
            total,
            admitted_at_ns,
            first_deadline_ns,
            reference_work_at_admission_ns: baseline,
            logical_high_water: committed_prefix_at_admission,
            first_token_committed: false,
        })
    }
}

/// Fixed per-incarnation accounting. No setter can refresh the original
/// admission, replace the curve, or reset useful high-water during recompute.
#[derive(Debug, Clone)]
pub struct RequestPrefillReferenceBinding {
    calibration: Arc<LoadedPrefillReference>,
    curve: Arc<PrefillReferenceWork>,
    request_id: RequestId,
    incarnation: NonZeroU64,
    total: NonZeroU32,
    admitted_at_ns: u64,
    first_deadline_ns: u64,
    reference_work_at_admission_ns: u64,
    logical_high_water: u32,
    first_token_committed: bool,
}
impl RequestPrefillReferenceBinding {
    pub fn identity(&self) -> ReferenceIdentity {
        self.calibration.identity()
    }
    pub fn tau_ref_ns(&self) -> NonZeroU64 {
        self.calibration.tau_ref_ns()
    }
    pub fn reference(&self) -> &Arc<PrefillReferenceWork> {
        &self.curve
    }
    pub fn admitted_at_ns(&self) -> u64 {
        self.admitted_at_ns
    }
    pub fn first_deadline_ns(&self) -> u64 {
        self.first_deadline_ns
    }
    pub fn logical_high_water(&self) -> u32 {
        self.logical_high_water
    }
    pub fn reference_work_at_admission_ns(&self) -> u64 {
        self.reference_work_at_admission_ns
    }
    pub fn first_token_committed(&self) -> bool {
        self.first_token_committed
    }

    /// Call only after one exact successful physical/host receipt. Failed,
    /// cancelled or NotSubmitted work has no corresponding operation here.
    /// Recomputing or restoring an already credited prefix earns zero work.
    pub fn record_committed(
        &mut self,
        request_id: &RequestId,
        incarnation: NonZeroU64,
        offset: u32,
        end: u32,
        first_token_committed: bool,
    ) -> Result<u64, ReferenceUnknown> {
        if request_id != &self.request_id || incarnation != self.incarnation {
            return Err(ReferenceUnknown::WrongIncarnation);
        }
        if offset >= end || end > self.total.get() || offset > self.logical_high_water {
            return Err(ReferenceUnknown::InvalidProgress);
        }
        self.curve
            .work_at(offset)
            .ok_or(ReferenceUnknown::MissingEndpoint)?;
        self.curve
            .work_at(end)
            .ok_or(ReferenceUnknown::MissingEndpoint)?;
        if (end == self.total.get()) != first_token_committed {
            return Err(ReferenceUnknown::FinalTokenNotCommitted);
        }
        let high_water = self.logical_high_water.max(end);
        let delta = self
            .curve
            .work_at(high_water)
            .ok_or(ReferenceUnknown::MissingEndpoint)?
            .checked_sub(
                self.curve
                    .work_at(self.logical_high_water)
                    .ok_or(ReferenceUnknown::MissingEndpoint)?,
            )
            .ok_or(ReferenceUnknown::ArithmeticOverflow)?;
        self.logical_high_water = high_water;
        self.first_token_committed |= first_token_committed;
        Ok(delta)
    }

    /// Checked projection keeps original time/baseline even after a rollback of
    /// physical offset. Milestones apply to remaining work and add the baseline
    /// once; the final first-token deadline has no quantization allowance.
    pub fn progress(
        &self,
        offset: u32,
        executable_until: u32,
        checkpoints_at_ns: &[u64],
    ) -> Result<PrefillProgressView, ReferenceUnknown> {
        self.progress_with_granule(offset, executable_until, checkpoints_at_ns, None)
    }

    /// V2 needs an actual legal execution granule to quantize milestones.
    /// Interpolation alone is never evidence that any fragment is executable.
    pub fn progress_with_granule(
        &self,
        offset: u32,
        executable_until: u32,
        checkpoints_at_ns: &[u64],
        legal_granule: Option<NonZeroU32>,
    ) -> Result<PrefillProgressView, ReferenceUnknown> {
        if offset > self.logical_high_water
            || offset >= self.total.get()
            || executable_until <= offset
            || executable_until > self.total.get()
        {
            return Err(ReferenceUnknown::InvalidProgress);
        }
        if checkpoints_at_ns.len() > 1024 {
            return Err(ReferenceUnknown::PointBudget);
        }
        self.curve
            .work_at(offset)
            .ok_or(ReferenceUnknown::MissingEndpoint)?;
        let total = self
            .curve
            .work_at(self.total.get())
            .ok_or(ReferenceUnknown::MissingEndpoint)?;
        let remaining = total
            .checked_sub(self.reference_work_at_admission_ns)
            .and_then(NonZeroU64::new)
            .ok_or(ReferenceUnknown::ArithmeticOverflow)?;
        let quantum = if self.calibration.piecewise.is_some() {
            0
        } else {
            self.curve
                .points
                .windows(2)
                .filter(|pair| pair[1].cumulative_work_ns > self.reference_work_at_admission_ns)
                .map(|pair| pair[1].cumulative_work_ns - pair[0].cumulative_work_ns)
                .max()
                .unwrap_or(0)
        };
        let mut milestones = linear_prefill_milestones(
            self.admitted_at_ns,
            self.first_deadline_ns,
            remaining,
            checkpoints_at_ns,
            quantum,
        )
        .ok_or(ReferenceUnknown::InvalidProgress)?;
        for milestone in &mut milestones {
            milestone.required_reference_work_ns = milestone
                .required_reference_work_ns
                .checked_add(self.reference_work_at_admission_ns)
                .ok_or(ReferenceUnknown::ArithmeticOverflow)?;
        }
        if self.calibration.piecewise.is_some() && !checkpoints_at_ns.is_empty() {
            let granule = legal_granule.ok_or(ReferenceUnknown::NoLegalChunk)?.get();
            // L already uses the original anchor and no V1 measured-segment
            // allowance. Round down by at most one actual legal granule.
            for milestone in &mut milestones {
                let ideal = milestone.required_reference_work_ns;
                if milestone.at_ns == self.first_deadline_ns {
                    milestone.required_reference_work_ns = total;
                    continue;
                }
                let prefix = self
                    .curve
                    .nonfinal_prefix_at_or_below(ideal)
                    .ok_or(ReferenceUnknown::MissingEndpoint)?;
                let prefix = prefix - prefix % granule;
                milestone.required_reference_work_ns = self
                    .curve
                    .work_at(prefix)
                    .ok_or(ReferenceUnknown::MissingEndpoint)?
                    .max(self.reference_work_at_admission_ns);
            }
        }
        Ok(PrefillProgressView {
            admitted_at_ns: self.admitted_at_ns,
            reference_work_at_admission_ns: self.reference_work_at_admission_ns,
            offset,
            total_prompt_tokens: self.total,
            logical_high_water: self.logical_high_water,
            reference: Arc::clone(&self.curve),
            milestones: milestones.into(),
            executable_until,
        })
    }
}
