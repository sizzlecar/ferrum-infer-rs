use super::*;

/// One immutable logical frontier. It contains no resolved route, resource
/// authority, or cost. A cursor belongs to exactly one search node; successors
/// rebuild readiness and urgency from their jointly advanced state.
pub(crate) struct FrontierCursor {
    decoders: Vec<usize>,
    prefills: Vec<usize>,
    fair_decoders: Vec<usize>,
    fair_prefills: Vec<usize>,
    decode_sizes: Vec<usize>,
    prefill_sizes: Vec<usize>,
    chunks: Vec<NonZeroU32>,
    required: Option<RequestWorkKey>,
    stage: Stage,
    seen: Vec<Vec<CandidateWork>>,
    attempt_limit: usize,
    attempts: usize,
    pub truncated: bool,
}

#[derive(Clone, Copy)]
enum Stage {
    WholeDecode,
    RoundDecode(usize),
    Prefill {
        round: usize,
        size: usize,
    },
    Mixed {
        round: usize,
        size: usize,
        decode: usize,
    },
    FairDecode(usize),
    FairPrefill {
        chunk: usize,
        size: usize,
    },
    Done,
}

#[derive(Clone, Copy)]
enum Action {
    Decode {
        size: usize,
        fair: bool,
    },
    Prefill {
        size: usize,
        chunk: NonZeroU32,
        fair: bool,
    },
    Mixed {
        prefill: usize,
        chunk: NonZeroU32,
        decode: usize,
    },
}

impl FrontierCursor {
    pub fn new(
        snapshot: &SchedulerSnapshot,
        requests: &[RequestSchedulingView],
        now_ns: u64,
        limit: usize,
        protection: Option<&PlanningObligationSet>,
        poll: &mut dyn FnMut() -> Result<(), PlanningUnknownReason>,
    ) -> Result<Self, PlanningUnknownReason> {
        let mut decoders = Vec::new();
        let mut prefills = Vec::new();
        for (index, row) in requests.iter().enumerate() {
            poll()?;
            if runnable(row) {
                match row.phase {
                    RequestPhaseView::Decode => decoders.push(index),
                    RequestPhaseView::Prefill(_) => prefills.push(index),
                }
            }
        }
        let priority = |index: usize| {
            (
                protection.is_some_and(|scope| !scope.protects(index)),
                urgency(&requests[index], now_ns),
            )
        };
        decoders.sort_by_key(|&index| priority(index));
        prefills.sort_by_key(|&index| priority(index));
        poll()?;
        let required = protection.and_then(|scope| scope.required_service(requests));
        if let Some(required) = required {
            for indices in [&mut decoders, &mut prefills] {
                if let Some(position) = indices.iter().position(|&index| index == required) {
                    indices.rotate_left(position);
                }
            }
        }
        let mut fair_decoders = decoders.clone();
        let mut fair_prefills = prefills.clone();
        for indices in [&mut fair_decoders, &mut fair_prefills] {
            if let Some((position, _)) = indices
                .iter()
                .enumerate()
                .min_by_key(|(_, index)| requests[**index].fairness_rank)
            {
                indices.rotate_left(position);
            }
        }
        let caps = &snapshot.capabilities;
        let mut decode_sizes = Vec::new();
        for size in &caps.decode_batch_sizes {
            poll()?;
            if size.get() <= decoders.len() && size.get() <= caps.max_wave_rows.get() {
                decode_sizes.push(size.get());
            }
        }
        let mut prefill_sizes = Vec::new();
        for size in &caps.prefill_batch_sizes {
            poll()?;
            if size.get() <= prefills.len() && size.get() <= caps.max_wave_rows.get() {
                prefill_sizes.push(size.get());
            }
        }
        let mut chunks = Vec::new();
        if !prefill_sizes.is_empty() {
            for &chunk in &caps.prefill_chunk_sizes {
                poll()?;
                // This only rules out mathematically illegal endpoints. It
                // never infers physical capacity, route coverage or cost.
                if prefill_work(requests, &prefills, 1, chunk, caps, poll)?.is_some() {
                    chunks.push(chunk);
                }
            }
        }
        Ok(Self {
            decoders,
            prefills,
            fair_decoders,
            fair_prefills,
            decode_sizes,
            prefill_sizes,
            chunks,
            required: required.map(|index| requests[index].key.clone()),
            stage: Stage::WholeDecode,
            seen: Vec::new(),
            attempt_limit: limit.min(256).saturating_mul(8),
            attempts: 0,
            truncated: false,
        })
    }

    /// Generate only until the next distinct legal action. Unseen siblings do
    /// not delay its physical projection or the construction of its shared tail.
    /// Raw attempts and successful physical expansions have separate limits.
    pub fn next(
        &mut self,
        snapshot: &SchedulerSnapshot,
        requests: &[RequestSchedulingView],
        global_attempts: &mut usize,
        global_limit: usize,
        poll: &mut dyn FnMut() -> Result<(), PlanningUnknownReason>,
    ) -> Result<Option<Vec<CandidateWork>>, PlanningUnknownReason> {
        loop {
            poll()?;
            let Some(action) = self.action(
                snapshot.capabilities.native_mixed && snapshot.capabilities.work_policy.allow_mixed,
                poll,
            )?
            else {
                return Ok(None);
            };
            if self.attempts >= self.attempt_limit || *global_attempts >= global_limit {
                self.truncated = true;
                self.stage = Stage::Done;
                return Ok(None);
            }
            self.attempts += 1;
            *global_attempts += 1;
            let caps = &snapshot.capabilities;
            let work = match action {
                Action::Decode { size, fair } => decode_work(
                    requests,
                    if fair {
                        &self.fair_decoders
                    } else {
                        &self.decoders
                    },
                    size,
                ),
                Action::Prefill { size, chunk, fair } => prefill_work(
                    requests,
                    if fair {
                        &self.fair_prefills
                    } else {
                        &self.prefills
                    },
                    size,
                    chunk,
                    caps,
                    poll,
                )?,
                Action::Mixed {
                    prefill,
                    chunk,
                    decode,
                } => {
                    if prefill + decode > caps.max_wave_rows.get() {
                        continue;
                    }
                    match (
                        decode_work(requests, &self.decoders, decode),
                        prefill_work(requests, &self.prefills, prefill, chunk, caps, poll)?,
                    ) {
                        (Some(mut rows), Some(prefill)) => {
                            rows.extend(prefill);
                            Some(rows)
                        }
                        _ => None,
                    }
                }
            };
            let Some(work) = work else { continue };
            if !within_work_envelope(caps, requests, &work, poll)? {
                continue;
            }
            let prefill_tokens: u64 = work
                .iter()
                .map(|row| match row.action {
                    WaveAction::Prefill { count, .. } => u64::from(count.get()),
                    WaveAction::Decode => 0,
                })
                .sum();
            if prefill_tokens > caps.max_prefill_tokens_per_wave.get() {
                continue;
            }
            if self
                .required
                .as_ref()
                .is_some_and(|key| !work.iter().any(|r| &r.key == key))
            {
                continue;
            }
            let mut duplicate = false;
            for old in &self.seen {
                poll()?;
                if old.len() == work.len() && work.iter().all(|row| old.contains(row)) {
                    duplicate = true;
                    break;
                }
            }
            if duplicate {
                continue;
            }
            self.seen.push(work.clone());
            return Ok(Some(work));
        }
    }

    pub fn may_have_more(&self) -> bool {
        !matches!(self.stage, Stage::Done)
    }

    fn action(
        &mut self,
        mixed: bool,
        poll: &mut dyn FnMut() -> Result<(), PlanningUnknownReason>,
    ) -> Result<Option<Action>, PlanningUnknownReason> {
        loop {
            poll()?;
            match self.stage {
                Stage::WholeDecode => {
                    self.stage = Stage::RoundDecode(0);
                    if let Some(&size) = self.decode_sizes.iter().max() {
                        return Ok(Some(Action::Decode { size, fair: false }));
                    }
                }
                Stage::RoundDecode(round) => {
                    if round >= self.decode_sizes.len().max(self.chunks.len()) {
                        self.stage = Stage::FairDecode(0);
                    } else {
                        self.stage = Stage::Prefill { round, size: 0 };
                        if let Some(&size) = self.decode_sizes.get(round) {
                            return Ok(Some(Action::Decode { size, fair: false }));
                        }
                    }
                }
                Stage::Prefill { round, size } => {
                    match (self.chunks.get(round), self.prefill_sizes.get(size)) {
                        (Some(&chunk), Some(&count)) => {
                            self.stage = Stage::Mixed {
                                round,
                                size,
                                decode: 0,
                            };
                            return Ok(Some(Action::Prefill {
                                size: count,
                                chunk,
                                fair: false,
                            }));
                        }
                        _ => self.stage = Stage::RoundDecode(round + 1),
                    }
                }
                Stage::Mixed {
                    round,
                    size,
                    decode,
                } => {
                    if mixed && decode < self.decode_sizes.len() {
                        self.stage = Stage::Mixed {
                            round,
                            size,
                            decode: decode + 1,
                        };
                        return Ok(Some(Action::Mixed {
                            prefill: self.prefill_sizes[size],
                            chunk: self.chunks[round],
                            decode: self.decode_sizes[decode],
                        }));
                    }
                    self.stage = Stage::Prefill {
                        round,
                        size: size + 1,
                    };
                }
                Stage::FairDecode(index) => {
                    self.stage = Stage::FairDecode(index + 1);
                    if let Some(&size) = self.decode_sizes.get(index) {
                        return Ok(Some(Action::Decode { size, fair: true }));
                    }
                    self.stage = Stage::FairPrefill { chunk: 0, size: 0 };
                }
                Stage::FairPrefill { chunk, size } => {
                    let Some(&tokens) = self.chunks.get(chunk) else {
                        self.stage = Stage::Done;
                        continue;
                    };
                    if let Some(&size_value) = self.prefill_sizes.get(size) {
                        self.stage = Stage::FairPrefill {
                            chunk,
                            size: size + 1,
                        };
                        return Ok(Some(Action::Prefill {
                            size: size_value,
                            chunk: tokens,
                            fair: true,
                        }));
                    }
                    self.stage = Stage::FairPrefill {
                        chunk: chunk + 1,
                        size: 0,
                    };
                }
                Stage::Done => return Ok(None),
            }
        }
    }
}
