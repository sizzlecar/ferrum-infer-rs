//! Observe actual completed Core waves; no output or timing is inferred from
//! poll count. Draft KV and main KV have different frontiers, so only main
//! settlement emits the existing monotonic token-wave protocol.
use super::super::diagnostics::{PhaseObservation, SharedOwner};
use super::*;
use serde::Serialize;
use std::time::{Duration, Instant};

#[derive(Default, Serialize)]
struct RoleFacts {
    computed_waves: u64,
    multi_participant_waves: u64,
    maximum_participants: usize,
    wave_wall_us: u128,
    poll_wall_us: u128,
    settlement_wall_us: u128,
}

#[derive(Default, Serialize)]
struct RequestFacts {
    main_committed_waves: u64,
    draft_committed_waves: u64,
    verified_proposals: u64,
    accepted_proposals: u64,
    rejected_proposals: u64,
    verified_output_tokens: u64,
}

pub(super) struct WaveObservation {
    wave: u64,
    kind: WaveKind,
    before: Vec<usize>,
    started: Instant,
    poll_wall: Duration,
}
impl WaveObservation {
    pub(super) fn polled(&mut self, elapsed: Duration) {
        self.poll_wall = self.poll_wall.saturating_add(elapsed);
    }
}

pub(super) struct Observations {
    observations: Vec<Option<PhaseObservation>>,
    ended: Vec<bool>,
    requests: Vec<RequestFacts>,
    roles: [RoleFacts; 4],
    next_wave: u64,
    retiring: bool,
}
impl Observations {
    pub(super) fn new(
        contexts: Vec<Option<(DiagnosticContext, Instant)>>,
        pending: &[Option<PreparedTextExpansion>],
    ) -> Self {
        let owner = SharedOwner {
            id: uuid::Uuid::new_v4(),
            participants: pending.len(),
        };
        let observations = contexts
            .into_iter()
            .zip(pending)
            .map(|(context, prepared)| {
                let prepared = prepared.as_ref().expect("validated unused preparation");
                context
                    .filter(|(c, _)| c.policy != PH3ImagePromptExpansionDiagnosticPolicy::Disabled)
                    .map(|(context, started)| {
                        PhaseObservation::new(
                            context,
                            prepared.tokens.len(),
                            prepared.generation.eos_token_id,
                            started,
                        )
                        .with_shared_owner(owner)
                        .with_vision_binding(prepared.vision_binding)
                    })
            })
            .collect();
        Self {
            observations,
            ended: vec![false; pending.len()],
            requests: (0..pending.len())
                .map(|_| RequestFacts::default())
                .collect(),
            roles: std::array::from_fn(|_| RoleFacts::default()),
            next_wave: 0,
            retiring: false,
        }
    }

    pub(super) fn begin(&mut self, kind: WaveKind, before: Vec<usize>) -> WaveObservation {
        let wave = self.next_wave;
        self.next_wave = self.next_wave.saturating_add(1);
        WaveObservation {
            wave,
            kind,
            before,
            started: Instant::now(),
            poll_wall: Duration::ZERO,
        }
    }

    pub(super) fn computed(&mut self, wave: &WaveObservation, participants: usize) {
        let role = &mut self.roles[slot(wave.kind)];
        role.computed_waves += 1;
        role.multi_participant_waves += u64::from(participants > 1);
        role.maximum_participants = role.maximum_participants.max(participants);
        role.wave_wall_us += wave.started.elapsed().as_micros();
        role.poll_wall_us += wave.poll_wall.as_micros();
    }

    /// Each entry comes from the original selected receipt: main/draft frontier,
    /// and, for Verify only, (accepted proposal, published output token count).
    pub(super) fn settled(
        &mut self,
        wave: &WaveObservation,
        indices: &[usize],
        outputs: &[Option<(u64, Option<(bool, usize)>)>],
        elapsed: Duration,
    ) {
        self.roles[slot(wave.kind)].settlement_wall_us += elapsed.as_micros();
        for (input_index, (&index, output)) in indices.iter().zip(outputs).enumerate() {
            if let Some((_, verification)) = output {
                let request = &mut self.requests[index];
                if wave.kind == WaveKind::Draft {
                    request.draft_committed_waves += 1;
                } else {
                    request.main_committed_waves += 1;
                }
                if let Some((accepted, tokens)) = verification {
                    request.verified_proposals += 1;
                    request.accepted_proposals += u64::from(*accepted);
                    request.rejected_proposals += u64::from(!accepted);
                    request.verified_output_tokens += *tokens as u64;
                }
            }
            if wave.kind != WaveKind::Draft {
                if let Some(observation) = &self.observations[index] {
                    observation.wave(
                        wave.wave,
                        indices.len(),
                        input_index,
                        wave.before[input_index],
                        output
                            .as_ref()
                            .and_then(|(frontier, _)| usize::try_from(*frontier).ok()),
                        if output.is_some() {
                            "completed"
                        } else {
                            "failed"
                        },
                        super::super::execute::ExpansionWaveOutcome::Completed,
                    );
                }
            }
        }
    }

    pub(super) fn ended(&mut self, index: usize, ids: &[u32], finish: FinishReason) {
        if !self.ended[index] {
            if let Some(observation) = &mut self.observations[index] {
                observation.tokens(ids, finish);
            }
            self.ended[index] = true;
        }
    }

    pub(super) fn retiring(&mut self) {
        if !self.retiring {
            for observation in self.observations.iter_mut().flatten() {
                observation.retiring(None);
            }
            self.retiring = true;
        }
    }

    pub(super) fn record(
        &mut self,
        results: &[Option<ExpansionParticipantResult>],
        normal: bool,
        failed: bool,
    ) {
        for (index, observation) in self.observations.iter_mut().enumerate() {
            let Some(mut observation) = observation.take() else {
                continue;
            };
            observation.set_metrics(serde_json::json!({"mtp": {
                "mode": "one_draft", "clock": "monotonic_host_wall",
                "roles_scope": "shared_cohort_do_not_sum_across_requests",
                "wave_wall_includes_capacity_wait_and_poll_gaps": true,
                "poll_wall_includes_synchronous_native_wait": true,
                "prefill": self.roles[0], "draft": self.roles[1],
                "verify": self.roles[2], "last": self.roles[3],
                "request": self.requests[index]
            }}));
            let (outcome, text) = if normal {
                match &results[index] {
                    Some(ExpansionParticipantResult::Expanded(text)) => {
                        ("expanded", Some(text.as_str()))
                    }
                    Some(ExpansionParticipantResult::Cancelled) => ("cancelled", None),
                    _ => ("failed", None),
                }
            } else if failed {
                ("failed", None)
            } else {
                ("cancelled", None)
            };
            observation.record(true, outcome, text);
        }
    }
}

fn slot(kind: WaveKind) -> usize {
    match kind {
        WaveKind::Prefill => 0,
        WaveKind::Draft => 1,
        WaveKind::Verify => 2,
        WaveKind::Last => 3,
    }
}
