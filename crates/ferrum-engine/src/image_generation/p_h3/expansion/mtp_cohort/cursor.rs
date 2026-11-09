//! One actual main/draft sequence pair. Main accepted position and draft KV
//! append position are separate; only committed main outputs enter history.
use super::*;
use ferrum_execution::{CausalAcceptedOutput, CausalVerifiedDraftOutput};
use ferrum_models::vnext::qwen35::{self, mtp as model};
use ferrum_runtime::{
    ElementType, HostMemoryClass, ProgramValueId, SequenceSession, SequenceSessionTerminalReceipt,
};
use ferrum_types::{FinishReason, TokenId};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum WaveKind {
    Prefill,
    Draft,
    Verify,
    Last,
}

pub(super) struct Cursor<R: DeviceRuntime> {
    pub(super) prepared: PreparedTextExpansion,
    pub(super) main: Arc<SequenceSession<R>>,
    pub(super) draft: Option<Arc<SequenceSession<R>>>,
    pub(super) inputs: Arc<ferrum_execution::CausalRequestInputs>,
    pub(super) output: execute::ManagedTokenHistory,
    pub(super) finish: Option<FinishReason>,
    history: Vec<u32>,
    draft_history: Vec<u32>,
    frontier: usize,
    accepted: Option<CausalAcceptedOutput>,
    proposal: Option<u32>,
    main_terminal: Option<SequenceSessionTerminalReceipt>,
    draft_terminal: Option<SequenceSessionTerminalReceipt>,
}

impl<R: DeviceRuntime> Cursor<R> {
    pub(super) fn new(
        prepared: PreparedTextExpansion,
        main: Arc<SequenceSession<R>>,
        inputs: Arc<ferrum_execution::CausalRequestInputs>,
        host: &HostMemoryAccount,
    ) -> Result<Self> {
        if prepared.tokens.is_empty() || prepared.prefill_chunk_tokens == 0 {
            return Err(FerrumError::request_validation(
                "MTP requires a nonempty prompt and chunk",
            ));
        }
        let output = execute::ManagedTokenHistory::new(host, prepared.generation.max_new_tokens)?;
        Ok(Self {
            history: prepared.tokens.iter().map(|token| token.get()).collect(),
            prepared,
            main,
            draft: None,
            inputs,
            output,
            finish: None,
            draft_history: Vec::new(),
            frontier: 0,
            accepted: None,
            proposal: None,
            main_terminal: None,
            draft_terminal: None,
        })
    }

    pub(super) fn remaining(&self) -> usize {
        self.prepared.generation.max_new_tokens as usize - self.output.ids().len()
    }

    /// Updated only after validating the original accepted main receipt.
    pub(super) fn accepted_frontier(&self) -> usize {
        self.frontier
    }

    pub(super) fn cancel(&mut self) -> Result<()> {
        if self.main_terminal.is_none() {
            self.main.request_cancel().map_err(core_error)?;
        }
        if self.draft_terminal.is_none() {
            if let Some(draft) = &self.draft {
                draft.request_cancel().map_err(core_error)?;
            }
        }
        self.finish = Some(FinishReason::Cancelled);
        Ok(())
    }

    /// Record each original receipt immediately. If the other close fails,
    /// retry only its still-live authority; Core correctly rejects closing an
    /// already terminal session a second time.
    pub(super) fn close_sequences(&mut self, abort: bool) -> Result<()> {
        if self.main_terminal.is_none() {
            self.main_terminal = Some(
                if abort {
                    self.main
                        .try_abort_if_quiescent()
                        .or_else(|_| self.main.try_abort())
                } else {
                    self.main.try_complete()
                }
                .map_err(core_error)?,
            );
        }
        if self.draft_terminal.is_none() {
            if let Some(draft) = &self.draft {
                self.draft_terminal = Some(
                    if abort {
                        draft
                            .try_abort_if_quiescent()
                            .or_else(|_| draft.try_abort())
                    } else {
                        draft.try_complete()
                    }
                    .map_err(core_error)?,
                );
            }
        }
        Ok(())
    }

    pub(super) fn kind(&self) -> Option<WaveKind> {
        if self.finish.is_some() {
            return None;
        }
        if self.frontier < self.prepared.tokens.len() {
            return Some(WaveKind::Prefill);
        }
        if self.proposal.is_some() {
            return Some(WaveKind::Verify);
        }
        if self.remaining() == 1 || self.inputs.maximum_tokens() - self.frontier == 1 {
            Some(WaveKind::Last)
        } else {
            Some(WaveKind::Draft)
        }
    }

    pub(super) fn draft_initial_history(&self) -> Result<Arc<[u32]>> {
        if self.kind() != Some(WaveKind::Draft) || self.draft.is_some() {
            return Err(FerrumError::internal(
                "draft admission is not at its first pending token",
            ));
        }
        Ok(Arc::from([self.pending()?]))
    }

    fn pending(&self) -> Result<u32> {
        let accepted = self
            .accepted
            .as_ref()
            .ok_or_else(|| FerrumError::internal("MTP lost accepted main hidden state"))?;
        if accepted.session() != self.main.fingerprint()
            || accepted.frontier() != self.frontier as u64
            || self.history.len() != self.frontier + 1
            || self.history.last().copied() != Some(accepted.selected_decision())
        {
            return Err(FerrumError::internal(
                "MTP main hidden/token/frontier identity changed",
            ));
        }
        Ok(accepted.selected_decision())
    }

    pub(super) fn member(&self, host: &HostMemoryAccount) -> Result<core::Member<R>> {
        let kind = self
            .kind()
            .ok_or_else(|| FerrumError::internal("finished MTP request cannot execute"))?;
        let mut history = self.history.clone();
        let (session, range, token_input, mut inputs) = if kind == WaveKind::Draft {
            let pending = self.pending()?;
            let session = self
                .draft
                .as_ref()
                .ok_or_else(|| FerrumError::internal("MTP draft is not admitted"))?;
            history = self.draft_history.clone();
            history.push(pending);
            let range = self.draft_history.len()..history.len();
            let mut inputs = vec![core::ExternalInput {
                value: id(model::ACCEPTED_HIDDEN)?,
                element_type: ElementType::F16,
                bytes: self.accepted.as_ref().unwrap().copy_hidden_input(host)?,
            }];
            // Select main continuation RoPE, then upload it at the independent
            // draft KV cursor. No synthetic future-hidden conditioning table.
            for (source, target) in [
                (qwen35::ROPE_COS_INPUT_ID, model::ROPE_COS),
                (qwen35::ROPE_SIN_INPUT_ID, model::ROPE_SIN),
            ] {
                let (element_type, bytes) = self.inputs.materialize_rows(
                    host,
                    &id(source)?,
                    self.frontier..self.frontier + 1,
                )?;
                inputs.push(core::ExternalInput {
                    value: id(target)?,
                    element_type,
                    bytes,
                });
            }
            (Arc::clone(session), range, model::TOKEN_IDS, inputs)
        } else {
            let end =
                match kind {
                    WaveKind::Prefill => (self.frontier + self.prepared.prefill_chunk_tokens)
                        .min(self.prepared.tokens.len()),
                    WaveKind::Verify => {
                        self.pending()?;
                        history.push(self.proposal.ok_or_else(|| {
                            FerrumError::internal("MTP verification lost proposal")
                        })?);
                        self.frontier + 2
                    }
                    WaveKind::Last => {
                        self.pending()?;
                        self.frontier + 1
                    }
                    WaveKind::Draft => unreachable!(),
                };
            let range = self.frontier..end;
            let inputs = self
                .inputs
                .schema()
                .specs()
                .iter()
                .map(|spec| {
                    let (element_type, bytes) =
                        self.inputs
                            .materialize_rows(host, &spec.value_id, range.clone())?;
                    Ok(core::ExternalInput {
                        value: spec.value_id.clone(),
                        element_type,
                        bytes,
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            (
                Arc::clone(&self.main),
                range,
                "value.input.token_ids",
                inputs,
            )
        };
        let mut tokens = host
            .allocate_zeroed(HostMemoryClass::InputPreprocess, range.len() * 4)
            .map_err(|error| FerrumError::resource_exhausted(error.to_string()))?;
        for (bytes, token) in tokens
            .as_mut()
            .chunks_exact_mut(4)
            .zip(&history[range.clone()])
        {
            bytes.copy_from_slice(&token.to_le_bytes());
        }
        inputs.push(core::ExternalInput {
            value: id(token_input)?,
            element_type: ElementType::U32,
            bytes: tokens.freeze(),
        });
        Ok(core::Member {
            session,
            history: history.into(),
            range,
            inputs,
        })
    }

    pub(super) fn accept_draft(&mut self, output: CausalAcceptedOutput) -> Result<()> {
        if self.kind() != Some(WaveKind::Draft)
            || self
                .draft
                .as_ref()
                .is_none_or(|session| output.session() != session.fingerprint())
            || output.frontier() != self.draft_history.len() as u64 + 1
        {
            return Err(FerrumError::internal(
                "draft receipt belongs to another pair or KV frontier",
            ));
        }
        let pending = self.pending()?;
        self.draft_history.push(pending);
        self.proposal = Some(output.selected_decision());
        Ok(())
    }

    pub(super) fn accept_main(&mut self, output: CausalAcceptedOutput) -> Result<()> {
        let kind = self.kind();
        let end = match kind {
            Some(WaveKind::Prefill) => {
                (self.frontier + self.prepared.prefill_chunk_tokens).min(self.prepared.tokens.len())
            }
            Some(WaveKind::Last) => self.frontier + 1,
            _ => {
                return Err(FerrumError::internal(
                    "ordinary main receipt at wrong MTP stage",
                ))
            }
        };
        self.check_main_receipt(&output, end)?;
        self.frontier = end;
        if end >= self.prepared.tokens.len() {
            self.append_output(output.selected_decision())?;
            self.accepted = Some(output);
        }
        Ok(())
    }

    pub(super) fn accept_verified(&mut self, output: CausalVerifiedDraftOutput) -> Result<()> {
        if self.kind() != Some(WaveKind::Verify)
            || output.tokens().is_empty()
            || output.tokens().len() > self.remaining()
        {
            return Err(FerrumError::internal(
                "verification output at wrong MTP stage or extent",
            ));
        }
        let tokens = output.tokens().to_vec();
        let state = output.into_state();
        self.check_main_receipt(&state, self.frontier + tokens.len())?;
        if tokens.last().copied() != Some(state.selected_decision()) {
            return Err(FerrumError::internal(
                "verified pending token differs from selected main row",
            ));
        }
        self.frontier += tokens.len();
        self.proposal = None;
        for token in tokens {
            self.append_output(token)?;
        }
        self.accepted = Some(state);
        Ok(())
    }

    fn check_main_receipt(&self, output: &CausalAcceptedOutput, end: usize) -> Result<()> {
        if output.session() != self.main.fingerprint() || output.frontier() != end as u64 {
            return Err(FerrumError::internal(
                "accepted main receipt belongs to another sequence or frontier",
            ));
        }
        Ok(())
    }

    fn append_output(&mut self, token: u32) -> Result<()> {
        if self.finish.is_some() {
            return Err(FerrumError::internal(
                "MTP attempted publication after EOS/limit",
            ));
        }
        self.output.push(TokenId::new(token))?;
        self.history.push(token);
        if token == self.prepared.generation.eos_token_id {
            self.finish = Some(FinishReason::EOS);
        } else if self.remaining() == 0 {
            self.finish = Some(FinishReason::Length);
        }
        Ok(())
    }
}

fn id(value: &str) -> Result<ProgramValueId> {
    ProgramValueId::new(value).map_err(|error| FerrumError::model(error.to_string()))
}
