//! Bounded output readback and selected-prefix settlement on the original
//! admitted Core frame. No accepted hidden row is published before retirement.
use super::*;

#[must_use = "failed causal work retains its original frame and recovery owners"]
pub struct CausalWaveFailure<R: DeviceRuntime> {
    message: String,
    step: Option<Arc<StepResourceLease<R>>>,
    reaper: Arc<CompletionReaper<R>>,
    completion: Option<CompletionHandle<R>>,
    indeterminate: Option<IndeterminateSubmissionHandle<R>>,
    dispatch: Option<SubmissionWaveDispatchError<R>>,
}
impl<R: DeviceRuntime> CausalWaveFailure<R> {
    fn new(step: Arc<StepResourceLease<R>>, reaper: Arc<CompletionReaper<R>>) -> Self {
        Self {
            message: String::new(),
            step: Some(step),
            reaper,
            completion: None,
            indeterminate: None,
            dispatch: None,
        }
    }
    fn error(mut self, message: impl std::fmt::Display) -> Self {
        self.message = message.to_string();
        self
    }
    pub fn completion(&self) -> Option<&CompletionHandle<R>> {
        self.completion.as_ref()
    }
    pub fn indeterminate(&self) -> Option<&IndeterminateSubmissionHandle<R>> {
        self.indeterminate.as_ref()
    }
    pub fn reaper(&self) -> &Arc<CompletionReaper<R>> {
        &self.reaper
    }
    pub fn dispatch_failure(&self) -> Option<&SubmissionWaveDispatchError<R>> {
        self.dispatch.as_ref()
    }

    /// Caller first recovers any indeterminate native work using the original
    /// handles. Core refuses abort while any submitted operation still holds
    /// this frame. Dropping this result is not a successful retirement.
    pub fn try_abort(mut self) -> std::result::Result<StepRetirementReceipt, Self> {
        // Recovery handles remain available if Core cannot abort the frame.
        // This retry owns an unsubmitted PreparedStepSubmissionWave, which
        // otherwise keeps a Step Arc alive and makes checked abort impossible.
        // Only the explicit definitely-not-submitted receipt authorizes drop.
        if matches!(
            self.dispatch,
            Some(SubmissionWaveDispatchError::DefinitelyNotSubmitted { .. })
        ) {
            drop(self.dispatch.take());
        }
        let step = self
            .step
            .take()
            .expect("causal failure owns its original frame");
        match step.try_abort() {
            Ok(receipt) => Ok(receipt),
            Err(failure) => {
                self.message = failure.error().to_string();
                self.step = Some(failure.into_step());
                Err(self)
            }
        }
    }
}
impl<R: DeviceRuntime> std::fmt::Display for CausalWaveFailure<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}
impl<R: DeviceRuntime> std::fmt::Debug for CausalWaveFailure<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self, f)
    }
}
impl<R: DeviceRuntime> std::error::Error for CausalWaveFailure<R> {}

struct ParticipantOutput {
    session: SequenceSessionFingerprint,
    start: u64,
    rows: u64,
    decisions: [u32; 2],
    hidden: ManagedHostBytes,
    proposed_token: Option<u32>,
}

/// Computed candidates can be inspected for a decision, but are not accepted
/// sequence state. This owner keeps the same Step through the copy fence.
#[must_use = "computed candidates need selected or ordinary Core retirement"]
pub struct CausalComputedWave<R: DeviceRuntime> {
    owner: CausalWaveFailure<R>,
    candidate: bool,
    outputs: Vec<ParticipantOutput>,
}

impl<R: DeviceRuntime> CausalProgramSession<R> {
    /// Execute one already admitted complete-plan wave. Admissions and backing
    /// deferrals stay with the caller's existing scheduler; this method does
    /// not queue requests or turn a client batch into serial model executions.
    pub fn execute_prepared_wave(
        &self,
        batch: &ExecutionBatchParticipants<R>,
        prepared: PreparedStepSubmissionWave<R>,
        uploads: Vec<SubmissionWaveInputUpload>,
        reaper: Arc<CompletionReaper<R>>,
    ) -> std::result::Result<CausalComputedWave<R>, CausalWaveFailure<R>> {
        let step = Arc::clone(prepared.step_resources());
        let mut owner = CausalWaveFailure::new(step, reaper);
        let terminal = match self.terminal.as_ref() {
            Some(terminal) => terminal,
            None => return Err(owner.error("installation-only graphs cannot execute token waves")),
        };
        let prepare = (|| -> Result<_> {
            let step = owner.step.as_ref().expect("original frame");
            if prepared.claimed_backing().plan_hash() != self.plan.execution_plan().plan_hash()
                || prepared.node_count() != self.plan.execution_plan().payload().nodes().len()
            {
                return Err(FerrumError::model(
                    "causal wave differs from the complete compiled plan",
                ));
            }
            let spans = step
                .work_shape()
                .participant_token_ranges()
                .map_err(|e| FerrumError::model(e.to_string()))?;
            if spans.len() != batch.sessions().len() || spans.is_empty() {
                return Err(FerrumError::model(
                    "causal wave participant membership differs",
                ));
            }
            validate_uploads(&self.plan, &self.host, spans, &uploads)?;
            let mut readbacks = Vec::with_capacity(spans.len() * 2);
            let mut sources = Vec::with_capacity(spans.len());
            for (index, (session, span)) in batch.sessions().iter().zip(spans).enumerate() {
                let proposed_token = if span.immediate_tokens() == 2 {
                    let (node, port) = self
                        .plan
                        .execution_plan()
                        .payload()
                        .nodes()
                        .iter()
                        .flat_map(|node| node.values().iter().map(move |port| (node, port)))
                        .find(|(_, port)| {
                            port.role() == ResolvedValueRole::Input
                                && port.value_id() == &terminal.spec.token_input
                        })
                        .ok_or_else(|| FerrumError::model("causal token port missing"))?;
                    let token_offset = span
                        .source_token_range()
                        .start
                        .checked_add(1)
                        .and_then(|row| row.checked_mul(4))
                        .ok_or_else(|| FerrumError::model("proposal token range overflow"))?;
                    let upload = uploads
                        .iter()
                        .find(|u| {
                            u.node_id() == node.id()
                                && u.input_ordinal() == port.ordinal()
                                && u.participant_index() == index as u32
                                && u.logical_offset_bytes() <= token_offset
                                && u.logical_offset_bytes()
                                    .checked_add(u.bytes().len() as u64)
                                    .is_some_and(|end| end >= token_offset + 4)
                        })
                        .ok_or_else(|| FerrumError::model("causal token upload missing"))?;
                    let offset = usize::try_from(token_offset - upload.logical_offset_bytes())
                        .map_err(|_| FerrumError::model("proposal token source offset overflow"))?;
                    Some(u32::from_le_bytes(
                        upload.bytes()[offset..offset + 4].try_into().unwrap(),
                    ))
                } else {
                    None
                };
                sources.push((
                    session.fingerprint().clone(),
                    span.source_token_range().start,
                    span.immediate_tokens(),
                    proposed_token,
                ));
                for port in [&terminal.decisions, &terminal.hidden] {
                    let component = &port.storage().components()[0];
                    let bytes = port
                        .tensor()
                        .minimum_storage_bytes()
                        .map_err(|e| FerrumError::model(e.to_string()))?;
                    let layout = HostTransferLayout::new(
                        port.tensor().element_type(),
                        bytes / port.tensor().element_type().size_bytes(),
                    )
                    .map_err(|e| FerrumError::model(e.to_string()))?;
                    let request = CompletionReadbackRequest::new_typed(
                        terminal.node.clone(),
                        u32::try_from(index)
                            .map_err(|_| FerrumError::model("participant index overflow"))?,
                        component.resource_id().clone(),
                        port.usage(),
                        component.offset_bytes(),
                        layout,
                    )
                    .map_err(|e| FerrumError::model(e.to_string()))?;
                    let output = self
                        .host
                        .allocate_zeroed(
                            HostMemoryClass::Output,
                            usize::try_from(bytes)
                                .map_err(|_| FerrumError::model("readback extent overflow"))?,
                        )
                        .map_err(|e| FerrumError::resource_exhausted(e.to_string()))?;
                    readbacks.push((request, output));
                }
            }
            let active = batch
                .sessions()
                .iter()
                .map(|s| TrustedActiveSequenceBinding::from_session(s))
                .collect::<std::result::Result<Vec<_>, _>>()
                .map_err(|e| FerrumError::model(e.to_string()))?;
            let identity = OperationDispatch::bind_submission_wave_identity(
                &self.plan,
                active.iter(),
                &prepared,
                step.execution_lane(),
            )
            .map_err(|e| FerrumError::model(e.to_string()))?;
            Ok((
                readbacks,
                sources,
                active,
                identity,
                Arc::clone(step.execution_lane()),
            ))
        })();
        let (readbacks, sources, active, identity, lane) = match prepare {
            Ok(prepared) => prepared,
            Err(error) => return Err(owner.error(error)),
        };
        let handle = match OperationDispatch::encode_and_submit_wave_with_inputs(
            self.providers.providers(),
            &self.plan,
            &identity,
            active.iter(),
            DeviceTimingMode::Off,
            &uploads,
            prepared,
            &lane,
            &owner.reaper,
        ) {
            Ok(handle) => handle,
            Err(error) => {
                owner.message = format!("causal dispatch: {error:?}");
                owner.dispatch = Some(error);
                return Err(owner);
            }
        };
        owner.completion = Some(handle);
        let observation = match owner
            .completion
            .as_ref()
            .unwrap()
            .wait_with_managed_readbacks(readbacks)
        {
            Ok(observation) => observation,
            Err(error) => return Err(owner.error(error)),
        };
        let (completion, outputs) = observation.into_parts();
        if !matches!(&completion, CompletionObservation::Terminal(receipt)
            if matches!(receipt.disposition(), OperationCompletionDisposition::Succeeded))
        {
            return Err(owner.error(format!("causal compute did not succeed: {completion:?}")));
        }
        let outputs = match outputs {
            Some(Ok(outputs)) if outputs.len() == sources.len() * 2 => outputs,
            Some(Err(error)) => return Err(owner.error(error)),
            _ => return Err(owner.error("causal readback omitted bounded participant outputs")),
        };
        let mut result = Vec::with_capacity(sources.len());
        let mut outputs = outputs.into_iter();
        for (session, start, rows, proposed_token) in sources {
            let decisions = outputs.next().unwrap();
            let hidden = outputs.next().unwrap();
            if decisions.len() != 8
                || !self.host.owns(&decisions)
                || !self.host.owns(&hidden)
                || decisions.class() != HostMemoryClass::Output
                || hidden.class() != HostMemoryClass::Output
            {
                return Err(
                    owner.error("causal readback has foreign owners or malformed decisions")
                );
            }
            let decisions = [
                u32::from_le_bytes(decisions.bytes()[..4].try_into().unwrap()),
                u32::from_le_bytes(decisions.bytes()[4..].try_into().unwrap()),
            ];
            if decisions
                .iter()
                .any(|&id| id as usize >= self.vocabulary_size)
            {
                return Err(owner.error("causal head did not select valid vocabulary tokens"));
            }
            result.push(ParticipantOutput {
                session,
                start,
                rows,
                decisions,
                hidden,
                proposed_token,
            });
        }
        drop((active, identity, lane, uploads));
        owner.completion.take();
        Ok(CausalComputedWave {
            owner,
            candidate: self.plan.execution_plan().requires_selected_prefix_commit(),
            outputs: result,
        })
    }
}

fn validate_uploads(
    plan: &ResolvedModelPlan,
    host: &HostMemoryAccount,
    spans: &[BatchParticipantTokenRange],
    uploads: &[SubmissionWaveInputUpload],
) -> Result<()> {
    let inputs = plan.parts().prepared_family.program().inputs();
    let mut expected_uploads = 0usize;
    for node in plan.execution_plan().payload().nodes() {
        for port in node
            .values()
            .iter()
            .filter(|v| v.role() == ResolvedValueRole::Input && inputs.contains(v.value_id()))
        {
            if port.access() != TensorAccess::Read
                || !matches!(port.tensor().layout(), ResolvedTensorLayout::Contiguous)
            {
                return Err(FerrumError::model(
                    "causal external inputs must be contiguous readonly values",
                ));
            }
            for (index, span) in spans.iter().enumerate() {
                let key = (node.id().clone(), index as u32, port.ordinal());
                let mut matching = uploads
                    .iter()
                    .filter(|u| {
                        u.node_id() == &key.0
                            && u.participant_index() == key.1
                            && u.input_ordinal() == key.2
                    })
                    .collect::<Vec<_>>();
                if matching.is_empty() {
                    return Err(FerrumError::model("causal wave input missing"));
                }
                let (offset, bytes) = if let Some(projection) =
                    node.work().token_projection(port.role(), port.ordinal())
                {
                    if projection.axis() != 0 {
                        return Err(FerrumError::unsupported(
                            "causal inputs require a leading token axis",
                        ));
                    }
                    let row = port.tensor().dimensions()[1..]
                        .iter()
                        .try_fold(port.tensor().element_type().size_bytes(), |a, &b| {
                            a.checked_mul(b)
                        })
                        .ok_or_else(|| FerrumError::model("causal row extent overflow"))?;
                    (
                        span.source_token_range().start.checked_mul(row),
                        span.immediate_tokens().checked_mul(row),
                    )
                } else {
                    (
                        Some(0),
                        Some(
                            port.tensor()
                                .minimum_storage_bytes()
                                .map_err(|e| FerrumError::model(e.to_string()))?,
                        ),
                    )
                };
                let mut cursor =
                    offset.ok_or_else(|| FerrumError::model("causal input offset overflow"))?;
                let end = bytes
                    .and_then(|bytes| cursor.checked_add(bytes))
                    .ok_or_else(|| FerrumError::model("causal input end overflow"))?;
                matching.sort_by_key(|u| u.logical_offset_bytes());
                expected_uploads = expected_uploads
                    .checked_add(matching.len())
                    .ok_or_else(|| FerrumError::model("causal input count overflow"))?;
                for upload in matching {
                    let source = upload.managed_bytes().ok_or_else(|| {
                        FerrumError::model("causal wave requires managed sources")
                    })?;
                    if !host.owns(source) || source.class() != HostMemoryClass::InputPreprocess {
                        return Err(FerrumError::model(
                            "causal input has foreign Host ownership/class",
                        ));
                    }
                    if source.is_empty()
                        || upload.logical_offset_bytes() != cursor
                        || upload.source_layout().element_type() != port.tensor().element_type()
                    {
                        return Err(FerrumError::model(
                            "causal input segments overlap, have gaps, or differ in dtype",
                        ));
                    }
                    cursor = cursor
                        .checked_add(source.len() as u64)
                        .filter(|&next| next <= end)
                        .ok_or_else(|| {
                            FerrumError::model("causal input segment exceeds admitted extent")
                        })?;
                }
                if cursor != end {
                    return Err(FerrumError::model(
                        "causal input segments omit admitted rows",
                    ));
                }
            }
        }
    }
    if expected_uploads != uploads.len() {
        return Err(FerrumError::model(
            "causal wave contains undeclared input uploads",
        ));
    }
    Ok(())
}

impl<R: DeviceRuntime> CausalComputedWave<R> {
    /// Abort through the same frame when its consumer is cancelled before
    /// selecting a prefix. No computed hidden output becomes accepted state.
    pub fn try_abort(self) -> std::result::Result<StepRetirementReceipt, CausalWaveFailure<R>> {
        let Self { owner, outputs, .. } = self;
        drop(outputs);
        owner.try_abort()
    }

    pub fn decisions(&self) -> impl ExactSizeIterator<Item = [u32; 2]> + '_ {
        self.outputs.iter().map(|p| p.decisions)
    }

    /// Verify the actual second input token against the first main decision.
    /// Each caller must have room for two new output tokens and two main input
    /// rows before scheduling this wave; the last-slot path uses ordinary N1.
    /// EOS at q0 commits only the root. No proposed token is ever published on
    /// rejection, and no publication survives cancellation or copy failure.
    pub fn commit_one_draft(
        self,
        stop_tokens: &[BTreeSet<u32>],
        remaining_output_tokens: &[usize],
    ) -> std::result::Result<Vec<Option<CausalVerifiedDraftOutput>>, CausalWaveFailure<R>> {
        if !self.candidate
            || stop_tokens.len() != self.outputs.len()
            || remaining_output_tokens.len() != self.outputs.len()
            || self
                .outputs
                .iter()
                .any(|p| p.rows != 2 || p.proposed_token.is_none())
            || remaining_output_tokens.iter().any(|&n| n < 2)
        {
            return Err(self.owner.error("one-draft verification needs actual two-row candidates and declared output capacity"));
        }
        let choices = self
            .outputs
            .iter()
            .zip(stop_tokens)
            .map(|(p, stops)| {
                let [first, second] = p.decisions;
                if stops.contains(&first) {
                    (SelectedPrefix::One, vec![first], true, false)
                } else if p.proposed_token == Some(first) {
                    (
                        SelectedPrefix::Two,
                        vec![first, second],
                        stops.contains(&second),
                        true,
                    )
                } else {
                    (SelectedPrefix::One, vec![first], false, false)
                }
            })
            .collect::<Vec<_>>();
        let selections = choices.iter().map(|choice| choice.0).collect();
        let committed = self.commit(selections)?;
        Ok(committed
            .into_outputs()
            .into_iter()
            .zip(choices)
            .map(|(state, (_, tokens, stopped, accepted_draft))| {
                state.map(|state| CausalVerifiedDraftOutput {
                    state,
                    tokens,
                    stopped,
                    accepted_draft,
                })
            })
            .collect())
    }

    pub fn commit(
        mut self,
        selections: Vec<SelectedPrefix>,
    ) -> std::result::Result<CausalCommittedWave, CausalWaveFailure<R>> {
        if selections.len() != self.outputs.len()
            || selections
                .iter()
                .zip(&self.outputs)
                .any(|(s, p)| matches!(s, SelectedPrefix::One | SelectedPrefix::Two) && p.rows != 2)
            || (!self.candidate && selections.iter().any(|s| *s != SelectedPrefix::All))
        {
            return Err(self
                .owner
                .error("causal selections differ from the actual wave"));
        }
        if self.candidate {
            let commit = match self
                .owner
                .step
                .as_ref()
                .unwrap()
                .try_prepare_selected_state_commit(selections.clone())
            {
                Ok(commit) => commit,
                Err(error) => return Err(self.owner.error(error)),
            };
            let submission = match self.owner.reaper.submit_selected_state_commit(commit) {
                Ok(submission) => submission,
                Err(error) => return Err(self.owner.error(error)),
            };
            match submission {
                SelectedStateSubmission::Submitted(handle) => self.owner.completion = Some(handle),
                SelectedStateSubmission::ContractAfterSubmission { error, handle } => {
                    self.owner.completion = Some(handle);
                    return Err(self.owner.error(error));
                }
                SelectedStateSubmission::Indeterminate(handle) => {
                    self.owner.indeterminate = Some(handle);
                    return Err(self
                        .owner
                        .error("selected state submission is indeterminate"));
                }
            }
            let observation = match self.owner.completion.as_ref().unwrap().wait() {
                Ok(observation) => observation,
                Err(error) => return Err(self.owner.error(error)),
            };
            if !matches!(&observation, CompletionObservation::Terminal(receipt)
                if matches!(receipt.disposition(), OperationCompletionDisposition::Succeeded))
            {
                return Err(self.owner.error(format!(
                    "selected state copy did not succeed: {observation:?}"
                )));
            }
            self.owner.completion.take();
        }
        let step = self.owner.step.take().unwrap();
        let receipt = match if self.candidate {
            step.try_retire_selected()
        } else {
            step.try_retire_normal()
        } {
            Ok(receipt) => receipt,
            Err(error) => {
                self.owner.message = error.error().to_string();
                self.owner.step = Some(error.into_step());
                return Err(self.owner);
            }
        };
        let outputs = self
            .outputs
            .into_iter()
            .zip(selections)
            .zip(receipt.participants())
            .map(|((p, selection), retired)| {
                if retired.disposition() != StepParticipantRetirementDisposition::Committed {
                    return None;
                }
                let count = if selection == SelectedPrefix::One {
                    1
                } else {
                    p.rows
                };
                let slot = if selection == SelectedPrefix::One {
                    0
                } else {
                    1
                };
                Some(CausalAcceptedOutput {
                    session: p.session,
                    frontier: p.start + count,
                    decisions: p.decisions,
                    hidden: p.hidden,
                    slot,
                })
            })
            .collect();
        Ok(CausalCommittedWave { receipt, outputs })
    }
}

pub struct CausalVerifiedDraftOutput {
    state: CausalAcceptedOutput,
    tokens: Vec<u32>,
    stopped: bool,
    accepted_draft: bool,
}
impl CausalVerifiedDraftOutput {
    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }
    pub fn stopped(&self) -> bool {
        self.stopped
    }
    pub fn accepted_draft(&self) -> bool {
        self.accepted_draft
    }
    pub fn into_state(self) -> CausalAcceptedOutput {
        self.state
    }
}

pub struct CausalCommittedWave {
    receipt: StepRetirementReceipt,
    outputs: Vec<Option<CausalAcceptedOutput>>,
}
impl CausalCommittedWave {
    pub fn receipt(&self) -> &StepRetirementReceipt {
        &self.receipt
    }
    pub fn into_outputs(self) -> Vec<Option<CausalAcceptedOutput>> {
        self.outputs
    }
}

/// Constructed only after the original frame commits. Cancelled participants
/// have no such output. The source identity and accepted frontier accompany
/// the exact selected original F16 row into the draft input handoff.
pub struct CausalAcceptedOutput {
    session: SequenceSessionFingerprint,
    frontier: u64,
    decisions: [u32; 2],
    hidden: ManagedHostBytes,
    slot: usize,
}
impl CausalAcceptedOutput {
    pub fn session(&self) -> &SequenceSessionFingerprint {
        &self.session
    }
    pub fn frontier(&self) -> u64 {
        self.frontier
    }
    pub fn decisions(&self) -> [u32; 2] {
        self.decisions
    }
    pub fn selected_decision(&self) -> u32 {
        self.decisions[self.slot]
    }
    pub fn copy_hidden_input(&self, host: &HostMemoryAccount) -> Result<ManagedHostBytes> {
        if !host.owns(&self.hidden) {
            return Err(FerrumError::model(
                "accepted hidden source belongs to another Host account",
            ));
        }
        let bytes = self.hidden.len() / 2;
        let mut input = host
            .allocate_zeroed(HostMemoryClass::InputPreprocess, bytes)
            .map_err(|e| FerrumError::resource_exhausted(e.to_string()))?;
        input
            .as_mut()
            .copy_from_slice(&self.hidden.bytes()[self.slot * bytes..(self.slot + 1) * bytes]);
        Ok(input.freeze())
    }
}
