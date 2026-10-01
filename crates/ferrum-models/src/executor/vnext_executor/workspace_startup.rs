//! Real resource preparation before product readiness. There is no encode,
//! dispatch, initialization upload, sampling, or synthetic cost observation.
//! Disposable startup owners are explicitly aborted; their idle lane slots
//! remain resident under the same plan/device capacity limits as ordinary work.
use super::*;
use ferrum_types::WorkspacePreparationMode;

#[derive(Debug, Clone, Serialize)]
pub(super) struct WorkspacePreparationReport {
    mode: WorkspacePreparationMode,
    prepared_buckets: Vec<WorkspaceBucketReceipt>,
    maximum_simultaneous_startup_owners: usize,
    encoded_model_waves: u64,
    submitted_model_waves: u64,
    before: WorkspaceCapacitySnapshot,
    after: WorkspaceCapacitySnapshot,
    elapsed_ms: u64,
}

#[derive(Debug, Clone, Serialize)]
struct WorkspaceBucketReceipt {
    bucket: ReusableExecutionBucketId,
    sequences: usize,
    tokens_per_sequence: usize,
    step: BatchStepId,
    invocation: BatchInvocationId,
}

#[derive(Debug, Clone, Serialize)]
struct WorkspaceCapacitySnapshot {
    epochs: ExecutorAdmissionEpochs,
    process_claimed_bytes: u64,
    effective_device_usable_ceiling_bytes: u64,
    pools: Vec<WorkspacePoolSnapshot>,
}
#[derive(Debug, Clone, Serialize)]
struct WorkspacePoolSnapshot {
    pool: DynamicBackingPoolId,
    resident_bytes: u64,
    free_bytes: u64,
}
impl WorkspaceCapacitySnapshot {
    fn capture<R: DeviceRuntime>(resources: &PlanRuntimeResources<R>) -> Result<Self> {
        let status = resources
            .dynamic_pool_status()
            .map_err(|error| FerrumError::backend(error.to_string()))?;
        Ok(Self {
            epochs: ExecutorAdmissionEpochs::from_capacity(status.epochs()),
            process_claimed_bytes: status.process_claimed_bytes(),
            effective_device_usable_ceiling_bytes: status.effective_device_usable_ceiling_bytes(),
            pools: status
                .pools()
                .iter()
                .map(|pool| WorkspacePoolSnapshot {
                    pool: pool.pool_id().clone(),
                    resident_bytes: pool.resident_bytes(),
                    free_bytes: pool.free_bytes(),
                })
                .collect(),
        })
    }
}

#[derive(Debug, Clone)]
struct WorkspaceCase {
    bucket: ReusableExecutionBucketId,
    kind: VNextExecutionWaveKind,
    sequences: usize,
    tokens_per_sequence: usize,
}

fn declared_cases(
    buckets: impl IntoIterator<Item = ReusableExecutionBucketSpec>,
    maximum_sequences: usize,
    maximum_tokens: usize,
    maximum_model_tokens: usize,
) -> Result<Vec<WorkspaceCase>> {
    let mut cases = Vec::new();
    let mut seen = BTreeSet::new();
    for bucket in buckets {
        let sequences = bucket.capacity().maximum_sequences() as usize;
        let tokens = usize::try_from(bucket.capacity().maximum_tokens())
            .map_err(|_| FerrumError::config("workspace token capacity exceeds usize"))?;
        if sequences == 0
            || sequences > maximum_sequences
            || sequences > MAXIMUM_REUSABLE_EXECUTION_STARTUP_CAPTURE_WIDTH
            || tokens == 0
            || tokens > maximum_tokens
            || !seen.insert(bucket.bucket_id().clone())
        {
            return Err(FerrumError::config(
                "workspace startup bucket exceeds declared capacity or is duplicated",
            ));
        }
        let (kind, tokens_per_sequence) = match bucket.class_id().as_str() {
            UNIFORM_QUERY_REUSABLE_CLASS if tokens == sequences => {
                (VNextExecutionWaveKind::Decode, 1)
            }
            PACKED_TOKEN_REUSABLE_CLASS if sequences == 1 && tokens <= maximum_model_tokens => {
                (VNextExecutionWaveKind::Prefill, tokens)
            }
            _ => {
                return Err(FerrumError::config(
                    "workspace startup does not support the declared bucket class/shape",
                ))
            }
        };
        cases.push(WorkspaceCase {
            bucket: bucket.bucket_id().clone(),
            kind,
            sequences,
            tokens_per_sequence,
        });
    }
    if cases.is_empty() {
        return Err(FerrumError::config(
            "workspace startup requires declared reusable resource buckets",
        ));
    }
    // Large claims first, but every declared bucket must succeed. This never
    // shrinks admission or silently drops a failed case.
    cases.sort_by_key(|case| std::cmp::Reverse((case.sequences, case.tokens_per_sequence)));
    Ok(cases)
}

/// Prepared-wave Drop releases its real flights/leases but is not a rollback
/// receipt. Explicit abort terminates only these disposable startup sessions.
fn finish_resource_only_wave<R: DeviceRuntime>(
    wave: PreparedStepSubmissionWave<R>,
    step: Arc<StepResourceLease<R>>,
) -> Result<(BatchStepId, BatchInvocationId)> {
    let identity = (wave.batch_step_id(), wave.batch_invocation_id());
    if identity.0 != step.batch_step_id() {
        return Err(FerrumError::internal(
            "workspace wave belongs to a different Step",
        ));
    }
    drop(wave);
    step.try_abort().map_err(|failure| {
        FerrumError::backend(format!(
            "workspace startup exact Step abort failed: {}",
            failure.error()
        ))
    })?;
    Ok(identity)
}

fn poll_resource_budget(budget: &mut dyn ResourcePlanningBudget) -> Result<()> {
    if budget.has_budget() {
        Ok(())
    } else {
        Err(FerrumError::resource_exhausted(
            "resource preparation original deadline expired",
        ))
    }
}

impl<R: DeviceRuntime> VNextModelExecutor<R> {
    pub(super) async fn prepare_workspace_startup(
        &self,
    ) -> Result<Option<WorkspacePreparationReport>> {
        if self.workspace_preparation == WorkspacePreparationMode::DemandDriven {
            return Ok(None);
        }
        let started = Instant::now();
        let memory = self
            .resolved_plan
            .execution_plan()
            .payload()
            .memory()
            .reusable_execution()
            .ok_or_else(|| {
                FerrumError::config(
                    "workspace startup is unsupported without a reusable memory plan",
                )
            })?;
        // A device-program policy adds lane-stable binding resources to the
        // same prepared wave. Claim them through the ordinary plan, then abort
        // without encoding. Actual program configuration/capture still belongs
        // to prepare_reusable_execution_startup, after this resource-only phase.
        if self.sequences.lock().total_len() != 0 {
            return Err(FerrumError::internal(
                "workspace startup requires an empty product registry",
            ));
        }
        let program_preparation = memory
            .program_policy()
            .map(|_| {
                let state = self
                    .lane
                    .cost_graph_stream_state()
                    .map_err(|error| FerrumError::backend(error.to_string()))?;
                state
                    .filter(|state| state.is_unconfigured_empty())
                    .ok_or_else(|| {
                        FerrumError::config(
                        "workspace startup requires an observed empty, unconfigured program cache",
                    )
                    })
            })
            .transpose()?;
        let cases = declared_cases(
            memory
                .buckets()
                .iter()
                .map(|resolved| resolved.bucket().clone()),
            self.policy.memory().maximum_active_sequences as usize,
            usize::try_from(self.policy.admission().maximum_scheduled_tokens)
                .map_err(|_| FerrumError::config("workspace scheduled capacity exceeds usize"))?,
            self.maximum_model_tokens,
        )?;
        let before = WorkspaceCapacitySnapshot::capture(&self.plan_resources)?;
        let submissions = self.metrics.submitted_waves.load(Ordering::Acquire);
        let mut prepared = Vec::with_capacity(cases.len());
        let maximum_simultaneous_startup_owners =
            cases.iter().map(|case| case.sequences).max().unwrap_or(0);
        for case in &cases {
            prepared.push(self.prepare_workspace_case(case).await?);
        }
        if self.sequences.lock().total_len() != 0 {
            return Err(FerrumError::internal(
                "workspace startup retained disposable owners",
            ));
        }
        if self.metrics.submitted_waves.load(Ordering::Acquire) != submissions {
            return Err(FerrumError::internal(
                "resource-only workspace startup unexpectedly submitted model work",
            ));
        }
        if let Some(before) = program_preparation {
            // Both program preparation receipts and the catalog require prior
            // configuration. Read the actual cache's unconfigured state; this
            // resource-only phase must leave it unchanged.
            let after = self
                .lane
                .cost_graph_stream_state()
                .map_err(|error| FerrumError::backend(error.to_string()))?;
            if after != Some(before) {
                return Err(FerrumError::internal(
                    "resource-only workspace startup unexpectedly prepared device programs",
                ));
            }
        }
        Ok(Some(WorkspacePreparationReport {
            mode: self.workspace_preparation,
            prepared_buckets: prepared,
            maximum_simultaneous_startup_owners,
            encoded_model_waves: 0,
            submitted_model_waves: 0,
            before,
            after: WorkspaceCapacitySnapshot::capture(&self.plan_resources)?,
            elapsed_ms: started.elapsed().as_millis().min(u64::MAX as u128) as u64,
        }))
    }

    pub(super) fn prepare_declared_resource_shapes(
        &self,
        requests: &[ferrum_interfaces::model_executor::ExecutorResourcePreparationRequest],
        budget: &mut dyn ResourcePlanningBudget,
    ) -> Result<ferrum_interfaces::model_executor::ExecutorResourcePreparationReceipt> {
        use ferrum_interfaces::model_executor::{
            ExecutorResourcePreparationOutcome as Outcome, ExecutorResourcePreparationReceipt,
        };
        if requests.is_empty() {
            return Err(FerrumError::invalid_request(
                "resource preparation has no declared shapes",
            ));
        }
        let mut ordered = Vec::with_capacity(requests.len());
        for &request in requests {
            if !budget.has_budget() {
                return Ok(ExecutorResourcePreparationReceipt {
                    outcome: Outcome::Unavailable,
                    prepared_participants: 0,
                });
            }
            let kind = match request.kind() {
                ferrum_interfaces::execution_cost::ActualWaveKind::Prefill => {
                    VNextExecutionWaveKind::Prefill
                }
                ferrum_interfaces::execution_cost::ActualWaveKind::Decode => {
                    VNextExecutionWaveKind::Decode
                }
                _ => {
                    return Err(FerrumError::invalid_request(
                        "unsupported resource preparation phase",
                    ))
                }
            };
            let retained = u32::try_from(request.participants())
                .ok()
                .and_then(|rows| {
                    request
                        .participants()
                        .checked_mul(request.tokens_per_sequence())
                        .and_then(|tokens| u64::try_from(tokens).ok())
                        .and_then(|tokens| self.reusable_bucket_for_shape(kind, rows, tokens, 0))
                })
                .is_some();
            ordered.push((!retained, request));
        }
        // Resident lane slots consume shared pools after their owner retires.
        // Materialize those real slots first so later transient preparation
        // accounts for all retained claims. No extra shape or allocation is added.
        ordered.sort_by_key(|(transient, _)| *transient);
        let mut prepared_participants = 0usize;
        for (_, request) in ordered {
            if !budget.has_budget() {
                return Ok(ExecutorResourcePreparationReceipt {
                    outcome: Outcome::Unavailable,
                    prepared_participants,
                });
            }
            let outcome = self.prepare_declared_resource_shape(request, budget)?;
            if outcome == Outcome::Prepared {
                prepared_participants = prepared_participants
                    .checked_add(request.participants())
                    .ok_or_else(|| {
                    FerrumError::internal("resource preparation receipt overflow")
                })?;
            }
            if outcome != Outcome::Prepared || !budget.has_budget() {
                return Ok(ExecutorResourcePreparationReceipt {
                    outcome: if outcome == Outcome::Prepared {
                        Outcome::Unavailable
                    } else {
                        outcome
                    },
                    prepared_participants,
                });
            }
        }
        Ok(ExecutorResourcePreparationReceipt {
            outcome: Outcome::Prepared,
            prepared_participants,
        })
    }

    fn prepare_declared_resource_shape(
        &self,
        request: ferrum_interfaces::model_executor::ExecutorResourcePreparationRequest,
        budget: &mut dyn ResourcePlanningBudget,
    ) -> Result<ferrum_interfaces::model_executor::ExecutorResourcePreparationOutcome> {
        use ferrum_interfaces::model_executor::ExecutorResourcePreparationOutcome as Outcome;
        if !budget.has_budget() {
            return Ok(Outcome::Unavailable);
        }
        let kind = match request.kind() {
            ferrum_interfaces::execution_cost::ActualWaveKind::Prefill => {
                VNextExecutionWaveKind::Prefill
            }
            ferrum_interfaces::execution_cost::ActualWaveKind::Decode => {
                VNextExecutionWaveKind::Decode
            }
            _ => {
                return Err(FerrumError::invalid_request(
                    "unsupported resource preparation phase",
                ))
            }
        };
        if request.participants() > self.policy.memory().maximum_active_sequences as usize
            || request.sequence_tokens() > self.maximum_model_tokens
            || request
                .participants()
                .checked_mul(request.tokens_per_sequence())
                .is_none_or(|tokens| {
                    tokens as u64 > self.policy.admission().maximum_scheduled_tokens
                })
        {
            return Ok(Outcome::Unavailable);
        }
        if self.sequences.lock().total_len() != 0 {
            return Err(FerrumError::invalid_request(
                "resource preparation requires an unused executor",
            ));
        }
        let before = WorkspaceCapacitySnapshot::capture(&self.plan_resources)?;
        let submissions = self.metrics.submitted_waves.load(Ordering::Acquire);
        let result = self.prepare_resource_shape(
            kind,
            request.participants(),
            request.sequence_tokens(),
            request.tokens_per_sequence(),
            None,
            budget,
        );
        if self.sequences.lock().total_len() != 0
            || self.metrics.submitted_waves.load(Ordering::Acquire) != submissions
        {
            return Err(FerrumError::internal(
                "resource preparation retained owners or submitted model work",
            ));
        }
        let after = WorkspaceCapacitySnapshot::capture(&self.plan_resources)?;
        tracing::info!(target: "ferrum::resource_readiness", ?request, ?before, ?after,
            failure = ?result.as_ref().err(),
            "Declared resource preparation without model execution");
        // Capacity failure is not a route proof and does not authorize actual
        // inference as a fallback. The caller must recapture its original gap.
        match result {
            Ok(_) => Ok(Outcome::Prepared),
            Err(FerrumError::ResourceExhausted { .. }) => Ok(Outcome::Unavailable),
            Err(error) => Err(error),
        }
    }

    async fn prepare_workspace_case(&self, case: &WorkspaceCase) -> Result<WorkspaceBucketReceipt> {
        let (step, invocation) = self.prepare_resource_shape(
            case.kind,
            case.sequences,
            case.tokens_per_sequence,
            case.tokens_per_sequence,
            Some(&case.bucket),
            &mut || true,
        )?;
        Ok(WorkspaceBucketReceipt {
            bucket: case.bucket.clone(),
            sequences: case.sequences,
            tokens_per_sequence: case.tokens_per_sequence,
            step,
            invocation,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn prepare_resource_shape(
        &self,
        kind: VNextExecutionWaveKind,
        participants: usize,
        sequence_tokens: usize,
        tokens_per_sequence: usize,
        expected_bucket: Option<&ReusableExecutionBucketId>,
        budget: &mut dyn ResourcePlanningBudget,
    ) -> Result<(BatchStepId, BatchInvocationId)> {
        // Actual pending-prefill owners provide legal token-span and Sequence
        // authority. No decode state or cache handle is fabricated: `kind` here
        // selects a resource class only, and this path never executes that wave.
        let tokens: Arc<[TokenId]> = vec![TokenId::new(0); sequence_tokens].into();
        let mut owners = Vec::with_capacity(participants);
        let mut sequences = BTreeMap::new();
        for _ in 0..participants {
            poll_resource_budget(budget)?;
            let mut owner = VNextStartupSequenceGuard::new(self);
            let request = self
                .reserve_resource_preparation_sequence(&mut owner, &tokens, tokens.len())?
                .ok_or_else(|| {
                    FerrumError::resource_exhausted(
                        "workspace startup admission could not fit the full declared bucket",
                    )
                })?;
            let sequence = {
                let registry = self.sequences.lock();
                let slot = registry.prefills.get(&request).ok_or_else(|| {
                    FerrumError::internal("workspace startup lost its actual prefill authority")
                })?;
                let state = slot.state.lock();
                match &*state {
                    VNextPrefillSlotState::Ready(sequence) => Arc::clone(sequence),
                    _ => {
                        return Err(FerrumError::internal(
                            "workspace startup admission is not Ready",
                        ))
                    }
                }
            };
            sequences.insert(sequence.session.sequence_authority(), sequence);
            owners.push(owner);
        }
        let batch = ExecutionBatchParticipants::new(
            sequences
                .values()
                .map(|sequence| Arc::clone(&sequence.session))
                .collect(),
        )
        .map_err(|error| FerrumError::backend(error.to_string()))?;
        let sequences = batch
            .sessions()
            .iter()
            .map(|session| {
                sequences
                    .remove(&session.sequence_authority())
                    .ok_or_else(|| {
                        FerrumError::internal("workspace startup canonical owner mapping changed")
                    })
            })
            .collect::<Result<Vec<_>>>()?;
        let ids = tokens.iter().map(|token| token.get()).collect::<Vec<_>>();
        let span = TokenSpanWork::from_token_ids(&ids, 0..tokens_per_sequence)
            .map_err(|error| FerrumError::backend(error.to_string()))?;
        let sequence_span = TokenSpanWork::from_token_ids(&ids, 0..ids.len())
            .map_err(|error| FerrumError::backend(error.to_string()))?;
        for sequence in &sequences {
            poll_resource_budget(budget)?;
            let work = ResourceWorkShape::single(sequence_span.clone())
                .map_err(|error| FerrumError::backend(error.to_string()))?;
            match self.extend_sequence_with_capacity(sequence, work)? {
                VNextExecutionCapacityDecision::Ready(()) => {}
                VNextExecutionCapacityDecision::Deferred(deferred) => {
                    return Err(FerrumError::resource_exhausted(format!(
                        "workspace Sequence backing deferred: {deferred:?}"
                    )))
                }
                VNextExecutionCapacityDecision::RequestStateDeferred(deferred) => {
                    return Err(FerrumError::resource_exhausted(format!(
                        "workspace Sequence Request-state deferred: {deferred:?}"
                    )))
                }
            }
        }
        poll_resource_budget(budget)?;
        let spans = vec![span; sequences.len()];
        let step =
            match self.begin_step_for_spans_with_capacity(&batch, &sequences, &spans, kind)? {
                VNextExecutionCapacityDecision::Ready(step) => step,
                VNextExecutionCapacityDecision::Deferred(deferred) => {
                    return Err(FerrumError::resource_exhausted(format!(
                        "workspace Step deferred: {deferred:?}"
                    )))
                }
                VNextExecutionCapacityDecision::RequestStateDeferred(deferred) => {
                    return Err(FerrumError::resource_exhausted(format!(
                        "workspace Step Request-state deferred: {deferred:?}"
                    )))
                }
            };
        if expected_bucket.is_some_and(|expected| {
            step.reusable_execution_bucket()
                .map(|bucket| bucket.bucket_id())
                != Some(expected)
        }) {
            return Err(self.abort_unsubmitted_step(
                step,
                FerrumError::internal(
                    "workspace startup first-fit differs from its declared bucket",
                ),
            ));
        }
        if let Err(error) = poll_resource_budget(budget) {
            return Err(self.abort_unsubmitted_step(step, error));
        }
        let wave = match self.prepare_wave_for_spans_with_capacity(&step, &sequences, &spans, kind)
        {
            Ok(VNextExecutionCapacityDecision::Ready(wave)) => wave,
            Ok(VNextExecutionCapacityDecision::Deferred(deferred)) => {
                return Err(self.abort_unsubmitted_step(
                    step,
                    FerrumError::resource_exhausted(format!(
                        "workspace Invocation deferred: {deferred:?}"
                    )),
                ))
            }
            Ok(VNextExecutionCapacityDecision::RequestStateDeferred(deferred)) => {
                return Err(self.abort_unsubmitted_step(
                    step,
                    FerrumError::resource_exhausted(format!(
                        "workspace Invocation Request-state deferred: {deferred:?}"
                    )),
                ))
            }
            Err(error) => return Err(self.abort_unsubmitted_step(step, error)),
        };
        let (step, invocation) = finish_resource_only_wave(wave, step)?;
        drop(batch);
        drop(sequences);
        for owner in owners {
            owner.complete()?;
        }
        Ok((step, invocation))
    }
}

#[cfg(test)]
mod tests;

#[cfg(all(test, feature = "cuda"))]
mod cuda_tests;
