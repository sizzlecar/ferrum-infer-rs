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

    async fn prepare_workspace_case(&self, case: &WorkspaceCase) -> Result<WorkspaceBucketReceipt> {
        // Actual pending-prefill owners provide legal token-span and Sequence
        // authority. No decode state or cache handle is fabricated: `kind` here
        // selects a resource class only, and this path never executes that wave.
        let tokens: Arc<[TokenId]> = vec![TokenId::new(0); case.tokens_per_sequence].into();
        let mut owners = Vec::with_capacity(case.sequences);
        let mut sequences = BTreeMap::new();
        for _ in 0..case.sequences {
            let mut owner = VNextStartupSequenceGuard::new(self);
            let request = self
                .reserve_startup_sequence(&mut owner, &tokens, tokens.len())
                .await?
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
        let span = TokenSpanWork::from_token_ids(&ids, 0..ids.len())
            .map_err(|error| FerrumError::backend(error.to_string()))?;
        for sequence in &sequences {
            let work = ResourceWorkShape::single(span.clone())
                .map_err(|error| FerrumError::backend(error.to_string()))?;
            match self.extend_sequence_with_capacity(sequence, work)? {
                VNextExecutionCapacityDecision::Ready(()) => {}
                _ => return Err(FerrumError::resource_exhausted("workspace startup Sequence backing remains deferred within the original maintenance/capacity limits")),
            }
        }
        let spans = vec![span; sequences.len()];
        let step = match self.begin_step_for_spans_with_capacity(&batch, &sequences, &spans, case.kind)? {
            VNextExecutionCapacityDecision::Ready(step) => step,
            _ => return Err(FerrumError::resource_exhausted("workspace startup Step remains deferred within the original maintenance/capacity limits")),
        };
        if step
            .reusable_execution_bucket()
            .map(|bucket| bucket.bucket_id())
            != Some(&case.bucket)
        {
            return Err(self.abort_unsubmitted_step(
                step,
                FerrumError::internal(
                    "workspace startup first-fit differs from its declared bucket",
                ),
            ));
        }
        let wave = match self.prepare_wave_for_spans_with_capacity(&step, &sequences, &spans, case.kind) {
            Ok(VNextExecutionCapacityDecision::Ready(wave)) => wave,
            Ok(_) => return Err(self.abort_unsubmitted_step(step, FerrumError::resource_exhausted("workspace startup full wave remains deferred within the original maintenance/capacity limits"))),
            Err(error) => return Err(self.abort_unsubmitted_step(step, error)),
        };
        let (step, invocation) = finish_resource_only_wave(wave, step)?;
        drop(batch);
        drop(sequences);
        for owner in owners {
            owner.complete()?;
        }
        Ok(WorkspaceBucketReceipt {
            bucket: case.bucket.clone(),
            sequences: case.sequences,
            tokens_per_sequence: case.tokens_per_sequence,
            step,
            invocation,
        })
    }
}

#[cfg(test)]
mod tests;

#[cfg(all(test, feature = "cuda"))]
mod cuda_tests;
