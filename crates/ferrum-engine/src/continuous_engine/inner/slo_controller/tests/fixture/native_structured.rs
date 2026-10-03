//! Actual core submission for the opt-in controlled CPU fill. The data used
//! for returned logits is produced inside TestRuntime::submit after its guard.
use super::*;

struct HostGuard<'a> {
    host: &'a dyn NonblockingHostSubmissionGuard,
    rows: usize,
    tokens: u64,
    full_logits: bool,
    layout: contract::ControlledCpuFillLayout,
}
impl vnext::PreparedWaveSubmissionGuard for HostGuard<'_> {
    fn check(
        &self,
        actual: &vnext::DeviceSubmissionAttribution,
        readback: CoreReadbackRoute,
    ) -> std::result::Result<(), GuardedNotSubmittedReason> {
        let native = self.layout.native_op_id(self.full_logits);
        if actual.commands().len() != 2
            || readback != CoreReadbackRoute::HostSynchronized
            || actual
                .commands()
                .iter()
                .enumerate()
                .any(|(index, command)| {
                    command.command_index() != index as u32
                        || command.node_index() != Some(index as u32)
                        || command.command_phase() != vnext::DeviceCommandPhase::Compute
                        || command.execution_path() != vnext::DeviceExecutionPath::Eager
                        || command.participant_start() != 0
                        || command.batching_form() != vnext::DeviceBatchingForm::Packed
                        || command.reusable_graph_node_count().is_some()
                        || command.native_op_id() != native
                        || command.participant_count() as usize != self.rows
                        || command.token_count() != self.tokens
                        || command.compute_dispatch_count()
                            != self.layout.compute_dispatch_count(self.rows)
                        || command.transfer_command_count() != 0
                })
        {
            eprintln!(
                "CPU fill native guard mismatch: expected rows={} tokens={} native={} readback=HostSynchronized; actual readback={readback:?}, attribution={actual:?}",
                self.rows, self.tokens, native
            );
            return Err(GuardedNotSubmittedReason::ResourceClaimMismatch);
        }
        self.host
            .check()
            .map_err(GuardedNotSubmittedReason::HostRejected)
    }
}

struct InstalledFill<'a> {
    fixture: &'a contract::Fixture,
}
impl Drop for InstalledFill<'_> {
    fn drop(&mut self) {
        self.fixture
            .provider_trace
            .lock()
            .unwrap()
            .controlled_cpu_fill = None;
        self.fixture
            .runtime_trace
            .lock()
            .unwrap()
            .controlled_cpu_fill = None;
    }
}

// Scoped controlled-stream facts; ordinary waves retain their original route.
struct PrefixOutsideStream<'a> {
    fixture: &'a contract::Fixture,
    before: Option<vnext::DeviceCostGraphStreamState>,
    evidence: Option<vnext::DeviceSubmissionGraphEvidence>,
}
impl Drop for PrefixOutsideStream<'_> {
    fn drop(&mut self) {
        let mut trace = self.fixture.runtime_trace.lock().unwrap();
        trace.cost_graph_state = self.before;
        trace.submitted_graph_evidence = self.evidence;
    }
}

impl ControlledExecutor {
    pub(super) fn cpu_fill_layout(
        &self,
        decode_contexts: impl IntoIterator<Item = usize>,
    ) -> contract::ControlledCpuFillLayout {
        if self.row_selected_cpu_fill.load(Ordering::Acquire) {
            contract::ControlledCpuFillLayout::per_row(decode_contexts)
        } else if self.context_partitioned_cpu_fill.load(Ordering::Acquire) {
            contract::ControlledCpuFillLayout::for_contexts(decode_contexts)
        } else {
            contract::ControlledCpuFillLayout::Uniform
        }
    }

    /// Original logical request authority, which can be smaller than the
    /// mock model capability or the separately declared numerical domain.
    /// Reading this immutable admission ceiling grants no additional capacity.
    pub(in crate::continuous_engine::inner) fn native_request_fit_tokens(&self) -> Option<u64> {
        self.evidence
            .sessions
            .snapshot()
            .iter()
            .map(|session| {
                session
                    .resources()
                    .request_resources()
                    .work_shape()
                    .fit_tokens()
            })
            .min()
    }

    pub(in crate::continuous_engine::inner) fn native_structured_counts(&self) -> (u64, usize) {
        let trace = self
            .evidence
            .fixture
            .as_ref()
            .unwrap()
            .runtime_trace
            .lock()
            .unwrap();
        let commands = trace
            .submitted_commands
            .iter()
            .flatten()
            .filter(|c| matches!(c, contract::TestCommand::ControlledCpuFill { .. }))
            .count();
        (trace.submit_calls, commands)
    }
    pub(in crate::continuous_engine::inner) fn native_structured_mixed_row_commands(
        &self,
    ) -> usize {
        self.evidence
            .fixture
            .as_ref()
            .unwrap()
            .runtime_trace
            .lock()
            .unwrap()
            .submitted_commands
            .iter()
            .flatten()
            .filter(|command| match command {
                contract::TestCommand::ControlledCpuFill {
                    layout: contract::ControlledCpuFillLayout::PerRow { packed_rows },
                    participants,
                    ..
                } => {
                    let count = packed_rows.count_ones();
                    count > 0 && count < *participants
                }
                _ => false,
            })
            .count()
    }
    pub(super) fn native_structured_output(
        &self,
        context: GuardedCostObservation<'_, '_>,
        prefills: &[PlanRuntimePrefillInput],
        decodes: &[PlanRuntimeDecodeInput],
        guard: &dyn NonblockingHostSubmissionGuard,
    ) -> Option<Result<Vec<Vec<f32>>>> {
        if !self.native_structured_submission.load(Ordering::Acquire) {
            return None;
        }
        Some(self.submit_cpu_fill(context, prefills, decodes, guard))
    }

    fn submit_cpu_fill(
        &self,
        mut context: GuardedCostObservation<'_, '_>,
        prefills: &[PlanRuntimePrefillInput],
        decodes: &[PlanRuntimeDecodeInput],
        guard: &dyn NonblockingHostSubmissionGuard,
    ) -> Result<Vec<Vec<f32>>> {
        let fail = |e: &dyn std::fmt::Display| FerrumError::backend(e.to_string());
        if self.single_row_prefill_only.load(Ordering::Acquire) && prefills.len() > 1 {
            return Err(FerrumError::unsupported(
                "CPU route supports only single-row prefill",
            ));
        }
        assert!(
            prefills.is_empty() != decodes.is_empty(),
            "CPU fixture uses split waves"
        );
        assert!(
            self.evidence.prefix.is_some() || prefills.iter().all(|input| input.chunk.is_final())
        );
        let full = !prefills.is_empty()
            || decodes
                .iter()
                .any(|input| input.logits_policy.requires_full_logits());
        let ids = prefills
            .iter()
            .map(|r| &r.request_id)
            .chain(decodes.iter().map(|r| &r.request_id))
            .collect::<Vec<_>>();
        let bindings = self.session_bindings.lock();
        let sessions = ids
            .iter()
            .map(|id| {
                self.evidence
                    .sessions
                    .get(
                        bindings
                            .iter()
                            .position(|old| old == *id)
                            .expect("actual admission bound CPU request"),
                    )
                    .unwrap()
            })
            .collect::<Vec<_>>();
        drop(bindings);
        let mut histories = self.native_structured_history.lock();
        let mut next_histories = Vec::new();
        let mut work = Vec::new();
        for input in prefills {
            let tokens = input
                .input_tokens
                .iter()
                .map(|t| t.get())
                .collect::<Vec<_>>();
            let span = vnext::TokenSpanWork::from_token_ids(
                &tokens,
                input.chunk.tokens_processed()..input.chunk.end(),
            )
            .map_err(|e| fail(&e))?;
            let span = if self.evidence.prefix.is_some() {
                span.with_checkpoint_tokens(Arc::from(tokens.clone()))
                    .map_err(|e| fail(&e))?
            } else {
                span
            };
            work.push(span);
            next_histories.push((input.request_id.clone(), tokens));
        }
        for input in decodes {
            let mut tokens = histories
                .get(&input.request_id)
                .expect("actual prior prefill")
                .clone();
            assert_eq!(tokens.len(), input.kv_cache.num_tokens());
            let start = tokens.len();
            tokens.push(input.input_token.get());
            let span = vnext::TokenSpanWork::from_token_ids(&tokens, start..tokens.len())
                .map_err(|e| fail(&e))?;
            work.push(if self.evidence.prefix.is_some() {
                span.with_checkpoint_tokens(Arc::from(tokens.clone()))
                    .map_err(|e| fail(&e))?
            } else {
                span
            });
            next_histories.push((input.request_id.clone(), tokens));
        }
        drop(histories);
        if self.evidence.prefix.is_some() {
            // Match production's selected-wave sequence extension: original
            // admission covers the prompt, not every future decode frontier.
            // Fixed checkpoint state needs no new physical bytes, but still
            // needs the typed backing publication before opening its Step.
            for (session, (_, tokens)) in sessions.iter().zip(&next_histories) {
                let span = vnext::TokenSpanWork::from_token_ids(tokens, 0..tokens.len())
                    .map_err(|e| fail(&e))?;
                let request = vnext::SequenceResourceExtensionRequest::new(
                    vnext::ResourceWorkShape::single(span).map_err(|e| fail(&e))?,
                    vnext::AdmissionPressureAction::WaitForRelease,
                )
                .map_err(|e| fail(&e))?;
                match session
                    .try_ensure_backing_covers(request)
                    .map_err(|e| fail(&e))?
                {
                    vnext::SequenceResourceExtensionDecision::Current(_)
                    | vnext::SequenceResourceExtensionDecision::Extended(_) => {}
                    _ => {
                        return Err(FerrumError::backend(
                            "CPU checkpoint fixture selected extension is not resident",
                        ))
                    }
                }
            }
        }
        let batch =
            vnext::ExecutionBatchParticipants::new(sessions.clone()).map_err(|e| fail(&e))?;
        let lane = self.evidence.lane.as_ref().unwrap();
        let request = vnext::StepResourceAdmissionRequest::new(
            batch.bind_work_shape(work.clone()).map_err(|e| fail(&e))?,
            vnext::AdmissionFitPolicy::ImmediateOnly,
            vnext::AdmissionPressureAction::WaitForRelease,
        )
        .map_err(|e| fail(&e))?;
        let mut selected = None;
        for _ in 0..4 {
            match batch
                .try_begin_step(request.clone(), lane)
                .map_err(|e| fail(&e))?
            {
                vnext::StepResourceAdmissionDecision::Admitted(step) => {
                    selected = Some(step);
                    break;
                }
                vnext::StepResourceAdmissionDecision::BackingDeferred(deferred) => {
                    deferred.maintain().map_err(|e| fail(&e))?;
                }
                _ => {
                    return Err(FerrumError::backend(
                        "CPU fixture original Step unavailable",
                    ))
                }
            }
        }
        let step =
            selected.ok_or_else(|| FerrumError::backend("CPU fixture Step budget exhausted"))?;
        let fixture = self.evidence.fixture.as_ref().unwrap();
        let requests = fixture
            .resolved
            .execution_plan()
            .payload()
            .nodes()
            .iter()
            .map(|node| {
                vnext::InvocationResourceAdmissionRequest::for_all_step_participants(
                    node.id().clone(),
                    step.bind_all_invocation_work_shape(work.clone()).unwrap(),
                    vnext::AdmissionFitPolicy::ImmediateOnly,
                    vnext::AdmissionPressureAction::WaitForRelease,
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        let mut selected = None;
        for _ in 0..4 {
            let prepared = if self.evidence.prefix.is_some() {
                step.try_prepare_full_plan_submission_wave(
                    Arc::new(step.work_shape().clone()),
                    vnext::AdmissionFitPolicy::ImmediateOnly,
                    vnext::AdmissionPressureAction::WaitForRelease,
                )
            } else {
                step.try_prepare_submission_wave(requests.clone())
            };
            match prepared.map_err(|e| fail(&e))? {
                vnext::StepSubmissionWaveAdmissionDecision::Prepared(wave) => {
                    selected = Some(wave);
                    break;
                }
                vnext::StepSubmissionWaveAdmissionDecision::BackingDeferred(deferred) => {
                    deferred.maintain().map_err(|e| fail(&e))?;
                }
                _ => {
                    return Err(FerrumError::backend(
                        "CPU fixture original invocation unavailable",
                    ))
                }
            }
        }
        let wave =
            selected.ok_or_else(|| FerrumError::backend("CPU fixture wave budget exhausted"))?;
        let active = sessions
            .iter()
            .map(|session| vnext::TrustedActiveSequenceBinding::from_session(session).unwrap())
            .collect::<Vec<_>>();
        let identity = vnext::OperationDispatch::bind_submission_wave_identity(
            &fixture.resolved,
            active.iter(),
            &wave,
            lane,
        )
        .map_err(|e| fail(&e))?;
        let layout = self.cpu_fill_layout(decodes.iter().map(|row| row.kv_cache.num_tokens()));
        let fill = contract::ControlledCpuFill::with_layout(
            ids.len(),
            self.info().vocab_size,
            full,
            layout,
        );
        assert_eq!(fixture.resolved.execution_plan().payload().nodes().len(), 2);
        fixture.provider_trace.lock().unwrap().controlled_cpu_fill = Some(fill.clone());
        fixture.runtime_trace.lock().unwrap().controlled_cpu_fill = Some(fill.clone());
        let _installed = InstalledFill { fixture };
        let providers = fixture
            .registry
            .bind_plan(&fixture.resolved)
            .map_err(|e| fail(&e))?;
        let provider_identities = structured::provider_identities(providers.providers());
        let outside_preparation = self
            .native_prefix_preparation_outside
            .load(Ordering::Acquire)
            && context.as_deref().is_some_and(|context| {
                ids.iter().all(|id| {
                    context
                        .participant(id)
                        .and_then(|p| p.host_features)
                        .is_some_and(|host| host.policy.empirical_content_domain.is_none())
                })
            });
        let _outside_stream = outside_preparation.then(|| {
            let mut trace = fixture.runtime_trace.lock().unwrap();
            let prior = PrefixOutsideStream {
                fixture,
                before: trace.cost_graph_state,
                evidence: trace.submitted_graph_evidence,
            };
            let state = vnext::DeviceCostGraphStreamState::new(
                vnext::DeviceCostGraphConfiguration::OnDemand,
                0,
                0,
                0,
            )
            .unwrap();
            trace.cost_graph_state = Some(state);
            trace.submitted_graph_evidence = Some(
                vnext::DeviceSubmissionGraphEvidence::new(state, state, false, 0, 0, 0, 0, 0)
                    .unwrap(),
            );
            prior
        });
        let mut outside_rows = None;
        let mut outside_started = None;
        if let Some(context) = context.as_deref_mut() {
            if outside_preparation {
                let catalog = lane
                    .reusable_execution_catalog()
                    .map_err(|e| fail(&e))?
                    .into_index()
                    .map_err(|e| fail(&e))?;
                let selection = vnext::OperationDispatch::select_reusable_execution_for_cost(
                    providers.providers(),
                    &fixture.resolved,
                    &wave,
                    lane,
                    Some(&catalog),
                    true,
                )
                .map_err(|e| fail(&e))?;
                let (program, route) = selection.into_parts();
                assert!(program.is_none());
                assert!(
                    route.class().is_outside(),
                    "original fixture selector must prove outside"
                );
                let rows: Vec<ActualWaveRow> = prefills
                    .iter()
                    .map(|input| {
                        let host = context.participant(&input.request_id).unwrap();
                        ActualWaveRow {
                            request_id: host.request_id.clone(),
                            owner_incarnation: host.owner_incarnation,
                            work_generation: host.work_generation,
                            input_index: host.input_index,
                            work: ActualRowWork::Prefill {
                                offset: input.chunk.tokens_processed().try_into().unwrap(),
                                count: input.chunk.tokens_to_process().try_into().unwrap(),
                                total_prompt_tokens: input
                                    .chunk
                                    .total_prompt_tokens()
                                    .try_into()
                                    .unwrap(),
                            },
                        }
                    })
                    .chain(decodes.iter().map(|input| {
                        let host = context.participant(&input.request_id).unwrap();
                        ActualWaveRow {
                            request_id: host.request_id.clone(),
                            owner_incarnation: host.owner_incarnation,
                            work_generation: host.work_generation,
                            input_index: host.input_index,
                            work: ActualRowWork::Decode {
                                kv_tokens: input.kv_cache.num_tokens().try_into().unwrap(),
                            },
                        }
                    }))
                    .collect();
                context.prepared_route(route, Ok(rows.clone()));
                outside_rows = Some(rows);
                outside_started = context.now_ns();
            } else {
                context.prepared_route(
                    vnext::OperationDispatch::observe_non_reusable_cost_route(lane),
                    Ok(Vec::new()),
                );
                structured::record_native_with_layout(
                    context,
                    prefills,
                    decodes,
                    &provider_identities,
                    layout,
                    structured::recurrent_state_bytes(self, ids.len()),
                );
            }
        }
        let tokens = prefills
            .iter()
            .map(|r| r.chunk.tokens_to_process() as u64)
            .sum::<u64>()
            + decodes.len() as u64;
        let reaper = self
            .evidence
            .prefix
            .as_ref()
            .map(|prefix| prefix.reaper.clone())
            .unwrap_or_else(vnext::CompletionReaper::new);
        let result = vnext::OperationDispatch::encode_and_submit_guarded_wave(
            providers.providers(),
            &fixture.resolved,
            &identity,
            active.iter(),
            &[],
            &HostGuard {
                host: guard,
                rows: ids.len(),
                tokens,
                full_logits: full,
                layout,
            },
            wave,
            lane,
            &reaper,
        );
        let submitted = match result {
            vnext::GuardedWaveSubmissionOutcome::Dispatch(result) => {
                result.map_err(|e| fail(&e))?
            }
            vnext::GuardedWaveSubmissionOutcome::NotSubmitted(rejection) => {
                let rejection = rejection.reconcile_step(step).map_err(|(e, _)| fail(&e))?;
                return Err(FerrumError::backend(format!(
                    "CPU fill rejected before actual submit: {:?}; rows={} tokens={tokens} full_logits={full} layout={layout:?}",
                    rejection.reason(), ids.len()
                )));
            }
        };
        let (handle, attribution) = submitted.into_parts();
        self.physical.fetch_add(1, Ordering::AcqRel);
        if outside_preparation {
            self.native_prefix_preparation_outside_submissions
                .fetch_add(1, Ordering::AcqRel);
        }
        if let Some(context) = context.as_deref_mut() {
            if let Some(rows) = outside_rows {
                let pending = structured::private_outside_pending(
                    context,
                    rows,
                    attribution.as_ref().expect("actual inference attribution"),
                    &provider_identities,
                );
                context.physical_wave_pending(Ok(pending), outside_started);
            }
            context.route_submission(attribution.as_ref());
        }
        assert!(matches!(
            handle.wait().map_err(|e| fail(&e))?,
            vnext::CompletionObservation::Terminal(_)
        ));
        assert_eq!(fill.executed.load(Ordering::Acquire), 2);
        assert_eq!(
            fill.executed_dispatches.load(Ordering::Acquire),
            2 * layout.compute_dispatch_count(ids.len()) as usize
        );
        let output = fill.take_output();
        let mut histories = self.native_structured_history.lock();
        for (id, tokens) in next_histories {
            histories.insert(id, tokens);
        }
        // A different live owner may be absent from this wave. Retire its
        // history only through actual successful complete_cache below.
        assert!(histories.len() <= self.evidence.sessions.len());
        drop(histories);
        drop(attribution);
        drop(handle);
        drop(identity);
        drop(active);
        drop(providers);
        step.try_retire_normal()
            .map_err(|_| FerrumError::backend("CPU fixture Step retirement failed"))?;
        Ok(output)
    }

    pub(super) fn prefill_output_from_logits(
        &self,
        input: &PlanRuntimePrefillInput,
        logits: Vec<f32>,
    ) -> Result<PlanRuntimePrefillCompletion> {
        assert_eq!(logits.len(), self.info().vocab_size);
        let output = if input.chunk.is_final() {
            PlanRuntimePrefillOutput::final_logits(
                input.request_id.clone(),
                input.chunk.end(),
                logits,
                self.cache(&input.request_id, input.chunk.end()),
            )?
        } else {
            assert!(self.evidence.prefix.is_some());
            PlanRuntimePrefillOutput::intermediate(
                input.request_id.clone(),
                input.chunk.end(),
                self.cache(&input.request_id, input.chunk.end()),
            )
        };
        PlanRuntimePrefillCompletion::new(output, input.chunk, input.chunk, 0)
    }
    pub(super) fn decode_output_from_logits(
        &self,
        input: &PlanRuntimeDecodeInput,
        values: Vec<f32>,
    ) -> PlanRuntimeDecodeOutput {
        let output = if input.logits_policy.requires_full_logits() {
            assert_eq!(values.len(), self.info().vocab_size);
            ExecutorSamplingOutput::FullLogits(values)
        } else {
            // A mixed pending cohort executes the FullLogits product for the
            // whole wave. Clean greedy rows still consume its real argmax.
            let selected = if values.len() == 1 {
                values[0] as u32
            } else {
                assert_eq!(values.len(), self.info().vocab_size);
                values
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .unwrap()
                    .0 as u32
            };
            ExecutorSamplingOutput::GreedyToken(ferrum_types::TokenId::new(selected))
        };
        PlanRuntimeDecodeOutput::new(
            output,
            self.cache(&input.request_id, input.kv_cache.num_tokens() + 1),
        )
    }
}
