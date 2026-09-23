//! CUDA native submission and the final optional cost guard.
use super::*;

impl CudaDeviceRuntime {
    pub(super) fn submit_with_timing_and_guard<S>(
        &self,
        stream: &mut CudaDeviceStream,
        commands: DeviceCommandBatch<CudaDeviceCommand>,
        timing_sink: &S,
        guard: Option<&dyn DeviceSubmissionGuard>,
    ) -> Result<CudaDeviceFence, GuardedDeviceSubmissionError<CudaDeviceRuntimeError>>
    where
        S: DeviceSubmissionTimingSink,
    {
        let validate_stage =
            CudaSubmissionStageTimer::start(timing_sink, DeviceSubmissionStage::ValidateAndPrepare);
        if let Err(error) = self.validate_stream(stream) {
            return Err(GuardedDeviceSubmissionError::Device(
                DefinitelyNotSubmitted::new(error),
            ));
        }
        if commands.is_empty() {
            return Err(GuardedDeviceSubmissionError::Device(
                DefinitelyNotSubmitted::new(CudaDeviceRuntimeError::contract(
                    "CUDA command batch is empty",
                )),
            ));
        }
        let timing_mode = commands.timing_mode();
        let compute_path_requirement = commands.compute_path_requirement();
        let declared_eager_compute_node_indices = commands
            .declared_eager_compute_node_indices()
            .iter()
            .copied()
            .collect::<BTreeSet<_>>();
        let declared_eager_compute_node_count =
            commands.declared_eager_compute_node_indices().len();
        let logical_attribution = commands
            .attribution_requirement()
            .logical_execution_path_required();
        let reusable_execution_capture = commands.reusable_execution_capture().cloned();
        let graph_before = stream.executable_cache.cost_graph_stream_state();
        if guard.is_some()
            && (timing_mode != DeviceTimingMode::Off
                || !logical_attribution
                || reusable_execution_capture.is_some()
                || !stream.state.is_quiescent()
                || !graph_before.is_some_and(|state| state.is_unconfigured_empty()))
        {
            // Reject before graph preparation could enqueue any work.
            return Err(GuardedDeviceSubmissionError::Rejected(
                ferrum_interfaces::execution_cost::GuardedNotSubmittedReason::AttributionUnavailable,
            ));
        }
        let entries = commands
            .into_entries()
            .into_iter()
            .map(|entry| {
                let (phase, node_index, logical_work, command) = entry.into_parts();
                let command = match logical_work {
                    Some(logical_work) => command.bind_core_logical_work(logical_work)?,
                    None => command,
                };
                Ok((phase, node_index, command))
            })
            .collect::<Result<Vec<_>, CudaDeviceRuntimeError>>()
            .map_err(DefinitelyNotSubmitted::new)?;
        let declaration_shape_matches = match compute_path_requirement {
            DeviceComputePathRequirement::ReplayedWithDeclaredEagerBoundaries => {
                !declared_eager_compute_node_indices.is_empty()
                    && declared_eager_compute_node_indices.len()
                        == declared_eager_compute_node_count
            }
            _ => declared_eager_compute_node_indices.is_empty(),
        };
        let mut compute_command_count = 0_usize;
        let mut direct_compute_command_count = 0_usize;
        let mut observed_eager_compute_node_indices = BTreeSet::new();
        let mut exact_boundary_shape = true;
        for (phase, node_index, command) in &entries {
            if *phase != DeviceCommandPhase::Compute {
                continue;
            }
            compute_command_count += 1;
            if let Some(invocation) = command.reusable_execution_invocation() {
                direct_compute_command_count += 1;
                if declared_eager_compute_node_indices
                    .iter()
                    .any(|node_index| invocation.segment().contains_node(*node_index))
                {
                    exact_boundary_shape = false;
                }
            } else if compute_path_requirement
                == DeviceComputePathRequirement::ReplayedWithDeclaredEagerBoundaries
            {
                exact_boundary_shape &= node_index.is_some_and(|node_index| {
                    declared_eager_compute_node_indices.contains(&node_index)
                        && observed_eager_compute_node_indices.insert(node_index)
                });
            }
        }
        let command_count = u32::try_from(entries.len()).map_err(|_| {
            DefinitelyNotSubmitted::new(CudaDeviceRuntimeError::contract(
                "CUDA command count exceeds u32",
            ))
        })?;
        let command_phases = entries
            .iter()
            .map(|(phase, _, _)| *phase)
            .collect::<Vec<_>>();
        let command_node_indices = (timing_mode.kernel_attribution_enabled()
            || logical_attribution
            || reusable_execution_capture.is_some())
        .then(|| {
            entries
                .iter()
                .map(|(_, node_index, _)| *node_index)
                .collect::<Vec<_>>()
        });
        let commands = entries
            .into_iter()
            .map(|(_, _, command)| command)
            .collect::<Vec<_>>();
        let physical_span_attribution = timing_mode.physical_span_attribution_enabled();
        let kernel_attribution = timing_mode.kernel_attribution_enabled();
        let native_attribution = kernel_attribution || logical_attribution;
        if kernel_attribution {
            vnext_tool_correlation::prepare();
        }
        let mut execution_paths =
            native_attribution.then(|| vec![DeviceExecutionPath::Eager; commands.len()]);
        let mut reusable_graph_node_counts = native_attribution.then(|| vec![None; commands.len()]);
        if commands
            .iter()
            .any(|command| command.runtime_instance != self.runtime_instance)
        {
            return Err(GuardedDeviceSubmissionError::Device(
                DefinitelyNotSubmitted::new(CudaDeviceRuntimeError::contract(
                    "CUDA command batch contains work from another runtime instance",
                )),
            ));
        }
        let contains_direct_execution = direct_compute_command_count != 0;
        let compute_path_matches = match compute_path_requirement {
            DeviceComputePathRequirement::Adaptive => true,
            DeviceComputePathRequirement::EagerOnly => direct_compute_command_count == 0,
            DeviceComputePathRequirement::ReplayedOnly => {
                compute_command_count > 0 && direct_compute_command_count == compute_command_count
            }
            DeviceComputePathRequirement::ReplayedWithDeclaredEagerBoundaries => {
                declaration_shape_matches
                    && exact_boundary_shape
                    && direct_compute_command_count > 0
                    && direct_compute_command_count < compute_command_count
                    && observed_eager_compute_node_indices == declared_eager_compute_node_indices
            }
        };
        if !compute_path_matches {
            return Err(GuardedDeviceSubmissionError::Device(
                DefinitelyNotSubmitted::new(CudaDeviceRuntimeError::contract(
                    "CUDA compute commands do not satisfy the required execution path",
                )),
            ));
        }
        if reusable_execution_capture.is_some() && contains_direct_execution {
            return Err(GuardedDeviceSubmissionError::Device(
                DefinitelyNotSubmitted::new(CudaDeviceRuntimeError::contract(
                    "CUDA reusable execution capture cannot contain direct program references",
                )),
            ));
        }
        if kernel_attribution && contains_direct_execution {
            return Err(GuardedDeviceSubmissionError::Device(
                DefinitelyNotSubmitted::new(CudaDeviceRuntimeError::contract(
                    "CUDA kernel attribution requires full logical command encoding",
                )),
            ));
        }
        if let (Some(command_node_indices), Some(execution_paths)) =
            (&command_node_indices, &execution_paths)
        {
            if let Err(error) = cuda_submission_attribution(
                &command_phases,
                command_node_indices,
                &commands,
                execution_paths,
                reusable_graph_node_counts.as_deref(),
                Vec::new(),
            ) {
                return Err(GuardedDeviceSubmissionError::Device(
                    DefinitelyNotSubmitted::new(error),
                ));
            }
        }
        if let Err(error) = self.context.bind_to_thread() {
            return Err(GuardedDeviceSubmissionError::Device(
                DefinitelyNotSubmitted::new(CudaDeviceRuntimeError::driver(
                    "submission context binding",
                    error,
                )),
            ));
        }
        for invocation in commands
            .iter()
            .filter_map(CudaDeviceCommand::reusable_execution_invocation)
        {
            let resident = if logical_attribution {
                stream
                    .executable_cache
                    .contains_attributable_program_segment(invocation)
            } else {
                stream.executable_cache.contains_program_segment(invocation)
            };
            match resident {
                Ok(true) => {}
                Ok(false) => {
                    return Err(GuardedDeviceSubmissionError::Device(
                        DefinitelyNotSubmitted::new(CudaDeviceRuntimeError::contract(
                            if logical_attribution {
                                "CUDA direct reusable execution lacks sealed logical attribution"
                            } else {
                                "CUDA direct reusable execution is not resident in the sealed catalog"
                            },
                        )),
                    ))
                }
                Err(error) => {
                    return Err(GuardedDeviceSubmissionError::Device(
                        DefinitelyNotSubmitted::new(CudaDeviceRuntimeError::contract(
                            error.to_string(),
                        )),
                    ))
                }
            }
        }
        let executable_candidates = match compute_path_requirement {
            DeviceComputePathRequirement::Adaptive => {
                let eager_boundary_node_indices = reusable_execution_capture
                    .as_ref()
                    .map(DeviceReusableExecutionCapture::eager_boundary_node_indices)
                    .unwrap_or_default();
                match cuda_executable_candidates(
                    &command_phases,
                    &commands,
                    command_node_indices.as_deref(),
                    eager_boundary_node_indices,
                ) {
                    Ok(candidates) => candidates,
                    Err(error) => {
                        return Err(GuardedDeviceSubmissionError::Device(
                            DefinitelyNotSubmitted::new(error),
                        ))
                    }
                }
            }
            DeviceComputePathRequirement::EagerOnly
            | DeviceComputePathRequirement::ReplayedOnly
            | DeviceComputePathRequirement::ReplayedWithDeclaredEagerBoundaries => Vec::new(),
        };
        let capture_allowed = stream.state.is_quiescent();
        if let Err(error) = stream.state.begin_submission() {
            return Err(GuardedDeviceSubmissionError::Device(
                DefinitelyNotSubmitted::new(error),
            ));
        }
        let mut replay_observation = DeviceReusableExecutionObservation::default();
        if S::ENABLED {
            for _ in &executable_candidates {
                replay_observation.observe_candidate_segment();
            }
        }
        let preparation = match stream.executable_cache.prepare_all(
            &self.context,
            &stream.stream,
            &stream.blas,
            &commands,
            &executable_candidates,
            capture_allowed,
        ) {
            Ok(preparation) => preparation,
            Err(error) => {
                stream.state.fail();
                self.quarantine(stream, commands);
                panic!(
                    "CUDA submission became indeterminate while preparing reusable executables: {error}"
                );
            }
        };
        if S::ENABLED {
            for _ in 0..preparation.captured_segments() {
                replay_observation.observe_captured_segment();
            }
            for _ in 0..preparation.uploaded_segments() {
                replay_observation.observe_uploaded_segment();
            }
            for _ in 0..preparation.cache_hit_segments() {
                replay_observation.observe_cache_hit_segment();
            }
            for _ in 0..preparation.cached_rejected_segments() {
                replay_observation.observe_cached_rejected_segment();
            }
            for _ in 0..preparation.capture_rejected_segments() {
                replay_observation.observe_capture_rejection();
            }
            for _ in 0..preparation.quiescence_deferred_segments() {
                replay_observation.observe_quiescence_deferred_segment();
            }
            for _ in 0..preparation.warmup_required_segments() {
                replay_observation.observe_warmup_required_segment();
            }
            for _ in 0..preparation.capacity_deferred_segments() {
                replay_observation.observe_capacity_deferred_segment();
            }
            for _ in 0..preparation.outside_preparation_segments() {
                replay_observation.observe_outside_preparation_segment();
            }
            for _ in 0..preparation.evicted_segments() {
                replay_observation.observe_evicted_segment();
            }
        }
        if let Some(capture) = reusable_execution_capture.as_ref() {
            let command_node_indices = command_node_indices
                .as_deref()
                .expect("reusable execution capture retained node attribution");
            if let Err(error) = stream.executable_cache.register_program(
                capture,
                &executable_candidates,
                &command_phases,
                command_node_indices,
                &commands,
                &preparation,
            ) {
                stream.state.fail();
                self.quarantine(stream, commands);
                panic!(
                    "CUDA submission became indeterminate while registering a reusable program: {error}"
                );
            }
        }
        let graph_after_preparation = stream.executable_cache.cost_graph_stream_state();
        let graph_evidence = |replayed_segments| {
            DeviceSubmissionGraphEvidence::new(
                graph_before?,
                graph_after_preparation?,
                reusable_execution_capture.is_some(),
                executable_candidates.len().try_into().ok()?,
                preparation.captured_segments().try_into().ok()?,
                preparation.capture_rejected_segments().try_into().ok()?,
                preparation.uploaded_segments().try_into().ok()?,
                replayed_segments,
            )
        };
        let guarded_attribution = if guard.is_some() {
            let evidence = graph_evidence(0).filter(|proof| proof.proves_unconfigured_eager());
            let attribution = command_node_indices
                .as_deref()
                .zip(execution_paths.as_deref())
                .and_then(|(nodes, paths)| {
                    cuda_submission_attribution(
                        &command_phases,
                        nodes,
                        &commands,
                        paths,
                        reusable_graph_node_counts.as_deref(),
                        Vec::new(),
                    )
                    .ok()
                })
                .and_then(|attribution| attribution.with_graph_evidence(evidence?));
            if attribution.is_none() {
                stream.state.cancel_recording();
                return Err(GuardedDeviceSubmissionError::Rejected(
                    ferrum_interfaces::execution_cost::GuardedNotSubmittedReason::AttributionUnavailable,
                ));
            }
            attribution
        } else {
            None
        };
        drop(validate_stage);

        let begin_timing_stage =
            CudaSubmissionStageTimer::start(timing_sink, DeviceSubmissionStage::BeginTiming);
        let timing = match timing_mode {
            DeviceTimingMode::Off => CudaFenceTiming::NotRequested,
            DeviceTimingMode::Completion
            | DeviceTimingMode::Replay
            | DeviceTimingMode::Kernel
            | DeviceTimingMode::Verification => {
                match stream
                    .stream
                    .record_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))
                {
                    Ok(start) => CudaFenceTiming::Events { start },
                    Err(_) => CudaFenceTiming::Unavailable,
                }
            }
        };
        drop(begin_timing_stage);

        let enqueue_stage =
            CudaSubmissionStageTimer::start(timing_sink, DeviceSubmissionStage::EnqueueCommands);
        let mut command_spans =
            physical_span_attribution.then(|| Vec::with_capacity(commands.len()));
        let mut replayed_segments = logical_attribution.then(Vec::new);
        let mut index = 0;
        let mut executable_candidate_index = 0;
        let mut actual_replayed_segments = 0_u64;
        if let Some(guard) = guard {
            // All blocking context/preparation work and host attribution
            // allocation is complete. Recheck the exact cache before enqueue.
            if stream.executable_cache.cost_graph_stream_state() != graph_after_preparation {
                stream.state.cancel_recording();
                return Err(GuardedDeviceSubmissionError::Rejected(
                    ferrum_interfaces::execution_cost::GuardedNotSubmittedReason::AttributionUnavailable,
                ));
            }
            if let Err(reason) = guard.check(guarded_attribution.as_ref()) {
                stream.state.cancel_recording();
                return Err(GuardedDeviceSubmissionError::Rejected(reason));
            }
        }
        while index < commands.len() {
            if let Some(invocation) = commands[index].reusable_execution_invocation() {
                let start = command_spans.as_ref().and_then(|_| {
                    stream
                        .stream
                        .record_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))
                        .ok()
                });
                let launched = stream.executable_cache.launch_program_segment(
                    &stream.stream,
                    invocation,
                    timing_mode,
                    logical_attribution,
                );
                let end = command_spans.as_ref().and_then(|_| {
                    stream
                        .stream
                        .record_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))
                        .ok()
                });
                match launched {
                    Ok(Some(launch)) => {
                        actual_replayed_segments += 1;
                        if let Some(replayed_segments) = replayed_segments.as_mut() {
                            let physical_command_index = u32::try_from(index).expect(
                                "CUDA direct replay command index was validated before submission",
                            );
                            let reusable_executable_fingerprint = launch
                                .reusable_executable_fingerprint()
                                .expect("attributable CUDA replay retained its fingerprint");
                            let logical_commands = launch
                                .replayed_logical_commands()
                                .expect("attributable CUDA replay retained its logical commands");
                            let reusable_graph_node_count =
                                logical_commands.iter().try_fold(0_u64, |total, command| {
                                    total.checked_add(command.reusable_graph_node_count())
                                });
                            let Some(reusable_graph_node_count) = reusable_graph_node_count else {
                                stream.state.fail();
                                self.quarantine(stream, commands);
                                panic!(
                                    "CUDA submission became indeterminate because replay graph attribution overflowed u64"
                                );
                            };
                            let replayed = DeviceReplayedSegmentAttribution::new(
                                physical_command_index,
                                invocation.program_id().clone(),
                                invocation.segment().clone(),
                                reusable_executable_fingerprint.to_string(),
                                logical_commands.as_ref().to_vec(),
                            );
                            let Some(replayed) = replayed else {
                                stream.state.fail();
                                self.quarantine(stream, commands);
                                panic!(
                                    "CUDA submission became indeterminate because sealed replay attribution drifted"
                                );
                            };
                            execution_paths
                                .as_mut()
                                .expect("logical CUDA attribution retained execution paths")
                                [index] = DeviceExecutionPath::Replayed;
                            reusable_graph_node_counts
                                .as_mut()
                                .expect("logical CUDA attribution retained graph counts")[index] =
                                Some(reusable_graph_node_count);
                            replayed_segments.push(replayed);
                        }
                        if let Some(command_spans) = command_spans.as_mut() {
                            command_spans.push(
                                CudaExecutionSpanEventTiming::new(
                                    index,
                                    index + 1,
                                    DeviceExecutionSpanKind::ReusableExecutable,
                                    DeviceExecutionIntervalKind::Compute,
                                    "cuda direct reusable executable",
                                    launch.reusable_executable_fingerprint(),
                                    start.zip(end),
                                )
                                .expect("CUDA direct replay index was validated as u32"),
                            );
                        }
                        if S::ENABLED {
                            replay_observation.observe_replayed_segment(
                                invocation.segment().logical_command_count() as usize,
                            );
                        }
                        index += 1;
                        continue;
                    }
                    Ok(None) => {
                        stream.state.fail();
                        self.quarantine(stream, commands);
                        panic!("CUDA reusable execution disappeared after pre-submit validation");
                    }
                    Err(error) => {
                        stream.state.fail();
                        self.quarantine(stream, commands);
                        panic!(
                            "CUDA submission became indeterminate while launching a reusable program: {error}"
                        );
                    }
                }
            }
            while executable_candidates
                .get(executable_candidate_index)
                .is_some_and(|candidate| candidate.start() < index)
            {
                executable_candidate_index += 1;
            }
            let replay_candidate = executable_candidates
                .get(executable_candidate_index)
                .filter(|candidate| candidate.start() == index);
            let replayed = match replay_candidate {
                Some(candidate)
                    if physical_span_attribution && stream.executable_cache.contains(candidate) =>
                {
                    let start = command_spans.as_ref().and_then(|_| {
                        stream
                            .stream
                            .record_event(Some(
                                cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT,
                            ))
                            .ok()
                    });
                    let launched =
                        stream
                            .executable_cache
                            .launch(&stream.stream, candidate, timing_mode);
                    let end = command_spans.as_ref().and_then(|_| {
                        stream
                            .stream
                            .record_event(Some(
                                cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT,
                            ))
                            .ok()
                    });
                    match launched {
                        Ok(Some(launch)) => Some(Ok((
                            candidate.end(),
                            start.zip(end),
                            launch.reusable_executable_fingerprint(),
                            launch.reusable_graph_node_counts(),
                        ))),
                        Ok(None) => None,
                        Err(error) => Some(Err(error)),
                    }
                }
                Some(candidate) if !physical_span_attribution => {
                    match stream
                        .executable_cache
                        .launch(&stream.stream, candidate, timing_mode)
                    {
                        Ok(Some(_)) => Some(Ok((candidate.end(), None, None, None))),
                        Ok(None) => None,
                        Err(error) => Some(Err(error)),
                    }
                }
                Some(_) | None => None,
            };
            match replayed {
                Some(Ok((
                    segment_end,
                    events,
                    reusable_executable_fingerprint,
                    graph_node_counts,
                ))) => {
                    actual_replayed_segments += 1;
                    if let Some(execution_paths) = execution_paths.as_mut() {
                        execution_paths[index..segment_end].fill(DeviceExecutionPath::Replayed);
                    }
                    if let (Some(target), Some(observed)) =
                        (reusable_graph_node_counts.as_mut(), graph_node_counts)
                    {
                        debug_assert_eq!(observed.len(), segment_end - index);
                        for (target, observed) in target[index..segment_end]
                            .iter_mut()
                            .zip(observed.iter().copied())
                        {
                            *target = Some(u64::from(observed));
                        }
                    }
                    if let Some(command_spans) = command_spans.as_mut() {
                        command_spans.push(
                            CudaExecutionSpanEventTiming::new(
                                index,
                                segment_end,
                                DeviceExecutionSpanKind::ReusableExecutable,
                                DeviceExecutionIntervalKind::Compute,
                                "cuda reusable executable",
                                reusable_executable_fingerprint,
                                events,
                            )
                            .expect("CUDA replay range was validated as u32"),
                        );
                    }
                    if S::ENABLED {
                        replay_observation.observe_replayed_segment(segment_end - index);
                    }
                    index = segment_end;
                    continue;
                }
                Some(Err(error)) => {
                    stream.state.fail();
                    self.quarantine(stream, commands);
                    panic!(
                        "CUDA submission became indeterminate while launching a reusable executable: {error}"
                    );
                }
                None => {}
            }
            let command_start = command_spans.as_ref().and_then(|_| {
                stream
                    .stream
                    .record_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))
                    .ok()
            });
            if let Err(error) = commands[index].enqueue(&stream.stream, &stream.blas) {
                stream.state.fail();
                self.quarantine(stream, commands);
                panic!("CUDA submission became indeterminate while enqueueing its batch: {error}");
            }
            if let Some(command_spans) = command_spans.as_mut() {
                let command_end = stream
                    .stream
                    .record_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))
                    .ok();
                let command = &commands[index];
                let interval_kind = if command.compute_dispatch_count > 0 {
                    DeviceExecutionIntervalKind::Compute
                } else {
                    DeviceExecutionIntervalKind::Transfer
                };
                command_spans.push(
                    CudaExecutionSpanEventTiming::new(
                        index,
                        index + 1,
                        DeviceExecutionSpanKind::EagerCommand,
                        interval_kind,
                        command.operation,
                        None,
                        command_start.zip(command_end),
                    )
                    .expect("CUDA eager command index was validated as u32"),
                );
            }
            if S::ENABLED {
                replay_observation.observe_eager_command();
            }
            index += 1;
        }
        // Snapshot non-aliasing typed binding statuses after all eager or
        // replayed computation and before the fence. The prelude commands own
        // both their source regions and pinned host destinations until terminal.
        let numerical_snapshot = commands
            .iter()
            .flat_map(|command| &command.completion_checks)
            .filter(|check| check.deferred)
            .try_for_each(|check| check.enqueue(&stream.stream));
        if let Err(error) = numerical_snapshot {
            stream.state.fail();
            self.quarantine(stream, commands);
            panic!(
                "CUDA submission became indeterminate while snapshotting numerical status: {error}"
            );
        }
        drop(enqueue_stage);
        if S::ENABLED {
            timing_sink.record_reusable_execution(replay_observation);
        }
        let attribution = match command_node_indices
            .as_ref()
            .zip(execution_paths.as_ref())
            .map(|(command_node_indices, execution_paths)| {
                let attribution = cuda_submission_attribution(
                    &command_phases,
                    command_node_indices,
                    &commands,
                    execution_paths,
                    reusable_graph_node_counts.as_deref(),
                    replayed_segments.unwrap_or_default(),
                )?;
                match graph_evidence(actual_replayed_segments) {
                    Some(evidence) => attribution.with_graph_evidence(evidence).ok_or_else(|| {
                        CudaDeviceRuntimeError::contract(
                            "CUDA actual graph evidence contradicts command attribution",
                        )
                    }),
                    None => Ok(attribution),
                }
            }) {
            None => None,
            Some(Ok(attribution)) => Some(attribution),
            Some(Err(error)) => {
                stream.state.fail();
                self.quarantine(stream, commands);
                panic!(
                    "CUDA submission became indeterminate while binding native attribution: {error}"
                );
            }
        };
        let command_timing = match timing_mode {
            DeviceTimingMode::Off | DeviceTimingMode::Completion => {
                CudaFenceCommandTiming::NotRequested
            }
            DeviceTimingMode::Replay
            | DeviceTimingMode::Kernel
            | DeviceTimingMode::Verification => command_spans.map_or(
                CudaFenceCommandTiming::Unavailable(
                    DeviceTimingUnavailableReason::BackendMeasurementFailed,
                ),
                |spans| CudaFenceCommandTiming::Spans {
                    command_count,
                    spans,
                },
            ),
        };

        let fence_stage = CudaSubmissionStageTimer::start(
            timing_sink,
            DeviceSubmissionStage::RecordFenceAndAccount,
        );
        let fence_flags = match timing_mode {
            DeviceTimingMode::Off => None,
            DeviceTimingMode::Completion
            | DeviceTimingMode::Replay
            | DeviceTimingMode::Kernel
            | DeviceTimingMode::Verification => {
                Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT)
            }
        };
        let event = match stream.stream.record_event(fence_flags) {
            Ok(event) => event,
            Err(error) => {
                stream.state.fail();
                self.quarantine(stream, commands);
                panic!("CUDA submission became indeterminate while recording its fence: {error:?}");
            }
        };
        if let Err(error) = stream.state.submission_recorded() {
            stream.state.fail();
            self.quarantine(stream, commands);
            panic!("CUDA submission became indeterminate while accounting its fence: {error}");
        }
        let fence = CudaDeviceFence {
            event,
            timing,
            command_timing,
            attribution,
            stream_state: Arc::clone(&stream.state),
            terminal_accounted: AtomicBool::new(false),
            _stream: Arc::clone(&stream.stream),
            _blas: Arc::clone(&stream.blas),
            _commands: commands,
        };
        drop(fence_stage);
        Ok(fence)
    }
}
