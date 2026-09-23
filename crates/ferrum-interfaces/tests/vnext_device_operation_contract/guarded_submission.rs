//! Shared actual test-backend preparation and submission. Ordinary and guarded
//! entrypoints execute the same code; the guard is after attribution preparation
//! and before any physical-submit trace or simulated device memory write.
use super::*;
use ferrum_interfaces::execution_cost::GuardedNotSubmittedReason;

impl TestRuntime {
    pub(super) fn submit_fixture(
        &self,
        _stream: &mut TestStream,
        commands: DeviceCommandBatch<TestCommand>,
        guard: Option<(
            &dyn DeviceSubmissionGuard,
            &mut Option<GuardedNotSubmittedReason>,
        )>,
    ) -> Result<TestFence, DefinitelyNotSubmitted<TestRuntimeError>> {
        assert!(!commands.is_empty(), "core must not submit an empty batch");
        let timing_mode = commands.timing_mode();
        let compute_path_requirement = commands.compute_path_requirement();
        let declared_eager_compute_node_count =
            commands.declared_eager_compute_node_indices().len();
        let declared_eager_compute_node_indices = commands
            .declared_eager_compute_node_indices()
            .iter()
            .copied()
            .collect::<BTreeSet<_>>();
        let attribution_requirement = commands.attribution_requirement();
        let reusable_execution_capture = commands.reusable_execution_capture().cloned();
        let entries = commands.into_entries();
        let mut compute_command_count = 0_usize;
        let mut replayed_compute_command_count = 0_usize;
        let mut observed_eager_compute_node_indices = BTreeSet::new();
        let mut exact_boundary_shape = true;
        for entry in &entries {
            if entry.phase() == DeviceCommandPhase::Compute {
                compute_command_count += 1;
                if *entry.command() == TestCommand::ReusableExecution {
                    replayed_compute_command_count += 1;
                } else if compute_path_requirement
                    == DeviceComputePathRequirement::ReplayedWithDeclaredEagerBoundaries
                {
                    exact_boundary_shape &= entry.node_index().is_some_and(|node_index| {
                        declared_eager_compute_node_indices.contains(&node_index)
                            && observed_eager_compute_node_indices.insert(node_index)
                    });
                }
            }
        }
        let reusable_invocations = {
            let mut trace = self.trace.lock().unwrap();
            if trace.pending_reusable_invocations.len() != replayed_compute_command_count {
                trace.pending_reusable_invocations.clear();
                return Err(DefinitelyNotSubmitted::new(TestRuntimeError(
                    "reusable invocation metadata differs from encoded commands",
                )));
            }
            std::mem::take(&mut trace.pending_reusable_invocations)
        };
        let compute_path_matches = match compute_path_requirement {
            DeviceComputePathRequirement::Adaptive => true,
            DeviceComputePathRequirement::EagerOnly => replayed_compute_command_count == 0,
            DeviceComputePathRequirement::ReplayedOnly => {
                compute_command_count > 0 && replayed_compute_command_count == compute_command_count
            }
            DeviceComputePathRequirement::ReplayedWithDeclaredEagerBoundaries => {
                !declared_eager_compute_node_indices.is_empty()
                    && declared_eager_compute_node_count
                        == declared_eager_compute_node_indices.len()
                    && exact_boundary_shape
                    && replayed_compute_command_count > 0
                    && replayed_compute_command_count < compute_command_count
                    && observed_eager_compute_node_indices == declared_eager_compute_node_indices
            }
        };
        if !compute_path_matches {
            return Err(DefinitelyNotSubmitted::new(TestRuntimeError(
                "compute-path requirement mismatch",
            )));
        }
        let command_phases = entries.iter().map(DeviceCommandEntry::phase).collect();
        let command_node_indices = entries.iter().map(DeviceCommandEntry::node_index).collect();
        let attribution = test_submission_attribution(
            timing_mode,
            attribution_requirement,
            &entries,
            reusable_invocations,
        )
        .map_err(DefinitelyNotSubmitted::new)?;
        if let Some((guard, rejection)) = guard {
            let missing_attribution = {
                let mut trace = self.trace.lock().unwrap();
                trace.guarded_submit_calls += 1;
                trace.guarded_missing_attribution
            };
            let evidence = if missing_attribution {
                None
            } else {
                attribution.as_ref()
            };
            if let Err(reason) = guard.check(evidence) {
                *rejection = Some(reason);
                return Err(DefinitelyNotSubmitted::new(TestRuntimeError(
                    "guard rejected after actual preparation",
                )));
            }
        }
        let scratch_events = entries
            .iter()
            .filter_map(|entry| {
                let node_index = entry.node_index()?;
                match (entry.phase(), entry.command()) {
                    (DeviceCommandPhase::Initialization, TestCommand::Zero) => {
                        Some((node_index, 0, false))
                    }
                    (
                        DeviceCommandPhase::Initialization,
                        TestCommand::Upload(value, BufferUsage::Scratch),
                    ) => Some((node_index, *value, false)),
                    (_, TestCommand::ScratchProvider | TestCommand::ScratchProviderWork(_, _)) => {
                        Some((node_index, 0xa5, true))
                    }
                    _ => None,
                }
            })
            .collect::<Vec<_>>();
        let commands = entries
            .into_iter()
            .map(DeviceCommandEntry::into_parts)
            .map(|(_, _, _, command)| command)
            .collect::<Vec<_>>();
        let command_count = commands.len();
        let memory_commands = commands.clone();
        let (drift, behavior, fence) = {
            let mut trace = self.trace.lock().unwrap();
            trace.submit_calls += 1;
            trace.submitted_command_counts.push(command_count);
            trace.submitted_command_phases.push(command_phases);
            trace
                .submitted_command_node_indices
                .push(command_node_indices);
            trace.submitted_commands.push(commands);
            trace
                .submitted_compute_path_requirements
                .push(compute_path_requirement);
            trace
                .submitted_attribution_requirements
                .push(attribution_requirement);
            for (node_index, value, observe_before_write) in scratch_events {
                if observe_before_write {
                    let observed = *trace.scratch_bytes.get(&node_index).unwrap_or(&0xa5);
                    trace.scratch_observations.push((node_index, observed));
                }
                trace.scratch_bytes.insert(node_index, value);
            }
            trace
                .submitted_reusable_captures
                .push(reusable_execution_capture);
            trace.next_fence += 1;
            (
                trace.drift_on_submit,
                trace.submit_behavior,
                TestFence(trace.next_fence, timing_mode, attribution),
            )
        };
        match behavior {
            SubmitBehavior::DefinitelyNotSubmitted => {
                return Err(DefinitelyNotSubmitted::new(TestRuntimeError(
                    "definitely-not-submitted",
                )));
            }
            SubmitBehavior::Panic => panic!("injected submit panic"),
            SubmitBehavior::Success => {}
        }
        if let Some(memory) = self.trace.lock().unwrap().memory.clone() {
            memory_fixture::execute(&memory, &memory_commands);
        }
        if drift {
            self.use_alternate_descriptor.store(true, Ordering::Release);
        }
        Ok(fence)
    }
}
