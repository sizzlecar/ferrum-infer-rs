use super::*;

// The public diagnostic traits accept a fixture-owned disabled sink; the
// production convenience sink is intentionally private to the interfaces crate.
struct NoHostTiming;
impl DeviceSubmissionTimingSink for NoHostTiming {
    const ENABLED: bool = false;
    fn record_device_submission(&self, _: DeviceSubmissionStage, _: std::time::Duration) {}
}
impl SubmissionWaveDispatchTimingSink for NoHostTiming {
    fn record(&self, _: SubmissionWaveDispatchStage, _: std::time::Duration) {}
}
#[derive(Clone, Copy, Debug)]
pub enum Path {
    Eager,
    Warm,
    Replay,
}

pub struct ObservedRun {
    pub outputs: Vec<Vec<u8>>,
    pub dependency_work: Vec<(u64, u64, u64)>,
}

#[derive(Default)]
pub struct SubmissionRendezvous {
    submitted: std::sync::Mutex<(u32, bool)>,
    ready: std::sync::Condvar,
}
impl SubmissionRendezvous {
    fn wait(&self) {
        let mut submitted = self.submitted.lock().unwrap();
        submitted.0 += 1;
        self.ready.notify_all();
        let (submitted, _) = self
            .ready
            .wait_timeout_while(submitted, std::time::Duration::from_secs(30), |state| {
                state.0 < 2 && !state.1
            })
            .unwrap();
        let state = *submitted;
        drop(submitted);
        assert_eq!(
            state,
            (2, false),
            "other lane failed before the shared-state submission rendezvous"
        );
    }
}

struct SubmissionGuard<'a> {
    rendezvous: &'a SubmissionRendezvous,
    completed: bool,
}
impl SubmissionGuard<'_> {
    fn complete(&mut self) {
        self.rendezvous.wait();
        self.completed = true;
    }
}
impl Drop for SubmissionGuard<'_> {
    fn drop(&mut self) {
        if !self.completed {
            self.rendezvous
                .submitted
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .1 = true;
            self.rendezvous.ready.notify_all();
        }
    }
}
impl Fixture {
    pub fn execute(
        &self,
        sessions: &[Arc<SequenceSession<Runtime>>],
        tokens: &[Arc<[u32]>],
        range: Range<usize>,
        path: Path,
    ) -> Vec<Vec<u8>> {
        self.on_lane(&self.lane, &self.reaper, sessions, tokens, range, path)
    }
    pub fn on_lane(
        &self,
        lane: &Arc<ExecutionLane<Runtime>>,
        reaper: &Arc<CompletionReaper<Runtime>>,
        sessions: &[Arc<SequenceSession<Runtime>>],
        tokens: &[Arc<[u32]>],
        range: Range<usize>,
        path: Path,
    ) -> Vec<Vec<u8>> {
        self.on_lane_observed(lane, reaper, sessions, tokens, range, path, None)
            .outputs
    }

    pub fn on_lane_observed(
        &self,
        lane: &Arc<ExecutionLane<Runtime>>,
        reaper: &Arc<CompletionReaper<Runtime>>,
        sessions: &[Arc<SequenceSession<Runtime>>],
        tokens: &[Arc<[u32]>],
        range: Range<usize>,
        path: Path,
        submitted: Option<&SubmissionRendezvous>,
    ) -> ObservedRun {
        let mut submission_guard = submitted.map(|rendezvous| SubmissionGuard {
            rendezvous,
            completed: false,
        });
        assert_eq!(sessions.len(), tokens.len());
        let participants = sessions.len() as u32;
        let batch = ExecutionBatchParticipants::new(sessions.to_vec()).unwrap();
        let spans = tokens
            .iter()
            .map(|t| token_span(Arc::clone(t), range.clone()))
            .collect();
        let request = StepResourceAdmissionRequest::new(
            batch.bind_work_shape(spans).unwrap(),
            AdmissionFitPolicy::ImmediateOnly,
            AdmissionPressureAction::WaitForRelease,
        )
        .unwrap()
        .with_reusable_execution_bucket(self.reusable_bucket.clone());
        let step = loop {
            match batch.try_begin_step(request.clone(), lane).unwrap() {
                StepResourceAdmissionDecision::Admitted(s) => break s,
                StepResourceAdmissionDecision::BackingDeferred(d) => {
                    require_progress(d.maintain().unwrap())
                }
                StepResourceAdmissionDecision::Deferred(r) => {
                    require_progress(self.resources.maintain_for_admission_deferred(&r).unwrap())
                }
                StepResourceAdmissionDecision::PermanentRejected(r) => {
                    panic!("step rejected: {r:?}")
                }
            }
        };
        let wave = loop {
            match step
                .try_prepare_full_plan_submission_wave(
                    Arc::new(step.work_shape().clone()),
                    AdmissionFitPolicy::ImmediateOnly,
                    AdmissionPressureAction::WaitForRelease,
                )
                .unwrap()
            {
                StepSubmissionWaveAdmissionDecision::Prepared(w) => break w,
                StepSubmissionWaveAdmissionDecision::BackingDeferred(d) => {
                    require_progress(d.maintain().unwrap())
                }
                StepSubmissionWaveAdmissionDecision::Deferred(r) => {
                    require_progress(self.resources.maintain_for_admission_deferred(&r).unwrap())
                }
                other => panic!("wave rejected: {:?}", std::mem::discriminant(&other)),
            }
        };
        let active: Vec<_> = sessions
            .iter()
            .map(|s| TrustedActiveSequenceBinding::from_session(s).unwrap())
            .collect();
        let executable = self.compilation.executable();
        let plan = executable.execution_plan();
        let identity = OperationDispatch::bind_submission_wave_identity(
            executable,
            active.iter(),
            &wave,
            lane,
        )
        .unwrap();
        let uploads: Vec<_> = tokens
            .iter()
            .enumerate()
            .map(|(p, t)| {
                SubmissionWaveInputUpload::new(
                    id("node.embedding"),
                    p as u32,
                    0,
                    range.start as u64 * 4,
                    HostTransferLayout::new(ElementType::U32, range.len() as u64).unwrap(),
                    t[range.clone()]
                        .iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect(),
                )
                .unwrap()
            })
            .collect();
        let node = plan
            .payload()
            .nodes()
            .iter()
            .find(|n| n.id().as_str() == "node.head")
            .unwrap();
        let output = node
            .values()
            .iter()
            .find(|v| v.role() == ResolvedValueRole::Output && v.ordinal() == 0)
            .unwrap();
        let component = &output.storage().components()[0];
        let readbacks =
            CompletionReadbackCollectionRequest::new(vec![CompletionReadbackBatchRequest::new(
                (0..participants)
                    .map(|p| {
                        CompletionReadbackRequest::new(
                            node.id().clone(),
                            p,
                            component.resource_id().clone(),
                            component.offset_bytes(),
                            HostTransferLayout::new(ElementType::F32, OUTPUTS).unwrap(),
                        )
                        .unwrap()
                    })
                    .collect(),
            )
            .unwrap()])
            .unwrap();
        let program_id = OperationDispatch::reusable_execution_program_id_for_wave(
            self.providers.providers(),
            executable,
            &wave,
            lane,
        )
        .unwrap();
        let head_index = wave
            .nodes()
            .iter()
            .position(|n| n.node_id() == node.id())
            .unwrap() as u32;
        let profiled = if matches!(path, Path::Replay) {
            let catalog = lane.reusable_execution_catalog().unwrap();
            let program = catalog
                .programs()
                .iter()
                .find(|p| Some(p.program_id()) == program_id.as_ref())
                .expect("actual warmed topology");
            assert!(program.is_determinism_ready());
            let index = wave
                .nodes()
                .iter()
                .position(|n| n.node_id() == node.id())
                .unwrap() as u32;
            assert!(
                !program.eager_boundary_node_indices().contains(&index),
                "head must actually replay"
            );
            if self.head == Head::Q6 {
                assert!(
                    program
                        .retained_plan_dependencies()
                        .iter()
                        .any(|d| d.node_id() == node.id()),
                    "replay must retain validated weight flag"
                );
            }
            OperationDispatch::encode_and_submit_reusable_wave_with_inputs_and_timing(
                self.providers.providers(),
                executable,
                &identity,
                active.iter(),
                DeviceTimingMode::Replay,
                &uploads,
                program,
                SubmissionExecutionPolicy::determinism_replayed(0xa5),
                &NoHostTiming,
                wave,
                lane,
                reaper,
            )
            .unwrap()
        } else {
            let policy = if matches!(path, Path::Eager) {
                SubmissionExecutionPolicy::determinism_eager(0x5a)
            } else {
                SubmissionExecutionPolicy::adaptive()
            };
            OperationDispatch::encode_and_submit_wave_with_inputs_and_timing(
                self.providers.providers(),
                executable,
                &identity,
                active.iter(),
                if matches!(path, Path::Eager) {
                    DeviceTimingMode::Kernel
                } else {
                    DeviceTimingMode::Replay
                },
                &uploads,
                policy,
                &NoHostTiming,
                wave,
                lane,
                reaper,
            )
            .unwrap()
        };
        let (handle, attribution) = profiled.into_parts();
        // Keep the first submission's command leases alive until the other
        // lane has encoded and submitted against the same retained state.
        if let Some(guard) = submission_guard.as_mut() {
            guard.complete();
        }
        let mut dependency_work = Vec::new();
        let observed = if matches!(path, Path::Eager) {
            let attribution = attribution.expect("eager command attribution is required");
            let device = attribution.device();
            let physical = device
                .commands()
                .iter()
                .filter(|c| {
                    c.node_index() == Some(head_index)
                        && c.native_op_id() == "vnext_q6_f32_last_token_linear"
                })
                .map(|c| {
                    (
                        c.batching_form(),
                        c.participant_count(),
                        c.token_count(),
                        c.compute_dispatch_count(),
                    )
                });
            let head_commands: Vec<_> = physical.collect();
            assert_eq!(
                head_commands.len(),
                1,
                "head must have one actual physical command"
            );
            let (form, count, token_count, dispatches) = head_commands[0];
            let packed = range.len() == 1 && participants > 1;
            assert_eq!(
                form,
                if packed {
                    DeviceBatchingForm::Packed
                } else {
                    DeviceBatchingForm::ParticipantLoop
                },
                "verify actual row merge, not offered concurrency"
            );
            assert_eq!(count, participants);
            assert_eq!(token_count, u64::from(participants));
            let launches = if packed { 1 } else { u64::from(participants) };
            let mmq = self.head == Head::Q6 && (!packed || participants <= 32);
            for dependency in device
                .commands()
                .iter()
                .filter(|row| row.native_op_id() == "upstream.retained_weight_validation")
            {
                assert_eq!(
                    dependency.command_phase(),
                    DeviceCommandPhase::DynamicBinding
                );
                assert_eq!(dependency.node_index(), None);
                assert_eq!(dependency.execution_path(), DeviceExecutionPath::Eager);
                assert_eq!(dependency.batching_form(), DeviceBatchingForm::Scalar);
                assert_eq!(dependency.participant_start(), 0);
                assert_eq!(dependency.participant_count(), 0);
                assert_eq!(dependency.token_count(), 0);
                let work = (
                    dependency.compute_dispatch_count(),
                    dependency.transfer_command_count(),
                    dependency.dependency_wait_count(),
                );
                assert!(
                    [(1, 1, 1), (0, 0, 1)].contains(&work),
                    "cold memset/scan/wait or warm wait: {work:?}"
                );
                dependency_work.push(work);
            }
            assert_eq!(
                dependency_work.len(),
                usize::from(mmq),
                "one retained dependency per MMQ weight leaf; strict fallback queues none"
            );
            if mmq {
                assert!(
                    [4 * launches, 5 * launches].contains(&dispatches),
                    "pack/dot/optional fixup/publish must execute MMQ"
                );
            } else {
                assert_eq!(
                    dispatches, launches,
                    "declared ineligible route uses strict projection"
                );
            }
            Some(
                serde_json::json!({"batching_form":form,"physical_rows_per_launch":if packed {participants} else {1},"compute_dispatches":dispatches,"retained_dependency_work":dependency_work}),
            )
        } else {
            // Normal Replay mode intentionally permits direct CUDA graph
            // replay. It does not expose per-node kernel attribution; actual
            // ReplayOnly catalog checks plus changed-input readback qualify
            // this path without inventing missing command observations.
            None
        };
        let receipt = match handle.wait_with_readback_collection(readbacks).unwrap() {
            CompletionReadbackBatchObservation::Terminal(r) => r,
            other => panic!("readback did not terminate: {other:?}"),
        };
        let mut values = BTreeMap::new();
        for disposition in receipt.dispositions() {
            let CompletionReadbackDisposition::Succeeded(r) = disposition else {
                panic!("head readback failed: {disposition:?}")
            };
            assert!(values
                .insert(r.request().participant_index(), r.bytes().to_vec())
                .is_none());
        }
        assert_eq!(values.len(), participants as usize);
        drop((receipt, handle, identity, active));
        step.try_retire_normal().unwrap();
        println!(
            "{}",
            serde_json::json!({"kind":"q6_head_provider_dispatch","participants":participants,"range":[range.start,range.end],"path":format!("{path:?}"),"head":self.head,"eager_command_observation":observed})
        );
        ObservedRun {
            outputs: values.into_values().collect(),
            dependency_work,
        }
    }
}
