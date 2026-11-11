//! Controlled contract failures. A real asynchronous original shutdown gate
//! owns ambiguous authority; these tests are not a model numerical fixture.
use super::*;
use async_trait::async_trait;
use ferrum_execution::{
    executor_lifecycle::ExecutorShutdownPermit,
    model_executor::{ExecutorCapabilities, ExecutorStatus},
    token_execution::*,
    ModelExecutor,
};
use ferrum_runtime::HostMemoryClass;
use ferrum_testkit::MockModelExecutor;
use ferrum_types::ModelInfo;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Mutex,
};
use tokio::sync::watch;

#[derive(Clone, Copy)]
enum Fault {
    StartupError,
    AdmissionError,
    Unknown,
    ContractViolation,
    BothCloseErrors,
}

#[derive(Default)]
struct Observed {
    admitted: AtomicUsize,
    submitted: AtomicUsize,
}

struct ContractExecutor {
    issuer: ExecutorSequenceIssuer,
    sequences: Mutex<Vec<ExecutionFrontier>>,
    recovery: Mutex<Option<ExecutionRecoveryKey>>,
    release: watch::Sender<bool>,
    observed: Arc<Observed>,
    fault: Fault,
    info: ModelInfo,
    capabilities: ExecutorCapabilities,
    status: ExecutorStatus,
}

impl ContractExecutor {
    fn held(&self, key: ExecutionRecoveryKey) -> Result<ExecutionHeld> {
        let issuer = self.issuer.clone();
        let sequence = key.sequence().clone();
        let mut release = self.release.subscribe();
        self.issuer.held(key, async move {
            while !*release.borrow_and_update() {
                release
                    .changed()
                    .await
                    .map_err(|_| FerrumError::backend("original gate disconnected"))?;
            }
            Ok(ExecutionTerminalOutcome::ended(issuer.ended(&sequence)?))
        })
    }

    fn close(&self, sequence: &ExecutorSequence) -> Result<ExecutionCloseOutcome> {
        assert!(self
            .sequences
            .lock()
            .unwrap()
            .iter()
            .any(|f| f.sequence() == sequence));
        if matches!(self.fault, Fault::BothCloseErrors) {
            return Err(FerrumError::backend("original close remains unresolved"));
        }
        if let Some(key) = self
            .recovery
            .lock()
            .unwrap()
            .as_ref()
            .filter(|k| k.sequence() == sequence)
        {
            return self.held(key.clone()).map(ExecutionCloseOutcome::held);
        }
        Ok(ExecutionCloseOutcome::ended(self.issuer.ended(sequence)?))
    }
}

#[async_trait]
impl ModelExecutor for ContractExecutor {
    fn info(&self) -> &ModelInfo {
        &self.info
    }
    fn capabilities(&self) -> ExecutorCapabilities {
        self.capabilities.clone()
    }
    fn status(&self) -> ExecutorStatus {
        self.status.clone()
    }

    async fn prepare_startup(&self) -> Result<()> {
        if matches!(self.fault, Fault::StartupError) {
            return Err(FerrumError::backend(
                "original startup configuration failed",
            ));
        }
        Ok(())
    }

    fn begin_shutdown(
        self: Arc<Self>,
        permit: &ExecutorShutdownPermit,
    ) -> ExecutorShutdownObserver {
        permit.check(self.as_ref()).unwrap();
        let (reporter, observer) = ExecutorShutdownObserver::channel(Arc::new(()));
        std::thread::spawn(move || {
            let mut release = self.release.subscribe();
            futures::executor::block_on(async {
                while !*release.borrow_and_update() {
                    release.changed().await.unwrap();
                }
            });
            for frontier in self.sequences.lock().unwrap().iter() {
                // Only the original retirement service may settle authority
                // whose failed admission never returned a public frontier.
                self.issuer.ended(frontier.sequence()).unwrap();
            }
            drop(self);
            reporter.retired(None);
        });
        observer
    }

    async fn admit_tokens(
        &self,
        admission: ExecutionAdmission<'_>,
    ) -> Result<TokenAdmissionOutcome> {
        admission.validate()?;
        self.observed.admitted.fetch_add(1, Ordering::SeqCst);
        let sequence = self.issuer.issue_sequence()?;
        let frontier = self.issuer.frontier(&sequence, 0)?;
        self.sequences.lock().unwrap().push(frontier.clone());
        if matches!(self.fault, Fault::AdmissionError) {
            return Err(FerrumError::backend(
                "admission failed after original authority acquisition",
            ));
        }
        Ok(TokenAdmissionOutcome::Admitted(frontier))
    }

    async fn execute_tokens(&self, work: &[ExecutionWork<'_>]) -> Result<ExecutionBatchOutcome> {
        assert_eq!(work.len(), 2);
        self.observed.submitted.fetch_add(1, Ordering::SeqCst);
        let batch = self.issuer.batch(work, self.info.vocab_size)?;
        let key = self.issuer.recovery_key(work[0].sequence())?;
        *self.recovery.lock().unwrap() = Some(key.clone());
        let participant = batch
            .participant(
                0,
                ExecutionProgress::Unknown {
                    last_known: LastKnownExecutionStage::Submitted,
                    known_executed_range: None,
                },
                ContinuationDisposition::Held(self.held(key)?),
                1,
            )
            .unwrap();
        if matches!(self.fault, Fault::BothCloseErrors) {
            let second = batch
                .participant(
                    1,
                    ExecutionProgress::Unknown {
                        last_known: LastKnownExecutionStage::Submitted,
                        known_executed_range: None,
                    },
                    ContinuationDisposition::Held(
                        self.held(self.issuer.recovery_key(work[1].sequence())?)?,
                    ),
                    0,
                )
                .unwrap();
            return Ok(batch
                .interrupted(
                    vec![],
                    vec![participant, second],
                    vec![
                        FerrumError::backend("second unknown submission").into(),
                        FerrumError::backend("original unknown submission").into(),
                    ],
                )
                .unwrap());
        }
        let TokenWork::Prefill(chunk) = work[1].work() else {
            panic!("fixture expects the survivor's first prefill")
        };
        let completed = batch
            .completion(
                1,
                ExecutedTokenWork::Prefill {
                    chunk,
                    capacity_probe_count: 0,
                },
                ExecutionProduct::Intermediate,
            )
            .unwrap();
        if matches!(self.fault, Fault::ContractViolation) {
            return Ok(ExecutionBatchOutcome::ContractViolation(
                ExecutionContractViolation::new(
                    FerrumError::backend("original native batch contract violation"),
                    (
                        vec![completed],
                        vec![participant],
                        vec![
                            FerrumError::backend("unrelated ledger cause").into(),
                            FerrumError::backend("original unknown submission").into(),
                        ],
                    ),
                ),
            ));
        }
        Ok(batch
            .interrupted(
                vec![completed],
                vec![participant],
                vec![
                    FerrumError::backend("unrelated ledger cause").into(),
                    FerrumError::backend("original unknown submission").into(),
                ],
            )
            .unwrap())
    }

    async fn finish_tokens(
        &self,
        sequence: &ExecutorSequence,
        _: ExecutionEndReason,
    ) -> Result<ExecutionCloseOutcome> {
        self.close(sequence)
    }
    fn cancel_tokens(&self, sequence: &ExecutorSequence) -> Result<ExecutionCloseOutcome> {
        self.close(sequence)
    }
}

fn controlled(
    fault: Fault,
) -> (
    Fixture,
    NativeExpansionCohort,
    HostMemoryAccount,
    watch::Sender<bool>,
    Arc<Observed>,
) {
    let mut fixture = Fixture::new_two_answers();
    let assets = fixture.take_expansion();
    let requests = requests(&assets, 2);
    let metadata = MockModelExecutor::instant(64);
    let (release, _) = watch::channel(false);
    let observed = Arc::new(Observed::default());
    let executor = ContractExecutor {
        issuer: ExecutorSequenceIssuer::new(),
        sequences: Mutex::new(vec![]),
        recovery: Mutex::new(None),
        release: release.clone(),
        observed: Arc::clone(&observed),
        fault,
        info: metadata.info().clone(),
        capabilities: metadata.capabilities(),
        status: metadata.status(),
    };
    let owner = ExecutorOwner::new(Box::new(executor));
    let scope = owner.scope().unwrap();
    let host = host();
    let history = requests
        .iter()
        .map(|r| execute::history_backing_bytes(&host, r.generation.max_new_tokens).unwrap())
        .sum();
    let phase = host
        .reserve_scope(&[(HostMemoryClass::Output, history)])
        .unwrap();
    let phase_account = phase.account();
    owner.shutdown_observer().retain_until_retired(phase);
    let cohort = NativeExpansionCohort::from_owner(
        assets,
        requests,
        owner,
        scope,
        phase_account,
        vec![None, None],
    );
    (fixture, cohort, host, release, observed)
}

#[tokio::test]
async fn startup_failure_keeps_original_phase_until_original_retirement() {
    let (_fixture, mut cohort, host, release, observed) = controlled(Fault::StartupError);
    drop(cohort.scope.take());
    let error = prepare_owned_startup(&cohort.owner).unwrap_err();
    assert!(error
        .to_string()
        .contains("original startup configuration failed"));
    let cleanup = error
        .cleanup()
        .expect("original startup cleanup must be retained")
        .clone();
    assert!(cohort.owner.scope().is_err());
    assert_eq!(observed.admitted.load(Ordering::SeqCst), 0);
    assert_eq!(observed.submitted.load(Ordering::SeqCst), 0);
    assert!(host.usage().unwrap().committed_bytes() > 0);
    assert_ne!(cleanup.snapshot().phase(), ExecutorShutdownPhase::Retired);
    release.send_replace(true);
    tokio::time::timeout(std::time::Duration::from_secs(10), cleanup.wait())
        .await
        .unwrap()
        .result()
        .unwrap();
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    assert_eq!(host.usage().unwrap().allocated_bytes(), 0);
}

#[tokio::test]
async fn bare_admission_error_isolates_siblings_and_retains_phase_until_original_retirement() {
    let (_fixture, mut cohort, host, release, observed) = controlled(Fault::AdmissionError);
    let wake = futures::task::noop_waker();
    cohort.advance(&[false, false], &wake).unwrap();
    assert!(matches!(cohort.state, Some(CohortState::Retiring { .. })));
    let failure = cohort.participants[0].failure.as_ref().unwrap();
    assert_eq!(failure.admission_probe, execute::AdmissionProbe::Entered);
    assert!(
        failure.frontier.is_none() && failure.admission.is_none() && failure.maintenance.is_none()
    );
    assert!(cohort.owner.scope().is_err());
    assert!(matches!(
        cohort.advance(&[false, false], &wake).unwrap(),
        ExpansionCohortProgress::Pending {
            made_progress: false
        }
    ));
    assert_eq!(observed.admitted.load(Ordering::SeqCst), 1);
    assert_eq!(observed.submitted.load(Ordering::SeqCst), 0);
    assert_ne!(
        cohort.cleanup().snapshot().phase(),
        ExecutorShutdownPhase::Retired
    );
    assert!(host.usage().unwrap().committed_bytes() > 0);
    release.send_replace(true);
    let error = complete(&mut cohort, &[false, false]).await.err().unwrap();
    assert!(error
        .to_string()
        .contains("admission failed after original authority acquisition"));
    assert_eq!(
        cohort.cleanup().snapshot().phase(),
        ExecutorShutdownPhase::Retired
    );
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    assert_eq!(host.usage().unwrap().allocated_bytes(), 0);
}

#[tokio::test]
async fn unknown_held_pauses_all_shared_submission_without_losing_original_outcome() {
    let (_fixture, mut cohort, host, release, observed) = controlled(Fault::Unknown);
    advance_until(&mut cohort, &[false, false], |c| {
        matches!(
            c.participants[0].state,
            Some(ParticipantState::Closing { .. })
        )
    });
    let failure = cohort.participants[0].failure.as_ref().unwrap();
    assert!(failure
        .frontier
        .as_ref()
        .unwrap()
        .sequence()
        .observed_end()
        .is_none());
    let reference = failure.wave_observation.unwrap();
    let retained = &cohort.retained_waves[reference.wave];
    assert_eq!(reference.input_index, 0);
    assert_eq!(retained._requests[0], cohort.participants[0].request);
    assert_eq!(
        &retained._input_frontiers[0],
        failure.frontier.as_ref().unwrap()
    );
    assert!(failure
        .error
        .to_string()
        .contains("original unknown submission"));
    let ExecutionBatchOutcome::Interrupted(outcome) = &retained._outcome else {
        panic!("original interruption missing")
    };
    assert_eq!(outcome.remaining()[0].input_index(), 0);
    assert_eq!(outcome.remaining()[0].cause_index(), 1);
    assert_eq!(outcome.completed()[0].input_index(), 1);
    let Some(ParticipantState::Tokens(survivor)) = &cohort.participants[1].state else {
        panic!("completed peer must keep its own live cursor")
    };
    assert_eq!(survivor.consumed_tokens(), 1);
    assert_eq!(observed.admitted.load(Ordering::SeqCst), 2);
    assert_eq!(observed.submitted.load(Ordering::SeqCst), 1);
    assert!(matches!(
        cohort
            .advance(&[false, true], &futures::task::noop_waker())
            .unwrap(),
        ExpansionCohortProgress::Pending {
            made_progress: false
        }
    ));
    assert!(matches!(
        cohort.participants[1].state,
        Some(ParticipantState::Tokens(_))
    ));
    assert!(cohort.participants[1].ended.is_none());
    assert_eq!(observed.submitted.load(Ordering::SeqCst), 1);
    assert!(host.usage().unwrap().committed_bytes() > 0);
    release.send_replace(true);
    let results = complete(&mut cohort, &[false, true]).await.unwrap();
    assert!(matches!(results[0], ExpansionParticipantResult::Failed(_)));
    assert!(matches!(results[1], ExpansionParticipantResult::Cancelled));
    assert_eq!(observed.submitted.load(Ordering::SeqCst), 1);
    assert!(cohort.participants.iter().all(|p| p
        .ended
        .as_ref()
        .unwrap()
        .sequence()
        .observed_end()
        .is_some()));
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}

#[tokio::test]
async fn contract_violation_retains_original_ledger_and_phase_until_owner_retires() {
    let (_fixture, mut cohort, host, release, observed) = controlled(Fault::ContractViolation);
    advance_until(&mut cohort, &[false, false], |c| {
        matches!(c.state, Some(CohortState::Retiring { .. }))
    });
    assert!(cohort.owner.scope().is_err());
    assert!(cohort.scope.is_none());
    assert_eq!(observed.admitted.load(Ordering::SeqCst), 2);
    assert_eq!(observed.submitted.load(Ordering::SeqCst), 1);
    for (index, participant) in cohort.participants.iter().enumerate() {
        assert!(participant.completed.is_none());
        let failure = participant.failure.as_ref().unwrap();
        assert!(failure.history.as_ref().unwrap().ids().is_empty());
        // A rejected completion cannot become the survivor's accepted frontier.
        assert_eq!(failure.frontier.as_ref().unwrap().consumed_tokens(), 0);
        let reference = failure.wave_observation.unwrap();
        assert_eq!(reference.input_index, index);
        assert_eq!(
            cohort.retained_waves[reference.wave]._requests[index],
            participant.request
        );
        assert_eq!(
            &cohort.retained_waves[reference.wave]._input_frontiers[index],
            failure.frontier.as_ref().unwrap()
        );
    }
    let ExecutionBatchOutcome::ContractViolation(original) = &cohort.retained_waves[0]._outcome
    else {
        panic!("original contract violation must retain its observations")
    };
    assert_eq!(original.completed[0].input_index(), 1);
    assert_eq!(original.remaining[0].input_index(), 0);
    assert_eq!(original.remaining[0].cause_index(), 1);
    assert!(matches!(
        original.remaining[0].continuation(),
        ContinuationDisposition::Held(_)
    ));
    assert!(host.usage().unwrap().allocated_bytes() > 0);
    assert!(host.usage().unwrap().committed_bytes() > 0);
    assert!(matches!(
        cohort
            .advance(&[false, false], &futures::task::noop_waker())
            .unwrap(),
        ExpansionCohortProgress::Pending {
            made_progress: false
        }
    ));
    assert_eq!(observed.submitted.load(Ordering::SeqCst), 1);
    release.send_replace(true);
    let error = complete(&mut cohort, &[false, false]).await.err().unwrap();
    assert!(error
        .to_string()
        .contains("original native batch contract violation"));
    assert_eq!(
        cohort.cleanup().snapshot().phase(),
        ExecutorShutdownPhase::Retired
    );
    assert!(cohort.retained_waves.is_empty());
    assert!(cohort.participants.iter().all(|p| p.failure.is_none()));
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    assert_eq!(host.usage().unwrap().allocated_bytes(), 0);
}

#[tokio::test]
async fn later_batch_failure_preserves_first_retirement_and_both_original_histories() {
    let (_fixture, mut cohort, host, release, observed) = controlled(Fault::BothCloseErrors);
    advance_until(&mut cohort, &[false, false], |c| {
        matches!(c.state, Some(CohortState::Retiring { .. }))
    });
    let Some(CohortState::Retiring {
        failure: Some(error),
        ..
    }) = &cohort.state
    else {
        panic!("ambiguous batch must seal the original owner")
    };
    assert!(error.to_string().contains("original unknown submission"));
    assert!(cohort.participants[1]
        .failure
        .as_ref()
        .unwrap()
        .error
        .to_string()
        .contains("second unknown submission"));
    for (index, participant) in cohort.participants.iter().enumerate() {
        let failure = participant.failure.as_ref().unwrap();
        assert!(failure.history.is_some());
        assert!(failure.close_error.is_some());
        assert_eq!(failure.wave_observation.unwrap().input_index, index);
    }
    assert_eq!(observed.submitted.load(Ordering::SeqCst), 1);
    assert!(host.usage().unwrap().allocated_bytes() > 0);
    assert!(matches!(
        cohort
            .advance(&[false, false], &futures::task::noop_waker())
            .unwrap(),
        ExpansionCohortProgress::Pending {
            made_progress: false
        }
    ));
    release.send_replace(true);
    let error = complete(&mut cohort, &[false, false]).await.err().unwrap();
    assert!(error.to_string().contains("original unknown submission"));
    assert_eq!(observed.submitted.load(Ordering::SeqCst), 1);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    assert_eq!(host.usage().unwrap().allocated_bytes(), 0);
}

#[tokio::test]
async fn dropping_ambiguous_cohort_transfers_histories_and_phase_to_original_cleanup() {
    let (_fixture, mut cohort, host, release, observed) = controlled(Fault::AdmissionError);
    cohort
        .advance(&[false, false], &futures::task::noop_waker())
        .unwrap();
    let cleanup = cohort.cleanup();
    drop(cohort);
    assert_ne!(cleanup.snapshot().phase(), ExecutorShutdownPhase::Retired);
    assert!(host.usage().unwrap().allocated_bytes() > 0);
    assert!(host.usage().unwrap().committed_bytes() > 0);
    assert_eq!(observed.admitted.load(Ordering::SeqCst), 1);
    release.send_replace(true);
    tokio::time::timeout(std::time::Duration::from_secs(10), cleanup.wait())
        .await
        .unwrap()
        .result()
        .unwrap();
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    assert_eq!(host.usage().unwrap().allocated_bytes(), 0);
}
