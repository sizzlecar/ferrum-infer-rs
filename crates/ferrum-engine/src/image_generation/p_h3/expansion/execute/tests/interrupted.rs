//! Controlled causal contract outcomes, not a model or GPU simulator. The
//! original issuer owns each frontier and a real watch gate controls Held end
//! publication, so dropping an observer cannot manufacture terminal progress.
use super::*;
use async_trait::async_trait;
use ferrum_execution::{
    executor_lifecycle::{ExecutorShutdownObserver, ExecutorShutdownPermit},
    model_executor::{ExecutorCapabilities, ExecutorStatus},
    ModelExecutor,
};
use ferrum_types::ModelInfo;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc, Mutex,
};
use tokio::sync::watch;

#[derive(Clone, Copy)]
enum Fault {
    Unknown,
    NotSubmitted,
    Rejected,
}
#[derive(Default)]
struct Observed {
    execute: AtomicUsize,
    cancel: AtomicUsize,
    finish: AtomicUsize,
}
struct State {
    frontier: Option<ExecutionFrontier>,
    recovery: Option<ExecutionRecoveryKey>,
}
struct ContractExecutor {
    issuer: ExecutorSequenceIssuer,
    state: Mutex<State>,
    fault: Fault,
    release: watch::Sender<bool>,
    observed: Arc<Observed>,
    info: ModelInfo,
    capabilities: ExecutorCapabilities,
    status: ExecutorStatus,
}
impl ContractExecutor {
    fn new(fault: Fault) -> (Self, watch::Sender<bool>, Arc<Observed>) {
        let metadata = MockModelExecutor::instant(64);
        let (release, _) = watch::channel(false);
        let observed = Arc::new(Observed::default());
        (
            Self {
                issuer: ExecutorSequenceIssuer::new(),
                state: Mutex::new(State {
                    frontier: None,
                    recovery: None,
                }),
                fault,
                release: release.clone(),
                observed: Arc::clone(&observed),
                info: metadata.info().clone(),
                capabilities: metadata.capabilities(),
                status: metadata.status(),
            },
            release,
            observed,
        )
    }
    fn held(&self, key: ExecutionRecoveryKey) -> ferrum_types::Result<ExecutionHeld> {
        let issuer = self.issuer.clone();
        let sequence = key.sequence().clone();
        let mut release = self.release.subscribe();
        self.issuer.held(key, async move {
            while !*release.borrow_and_update() {
                release
                    .changed()
                    .await
                    .map_err(|_| FerrumError::backend("original held gate disconnected"))?;
            }
            Ok(ExecutionTerminalOutcome::ended(issuer.ended(&sequence)?))
        })
    }
    fn close(&self, sequence: &ExecutorSequence) -> ferrum_types::Result<ExecutionCloseOutcome> {
        let state = self.state.lock().unwrap();
        if state.frontier.as_ref().map(ExecutionFrontier::sequence) != Some(sequence) {
            return Err(FerrumError::request_validation("foreign fixture close"));
        }
        if let Some(key) = &state.recovery {
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
    fn begin_shutdown(
        self: Arc<Self>,
        permit: &ExecutorShutdownPermit,
    ) -> ExecutorShutdownObserver {
        permit.check(self.as_ref()).unwrap();
        let original = Arc::try_unwrap(self)
            .ok()
            .expect("original contract owner must have no live scope");
        if let Some(frontier) = &original.state.lock().unwrap().frontier {
            assert!(
                frontier.sequence().observed_end().is_some(),
                "the original held gate must prove end before fixture retirement"
            );
        }
        drop(original);
        let (reporter, observer) = ExecutorShutdownObserver::channel(Arc::new(()));
        reporter.retired(None);
        observer
    }
    async fn admit_tokens(
        &self,
        admission: ExecutionAdmission<'_>,
    ) -> ferrum_types::Result<TokenAdmissionOutcome> {
        admission.validate()?;
        let sequence = self.issuer.issue_sequence()?;
        let frontier = self.issuer.frontier(&sequence, 0)?;
        let mut state = self.state.lock().unwrap();
        assert!(state.frontier.is_none());
        state.frontier = Some(frontier.clone());
        Ok(TokenAdmissionOutcome::Admitted(frontier))
    }
    async fn execute_tokens(
        &self,
        work: &[ExecutionWork<'_>],
    ) -> ferrum_types::Result<ExecutionBatchOutcome> {
        assert_eq!(work.len(), 1);
        self.observed.execute.fetch_add(1, Ordering::SeqCst);
        let mut state = self.state.lock().unwrap();
        assert_eq!(state.frontier.as_ref(), Some(work[0].expected_frontier()));
        let batch = self.issuer.batch(work, self.info.vocab_size)?;
        match self.fault {
            Fault::NotSubmitted => Ok(batch.not_submitted(NoSubmissionReason::UnsupportedShape)),
            Fault::Rejected => Ok(batch.not_submitted(NoSubmissionReason::Rejected(
                FerrumError::device("original fixture device failure"),
            ))),
            Fault::Unknown => {
                let key = self.issuer.recovery_key(work[0].sequence())?;
                state.recovery = Some(key.clone());
                let participant = batch
                    .participant(
                        0,
                        ExecutionProgress::Unknown {
                            last_known: LastKnownExecutionStage::Submitted,
                            known_executed_range: None,
                        },
                        ContinuationDisposition::Held(self.held(key)?),
                        0,
                    )
                    .unwrap();
                Ok(batch
                    .interrupted(
                        vec![],
                        vec![participant],
                        vec![FerrumError::backend("original unknown fixture submission").into()],
                    )
                    .unwrap())
            }
        }
    }
    async fn finish_tokens(
        &self,
        sequence: &ExecutorSequence,
        _: ExecutionEndReason,
    ) -> ferrum_types::Result<ExecutionCloseOutcome> {
        self.observed.finish.fetch_add(1, Ordering::SeqCst);
        self.close(sequence)
    }
    fn cancel_tokens(
        &self,
        sequence: &ExecutorSequence,
    ) -> ferrum_types::Result<ExecutionCloseOutcome> {
        self.observed.cancel.fetch_add(1, Ordering::SeqCst);
        self.close(sequence)
    }
}

#[tokio::test]
async fn interrupted_unknown_keeps_both_original_held_observers_without_replay_or_end() {
    let (executor, release, observed) = ContractExecutor::new(Fault::Unknown);
    let owner = ExecutorOwner::new(Box::new(executor));
    let scope = owner.scope().unwrap();
    let host = host();
    let mut failure = execute(
        &scope,
        &RequestId::new(),
        &prompt(),
        generation(8),
        2,
        &host,
        &|| false,
    )
    .await
    .unwrap_err();
    assert_eq!(failure.finish, FinishReason::Error);
    assert!(matches!(
        &failure.error,
        FerrumError::Backend { message } if message == "original unknown fixture submission"
    ));
    let frontier = failure.frontier.as_ref().unwrap();
    assert_eq!(frontier.consumed_tokens(), 0);
    assert!(frontier.sequence().observed_end().is_none());
    assert!(failure.history.as_ref().unwrap().ids().is_empty());
    let Some(ExecutionBatchOutcome::Interrupted(interrupted)) = &failure.outcome else {
        panic!("original unknown batch must remain owned");
    };
    assert_eq!(interrupted.remaining()[0].sequence(), frontier.sequence());
    assert!(matches!(
        interrupted.remaining()[0].progress(),
        ExecutionProgress::Unknown { .. }
    ));
    let ContinuationDisposition::Held(original) = interrupted.remaining()[0].continuation() else {
        panic!("original continuation must stay Held");
    };
    let ExecutionCloseDisposition::Held(close) = &failure.close.as_ref().unwrap().disposition
    else {
        panic!("cancel must preserve the original retained work");
    };
    assert_eq!(close.key(), original.key());
    assert_eq!(observed.execute.load(Ordering::SeqCst), 1);
    assert_eq!(observed.cancel.load(Ordering::SeqCst), 1);
    assert_eq!(observed.finish.load(Ordering::SeqCst), 0);
    let close = failure.close.take().unwrap();
    let ExecutionCloseDisposition::Held(held) = close.disposition else {
        unreachable!()
    };
    release.send_replace(true);
    let terminal = held.wait().await.unwrap();
    assert_eq!(
        terminal.ended.sequence(),
        failure.frontier.as_ref().unwrap().sequence()
    );
    assert!(terminal.error.is_none());
    drop(failure);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    drop(scope);
    retire(owner).await;
}

#[tokio::test]
async fn not_submitted_keeps_its_exact_reason_and_never_implicitly_replays() {
    let (executor, _, observed) = ContractExecutor::new(Fault::NotSubmitted);
    let owner = ExecutorOwner::new(Box::new(executor));
    let scope = owner.scope().unwrap();
    let host = host();
    let failure = execute(
        &scope,
        &RequestId::new(),
        &prompt(),
        generation(8),
        2,
        &host,
        &|| false,
    )
    .await
    .unwrap_err();
    let Some(ExecutionBatchOutcome::NotSubmitted(original)) = &failure.outcome else {
        panic!("original no-submit certificate must remain owned");
    };
    assert!(matches!(
        original.reason(),
        NoSubmissionReason::UnsupportedShape
    ));
    assert!(failure.error.to_string().contains("unsupported shape"));
    assert_eq!(observed.execute.load(Ordering::SeqCst), 1);
    assert_eq!(observed.cancel.load(Ordering::SeqCst), 1);
    assert_eq!(observed.finish.load(Ordering::SeqCst), 0);
    assert_eq!(failure.frontier.as_ref().unwrap().consumed_tokens(), 0);
    assert!(matches!(
        failure.close.as_ref().unwrap().disposition,
        ExecutionCloseDisposition::Ended(_)
    ));
    drop(failure);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    drop(scope);
    retire(owner).await;
}

#[tokio::test]
async fn rejected_wave_exposes_original_typed_error_and_keeps_its_no_submit_outcome() {
    let (executor, _, observed) = ContractExecutor::new(Fault::Rejected);
    let owner = ExecutorOwner::new(Box::new(executor));
    let scope = owner.scope().unwrap();
    let host = host();
    let failure = execute(
        &scope,
        &RequestId::new(),
        &prompt(),
        generation(8),
        2,
        &host,
        &|| false,
    )
    .await
    .unwrap_err();
    assert!(matches!(
        &failure.error,
        FerrumError::Device { message } if message == "original fixture device failure"
    ));
    let Some(ExecutionBatchOutcome::NotSubmitted(original)) = &failure.outcome else {
        panic!("the original rejected outcome must remain owned");
    };
    assert!(matches!(
        original.reason(),
        NoSubmissionReason::Rejected(FerrumError::Device { message })
            if message == "original fixture device failure"
    ));
    assert_eq!(observed.execute.load(Ordering::SeqCst), 1);
    assert_eq!(observed.cancel.load(Ordering::SeqCst), 1);
    assert_eq!(observed.finish.load(Ordering::SeqCst), 0);
    assert!(failure.history.as_ref().unwrap().ids().is_empty());
    assert!(matches!(
        failure.close.as_ref().unwrap().disposition,
        ExecutionCloseDisposition::Ended(_)
    ));
    assert!(host.usage().unwrap().committed_bytes() > 0);
    drop(failure);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    drop(scope);
    retire(owner).await;
}

#[test]
fn contract_violation_diagnostics_preserve_the_original_typed_error() {
    let outcome = ExecutionBatchOutcome::ContractViolation(ExecutionContractViolation::new(
        FerrumError::device("original contract diagnostic"),
        (vec![], vec![], vec![]),
    ));
    assert!(matches!(
        incomplete_wave_error(&outcome),
        FerrumError::Device { message } if message == "original contract diagnostic"
    ));
    let ExecutionBatchOutcome::ContractViolation(original) = &outcome else {
        unreachable!()
    };
    assert!(matches!(
        &original.error,
        FerrumError::Device { message } if message == "original contract diagnostic"
    ));
}
