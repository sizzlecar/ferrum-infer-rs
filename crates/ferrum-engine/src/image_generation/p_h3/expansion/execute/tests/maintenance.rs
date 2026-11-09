//! Controlled admission protocol observations over actual managed Host backing.
//! These project changing allocator epochs, without claiming native allocation,
//! model numerical execution or GPU qualification.
use super::*;
use async_trait::async_trait;
use ferrum_execution::{
    executor_lifecycle::{ExecutorShutdownObserver, ExecutorShutdownPermit},
    model_executor::{
        ExecutorCapabilities, ExecutorPrefillMaintenanceBlocker,
        ExecutorPrefillMaintenanceDeferral, ExecutorPrefillMaintenanceStage, ExecutorStatus,
    },
    ModelExecutor,
};
use ferrum_runtime::{
    CapacityAvailabilityEpoch, CapacityAvailabilitySource, CapacityShortfallKind,
    CapacityWaitCondition,
};
use ferrum_types::ModelInfo;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc, Mutex,
};

#[derive(Default)]
struct Observed {
    admissions: AtomicUsize,
    maintenance: AtomicUsize,
    request: Mutex<Option<RequestId>>,
}
struct MaintenanceExecutor {
    metadata: MockModelExecutor,
    observed: Arc<Observed>,
    changing: bool,
}
fn epochs(epoch: usize) -> ExecutorAdmissionEpochs {
    ExecutorAdmissionEpochs::new(
        std::num::NonZeroU64::new(71).unwrap(),
        0,
        u64::try_from(epoch).unwrap(),
    )
}
impl MaintenanceExecutor {
    fn new(changing: bool) -> (Self, Arc<Observed>) {
        let observed = Arc::new(Observed::default());
        (
            Self {
                metadata: MockModelExecutor::instant(64),
                observed: observed.clone(),
                changing,
            },
            observed,
        )
    }
}
#[async_trait]
impl ModelExecutor for MaintenanceExecutor {
    fn info(&self) -> &ModelInfo {
        self.metadata.info()
    }
    fn capabilities(&self) -> ExecutorCapabilities {
        self.metadata.capabilities()
    }
    fn status(&self) -> ExecutorStatus {
        self.metadata.status()
    }
    fn begin_shutdown(
        self: Arc<Self>,
        permit: &ExecutorShutdownPermit,
    ) -> ExecutorShutdownObserver {
        permit.check(self.as_ref()).unwrap();
        let original = Arc::try_unwrap(self)
            .ok()
            .expect("controlled admission has no live scope at retirement");
        // No frontier or native allocation was issued by this fixture.
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
        let ordinal = self.observed.admissions.fetch_add(1, Ordering::SeqCst) + 1;
        let request = admission.context.request_id;
        let mut original = self.observed.request.lock().unwrap();
        if let Some(original) = original.as_ref() {
            assert_eq!(original, request);
        } else {
            *original = Some(request.clone());
        }
        let epoch = if self.changing { ordinal } else { 1 };
        Ok(TokenAdmissionOutcome::MaintenanceDeferred(
            ExecutorPrefillMaintenanceDeferral::new(
                request.clone(),
                epochs(epoch),
                CapacityWaitCondition::from_observation(
                    71,
                    vec![CapacityAvailabilityEpoch::new(
                        CapacityAvailabilitySource::ActiveSequenceSlots,
                        u64::try_from(epoch).unwrap(),
                    )
                    .unwrap()],
                )
                .unwrap(),
                ExecutorPrefillMaintenanceStage::PhysicalBacking,
                vec![ExecutorPrefillMaintenanceBlocker::Capacity {
                    domain_id: Some(1),
                    kind: CapacityShortfallKind::BackingGrowthRequired,
                    requested: 64,
                    available: 0,
                    current_total: 0,
                    maximum_total: 64,
                }],
            )?,
        ))
    }
    fn maintain_prefill_backing(
        &self,
        request: &RequestId,
    ) -> ferrum_types::Result<ExecutorPrefillMaintenanceOutcome> {
        assert_eq!(
            self.observed.request.lock().unwrap().as_ref(),
            Some(request)
        );
        let ordinal = self.observed.maintenance.fetch_add(1, Ordering::SeqCst) + 1;
        Ok(ExecutorPrefillMaintenanceOutcome::RetryAdmission {
            current: epochs(ordinal + 1),
        })
    }
}

fn original_deferred(failure: &ExpansionExecutionFailure) -> &ExecutorPrefillMaintenanceDeferral {
    let Some(TokenAdmissionOutcome::MaintenanceDeferred(deferred)) = &failure.admission else {
        panic!("original admission deferral must remain owned");
    };
    deferred
}

#[tokio::test]
async fn changing_epochs_return_each_transition_and_cancel_keeps_original_history_and_facts() {
    let (model, observed) = MaintenanceExecutor::new(true);
    let owner = ExecutorOwner::new(Box::new(model));
    let scope = owner.scope().unwrap();
    let host = host();
    let required = history_backing_bytes(&host, 6).unwrap();
    let grant = host
        .reserve_scope(&[(HostMemoryClass::Output, required)])
        .unwrap();
    let request = RequestId::new();
    let mut admission = ExpansionAdmission::new(
        &scope,
        &request,
        &prompt(),
        generation(6),
        2,
        &grant.account(),
    )
    .unwrap();
    let original = admission.history.buffer.as_ref().as_ptr();
    assert_eq!(observed.admissions.load(Ordering::SeqCst), 0);
    for ordinal in 1..=3 {
        let ExpansionAdmissionStep::Pending(next) = admission
            .advance(&scope, &prompt(), &|| false)
            .await
            .unwrap()
        else {
            panic!("a maintenance transition cannot issue a token frontier");
        };
        admission = next;
        assert_eq!(observed.admissions.load(Ordering::SeqCst), ordinal);
        assert_eq!(observed.maintenance.load(Ordering::SeqCst), ordinal);
        assert_eq!(admission.history.buffer.as_ref().as_ptr(), original);
        assert!(admission.history.ids().is_empty());
        assert_eq!(host.usage().unwrap().allocated_bytes(), required);
        assert_eq!(admission.prior_maintenance, Some(epochs(ordinal)));
    }
    let failure = admission
        .advance(&scope, &prompt(), &|| true)
        .await
        .err()
        .expect("new cancellation must be observed before another admission");
    assert_eq!(failure.finish, FinishReason::Cancelled);
    assert_eq!(observed.admissions.load(Ordering::SeqCst), 3);
    assert_eq!(observed.maintenance.load(Ordering::SeqCst), 3);
    assert_eq!(original_deferred(&failure).request_id(), &request);
    assert_eq!(original_deferred(&failure).observed(), epochs(3));
    assert!(matches!(
        failure.maintenance,
        Some(ExecutorPrefillMaintenanceOutcome::RetryAdmission { current }) if current == epochs(4)
    ));
    assert!(failure.frontier.is_none());
    assert!(failure.outcome.is_none());
    assert_eq!(
        failure.history.as_ref().unwrap().buffer.as_ref().as_ptr(),
        original
    );
    grant.close().unwrap();
    assert_eq!(host.usage().unwrap().committed_bytes(), required);
    drop(failure);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    drop(scope);
    retire(owner).await;
}

#[tokio::test]
async fn unchanged_epoch_rejection_keeps_latest_deferral_and_does_not_repeat_maintenance() {
    let (model, observed) = MaintenanceExecutor::new(false);
    let owner = ExecutorOwner::new(Box::new(model));
    let scope = owner.scope().unwrap();
    let host = host();
    let admission = ExpansionAdmission::new(
        &scope,
        &RequestId::new(),
        &prompt(),
        generation(6),
        2,
        &host,
    )
    .unwrap();
    let ExpansionAdmissionStep::Pending(admission) = admission
        .advance(&scope, &prompt(), &|| false)
        .await
        .unwrap()
    else {
        panic!("first observed maintenance should return to its caller");
    };
    let failure = admission
        .advance(&scope, &prompt(), &|| false)
        .await
        .err()
        .expect("a duplicate capacity observation cannot authorize another maintenance");
    assert_eq!(failure.finish, FinishReason::Error);
    assert_eq!(original_deferred(&failure).observed(), epochs(1));
    assert_eq!(observed.admissions.load(Ordering::SeqCst), 2);
    assert_eq!(observed.maintenance.load(Ordering::SeqCst), 1);
    assert!(failure.history.as_ref().unwrap().ids().is_empty());
    drop(failure);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    drop(scope);
    retire(owner).await;
}

#[tokio::test]
async fn borrowed_begin_wrapper_reads_new_cancel_between_identical_state_machine_transitions() {
    let (model, observed) = MaintenanceExecutor::new(true);
    let owner = ExecutorOwner::new(Box::new(model));
    let scope = owner.scope().unwrap();
    let host = host();
    let failure = begin(
        &scope,
        &RequestId::new(),
        &prompt(),
        generation(6),
        2,
        &host,
        &|| observed.maintenance.load(Ordering::SeqCst) > 0,
    )
    .await
    .err()
    .expect("borrowed wrapper must observe cancellation before the next probe");
    assert_eq!(failure.finish, FinishReason::Cancelled);
    assert_eq!(observed.admissions.load(Ordering::SeqCst), 1);
    assert_eq!(observed.maintenance.load(Ordering::SeqCst), 1);
    assert_eq!(original_deferred(&failure).observed(), epochs(1));
    drop(failure);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    drop(scope);
    retire(owner).await;
}
