use super::*;
use ferrum_execution::model_executor::{ExecutorCapabilities, ExecutorStatus};
use ferrum_execution::{ExecutorShutdownPermit, ExecutorShutdownPhase};
use ferrum_types::{
    DeviceMemorySnapshot, StartupMemoryPlan, StartupMemoryRequest, StartupResourceLimits,
};
use std::sync::atomic::{AtomicUsize, Ordering};

struct SelectedExecutor {
    inner: ferrum_testkit::MockModelExecutor,
    plan: StartupMemoryPlan,
    startup_calls: Arc<AtomicUsize>,
}

#[async_trait::async_trait]
impl ModelExecutor for SelectedExecutor {
    fn info(&self) -> &ferrum_types::ModelInfo {
        self.inner.info()
    }
    fn capabilities(&self) -> ExecutorCapabilities {
        self.inner.capabilities()
    }
    fn status(&self) -> ExecutorStatus {
        self.inner.status()
    }
    fn startup_memory_plan(&self) -> Option<&StartupMemoryPlan> {
        Some(&self.plan)
    }
    async fn prepare_startup(&self) -> Result<()> {
        self.startup_calls.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
    fn begin_shutdown(
        self: Arc<Self>,
        permit: &ExecutorShutdownPermit,
    ) -> ExecutorShutdownObserver {
        if let Err(error) = permit.check(self.as_ref()) {
            return ExecutorShutdownObserver::retained(Arc::new(self), error);
        }
        match Arc::try_unwrap(self) {
            Ok(original) => ExecutorOwner::new(Box::new(original.inner)).begin_shutdown(),
            Err(original) => ExecutorShutdownObserver::retained(
                Arc::new(original),
                FerrumError::internal("fixture executor still retained"),
            ),
        }
    }
}

#[tokio::test]
async fn binding_sees_actual_executor_and_selected_memory_before_startup() {
    let starts = Arc::new(AtomicUsize::new(0));
    let selected = StartupResourceLimits {
        context_tokens: 64,
        max_sequences: 2,
        max_batch_tokens: 8,
    };
    let executor = ExecutorOwner::new(Box::new(SelectedExecutor {
        inner: ferrum_testkit::MockModelExecutor::instant(128),
        startup_calls: Arc::clone(&starts),
        plan: StartupMemoryPlan {
            request: StartupMemoryRequest {
                device: DeviceMemorySnapshot {
                    capacity_bytes: 1024 * 1024,
                    available_bytes: 1024 * 1024,
                    source: "fixture actual owner".into(),
                },
                usable_capacity_bytes: 512 * 1024,
                context_is_explicit: false,
                sequences_is_explicit: false,
                batch_is_explicit: false,
            },
            requested: StartupResourceLimits {
                context_tokens: 128,
                max_sequences: 4,
                max_batch_tokens: 16,
            },
            selected,
            context_peak_bytes: 1024,
            decode_peak_bytes: 1024,
            reasons: vec![],
        },
    }));
    let collector = ferrum_slo::ReferenceEstimator::spawn(Default::default()).unwrap();
    let observer = collector.observer();
    let called = Arc::new(AtomicUsize::new(0));
    let recorded = Arc::clone(&called);
    let startup = Arc::clone(&starts);
    let engine = EngineBuilder::new(EngineConfig::default())
        .with_owned_executor(executor)
        .unwrap()
        .with_execution_feedback_factory(move |config, actual| {
            assert_eq!(actual.info().model_id.to_string(), "mock-model");
            assert_eq!(config.runtime.max_model_len, Some(64));
            assert_eq!(config.batching.max_num_batched_tokens, 8);
            assert_eq!(config.scheduler.max_running_requests, 2);
            assert_eq!(startup.load(Ordering::Relaxed), 0);
            recorded.fetch_add(1, Ordering::Relaxed);
            Ok(ExecutionFeedbackBinding {
                execution_observer: observer,
                cost_source: None,
                stopping_observer: None,
            })
        })
        .build()
        .await
        .unwrap();
    assert_eq!(called.load(Ordering::Relaxed), 1);
    assert_eq!(starts.load(Ordering::Relaxed), 1);
    engine.shutdown().await.unwrap();
    engine.wait_for_shutdown_completion().await.unwrap();
    let _ = collector.shutdown().await.unwrap();
}

#[tokio::test]
async fn rejected_binding_retains_the_original_executor_cleanup() {
    let executor = ExecutorOwner::new(Box::new(ferrum_models::executor::StubModelExecutor::new(
        "actual-stub",
        128,
    )));
    let original = executor.shutdown_observer();
    let error = EngineBuilder::new(EngineConfig::default())
        .with_owned_executor(executor)
        .unwrap()
        .with_execution_feedback_factory(|_, _| Err(FerrumError::config("identity mismatch")))
        .build()
        .await
        .err()
        .expect("binding must reject before engine publication");
    assert!(error.to_string().contains("identity mismatch"));
    error
        .cleanup()
        .expect("original cleanup")
        .clone()
        .wait()
        .await
        .result()
        .unwrap();
    assert_eq!(original.snapshot().phase(), ExecutorShutdownPhase::Retired);
}

#[tokio::test]
async fn factory_cannot_override_direct_observer_injection() {
    let collector = ferrum_slo::ReferenceEstimator::spawn(Default::default()).unwrap();
    let error = EngineBuilder::new(EngineConfig::default())
        .with_execution_attempt_observer(collector.observer())
        .with_execution_feedback_factory(|_, _| panic!("ambiguous factory must not execute"))
        .build()
        .await
        .err()
        .expect("ambiguous injection must fail");
    assert!(error.to_string().contains("cannot replace"));
    let _ = collector.shutdown().await.unwrap();
}
