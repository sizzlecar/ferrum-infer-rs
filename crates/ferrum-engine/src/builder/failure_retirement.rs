//! Builder failures observe the same original CPU model's retirement.

use super::*;
use ferrum_execution::{ExecutorShutdownPermit, ExecutorShutdownPhase};
use ferrum_models::test_support::{tiny_llama_executor_owned, TinyLlamaConfig};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

async fn retired(observer: &ExecutorShutdownObserver) {
    let mut observer = observer.clone();
    tokio::time::timeout(Duration::from_secs(5), async {
        while observer.snapshot().phase() != ExecutorShutdownPhase::Retired {
            observer.changed().await;
        }
    })
    .await
    .expect("original CPU model cleanup must complete");
}

#[tokio::test]
async fn builder_failure_before_factories_keeps_original_owned_cpu_cleanup() {
    let owner = ExecutorOwner::new(tiny_llama_executor_owned(&TinyLlamaConfig::default()));
    let peer = owner.scope().unwrap();
    let original = owner.shutdown_observer();
    let mut config = EngineConfig::default();
    config
        .backend
        .backend_options
        .insert("spec_n".to_owned(), 2.into());
    let error = EngineBuilder::new(config)
        .with_owned_executor(owner)
        .unwrap()
        .build()
        .await
        .err()
        .expect("retired option rejects before component factories");
    assert!(error.cause().to_string().contains("retired experimental"));
    let cleanup = error.cleanup().expect("original owner observation").clone();
    assert_eq!(cleanup.snapshot().phase(), ExecutorShutdownPhase::Draining);
    drop(error);
    assert_eq!(original.snapshot().phase(), ExecutorShutdownPhase::Draining);
    drop(peer);
    retired(&cleanup).await;
    cleanup.snapshot().result().unwrap();
    assert_eq!(original.snapshot().phase(), ExecutorShutdownPhase::Retired);
}

struct StartupFailure {
    scope: ExecutorScope,
    owner: ExecutorOwner,
    starts: Arc<AtomicUsize>,
    closes: Arc<AtomicUsize>,
}

#[async_trait::async_trait]
impl ModelExecutor for StartupFailure {
    fn info(&self) -> &ferrum_types::ModelInfo {
        self.scope.info()
    }
    fn capabilities(&self) -> ferrum_execution::model_executor::ExecutorCapabilities {
        self.scope.capabilities()
    }
    fn status(&self) -> ferrum_execution::model_executor::ExecutorStatus {
        self.scope.status()
    }
    async fn prepare_startup(&self) -> Result<()> {
        self.scope.prepare_startup().await?;
        self.starts.fetch_add(1, Ordering::SeqCst);
        Err(FerrumError::backend("original startup rejection"))
    }
    fn begin_shutdown(
        self: Arc<Self>,
        permit: &ExecutorShutdownPermit,
    ) -> ExecutorShutdownObserver {
        if let Err(error) = permit.check(self.as_ref()) {
            return ExecutorShutdownObserver::retained(Arc::new(self), error);
        }
        match Arc::try_unwrap(self) {
            Ok(original) => {
                original.closes.fetch_add(1, Ordering::SeqCst);
                drop(original.scope);
                original.owner.begin_shutdown()
            }
            Err(original) => ExecutorShutdownObserver::retained(
                Arc::new(original),
                FerrumError::internal("startup failure still has execution scopes"),
            ),
        }
    }
}

fn startup_failure_fixture() -> (
    EngineBuilder,
    ExecutorScope,
    ExecutorShutdownObserver,
    Arc<AtomicUsize>,
    Arc<AtomicUsize>,
) {
    let owner = ExecutorOwner::new(tiny_llama_executor_owned(&TinyLlamaConfig::default()));
    let peer = owner.scope().unwrap();
    let original = owner.shutdown_observer();
    let starts = Arc::new(AtomicUsize::new(0));
    let closes = Arc::new(AtomicUsize::new(0));
    let wrapper = StartupFailure {
        scope: owner.scope().unwrap(),
        owner,
        starts: Arc::clone(&starts),
        closes: Arc::clone(&closes),
    };
    let builder = EngineBuilder::new(EngineConfig::default())
        .with_owned_executor(ExecutorOwner::new(Box::new(wrapper)))
        .unwrap();
    (builder, peer, original, starts, closes)
}

#[tokio::test]
async fn builder_startup_failure_waits_for_original_first_close_report() {
    let (builder, peer, original, starts, closes) = startup_failure_fixture();
    let mut build = Box::pin(builder.build());
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            tokio::select! {
                _ = build.as_mut() => panic!("original peer still blocks the first close report"),
                _ = tokio::task::yield_now() => {
                    if starts.load(Ordering::SeqCst) == 1 { break; }
                }
            }
        }
    })
    .await
    .unwrap();
    assert!(futures::poll!(build.as_mut()).is_pending());
    assert_eq!(original.snapshot().phase(), ExecutorShutdownPhase::Draining);
    drop(peer);
    let error = tokio::time::timeout(Duration::from_secs(5), build)
        .await
        .unwrap()
        .err()
        .expect("startup rejects publication");
    assert!(error
        .cause()
        .to_string()
        .contains("original startup rejection"));
    let cleanup = error
        .cleanup()
        .expect("complete engine cleanup observation")
        .clone();
    drop(error);
    retired(&cleanup).await;
    cleanup.snapshot().result().unwrap();
    assert_eq!(original.snapshot().phase(), ExecutorShutdownPhase::Retired);
    assert_eq!(starts.load(Ordering::SeqCst), 1);
    assert_eq!(closes.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn cancelled_builder_startup_wait_keeps_original_cpu_close_service() {
    let (builder, peer, original, starts, closes) = startup_failure_fixture();
    let mut build = Box::pin(builder.build());
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            tokio::select! {
                _ = build.as_mut() => panic!("original peer still blocks the first close report"),
                _ = tokio::task::yield_now() => {
                    if starts.load(Ordering::SeqCst) == 1 { break; }
                }
            }
        }
    })
    .await
    .unwrap();
    assert!(futures::poll!(build.as_mut()).is_pending());
    drop(build);
    assert_eq!(original.snapshot().phase(), ExecutorShutdownPhase::Draining);
    drop(peer);
    retired(&original).await;
    original.snapshot().result().unwrap();
    assert_eq!(starts.load(Ordering::SeqCst), 1);
    assert_eq!(closes.load(Ordering::SeqCst), 1);
}
