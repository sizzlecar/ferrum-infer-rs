use super::*;

struct SamplingExecutor {
    inner: MockModelExecutor,
    finished: AtomicU64,
    fail_finish: bool,
}

#[async_trait::async_trait]
impl ModelExecutor for SamplingExecutor {
    fn info(&self) -> &ferrum_types::ModelInfo {
        self.inner.info()
    }

    async fn prefill(&self, input: &PrefillInput) -> Result<PrefillOutput> {
        self.inner.prefill(input).await
    }

    async fn decode(&self, input: &DecodeInput) -> Result<DecodeOutput> {
        self.inner.decode(input).await
    }

    fn capabilities(&self) -> ExecutorCapabilities {
        self.inner.capabilities()
    }

    fn status(&self) -> ExecutorStatus {
        self.inner.status()
    }

    fn finish_device_memory_sampling(&self) -> Result<()> {
        self.finished.fetch_add(1, Ordering::SeqCst);
        if self.fail_finish {
            Err(FerrumError::device("sampler final write failed"))
        } else {
            Ok(())
        }
    }
}

fn engine(executor: Arc<SamplingExecutor>) -> ContinuousBatchEngine {
    let config = EngineConfig::default();
    let scheduler = Arc::new(ContinuousBatchScheduler::new(config.scheduler.clone()));
    ContinuousBatchEngine::new(
        config,
        scheduler,
        Arc::new(ferrum_testkit::MockTokenizer::new(128)),
        Arc::new(ferrum_testkit::MockSampler),
        Arc::new(ferrum_testkit::MockKvCacheManager::new(256)),
        executor,
        Arc::new(ferrum_testkit::MockTensorFactory),
    )
    .unwrap()
}

#[tokio::test]
async fn device_memory_shutdown_finalizes_after_background_failure_without_hiding_it() {
    for fail_finish in [false, true] {
        let executor = Arc::new(SamplingExecutor {
            inner: MockModelExecutor::instant(128),
            finished: AtomicU64::new(0),
            fail_finish,
        });
        let engine = engine(Arc::clone(&executor));
        // A real failed join exercises shutdown's error path before sampling.
        *engine.inner.background_loop.lock() = Some(tokio::spawn(async {
            panic!("background failure before sampler finalization");
        }));
        let error = engine.shutdown().await.unwrap_err();
        assert!(error
            .to_string()
            .contains("background iteration loop failed"));
        assert_eq!(executor.finished.load(Ordering::SeqCst), 1);
    }
}

#[tokio::test]
async fn device_memory_shutdown_reports_sampler_failure_after_other_work_drains() {
    let executor = Arc::new(SamplingExecutor {
        inner: MockModelExecutor::instant(128),
        finished: AtomicU64::new(0),
        fail_finish: true,
    });
    let engine = engine(Arc::clone(&executor));
    let error = engine.shutdown().await.unwrap_err();
    assert!(error.to_string().contains("sampler final write failed"));
    assert_eq!(executor.finished.load(Ordering::SeqCst), 1);
    assert!(!engine.inner.is_running.load(Ordering::SeqCst));
}
