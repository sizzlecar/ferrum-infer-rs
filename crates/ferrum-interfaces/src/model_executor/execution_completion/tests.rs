use super::*;
use crate::model_executor::*;
use crate::ModelExecutor;
use ferrum_types::{FerrumError, ModelInfo, RequestId};
use std::sync::atomic::{AtomicUsize, Ordering};

struct CompletionOnly {
    completed: AtomicUsize,
    cancelled: AtomicUsize,
}
#[async_trait::async_trait]
impl ModelExecutor for CompletionOnly {
    fn info(&self) -> &ModelInfo {
        panic!("completion does not inspect model metadata")
    }
    fn capabilities(&self) -> ExecutorCapabilities {
        panic!("not an execution call")
    }
    fn status(&self) -> ExecutorStatus {
        panic!("not an execution call")
    }
    async fn prefill(&self, _: &PrefillInput) -> Result<PrefillOutput> {
        panic!("completion cannot replay prefill")
    }
    async fn decode(&self, _: &DecodeInput) -> Result<DecodeOutput> {
        panic!("completion cannot replay decode")
    }
    async fn complete_cache(&self, _: ExecutorSequenceCompletion) -> Result<()> {
        self.completed.fetch_add(1, Ordering::Relaxed);
        Err(FerrumError::backend("actual completion failure"))
    }
    fn cancel_prefill_admission(&self, _: &RequestId) -> bool {
        self.cancelled.fetch_add(1, Ordering::Relaxed);
        true
    }
}

#[tokio::test]
async fn observed_completion_default_executes_once_and_preserves_failure_and_unknown_scope() {
    let executor = CompletionOnly {
        completed: AtomicUsize::new(0),
        cancelled: AtomicUsize::new(0),
    };
    let id = RequestId::new();
    let observed = executor
        .complete_cache_observed(
            ExecutorSequenceCompletion::new(id.clone(), "cache".into(), 4, 1).unwrap(),
        )
        .await;
    assert!(observed
        .result
        .unwrap_err()
        .to_string()
        .contains("actual completion failure"));
    assert_eq!(observed.work, ExecutorCompletionWork::Unknown);
    assert_eq!(executor.completed.load(Ordering::Relaxed), 1);
    let cancellation = executor.cancel_prefill_admission_observed(&id);
    assert!(cancellation.released);
    assert_eq!(cancellation.work, ExecutorCompletionWork::Unknown);
    assert_eq!(executor.cancelled.load(Ordering::Relaxed), 1);
}
