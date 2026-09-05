//! Only model logits and opaque cache storage are simulated. Stop detection,
//! sampling, finish reasons, and response chunks belong to the production engine.
use super::*;
use ferrum_interfaces::{
    model_executor::{
        DecodeInput, DecodeOutput, ExecutionResourceAuthority, ExecutorAdmissionEpochs,
        ExecutorCapabilities, ExecutorPrefillAdmission, ExecutorPrefillAdmissionDecision,
        ExecutorPrefillAdmissionReceipt, ExecutorSamplingOutput, ExecutorSequenceCompletion,
        ExecutorStatus, PlanRuntimeBatchDecodeOutcome, PlanRuntimeDecodeInput,
        PlanRuntimeDecodeOutput, PlanRuntimePrefillCompletion, PlanRuntimePrefillInput,
        PlanRuntimePrefillOutcome, PlanRuntimePrefillOutput, PlanRuntimeResourceSnapshot,
        PrefillInput, PrefillOutput,
    },
    KvCacheHandle, ModelExecutor,
};
use ferrum_testkit::{MockKvCacheHandle, MockModelExecutor};
use ferrum_types::{ExecutorAdmissionLimits, ModelInfo, RequestId, Result};
use std::{collections::HashSet, num::NonZeroU64, sync::atomic::AtomicU64};

struct Cursor {
    request_id: RequestId,
    next: usize,
    committed: usize,
}

pub(super) struct ScriptedExecutor {
    metadata: MockModelExecutor,
    script: Vec<TokenId>,
    admitted: Mutex<HashSet<RequestId>>,
    active: Mutex<HashMap<String, Cursor>>,
    pub prompt_tokens: AtomicUsize,
    pub generated_tokens: AtomicUsize,
    releases: AtomicU64,
    completed: Mutex<Vec<ExecutorSequenceCompletion>>,
}

impl ScriptedExecutor {
    pub fn new(vocab_size: usize, script: Vec<TokenId>) -> Self {
        assert!(!script.is_empty());
        Self {
            metadata: MockModelExecutor::instant(vocab_size),
            script,
            admitted: Mutex::new(HashSet::new()),
            active: Mutex::new(HashMap::new()),
            prompt_tokens: AtomicUsize::new(0),
            generated_tokens: AtomicUsize::new(0),
            releases: AtomicU64::new(0),
            completed: Mutex::new(Vec::new()),
        }
    }

    fn logits(&self, index: usize) -> Result<Vec<f32>> {
        let token = self.script.get(index).ok_or_else(|| {
            FerrumError::backend("script exhausted: engine did not stop on the final token")
        })?;
        let mut logits = vec![f32::NEG_INFINITY; self.info().vocab_size];
        logits[token.get() as usize] = 1.0;
        self.generated_tokens.fetch_add(1, Ordering::Relaxed);
        Ok(logits)
    }

    pub fn assert_released(&self) {
        assert!(self.admitted.lock().unwrap().is_empty());
        assert!(self.active.lock().unwrap().is_empty());
    }

    pub fn assert_completed(&self) {
        let completed = self.completed.lock().unwrap();
        assert_eq!(
            completed.len(),
            1,
            "one request completed inside the engine"
        );
        assert_eq!(
            completed[0].input_tokens(),
            self.prompt_tokens.load(Ordering::Relaxed) as u64
        );
        assert_eq!(
            completed[0].output_tokens(),
            self.generated_tokens.load(Ordering::Relaxed) as u64
        );
    }
}

#[async_trait]
impl ModelExecutor for ScriptedExecutor {
    fn info(&self) -> &ModelInfo {
        self.metadata.info()
    }

    fn capabilities(&self) -> ExecutorCapabilities {
        self.metadata.capabilities()
    }

    fn status(&self) -> ExecutorStatus {
        self.metadata.status()
    }

    fn execution_resource_authority(&self) -> ExecutionResourceAuthority {
        ExecutionResourceAuthority::PlanRuntime
    }

    fn admission_limits(&self) -> Result<Option<ExecutorAdmissionLimits>> {
        ExecutorAdmissionLimits::new(1, 256)
            .map(Some)
            .map_err(FerrumError::internal)
    }

    fn plan_runtime_resource_snapshot(&self) -> Result<Option<PlanRuntimeResourceSnapshot>> {
        PlanRuntimeResourceSnapshot::new(0, 0, 0, 0, 0, 0, 0, 0, 0).map(Some)
    }

    fn execution_capacity_epochs(&self) -> Result<Option<ExecutorAdmissionEpochs>> {
        Ok(Some(ExecutorAdmissionEpochs::new(
            NonZeroU64::new(1).unwrap(),
            self.releases.load(Ordering::Relaxed),
            0,
        )))
    }

    fn try_admit_prefill(
        &self,
        input: ExecutorPrefillAdmission<'_>,
    ) -> Result<ExecutorPrefillAdmissionDecision> {
        input.validate()?;
        assert!(self
            .admitted
            .lock()
            .unwrap()
            .insert(input.request_id.clone()));
        Ok(ExecutorPrefillAdmissionDecision::Admitted(
            ExecutorPrefillAdmissionReceipt {
                request_id: input.request_id.clone(),
            },
        ))
    }

    fn cancel_prefill_admission(&self, request_id: &RequestId) -> bool {
        self.admitted.lock().unwrap().remove(request_id)
    }

    async fn prefill(&self, _: &PrefillInput) -> Result<PrefillOutput> {
        Err(FerrumError::unsupported("test must use typed plan prefill"))
    }

    async fn decode(&self, _: &DecodeInput) -> Result<DecodeOutput> {
        Err(FerrumError::unsupported("test must use typed plan decode"))
    }

    async fn plan_runtime_prefill_with_capacity(
        &self,
        input: &PlanRuntimePrefillInput,
    ) -> Result<PlanRuntimePrefillOutcome> {
        assert!(self.admitted.lock().unwrap().contains(&input.request_id));
        let cache: Arc<dyn KvCacheHandle> = Arc::new(MockKvCacheHandle::new(
            input.request_id.clone(),
            1,
            input.chunk.end(),
        ));
        let output = if input.chunk.is_final() {
            self.prompt_tokens
                .store(input.input_tokens.len(), Ordering::Relaxed);
            assert!(self.admitted.lock().unwrap().remove(&input.request_id));
            assert!(self
                .active
                .lock()
                .unwrap()
                .insert(
                    cache.cache_id(),
                    Cursor {
                        request_id: input.request_id.clone(),
                        next: 1,
                        committed: input.chunk.end(),
                    }
                )
                .is_none());
            PlanRuntimePrefillOutput::final_logits(
                input.request_id.clone(),
                input.chunk.end(),
                self.logits(0)?,
                cache,
            )?
        } else {
            PlanRuntimePrefillOutput::intermediate(
                input.request_id.clone(),
                input.chunk.end(),
                cache,
            )
        };
        Ok(PlanRuntimePrefillOutcome::Completed(
            PlanRuntimePrefillCompletion::exact(output, input.chunk),
        ))
    }

    async fn plan_runtime_batch_decode_with_capacity(
        &self,
        inputs: &[PlanRuntimeDecodeInput],
    ) -> Result<PlanRuntimeBatchDecodeOutcome> {
        let mut active = self.active.lock().unwrap();
        let outputs = inputs
            .iter()
            .map(|input| {
                let cursor = active
                    .get_mut(&input.kv_cache.cache_id())
                    .expect("active cache");
                assert_eq!(cursor.request_id, input.request_id);
                assert_eq!(input.input_token, self.script[cursor.next - 1]);
                let logits = self.logits(cursor.next)?;
                cursor.next += 1;
                cursor.committed += 1;
                let cache = Arc::new(MockKvCacheHandle::new(
                    input.request_id.clone(),
                    1,
                    cursor.committed,
                ));
                Ok(PlanRuntimeDecodeOutput::new(
                    ExecutorSamplingOutput::FullLogits(logits),
                    cache,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(PlanRuntimeBatchDecodeOutcome::Completed(outputs))
    }

    fn complete_cache(&self, completion: ExecutorSequenceCompletion) -> Result<()> {
        {
            let active = self.active.lock().unwrap();
            let cursor = active
                .get(completion.cache_id())
                .expect("completed active cache");
            assert_eq!(&cursor.request_id, completion.request_id());
            assert_eq!(cursor.next as u64, completion.output_tokens());
        }
        self.release_cache(completion.cache_id());
        self.completed.lock().unwrap().push(completion);
        Ok(())
    }

    fn release_cache(&self, cache_id: &str) {
        if self.active.lock().unwrap().remove(cache_id).is_some() {
            self.releases.fetch_add(1, Ordering::Relaxed);
        }
    }
}
