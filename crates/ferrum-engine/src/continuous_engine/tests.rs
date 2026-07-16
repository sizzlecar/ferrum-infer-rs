use super::*;
use crate::recurrent_state::{InMemoryRecurrentStateConfig, InMemoryRecurrentStateManager};
use ferrum_interfaces::kv_cache::{
    AllocationRequest, CacheGcStats, CacheManagerStats, MemoryPressure,
};
use ferrum_interfaces::tokenizer::{TokenizerInfo, TokenizerType};
use ferrum_interfaces::{
    model_executor::{
        DecodeInput, DecodeOutput, ExecutorAdmissionEpochs, ExecutorCapabilities,
        ExecutorPrefillAdmission, ExecutorPrefillAdmissionDecision,
        ExecutorPrefillAdmissionReceipt, ExecutorPrefillMaintenanceBlocker,
        ExecutorPrefillMaintenanceDeferral, ExecutorPrefillMaintenanceOutcome,
        ExecutorPrefillMaintenanceStage, ExecutorStatus, PrefillInput, PrefillOutput, UnifiedBatch,
    },
    KvCacheHandle, KvCacheManager, ModelExecutor, RecurrentStateManager, RecurrentStateSpec,
    RecurrentStateTensorSpec, TensorRef,
};
use ferrum_models::{DecoderOnlyLLM, LlmExecutor, LlmRuntimeConfig};
use ferrum_testkit::{MockKvCacheManager, MockModelExecutor, MockTensor, MockTensorFactory};
use std::time::Duration;

struct ExecutorManagedAdmissionTestExecutor {
    inner: MockModelExecutor,
    retained: std::sync::Mutex<HashSet<RequestId>>,
    admission_probes: AtomicU64,
    prefill_calls: AtomicU64,
}

impl ExecutorManagedAdmissionTestExecutor {
    fn new(vocab_size: usize) -> Self {
        Self {
            inner: MockModelExecutor::instant(vocab_size),
            retained: std::sync::Mutex::new(HashSet::new()),
            admission_probes: AtomicU64::new(0),
            prefill_calls: AtomicU64::new(0),
        }
    }
}

#[async_trait::async_trait]
impl ModelExecutor for ExecutorManagedAdmissionTestExecutor {
    fn info(&self) -> &ferrum_types::ModelInfo {
        self.inner.info()
    }

    fn execution_resource_ownership(&self) -> ExecutionResourceOwnership {
        ExecutionResourceOwnership::ExecutorManaged
    }

    fn prefill_admission_epochs(&self) -> Result<Option<ExecutorAdmissionEpochs>> {
        Ok(Some(ExecutorAdmissionEpochs::new(
            std::num::NonZeroU64::new(41).unwrap(),
            0,
            0,
        )))
    }

    fn try_admit_prefill(
        &self,
        input: ExecutorPrefillAdmission<'_>,
    ) -> Result<ExecutorPrefillAdmissionDecision> {
        self.admission_probes.fetch_add(1, Ordering::Relaxed);
        let inserted = self
            .retained
            .lock()
            .expect("retained admission mutex poisoned")
            .insert(input.request_id.clone());
        if !inserted {
            return Err(FerrumError::already_exists("duplicate test admission"));
        }
        Ok(ExecutorPrefillAdmissionDecision::Admitted(
            ExecutorPrefillAdmissionReceipt {
                request_id: input.request_id.clone(),
            },
        ))
    }

    fn cancel_prefill_admission(&self, request_id: &RequestId) -> bool {
        self.retained
            .lock()
            .expect("retained admission mutex poisoned")
            .remove(request_id)
    }

    async fn prefill(&self, input: &PrefillInput) -> Result<PrefillOutput> {
        let request_id = input
            .request_id
            .as_ref()
            .ok_or_else(|| FerrumError::request_validation("missing request id"))?;
        if !self
            .retained
            .lock()
            .expect("retained admission mutex poisoned")
            .remove(request_id)
        {
            return Err(FerrumError::internal(
                "test prefill reached submit without retained admission",
            ));
        }
        self.prefill_calls.fetch_add(1, Ordering::Relaxed);
        self.inner.prefill(input).await
    }

    async fn decode(&self, input: &DecodeInput) -> Result<DecodeOutput> {
        self.inner.decode(input).await
    }

    fn release_cache(&self, cache_id: &str) {
        self.inner.release_cache(cache_id);
    }

    fn capabilities(&self) -> ExecutorCapabilities {
        self.inner.capabilities()
    }

    fn status(&self) -> ExecutorStatus {
        self.inner.status()
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum TestMaintenanceBehavior {
    Advance,
    WaitForRelease,
    StaleEpoch,
    Fail,
}

struct ExecutorManagedMaintenanceTestExecutor {
    inner: MockModelExecutor,
    retained: std::sync::Mutex<HashSet<RequestId>>,
    release_epoch: AtomicU64,
    capacity_epoch: AtomicU64,
    admission_probes: AtomicU64,
    maintenance_calls: AtomicU64,
    prefill_calls: AtomicU64,
    call_order: std::sync::Mutex<Vec<&'static str>>,
    maintenance_behavior: TestMaintenanceBehavior,
}

impl ExecutorManagedMaintenanceTestExecutor {
    fn new(vocab_size: usize) -> Self {
        Self::with_behavior(vocab_size, TestMaintenanceBehavior::Advance)
    }

    fn with_behavior(vocab_size: usize, maintenance_behavior: TestMaintenanceBehavior) -> Self {
        Self {
            inner: MockModelExecutor::instant(vocab_size),
            retained: std::sync::Mutex::new(HashSet::new()),
            release_epoch: AtomicU64::new(0),
            capacity_epoch: AtomicU64::new(0),
            admission_probes: AtomicU64::new(0),
            maintenance_calls: AtomicU64::new(0),
            prefill_calls: AtomicU64::new(0),
            call_order: std::sync::Mutex::new(Vec::new()),
            maintenance_behavior,
        }
    }

    fn epochs(&self) -> ExecutorAdmissionEpochs {
        ExecutorAdmissionEpochs::new(
            std::num::NonZeroU64::new(43).unwrap(),
            self.release_epoch.load(Ordering::Acquire),
            self.capacity_epoch.load(Ordering::Acquire),
        )
    }
}

#[async_trait::async_trait]
impl ModelExecutor for ExecutorManagedMaintenanceTestExecutor {
    fn info(&self) -> &ferrum_types::ModelInfo {
        self.inner.info()
    }

    fn execution_resource_ownership(&self) -> ExecutionResourceOwnership {
        ExecutionResourceOwnership::ExecutorManaged
    }

    fn prefill_admission_epochs(&self) -> Result<Option<ExecutorAdmissionEpochs>> {
        Ok(Some(self.epochs()))
    }

    fn try_admit_prefill(
        &self,
        input: ExecutorPrefillAdmission<'_>,
    ) -> Result<ExecutorPrefillAdmissionDecision> {
        self.admission_probes.fetch_add(1, Ordering::Relaxed);
        let inserted = self
            .retained
            .lock()
            .expect("retained admission mutex poisoned")
            .insert(input.request_id.clone());
        if !inserted {
            return Err(FerrumError::already_exists("duplicate test admission"));
        }
        if self.capacity_epoch.load(Ordering::Acquire) == 0
            && !(self.maintenance_behavior == TestMaintenanceBehavior::WaitForRelease
                && self.release_epoch.load(Ordering::Acquire) > 0)
        {
            self.call_order
                .lock()
                .expect("call order mutex poisoned")
                .push("defer");
            return Ok(ExecutorPrefillAdmissionDecision::MaintenanceDeferred(
                ExecutorPrefillMaintenanceDeferral::new(
                    input.request_id.clone(),
                    self.epochs(),
                    ExecutorPrefillMaintenanceStage::LogicalCapacity,
                    vec![ExecutorPrefillMaintenanceBlocker::Capacity {
                        domain_id: Some(1),
                        kind:
                            ferrum_interfaces::vnext::CapacityShortfallKind::BackingGrowthRequired,
                        requested: 4096,
                        available: 0,
                        current_total: 0,
                        maximum_total: 4096,
                    }],
                )?,
            ));
        }
        self.call_order
            .lock()
            .expect("call order mutex poisoned")
            .push("admit");
        Ok(ExecutorPrefillAdmissionDecision::Admitted(
            ExecutorPrefillAdmissionReceipt {
                request_id: input.request_id.clone(),
            },
        ))
    }

    fn cancel_prefill_admission(&self, request_id: &RequestId) -> bool {
        self.retained
            .lock()
            .expect("retained admission mutex poisoned")
            .remove(request_id)
    }

    fn maintain_prefill_backing(
        &self,
        request_id: &RequestId,
    ) -> Result<ExecutorPrefillMaintenanceOutcome> {
        if !self
            .retained
            .lock()
            .expect("retained admission mutex poisoned")
            .remove(request_id)
        {
            return Ok(ExecutorPrefillMaintenanceOutcome::NoLongerPending);
        }
        self.maintenance_calls.fetch_add(1, Ordering::Relaxed);
        self.call_order
            .lock()
            .expect("call order mutex poisoned")
            .push("maintain");
        match self.maintenance_behavior {
            TestMaintenanceBehavior::Advance => {
                self.capacity_epoch.store(1, Ordering::Release);
                Ok(ExecutorPrefillMaintenanceOutcome::Maintained {
                    current: self.epochs(),
                    pools_grown: 1,
                    allocated_bytes: 4096,
                })
            }
            TestMaintenanceBehavior::WaitForRelease => {
                Ok(ExecutorPrefillMaintenanceOutcome::WaitForRelease {
                    current: self.epochs(),
                    pressure: ferrum_interfaces::vnext::DeviceCapacityPressure::new(
                        ferrum_interfaces::vnext::DeviceCapacityPressureScope::PlanBudget,
                        "device.test-capacity-pressure".to_owned(),
                        4096,
                        4096,
                        4096,
                        4096,
                        4096,
                    )
                    .unwrap(),
                })
            }
            TestMaintenanceBehavior::StaleEpoch => {
                Ok(ExecutorPrefillMaintenanceOutcome::RetryWithoutGrowth {
                    current: self.epochs(),
                })
            }
            TestMaintenanceBehavior::Fail => {
                Err(FerrumError::backend("synthetic backing allocation failure"))
            }
        }
    }

    async fn prefill(&self, input: &PrefillInput) -> Result<PrefillOutput> {
        let request_id = input
            .request_id
            .as_ref()
            .ok_or_else(|| FerrumError::request_validation("missing request id"))?;
        if !self
            .retained
            .lock()
            .expect("retained admission mutex poisoned")
            .remove(request_id)
        {
            return Err(FerrumError::internal(
                "test prefill reached submit without retained admission",
            ));
        }
        self.prefill_calls.fetch_add(1, Ordering::Relaxed);
        self.call_order
            .lock()
            .expect("call order mutex poisoned")
            .push("prefill");
        self.inner.prefill(input).await
    }

    async fn decode(&self, input: &DecodeInput) -> Result<DecodeOutput> {
        self.inner.decode(input).await
    }

    fn release_cache(&self, cache_id: &str) {
        self.inner.release_cache(cache_id);
    }

    fn capabilities(&self) -> ExecutorCapabilities {
        self.inner.capabilities()
    }

    fn status(&self) -> ExecutorStatus {
        self.inner.status()
    }
}

#[derive(Debug, Clone)]
struct TestRecurrentStateHandle {
    request_id: RequestId,
    slots: usize,
}

impl TestRecurrentStateHandle {
    fn new(request_id: RequestId, slots: usize) -> Self {
        Self {
            request_id,
            slots: slots.max(1),
        }
    }
}

impl ferrum_interfaces::RecurrentStateHandle for TestRecurrentStateHandle {
    fn request_id(&self) -> RequestId {
        self.request_id.clone()
    }

    fn device(&self) -> Device {
        Device::CPU
    }

    fn num_layers(&self) -> usize {
        1
    }

    fn state_bytes(&self) -> usize {
        self.slots * std::mem::size_of::<f32>()
    }

    fn clone_handle(
        &self,
    ) -> ferrum_types::Result<Arc<dyn ferrum_interfaces::RecurrentStateHandle>> {
        Ok(Arc::new(self.clone()))
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn stats(&self) -> ferrum_interfaces::RecurrentStateHandleStats {
        ferrum_interfaces::RecurrentStateHandleStats {
            memory_bytes: self.state_bytes(),
            state_tensors: 1,
            batch_slots: self.slots,
            last_access: Instant::now(),
        }
    }

    fn is_valid(&self) -> bool {
        true
    }

    fn cache_id(&self) -> String {
        format!("test-recurrent-state-{}", self.request_id)
    }
}

fn test_recurrent_state_handle(
    request_id: RequestId,
    slots: usize,
) -> Arc<dyn ferrum_interfaces::RecurrentStateHandle> {
    Arc::new(TestRecurrentStateHandle::new(request_id, slots))
}

struct PolicyTokenizer {
    vocab_size: usize,
    special: ferrum_types::SpecialTokens,
    ids: HashMap<String, TokenId>,
    texts: Vec<Option<String>>,
}

impl PolicyTokenizer {
    fn new(vocab_size: usize, pairs: &[(&str, u32)]) -> Self {
        let max_id = pairs.iter().map(|(_, id)| *id as usize).max().unwrap_or(0);
        let mut texts = vec![None; max_id + 1];
        let mut ids = HashMap::new();
        for (text, id) in pairs {
            ids.insert((*text).to_string(), TokenId::new(*id));
            texts[*id as usize] = Some((*text).to_string());
        }
        Self {
            vocab_size,
            special: ferrum_types::SpecialTokens {
                bos_token: Some(TokenId::new(1)),
                eos_token: Some(TokenId::new(3)),
                unk_token: Some(TokenId::new(2)),
                pad_token: Some(TokenId::new(4)),
                sep_token: None,
                cls_token: None,
                mask_token: None,
                extra_eos_tokens: Vec::new(),
            },
            ids,
            texts,
        }
    }
}

impl Tokenizer for PolicyTokenizer {
    fn encode(&self, text: &str, _add_special: bool) -> Result<Vec<TokenId>> {
        if let Some(id) = self.ids.get(text) {
            return Ok(vec![*id]);
        }
        let split_tokens = text
            .split_whitespace()
            .map(|part| self.ids.get(part).copied())
            .collect::<Option<Vec<_>>>();
        if let Some(tokens) = split_tokens.filter(|tokens| !tokens.is_empty()) {
            return Ok(tokens);
        }
        Ok(vec![TokenId::new(0)])
    }

    fn decode(&self, tokens: &[TokenId], skip_special: bool) -> Result<String> {
        let mut output = String::new();
        let mut pending_bad_byte = false;
        for token in tokens {
            let Some(text) = self.token_text(*token) else {
                continue;
            };
            if skip_special && matches!(text, "<think>") {
                continue;
            }
            match text {
                "byte-fallback" => output.push('\u{FFFD}'),
                "bad-byte-lead" => pending_bad_byte = true,
                "valid-byte-cont" if pending_bad_byte => {
                    output.push('好');
                    pending_bad_byte = false;
                }
                text => {
                    if pending_bad_byte {
                        output.push('\u{FFFD}');
                        pending_bad_byte = false;
                    }
                    output.push_str(text);
                }
            }
        }
        Ok(output)
    }

    fn decode_incremental(&self, _prev: &[TokenId], next: TokenId) -> Result<String> {
        Ok(self.token_text(next).unwrap_or_default().to_string())
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    fn special_tokens(&self) -> &ferrum_types::SpecialTokens {
        &self.special
    }

    fn token_id(&self, text: &str) -> Option<TokenId> {
        self.ids.get(text).copied()
    }

    fn token_text(&self, token_id: TokenId) -> Option<&str> {
        self.texts
            .get(token_id.get() as usize)
            .and_then(|text| text.as_deref())
    }

    fn info(&self) -> TokenizerInfo {
        TokenizerInfo {
            tokenizer_type: TokenizerType::Custom,
            vocab_size: self.vocab_size,
            special_tokens: self.special.clone(),
            supports_incremental: true,
            supports_chat_template: false,
            max_token_length: None,
            model_name: Some("policy-tokenizer-test".to_string()),
        }
    }
}

fn policy_request() -> InferenceRequest {
    InferenceRequest {
        id: RequestId::new(),
        prompt: "test".to_string(),
        model_id: ferrum_types::ModelId::new("test"),
        sampling_params: SamplingParams::greedy(),
        stream: false,
        priority: Priority::Normal,
        client_id: None,
        session_id: None,
        created_at: chrono::Utc::now(),
        api_request: None,
        metadata: HashMap::new(),
    }
}

#[derive(Debug, Clone, Default)]
struct TestResourceTraceState {
    reserved: i64,
    committed: i64,
    released: i64,
    rolled_back: i64,
}

impl TestResourceTraceState {
    fn outstanding_reserved(&self) -> i64 {
        self.reserved - self.released - self.rolled_back
    }

    fn outstanding_committed(&self) -> i64 {
        self.committed - self.released - self.rolled_back
    }

    fn assert_zero_outstanding(&self, key: &(String, String, String)) {
        assert_eq!(
            self.outstanding_reserved(),
            0,
            "resource {key:?} closed with outstanding reserved state: {self:?}"
        );
        assert_eq!(
            self.outstanding_committed(),
            0,
            "resource {key:?} closed with outstanding committed state: {self:?}"
        );
    }
}

fn resource_trace_temp_path(label: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system clock should be after UNIX_EPOCH")
        .as_nanos();
    std::env::temp_dir().join(format!(
        "ferrum-engine-{label}-{}-{nanos}.jsonl",
        std::process::id()
    ))
}

fn read_engine_profile_events(path: &Path) -> Vec<FerrumProfileEvent> {
    let contents = std::fs::read_to_string(path).unwrap_or_else(|error| {
        panic!("failed to read resource trace {}: {error}", path.display())
    });
    assert!(
        !contents.trim().is_empty(),
        "resource trace {} should not be empty",
        path.display()
    );

    contents
        .lines()
        .enumerate()
        .filter_map(|(line_no, line)| {
            if line.trim().is_empty() {
                return None;
            }
            let event: FerrumProfileEvent = serde_json::from_str(line).unwrap_or_else(|error| {
                panic!(
                    "line {} in {} is not a FerrumProfileEvent: {error}\n{line}",
                    line_no + 1,
                    path.display()
                )
            });
            event.validate().unwrap_or_else(|error| {
                panic!(
                    "line {} in {} failed profile validation: {error}\n{line}",
                    line_no + 1,
                    path.display()
                )
            });
            Some(event)
        })
        .collect()
}

fn assert_engine_resource_trace_balanced(path: &Path) -> Vec<ResourceTraceEvent> {
    let mut states: HashMap<(String, String, String), TestResourceTraceState> = HashMap::new();
    let mut resources = Vec::new();

    for event in read_engine_profile_events(path) {
        let Some(resource) = event.resource else {
            continue;
        };
        resource.validate().unwrap();
        let key = (
            resource.owner_kind.clone(),
            resource.owner_id.clone(),
            resource.resource_kind.clone(),
        );
        match resource.action {
            ResourceAction::Reserve => {
                let amount = resource.amount.expect("reserve amount");
                let state = states.entry(key).or_default();
                assert!(amount > 0, "reserve amount must be positive: {resource:?}");
                assert_eq!(resource.before, Some(state.outstanding_reserved()));
                state.reserved += amount;
                assert_eq!(resource.after, Some(state.outstanding_reserved()));
            }
            ResourceAction::Commit => {
                let amount = resource.amount.expect("commit amount");
                let state = states.entry(key).or_default();
                assert!(amount > 0, "commit amount must be positive: {resource:?}");
                assert_eq!(resource.before, Some(state.outstanding_committed()));
                state.committed += amount;
                assert_eq!(resource.after, Some(state.outstanding_committed()));
            }
            ResourceAction::Release => {
                let amount = resource.amount.expect("release amount");
                let state = states.entry(key).or_default();
                assert!(amount > 0, "release amount must be positive: {resource:?}");
                let before = state.outstanding_committed();
                assert!(
                    amount <= before,
                    "release underflow for {resource:?}; state before release: {state:?}"
                );
                assert_eq!(resource.before, Some(before));
                state.released += amount;
                assert_eq!(resource.after, Some(state.outstanding_committed()));
            }
            ResourceAction::Rollback => {
                let amount = resource.amount.expect("rollback amount");
                let state = states.entry(key).or_default();
                assert!(amount > 0, "rollback amount must be positive: {resource:?}");
                let before = state.outstanding_reserved();
                assert!(
                    amount <= before,
                    "rollback underflow for {resource:?}; state before rollback: {state:?}"
                );
                assert_eq!(resource.before, Some(before));
                state.rolled_back += amount;
                assert_eq!(resource.after, Some(state.outstanding_reserved()));
            }
            ResourceAction::RequestClose => {
                let owner_kind = resource.owner_kind.clone();
                let owner_id = resource.owner_id.clone();
                for (state_key, state) in states
                    .iter()
                    .filter(|(state_key, _)| state_key.0 == owner_kind && state_key.1 == owner_id)
                {
                    state.assert_zero_outstanding(state_key);
                }
                states.retain(|state_key, _| state_key.0 != owner_kind || state_key.1 != owner_id);
            }
            ResourceAction::RequestOpen
            | ResourceAction::Defer
            | ResourceAction::Reject
            | ResourceAction::CapacitySnapshot => {}
        }
        resources.push(resource);
    }

    for (key, state) in &states {
        state.assert_zero_outstanding(key);
    }

    resources
}

struct RecurrentSpecExecutor {
    inner: MockModelExecutor,
}

struct FailingBatchPrefillExecutor {
    inner: RecurrentSpecExecutor,
}

struct FailingUnifiedReserveExecutor {
    inner: FailingBatchPrefillExecutor,
}

struct StructuredPressureOnceUnifiedReserveExecutor {
    inner: RecurrentSpecExecutor,
    remaining_failures: std::sync::atomic::AtomicUsize,
    message: &'static str,
}

struct FailingUnifiedForwardExecutor {
    inner: RecurrentSpecExecutor,
    resource_exhausted: bool,
}

struct BadShapePrefillExecutor {
    inner: RecurrentSpecExecutor,
}

struct ShortBatchPrefillExecutor {
    inner: RecurrentSpecExecutor,
}

struct FailingFromSliceTensorFactory;

struct ShortUnifiedResultExecutor {
    inner: RecurrentSpecExecutor,
}

struct MissingFinalUnifiedResultExecutor {
    inner: RecurrentSpecExecutor,
}

struct GreedySentinelUnifiedExecutor {
    inner: RecurrentSpecExecutor,
    token: u32,
}

struct FailingDecodeExecutor {
    inner: RecurrentSpecExecutor,
}

struct FirstAllocateThenFailKvCacheManager {
    inner: MockKvCacheManager,
    allocate_calls: std::sync::atomic::AtomicU64,
}

impl FirstAllocateThenFailKvCacheManager {
    fn new(total_blocks: usize) -> Self {
        Self {
            inner: MockKvCacheManager::new(total_blocks),
            allocate_calls: std::sync::atomic::AtomicU64::new(0),
        }
    }
}

struct FailingDeallocateKvCacheManager {
    inner: MockKvCacheManager,
}

impl FailingDeallocateKvCacheManager {
    fn new(total_blocks: usize) -> Self {
        Self {
            inner: MockKvCacheManager::new(total_blocks),
        }
    }
}

struct FailingDeallocateRecurrentStateManager {
    inner: InMemoryRecurrentStateManager,
}

impl FailingDeallocateRecurrentStateManager {
    fn new(config: InMemoryRecurrentStateConfig) -> Self {
        Self {
            inner: InMemoryRecurrentStateManager::new(config),
        }
    }
}

struct RecurrentSpecLlm {
    config: LlmRuntimeConfig,
}

impl RecurrentSpecLlm {
    fn new() -> Self {
        Self {
            config: LlmRuntimeConfig {
                hidden_size: 4,
                num_layers: 1,
                num_kv_heads: 1,
                head_dim: 4,
                vocab_size: 64,
                max_seq_len: 16,
            },
        }
    }
}

impl DecoderOnlyLLM for RecurrentSpecLlm {
    fn config(&self) -> &LlmRuntimeConfig {
        &self.config
    }

    fn recurrent_state_spec(
        &self,
        request_id: &RequestId,
        _input_tokens: &[TokenId],
    ) -> Result<Option<RecurrentStateSpec>> {
        Ok(Some(RecurrentStateSpec {
            request_id: request_id.clone(),
            num_layers: 1,
            tensors: vec![RecurrentStateTensorSpec::new(0, "delta_state", vec![4])],
            dtype: DataType::FP32,
            device: Device::CPU,
            max_batch_slots: 1,
        }))
    }

    fn prefill(&mut self, _cache_id: &str, _tokens: &[u32]) -> Vec<f32> {
        let mut logits = vec![0.0; self.config.vocab_size];
        logits[0] = 1.0;
        logits
    }

    fn decode(&mut self, _cache_id: &str, _token: u32, _pos: u32) -> Vec<f32> {
        let mut logits = vec![0.0; self.config.vocab_size];
        logits[0] = 1.0;
        logits
    }

    fn release(&mut self, _cache_id: &str) {}
}

#[derive(Clone, Debug)]
struct CapturedUnifiedItem {
    q_len: usize,
    pos_offset: usize,
    is_final_chunk: bool,
    admission_target_len: Option<usize>,
    logits_policy: ferrum_interfaces::model_executor::LogitsReturnPolicy,
}

struct CapturingUnifiedExecutor {
    inner: MockModelExecutor,
    captured: Arc<std::sync::Mutex<Vec<Vec<CapturedUnifiedItem>>>>,
    output_token: u32,
}

#[async_trait::async_trait]
impl ModelExecutor for CapturingUnifiedExecutor {
    fn info(&self) -> &ferrum_types::ModelInfo {
        self.inner.info()
    }

    fn supports_native_unified_decode(&self) -> bool {
        true
    }

    async fn prefill(&self, input: &PrefillInput) -> Result<PrefillOutput> {
        self.inner.prefill(input).await
    }

    async fn decode(&self, input: &DecodeInput) -> Result<DecodeOutput> {
        self.inner.decode(input).await
    }

    async fn unified_decode(&self, batch: &UnifiedBatch) -> Result<Vec<Option<Vec<f32>>>> {
        let captured = batch
            .items
            .iter()
            .map(|item| CapturedUnifiedItem {
                q_len: item.q_tokens.len(),
                pos_offset: item.pos_offset,
                is_final_chunk: item.is_final_chunk,
                admission_target_len: item
                    .metadata
                    .get("ferrum_kv_admission_target_len")
                    .and_then(|value| value.as_u64())
                    .map(|value| value as usize),
                logits_policy: item.logits_policy.clone(),
            })
            .collect::<Vec<_>>();
        self.captured
            .lock()
            .expect("capture mutex poisoned")
            .push(captured);

        Ok(batch
            .items
            .iter()
            .map(|item| item.is_final_chunk.then(|| vec![self.output_token as f32]))
            .collect())
    }

    fn capabilities(&self) -> ExecutorCapabilities {
        self.inner.capabilities()
    }

    fn status(&self) -> ExecutorStatus {
        self.inner.status()
    }
}

#[async_trait::async_trait]
impl ModelExecutor for RecurrentSpecExecutor {
    fn info(&self) -> &ferrum_types::ModelInfo {
        self.inner.info()
    }

    fn supports_native_unified_decode(&self) -> bool {
        true
    }

    fn recurrent_state_spec(
        &self,
        request_id: &RequestId,
        _input_tokens: &[TokenId],
    ) -> Result<Option<RecurrentStateSpec>> {
        Ok(Some(RecurrentStateSpec {
            request_id: request_id.clone(),
            num_layers: 1,
            tensors: vec![RecurrentStateTensorSpec::new(0, "delta_state", vec![4])],
            dtype: DataType::BF16,
            device: Device::CPU,
            max_batch_slots: 1,
        }))
    }

    async fn prefill(&self, input: &PrefillInput) -> Result<PrefillOutput> {
        self.inner.prefill(input).await
    }

    async fn decode(&self, input: &DecodeInput) -> Result<DecodeOutput> {
        self.inner.decode(input).await
    }

    async fn unified_decode(&self, batch: &UnifiedBatch) -> Result<Vec<Option<Vec<f32>>>> {
        Ok(batch
            .items
            .iter()
            .map(|item| {
                item.is_final_chunk.then(|| {
                    let mut logits = vec![0.0; self.info().vocab_size];
                    logits[0] = 1.0;
                    logits
                })
            })
            .collect())
    }

    fn capabilities(&self) -> ExecutorCapabilities {
        self.inner.capabilities()
    }

    fn status(&self) -> ExecutorStatus {
        self.inner.status()
    }
}

#[async_trait::async_trait]
impl ModelExecutor for FailingBatchPrefillExecutor {
    fn info(&self) -> &ferrum_types::ModelInfo {
        self.inner.info()
    }

    fn supports_native_unified_decode(&self) -> bool {
        self.inner.supports_native_unified_decode()
    }

    fn recurrent_state_spec(
        &self,
        request_id: &RequestId,
        input_tokens: &[TokenId],
    ) -> Result<Option<RecurrentStateSpec>> {
        self.inner.recurrent_state_spec(request_id, input_tokens)
    }

    async fn prefill(&self, _input: &PrefillInput) -> Result<PrefillOutput> {
        Err(FerrumError::resource_exhausted(
            "synthetic single prefill model-side KV reserve failure",
        ))
    }

    async fn batch_prefill(&self, _inputs: &[PrefillInput]) -> Result<Vec<PrefillOutput>> {
        Err(FerrumError::resource_exhausted(
            "synthetic model-side KV reserve failure",
        ))
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
}

#[async_trait::async_trait]
impl ModelExecutor for FailingUnifiedReserveExecutor {
    fn info(&self) -> &ferrum_types::ModelInfo {
        self.inner.info()
    }

    fn supports_native_unified_decode(&self) -> bool {
        true
    }

    fn recurrent_state_spec(
        &self,
        request_id: &RequestId,
        input_tokens: &[TokenId],
    ) -> Result<Option<RecurrentStateSpec>> {
        self.inner.recurrent_state_spec(request_id, input_tokens)
    }

    fn reserve_kv_slots(
        &self,
        _requests: &[ferrum_interfaces::model_executor::KvSlotRequest],
    ) -> Result<Option<ferrum_interfaces::model_executor::KvSlotReservation>> {
        Err(FerrumError::resource_exhausted(
            "synthetic unified reserve failure",
        ))
    }

    async fn prefill(&self, input: &PrefillInput) -> Result<PrefillOutput> {
        self.inner.prefill(input).await
    }

    async fn batch_prefill(&self, inputs: &[PrefillInput]) -> Result<Vec<PrefillOutput>> {
        self.inner.batch_prefill(inputs).await
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
}

#[async_trait::async_trait]
impl ModelExecutor for StructuredPressureOnceUnifiedReserveExecutor {
    fn info(&self) -> &ferrum_types::ModelInfo {
        self.inner.info()
    }

    fn supports_native_unified_decode(&self) -> bool {
        true
    }

    fn recurrent_state_spec(
        &self,
        request_id: &RequestId,
        input_tokens: &[TokenId],
    ) -> Result<Option<RecurrentStateSpec>> {
        self.inner.recurrent_state_spec(request_id, input_tokens)
    }

    fn reserve_kv_slots(
        &self,
        _requests: &[ferrum_interfaces::model_executor::KvSlotRequest],
    ) -> Result<Option<ferrum_interfaces::model_executor::KvSlotReservation>> {
        if self
            .remaining_failures
            .fetch_update(
                std::sync::atomic::Ordering::Relaxed,
                std::sync::atomic::Ordering::Relaxed,
                |remaining| remaining.checked_sub(1),
            )
            .is_ok()
        {
            return Err(FerrumError::resource_exhausted(self.message));
        }
        Ok(None)
    }

    async fn prefill(&self, input: &PrefillInput) -> Result<PrefillOutput> {
        self.inner.prefill(input).await
    }

    async fn batch_prefill(&self, inputs: &[PrefillInput]) -> Result<Vec<PrefillOutput>> {
        self.inner.batch_prefill(inputs).await
    }

    async fn decode(&self, input: &DecodeInput) -> Result<DecodeOutput> {
        self.inner.decode(input).await
    }

    async fn unified_decode(&self, batch: &UnifiedBatch) -> Result<Vec<Option<Vec<f32>>>> {
        self.inner.unified_decode(batch).await
    }

    fn capabilities(&self) -> ExecutorCapabilities {
        self.inner.capabilities()
    }

    fn status(&self) -> ExecutorStatus {
        self.inner.status()
    }
}

#[async_trait::async_trait]
impl ModelExecutor for FailingUnifiedForwardExecutor {
    fn info(&self) -> &ferrum_types::ModelInfo {
        self.inner.info()
    }

    fn supports_native_unified_decode(&self) -> bool {
        true
    }

    fn recurrent_state_spec(
        &self,
        request_id: &RequestId,
        input_tokens: &[TokenId],
    ) -> Result<Option<RecurrentStateSpec>> {
        self.inner.recurrent_state_spec(request_id, input_tokens)
    }

    async fn prefill(&self, input: &PrefillInput) -> Result<PrefillOutput> {
        self.inner.prefill(input).await
    }

    async fn batch_prefill(&self, inputs: &[PrefillInput]) -> Result<Vec<PrefillOutput>> {
        self.inner.batch_prefill(inputs).await
    }

    async fn decode(&self, input: &DecodeInput) -> Result<DecodeOutput> {
        self.inner.decode(input).await
    }

    async fn unified_decode(&self, _batch: &UnifiedBatch) -> Result<Vec<Option<Vec<f32>>>> {
        if self.resource_exhausted {
            Err(FerrumError::resource_exhausted(
                "synthetic unified forward resource exhaustion",
            ))
        } else {
            Err(FerrumError::internal(
                "synthetic unified forward internal failure",
            ))
        }
    }

    fn capabilities(&self) -> ExecutorCapabilities {
        self.inner.capabilities()
    }

    fn status(&self) -> ExecutorStatus {
        self.inner.status()
    }
}

#[async_trait::async_trait]
impl ModelExecutor for BadShapePrefillExecutor {
    fn info(&self) -> &ferrum_types::ModelInfo {
        self.inner.info()
    }

    fn supports_native_unified_decode(&self) -> bool {
        false
    }

    fn recurrent_state_spec(
        &self,
        request_id: &RequestId,
        input_tokens: &[TokenId],
    ) -> Result<Option<RecurrentStateSpec>> {
        self.inner.recurrent_state_spec(request_id, input_tokens)
    }

    async fn prefill(&self, input: &PrefillInput) -> Result<PrefillOutput> {
        let kv = input
            .kv_cache
            .clone()
            .ok_or_else(|| FerrumError::internal("bad-shape prefill missing kv"))?;
        let logits = MockTensor::zeros(&[1, self.info().vocab_size], DataType::FP32).into_ref();
        let output = PrefillOutput::new(logits, kv);
        Ok(if let Some(state) = input.recurrent_state.clone() {
            output.with_recurrent_state(state)
        } else {
            output
        })
    }

    async fn batch_prefill(&self, inputs: &[PrefillInput]) -> Result<Vec<PrefillOutput>> {
        let mut outputs = Vec::with_capacity(inputs.len());
        for input in inputs {
            outputs.push(self.prefill(input).await?);
        }
        Ok(outputs)
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
}

#[async_trait::async_trait]
impl ModelExecutor for ShortBatchPrefillExecutor {
    fn info(&self) -> &ferrum_types::ModelInfo {
        self.inner.info()
    }

    fn supports_native_unified_decode(&self) -> bool {
        false
    }

    fn recurrent_state_spec(
        &self,
        request_id: &RequestId,
        input_tokens: &[TokenId],
    ) -> Result<Option<RecurrentStateSpec>> {
        self.inner.recurrent_state_spec(request_id, input_tokens)
    }

    async fn prefill(&self, input: &PrefillInput) -> Result<PrefillOutput> {
        self.inner.prefill(input).await
    }

    async fn batch_prefill(&self, _inputs: &[PrefillInput]) -> Result<Vec<PrefillOutput>> {
        Ok(Vec::new())
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
}

#[async_trait::async_trait]
impl ModelExecutor for ShortUnifiedResultExecutor {
    fn info(&self) -> &ferrum_types::ModelInfo {
        self.inner.info()
    }

    fn supports_native_unified_decode(&self) -> bool {
        true
    }

    fn recurrent_state_spec(
        &self,
        request_id: &RequestId,
        input_tokens: &[TokenId],
    ) -> Result<Option<RecurrentStateSpec>> {
        self.inner.recurrent_state_spec(request_id, input_tokens)
    }

    async fn prefill(&self, input: &PrefillInput) -> Result<PrefillOutput> {
        self.inner.prefill(input).await
    }

    async fn decode(&self, input: &DecodeInput) -> Result<DecodeOutput> {
        self.inner.decode(input).await
    }

    async fn unified_decode(&self, _batch: &UnifiedBatch) -> Result<Vec<Option<Vec<f32>>>> {
        Ok(Vec::new())
    }

    fn capabilities(&self) -> ExecutorCapabilities {
        self.inner.capabilities()
    }

    fn status(&self) -> ExecutorStatus {
        self.inner.status()
    }
}

#[async_trait::async_trait]
impl ModelExecutor for MissingFinalUnifiedResultExecutor {
    fn info(&self) -> &ferrum_types::ModelInfo {
        self.inner.info()
    }

    fn supports_native_unified_decode(&self) -> bool {
        true
    }

    fn recurrent_state_spec(
        &self,
        request_id: &RequestId,
        input_tokens: &[TokenId],
    ) -> Result<Option<RecurrentStateSpec>> {
        self.inner.recurrent_state_spec(request_id, input_tokens)
    }

    async fn prefill(&self, input: &PrefillInput) -> Result<PrefillOutput> {
        self.inner.prefill(input).await
    }

    async fn decode(&self, input: &DecodeInput) -> Result<DecodeOutput> {
        self.inner.decode(input).await
    }

    async fn unified_decode(&self, batch: &UnifiedBatch) -> Result<Vec<Option<Vec<f32>>>> {
        Ok(batch.items.iter().map(|_| None).collect())
    }

    fn capabilities(&self) -> ExecutorCapabilities {
        self.inner.capabilities()
    }

    fn status(&self) -> ExecutorStatus {
        self.inner.status()
    }
}

#[async_trait::async_trait]
impl ModelExecutor for GreedySentinelUnifiedExecutor {
    fn info(&self) -> &ferrum_types::ModelInfo {
        self.inner.info()
    }

    fn supports_native_unified_decode(&self) -> bool {
        true
    }

    fn recurrent_state_spec(
        &self,
        request_id: &RequestId,
        input_tokens: &[TokenId],
    ) -> Result<Option<RecurrentStateSpec>> {
        self.inner.recurrent_state_spec(request_id, input_tokens)
    }

    async fn prefill(&self, input: &PrefillInput) -> Result<PrefillOutput> {
        self.inner.prefill(input).await
    }

    async fn decode(&self, input: &DecodeInput) -> Result<DecodeOutput> {
        self.inner.decode(input).await
    }

    async fn unified_decode(&self, batch: &UnifiedBatch) -> Result<Vec<Option<Vec<f32>>>> {
        Ok(batch
            .items
            .iter()
            .map(|item| item.is_final_chunk.then(|| vec![self.token as f32]))
            .collect())
    }

    fn capabilities(&self) -> ExecutorCapabilities {
        self.inner.capabilities()
    }

    fn status(&self) -> ExecutorStatus {
        self.inner.status()
    }
}

#[async_trait::async_trait]
impl ModelExecutor for FailingDecodeExecutor {
    fn info(&self) -> &ferrum_types::ModelInfo {
        self.inner.info()
    }

    fn supports_native_unified_decode(&self) -> bool {
        self.inner.supports_native_unified_decode()
    }

    fn recurrent_state_spec(
        &self,
        request_id: &RequestId,
        input_tokens: &[TokenId],
    ) -> Result<Option<RecurrentStateSpec>> {
        self.inner.recurrent_state_spec(request_id, input_tokens)
    }

    async fn prefill(&self, input: &PrefillInput) -> Result<PrefillOutput> {
        self.inner.prefill(input).await
    }

    async fn decode(&self, _input: &DecodeInput) -> Result<DecodeOutput> {
        Err(FerrumError::resource_exhausted(
            "synthetic decode model-side KV reserve failure",
        ))
    }

    fn capabilities(&self) -> ExecutorCapabilities {
        self.inner.capabilities()
    }

    fn status(&self) -> ExecutorStatus {
        self.inner.status()
    }
}

#[async_trait::async_trait]
impl KvCacheManager for FirstAllocateThenFailKvCacheManager {
    async fn allocate(&self, request: &AllocationRequest) -> Result<Arc<dyn KvCacheHandle>> {
        let call = self
            .allocate_calls
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if call == 0 {
            self.inner.allocate(request).await
        } else {
            Err(FerrumError::resource_exhausted(
                "synthetic fallback KV allocation exhaustion",
            ))
        }
    }

    async fn extend(&self, handle: &mut dyn KvCacheHandle, additional_tokens: usize) -> Result<()> {
        self.inner.extend(handle, additional_tokens).await
    }

    async fn deallocate(&self, request_id: RequestId) -> Result<()> {
        self.inner.deallocate(request_id).await
    }

    fn can_allocate(&self, request: &AllocationRequest) -> bool {
        self.inner.can_allocate(request)
    }

    fn stats(&self) -> CacheManagerStats {
        self.inner.stats()
    }

    async fn gc(&self) -> Result<CacheGcStats> {
        self.inner.gc().await
    }

    fn set_pressure_callback(&self, callback: Box<dyn Fn(MemoryPressure) + Send + Sync>) {
        self.inner.set_pressure_callback(callback);
    }

    fn get_handle(&self, request_id: RequestId) -> Option<Arc<dyn KvCacheHandle>> {
        self.inner.get_handle(request_id)
    }

    fn list_handles(&self) -> Vec<(RequestId, Arc<dyn KvCacheHandle>)> {
        self.inner.list_handles()
    }
}

#[async_trait::async_trait]
impl KvCacheManager for FailingDeallocateKvCacheManager {
    async fn allocate(&self, request: &AllocationRequest) -> Result<Arc<dyn KvCacheHandle>> {
        self.inner.allocate(request).await
    }

    async fn extend(&self, handle: &mut dyn KvCacheHandle, additional_tokens: usize) -> Result<()> {
        self.inner.extend(handle, additional_tokens).await
    }

    async fn deallocate(&self, request_id: RequestId) -> Result<()> {
        Err(FerrumError::internal(format!(
            "synthetic KV release failure for {request_id}"
        )))
    }

    fn can_allocate(&self, request: &AllocationRequest) -> bool {
        self.inner.can_allocate(request)
    }

    fn stats(&self) -> CacheManagerStats {
        self.inner.stats()
    }

    async fn gc(&self) -> Result<CacheGcStats> {
        self.inner.gc().await
    }

    fn set_pressure_callback(&self, callback: Box<dyn Fn(MemoryPressure) + Send + Sync>) {
        self.inner.set_pressure_callback(callback);
    }

    fn get_handle(&self, request_id: RequestId) -> Option<Arc<dyn KvCacheHandle>> {
        self.inner.get_handle(request_id)
    }

    fn list_handles(&self) -> Vec<(RequestId, Arc<dyn KvCacheHandle>)> {
        self.inner.list_handles()
    }
}

#[async_trait::async_trait]
impl RecurrentStateManager for FailingDeallocateRecurrentStateManager {
    async fn allocate(&self, spec: &RecurrentStateSpec) -> Result<Arc<dyn RecurrentStateHandle>> {
        self.inner.allocate(spec).await
    }

    async fn deallocate(&self, request_id: RequestId) -> Result<()> {
        Err(FerrumError::internal(format!(
            "synthetic recurrent-state release failure for {request_id}"
        )))
    }

    fn can_allocate(&self, spec: &RecurrentStateSpec) -> bool {
        self.inner.can_allocate(spec)
    }

    fn get_handle(&self, request_id: RequestId) -> Option<Arc<dyn RecurrentStateHandle>> {
        self.inner.get_handle(request_id)
    }

    fn list_handles(&self) -> Vec<(RequestId, Arc<dyn RecurrentStateHandle>)> {
        self.inner.list_handles()
    }

    fn stats(&self) -> ferrum_interfaces::RecurrentStateManagerStats {
        self.inner.stats()
    }

    async fn reset(&self) -> Result<()> {
        self.inner.reset().await
    }
}

impl TensorFactory for FailingFromSliceTensorFactory {
    fn empty(&self, shape: &[usize], dtype: DataType, device: Device) -> Result<TensorRef> {
        MockTensorFactory.empty(shape, dtype, device)
    }

    fn zeros_like(&self, tensor: &TensorRef) -> Result<TensorRef> {
        MockTensorFactory.zeros_like(tensor)
    }

    fn from_slice(
        &self,
        _data: &[f32],
        _shape: &[usize],
        _dtype: DataType,
        _device: Device,
    ) -> Result<TensorRef> {
        Err(FerrumError::backend("synthetic tokens_to_tensor failure"))
    }

    fn to_device(&self, tensor: &TensorRef, device: Device) -> Result<TensorRef> {
        MockTensorFactory.to_device(tensor, device)
    }

    fn narrow(
        &self,
        tensor: &TensorRef,
        dim: usize,
        start: usize,
        length: usize,
    ) -> Result<TensorRef> {
        MockTensorFactory.narrow(tensor, dim, start, length)
    }

    fn reshape(&self, tensor: &TensorRef, shape: &[usize]) -> Result<TensorRef> {
        MockTensorFactory.reshape(tensor, shape)
    }

    fn zeros(&self, shape: &[usize], dtype: DataType, device: &Device) -> Result<TensorRef> {
        MockTensorFactory.zeros(shape, dtype, device)
    }

    fn ones(&self, shape: &[usize], dtype: DataType, device: &Device) -> Result<TensorRef> {
        MockTensorFactory.ones(shape, dtype, device)
    }

    fn uniform(
        &self,
        shape: &[usize],
        low: f32,
        high: f32,
        dtype: DataType,
        device: &Device,
    ) -> Result<TensorRef> {
        MockTensorFactory.uniform(shape, low, high, dtype, device)
    }

    fn normal(
        &self,
        shape: &[usize],
        mean: f32,
        std: f32,
        dtype: DataType,
        device: &Device,
    ) -> Result<TensorRef> {
        MockTensorFactory.normal(shape, mean, std, dtype, device)
    }

    fn from_tensor(&self, tensor: &TensorRef, device: &Device) -> Result<TensorRef> {
        MockTensorFactory.from_tensor(tensor, device)
    }
}

#[tokio::test]
async fn process_batch_unified_forwards_prefill_logits_policy() {
    let mut config = EngineConfig::default();
    config.kv_cache.max_blocks = 128;
    let scheduler = Arc::new(ContinuousBatchScheduler::new(config.scheduler.clone()));
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> = Arc::new(PolicyTokenizer::new(
        64,
        &[("test", 5), ("ok", 6), ("<unk>", 2), ("<pad>", 4)],
    ));
    let sampler: Arc<dyn Sampler + Send + Sync> = Arc::new(crate::registry::GreedySampler);
    let kv_cache: Arc<dyn KvCacheManager + Send + Sync> = Arc::new(MockKvCacheManager::new(128));
    let tensor_factory: Arc<dyn TensorFactory> = Arc::new(MockTensorFactory);
    let captured = Arc::new(std::sync::Mutex::new(Vec::new()));
    let executor: Arc<dyn ModelExecutor + Send + Sync> = Arc::new(CapturingUnifiedExecutor {
        inner: MockModelExecutor::instant(64),
        captured: captured.clone(),
        output_token: 6,
    });
    let engine = ContinuousBatchEngine::new(
        config,
        scheduler,
        tokenizer,
        sampler,
        kv_cache,
        executor,
        tensor_factory,
    );
    let mut request = policy_request();
    request.sampling_params.max_tokens = 1;

    let response = engine.infer(request).await.unwrap();
    assert_eq!(response.finish_reason, FinishReason::Length);

    let captured = captured.lock().expect("capture mutex poisoned");
    assert_eq!(captured.len(), 1);
    assert_eq!(captured[0].len(), 1);
    let item = &captured[0][0];
    assert_eq!(item.q_len, 1);
    assert_eq!(item.pos_offset, 0);
    assert!(item.is_final_chunk);
    let ferrum_interfaces::model_executor::LogitsReturnPolicy::GreedyArgmax {
        token_mask: Some(mask),
        repetition_penalty: None,
    } = &item.logits_policy
    else {
        panic!("final product prefill should use model-side greedy argmax policy");
    };
    assert_eq!(mask.valid_token_mask[2], 0, "unk token must stay masked");
    assert_eq!(
        mask.valid_token_mask[6], 1,
        "normal generated token must be selectable"
    );
}

#[tokio::test]
async fn process_batch_unified_honors_runtime_chunked_prefill() {
    let mut config = EngineConfig::default();
    config.kv_cache.max_blocks = 128;
    config.runtime.chunked_prefill_size = Some(1);
    let scheduler = Arc::new(ContinuousBatchScheduler::new(config.scheduler.clone()));
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> = Arc::new(PolicyTokenizer::new(
        64,
        &[("test", 5), ("ok", 6), ("<unk>", 2), ("<pad>", 4)],
    ));
    let sampler: Arc<dyn Sampler + Send + Sync> = Arc::new(crate::registry::GreedySampler);
    let kv_cache: Arc<dyn KvCacheManager + Send + Sync> = Arc::new(MockKvCacheManager::new(128));
    let tensor_factory: Arc<dyn TensorFactory> = Arc::new(MockTensorFactory);
    let captured = Arc::new(std::sync::Mutex::new(Vec::new()));
    let executor: Arc<dyn ModelExecutor + Send + Sync> = Arc::new(CapturingUnifiedExecutor {
        inner: MockModelExecutor::instant(64),
        captured: captured.clone(),
        output_token: 6,
    });
    let engine = ContinuousBatchEngine::new(
        config,
        scheduler,
        tokenizer,
        sampler,
        kv_cache,
        executor,
        tensor_factory,
    );
    let mut request = policy_request();
    request.prompt = "test ok".to_string();
    request.sampling_params.max_tokens = 1;

    let response = engine.infer(request).await.unwrap();
    assert_eq!(response.finish_reason, FinishReason::Length);

    let captured = captured.lock().expect("capture mutex poisoned");
    assert_eq!(captured.len(), 2);
    assert_eq!(captured[0].len(), 1);
    assert_eq!(captured[0][0].q_len, 1);
    assert_eq!(captured[0][0].pos_offset, 0);
    assert!(!captured[0][0].is_final_chunk);
    assert_eq!(captured[1].len(), 1);
    assert_eq!(captured[1][0].q_len, 1);
    assert_eq!(captured[1][0].pos_offset, 1);
    assert!(captured[1][0].is_final_chunk);
}

#[tokio::test]
async fn process_batch_unified_co_batches_active_decode_with_fresh_prefill_chunk() {
    let mut config = EngineConfig::default();
    config.kv_cache.max_blocks = 128;
    config.runtime.chunked_prefill_size = Some(1);
    let scheduler = Arc::new(ContinuousBatchScheduler::new(config.scheduler.clone()));
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> = Arc::new(PolicyTokenizer::new(
        64,
        &[("test", 5), ("ok", 6), ("<unk>", 2), ("<pad>", 4)],
    ));
    let sampler: Arc<dyn Sampler + Send + Sync> = Arc::new(crate::registry::GreedySampler);
    let kv_cache: Arc<dyn KvCacheManager + Send + Sync> = Arc::new(MockKvCacheManager::new(128));
    let tensor_factory: Arc<dyn TensorFactory> = Arc::new(MockTensorFactory);
    let captured = Arc::new(std::sync::Mutex::new(Vec::new()));
    let executor: Arc<dyn ModelExecutor + Send + Sync> = Arc::new(CapturingUnifiedExecutor {
        inner: MockModelExecutor::instant(64),
        captured: captured.clone(),
        output_token: 6,
    });
    let engine = ContinuousBatchEngine::new(
        config,
        scheduler,
        tokenizer.clone(),
        sampler,
        kv_cache,
        executor,
        tensor_factory,
    );

    let mut decode_request = policy_request();
    decode_request.prompt = "test".to_string();
    decode_request.sampling_params.max_tokens = 3;
    let decode_id = decode_request.id.clone();
    let decode_kv = engine
        .inner
        .make_model_kv_handle_with_seq("decode-cache".to_string(), 2);
    let mut decode_seq = SequenceState::new_with_tokenizer_and_model_vocab_size(
        decode_request.clone(),
        vec![TokenId::new(5)],
        Some(tokenizer.clone()),
        Some(64),
    );
    decode_seq.generated_tokens.push(TokenId::new(6));
    decode_seq.prefill_complete = true;
    decode_seq.prefill_tokens_processed = 1;
    decode_seq.install_model_kv_without_owned_blocks(decode_kv);
    decode_seq.phase = RequestPhase::Decoding;
    {
        let mut sequences = engine.inner.sequences.write();
        sequences.insert(decode_id.clone(), decode_seq);
    }

    let mut prefill_request = policy_request();
    prefill_request.prompt = "test ok".to_string();
    prefill_request.sampling_params.max_tokens = 1;
    let mut prefill_scheduled =
        ferrum_interfaces::scheduler::ScheduledRequest::new(prefill_request);
    prefill_scheduled.tokens_to_process = Some(1);
    let mut decode_scheduled = ferrum_interfaces::scheduler::ScheduledRequest::new(decode_request);
    decode_scheduled.tokens_to_process = Some(1);
    let batch = ferrum_interfaces::BatchPlan {
        batch_id: ferrum_types::BatchId::new(),
        requests: vec![prefill_scheduled, decode_scheduled],
        max_sequence_length: 2,
        estimated_time_ms: None,
        resource_requirements: ferrum_interfaces::scheduler::BatchResourceRequirements::default(),
        created_at: chrono::Utc::now(),
    };

    engine.inner.process_batch(&batch).await.unwrap();

    let captured = captured.lock().expect("capture mutex poisoned");
    assert_eq!(captured.len(), 1, "mixed work must use one unified call");
    assert_eq!(captured[0].len(), 2);
    let prefill = &captured[0][0];
    assert_eq!(prefill.q_len, 1);
    assert_eq!(prefill.pos_offset, 0);
    assert!(
        !prefill.is_final_chunk,
        "fresh first chunk should stay non-final in the mixed batch"
    );
    assert_eq!(
        prefill.admission_target_len, None,
        "mixed non-final prefill chunk should reserve only immediate KV slots"
    );
    let decode = &captured[0][1];
    assert_eq!(decode.q_len, 1);
    assert_eq!(decode.pos_offset, 1);
    assert!(decode.is_final_chunk);
    assert_eq!(
        decode.admission_target_len,
        Some(2),
        "decode metadata should keep its current-context admission target"
    );
}

#[test]
fn continuous_engine_runtime_config_parses_env_snapshot() {
    let cfg = ContinuousEngineRuntimeConfig::from_env_vars(
        Some(64),
        [
            (BATCH_DECODE_PROF_ENV, "1"),
            (CHUNKED_PREFILL_ENV, "128"),
            (KV_CAPACITY_ENV, "2048"),
            (MAX_MODEL_LEN_ENV, "4096"),
            (NEXT_BATCH_PROF_ENV, "1"),
            (WHOLE_PROMPT_PREFIX_CACHE_ENV, "1"),
            (RBD_PROF_ENV, "1"),
            ("FERRUM_SCHEDULER_TRACE_JSONL", "/tmp/scheduler-trace.jsonl"),
            (UNIFIED_POST_PROF_ENV, "1"),
        ],
    );

    assert_eq!(cfg.active_decode_prefill_chunk, Some(64));
    assert!(cfg.batch_decode_prof);
    assert!(cfg.chunked_prefill_present);
    assert_eq!(cfg.chunked_prefill_size, Some(128));
    assert_eq!(cfg.chunked_prefill_size_for(200), Some(128));
    assert_eq!(cfg.chunked_prefill_size_for(128), None);
    assert_eq!(cfg.kv_capacity, Some(2048));
    assert_eq!(cfg.max_model_len, Some(4096));
    assert!(cfg.next_batch_prof);
    assert!(cfg.prefix_cache_enabled);
    assert!(cfg.rbd_prof);
    assert_eq!(
        cfg.scheduler_trace_jsonl.as_deref(),
        Some(std::path::Path::new("/tmp/scheduler-trace.jsonl"))
    );
    assert!(cfg.unified_post_prof);
}

#[test]
fn continuous_engine_runtime_config_keeps_invalid_chunk_presence() {
    let cfg = ContinuousEngineRuntimeConfig::from_env_vars(
        None,
        [
            (CHUNKED_PREFILL_ENV, "invalid"),
            (WHOLE_PROMPT_PREFIX_CACHE_ENV, "0"),
        ],
    );

    assert!(cfg.chunked_prefill_present);
    assert_eq!(cfg.chunked_prefill_size, None);
    assert_eq!(cfg.chunked_prefill_size_for(200), None);
    assert!(!cfg.prefix_cache_enabled);
}

#[test]
fn performance_breakdown_reports_engine_timing_counters() {
    let config = EngineConfig::default();
    let scheduler = Arc::new(ContinuousBatchScheduler::new(config.scheduler.clone()));
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> =
        Arc::new(ferrum_testkit::MockTokenizer::new(128));
    let sampler: Arc<dyn Sampler + Send + Sync> = Arc::new(ferrum_testkit::MockSampler);
    let kv_cache: Arc<dyn KvCacheManager + Send + Sync> =
        Arc::new(ferrum_testkit::MockKvCacheManager::new(256));
    let tensor_factory: Arc<dyn TensorFactory> = Arc::new(ferrum_testkit::MockTensorFactory);
    let model_executor: Arc<dyn ModelExecutor + Send + Sync> =
        Arc::new(ferrum_testkit::MockModelExecutor::instant(128));
    let engine = ContinuousBatchEngine::new(
        config,
        scheduler,
        tokenizer,
        sampler,
        kv_cache,
        model_executor,
        tensor_factory,
    );

    engine
        .inner
        .record_scheduling_time(Duration::from_micros(1500));
    engine
        .inner
        .record_scheduling_time(Duration::from_micros(2500));
    engine
        .inner
        .record_model_execution_time(Duration::from_micros(10_000));
    engine
        .inner
        .record_model_execution_time(Duration::from_micros(14_000));
    engine
        .inner
        .record_iteration_lock_wait(Duration::from_micros(300));
    engine
        .inner
        .record_iteration_lock_wait(Duration::from_micros(700));

    let breakdown = engine.metrics().performance_breakdown;
    assert_eq!(breakdown.scheduling_time_ms, 2.0);
    assert_eq!(breakdown.model_execution_time_ms, 12.0);
    assert_eq!(breakdown.other_overhead_time_ms, 0.5);
}

fn test_continuous_engine() -> ContinuousBatchEngine {
    test_continuous_engine_with_config(EngineConfig::default())
}

fn test_continuous_engine_with_config(config: EngineConfig) -> ContinuousBatchEngine {
    let scheduler = Arc::new(ContinuousBatchScheduler::new(config.scheduler.clone()));
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> =
        Arc::new(ferrum_testkit::MockTokenizer::new(128));
    let sampler: Arc<dyn Sampler + Send + Sync> = Arc::new(ferrum_testkit::MockSampler);
    let kv_cache: Arc<dyn KvCacheManager + Send + Sync> =
        Arc::new(ferrum_testkit::MockKvCacheManager::new(256));
    let tensor_factory: Arc<dyn TensorFactory> = Arc::new(ferrum_testkit::MockTensorFactory);
    let model_executor: Arc<dyn ModelExecutor + Send + Sync> =
        Arc::new(ferrum_testkit::MockModelExecutor::instant(128));

    ContinuousBatchEngine::new(
        config,
        scheduler,
        tokenizer,
        sampler,
        kv_cache,
        model_executor,
        tensor_factory,
    )
}

#[tokio::test]
async fn executor_managed_product_path_requires_typed_admission_before_prefill() {
    let mut config = EngineConfig::default();
    config.kv_cache.max_blocks = 128;
    let scheduler = Arc::new(ContinuousBatchScheduler::new(config.scheduler.clone()));
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> =
        Arc::new(ferrum_testkit::MockTokenizer::new(128));
    let sampler: Arc<dyn Sampler + Send + Sync> = Arc::new(ferrum_testkit::MockSampler);
    let kv_cache: Arc<dyn KvCacheManager + Send + Sync> = Arc::new(MockKvCacheManager::new(128));
    let executor = Arc::new(ExecutorManagedAdmissionTestExecutor::new(128));
    let tensor_factory: Arc<dyn TensorFactory> = Arc::new(MockTensorFactory);
    let engine = ContinuousBatchEngine::new(
        config,
        Arc::clone(&scheduler),
        tokenizer,
        sampler,
        kv_cache,
        executor.clone(),
        tensor_factory,
    );
    let mut request = policy_request();
    request.sampling_params.max_tokens = 1;

    let response = engine.infer(request).await.unwrap();

    assert_eq!(response.finish_reason, FinishReason::Length);
    assert_eq!(executor.admission_probes.load(Ordering::Relaxed), 1);
    assert_eq!(executor.prefill_calls.load(Ordering::Relaxed), 1);
    assert!(executor
        .retained
        .lock()
        .expect("retained admission mutex poisoned")
        .is_empty());
    let trace = scheduler.trace_snapshot();
    assert_eq!(trace.legacy_waiting_admission_ticks, 0);
    assert!(trace.dynamic_admission_ticks >= 1);
}

#[tokio::test]
async fn executor_managed_backing_maintenance_advances_epoch_before_prefill() {
    let trace_path = resource_trace_temp_path("executor-prefill-maintenance");
    let _ = std::fs::remove_file(&trace_path);
    let mut config = EngineConfig::default();
    config.runtime.scheduler_trace_jsonl = Some(trace_path.clone());
    config.runtime.profile_entrypoint = Some(ProfileEntrypoint::Serve);
    config.kv_cache.max_blocks = 128;
    let scheduler = Arc::new(ContinuousBatchScheduler::new(config.scheduler.clone()));
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> =
        Arc::new(ferrum_testkit::MockTokenizer::new(128));
    let sampler: Arc<dyn Sampler + Send + Sync> = Arc::new(ferrum_testkit::MockSampler);
    let kv_cache: Arc<dyn KvCacheManager + Send + Sync> = Arc::new(MockKvCacheManager::new(128));
    let executor = Arc::new(ExecutorManagedMaintenanceTestExecutor::new(128));
    let tensor_factory: Arc<dyn TensorFactory> = Arc::new(MockTensorFactory);
    let engine = ContinuousBatchEngine::new(
        config,
        Arc::clone(&scheduler),
        tokenizer,
        sampler,
        kv_cache,
        executor.clone(),
        tensor_factory,
    );
    let mut request = policy_request();
    request.sampling_params.max_tokens = 1;
    let request_id = request.id.clone();

    let response = engine.infer(request).await.unwrap();

    assert_eq!(response.finish_reason, FinishReason::Length);
    assert_eq!(executor.admission_probes.load(Ordering::Relaxed), 2);
    assert_eq!(executor.maintenance_calls.load(Ordering::Relaxed), 1);
    assert_eq!(executor.prefill_calls.load(Ordering::Relaxed), 1);
    assert_eq!(
        *executor
            .call_order
            .lock()
            .expect("call order mutex poisoned"),
        vec!["defer", "maintain", "admit", "prefill"]
    );
    assert!(executor
        .retained
        .lock()
        .expect("retained admission mutex poisoned")
        .is_empty());
    let snapshot = scheduler.trace_snapshot();
    assert_eq!(snapshot.legacy_waiting_admission_ticks, 0);
    assert_eq!(snapshot.dynamic_backing_growth_requested, 1);
    assert_eq!(snapshot.dynamic_admission_deferred, 1);

    let events = read_engine_profile_events(&trace_path);
    let deferred_index = events
        .iter()
        .position(|event| {
            event.request_id == request_id.to_string()
                && event.phase == "vnext.prefill_admission"
                && event.shape.get("decision") == Some(&serde_json::json!("maintenance_deferred"))
        })
        .expect("maintenance deferral trace event");
    let maintenance_index = events
        .iter()
        .position(|event| {
            event.request_id == request_id.to_string()
                && event.phase == "vnext.prefill_backing_maintenance"
                && event.shape.get("outcome") == Some(&serde_json::json!("maintained"))
        })
        .expect("maintenance completion trace event");
    let admitted_index = events
        .iter()
        .position(|event| {
            event.request_id == request_id.to_string()
                && event.phase == "vnext.prefill_admission"
                && event.shape.get("decision") == Some(&serde_json::json!("admitted"))
        })
        .expect("admitted trace event");
    assert!(deferred_index < maintenance_index);
    assert!(maintenance_index < admitted_index);
    assert_eq!(
        events[deferred_index].shape.get("prefill_submit_observed"),
        Some(&serde_json::json!(false))
    );
    assert!(events[deferred_index]
        .attributes
        .get("monotonic_nanos")
        .and_then(serde_json::Value::as_u64)
        .is_some());

    let _ = std::fs::remove_file(trace_path);
}

#[tokio::test]
async fn executor_managed_capacity_pressure_waits_without_spinning_then_retries_on_release() {
    let mut config = EngineConfig::default();
    config.kv_cache.max_blocks = 128;
    let scheduler = Arc::new(ContinuousBatchScheduler::new(config.scheduler.clone()));
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> =
        Arc::new(ferrum_testkit::MockTokenizer::new(128));
    let sampler: Arc<dyn Sampler + Send + Sync> = Arc::new(ferrum_testkit::MockSampler);
    let kv_cache: Arc<dyn KvCacheManager + Send + Sync> = Arc::new(MockKvCacheManager::new(128));
    let executor = Arc::new(ExecutorManagedMaintenanceTestExecutor::with_behavior(
        128,
        TestMaintenanceBehavior::WaitForRelease,
    ));
    let tensor_factory: Arc<dyn TensorFactory> = Arc::new(MockTensorFactory);
    let engine = Arc::new(ContinuousBatchEngine::new(
        config,
        Arc::clone(&scheduler),
        tokenizer,
        sampler,
        kv_cache,
        executor.clone(),
        tensor_factory,
    ));
    let mut request = policy_request();
    request.sampling_params.max_tokens = 1;
    let infer_engine = Arc::clone(&engine);
    let inference = tokio::spawn(async move { infer_engine.infer(request).await });

    tokio::time::timeout(Duration::from_secs(1), async {
        while executor.maintenance_calls.load(Ordering::Acquire) == 0 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("typed capacity maintenance must run");
    tokio::time::sleep(Duration::from_millis(20)).await;
    assert_eq!(executor.admission_probes.load(Ordering::Relaxed), 1);
    assert_eq!(executor.maintenance_calls.load(Ordering::Relaxed), 1);
    assert_eq!(executor.prefill_calls.load(Ordering::Relaxed), 0);
    assert_eq!(scheduler.waiting_count(), 1);
    assert_eq!(scheduler.trace_snapshot().dynamic_admission_failed, 0);

    executor.release_epoch.store(1, Ordering::Release);
    let response = tokio::time::timeout(Duration::from_secs(2), inference)
        .await
        .expect("release epoch must wake the waiting request")
        .expect("inference task must not panic")
        .unwrap();
    assert_eq!(response.finish_reason, FinishReason::Length);
    assert_eq!(executor.admission_probes.load(Ordering::Relaxed), 2);
    assert_eq!(executor.maintenance_calls.load(Ordering::Relaxed), 1);
    assert_eq!(executor.prefill_calls.load(Ordering::Relaxed), 1);
    assert_eq!(scheduler.waiting_count(), 0);
    assert_eq!(
        *executor
            .call_order
            .lock()
            .expect("call order mutex poisoned"),
        vec!["defer", "maintain", "admit", "prefill"]
    );
    engine.shutdown().await.unwrap();
}

#[tokio::test]
async fn executor_managed_invalid_maintenance_fails_waiting_request_without_retry_loop() {
    for behavior in [
        TestMaintenanceBehavior::StaleEpoch,
        TestMaintenanceBehavior::Fail,
    ] {
        let mut config = EngineConfig::default();
        config.kv_cache.max_blocks = 128;
        let scheduler = Arc::new(ContinuousBatchScheduler::new(config.scheduler.clone()));
        let tokenizer: Arc<dyn Tokenizer + Send + Sync> =
            Arc::new(ferrum_testkit::MockTokenizer::new(128));
        let sampler: Arc<dyn Sampler + Send + Sync> = Arc::new(ferrum_testkit::MockSampler);
        let kv_cache: Arc<dyn KvCacheManager + Send + Sync> =
            Arc::new(MockKvCacheManager::new(128));
        let executor = Arc::new(ExecutorManagedMaintenanceTestExecutor::with_behavior(
            128, behavior,
        ));
        let tensor_factory: Arc<dyn TensorFactory> = Arc::new(MockTensorFactory);
        let engine = ContinuousBatchEngine::new(
            config,
            Arc::clone(&scheduler),
            tokenizer,
            sampler,
            kv_cache,
            executor.clone(),
            tensor_factory,
        );
        let mut request = policy_request();
        request.sampling_params.max_tokens = 1;

        let response = tokio::time::timeout(Duration::from_secs(2), engine.infer(request))
            .await
            .expect("invalid maintenance must not leave the request waiting forever")
            .unwrap();

        assert_eq!(response.finish_reason, FinishReason::Error);
        assert_eq!(executor.admission_probes.load(Ordering::Relaxed), 1);
        assert_eq!(executor.maintenance_calls.load(Ordering::Relaxed), 1);
        assert_eq!(executor.prefill_calls.load(Ordering::Relaxed), 0);
        assert_eq!(scheduler.waiting_count(), 0);
        assert_eq!(scheduler.trace_snapshot().dynamic_admission_failed, 1);
        assert!(executor
            .retained
            .lock()
            .expect("retained admission mutex poisoned")
            .is_empty());
        engine.shutdown().await.unwrap();
    }
}

#[test]
#[should_panic(expected = "request slot lease dropped without explicit reject or close")]
fn request_slot_lease_drop_without_consumption_panics_in_tests() {
    let engine = test_continuous_engine();
    let _lease = RequestSlotLease::open(&engine.inner, RequestId::new());
}

#[test]
#[should_panic(expected = "sequence state dropped with owned request slot")]
fn sequence_state_drop_with_owned_request_slot_panics_in_tests() {
    let engine = test_continuous_engine();
    let request = policy_request();
    let mut sequence = SequenceState::new(request.clone(), vec![TokenId::new(1)]);
    sequence.request_slot = Some(RequestSlotLease::open(&engine.inner, request.id));
}

#[test]
#[should_panic(expected = "unified prefill resources dropped without explicit release or commit")]
fn unified_prefill_owned_resources_drop_without_release_or_commit_panics_in_tests() {
    let _resources = UnifiedPrefillOwnedResources::default().with_fresh_kv(RequestId::new(), 1);
}

#[test]
fn unified_prefill_owned_resources_commit_consumes_transaction() {
    let mut resources = UnifiedPrefillOwnedResources::default().with_fresh_kv(RequestId::new(), 1);
    assert!(!resources.is_empty());

    std::mem::take(&mut resources).commit();

    assert!(resources.is_empty());
}

#[test]
fn sequence_take_completion_resources_moves_request_slot_and_physical_resources_together() {
    let engine = test_continuous_engine();
    let request = policy_request();
    let request_id = request.id.clone();
    let mut sequence = SequenceState::new(request, vec![TokenId::new(1)]);
    let model_kv: Arc<dyn KvCacheHandle> = Arc::new(ferrum_testkit::MockKvCacheHandle::new(
        request_id.clone(),
        1,
        1,
    ));
    let model_cache_id = model_kv.cache_id();
    sequence.install_model_kv(model_kv, Some(2));

    let mut request_slot = RequestSlotLease::open(&engine.inner, request_id.clone());
    request_slot.admit(&engine.inner);
    sequence.request_slot = Some(request_slot);

    let completion_resources = sequence.take_completion_resources();

    assert_eq!(
        completion_resources.physical.kv_allocation,
        Some(SequenceKvAllocation::new(request_id.clone(), Some(2)))
    );
    assert_eq!(
        completion_resources.physical.model_cache_id(),
        Some(model_cache_id.as_str())
    );
    assert!(completion_resources.request_slot.is_some());
    assert!(sequence.kv_cache_handle().is_none());
    assert!(sequence.model_cache_id().is_none());
    assert!(sequence.request_slot.is_none());

    completion_resources
        .request_slot
        .expect("request slot")
        .close(&engine.inner);
}

#[tokio::test]
async fn sequence_take_physical_resources_for_recompute_clears_owned_resources() {
    let request = policy_request();
    let request_id = request.id.clone();
    let draft_request_id = RequestId::new();
    let mut sequence = SequenceState::new(request, vec![TokenId::new(1)]);
    let model_kv: Arc<dyn KvCacheHandle> = Arc::new(ferrum_testkit::MockKvCacheHandle::new(
        request_id.clone(),
        1,
        1,
    ));
    let model_cache_id = model_kv.cache_id();
    sequence.install_model_kv(model_kv, Some(2));
    sequence.commit_draft_kv_allocation(
        Arc::new(ferrum_testkit::MockKvCacheHandle::new(
            draft_request_id.clone(),
            1,
            1,
        )),
        draft_request_id.clone(),
        3,
    );
    sequence.prefill_complete = true;
    sequence.phase = RequestPhase::Decoding;
    sequence.tokens_this_iteration = 4;

    let recurrent_manager = InMemoryRecurrentStateManager::new(InMemoryRecurrentStateConfig {
        total_memory_bytes: 8,
        total_batch_slots: 1,
    });
    let recurrent_spec = RecurrentStateSpec {
        request_id: request_id.clone(),
        num_layers: 1,
        tensors: vec![RecurrentStateTensorSpec::new(0, "state", vec![1])],
        dtype: DataType::FP32,
        device: Device::CPU,
        max_batch_slots: 1,
    };
    let recurrent_state = recurrent_manager.allocate(&recurrent_spec).await.unwrap();
    sequence.commit_recurrent_state_admission(recurrent_state, 1);

    let resources = sequence.take_physical_resources_for_recompute();

    assert_eq!(
        resources.kv_allocation,
        Some(SequenceKvAllocation::new(request_id.clone(), Some(2)))
    );
    assert_eq!(
        resources.draft_kv_allocation,
        Some(SequenceKvAllocation::new(draft_request_id, Some(3)))
    );
    assert_eq!(
        resources.recurrent_state_allocation,
        Some(SequenceRecurrentAllocation::new(Some(1)))
    );
    assert_eq!(resources.model_cache_id(), Some(model_cache_id.as_str()));
    assert!(sequence.kv_cache_handle().is_none());
    assert!(sequence.kv_resource_blocks().is_none());
    assert!(sequence.draft_kv.is_none());
    assert!(sequence.recurrent_state.is_none());
    assert!(sequence.recurrent_state_slots().is_none());
    assert!(sequence.model_cache_id().is_none());
    assert!(!sequence.prefill_complete);
    assert_eq!(sequence.phase, RequestPhase::Waiting);
    assert_eq!(sequence.tokens_this_iteration, 0);
}

#[test]
fn sequence_take_physical_resources_skips_manager_without_owned_blocks() {
    let request = policy_request();
    let request_id = request.id.clone();
    let mut sequence = SequenceState::new(request, vec![TokenId::new(1)]);
    let model_kv: Arc<dyn KvCacheHandle> = Arc::new(ferrum_testkit::MockKvCacheHandle::new(
        request_id.clone(),
        1,
        1,
    ));
    let model_cache_id = model_kv.cache_id();

    sequence.install_model_kv_without_owned_blocks(model_kv);
    let resources = sequence.take_physical_resources();

    assert!(resources.kv_allocation.is_none());
    assert!(resources.draft_kv_allocation.is_none());
    assert!(resources.recurrent_state_allocation.is_none());
    assert_eq!(resources.model_cache_id(), Some(model_cache_id.as_str()));
    assert!(sequence.kv_cache_handle().is_none());
    assert!(sequence.kv_resource_blocks().is_none());
}

#[test]
#[should_panic(expected = "draft KV cache updated without owned allocation metadata")]
fn sequence_speculative_decode_commit_rejects_draft_kv_without_allocation_metadata() {
    let request = policy_request();
    let request_id = request.id.clone();
    let mut sequence = SequenceState::new(request, vec![TokenId::new(1)]);
    let target_kv: Arc<dyn KvCacheHandle> =
        Arc::new(ferrum_testkit::MockKvCacheHandle::new(request_id, 1, 1));
    let draft_kv: Arc<dyn KvCacheHandle> = Arc::new(ferrum_testkit::MockKvCacheHandle::new(
        RequestId::new(),
        1,
        1,
    ));

    sequence.commit_speculative_decode_physical_resources(target_kv, draft_kv);
}

#[test]
fn sequence_prefill_commit_helpers_keep_resource_metadata_together() {
    let request = policy_request();
    let request_id = request.id.clone();
    let mut sequence = SequenceState::new(request, vec![TokenId::new(1)]);
    let model_kv: Arc<dyn KvCacheHandle> = Arc::new(ferrum_testkit::MockKvCacheHandle::new(
        request_id.clone(),
        1,
        1,
    ));
    let model_cache_id = model_kv.cache_id();

    sequence.install_model_kv_without_owned_blocks(model_kv.clone());
    assert!(sequence.kv_cache_handle().is_some());
    assert_eq!(sequence.model_cache_id(), Some(model_cache_id.as_str()));
    assert!(sequence.kv_resource_blocks().is_none());

    let cached_kv: Arc<dyn KvCacheHandle> = Arc::new(ferrum_testkit::MockKvCacheHandle::new(
        RequestId::new(),
        1,
        2,
    ));
    let cached_cache_id = cached_kv.cache_id();
    sequence.commit_cached_prefill_physical_resources(cached_kv, 1);
    assert_eq!(sequence.model_cache_id(), Some(cached_cache_id.as_str()));
    assert!(sequence.kv_cache_handle().is_some());
    assert_eq!(sequence.kv_resource_blocks(), None);
    assert_eq!(sequence.prefill_tokens_processed, 1);
    assert!(sequence.prefill_complete);
    assert_eq!(sequence.phase, RequestPhase::Decoding);

    let owned_kv: Arc<dyn KvCacheHandle> = Arc::new(ferrum_testkit::MockKvCacheHandle::new(
        request_id.clone(),
        1,
        2,
    ));
    let owned_cache_id = owned_kv.cache_id();
    sequence.commit_prefill_physical_resources(owned_kv.clone(), 4, None, None);

    assert_eq!(sequence.model_cache_id(), Some(owned_cache_id.as_str()));
    assert!(sequence.kv_cache_handle().is_some());
    assert_eq!(sequence.kv_resource_blocks(), Some(4));
    assert!(sequence.recurrent_state.is_none());
    assert!(sequence.recurrent_state_slots().is_none());
    assert!(sequence.prefill_complete);
    assert_eq!(sequence.phase, RequestPhase::Decoding);

    let resources = sequence.prefill_resources();
    assert!(resources.kv_cache_handle().is_some());
    assert_eq!(resources.kv_resource_blocks(), Some(4));
    assert!(resources.recurrent_state.is_none());
    assert_eq!(
        resources.prefill_tokens_processed,
        sequence.prefill_tokens_processed
    );
}

#[test]
fn sequence_prefill_chunk_commit_tracks_partial_and_final_state() {
    let request = policy_request();
    let request_id = request.id.clone();
    let mut sequence = SequenceState::new(request, vec![TokenId::new(1), TokenId::new(2)]);

    let partial_kv: Arc<dyn KvCacheHandle> = Arc::new(ferrum_testkit::MockKvCacheHandle::new(
        RequestId::new(),
        1,
        2,
    ));
    let partial_cache_id = partial_kv.cache_id();
    let recurrent_state = test_recurrent_state_handle(request_id.clone(), 3);
    sequence.commit_recurrent_state_admission(recurrent_state, 3);
    sequence.commit_prefill_chunk_physical_resources(partial_kv, Some(2), None, 1, false);

    assert_eq!(sequence.model_cache_id(), Some(partial_cache_id.as_str()));
    assert!(sequence.kv_cache_handle().is_some());
    assert_eq!(sequence.kv_resource_blocks(), Some(2));
    assert!(sequence.recurrent_state.is_none());
    assert!(sequence.recurrent_state_slots().is_none());
    assert_eq!(sequence.prefill_tokens_processed, 1);
    assert!(!sequence.prefill_complete);
    assert_eq!(sequence.phase, RequestPhase::Prefilling);

    let final_kv: Arc<dyn KvCacheHandle> =
        Arc::new(ferrum_testkit::MockKvCacheHandle::new(request_id, 1, 4));
    let final_cache_id = final_kv.cache_id();
    sequence.commit_prefill_chunk_physical_resources(final_kv, None, None, 2, true);

    assert_eq!(sequence.model_cache_id(), Some(final_cache_id.as_str()));
    assert_eq!(sequence.kv_resource_blocks(), None);
    assert_eq!(sequence.prefill_tokens_processed, 2);
    assert!(sequence.prefill_complete);
    assert_eq!(sequence.phase, RequestPhase::Decoding);
}

#[test]
fn sequence_decode_commit_helpers_keep_resource_metadata_together() {
    let request = policy_request();
    let request_id = request.id.clone();
    let mut sequence = SequenceState::new(request, vec![TokenId::new(1), TokenId::new(2)]);

    assert_eq!(
        sequence.decode_model_cache_id_or_request_id(&request_id),
        request_id.to_string()
    );
    let decode_kv: Arc<dyn KvCacheHandle> = Arc::new(ferrum_testkit::MockKvCacheHandle::new(
        request_id.clone(),
        1,
        2,
    ));
    let decode_cache_id = decode_kv.cache_id();
    sequence.install_model_kv_without_owned_blocks(decode_kv.clone());
    sequence.generated_tokens.push(TokenId::new(7));
    assert_eq!(
        sequence.decode_model_cache_id_or_request_id(&request_id),
        decode_cache_id
    );
    assert_eq!(sequence.decode_model_kv_len_after_last_generated_token(), 2);

    sequence.commit_decode_step_physical_resources(decode_kv);
    assert!(sequence.kv_cache_handle().is_some());
    assert_eq!(sequence.tokens_this_iteration, 1);
    assert_eq!(sequence.model_cache_id(), Some(decode_cache_id.as_str()));
    let decode_resources = sequence
        .decode_resources(&request_id)
        .expect("decode resources");
    assert_eq!(decode_resources.seq_id, decode_cache_id);
    assert_eq!(decode_resources.last_token, TokenId::new(7));
    assert_eq!(decode_resources.pos_offset, 2);

    let recurrent_state = test_recurrent_state_handle(request_id.clone(), 1);
    sequence.commit_recurrent_state_admission(recurrent_state, 1);
    sequence.commit_decode_recurrent_state(None);
    assert!(sequence.recurrent_state.is_none());
    assert!(sequence.recurrent_state_slots().is_none());

    let target_kv: Arc<dyn KvCacheHandle> = Arc::new(ferrum_testkit::MockKvCacheHandle::new(
        request_id.clone(),
        1,
        3,
    ));
    let draft_request_id = RequestId::new();
    let draft_kv: Arc<dyn KvCacheHandle> = Arc::new(ferrum_testkit::MockKvCacheHandle::new(
        draft_request_id.clone(),
        1,
        3,
    ));
    sequence.commit_draft_kv_allocation(draft_kv.clone(), draft_request_id.clone(), 0);
    sequence.commit_speculative_decode_physical_resources(target_kv, draft_kv);
    assert!(sequence.kv_cache_handle().is_some());
    let draft = sequence.draft_kv.as_ref().expect("draft kv state");
    assert_eq!(draft.request_id, draft_request_id);
    assert_eq!(draft.resource_blocks, 1);
    assert!(sequence.draft_kv_cache_handle().is_some());
}

#[tokio::test]
async fn sequence_recurrent_admission_helpers_keep_handle_and_slots_together() {
    let request = policy_request();
    let request_id = request.id.clone();
    let mut sequence = SequenceState::new(request, vec![TokenId::new(1)]);
    let recurrent_manager = InMemoryRecurrentStateManager::new(InMemoryRecurrentStateConfig {
        total_memory_bytes: 8,
        total_batch_slots: 1,
    });
    let recurrent_spec = RecurrentStateSpec {
        request_id,
        num_layers: 1,
        tensors: vec![RecurrentStateTensorSpec::new(0, "state", vec![1])],
        dtype: DataType::FP32,
        device: Device::CPU,
        max_batch_slots: 1,
    };
    let recurrent_state = recurrent_manager.allocate(&recurrent_spec).await.unwrap();

    sequence.commit_recurrent_state_admission(recurrent_state, 1);
    assert!(sequence.recurrent_state.is_some());
    assert_eq!(sequence.recurrent_state_slots(), Some(1));

    let slots = sequence.take_recurrent_state_allocation();
    assert_eq!(slots, Some(1));
    assert!(sequence.recurrent_state.is_none());
    assert!(sequence.recurrent_state_slots().is_none());
}

#[test]
#[should_panic(expected = "KV allocation lease dropped without explicit commit or async release")]
fn kv_allocation_lease_drop_without_consumption_panics_in_tests() {
    let request_id = RequestId::new();
    let handle: Arc<dyn KvCacheHandle> = Arc::new(ferrum_testkit::MockKvCacheHandle::new(
        request_id.clone(),
        1,
        1,
    ));

    let _lease = KvAllocationLease::new(request_id.clone(), request_id, handle, 1);
}

#[tokio::test]
async fn kv_release_failure_traces_reject_without_successful_release() {
    let trace_path = resource_trace_temp_path("kv-release-failure");
    let _ = std::fs::remove_file(&trace_path);
    let mut config = EngineConfig::default();
    config.runtime.scheduler_trace_jsonl = Some(trace_path.clone());

    let scheduler = Arc::new(ContinuousBatchScheduler::new(config.scheduler.clone()));
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> =
        Arc::new(ferrum_testkit::MockTokenizer::new(128));
    let sampler: Arc<dyn Sampler + Send + Sync> = Arc::new(ferrum_testkit::MockSampler);
    let kv_cache: Arc<dyn KvCacheManager + Send + Sync> =
        Arc::new(FailingDeallocateKvCacheManager::new(256));
    let tensor_factory: Arc<dyn TensorFactory> = Arc::new(ferrum_testkit::MockTensorFactory);
    let model_executor: Arc<dyn ModelExecutor + Send + Sync> =
        Arc::new(ferrum_testkit::MockModelExecutor::instant(128));
    let engine = ContinuousBatchEngine::new(
        config,
        scheduler,
        tokenizer,
        sampler,
        kv_cache,
        model_executor,
        tensor_factory,
    );

    let request_id = RequestId::new();
    engine.inner.trace_kv_allocate(&request_id, 2);
    engine
        .inner
        .release_kv_allocation(&request_id, request_id.clone(), Some(2))
        .await;

    let resources: Vec<_> = read_engine_profile_events(&trace_path)
        .into_iter()
        .filter_map(|event| event.resource)
        .collect();
    assert!(
        !resources
            .iter()
            .any(|resource| resource.resource_kind == "kv_block"
                && resource.action == ResourceAction::Release),
        "failed deallocate must not be traced as a successful release: {resources:?}"
    );
    let reject = resources
        .iter()
        .find(|resource| {
            resource.resource_kind == "kv_block" && resource.action == ResourceAction::Reject
        })
        .expect("failed KV release should be traced as resource reject");
    assert!(
        reject
            .reason
            .as_deref()
            .is_some_and(|reason| reason.contains("kv release failed")),
        "reject reason should explain release failure: {reject:?}"
    );
    let _ = std::fs::remove_file(trace_path);
}

#[tokio::test]
async fn recurrent_release_failure_traces_reject_without_successful_release() {
    let trace_path = resource_trace_temp_path("recurrent-release-failure");
    let _ = std::fs::remove_file(&trace_path);
    let mut config = EngineConfig::default();
    config.runtime.scheduler_trace_jsonl = Some(trace_path.clone());

    let scheduler = Arc::new(ContinuousBatchScheduler::new(config.scheduler.clone()));
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> =
        Arc::new(ferrum_testkit::MockTokenizer::new(128));
    let sampler: Arc<dyn Sampler + Send + Sync> = Arc::new(ferrum_testkit::MockSampler);
    let kv_cache: Arc<dyn KvCacheManager + Send + Sync> =
        Arc::new(ferrum_testkit::MockKvCacheManager::new(256));
    let tensor_factory: Arc<dyn TensorFactory> = Arc::new(ferrum_testkit::MockTensorFactory);
    let model_executor: Arc<dyn ModelExecutor + Send + Sync> =
        Arc::new(ferrum_testkit::MockModelExecutor::instant(128));
    let recurrent_manager: Arc<dyn RecurrentStateManager + Send + Sync> = Arc::new(
        FailingDeallocateRecurrentStateManager::new(InMemoryRecurrentStateConfig {
            total_memory_bytes: 8,
            total_batch_slots: 1,
        }),
    );
    let engine = ContinuousBatchEngine::new_with_speculation_and_recurrent_state_manager(
        config,
        scheduler,
        tokenizer,
        sampler,
        kv_cache,
        model_executor,
        tensor_factory,
        None,
        None,
        Some(recurrent_manager),
    );

    let request_id = RequestId::new();
    engine
        .inner
        .trace_recurrent_allocate(&request_id, 1, Some(1));
    engine
        .inner
        .release_recurrent_allocation(&request_id, Some(1))
        .await;

    let resources: Vec<_> = read_engine_profile_events(&trace_path)
        .into_iter()
        .filter_map(|event| event.resource)
        .collect();
    assert!(
        !resources
            .iter()
            .any(|resource| resource.resource_kind == "recurrent_state_slot"
                && resource.action == ResourceAction::Release),
        "failed recurrent deallocate must not be traced as a successful release: {resources:?}"
    );
    let reject = resources
        .iter()
        .find(|resource| {
            resource.resource_kind == "recurrent_state_slot"
                && resource.action == ResourceAction::Reject
        })
        .expect("failed recurrent release should be traced as resource reject");
    assert!(
        reject
            .reason
            .as_deref()
            .is_some_and(|reason| reason.contains("recurrent-state release failed")),
        "reject reason should explain release failure: {reject:?}"
    );
    let _ = std::fs::remove_file(trace_path);
}

#[test]
fn request_owner_close_event_reports_outstanding_resources() {
    let trace_path = resource_trace_temp_path("request-owner-close-outstanding");
    let _ = std::fs::remove_file(&trace_path);
    let mut config = EngineConfig::default();
    config.runtime.scheduler_trace_jsonl = Some(trace_path.clone());

    let engine = test_continuous_engine_with_config(config);
    let request_id = RequestId::new();

    engine.inner.trace_request_open(&request_id);
    engine.inner.trace_request_admitted(&request_id);
    engine.inner.trace_request_owner_close(&request_id);

    let events = read_engine_profile_events(&trace_path);
    let close = events
        .iter()
        .find(|event| {
            event
                .resource
                .as_ref()
                .is_some_and(|resource| resource.action == ResourceAction::RequestClose)
        })
        .expect("request close event should be present");

    assert_eq!(close.status, ProfileStatus::Failure);
    assert_eq!(
        close.error.as_ref().map(|error| error.kind.as_str()),
        Some("resource_owner_close_outstanding")
    );
    assert_eq!(
        close
            .resource
            .as_ref()
            .and_then(|resource| resource.resource_error_kind.as_deref()),
        Some("resource_leak")
    );
    assert_eq!(
        close
            .attributes
            .get("resource_owner_outstanding_count")
            .and_then(serde_json::Value::as_u64),
        Some(1)
    );
    let summary = close
        .attributes
        .get("resource_owner_close_summary")
        .and_then(serde_json::Value::as_array)
        .expect("close summary should be an array");
    assert!(
        summary.iter().any(|item| {
            item.get("resource_kind")
                .and_then(serde_json::Value::as_str)
                == Some("request_slot")
                && item
                    .get("outstanding_reserved")
                    .and_then(serde_json::Value::as_i64)
                    == Some(1)
                && item
                    .get("outstanding_committed")
                    .and_then(serde_json::Value::as_i64)
                    == Some(1)
        }),
        "close summary must identify the leaked request_slot: {summary:?}"
    );

    let _ = std::fs::remove_file(trace_path);
}

#[tokio::test]
#[should_panic(expected = "recurrent-state lease dropped without explicit commit or async release")]
async fn recurrent_state_lease_drop_without_consumption_panics_in_tests() {
    let request_id = RequestId::new();
    let manager = InMemoryRecurrentStateManager::new(InMemoryRecurrentStateConfig {
        total_memory_bytes: 8,
        total_batch_slots: 1,
    });
    let spec = RecurrentStateSpec {
        request_id: request_id.clone(),
        num_layers: 1,
        tensors: vec![RecurrentStateTensorSpec::new(0, "state", vec![1])],
        dtype: DataType::FP32,
        device: Device::CPU,
        max_batch_slots: 1,
    };
    let handle = manager.allocate(&spec).await.unwrap();

    let _lease = RecurrentStateLease::new(request_id, handle, 1, Some(1));
}

#[test]
fn model_kv_handle_with_seq_is_executor_decode_handle() {
    let engine = test_continuous_engine();

    let handle = engine
        .inner
        .make_model_kv_handle_with_seq("cache-a".to_string(), 17);
    let generic = handle
        .as_any()
        .downcast_ref::<ferrum_models::executor::common::GenericKvCacheHandle>()
        .expect("model KV handle must be GenericKvCacheHandle");

    assert_eq!(generic.request_cache_id(), "cache-a");
    assert_eq!(handle.block_table().sequence_length, 17);
}

#[test]
fn decode_ready_request_ids_skip_preempted_sequences_without_kv() {
    let engine = test_continuous_engine();
    let ready_request = policy_request();
    let ready_id = ready_request.id.clone();
    let preempted_request = policy_request();
    let preempted_id = preempted_request.id.clone();

    let ready_kv = engine
        .inner
        .make_model_kv_handle_with_seq("ready-cache".to_string(), 2);
    let mut ready_seq = SequenceState::new(ready_request, vec![TokenId::new(1)]);
    ready_seq.generated_tokens.push(TokenId::new(2));
    ready_seq.prefill_complete = true;
    ready_seq.install_model_kv_without_owned_blocks(ready_kv);

    let mut preempted_seq = SequenceState::new(preempted_request, vec![TokenId::new(1)]);
    preempted_seq.generated_tokens.push(TokenId::new(2));
    preempted_seq.prefill_complete = false;

    {
        let mut sequences = engine.inner.sequences.write();
        sequences.insert(ready_id.clone(), ready_seq);
        sequences.insert(preempted_id.clone(), preempted_seq);
    }

    let ready = engine
        .inner
        .decode_ready_request_ids(&[ready_id.clone(), preempted_id]);

    assert_eq!(ready, vec![ready_id]);
}

#[tokio::test]
async fn scheduler_trace_plan_stats_reports_request_details() {
    let engine = test_continuous_engine();
    let request = policy_request();
    let request_id = request.id.clone();

    engine
        .inner
        .scheduler
        .submit(request.clone())
        .await
        .unwrap();
    let batch = engine
        .inner
        .scheduler
        .next_batch(ferrum_interfaces::BatchHint {
            max_batch_size: 4,
            max_tokens: 4,
            target_latency_ms: None,
            available_memory: None,
            resource_constraints: Default::default(),
        })
        .await
        .expect("batch should schedule submitted request");

    let mut seq = SequenceState::new(
        request,
        vec![
            TokenId::new(10),
            TokenId::new(11),
            TokenId::new(12),
            TokenId::new(13),
        ],
    );
    seq.prefill_tokens_processed = 1;
    {
        let mut sequences = engine.inner.sequences.write();
        sequences.insert(request_id.clone(), seq);
    }

    let stats = engine.inner.scheduler_trace_plan_stats(&batch);
    assert_eq!(stats.batch_size, 1);
    assert_eq!(stats.prefill_items, 1);
    assert_eq!(stats.prefill_tokens, 4);
    assert_eq!(stats.requests.len(), 1);

    let request_stats = &stats.requests[0];
    assert_eq!(request_stats.request_id, request_id.to_string());
    assert_eq!(request_stats.phase.as_deref(), Some("Prefilling"));
    assert_eq!(request_stats.scheduled_tokens, 4);
    assert_eq!(request_stats.prompt_tokens, Some(4));
    assert_eq!(request_stats.generated_tokens, Some(0));
    assert_eq!(request_stats.prefill_tokens_processed, Some(1));
    assert_eq!(request_stats.prefill_tokens_remaining_before, Some(3));
    assert_eq!(request_stats.is_final_prefill_chunk, Some(true));
}

#[tokio::test]
async fn scheduler_trace_jsonl_resource_events_balance_successful_infer() {
    let trace_path = resource_trace_temp_path("successful-infer");
    let _ = std::fs::remove_file(&trace_path);
    let mut config = EngineConfig::default();
    config.runtime.scheduler_trace_jsonl = Some(trace_path.clone());
    config.runtime.profile_entrypoint = Some(ProfileEntrypoint::Run);
    config.kv_cache.max_blocks = 128;
    let engine = test_continuous_engine_with_config(config);

    let mut request = policy_request();
    request.sampling_params.max_tokens = 1;
    let request_id = request.id.clone();
    let response = engine.infer(request).await.unwrap();

    assert_eq!(response.request_id, request_id);
    assert_eq!(response.finish_reason, FinishReason::Length);

    let resources = assert_engine_resource_trace_balanced(&trace_path);
    let saw = |kind: &str, action: ResourceAction| {
        resources
            .iter()
            .any(|resource| resource.resource_kind == kind && resource.action == action)
    };

    assert!(saw("request_slot", ResourceAction::RequestOpen));
    assert!(saw("request_slot", ResourceAction::Reserve));
    assert!(saw("request_slot", ResourceAction::Commit));
    assert!(saw("request_slot", ResourceAction::Release));
    assert!(saw("request_slot", ResourceAction::RequestClose));
    assert!(saw("kv_block", ResourceAction::Reserve));
    assert!(saw("kv_block", ResourceAction::Commit));
    assert!(saw("kv_block", ResourceAction::Release));
    assert!(saw("model_cache_ref", ResourceAction::Reserve));
    assert!(saw("model_cache_ref", ResourceAction::Commit));
    assert!(saw("model_cache_ref", ResourceAction::Release));
    assert!(saw("backend_workspace", ResourceAction::Reserve));
    assert!(saw("backend_workspace", ResourceAction::Commit));
    assert!(saw("backend_workspace", ResourceAction::Release));

    let _ = std::fs::remove_file(trace_path);
}

#[test]
fn vnext_execution_events_use_the_canonical_scheduler_trace_schema() {
    use ferrum_interfaces::vnext::{
        ExecutionEventDetail, ExecutionEventEmitter, ExecutionEventKind, ExecutionIdentityEnvelope,
        ExecutionIdentityParts, ExecutionPhase, MonotonicTimestamp, RequestIdentity, RunId, SpanId,
        TrustedExecutionEventContext, EXECUTION_IDENTITY_VERSION,
    };

    let trace_path = resource_trace_temp_path("vnext-execution-event");
    let _ = std::fs::remove_file(&trace_path);
    let file = create_scheduler_trace_sink(Some(&trace_path)).unwrap();
    let config = EngineConfig::default();
    let sink = VNextProfileExecutionEventSink::new(file, ProfileEntrypoint::Run, &config);
    assert_eq!(
        sink.capture_policy(),
        ExecutionEventCapturePolicy::FirstFramePerRequest
    );
    assert_eq!(sink.capture_policy().as_str(), "first_frame_per_request");
    let run_id = RunId::new("run.vnext.engine-profile-test").unwrap();
    let request_id = RequestIdentity::new("request.vnext.engine-profile-test").unwrap();
    let event = ExecutionEvent::new(
        MonotonicTimestamp {
            nanos_since_run_start: 1,
        },
        ExecutionPhase::Resolution,
        ExecutionEventKind::RequestAccepted,
        ExecutionIdentityEnvelope::new(ExecutionIdentityParts {
            version: EXECUTION_IDENTITY_VERSION,
            run_id: run_id.clone(),
            request_id: request_id.clone(),
            sequence: 1,
            plan_id: None,
            plan_hash: None,
            frame_id: None,
            node_invocation_id: None,
            node_id: None,
            operation_id: None,
            provider_id: None,
            device_id: None,
            resource_pool_id: None,
            resource_pool_identity_fingerprint: None,
            provisioning_run_id: None,
            provisioning_request_id: None,
            transaction_id: None,
            active_sequence_slot: None,
            admission_generation: None,
            activation_epoch: None,
            runtime_implementation_fingerprint: None,
            active_sequence_fingerprint: None,
            completed_sequence_fingerprint: None,
            aborted_sequence_fingerprint: None,
            resource_id: None,
            resource_generation: None,
            resource_batch_fingerprint: None,
            span_id: SpanId::new("vnext/request/engine-profile-test").unwrap(),
            parent_span_id: None,
            async_links: Vec::new(),
        })
        .unwrap(),
        ExecutionEventDetail::None,
    )
    .unwrap();
    let mut emitter = ExecutionEventEmitter::new(&sink, run_id.clone(), request_id.clone());
    emitter
        .emit(
            &event,
            &TrustedExecutionEventContext::pre_plan(&run_id, &request_id),
        )
        .unwrap();

    let rows = std::fs::read_to_string(&trace_path).unwrap();
    let profile: FerrumProfileEvent = serde_json::from_str(rows.trim()).unwrap();
    profile.validate().unwrap();
    assert_eq!(profile.phase, "vnext.request_accepted");
    assert_eq!(profile.event_kind, ProfileEventKind::Instant);
    assert_eq!(profile.request_id, request_id.to_string());
    assert_eq!(
        profile.attributes.get("execution_trace_source"),
        Some(&serde_json::json!("vnext"))
    );
    assert_eq!(
        profile.attributes.get("execution_capture_policy"),
        Some(&serde_json::json!("first_frame_per_request"))
    );

    let _ = std::fs::remove_file(trace_path);
}

#[test]
fn request_context_capacity_uses_executor_kv_capacity_when_smaller() {
    let mut config = EngineConfig::default();
    config.kv_cache.max_blocks = 2048;
    let runtime = ContinuousEngineRuntimeConfig::from_env_vars(None, Vec::<(&str, &str)>::new());

    assert_eq!(
        effective_request_context_capacity(&config, &runtime, Some(512)),
        Some(512)
    );
}

#[test]
fn test_sequence_state() {
    let request = InferenceRequest {
        id: RequestId::new(),
        prompt: "test".to_string(),
        model_id: ferrum_types::ModelId::new("test"),
        sampling_params: SamplingParams::default(),
        stream: false,
        priority: Priority::Normal,
        client_id: None,
        session_id: None,
        created_at: chrono::Utc::now(),
        api_request: None,
        metadata: HashMap::new(),
    };

    let tokens = vec![TokenId::new(1), TokenId::new(2)];
    let state = SequenceState::new(request, tokens);

    assert_eq!(state.phase, RequestPhase::Waiting);
    assert_eq!(state.total_tokens(), 2);
    assert!(!state.prefill_complete);
    assert!(state.recurrent_state.is_none());
}

#[tokio::test]
async fn engine_allocates_and_deallocates_model_declared_recurrent_state() {
    let scheduler = Arc::new(ContinuousBatchScheduler::new(
        ferrum_types::SchedulerConfig::default(),
    ));
    let tokenizer = Arc::new(PolicyTokenizer::new(64, &[]));
    let sampler = Arc::new(crate::registry::GreedySampler);
    let kv_cache = Arc::new(MockKvCacheManager::new(128));
    let executor = Arc::new(RecurrentSpecExecutor {
        inner: MockModelExecutor::new(64, Duration::ZERO, Duration::ZERO),
    });
    let tensor_factory = Arc::new(MockTensorFactory);
    let recurrent_manager = Arc::new(InMemoryRecurrentStateManager::new(
        InMemoryRecurrentStateConfig {
            total_memory_bytes: 1024,
            total_batch_slots: 4,
        },
    ));
    let engine = ContinuousBatchEngine::new_with_speculation_and_recurrent_state_manager(
        EngineConfig::default(),
        scheduler,
        tokenizer,
        sampler,
        kv_cache,
        executor,
        tensor_factory,
        None,
        None,
        Some(recurrent_manager.clone()),
    );
    let mut request = policy_request();
    request.sampling_params.max_tokens = 1;

    let response = engine.infer(request).await.unwrap();

    assert_eq!(response.finish_reason, FinishReason::Length);
    let stats = recurrent_manager.stats();
    assert_eq!(stats.allocation_count, 1);
    assert_eq!(stats.allocation_failures, 0);
    assert_eq!(stats.active_states, 0);
    assert_eq!(stats.used_memory_bytes, 0);
}

#[tokio::test]
async fn run_iteration_cancels_disconnected_client_and_releases_recurrent_state() {
    let mut config = EngineConfig::default();
    config.kv_cache.max_blocks = 128;
    let scheduler = Arc::new(ContinuousBatchScheduler::new(config.scheduler.clone()));
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> = Arc::new(PolicyTokenizer::new(64, &[]));
    let sampler: Arc<dyn Sampler + Send + Sync> = Arc::new(crate::registry::GreedySampler);
    let kv_cache: Arc<dyn KvCacheManager + Send + Sync> = Arc::new(MockKvCacheManager::new(128));
    let executor: Arc<dyn ModelExecutor + Send + Sync> = Arc::new(RecurrentSpecExecutor {
        inner: MockModelExecutor::new(64, Duration::ZERO, Duration::ZERO),
    });
    let tensor_factory: Arc<dyn TensorFactory> = Arc::new(MockTensorFactory);
    let recurrent_manager = Arc::new(InMemoryRecurrentStateManager::new(
        InMemoryRecurrentStateConfig {
            total_memory_bytes: 1024,
            total_batch_slots: 1,
        },
    ));
    let engine = ContinuousBatchEngine::new_with_speculation_and_recurrent_state_manager(
        config,
        scheduler.clone(),
        tokenizer,
        sampler,
        kv_cache,
        executor,
        tensor_factory,
        None,
        None,
        Some(recurrent_manager.clone()),
    );

    let request = policy_request();
    let request_id = request.id.clone();
    scheduler.submit(request.clone()).await.unwrap();
    let recurrent_spec = RecurrentStateSpec {
        request_id: request_id.clone(),
        num_layers: 1,
        tensors: vec![RecurrentStateTensorSpec::new(0, "state", vec![1])],
        dtype: DataType::FP32,
        device: Device::CPU,
        max_batch_slots: 1,
    };
    let recurrent_state = recurrent_manager.allocate(&recurrent_spec).await.unwrap();
    let (stream_sender, stream_receiver) = tokio::sync::mpsc::channel(1);
    drop(stream_receiver);

    let mut sequence = SequenceState::new(request, vec![TokenId::new(1)]);
    sequence.stream_sender = Some(stream_sender);
    sequence.commit_recurrent_state_admission(recurrent_state, 1);
    let mut request_slot = RequestSlotLease::open(&engine.inner, request_id.clone());
    request_slot.admit(&engine.inner);
    sequence.request_slot = Some(request_slot);
    engine
        .inner
        .sequences
        .write()
        .insert(request_id.clone(), sequence);

    engine.inner.run_iteration().await.unwrap();

    assert!(!engine.inner.sequences.read().contains_key(&request_id));
    assert_eq!(scheduler.trace_phase(&request_id), None);
    let scheduler_stats = scheduler.trace_snapshot();
    assert_eq!(scheduler_stats.waiting_queue_len, 0);
    assert_eq!(scheduler_stats.cancelled_total, 1);
    assert_eq!(scheduler_stats.capacity_release_epoch, 1);
    let recurrent_stats = recurrent_manager.stats();
    assert_eq!(recurrent_stats.active_states, 0);
    assert_eq!(recurrent_stats.used_batch_slots, 0);
    assert_eq!(recurrent_stats.used_memory_bytes, 0);
}

#[test]
fn sequence_state_detects_closed_run_and_serve_receivers() {
    let mut run_sequence = SequenceState::new(policy_request(), vec![TokenId::new(1)]);
    let (response_sender, response_receiver) = tokio::sync::oneshot::channel();
    run_sequence.response_sender = Some(response_sender);
    assert!(!run_sequence.client_receiver_closed());
    drop(response_receiver);
    assert!(run_sequence.client_receiver_closed());

    let mut serve_sequence = SequenceState::new(policy_request(), vec![TokenId::new(1)]);
    let (stream_sender, stream_receiver) = tokio::sync::mpsc::channel(1);
    serve_sequence.stream_sender = Some(stream_sender);
    assert!(!serve_sequence.client_receiver_closed());
    drop(stream_receiver);
    assert!(serve_sequence.client_receiver_closed());
}

#[tokio::test]
async fn engine_status_reports_kv_cache_memory_from_manager_stats() {
    let mut config = EngineConfig::default();
    config.kv_cache.max_blocks = 128;
    let scheduler = Arc::new(ContinuousBatchScheduler::new(config.scheduler.clone()));
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> = Arc::new(PolicyTokenizer::new(64, &[]));
    let sampler: Arc<dyn Sampler + Send + Sync> = Arc::new(crate::registry::GreedySampler);
    let kv_cache = Arc::new(MockKvCacheManager::new(128));
    kv_cache
        .allocate(&AllocationRequest {
            request_id: RequestId::new(),
            initial_tokens: 17,
            max_sequence_length: 32,
            num_layers: 1,
            num_heads: 1,
            head_dim: 1,
            device: Device::CPU,
            dtype: DataType::FP16,
            priority: Priority::Normal,
        })
        .await
        .unwrap();
    let executor: Arc<dyn ModelExecutor + Send + Sync> = Arc::new(MockModelExecutor::instant(64));
    let tensor_factory: Arc<dyn TensorFactory> = Arc::new(MockTensorFactory);
    let engine = ContinuousBatchEngine::new(
        config,
        scheduler,
        tokenizer,
        sampler,
        kv_cache,
        executor,
        tensor_factory,
    );

    let status = engine.status().await;

    assert_eq!(status.memory_usage.total_bytes, 128 * 16 * 1024);
    assert_eq!(status.memory_usage.used_bytes, 2 * 16 * 1024);
    assert_eq!(status.memory_usage.cache_memory_bytes, 2 * 16 * 1024);
    assert_eq!(
        status.memory_usage.free_bytes,
        status.memory_usage.total_bytes - status.memory_usage.used_bytes
    );
    assert_eq!(status.memory_usage.cpu_memory_bytes, Some(2 * 16 * 1024));
    assert_eq!(status.memory_usage.gpu_memory_bytes, None);
    assert!(status.memory_usage.utilization_percent > 0.0);
}

#[tokio::test]
async fn engine_allocates_and_deallocates_llm_executor_declared_recurrent_state() {
    let scheduler = Arc::new(ContinuousBatchScheduler::new(
        ferrum_types::SchedulerConfig::default(),
    ));
    let tokenizer = Arc::new(PolicyTokenizer::new(64, &[]));
    let sampler = Arc::new(crate::registry::GreedySampler);
    let kv_cache = Arc::new(MockKvCacheManager::new(128));
    let model_info = ferrum_types::ModelInfo {
        model_id: ferrum_types::ModelId::new("recurrent-llm"),
        model_type: ferrum_types::ModelType::Custom("recurrent-llm".to_string()),
        num_parameters: 0,
        hidden_size: 4,
        num_layers: 1,
        num_heads: 1,
        num_kv_heads: 1,
        vocab_size: 64,
        max_sequence_length: 16,
        dtype: DataType::FP32,
        device: Device::CUDA(0),
        version: None,
        license: None,
        metadata: HashMap::new(),
    };
    let executor: Arc<dyn ModelExecutor + Send + Sync> = Arc::new(LlmExecutor::new(
        Box::new(RecurrentSpecLlm::new()),
        model_info,
    ));
    let tensor_factory = Arc::new(MockTensorFactory);
    let recurrent_manager = Arc::new(InMemoryRecurrentStateManager::new(
        InMemoryRecurrentStateConfig {
            total_memory_bytes: 1024,
            total_batch_slots: 1,
        },
    ));
    let engine = ContinuousBatchEngine::new_with_speculation_and_recurrent_state_manager(
        EngineConfig::default(),
        scheduler,
        tokenizer,
        sampler,
        kv_cache,
        executor,
        tensor_factory,
        None,
        None,
        Some(recurrent_manager.clone()),
    );
    let mut request = policy_request();
    request.sampling_params.max_tokens = 1;

    let response = engine.infer(request).await.unwrap();

    assert_eq!(response.finish_reason, FinishReason::Length);
    let stats = recurrent_manager.stats();
    assert_eq!(stats.total_batch_slots, 1);
    assert_eq!(stats.allocation_count, 1);
    assert_eq!(stats.allocation_failures, 0);
    assert_eq!(stats.active_states, 0);
    assert_eq!(stats.used_batch_slots, 0);
}

#[tokio::test]
async fn process_batch_unified_defers_prefill_for_recurrent_state_capacity() {
    let mut config = EngineConfig::default();
    config.kv_cache.max_blocks = 128;
    let scheduler = Arc::new(ContinuousBatchScheduler::new(config.scheduler.clone()));
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> =
        Arc::new(PolicyTokenizer::new(64, &[("test", 5), ("ok", 6)]));
    let sampler: Arc<dyn Sampler + Send + Sync> = Arc::new(crate::registry::GreedySampler);
    let kv_cache: Arc<dyn KvCacheManager + Send + Sync> = Arc::new(MockKvCacheManager::new(128));
    let executor: Arc<dyn ModelExecutor + Send + Sync> = Arc::new(RecurrentSpecExecutor {
        inner: MockModelExecutor::new(64, Duration::ZERO, Duration::ZERO),
    });
    let tensor_factory: Arc<dyn TensorFactory> = Arc::new(MockTensorFactory);
    let recurrent_manager = Arc::new(InMemoryRecurrentStateManager::new(
        InMemoryRecurrentStateConfig {
            total_memory_bytes: 8,
            total_batch_slots: 1,
        },
    ));
    let engine = ContinuousBatchEngine::new_with_speculation_and_recurrent_state_manager(
        config,
        scheduler,
        tokenizer.clone(),
        sampler,
        kv_cache,
        executor,
        tensor_factory,
        None,
        None,
        Some(recurrent_manager.clone()),
    );

    let mut victim_request = policy_request();
    victim_request.prompt = "test".to_string();
    victim_request.sampling_params.max_tokens = 4;
    let victim_id = victim_request.id.clone();
    let victim_spec = RecurrentStateSpec {
        request_id: victim_id.clone(),
        num_layers: 1,
        tensors: vec![RecurrentStateTensorSpec::new(0, "delta_state", vec![4])],
        dtype: DataType::BF16,
        device: Device::CPU,
        max_batch_slots: 1,
    };
    let victim_recurrent_state = recurrent_manager.allocate(&victim_spec).await.unwrap();
    let victim_kv = engine
        .inner
        .make_model_kv_handle_with_seq("victim-cache".to_string(), 1);
    let mut victim_seq = SequenceState::new_with_tokenizer_and_model_vocab_size(
        victim_request.clone(),
        vec![TokenId::new(5)],
        Some(tokenizer.clone()),
        Some(64),
    );
    victim_seq.generated_tokens.push(TokenId::new(6));
    victim_seq.prefill_complete = true;
    victim_seq.prefill_tokens_processed = 1;
    victim_seq.install_model_kv_without_owned_blocks(victim_kv);
    victim_seq.commit_recurrent_state_admission(victim_recurrent_state, 1);
    victim_seq.phase = RequestPhase::Decoding;
    {
        let mut sequences = engine.inner.sequences.write();
        sequences.insert(victim_id.clone(), victim_seq);
    }

    let mut fresh_request = policy_request();
    fresh_request.prompt = "test".to_string();
    fresh_request.sampling_params.max_tokens = 2;
    let fresh_id = fresh_request.id.clone();
    let batch = ferrum_interfaces::BatchPlan {
        batch_id: ferrum_types::BatchId::new(),
        requests: vec![ferrum_interfaces::scheduler::ScheduledRequest::new(
            fresh_request,
        )],
        max_sequence_length: 1,
        estimated_time_ms: None,
        resource_requirements: ferrum_interfaces::scheduler::BatchResourceRequirements::default(),
        created_at: chrono::Utc::now(),
    };

    engine.inner.process_batch(&batch).await.unwrap();

    let stats = recurrent_manager.stats();
    assert_eq!(stats.allocation_count, 1);
    assert_eq!(stats.allocation_failures, 0);
    assert_eq!(stats.active_states, 1);
    assert_eq!(stats.used_batch_slots, 1);

    let sequences = engine.inner.sequences.read();
    let victim = sequences
        .get(&victim_id)
        .expect("decode request should stay active while prefill waits");
    assert!(victim.prefill_complete);
    assert!(victim.kv_cache_handle().is_some());
    assert!(victim.recurrent_state.is_some());
    assert_eq!(victim.generated_tokens, vec![TokenId::new(6)]);
    assert_eq!(victim.preemption_count, 0);

    let fresh = sequences
        .get(&fresh_id)
        .expect("fresh request should remain queued for retry");
    assert!(!fresh.prefill_complete);
    assert!(fresh.kv_cache_handle().is_none());
    assert!(fresh.recurrent_state.is_none());
}

#[tokio::test]
async fn process_batch_unified_releases_recurrent_state_when_kv_alloc_defers() {
    let mut config = EngineConfig::default();
    config.kv_cache.max_blocks = 0;
    let scheduler = Arc::new(ContinuousBatchScheduler::new(config.scheduler.clone()));
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> =
        Arc::new(PolicyTokenizer::new(64, &[("test", 5), ("ok", 6)]));
    let sampler: Arc<dyn Sampler + Send + Sync> = Arc::new(crate::registry::GreedySampler);
    let kv_cache = Arc::new(MockKvCacheManager::new(0));
    let executor: Arc<dyn ModelExecutor + Send + Sync> = Arc::new(RecurrentSpecExecutor {
        inner: MockModelExecutor::new(64, Duration::ZERO, Duration::ZERO),
    });
    let tensor_factory: Arc<dyn TensorFactory> = Arc::new(MockTensorFactory);
    let recurrent_manager = Arc::new(InMemoryRecurrentStateManager::new(
        InMemoryRecurrentStateConfig {
            total_memory_bytes: 8,
            total_batch_slots: 1,
        },
    ));
    let engine = ContinuousBatchEngine::new_with_speculation_and_recurrent_state_manager(
        config,
        scheduler,
        tokenizer,
        sampler,
        kv_cache.clone(),
        executor,
        tensor_factory,
        None,
        None,
        Some(recurrent_manager.clone()),
    );

    let mut request = policy_request();
    request.prompt = "test".to_string();
    request.sampling_params.max_tokens = 2;
    let request_id = request.id.clone();
    let batch = ferrum_interfaces::BatchPlan {
        batch_id: ferrum_types::BatchId::new(),
        requests: vec![ferrum_interfaces::scheduler::ScheduledRequest::new(request)],
        max_sequence_length: 1,
        estimated_time_ms: None,
        resource_requirements: ferrum_interfaces::scheduler::BatchResourceRequirements::default(),
        created_at: chrono::Utc::now(),
    };

    engine.inner.process_batch(&batch).await.unwrap();

    let recurrent_stats = recurrent_manager.stats();
    assert_eq!(recurrent_stats.allocation_count, 1);
    assert_eq!(recurrent_stats.active_states, 0);
    assert_eq!(recurrent_stats.used_batch_slots, 0);
    let active_kv = kv_cache.list_handles();
    assert_eq!(active_kv.len(), 0, "active kv handles: {active_kv:?}");

    let sequences = engine.inner.sequences.read();
    let sequence = sequences
        .get(&request_id)
        .expect("deferred request should remain queued");
    assert!(!sequence.prefill_complete);
    assert!(sequence.kv_cache_handle().is_none());
    assert!(sequence.recurrent_state.is_none());
}

#[tokio::test]
async fn process_batch_unified_kv_defer_does_not_preempt_decode_for_fresh_prefill() {
    let mut config = EngineConfig::default();
    config.kv_cache.max_blocks = 1;
    let scheduler = Arc::new(ContinuousBatchScheduler::new(config.scheduler.clone()));
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> =
        Arc::new(PolicyTokenizer::new(64, &[("test", 5), ("ok", 6)]));
    let sampler: Arc<dyn Sampler + Send + Sync> = Arc::new(crate::registry::GreedySampler);
    let kv_cache = Arc::new(MockKvCacheManager::new(1));
    let executor: Arc<dyn ModelExecutor + Send + Sync> = Arc::new(CapturingUnifiedExecutor {
        inner: MockModelExecutor::new(64, Duration::ZERO, Duration::ZERO),
        captured: Arc::new(std::sync::Mutex::new(Vec::new())),
        output_token: 6,
    });
    let tensor_factory: Arc<dyn TensorFactory> = Arc::new(MockTensorFactory);
    let engine = ContinuousBatchEngine::new_with_speculation_and_recurrent_state_manager(
        config,
        scheduler.clone(),
        tokenizer.clone(),
        sampler,
        kv_cache.clone(),
        executor,
        tensor_factory,
        None,
        None,
        None,
    );

    let mut decode_request = policy_request();
    decode_request.prompt = "test".to_string();
    decode_request.sampling_params.max_tokens = 4;
    let decode_id = decode_request.id.clone();
    scheduler.submit(decode_request.clone()).await.unwrap();
    let initial_batch = scheduler
        .next_batch(ferrum_interfaces::BatchHint::simple(4))
        .await
        .expect("decode request should first be scheduled as prefill");
    assert_eq!(initial_batch.requests.len(), 1);
    scheduler.mark_prefill_complete(&decode_id, 1);

    let decode_kv = kv_cache
        .allocate(&AllocationRequest {
            request_id: decode_id.clone(),
            initial_tokens: 1,
            max_sequence_length: 16,
            num_layers: 1,
            num_heads: 1,
            head_dim: 4,
            device: Device::CPU,
            dtype: DataType::FP32,
            priority: Priority::Normal,
        })
        .await
        .unwrap();
    let mut decode_seq = SequenceState::new_with_tokenizer_and_model_vocab_size(
        decode_request.clone(),
        vec![TokenId::new(5)],
        Some(tokenizer),
        Some(64),
    );
    decode_seq.generated_tokens.push(TokenId::new(6));
    decode_seq.prefill_complete = true;
    decode_seq.prefill_tokens_processed = 1;
    decode_seq.install_model_kv_without_owned_blocks(decode_kv);
    decode_seq.phase = RequestPhase::Decoding;
    engine
        .inner
        .sequences
        .write()
        .insert(decode_id.clone(), decode_seq);

    let mut fresh_request = policy_request();
    fresh_request.prompt = "test".to_string();
    fresh_request.sampling_params.max_tokens = 2;
    let fresh_id = fresh_request.id.clone();
    scheduler.submit(fresh_request).await.unwrap();
    let batch = scheduler
        .next_batch(ferrum_interfaces::BatchHint::simple(4))
        .await
        .expect("mixed decode/fresh prefill batch should be scheduled");
    assert_eq!(batch.requests.len(), 2);

    engine.inner.process_batch(&batch).await.unwrap();

    let trace = scheduler.trace_snapshot();
    assert_eq!(
        trace.cancelled_total, 0,
        "fresh waiting prefill KV pressure must not cancel an active decode victim"
    );
    assert_eq!(
        scheduler.trace_phase(&fresh_id),
        Some(RequestPhase::Waiting),
        "fresh prefill should be retried later from waiting"
    );

    let sequences = engine.inner.sequences.read();
    let decode = sequences
        .get(&decode_id)
        .expect("decode victim should remain active");
    assert!(decode.prefill_complete);
    assert!(decode.kv_cache_handle().is_some());
    assert_eq!(decode.preemption_count, 0);

    let fresh = sequences
        .get(&fresh_id)
        .expect("deferred fresh request should remain in sequence state");
    assert!(!fresh.prefill_complete);
    assert!(fresh.kv_cache_handle().is_none());
    assert_eq!(kv_cache.active_count(), 1);
}

#[tokio::test]
async fn process_batch_unified_reserve_defer_requeues_decode_for_recompute() {
    let mut config = EngineConfig::default();
    config.kv_cache.max_blocks = 128;
    let scheduler = Arc::new(ContinuousBatchScheduler::new(config.scheduler.clone()));
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> =
        Arc::new(PolicyTokenizer::new(64, &[("test", 5), ("ok", 6)]));
    let sampler: Arc<dyn Sampler + Send + Sync> = Arc::new(crate::registry::GreedySampler);
    let kv_cache = Arc::new(MockKvCacheManager::new(128));
    let executor: Arc<dyn ModelExecutor + Send + Sync> = Arc::new(FailingUnifiedReserveExecutor {
        inner: FailingBatchPrefillExecutor {
            inner: RecurrentSpecExecutor {
                inner: MockModelExecutor::new(64, Duration::ZERO, Duration::ZERO),
            },
        },
    });
    let tensor_factory: Arc<dyn TensorFactory> = Arc::new(MockTensorFactory);
    let recurrent_manager = Arc::new(InMemoryRecurrentStateManager::new(
        InMemoryRecurrentStateConfig {
            total_memory_bytes: 32,
            total_batch_slots: 4,
        },
    ));
    let engine = ContinuousBatchEngine::new_with_speculation_and_recurrent_state_manager(
        config,
        scheduler.clone(),
        tokenizer.clone(),
        sampler,
        kv_cache.clone(),
        executor,
        tensor_factory,
        None,
        None,
        Some(recurrent_manager),
    );

    let mut first_decode_request = policy_request();
    first_decode_request.prompt = "test".to_string();
    first_decode_request.sampling_params.max_tokens = 4;
    let first_decode_id = first_decode_request.id.clone();

    let mut second_decode_request = policy_request();
    second_decode_request.prompt = "ok".to_string();
    second_decode_request.sampling_params.max_tokens = 4;
    let second_decode_id = second_decode_request.id.clone();

    scheduler
        .submit(first_decode_request.clone())
        .await
        .unwrap();
    scheduler
        .submit(second_decode_request.clone())
        .await
        .unwrap();
    let initial_batch = scheduler
        .next_batch(ferrum_interfaces::BatchHint::simple(4))
        .await
        .expect("decode requests should first be scheduled as prefills");
    assert_eq!(initial_batch.requests.len(), 2);
    scheduler.mark_prefill_complete(&first_decode_id, 1);
    scheduler.mark_prefill_complete(&second_decode_id, 1);

    for (request, request_id, token, cache_id) in [
        (
            first_decode_request.clone(),
            first_decode_id.clone(),
            TokenId::new(5),
            "first-decode-cache",
        ),
        (
            second_decode_request.clone(),
            second_decode_id.clone(),
            TokenId::new(6),
            "second-decode-cache",
        ),
    ] {
        let kv = engine
            .inner
            .make_model_kv_handle_with_seq(cache_id.to_string(), 1);
        let mut sequence = SequenceState::new_with_tokenizer_and_model_vocab_size(
            request,
            vec![token],
            Some(tokenizer.clone()),
            Some(64),
        );
        sequence.generated_tokens.push(token);
        sequence.prefill_complete = true;
        sequence.prefill_tokens_processed = 1;
        sequence.install_model_kv_without_owned_blocks(kv);
        sequence.phase = RequestPhase::Decoding;
        engine.inner.sequences.write().insert(request_id, sequence);
    }

    let mut fresh_request = policy_request();
    fresh_request.prompt = "test ok".to_string();
    fresh_request.sampling_params.max_tokens = 2;
    let fresh_id = fresh_request.id.clone();
    scheduler.submit(fresh_request).await.unwrap();
    let mixed_batch = scheduler
        .next_batch(ferrum_interfaces::BatchHint::simple(4))
        .await
        .expect("mixed decode/fresh prefill batch should be scheduled");
    assert_eq!(mixed_batch.requests.len(), 3);

    engine.inner.process_batch(&mixed_batch).await.unwrap();

    let trace = scheduler.trace_snapshot();
    assert_eq!(
        trace.cancelled_total, 0,
        "model-owned KV reserve pressure must not cancel active decodes"
    );
    assert_eq!(trace.waiting_queue_len, 3);
    assert_eq!(trace.decode_queue_len, 0);
    assert_eq!(trace.active_len, 0);
    assert_eq!(
        scheduler.trace_phase(&fresh_id),
        Some(RequestPhase::Waiting)
    );

    let sequences = engine.inner.sequences.read();
    for (request_id, token) in [
        (&first_decode_id, TokenId::new(5)),
        (&second_decode_id, TokenId::new(6)),
    ] {
        let decode = sequences
            .get(request_id)
            .expect("decode request should remain available for recompute");
        assert_eq!(decode.phase, RequestPhase::Waiting);
        assert!(!decode.prefill_complete);
        assert!(decode.kv_cache_handle().is_none());
        assert!(decode.model_cache_id().is_none());
        assert_eq!(decode.generated_tokens, vec![token]);
        assert_eq!(decode.preemption_count, 1);
        assert_eq!(
            scheduler.trace_phase(request_id),
            Some(RequestPhase::Waiting)
        );
    }
    let fresh = sequences
        .get(&fresh_id)
        .expect("deferred fresh request should remain in sequence state");
    assert_eq!(fresh.phase, RequestPhase::Waiting);
    assert!(!fresh.prefill_complete);
    assert!(fresh.kv_cache_handle().is_none());
    assert_eq!(kv_cache.active_count(), 0);
}

#[tokio::test]
async fn process_batch_unified_structured_pressure_reopens_capacity_recompute_next_epoch() {
    let mut config = EngineConfig::default();
    config.kv_cache.max_blocks = 128;
    let scheduler = Arc::new(ContinuousBatchScheduler::new(config.scheduler.clone()));
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> =
        Arc::new(PolicyTokenizer::new(64, &[("test", 5), ("ok", 6)]));
    let sampler: Arc<dyn Sampler + Send + Sync> = Arc::new(crate::registry::GreedySampler);
    let kv_cache = Arc::new(MockKvCacheManager::new(128));
    let executor: Arc<dyn ModelExecutor + Send + Sync> =
        Arc::new(StructuredPressureOnceUnifiedReserveExecutor {
            inner: RecurrentSpecExecutor {
                inner: MockModelExecutor::new(64, Duration::ZERO, Duration::ZERO),
            },
            remaining_failures: std::sync::atomic::AtomicUsize::new(1),
            message:
                "Qwen3.5 paged KV admission: need 4 admission blocks (4 immediate) but only 5 free",
        });
    let tensor_factory: Arc<dyn TensorFactory> = Arc::new(MockTensorFactory);
    let recurrent_manager = Arc::new(InMemoryRecurrentStateManager::new(
        InMemoryRecurrentStateConfig {
            total_memory_bytes: 32,
            total_batch_slots: 4,
        },
    ));
    let engine = ContinuousBatchEngine::new_with_speculation_and_recurrent_state_manager(
        config,
        scheduler.clone(),
        tokenizer.clone(),
        sampler,
        kv_cache,
        executor,
        tensor_factory,
        None,
        None,
        Some(recurrent_manager),
    );

    let mut requests = Vec::new();
    for _ in 0..4 {
        let mut request = policy_request();
        request.prompt = "test".to_string();
        request.sampling_params.max_tokens = 4;
        scheduler.submit(request.clone()).await.unwrap();
        requests.push(request);
    }

    let initial_batch = scheduler
        .next_batch(ferrum_interfaces::BatchHint::simple(4))
        .await
        .expect("decode requests should first be scheduled as prefills");
    assert_eq!(initial_batch.requests.len(), 4);
    for request in &requests {
        scheduler.mark_prefill_complete(&request.id, 1);
        let kv = engine
            .inner
            .make_model_kv_handle_with_seq(format!("decode-cache-{}", request.id), 1);
        let mut sequence = SequenceState::new_with_tokenizer_and_model_vocab_size(
            request.clone(),
            vec![TokenId::new(5)],
            Some(tokenizer.clone()),
            Some(64),
        );
        sequence.generated_tokens.push(TokenId::new(6));
        sequence.prefill_complete = true;
        sequence.prefill_tokens_processed = 1;
        sequence.install_model_kv_without_owned_blocks(kv);
        sequence.phase = RequestPhase::Decoding;
        engine
            .inner
            .sequences
            .write()
            .insert(request.id.clone(), sequence);
    }

    let recompute_id = requests[0].id.clone();
    {
        let mut sequences = engine.inner.sequences.write();
        let sequence = sequences
            .get_mut(&recompute_id)
            .expect("recompute sequence should exist");
        sequence.clear_model_kv_for_test();
        sequence.prefill_complete = false;
        sequence.prefill_tokens_processed = 0;
        sequence.phase = RequestPhase::Waiting;
        sequence.tokens_this_iteration = 0;
        sequence.preemption_count += 1;
    }
    assert!(
        scheduler.defer_decode_to_waiting_for_capacity_with_pressure(&recompute_id, 4, Some(0))
    );

    let mixed_batch = scheduler
        .next_batch(ferrum_interfaces::BatchHint::simple(4))
        .await
        .expect("decode plus capacity-deferred recompute should be scheduled");
    assert_eq!(mixed_batch.requests.len(), 4);
    assert!(
        mixed_batch.requests.iter().any(|request| {
            request.request.id == recompute_id && request.tokens_to_process != Some(1)
        }),
        "the first mixed attempt must contain the capacity-deferred recompute"
    );

    engine.inner.process_batch(&mixed_batch).await.unwrap();

    let retry_batch = scheduler
        .next_batch(ferrum_interfaces::BatchHint::simple(4))
        .await
        .expect("structured KV feedback should reopen recompute in the next epoch");
    assert!(
        retry_batch.requests.iter().any(|request| {
            request.request.id == recompute_id && request.tokens_to_process != Some(1)
        }),
        "failed recompute must not be marked attempted in the same epoch that structured KV feedback reopens"
    );
}

#[tokio::test]
async fn process_batch_unified_capacity_defer_releases_existing_kv() {
    let mut config = EngineConfig::default();
    config.kv_cache.max_blocks = 128;
    let scheduler = Arc::new(ContinuousBatchScheduler::new(config.scheduler.clone()));
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> =
        Arc::new(PolicyTokenizer::new(64, &[("test", 5), ("ok", 6)]));
    let sampler: Arc<dyn Sampler + Send + Sync> = Arc::new(crate::registry::GreedySampler);
    let kv_cache = Arc::new(MockKvCacheManager::new(128));
    let executor: Arc<dyn ModelExecutor + Send + Sync> = Arc::new(RecurrentSpecExecutor {
        inner: MockModelExecutor::new(64, Duration::ZERO, Duration::ZERO),
    });
    let tensor_factory: Arc<dyn TensorFactory> = Arc::new(MockTensorFactory);
    let recurrent_manager = Arc::new(InMemoryRecurrentStateManager::new(
        InMemoryRecurrentStateConfig {
            total_memory_bytes: 0,
            total_batch_slots: 0,
        },
    ));
    let engine = ContinuousBatchEngine::new_with_speculation_and_recurrent_state_manager(
        config,
        scheduler.clone(),
        tokenizer.clone(),
        sampler,
        kv_cache.clone(),
        executor,
        tensor_factory,
        None,
        None,
        Some(recurrent_manager.clone()),
    );

    let mut request = policy_request();
    request.prompt = "test ok".to_string();
    request.sampling_params.max_tokens = 2;
    let request_id = request.id.clone();
    scheduler.submit(request.clone()).await.unwrap();
    let batch = scheduler
        .next_batch(ferrum_interfaces::BatchHint::simple(4))
        .await
        .expect("submitted request should be scheduled");

    let allocated_kv = kv_cache
        .allocate(&AllocationRequest {
            request_id: request_id.clone(),
            initial_tokens: 1,
            max_sequence_length: 16,
            num_layers: 1,
            num_heads: 1,
            head_dim: 4,
            device: Device::CPU,
            dtype: DataType::FP32,
            priority: Priority::Normal,
        })
        .await
        .unwrap();
    let mut sequence = SequenceState::new_with_tokenizer_and_model_vocab_size(
        request,
        vec![TokenId::new(5), TokenId::new(6)],
        Some(tokenizer),
        Some(64),
    );
    sequence.install_model_kv_without_owned_blocks(allocated_kv);
    sequence.prefill_tokens_processed = 1;
    sequence.phase = RequestPhase::Prefilling;
    engine
        .inner
        .sequences
        .write()
        .insert(request_id.clone(), sequence);

    engine.inner.process_batch(&batch).await.unwrap();

    let deferred = scheduler.trace_snapshot();
    assert_eq!(deferred.waiting_queue_len, 1);
    assert_eq!(deferred.prefill_queue_len, 0);
    assert_eq!(deferred.active_len, 0);
    let active_kv = kv_cache.list_handles();
    assert_eq!(
        active_kv.len(),
        0,
        "capacity-deferred prefill must not leak KV handles: {active_kv:?}"
    );
    {
        let sequences = engine.inner.sequences.read();
        let sequence = sequences
            .get(&request_id)
            .expect("deferred request should remain available for retry");
        assert_eq!(sequence.phase, RequestPhase::Waiting);
        assert!(sequence.kv_cache_handle().is_none());
        assert!(sequence.recurrent_state.is_none());
    }
}

#[tokio::test]
async fn process_batch_unified_kv_defer_moves_active_prefill_back_to_waiting() {
    let mut config = EngineConfig::default();
    config.kv_cache.max_blocks = 0;
    let scheduler = Arc::new(ContinuousBatchScheduler::new(config.scheduler.clone()));
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> =
        Arc::new(PolicyTokenizer::new(64, &[("test", 5), ("ok", 6)]));
    let sampler: Arc<dyn Sampler + Send + Sync> = Arc::new(crate::registry::GreedySampler);
    let kv_cache = Arc::new(MockKvCacheManager::new(0));
    let executor: Arc<dyn ModelExecutor + Send + Sync> = Arc::new(RecurrentSpecExecutor {
        inner: MockModelExecutor::new(64, Duration::ZERO, Duration::ZERO),
    });
    let tensor_factory: Arc<dyn TensorFactory> = Arc::new(MockTensorFactory);
    let recurrent_manager = Arc::new(InMemoryRecurrentStateManager::new(
        InMemoryRecurrentStateConfig {
            total_memory_bytes: 8,
            total_batch_slots: 1,
        },
    ));
    let engine = ContinuousBatchEngine::new_with_speculation_and_recurrent_state_manager(
        config,
        scheduler.clone(),
        tokenizer,
        sampler,
        kv_cache,
        executor,
        tensor_factory,
        None,
        None,
        Some(recurrent_manager),
    );

    let mut request = policy_request();
    request.prompt = "test".to_string();
    request.sampling_params.max_tokens = 2;
    let request_id = request.id.clone();
    scheduler.submit(request).await.unwrap();
    let batch = scheduler
        .next_batch(ferrum_interfaces::BatchHint::simple(4))
        .await
        .expect("submitted request should be scheduled");
    let active = scheduler.trace_snapshot();
    assert_eq!(active.prefill_queue_len, 1);
    assert_eq!(active.active_len, 1);

    engine.inner.process_batch(&batch).await.unwrap();

    let deferred = scheduler.trace_snapshot();
    assert_eq!(deferred.waiting_queue_len, 1);
    assert_eq!(deferred.prefill_queue_len, 0);
    assert_eq!(deferred.active_len, 0);
    assert_eq!(deferred.cancelled_total, 0);
    assert_eq!(
        scheduler.trace_phase(&request_id),
        Some(RequestPhase::Waiting)
    );
    let sequences = engine.inner.sequences.read();
    let sequence = sequences
        .get(&request_id)
        .expect("deferred request should remain in sequence state");
    assert_eq!(sequence.phase, RequestPhase::Waiting);
    assert!(!sequence.prefill_complete);
    assert!(sequence.kv_cache_handle().is_none());
    assert!(sequence.recurrent_state.is_none());
}

#[tokio::test]
async fn process_batch_releases_kv_and_recurrent_state_when_model_admission_fails() {
    let trace_path = resource_trace_temp_path("model-admission-failure");
    let _ = std::fs::remove_file(&trace_path);
    let mut config = EngineConfig::default();
    config.kv_cache.max_blocks = 128;
    config.runtime.scheduler_trace_jsonl = Some(trace_path.clone());
    config.runtime.profile_entrypoint = Some(ProfileEntrypoint::Synthetic);
    let scheduler = Arc::new(ContinuousBatchScheduler::new(config.scheduler.clone()));
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> =
        Arc::new(PolicyTokenizer::new(64, &[("test", 5), ("ok", 6)]));
    let sampler: Arc<dyn Sampler + Send + Sync> = Arc::new(crate::registry::GreedySampler);
    let kv_cache = Arc::new(MockKvCacheManager::new(128));
    let executor: Arc<dyn ModelExecutor + Send + Sync> = Arc::new(FailingBatchPrefillExecutor {
        inner: RecurrentSpecExecutor {
            inner: MockModelExecutor::new(64, Duration::ZERO, Duration::ZERO),
        },
    });
    let tensor_factory: Arc<dyn TensorFactory> = Arc::new(MockTensorFactory);
    let recurrent_manager = Arc::new(InMemoryRecurrentStateManager::new(
        InMemoryRecurrentStateConfig {
            total_memory_bytes: 8,
            total_batch_slots: 1,
        },
    ));
    let engine = ContinuousBatchEngine::new_with_speculation_and_recurrent_state_manager(
        config,
        scheduler,
        tokenizer.clone(),
        sampler,
        kv_cache.clone(),
        executor,
        tensor_factory,
        None,
        Some(crate::speculative::SpeculativeDecodingConfig::default()),
        Some(recurrent_manager.clone()),
    );

    let mut request = policy_request();
    request.prompt = "test".to_string();
    request.sampling_params.max_tokens = 2;
    let request_id = request.id.clone();
    let batch = ferrum_interfaces::BatchPlan {
        batch_id: ferrum_types::BatchId::new(),
        requests: vec![ferrum_interfaces::scheduler::ScheduledRequest::new(request)],
        max_sequence_length: 1,
        estimated_time_ms: None,
        resource_requirements: ferrum_interfaces::scheduler::BatchResourceRequirements::default(),
        created_at: chrono::Utc::now(),
    };

    engine.inner.process_batch(&batch).await.unwrap();

    let active_kv = kv_cache.list_handles();
    assert_eq!(active_kv.len(), 0, "active kv handles: {active_kv:?}");
    let recurrent_stats = recurrent_manager.stats();
    assert!(recurrent_stats.allocation_count >= 2);
    assert_eq!(recurrent_stats.active_states, 0);
    assert_eq!(recurrent_stats.used_batch_slots, 0);
    let resources = assert_engine_resource_trace_balanced(&trace_path);
    let saw = |kind: &str, action: ResourceAction| {
        resources
            .iter()
            .any(|resource| resource.resource_kind == kind && resource.action == action)
    };
    assert!(saw("kv_block", ResourceAction::Reserve));
    assert!(saw("kv_block", ResourceAction::Commit));
    assert!(saw("kv_block", ResourceAction::Release));
    assert!(saw("recurrent_state_slot", ResourceAction::Reserve));
    assert!(saw("recurrent_state_slot", ResourceAction::Commit));
    assert!(saw("recurrent_state_slot", ResourceAction::Release));
    assert!(saw("backend_workspace", ResourceAction::Reserve));
    assert!(saw("backend_workspace", ResourceAction::Commit));
    assert!(saw("backend_workspace", ResourceAction::Release));
    let _ = std::fs::remove_file(trace_path);

    let sequences = engine.inner.sequences.read();
    let sequence = sequences
        .get(&request_id)
        .expect("failed request should remain available for retry");
    assert!(!sequence.prefill_complete);
    assert!(sequence.kv_cache_handle().is_none());
    assert!(sequence.recurrent_state.is_none());
}

#[tokio::test]
async fn process_batch_chunked_prefill_postprocess_error_releases_kv_and_recurrent_state() {
    let mut config = EngineConfig::default();
    config.kv_cache.max_blocks = 128;
    config.runtime.chunked_prefill_size = Some(1);
    let scheduler = Arc::new(ContinuousBatchScheduler::new(config.scheduler.clone()));
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> =
        Arc::new(PolicyTokenizer::new(64, &[("test", 5), ("ok", 6)]));
    let sampler: Arc<dyn Sampler + Send + Sync> = Arc::new(crate::registry::GreedySampler);
    let kv_cache = Arc::new(MockKvCacheManager::new(128));
    let executor: Arc<dyn ModelExecutor + Send + Sync> = Arc::new(BadShapePrefillExecutor {
        inner: RecurrentSpecExecutor {
            inner: MockModelExecutor::new(64, Duration::ZERO, Duration::ZERO),
        },
    });
    let tensor_factory: Arc<dyn TensorFactory> = Arc::new(MockTensorFactory);
    let recurrent_manager = Arc::new(InMemoryRecurrentStateManager::new(
        InMemoryRecurrentStateConfig {
            total_memory_bytes: 8,
            total_batch_slots: 1,
        },
    ));
    let engine = ContinuousBatchEngine::new_with_speculation_and_recurrent_state_manager(
        config,
        scheduler,
        tokenizer.clone(),
        sampler,
        kv_cache.clone(),
        executor,
        tensor_factory,
        None,
        Some(crate::speculative::SpeculativeDecodingConfig::default()),
        Some(recurrent_manager.clone()),
    );

    let mut request = policy_request();
    request.prompt = "test ok".to_string();
    request.sampling_params.max_tokens = 2;
    let request_id = request.id.clone();
    let batch = ferrum_interfaces::BatchPlan {
        batch_id: ferrum_types::BatchId::new(),
        requests: vec![ferrum_interfaces::scheduler::ScheduledRequest::new(request)],
        max_sequence_length: 2,
        estimated_time_ms: None,
        resource_requirements: ferrum_interfaces::scheduler::BatchResourceRequirements::default(),
        created_at: chrono::Utc::now(),
    };

    let _ = engine.inner.process_batch(&batch).await;

    let active_kv = kv_cache.list_handles();
    assert_eq!(active_kv.len(), 0, "active kv handles: {active_kv:?}");
    let recurrent_stats = recurrent_manager.stats();
    assert_eq!(recurrent_stats.allocation_count, 1);
    assert_eq!(recurrent_stats.active_states, 0);
    assert_eq!(recurrent_stats.used_batch_slots, 0);

    assert!(
        !engine.inner.sequences.read().contains_key(&request_id),
        "error-completed request should be removed from active sequences"
    );
}

#[tokio::test]
async fn process_batch_batch_prefill_len_mismatch_releases_kv_and_recurrent_state() {
    let mut config = EngineConfig::default();
    config.kv_cache.max_blocks = 128;
    let scheduler = Arc::new(ContinuousBatchScheduler::new(config.scheduler.clone()));
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> =
        Arc::new(PolicyTokenizer::new(64, &[("test", 5), ("ok", 6)]));
    let sampler: Arc<dyn Sampler + Send + Sync> = Arc::new(crate::registry::GreedySampler);
    let kv_cache = Arc::new(FirstAllocateThenFailKvCacheManager::new(128));
    let executor: Arc<dyn ModelExecutor + Send + Sync> = Arc::new(ShortBatchPrefillExecutor {
        inner: RecurrentSpecExecutor {
            inner: MockModelExecutor::new(64, Duration::ZERO, Duration::ZERO),
        },
    });
    let tensor_factory: Arc<dyn TensorFactory> = Arc::new(MockTensorFactory);
    let recurrent_manager = Arc::new(InMemoryRecurrentStateManager::new(
        InMemoryRecurrentStateConfig {
            total_memory_bytes: 8,
            total_batch_slots: 1,
        },
    ));
    let engine = ContinuousBatchEngine::new_with_speculation_and_recurrent_state_manager(
        config,
        scheduler,
        tokenizer.clone(),
        sampler,
        kv_cache.clone(),
        executor,
        tensor_factory,
        None,
        Some(crate::speculative::SpeculativeDecodingConfig::default()),
        Some(recurrent_manager.clone()),
    );

    let mut request = policy_request();
    request.prompt = "test".to_string();
    request.sampling_params.max_tokens = 2;
    let request_id = request.id.clone();
    let batch = ferrum_interfaces::BatchPlan {
        batch_id: ferrum_types::BatchId::new(),
        requests: vec![ferrum_interfaces::scheduler::ScheduledRequest::new(request)],
        max_sequence_length: 1,
        estimated_time_ms: None,
        resource_requirements: ferrum_interfaces::scheduler::BatchResourceRequirements::default(),
        created_at: chrono::Utc::now(),
    };

    engine.inner.process_batch(&batch).await.unwrap();

    let active_kv = kv_cache.list_handles();
    assert_eq!(active_kv.len(), 0, "active kv handles: {active_kv:?}");
    let recurrent_stats = recurrent_manager.stats();
    assert_eq!(recurrent_stats.allocation_count, 2);
    assert_eq!(recurrent_stats.active_states, 0);
    assert_eq!(recurrent_stats.used_batch_slots, 0);

    let sequences = engine.inner.sequences.read();
    let sequence = sequences
        .get(&request_id)
        .expect("deferred fallback request should remain available for retry");
    assert!(!sequence.prefill_complete);
    assert!(sequence.kv_cache_handle().is_none());
    assert!(sequence.recurrent_state.is_none());
}

#[tokio::test]
async fn process_batch_batch_prefill_postprocess_error_releases_kv_and_recurrent_state() {
    let mut config = EngineConfig::default();
    config.kv_cache.max_blocks = 128;
    let scheduler = Arc::new(ContinuousBatchScheduler::new(config.scheduler.clone()));
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> =
        Arc::new(PolicyTokenizer::new(64, &[("test", 5), ("ok", 6)]));
    let sampler: Arc<dyn Sampler + Send + Sync> = Arc::new(crate::registry::GreedySampler);
    let kv_cache = Arc::new(MockKvCacheManager::new(128));
    let executor: Arc<dyn ModelExecutor + Send + Sync> = Arc::new(BadShapePrefillExecutor {
        inner: RecurrentSpecExecutor {
            inner: MockModelExecutor::new(64, Duration::ZERO, Duration::ZERO),
        },
    });
    let tensor_factory: Arc<dyn TensorFactory> = Arc::new(MockTensorFactory);
    let recurrent_manager = Arc::new(InMemoryRecurrentStateManager::new(
        InMemoryRecurrentStateConfig {
            total_memory_bytes: 8,
            total_batch_slots: 1,
        },
    ));
    let engine = ContinuousBatchEngine::new_with_speculation_and_recurrent_state_manager(
        config,
        scheduler,
        tokenizer.clone(),
        sampler,
        kv_cache.clone(),
        executor,
        tensor_factory,
        None,
        Some(crate::speculative::SpeculativeDecodingConfig::default()),
        Some(recurrent_manager.clone()),
    );

    let mut request = policy_request();
    request.prompt = "test".to_string();
    request.sampling_params.max_tokens = 2;
    let request_id = request.id.clone();
    let batch = ferrum_interfaces::BatchPlan {
        batch_id: ferrum_types::BatchId::new(),
        requests: vec![ferrum_interfaces::scheduler::ScheduledRequest::new(request)],
        max_sequence_length: 1,
        estimated_time_ms: None,
        resource_requirements: ferrum_interfaces::scheduler::BatchResourceRequirements::default(),
        created_at: chrono::Utc::now(),
    };

    let _ = engine.inner.process_batch(&batch).await;

    let active_kv = kv_cache.list_handles();
    assert_eq!(active_kv.len(), 0, "active kv handles: {active_kv:?}");
    let recurrent_stats = recurrent_manager.stats();
    assert_eq!(recurrent_stats.allocation_count, 1);
    assert_eq!(recurrent_stats.active_states, 0);
    assert_eq!(recurrent_stats.used_batch_slots, 0);
    assert!(
        !engine.inner.sequences.read().contains_key(&request_id),
        "error-completed request should be removed from active sequences"
    );
}

#[tokio::test]
async fn process_batch_chunked_prefill_tensor_error_releases_kv_and_recurrent_state() {
    let mut config = EngineConfig::default();
    config.kv_cache.max_blocks = 128;
    config.runtime.chunked_prefill_size = Some(1);
    let scheduler = Arc::new(ContinuousBatchScheduler::new(config.scheduler.clone()));
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> =
        Arc::new(PolicyTokenizer::new(64, &[("test", 5), ("ok", 6)]));
    let sampler: Arc<dyn Sampler + Send + Sync> = Arc::new(crate::registry::GreedySampler);
    let kv_cache = Arc::new(MockKvCacheManager::new(128));
    let executor: Arc<dyn ModelExecutor + Send + Sync> = Arc::new(RecurrentSpecExecutor {
        inner: MockModelExecutor::new(64, Duration::ZERO, Duration::ZERO),
    });
    let tensor_factory: Arc<dyn TensorFactory> = Arc::new(FailingFromSliceTensorFactory);
    let recurrent_manager = Arc::new(InMemoryRecurrentStateManager::new(
        InMemoryRecurrentStateConfig {
            total_memory_bytes: 8,
            total_batch_slots: 1,
        },
    ));
    let engine = ContinuousBatchEngine::new_with_speculation_and_recurrent_state_manager(
        config,
        scheduler,
        tokenizer.clone(),
        sampler,
        kv_cache.clone(),
        executor,
        tensor_factory,
        None,
        Some(crate::speculative::SpeculativeDecodingConfig::default()),
        Some(recurrent_manager.clone()),
    );

    let mut request = policy_request();
    request.prompt = "test ok".to_string();
    request.sampling_params.max_tokens = 2;
    let request_id = request.id.clone();
    let batch = ferrum_interfaces::BatchPlan {
        batch_id: ferrum_types::BatchId::new(),
        requests: vec![ferrum_interfaces::scheduler::ScheduledRequest::new(request)],
        max_sequence_length: 2,
        estimated_time_ms: None,
        resource_requirements: ferrum_interfaces::scheduler::BatchResourceRequirements::default(),
        created_at: chrono::Utc::now(),
    };

    let _ = engine.inner.process_batch(&batch).await;

    let active_kv = kv_cache.list_handles();
    assert_eq!(active_kv.len(), 0, "active kv handles: {active_kv:?}");
    let recurrent_stats = recurrent_manager.stats();
    assert_eq!(recurrent_stats.allocation_count, 1);
    assert_eq!(recurrent_stats.active_states, 0);
    assert_eq!(recurrent_stats.used_batch_slots, 0);
    assert!(
        !engine.inner.sequences.read().contains_key(&request_id),
        "error-completed request should be removed from active sequences"
    );
}

#[tokio::test]
async fn process_batch_batch_prefill_tensor_error_releases_kv_and_recurrent_state() {
    let mut config = EngineConfig::default();
    config.kv_cache.max_blocks = 128;
    let scheduler = Arc::new(ContinuousBatchScheduler::new(config.scheduler.clone()));
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> =
        Arc::new(PolicyTokenizer::new(64, &[("test", 5), ("ok", 6)]));
    let sampler: Arc<dyn Sampler + Send + Sync> = Arc::new(crate::registry::GreedySampler);
    let kv_cache = Arc::new(MockKvCacheManager::new(128));
    let executor: Arc<dyn ModelExecutor + Send + Sync> = Arc::new(RecurrentSpecExecutor {
        inner: MockModelExecutor::new(64, Duration::ZERO, Duration::ZERO),
    });
    let tensor_factory: Arc<dyn TensorFactory> = Arc::new(FailingFromSliceTensorFactory);
    let recurrent_manager = Arc::new(InMemoryRecurrentStateManager::new(
        InMemoryRecurrentStateConfig {
            total_memory_bytes: 8,
            total_batch_slots: 1,
        },
    ));
    let engine = ContinuousBatchEngine::new_with_speculation_and_recurrent_state_manager(
        config,
        scheduler,
        tokenizer.clone(),
        sampler,
        kv_cache.clone(),
        executor,
        tensor_factory,
        None,
        Some(crate::speculative::SpeculativeDecodingConfig::default()),
        Some(recurrent_manager.clone()),
    );

    let mut request = policy_request();
    request.prompt = "test".to_string();
    request.sampling_params.max_tokens = 2;
    let request_id = request.id.clone();
    let batch = ferrum_interfaces::BatchPlan {
        batch_id: ferrum_types::BatchId::new(),
        requests: vec![ferrum_interfaces::scheduler::ScheduledRequest::new(request)],
        max_sequence_length: 1,
        estimated_time_ms: None,
        resource_requirements: ferrum_interfaces::scheduler::BatchResourceRequirements::default(),
        created_at: chrono::Utc::now(),
    };

    let _ = engine.inner.process_batch(&batch).await;

    let active_kv = kv_cache.list_handles();
    assert_eq!(active_kv.len(), 0, "active kv handles: {active_kv:?}");
    let recurrent_stats = recurrent_manager.stats();
    assert!(recurrent_stats.allocation_count >= 1);
    assert_eq!(recurrent_stats.active_states, 0);
    assert_eq!(recurrent_stats.used_batch_slots, 0);
    assert!(
        !engine.inner.sequences.read().contains_key(&request_id),
        "error-completed request should be removed from active sequences"
    );
}

#[tokio::test]
async fn process_batch_speculative_draft_tensor_error_releases_target_and_draft_kv() {
    let mut config = EngineConfig::default();
    config.kv_cache.max_blocks = 128;
    let scheduler = Arc::new(ContinuousBatchScheduler::new(config.scheduler.clone()));
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> =
        Arc::new(PolicyTokenizer::new(64, &[("test", 5), ("ok", 6)]));
    let sampler: Arc<dyn Sampler + Send + Sync> = Arc::new(crate::registry::GreedySampler);
    let kv_cache = Arc::new(MockKvCacheManager::new(128));
    let target_executor: Arc<dyn ModelExecutor + Send + Sync> =
        Arc::new(MockModelExecutor::new(64, Duration::ZERO, Duration::ZERO));
    let draft_executor: Arc<dyn ModelExecutor + Send + Sync> =
        Arc::new(MockModelExecutor::new(64, Duration::ZERO, Duration::ZERO));
    let tensor_factory: Arc<dyn TensorFactory> = Arc::new(FailingFromSliceTensorFactory);
    let engine = ContinuousBatchEngine::new_with_speculation_and_recurrent_state_manager(
        config,
        scheduler,
        tokenizer.clone(),
        sampler,
        kv_cache.clone(),
        target_executor,
        tensor_factory,
        Some(draft_executor),
        Some(crate::speculative::SpeculativeDecodingConfig::default()),
        None,
    );

    let mut request = policy_request();
    request.prompt = "test".to_string();
    request.sampling_params.max_tokens = 2;
    let request_id = request.id.clone();
    let target_alloc = AllocationRequest {
        request_id: request_id.clone(),
        initial_tokens: 1,
        max_sequence_length: 16,
        num_layers: 1,
        num_heads: 1,
        head_dim: 4,
        device: Device::CPU,
        dtype: DataType::FP32,
        priority: Priority::Normal,
    };
    let target_kv = kv_cache.allocate(&target_alloc).await.unwrap();
    let mut sequence = SequenceState::new_with_tokenizer_and_model_vocab_size(
        request.clone(),
        vec![TokenId::new(5)],
        Some(tokenizer),
        Some(64),
    );
    sequence.generated_tokens.push(TokenId::new(6));
    sequence.prefill_complete = true;
    sequence.prefill_tokens_processed = 1;
    sequence.install_model_kv_without_owned_blocks(target_kv);
    sequence.phase = RequestPhase::Decoding;
    engine
        .inner
        .sequences
        .write()
        .insert(request_id.clone(), sequence);

    let batch = ferrum_interfaces::BatchPlan {
        batch_id: ferrum_types::BatchId::new(),
        requests: vec![ferrum_interfaces::scheduler::ScheduledRequest::new(request)],
        max_sequence_length: 1,
        estimated_time_ms: None,
        resource_requirements: ferrum_interfaces::scheduler::BatchResourceRequirements::default(),
        created_at: chrono::Utc::now(),
    };

    let _ = engine.inner.process_batch(&batch).await;

    let stats = kv_cache.stats();
    assert_eq!(
        stats.allocation_count, 2,
        "target and draft KV allocations should both be attempted"
    );
    assert_eq!(
        stats.active_caches, 0,
        "target and draft KV resources should both be released"
    );
    let active_kv = kv_cache.list_handles();
    assert_eq!(active_kv.len(), 0, "active kv handles: {active_kv:?}");
    assert!(
        !engine.inner.sequences.read().contains_key(&request_id),
        "error-completed request should be removed from active sequences"
    );
}

#[tokio::test]
async fn process_batch_unified_reserve_resource_exhausted_defers_without_fallback() {
    let mut config = EngineConfig::default();
    config.kv_cache.max_blocks = 128;
    let scheduler = Arc::new(ContinuousBatchScheduler::new(config.scheduler.clone()));
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> =
        Arc::new(PolicyTokenizer::new(64, &[("test", 5), ("ok", 6)]));
    let sampler: Arc<dyn Sampler + Send + Sync> = Arc::new(crate::registry::GreedySampler);
    let kv_cache = Arc::new(MockKvCacheManager::new(128));
    let executor: Arc<dyn ModelExecutor + Send + Sync> = Arc::new(FailingUnifiedReserveExecutor {
        inner: FailingBatchPrefillExecutor {
            inner: RecurrentSpecExecutor {
                inner: MockModelExecutor::new(64, Duration::ZERO, Duration::ZERO),
            },
        },
    });
    let tensor_factory: Arc<dyn TensorFactory> = Arc::new(MockTensorFactory);
    let recurrent_manager = Arc::new(InMemoryRecurrentStateManager::new(
        InMemoryRecurrentStateConfig {
            total_memory_bytes: 8,
            total_batch_slots: 1,
        },
    ));
    let engine = ContinuousBatchEngine::new_with_speculation_and_recurrent_state_manager(
        config,
        scheduler,
        tokenizer.clone(),
        sampler,
        kv_cache.clone(),
        executor,
        tensor_factory,
        None,
        None,
        Some(recurrent_manager.clone()),
    );

    let mut request = policy_request();
    request.prompt = "test".to_string();
    request.sampling_params.max_tokens = 2;
    let request_id = request.id.clone();
    let batch = ferrum_interfaces::BatchPlan {
        batch_id: ferrum_types::BatchId::new(),
        requests: vec![ferrum_interfaces::scheduler::ScheduledRequest::new(request)],
        max_sequence_length: 1,
        estimated_time_ms: None,
        resource_requirements: ferrum_interfaces::scheduler::BatchResourceRequirements::default(),
        created_at: chrono::Utc::now(),
    };

    engine.inner.process_batch(&batch).await.unwrap();

    let active_kv = kv_cache.list_handles();
    assert_eq!(active_kv.len(), 0, "active kv handles: {active_kv:?}");
    let recurrent_stats = recurrent_manager.stats();
    assert_eq!(
        recurrent_stats.allocation_count, 1,
        "ResourceExhausted admission should wait instead of entering legacy fallback"
    );
    assert_eq!(recurrent_stats.active_states, 0);
    assert_eq!(recurrent_stats.used_batch_slots, 0);

    let sequences = engine.inner.sequences.read();
    let sequence = sequences
        .get(&request_id)
        .expect("failed request should remain available for retry");
    assert!(!sequence.prefill_complete);
    assert!(sequence.kv_cache_handle().is_none());
    assert!(sequence.recurrent_state.is_none());
}

#[tokio::test]
async fn process_batch_unified_reserve_resource_exhausted_defers_existing_kv_prefill() {
    let mut config = EngineConfig::default();
    config.kv_cache.max_blocks = 128;
    let scheduler = Arc::new(ContinuousBatchScheduler::new(config.scheduler.clone()));
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> =
        Arc::new(PolicyTokenizer::new(64, &[("test", 5), ("ok", 6)]));
    let sampler: Arc<dyn Sampler + Send + Sync> = Arc::new(crate::registry::GreedySampler);
    let kv_cache = Arc::new(MockKvCacheManager::new(128));
    let executor: Arc<dyn ModelExecutor + Send + Sync> = Arc::new(FailingUnifiedReserveExecutor {
        inner: FailingBatchPrefillExecutor {
            inner: RecurrentSpecExecutor {
                inner: MockModelExecutor::new(64, Duration::ZERO, Duration::ZERO),
            },
        },
    });
    let tensor_factory: Arc<dyn TensorFactory> = Arc::new(MockTensorFactory);
    let recurrent_manager = Arc::new(InMemoryRecurrentStateManager::new(
        InMemoryRecurrentStateConfig {
            total_memory_bytes: 8,
            total_batch_slots: 1,
        },
    ));
    let engine = ContinuousBatchEngine::new_with_speculation_and_recurrent_state_manager(
        config,
        scheduler.clone(),
        tokenizer.clone(),
        sampler,
        kv_cache.clone(),
        executor,
        tensor_factory,
        None,
        None,
        Some(recurrent_manager.clone()),
    );

    let mut request = policy_request();
    request.prompt = "test ok".to_string();
    request.sampling_params.max_tokens = 2;
    let request_id = request.id.clone();
    scheduler.submit(request.clone()).await.unwrap();
    let batch = scheduler
        .next_batch(ferrum_interfaces::BatchHint::simple(4))
        .await
        .expect("submitted request should be scheduled");
    let active = scheduler.trace_snapshot();
    assert_eq!(active.prefill_queue_len, 1);
    assert_eq!(active.active_len, 1);

    let allocated_kv = kv_cache
        .allocate(&AllocationRequest {
            request_id: request_id.clone(),
            initial_tokens: 1,
            max_sequence_length: 16,
            num_layers: 1,
            num_heads: 1,
            head_dim: 4,
            device: Device::CPU,
            dtype: DataType::FP32,
            priority: Priority::Normal,
        })
        .await
        .unwrap();
    let mut sequence = SequenceState::new_with_tokenizer_and_model_vocab_size(
        request,
        vec![TokenId::new(5), TokenId::new(6)],
        Some(tokenizer),
        Some(64),
    );
    sequence.install_model_kv_without_owned_blocks(allocated_kv);
    sequence.prefill_tokens_processed = 1;
    sequence.phase = RequestPhase::Prefilling;
    engine
        .inner
        .sequences
        .write()
        .insert(request_id.clone(), sequence);

    engine.inner.process_batch(&batch).await.unwrap();

    let deferred = scheduler.trace_snapshot();
    assert_eq!(deferred.waiting_queue_len, 1);
    assert_eq!(deferred.prefill_queue_len, 0);
    assert_eq!(deferred.active_len, 0);
    assert_eq!(deferred.cancelled_total, 0);
    assert_eq!(
        scheduler.trace_phase(&request_id),
        Some(RequestPhase::Waiting)
    );
    let active_kv = kv_cache.list_handles();
    assert_eq!(active_kv.len(), 0, "active kv handles: {active_kv:?}");
    let recurrent_stats = recurrent_manager.stats();
    assert_eq!(recurrent_stats.active_states, 0);
    assert_eq!(recurrent_stats.used_batch_slots, 0);

    let sequences = engine.inner.sequences.read();
    let sequence = sequences
        .get(&request_id)
        .expect("deferred request should remain available for retry");
    assert_eq!(sequence.phase, RequestPhase::Waiting);
    assert!(!sequence.prefill_complete);
    assert!(sequence.kv_cache_handle().is_none());
    assert!(sequence.recurrent_state.is_none());
    assert!(sequence.model_cache_id().is_none());
    assert_eq!(
        sequence.prefill_tokens_processed, 0,
        "retry must rebuild KV from the full logical context"
    );
}

#[tokio::test]
async fn process_batch_unified_forward_resource_exhausted_defers_existing_kv_prefill() {
    let trace_path = resource_trace_temp_path("unified-forward-resource-exhausted");
    let _ = std::fs::remove_file(&trace_path);
    let mut config = EngineConfig::default();
    config.kv_cache.max_blocks = 128;
    config.runtime.scheduler_trace_jsonl = Some(trace_path.clone());
    config.runtime.profile_entrypoint = Some(ProfileEntrypoint::Synthetic);
    let scheduler = Arc::new(ContinuousBatchScheduler::new(config.scheduler.clone()));
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> =
        Arc::new(PolicyTokenizer::new(64, &[("test", 5), ("ok", 6)]));
    let sampler: Arc<dyn Sampler + Send + Sync> = Arc::new(crate::registry::GreedySampler);
    let kv_cache = Arc::new(MockKvCacheManager::new(128));
    let executor: Arc<dyn ModelExecutor + Send + Sync> = Arc::new(FailingUnifiedForwardExecutor {
        inner: RecurrentSpecExecutor {
            inner: MockModelExecutor::new(64, Duration::ZERO, Duration::ZERO),
        },
        resource_exhausted: true,
    });
    let tensor_factory: Arc<dyn TensorFactory> = Arc::new(MockTensorFactory);
    let recurrent_manager = Arc::new(InMemoryRecurrentStateManager::new(
        InMemoryRecurrentStateConfig {
            total_memory_bytes: 8,
            total_batch_slots: 1,
        },
    ));
    let engine = ContinuousBatchEngine::new_with_speculation_and_recurrent_state_manager(
        config,
        scheduler.clone(),
        tokenizer.clone(),
        sampler,
        kv_cache.clone(),
        executor,
        tensor_factory,
        None,
        None,
        Some(recurrent_manager.clone()),
    );

    let mut request = policy_request();
    request.prompt = "test ok".to_string();
    request.sampling_params.max_tokens = 2;
    let request_id = request.id.clone();
    scheduler.submit(request.clone()).await.unwrap();
    let batch = scheduler
        .next_batch(ferrum_interfaces::BatchHint::simple(4))
        .await
        .expect("submitted request should be scheduled");
    let active = scheduler.trace_snapshot();
    assert_eq!(active.prefill_queue_len, 1);
    assert_eq!(active.active_len, 1);

    let allocated_kv = kv_cache
        .allocate(&AllocationRequest {
            request_id: request_id.clone(),
            initial_tokens: 1,
            max_sequence_length: 16,
            num_layers: 1,
            num_heads: 1,
            head_dim: 4,
            device: Device::CPU,
            dtype: DataType::FP32,
            priority: Priority::Normal,
        })
        .await
        .unwrap();
    let mut sequence = SequenceState::new_with_tokenizer_and_model_vocab_size(
        request,
        vec![TokenId::new(5), TokenId::new(6)],
        Some(tokenizer),
        Some(64),
    );
    sequence.install_model_kv_without_owned_blocks(allocated_kv);
    sequence.prefill_tokens_processed = 1;
    sequence.phase = RequestPhase::Prefilling;
    engine
        .inner
        .sequences
        .write()
        .insert(request_id.clone(), sequence);

    engine.inner.trace_model_cache_ref_acquire(&request_id);
    engine.inner.process_batch(&batch).await.unwrap();

    let deferred = scheduler.trace_snapshot();
    assert_eq!(deferred.waiting_queue_len, 1);
    assert_eq!(deferred.prefill_queue_len, 0);
    assert_eq!(deferred.active_len, 0);
    assert_eq!(deferred.cancelled_total, 0);
    assert_eq!(
        scheduler.trace_phase(&request_id),
        Some(RequestPhase::Waiting)
    );
    let active_kv = kv_cache.list_handles();
    assert_eq!(active_kv.len(), 0, "active kv handles: {active_kv:?}");
    let recurrent_stats = recurrent_manager.stats();
    assert_eq!(recurrent_stats.active_states, 0);
    assert_eq!(recurrent_stats.used_batch_slots, 0);
    let resources = assert_engine_resource_trace_balanced(&trace_path);
    let saw = |kind: &str, action: ResourceAction| {
        resources
            .iter()
            .any(|resource| resource.resource_kind == kind && resource.action == action)
    };
    assert!(saw("backend_workspace", ResourceAction::Reserve));
    assert!(saw("backend_workspace", ResourceAction::Commit));
    assert!(saw("backend_workspace", ResourceAction::Release));
    let profile_events = read_engine_profile_events(&trace_path);
    let defer_event = profile_events
        .iter()
        .find(|event| {
            event
                .resource
                .as_ref()
                .is_some_and(|resource| resource.action == ResourceAction::Defer)
        })
        .expect("capacity defer event should be traced");
    let scheduler_snapshot = defer_event
        .attributes
        .get("scheduler_snapshot")
        .and_then(serde_json::Value::as_object)
        .expect("defer event should include scheduler snapshot");
    assert_eq!(
        scheduler_snapshot
            .get("waiting_queue_len")
            .and_then(serde_json::Value::as_u64),
        Some(1)
    );
    assert_eq!(
        scheduler_snapshot
            .get("capacity_deferred_total")
            .and_then(serde_json::Value::as_u64),
        Some(1)
    );
    assert!(
        scheduler_snapshot.contains_key("capacity_release_epoch"),
        "scheduler snapshot must expose release epoch"
    );
    assert!(
        scheduler_snapshot.contains_key("capacity_mixed_recompute_epoch"),
        "scheduler snapshot must expose mixed recompute epoch"
    );
    let _ = std::fs::remove_file(trace_path);

    let sequences = engine.inner.sequences.read();
    let sequence = sequences
        .get(&request_id)
        .expect("deferred request should remain available for retry");
    assert_eq!(sequence.phase, RequestPhase::Waiting);
    assert!(!sequence.prefill_complete);
    assert!(sequence.kv_cache_handle().is_none());
    assert!(sequence.recurrent_state.is_none());
    assert!(sequence.model_cache_id().is_none());
    assert_eq!(
        sequence.prefill_tokens_processed, 0,
        "retry must rebuild KV from the full logical context"
    );
}

#[tokio::test]
async fn process_batch_unified_forward_failure_then_fallback_kv_defer_releases_recurrent_state() {
    let mut config = EngineConfig::default();
    config.kv_cache.max_blocks = 128;
    let scheduler = Arc::new(ContinuousBatchScheduler::new(config.scheduler.clone()));
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> =
        Arc::new(PolicyTokenizer::new(64, &[("test", 5), ("ok", 6)]));
    let sampler: Arc<dyn Sampler + Send + Sync> = Arc::new(crate::registry::GreedySampler);
    let kv_cache = Arc::new(FirstAllocateThenFailKvCacheManager::new(128));
    let executor: Arc<dyn ModelExecutor + Send + Sync> = Arc::new(FailingUnifiedForwardExecutor {
        inner: RecurrentSpecExecutor {
            inner: MockModelExecutor::new(64, Duration::ZERO, Duration::ZERO),
        },
        resource_exhausted: false,
    });
    let tensor_factory: Arc<dyn TensorFactory> = Arc::new(MockTensorFactory);
    let recurrent_manager = Arc::new(InMemoryRecurrentStateManager::new(
        InMemoryRecurrentStateConfig {
            total_memory_bytes: 8,
            total_batch_slots: 1,
        },
    ));
    let engine = ContinuousBatchEngine::new_with_speculation_and_recurrent_state_manager(
        config,
        scheduler,
        tokenizer.clone(),
        sampler,
        kv_cache.clone(),
        executor,
        tensor_factory,
        None,
        None,
        Some(recurrent_manager.clone()),
    );

    let mut request = policy_request();
    request.prompt = "test".to_string();
    request.sampling_params.max_tokens = 2;
    let request_id = request.id.clone();
    let batch = ferrum_interfaces::BatchPlan {
        batch_id: ferrum_types::BatchId::new(),
        requests: vec![ferrum_interfaces::scheduler::ScheduledRequest::new(request)],
        max_sequence_length: 1,
        estimated_time_ms: None,
        resource_requirements: ferrum_interfaces::scheduler::BatchResourceRequirements::default(),
        created_at: chrono::Utc::now(),
    };

    engine.inner.process_batch(&batch).await.unwrap();

    let active_kv = kv_cache.list_handles();
    assert_eq!(active_kv.len(), 0, "active kv handles: {active_kv:?}");
    let recurrent_stats = recurrent_manager.stats();
    assert_eq!(recurrent_stats.allocation_count, 1);
    assert_eq!(recurrent_stats.active_states, 0);
    assert_eq!(recurrent_stats.used_batch_slots, 0);

    let sequences = engine.inner.sequences.read();
    let sequence = sequences
        .get(&request_id)
        .expect("deferred request should remain available for retry");
    assert!(!sequence.prefill_complete);
    assert!(sequence.kv_cache_handle().is_none());
    assert!(sequence.recurrent_state.is_none());
}

#[tokio::test]
async fn process_batch_unified_result_len_mismatch_releases_recurrent_state() {
    let mut config = EngineConfig::default();
    config.kv_cache.max_blocks = 128;
    let scheduler = Arc::new(ContinuousBatchScheduler::new(config.scheduler.clone()));
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> =
        Arc::new(PolicyTokenizer::new(64, &[("test", 5), ("ok", 6)]));
    let sampler: Arc<dyn Sampler + Send + Sync> = Arc::new(crate::registry::GreedySampler);
    let kv_cache = Arc::new(MockKvCacheManager::new(128));
    let executor: Arc<dyn ModelExecutor + Send + Sync> = Arc::new(ShortUnifiedResultExecutor {
        inner: RecurrentSpecExecutor {
            inner: MockModelExecutor::new(64, Duration::ZERO, Duration::ZERO),
        },
    });
    let tensor_factory: Arc<dyn TensorFactory> = Arc::new(MockTensorFactory);
    let recurrent_manager = Arc::new(InMemoryRecurrentStateManager::new(
        InMemoryRecurrentStateConfig {
            total_memory_bytes: 8,
            total_batch_slots: 1,
        },
    ));
    let engine = ContinuousBatchEngine::new_with_speculation_and_recurrent_state_manager(
        config,
        scheduler,
        tokenizer.clone(),
        sampler,
        kv_cache.clone(),
        executor,
        tensor_factory,
        None,
        None,
        Some(recurrent_manager.clone()),
    );

    let mut request = policy_request();
    request.prompt = "test".to_string();
    request.sampling_params.max_tokens = 2;
    let request_id = request.id.clone();
    let batch = ferrum_interfaces::BatchPlan {
        batch_id: ferrum_types::BatchId::new(),
        requests: vec![ferrum_interfaces::scheduler::ScheduledRequest::new(request)],
        max_sequence_length: 1,
        estimated_time_ms: None,
        resource_requirements: ferrum_interfaces::scheduler::BatchResourceRequirements::default(),
        created_at: chrono::Utc::now(),
    };

    let err = engine.inner.process_batch(&batch).await.unwrap_err();
    assert!(
        err.to_string().contains("unified_decode returned"),
        "unexpected error: {err}"
    );

    let active_kv = kv_cache.list_handles();
    assert_eq!(active_kv.len(), 0, "active kv handles: {active_kv:?}");
    let recurrent_stats = recurrent_manager.stats();
    assert_eq!(recurrent_stats.allocation_count, 1);
    assert_eq!(recurrent_stats.active_states, 0);
    assert_eq!(recurrent_stats.used_batch_slots, 0);

    let sequences = engine.inner.sequences.read();
    let sequence = sequences
        .get(&request_id)
        .expect("failed request should remain inspectable");
    assert!(!sequence.prefill_complete);
    assert!(sequence.kv_cache_handle().is_none());
    assert!(sequence.recurrent_state.is_none());
}

#[tokio::test]
async fn process_batch_unified_missing_final_prefill_result_releases_fresh_kv() {
    let mut config = EngineConfig::default();
    config.kv_cache.max_blocks = 128;
    let scheduler = Arc::new(ContinuousBatchScheduler::new(config.scheduler.clone()));
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> =
        Arc::new(PolicyTokenizer::new(64, &[("test", 5), ("ok", 6)]));
    let sampler: Arc<dyn Sampler + Send + Sync> = Arc::new(crate::registry::GreedySampler);
    let kv_cache = Arc::new(MockKvCacheManager::new(128));
    let executor: Arc<dyn ModelExecutor + Send + Sync> =
        Arc::new(MissingFinalUnifiedResultExecutor {
            inner: RecurrentSpecExecutor {
                inner: MockModelExecutor::new(64, Duration::ZERO, Duration::ZERO),
            },
        });
    let tensor_factory: Arc<dyn TensorFactory> = Arc::new(MockTensorFactory);
    let recurrent_manager = Arc::new(InMemoryRecurrentStateManager::new(
        InMemoryRecurrentStateConfig {
            total_memory_bytes: 8,
            total_batch_slots: 1,
        },
    ));
    let engine = ContinuousBatchEngine::new_with_speculation_and_recurrent_state_manager(
        config,
        scheduler,
        tokenizer.clone(),
        sampler,
        kv_cache.clone(),
        executor,
        tensor_factory,
        None,
        None,
        Some(recurrent_manager.clone()),
    );

    let mut request = policy_request();
    request.prompt = "test".to_string();
    request.sampling_params.max_tokens = 2;
    let request_id = request.id.clone();
    let batch = ferrum_interfaces::BatchPlan {
        batch_id: ferrum_types::BatchId::new(),
        requests: vec![ferrum_interfaces::scheduler::ScheduledRequest::new(request)],
        max_sequence_length: 1,
        estimated_time_ms: None,
        resource_requirements: ferrum_interfaces::scheduler::BatchResourceRequirements::default(),
        created_at: chrono::Utc::now(),
    };

    let _ = engine.inner.process_batch(&batch).await;

    let active_kv = kv_cache.list_handles();
    assert_eq!(active_kv.len(), 0, "active kv handles: {active_kv:?}");
    let recurrent_stats = recurrent_manager.stats();
    assert_eq!(recurrent_stats.allocation_count, 1);
    assert_eq!(recurrent_stats.active_states, 0);
    assert_eq!(recurrent_stats.used_batch_slots, 0);
    assert!(
        !engine.inner.sequences.read().contains_key(&request_id),
        "error-completed request should be removed from active sequences"
    );
}

#[tokio::test]
async fn process_batch_unified_prefill_postprocess_error_releases_fresh_kv() {
    let mut config = EngineConfig::default();
    config.kv_cache.max_blocks = 128;
    let scheduler = Arc::new(ContinuousBatchScheduler::new(config.scheduler.clone()));
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> = Arc::new(PolicyTokenizer::new(
        64,
        &[("test", 5), ("ok", 6), ("<unk>", 2)],
    ));
    let sampler: Arc<dyn Sampler + Send + Sync> = Arc::new(crate::registry::GreedySampler);
    let kv_cache = Arc::new(MockKvCacheManager::new(128));
    let executor: Arc<dyn ModelExecutor + Send + Sync> = Arc::new(GreedySentinelUnifiedExecutor {
        inner: RecurrentSpecExecutor {
            inner: MockModelExecutor::new(64, Duration::ZERO, Duration::ZERO),
        },
        token: 6,
    });
    let tensor_factory: Arc<dyn TensorFactory> = Arc::new(MockTensorFactory);
    let recurrent_manager = Arc::new(InMemoryRecurrentStateManager::new(
        InMemoryRecurrentStateConfig {
            total_memory_bytes: 8,
            total_batch_slots: 1,
        },
    ));
    let engine = ContinuousBatchEngine::new_with_speculation_and_recurrent_state_manager(
        config,
        scheduler,
        tokenizer.clone(),
        sampler,
        kv_cache.clone(),
        executor,
        tensor_factory,
        None,
        None,
        Some(recurrent_manager.clone()),
    );

    let mut request = policy_request();
    request.prompt = "test".to_string();
    request.sampling_params.max_tokens = 2;
    request.sampling_params.response_format = ferrum_types::ResponseFormat::JsonObject;
    let request_id = request.id.clone();
    let batch = ferrum_interfaces::BatchPlan {
        batch_id: ferrum_types::BatchId::new(),
        requests: vec![ferrum_interfaces::scheduler::ScheduledRequest::new(request)],
        max_sequence_length: 1,
        estimated_time_ms: None,
        resource_requirements: ferrum_interfaces::scheduler::BatchResourceRequirements::default(),
        created_at: chrono::Utc::now(),
    };

    let _ = engine.inner.process_batch(&batch).await;

    let active_kv = kv_cache.list_handles();
    assert_eq!(active_kv.len(), 0, "active kv handles: {active_kv:?}");
    let recurrent_stats = recurrent_manager.stats();
    assert_eq!(recurrent_stats.allocation_count, 1);
    assert_eq!(recurrent_stats.active_states, 0);
    assert_eq!(recurrent_stats.used_batch_slots, 0);
    assert!(
        !engine.inner.sequences.read().contains_key(&request_id),
        "error-completed request should be removed from active sequences"
    );
}

#[tokio::test]
async fn process_batch_unified_decode_postprocess_error_releases_recurrent_state() {
    let mut config = EngineConfig::default();
    config.kv_cache.max_blocks = 128;
    let scheduler = Arc::new(ContinuousBatchScheduler::new(config.scheduler.clone()));
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> =
        Arc::new(PolicyTokenizer::new(64, &[("test", 5), ("ok", 6)]));
    let sampler: Arc<dyn Sampler + Send + Sync> = Arc::new(crate::registry::GreedySampler);
    let kv_cache: Arc<dyn KvCacheManager + Send + Sync> = Arc::new(MockKvCacheManager::new(128));
    let executor: Arc<dyn ModelExecutor + Send + Sync> = Arc::new(GreedySentinelUnifiedExecutor {
        inner: RecurrentSpecExecutor {
            inner: MockModelExecutor::new(64, Duration::ZERO, Duration::ZERO),
        },
        token: 6,
    });
    let tensor_factory: Arc<dyn TensorFactory> = Arc::new(MockTensorFactory);
    let recurrent_manager = Arc::new(InMemoryRecurrentStateManager::new(
        InMemoryRecurrentStateConfig {
            total_memory_bytes: 8,
            total_batch_slots: 1,
        },
    ));
    let engine = ContinuousBatchEngine::new_with_speculation_and_recurrent_state_manager(
        config,
        scheduler,
        tokenizer.clone(),
        sampler,
        kv_cache,
        executor,
        tensor_factory,
        None,
        None,
        Some(recurrent_manager.clone()),
    );

    let mut request = policy_request();
    request.prompt = "test".to_string();
    request.sampling_params.max_tokens = 4;
    request.sampling_params.response_format = ferrum_types::ResponseFormat::JsonObject;
    let request_id = request.id.clone();
    let recurrent_spec = RecurrentStateSpec {
        request_id: request_id.clone(),
        num_layers: 1,
        tensors: vec![RecurrentStateTensorSpec::new(0, "delta_state", vec![4])],
        dtype: DataType::BF16,
        device: Device::CPU,
        max_batch_slots: 1,
    };
    let recurrent_state = recurrent_manager.allocate(&recurrent_spec).await.unwrap();
    let kv = engine
        .inner
        .make_model_kv_handle_with_seq("decode-cache".to_string(), 1);
    let mut sequence = SequenceState::new_with_tokenizer_and_model_vocab_size(
        request.clone(),
        vec![TokenId::new(5)],
        Some(tokenizer),
        Some(64),
    );
    sequence.generated_tokens.push(TokenId::new(6));
    sequence.prefill_complete = true;
    sequence.prefill_tokens_processed = 1;
    sequence.install_model_kv_without_owned_blocks(kv);
    sequence.commit_recurrent_state_admission(recurrent_state, 1);
    sequence.phase = RequestPhase::Decoding;
    engine
        .inner
        .sequences
        .write()
        .insert(request_id.clone(), sequence);

    let batch = ferrum_interfaces::BatchPlan {
        batch_id: ferrum_types::BatchId::new(),
        requests: vec![ferrum_interfaces::scheduler::ScheduledRequest::new(request)],
        max_sequence_length: 1,
        estimated_time_ms: None,
        resource_requirements: ferrum_interfaces::scheduler::BatchResourceRequirements::default(),
        created_at: chrono::Utc::now(),
    };

    let _ = engine.inner.process_batch(&batch).await;

    assert!(
        !engine.inner.sequences.read().contains_key(&request_id),
        "error-completed decode request should be removed from active sequences"
    );
    let recurrent_stats = recurrent_manager.stats();
    assert_eq!(recurrent_stats.active_states, 0);
    assert_eq!(recurrent_stats.used_batch_slots, 0);
}

#[tokio::test]
async fn process_batch_single_decode_resource_exhausted_keeps_recurrent_state_waiting() {
    let mut config = EngineConfig::default();
    config.kv_cache.max_blocks = 128;
    let scheduler = Arc::new(ContinuousBatchScheduler::new(config.scheduler.clone()));
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> =
        Arc::new(PolicyTokenizer::new(64, &[("test", 5), ("ok", 6)]));
    let sampler: Arc<dyn Sampler + Send + Sync> = Arc::new(crate::registry::GreedySampler);
    let kv_cache: Arc<dyn KvCacheManager + Send + Sync> = Arc::new(MockKvCacheManager::new(128));
    let executor: Arc<dyn ModelExecutor + Send + Sync> = Arc::new(FailingDecodeExecutor {
        inner: RecurrentSpecExecutor {
            inner: MockModelExecutor::new(64, Duration::ZERO, Duration::ZERO),
        },
    });
    let tensor_factory: Arc<dyn TensorFactory> = Arc::new(MockTensorFactory);
    let recurrent_manager = Arc::new(InMemoryRecurrentStateManager::new(
        InMemoryRecurrentStateConfig {
            total_memory_bytes: 8,
            total_batch_slots: 1,
        },
    ));
    let engine = ContinuousBatchEngine::new_with_speculation_and_recurrent_state_manager(
        config,
        scheduler,
        tokenizer.clone(),
        sampler,
        kv_cache,
        executor,
        tensor_factory,
        None,
        Some(crate::speculative::SpeculativeDecodingConfig::default()),
        Some(recurrent_manager.clone()),
    );

    let mut request = policy_request();
    request.prompt = "test".to_string();
    request.sampling_params.max_tokens = 4;
    let request_id = request.id.clone();
    let recurrent_spec = RecurrentStateSpec {
        request_id: request_id.clone(),
        num_layers: 1,
        tensors: vec![RecurrentStateTensorSpec::new(0, "delta_state", vec![4])],
        dtype: DataType::BF16,
        device: Device::CPU,
        max_batch_slots: 1,
    };
    let recurrent_state = recurrent_manager.allocate(&recurrent_spec).await.unwrap();
    let kv = engine
        .inner
        .make_model_kv_handle_with_seq("decode-cache".to_string(), 1);
    let mut sequence = SequenceState::new_with_tokenizer_and_model_vocab_size(
        request.clone(),
        vec![TokenId::new(5)],
        Some(tokenizer),
        Some(64),
    );
    sequence.generated_tokens.push(TokenId::new(6));
    sequence.prefill_complete = true;
    sequence.prefill_tokens_processed = 1;
    sequence.install_model_kv_without_owned_blocks(kv);
    sequence.commit_recurrent_state_admission(recurrent_state, 1);
    sequence.phase = RequestPhase::Decoding;
    engine
        .inner
        .sequences
        .write()
        .insert(request_id.clone(), sequence);

    let batch = ferrum_interfaces::BatchPlan {
        batch_id: ferrum_types::BatchId::new(),
        requests: vec![ferrum_interfaces::scheduler::ScheduledRequest::new(request)],
        max_sequence_length: 1,
        estimated_time_ms: None,
        resource_requirements: ferrum_interfaces::scheduler::BatchResourceRequirements::default(),
        created_at: chrono::Utc::now(),
    };

    engine.inner.process_batch(&batch).await.unwrap();

    let sequences = engine.inner.sequences.read();
    let sequence = sequences
        .get(&request_id)
        .expect("resource-exhausted decode should remain queued");
    assert!(sequence.prefill_complete);
    assert!(sequence.kv_cache_handle().is_some());
    assert!(sequence.recurrent_state.is_some());
    assert_eq!(sequence.generated_tokens, vec![TokenId::new(6)]);
    assert_eq!(recurrent_manager.stats().active_states, 1);
}

#[tokio::test]
async fn process_batch_unified_decode_resource_exhausted_keeps_recurrent_state_waiting() {
    let mut config = EngineConfig::default();
    config.kv_cache.max_blocks = 128;
    let scheduler = Arc::new(ContinuousBatchScheduler::new(config.scheduler.clone()));
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> =
        Arc::new(PolicyTokenizer::new(64, &[("test", 5), ("ok", 6)]));
    let sampler: Arc<dyn Sampler + Send + Sync> = Arc::new(crate::registry::GreedySampler);
    let kv_cache: Arc<dyn KvCacheManager + Send + Sync> = Arc::new(MockKvCacheManager::new(128));
    let executor: Arc<dyn ModelExecutor + Send + Sync> = Arc::new(FailingUnifiedReserveExecutor {
        inner: FailingBatchPrefillExecutor {
            inner: RecurrentSpecExecutor {
                inner: MockModelExecutor::new(64, Duration::ZERO, Duration::ZERO),
            },
        },
    });
    let tensor_factory: Arc<dyn TensorFactory> = Arc::new(MockTensorFactory);
    let recurrent_manager = Arc::new(InMemoryRecurrentStateManager::new(
        InMemoryRecurrentStateConfig {
            total_memory_bytes: 8,
            total_batch_slots: 1,
        },
    ));
    let engine = ContinuousBatchEngine::new_with_speculation_and_recurrent_state_manager(
        config,
        scheduler.clone(),
        tokenizer.clone(),
        sampler,
        kv_cache,
        executor,
        tensor_factory,
        None,
        None,
        Some(recurrent_manager.clone()),
    );

    let mut request = policy_request();
    request.prompt = "test".to_string();
    request.sampling_params.max_tokens = 4;
    let request_id = request.id.clone();
    scheduler.submit(request.clone()).await.unwrap();
    let initial_batch = scheduler
        .next_batch(ferrum_interfaces::BatchHint::simple(4))
        .await
        .expect("decode request should first be scheduled as prefill");
    assert_eq!(initial_batch.requests.len(), 1);
    scheduler.mark_prefill_complete(&request_id, 1);

    let recurrent_spec = RecurrentStateSpec {
        request_id: request_id.clone(),
        num_layers: 1,
        tensors: vec![RecurrentStateTensorSpec::new(0, "delta_state", vec![4])],
        dtype: DataType::BF16,
        device: Device::CPU,
        max_batch_slots: 1,
    };
    let recurrent_state = recurrent_manager.allocate(&recurrent_spec).await.unwrap();
    let kv = engine
        .inner
        .make_model_kv_handle_with_seq("decode-cache".to_string(), 1);
    let mut sequence = SequenceState::new_with_tokenizer_and_model_vocab_size(
        request.clone(),
        vec![TokenId::new(5)],
        Some(tokenizer),
        Some(64),
    );
    sequence.generated_tokens.push(TokenId::new(6));
    sequence.prefill_complete = true;
    sequence.prefill_tokens_processed = 1;
    sequence.install_model_kv_without_owned_blocks(kv);
    sequence.commit_recurrent_state_admission(recurrent_state, 1);
    sequence.phase = RequestPhase::Decoding;
    engine
        .inner
        .sequences
        .write()
        .insert(request_id.clone(), sequence);

    let batch = scheduler
        .next_batch(ferrum_interfaces::BatchHint::simple(4))
        .await
        .expect("decode request should be scheduled");
    assert_eq!(batch.requests.len(), 1);

    engine.inner.process_batch(&batch).await.unwrap();

    let trace = scheduler.trace_snapshot();
    assert_eq!(trace.cancelled_total, 0);
    assert_eq!(trace.waiting_queue_len, 1);
    assert_eq!(trace.decode_queue_len, 0);
    assert_eq!(
        scheduler.trace_phase(&request_id),
        Some(RequestPhase::Waiting)
    );

    let sequences = engine.inner.sequences.read();
    let sequence = sequences
        .get(&request_id)
        .expect("resource-exhausted unified decode should remain queued");
    assert_eq!(sequence.phase, RequestPhase::Waiting);
    assert!(!sequence.prefill_complete);
    assert!(sequence.kv_cache_handle().is_none());
    assert!(sequence.recurrent_state.is_none());
    assert!(sequence.model_cache_id().is_none());
    assert_eq!(sequence.generated_tokens, vec![TokenId::new(6)]);
    assert_eq!(sequence.preemption_count, 1);
    let recurrent_stats = recurrent_manager.stats();
    assert_eq!(recurrent_stats.active_states, 0);
    assert_eq!(recurrent_stats.used_batch_slots, 0);
}

#[test]
fn sequence_state_detects_text_stop_before_length() {
    let tokenizer = PolicyTokenizer::new(8, &[("OK", 5), ("<END>", 6), ("TAIL", 7)]);
    let mut request = policy_request();
    request.sampling_params.max_tokens = 3;
    let mut state = SequenceState::new(request, vec![TokenId::new(0)]);
    state.generated_tokens = vec![TokenId::new(5), TokenId::new(6), TokenId::new(7)];
    state.stop_text_seqs = vec!["<END>".to_string()];

    assert_eq!(
        state.stop_reason(Some(&tokenizer)),
        Some(FinishReason::Stop)
    );
}

#[test]
fn model_decode_metadata_marks_structured_requests_for_full_logits() {
    let plain = SequenceState::new(policy_request(), vec![TokenId::new(0)]);
    assert_eq!(
        plain
            .model_decode_metadata()
            .get("ferrum_require_full_logits")
            .and_then(|value| value.as_bool()),
        None
    );
    assert_eq!(
        plain
            .model_decode_metadata()
            .get("ferrum_kv_capacity_hint")
            .and_then(|value| value.as_u64()),
        Some((1 + plain.sampling_params.max_tokens.saturating_sub(1)) as u64)
    );
    assert_eq!(
        plain
            .model_decode_metadata()
            .get("ferrum_kv_admission_target_len")
            .and_then(|value| value.as_u64()),
        Some(plain.prefill_context_len() as u64)
    );

    let mut request = policy_request();
    request.sampling_params.response_format = ferrum_types::ResponseFormat::JsonObject;
    let structured = SequenceState::new(request, vec![TokenId::new(0)]);
    assert_eq!(
        structured
            .model_decode_metadata()
            .get("ferrum_require_full_logits")
            .and_then(|value| value.as_bool()),
        Some(true)
    );

    let mut request = policy_request();
    request.sampling_params.response_format = ferrum_types::ResponseFormat::JsonSchema(
        r#"{"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"]}"#
            .to_string(),
    );
    let json_schema_without_tokenizer = SequenceState::new(request, vec![TokenId::new(0)]);
    assert_eq!(
        json_schema_without_tokenizer
            .model_decode_metadata()
            .get("ferrum_require_full_logits")
            .and_then(|value| value.as_bool()),
        Some(true)
    );
}

#[test]
fn sequence_state_prefill_context_preserves_generated_tokens_for_kv_recompute() {
    let mut state = SequenceState::new(policy_request(), vec![TokenId::new(10), TokenId::new(11)]);
    state.generated_tokens = vec![TokenId::new(12), TokenId::new(13)];

    assert_eq!(
        state.prefill_context_tokens(),
        vec![
            TokenId::new(10),
            TokenId::new(11),
            TokenId::new(12),
            TokenId::new(13)
        ]
    );
    assert_eq!(state.prefill_context_len(), 4);
    assert!(
        state
            .model_decode_metadata()
            .get("ferrum_kv_capacity_hint")
            .and_then(|value| value.as_u64())
            .unwrap()
            >= state.prefill_context_len() as u64
    );
    assert_eq!(
        state
            .model_decode_metadata()
            .get("ferrum_kv_admission_target_len")
            .and_then(|value| value.as_u64()),
        Some(state.prefill_context_len() as u64)
    );
}

#[test]
fn model_decode_metadata_keeps_sampling_masks_on_model_argmax_path() {
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> = Arc::new(PolicyTokenizer::new(
        4,
        &[("normal", 0), ("<s>", 1), ("<unk>", 2), ("ok", 3)],
    ));
    let state =
        SequenceState::new_with_tokenizer(policy_request(), vec![TokenId::new(0)], Some(tokenizer));

    assert_eq!(
        state
            .model_decode_metadata()
            .get("ferrum_require_full_logits")
            .and_then(|value| value.as_bool()),
        None
    );
    let LogitsReturnPolicy::GreedyArgmax {
        token_mask: Some(mask),
        repetition_penalty: None,
    } = state.model_decode_logits_policy()
    else {
        panic!("plain greedy decode should use a masked model-side argmax policy");
    };
    assert_eq!(mask.valid_token_mask[0], 1);
    assert_eq!(mask.valid_token_mask[2], 0);
}

#[test]
fn model_decode_argmax_mask_uses_model_vocab_for_extended_stop_tokens() {
    let mut tok = PolicyTokenizer::new(
        4,
        &[
            ("normal", 0),
            ("<s>", 1),
            ("<unk>", 2),
            ("ok", 3),
            ("<pad>", 4),
            ("<|im_end|>", 5),
        ],
    );
    tok.special.eos_token = Some(TokenId::new(5));
    tok.special.pad_token = Some(TokenId::new(4));
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> = Arc::new(tok);

    let state = SequenceState::new_with_tokenizer_and_model_vocab_size(
        policy_request(),
        vec![TokenId::new(0)],
        Some(tokenizer),
        Some(6),
    );

    let LogitsReturnPolicy::GreedyArgmax {
        token_mask: Some(mask),
        repetition_penalty: None,
    } = state.model_decode_logits_policy()
    else {
        panic!("plain greedy decode should use a model-side argmax policy");
    };
    assert_eq!(mask.len(), 6);
    assert_eq!(
        mask.valid_token_mask[5], 1,
        "extended EOS must remain selectable"
    );
    assert_eq!(
        mask.valid_token_mask[4], 0,
        "unallowed extended PAD must stay masked"
    );
}

#[test]
fn model_decode_logits_policy_keeps_repetition_penalty_on_greedy_argmax_path() {
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> = Arc::new(PolicyTokenizer::new(
        4,
        &[("normal", 0), ("<s>", 1), ("<unk>", 2), ("ok", 3)],
    ));
    let mut request = policy_request();
    request.sampling_params.repetition_penalty = 1.1;
    let mut state =
        SequenceState::new_with_tokenizer(request, vec![TokenId::new(0)], Some(tokenizer));
    state.generated_tokens = vec![TokenId::new(3), TokenId::new(3), TokenId::new(0)];

    let LogitsReturnPolicy::GreedyArgmax {
        token_mask: Some(mask),
        repetition_penalty: Some(penalty),
    } = state.model_decode_logits_policy()
    else {
        panic!("greedy repetition penalty should use model-side argmax policy");
    };
    assert_eq!(mask.valid_token_mask[2], 0);
    assert_eq!(penalty.penalty, 1.1);
    assert_eq!(penalty.token_ids.as_ref(), &[3, 0]);
}

#[test]
fn model_decode_logits_policy_requires_full_for_structured_output() {
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> = Arc::new(PolicyTokenizer::new(
        4,
        &[("normal", 0), ("<s>", 1), ("<unk>", 2), ("ok", 3)],
    ));
    let mut request = policy_request();
    request.sampling_params.response_format = ferrum_types::ResponseFormat::JsonObject;
    let state = SequenceState::new_with_tokenizer(request, vec![TokenId::new(0)], Some(tokenizer));

    assert!(matches!(
        state.model_decode_logits_policy(),
        LogitsReturnPolicy::FullLogits
    ));
}

#[test]
fn model_greedy_argmax_sentinel_accepts_masked_policy_token() {
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> = Arc::new(PolicyTokenizer::new(
        4,
        &[("normal", 0), ("<s>", 1), ("<unk>", 2), ("ok", 3)],
    ));
    let state = SequenceState::new_with_tokenizer(
        policy_request(),
        vec![TokenId::new(0)],
        Some(tokenizer.clone()),
    );

    assert!(state.requires_full_logits_for_sampling());
    state
        .accept_model_greedy_argmax_token(Some(tokenizer.as_ref()), TokenId::new(0))
        .unwrap();
    let err = state
        .accept_model_greedy_argmax_token(Some(tokenizer.as_ref()), TokenId::new(2))
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("model greedy argmax returned a forbidden token"),
        "model-side greedy argmax must not bypass forbidden token masks: {err}"
    );
    assert!(err.contains("token_id=2"), "{err}");
    assert!(err.contains("token_text=\"<unk>\""), "{err}");
    assert!(err.contains("forbidden_count="), "{err}");
    assert!(err.contains("argmax_mask="), "{err}");
    assert!(err.contains("value=0"), "{err}");
}

#[test]
fn model_greedy_argmax_sentinel_rejects_non_greedy_request() {
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> = Arc::new(PolicyTokenizer::new(
        4,
        &[("normal", 0), ("<s>", 1), ("<unk>", 2), ("ok", 3)],
    ));
    let mut request = policy_request();
    request.sampling_params.top_p = 0.8;
    let state =
        SequenceState::new_with_tokenizer(request, vec![TokenId::new(0)], Some(tokenizer.clone()));

    assert!(state
        .accept_model_greedy_argmax_token(Some(tokenizer.as_ref()), TokenId::new(0))
        .is_err());
}

#[test]
fn single_token_stop_sequence_also_matches_composite_decoded_token() {
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> = Arc::new(PolicyTokenizer::new(
        8,
        &[("OK", 5), ("\n", 6), ("OK \n\n", 7)],
    ));
    let mut request = policy_request();
    request.sampling_params.stop_sequences = vec!["\n".to_string()];
    let mut state =
        SequenceState::new_with_tokenizer(request, vec![TokenId::new(0)], Some(tokenizer.clone()));

    assert!(state.stop_token_ids.contains(&6));
    assert!(state.stop_text_seqs.contains(&"\n".to_string()));

    state.generated_tokens.push(TokenId::new(7));

    assert_eq!(
        state.stop_reason(Some(tokenizer.as_ref())),
        Some(FinishReason::Stop)
    );
}

#[test]
fn schema_guided_sampling_masks_extended_stop_tokens_before_accept() {
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> = Arc::new(PolicyTokenizer::new(
        6,
        &[
            ("{", 0),
            (" ", 1),
            ("x", 2),
            ("</s>", 3),
            ("}", 4),
            ("\"", 5),
            ("<|eot_id|>", 8),
        ],
    ));
    let mut request = policy_request();
    request.sampling_params.response_format = ferrum_types::ResponseFormat::JsonSchema(
        r#"{"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"]}"#
            .to_string(),
    );
    let mut state =
        SequenceState::new_with_tokenizer(request, vec![TokenId::new(0)], Some(tokenizer));

    assert!(state.regex_processor.is_some());
    assert!(
        state.stop_token_ids.contains(&8),
        "common eot token should be a resolved stop token"
    );

    let mut logits = vec![f32::NEG_INFINITY; 9];
    logits[0] = 1.0;
    logits[1] = 0.5;
    logits[8] = 100.0;

    let token = state.sample_with_processors(&mut logits).unwrap();

    assert_eq!(token.get(), 0);
    assert!(
        logits[8].is_infinite() && logits[8].is_sign_negative(),
        "schema-guided generation must not sample eot before the schema accepts"
    );
}

#[test]
fn schema_guided_sampling_masks_extended_control_tokens_before_accept() {
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> = Arc::new(PolicyTokenizer::new(
        6,
        &[
            ("{", 0),
            (" ", 1),
            ("x", 2),
            ("</s>", 3),
            ("}", 4),
            ("\"", 5),
            ("<think>", 7),
            ("<|eot_id|>", 8),
        ],
    ));
    let mut request = policy_request();
    request.sampling_params.response_format = ferrum_types::ResponseFormat::JsonSchema(
        r#"{"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"]}"#
            .to_string(),
    );
    let mut state =
        SequenceState::new_with_tokenizer(request, vec![TokenId::new(0)], Some(tokenizer));

    assert!(state.regex_processor.is_some());
    assert!(
        state.allowed_extended_token_ids.contains(&7),
        "think token should be an allowed generated control token outside base vocab"
    );
    assert!(
        !state.stop_token_ids.contains(&7),
        "think token should not be treated as a terminator"
    );

    let mut logits = vec![f32::NEG_INFINITY; 9];
    logits[0] = 1.0;
    logits[1] = 0.5;
    logits[7] = 100.0;
    logits[8] = 90.0;

    let token = state.sample_with_processors(&mut logits).unwrap();

    assert_eq!(token.get(), 0);
    assert!(
        logits[7].is_infinite() && logits[7].is_sign_negative(),
        "schema-guided generation must not sample invisible control tokens before accept"
    );
    assert!(
        logits[8].is_infinite() && logits[8].is_sign_negative(),
        "schema-guided generation must not sample stop tokens before accept"
    );
}

#[test]
fn schema_guided_sampling_allows_extended_stop_after_accept() {
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> = Arc::new(PolicyTokenizer::new(
        6,
        &[
            ("{", 0),
            (" ", 1),
            ("x", 2),
            ("</s>", 3),
            ("}", 4),
            ("\"", 5),
            ("<think>", 7),
            ("<|eot_id|>", 8),
        ],
    ));
    let mut request = policy_request();
    request.sampling_params.response_format =
        ferrum_types::ResponseFormat::JsonSchema(r#"{"enum":["x"]}"#.to_string());
    let mut state =
        SequenceState::new_with_tokenizer(request, vec![TokenId::new(0)], Some(tokenizer));
    state.generated_tokens = vec![TokenId::new(5), TokenId::new(2), TokenId::new(5)];

    let mut logits = vec![f32::NEG_INFINITY; 9];
    logits[1] = 80.0;
    logits[7] = 100.0;
    logits[8] = 90.0;

    let token = state.sample_with_processors(&mut logits).unwrap();

    assert_eq!(token.get(), 8);
    assert!(
        logits[7].is_infinite() && logits[7].is_sign_negative(),
        "completed schema output should still reject non-stop control tokens"
    );
    assert!(
        logits[8].is_finite(),
        "completed schema output should allow the resolved stop token"
    );
}

#[test]
fn sample_masks_unknown_pad_reserved_and_bos_tokens() {
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> = Arc::new(PolicyTokenizer::new(
        10,
        &[
            ("normal", 0),
            ("<s>", 1),
            ("<unk>", 2),
            ("</s>", 3),
            ("[PAD151935]", 4),
            ("<|reserved_special_token_0|>", 5),
            ("ok", 6),
            ("other", 7),
            ("byte-fallback", 8),
            ("\u{00ef}\u{00bf}\u{00bd}", 9),
        ],
    ));
    let mut state =
        SequenceState::new_with_tokenizer(policy_request(), vec![TokenId::new(0)], Some(tokenizer));
    let mut logits = vec![0.0f32; 10];
    logits[1] = 100.0;
    logits[2] = 99.0;
    logits[4] = 98.0;
    logits[5] = 97.0;
    logits[8] = 96.0;
    logits[9] = 95.0;
    logits[6] = 1.0;

    let token = state.sample_with_processors(&mut logits).unwrap();

    assert_eq!(token.get(), 6);
    for token_id in [1usize, 2, 4, 5, 8, 9] {
        assert_eq!(logits[token_id], f32::NEG_INFINITY);
    }
}

#[test]
fn sample_masks_tokenizer_vocab_holes() {
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> = Arc::new(PolicyTokenizer::new(
        12,
        &[
            ("normal", 0),
            ("<s>", 1),
            ("<unk>", 2),
            ("</s>", 3),
            ("<pad>", 4),
            ("ok", 6),
        ],
    ));
    let mut state =
        SequenceState::new_with_tokenizer(policy_request(), vec![TokenId::new(0)], Some(tokenizer));
    let mut logits = vec![0.0f32; 12];
    logits[11] = 100.0;
    logits[6] = 1.0;

    let token = state.sample_with_processors(&mut logits).unwrap();

    assert_eq!(token.get(), 6);
    assert_eq!(logits[11], f32::NEG_INFINITY);
}

#[test]
fn sample_resamples_candidate_that_would_flush_replacement_char() {
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> = Arc::new(PolicyTokenizer::new(
        8,
        &[
            ("normal", 0),
            ("<s>", 1),
            ("<unk>", 2),
            ("</s>", 3),
            ("<pad>", 4),
            ("bad-byte-lead", 5),
            ("ok", 6),
            ("valid-byte-cont", 7),
        ],
    ));
    let mut state = SequenceState::new_with_tokenizer(
        policy_request(),
        vec![TokenId::new(0)],
        Some(tokenizer.clone()),
    );
    state.generated_tokens.push(TokenId::new(5));

    let mut logits = vec![0.0f32; 8];
    logits[6] = 100.0;
    logits[7] = 1.0;

    let token = state
        .sample_with_processors_with_tokenizer(&mut logits, Some(tokenizer.as_ref()))
        .unwrap();

    assert_eq!(token.get(), 7);
    assert_eq!(logits[6], f32::NEG_INFINITY);
}

#[test]
fn sample_candidate_checks_from_streamed_text_boundary() {
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> = Arc::new(PolicyTokenizer::new(
        7,
        &[
            ("normal", 0),
            ("<s>", 1),
            ("<unk>", 2),
            ("</s>", 3),
            ("<pad>", 4),
            ("byte-fallback", 5),
            ("ok", 6),
        ],
    ));
    let mut state = SequenceState::new_with_tokenizer(
        policy_request(),
        vec![TokenId::new(0)],
        Some(tokenizer.clone()),
    );
    state.generated_tokens.push(TokenId::new(5));
    state.streamed_text_len = 0;

    assert!(state.sample_candidate_decodes_to_forbidden_output(
        Some(tokenizer.as_ref()),
        state.streamed_text_len,
        TokenId::new(6),
    ));
}

#[test]
fn sample_allows_generated_control_tokens_above_base_vocab() {
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> = Arc::new(PolicyTokenizer::new(
        6,
        &[
            ("normal", 0),
            ("<s>", 1),
            ("<unk>", 2),
            ("</s>", 3),
            ("ok", 4),
            ("</think>", 5),
            ("[PAD151935]", 6),
        ],
    ));
    let mut state =
        SequenceState::new_with_tokenizer(policy_request(), vec![TokenId::new(0)], Some(tokenizer));
    assert!(
        state.requires_full_logits_for_sampling(),
        "extended control-token masks require full logits; GPU argmax would bypass them"
    );
    let mut logits = vec![0.0f32; 7];
    logits[4] = 1.0;
    logits[5] = 90.0;
    logits[6] = 100.0;

    let token = state.sample_with_processors(&mut logits).unwrap();

    assert_eq!(token.get(), 5);
    assert_eq!(logits[5], 90.0);
    assert_eq!(logits[6], f32::NEG_INFINITY);
}

#[test]
fn sample_resamples_hidden_non_stop_control_tokens_above_base_vocab() {
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> = Arc::new(PolicyTokenizer::new(
        6,
        &[
            ("normal", 0),
            ("<s>", 1),
            ("<unk>", 2),
            ("</s>", 3),
            ("ok", 4),
            ("x", 5),
            ("<think>", 7),
        ],
    ));
    let mut state = SequenceState::new_with_tokenizer(
        policy_request(),
        vec![TokenId::new(0)],
        Some(tokenizer.clone()),
    );
    state.generated_tokens.push(TokenId::new(4));
    state.streamed_text_len = tokenizer
        .decode(&state.generated_tokens, true)
        .expect("generated prefix decodes")
        .len();

    assert!(
        state.allowed_extended_token_ids.contains(&7),
        "think token should be whitelisted as a generated control token"
    );
    assert!(
        !state.stop_token_ids.contains(&7),
        "think token should not be treated as a stop token"
    );

    let mut logits = vec![f32::NEG_INFINITY; 8];
    logits[5] = 1.0;
    logits[7] = 100.0;

    let token = state
        .sample_with_processors_with_tokenizer(&mut logits, Some(tokenizer.as_ref()))
        .unwrap();

    assert_eq!(token.get(), 5);
    assert_eq!(logits[7], f32::NEG_INFINITY);
}

#[test]
fn sample_masks_metadata_initial_token_text_only_before_first_generation() {
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> = Arc::new(PolicyTokenizer::new(
        6,
        &[
            ("normal", 0),
            ("<s>", 1),
            ("<unk>", 2),
            ("</s>", 3),
            ("ok", 4),
            ("</think>", 5),
        ],
    ));
    let mut request = policy_request();
    request.metadata.insert(
        "ferrum_initial_forbidden_token_texts".to_string(),
        serde_json::json!(["</think>"]),
    );
    let mut state =
        SequenceState::new_with_tokenizer(request, vec![TokenId::new(0)], Some(tokenizer));
    let LogitsReturnPolicy::GreedyArgmax {
        token_mask: Some(mask),
        repetition_penalty: None,
    } = state.model_decode_logits_policy()
    else {
        panic!("first greedy decode should use the initial token mask");
    };
    assert_eq!(mask.valid_token_mask[5], 0);

    let mut first_logits = vec![0.0f32; 6];
    first_logits[0] = 1.0;
    first_logits[5] = 100.0;
    let first = state.sample_with_processors(&mut first_logits).unwrap();
    assert_eq!(first.get(), 0);
    assert_eq!(first_logits[5], f32::NEG_INFINITY);

    state.generated_tokens.push(first);
    let LogitsReturnPolicy::GreedyArgmax {
        token_mask: Some(mask),
        repetition_penalty: None,
    } = state.model_decode_logits_policy()
    else {
        panic!("subsequent greedy decode should use the regular token mask");
    };
    assert_eq!(mask.valid_token_mask[5], 1);
    let mut next_logits = vec![0.0f32; 6];
    next_logits[0] = 1.0;
    next_logits[5] = 100.0;
    let next = state.sample_with_processors(&mut next_logits).unwrap();
    assert_eq!(next.get(), 5);
    assert_eq!(next_logits[5], 100.0);
}
