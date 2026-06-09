use ferrum_interfaces::kv_dtype::KvInt8;
use ferrum_kernels::backend::{BackendInt8KvOps, KvLayer, MoeLlmBackend};
use ferrum_types::{FerrumError, Result};

use crate::common::{DecoderOnlyLLM, LlmRuntimeConfig};

use super::llama_family::{LlamaFamilyModel, LlamaStageHiddenBridgeTiming};

fn pipeline_decode_profile_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("FERRUM_DECODE_OP_PROFILE").is_some())
}

fn elapsed_micros_u64(t0: std::time::Instant) -> u64 {
    t0.elapsed().as_micros().min(u64::MAX as u128) as u64
}

const MIN_OVERLAPPED_DECODE_BATCH: usize = 16;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PipelineHiddenDtype {
    F32,
}

impl PipelineHiddenDtype {
    fn as_str(self) -> &'static str {
        match self {
            Self::F32 => "f32",
        }
    }

    fn elem_size_bytes(self) -> usize {
        match self {
            Self::F32 => size_of::<f32>(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PipelineHiddenDevice {
    Host,
    BackendDevice { ordinal: Option<usize> },
}

impl PipelineHiddenDevice {
    fn as_str(self) -> &'static str {
        match self {
            Self::Host => "host",
            Self::BackendDevice { .. } => "backend_device",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PipelineHiddenLayout {
    RowMajor,
}

impl PipelineHiddenLayout {
    fn as_str(self) -> &'static str {
        match self {
            Self::RowMajor => "row_major",
        }
    }
}

#[allow(dead_code)]
pub(crate) enum PipelineHiddenStorage<B: MoeLlmBackend> {
    Host(Vec<f32>),
    Device(B::Buffer),
}

pub(crate) struct PipelineHidden<B: MoeLlmBackend> {
    shape: [usize; 2],
    dtype: PipelineHiddenDtype,
    device: PipelineHiddenDevice,
    layout: PipelineHiddenLayout,
    storage: PipelineHiddenStorage<B>,
}

impl<B: MoeLlmBackend> PipelineHidden<B> {
    pub(crate) fn host(data: Vec<f32>, batch: usize, hidden_size: usize) -> Self {
        assert_eq!(
            data.len(),
            batch * hidden_size,
            "pipeline hidden host buffer length {} != batch * hidden_size {}",
            data.len(),
            batch * hidden_size
        );
        Self {
            shape: [batch, hidden_size],
            dtype: PipelineHiddenDtype::F32,
            device: PipelineHiddenDevice::Host,
            layout: PipelineHiddenLayout::RowMajor,
            storage: PipelineHiddenStorage::Host(data),
        }
    }

    pub(crate) fn row_count(&self) -> usize {
        self.shape[0]
    }

    pub(crate) fn hidden_size(&self) -> usize {
        self.shape[1]
    }

    pub(crate) fn len_bytes(&self) -> usize {
        self.shape[0] * self.shape[1] * self.dtype.elem_size_bytes()
    }

    pub(crate) fn host_slice(&self) -> &[f32] {
        match &self.storage {
            PipelineHiddenStorage::Host(data) => data,
            PipelineHiddenStorage::Device(_) => {
                panic!("device-resident PipelineHidden cannot be read through host_slice")
            }
        }
    }

    fn metadata_json(&self) -> serde_json::Value {
        serde_json::json!({
            "shape": self.shape,
            "dtype": self.dtype.as_str(),
            "device": self.device.as_str(),
            "layout": self.layout.as_str(),
            "len_bytes": self.len_bytes(),
        })
    }
}

pub(crate) trait LlamaPipelineStageBatchOps<B: MoeLlmBackend> {
    fn decode_stage_tokens_to_hidden_batch(
        &mut self,
        batch: &[(String, u32, u32)],
    ) -> PipelineHidden<B>;

    fn decode_stage_hidden_from_host_batch(
        &mut self,
        batch: &[(String, u32, u32)],
        hidden: &PipelineHidden<B>,
    ) -> (PipelineHidden<B>, LlamaStageHiddenBridgeTiming);

    fn logits_from_hidden_batch(
        &mut self,
        hidden: &PipelineHidden<B>,
        force_full_logits: bool,
    ) -> Vec<Vec<f32>>;
}

impl<B> LlamaPipelineStageBatchOps<B> for LlamaFamilyModel<B, KvInt8>
where
    B: MoeLlmBackend + BackendInt8KvOps,
{
    fn decode_stage_tokens_to_hidden_batch(
        &mut self,
        batch: &[(String, u32, u32)],
    ) -> PipelineHidden<B> {
        let h = self.config().hidden_size;
        let mut hidden = Vec::with_capacity(batch.len() * h);
        for (cache_id, token, pos) in batch {
            hidden.extend_from_slice(&self.decode_stage_token_to_hidden(cache_id, *token, *pos));
        }
        PipelineHidden::host(hidden, batch.len(), h)
    }

    fn decode_stage_hidden_from_host_batch(
        &mut self,
        batch: &[(String, u32, u32)],
        hidden: &PipelineHidden<B>,
    ) -> (PipelineHidden<B>, LlamaStageHiddenBridgeTiming) {
        let h = self.config().hidden_size;
        let hidden_slice = hidden.host_slice();
        assert_eq!(
            hidden_slice.len(),
            batch.len() * h,
            "hidden length {} != batch * hidden_size {}",
            hidden_slice.len(),
            batch.len() * h
        );
        let mut out = Vec::with_capacity(hidden_slice.len());
        let mut bridge_timing = LlamaStageHiddenBridgeTiming::default();
        for (row, (cache_id, _, pos)) in batch.iter().enumerate() {
            let start = row * h;
            let (row_hidden, row_timing) = self.decode_stage_hidden_from_host_with_timing(
                cache_id,
                &hidden_slice[start..start + h],
                *pos,
            );
            out.extend_from_slice(&row_hidden);
            bridge_timing = bridge_timing.add(row_timing);
        }
        (PipelineHidden::host(out, batch.len(), h), bridge_timing)
    }

    fn logits_from_hidden_batch(
        &mut self,
        hidden: &PipelineHidden<B>,
        _force_full_logits: bool,
    ) -> Vec<Vec<f32>> {
        let h = self.config().hidden_size;
        let hidden_slice = hidden.host_slice();
        assert_eq!(
            hidden_slice.len(),
            hidden.row_count() * h,
            "hidden length {} != row_count * hidden_size {}",
            hidden_slice.len(),
            hidden.row_count() * h
        );
        (0..hidden.row_count())
            .map(|row| {
                let start = row * h;
                self.logits_from_hidden(&hidden_slice[start..start + h])
            })
            .collect()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LlamaPipelineTransport {
    HostHiddenBridge,
}

impl LlamaPipelineTransport {
    fn as_str(self) -> &'static str {
        match self {
            Self::HostHiddenBridge => "host-hidden-bridge",
        }
    }

    fn stage_bridge(self) -> LlamaPipelineStageBridge {
        match self {
            Self::HostHiddenBridge => LlamaPipelineStageBridge::Host,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LlamaPipelineStageBridge {
    Host,
    CudaPeer,
    CudaDeviceStaged,
}

impl LlamaPipelineStageBridge {
    fn as_str(self) -> &'static str {
        match self {
            Self::Host => "host",
            Self::CudaPeer => "cuda_peer",
            Self::CudaDeviceStaged => "cuda_device_staged",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LlamaPipelineMode {
    Batch,
    Overlapped,
}

impl LlamaPipelineMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Batch => "batch",
            Self::Overlapped => "overlapped",
        }
    }

    pub fn default_for_stage_count(stage_count: usize) -> Self {
        if stage_count == 2 {
            Self::Overlapped
        } else {
            Self::Batch
        }
    }

    pub fn from_config_value(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "batch" => Ok(Self::Batch),
            "overlapped" => Ok(Self::Overlapped),
            other => Err(FerrumError::config(format!(
                "layer_split_pipeline_mode must be batch or overlapped, got {other:?}"
            ))),
        }
    }
}

#[derive(Debug, Clone)]
struct PipelineDecodeStats {
    calls: u64,
    overlapped_calls: u64,
    rows: u64,
    max_batch: u64,
    last_batch: u64,
    microbatch_count_max: u64,
    microbatch_count_last: u64,
    microbatch_size_max: u64,
    microbatch_size_last: u64,
    in_flight_stage_count_max: u64,
    in_flight_stage_count_last: u64,
    queue_depth_max: u64,
    queue_depth_last: u64,
    host_bridge_bytes_total: u64,
    host_bridge_bytes_last: u64,
    bridge_us_total: u64,
    bridge_us_last: u64,
    host_copy_us_total: u64,
    host_copy_us_last: u64,
    device_copy_us_total: u64,
    device_copy_us_last: u64,
    stage_us_total: Vec<u64>,
    stage_us_last: Vec<u64>,
    logits_us_total: u64,
    logits_us_last: u64,
    total_us_total: u64,
    total_us_last: u64,
}

impl PipelineDecodeStats {
    fn new(stage_count: usize) -> Self {
        Self {
            calls: 0,
            overlapped_calls: 0,
            rows: 0,
            max_batch: 0,
            last_batch: 0,
            microbatch_count_max: 0,
            microbatch_count_last: 0,
            microbatch_size_max: 0,
            microbatch_size_last: 0,
            in_flight_stage_count_max: 0,
            in_flight_stage_count_last: 0,
            queue_depth_max: 0,
            queue_depth_last: 0,
            host_bridge_bytes_total: 0,
            host_bridge_bytes_last: 0,
            bridge_us_total: 0,
            bridge_us_last: 0,
            host_copy_us_total: 0,
            host_copy_us_last: 0,
            device_copy_us_total: 0,
            device_copy_us_last: 0,
            stage_us_total: vec![0; stage_count],
            stage_us_last: vec![0; stage_count],
            logits_us_total: 0,
            logits_us_last: 0,
            total_us_total: 0,
            total_us_last: 0,
        }
    }

    fn record(
        &mut self,
        batch: usize,
        microbatch_count: usize,
        microbatch_size: usize,
        in_flight_stage_count: usize,
        queue_depth: usize,
        overlapped: bool,
        host_bridge_bytes: usize,
        bridge_timing: LlamaStageHiddenBridgeTiming,
        stage_us: &[u64],
        logits_us: u64,
        total_us: u64,
    ) {
        if self.stage_us_total.len() != stage_us.len() {
            self.stage_us_total.resize(stage_us.len(), 0);
            self.stage_us_last.resize(stage_us.len(), 0);
        }
        self.calls = self.calls.saturating_add(1);
        if overlapped {
            self.overlapped_calls = self.overlapped_calls.saturating_add(1);
        }
        self.rows = self.rows.saturating_add(batch as u64);
        self.max_batch = self.max_batch.max(batch as u64);
        self.last_batch = batch as u64;
        self.microbatch_count_max = self.microbatch_count_max.max(microbatch_count as u64);
        self.microbatch_count_last = microbatch_count as u64;
        self.microbatch_size_max = self.microbatch_size_max.max(microbatch_size as u64);
        self.microbatch_size_last = microbatch_size as u64;
        self.in_flight_stage_count_max = self
            .in_flight_stage_count_max
            .max(in_flight_stage_count as u64);
        self.in_flight_stage_count_last = in_flight_stage_count as u64;
        self.queue_depth_max = self.queue_depth_max.max(queue_depth as u64);
        self.queue_depth_last = queue_depth as u64;
        self.host_bridge_bytes_total = self
            .host_bridge_bytes_total
            .saturating_add(host_bridge_bytes as u64);
        self.host_bridge_bytes_last = host_bridge_bytes as u64;
        self.bridge_us_total = self.bridge_us_total.saturating_add(bridge_timing.bridge_us);
        self.bridge_us_last = bridge_timing.bridge_us;
        self.host_copy_us_total = self
            .host_copy_us_total
            .saturating_add(bridge_timing.host_copy_us);
        self.host_copy_us_last = bridge_timing.host_copy_us;
        self.device_copy_us_total = self
            .device_copy_us_total
            .saturating_add(bridge_timing.device_copy_us);
        self.device_copy_us_last = bridge_timing.device_copy_us;
        for (idx, value) in stage_us.iter().copied().enumerate() {
            self.stage_us_total[idx] = self.stage_us_total[idx].saturating_add(value);
            self.stage_us_last[idx] = value;
        }
        self.logits_us_total = self.logits_us_total.saturating_add(logits_us);
        self.logits_us_last = logits_us;
        self.total_us_total = self.total_us_total.saturating_add(total_us);
        self.total_us_last = total_us;
    }

    fn avg_per_call(&self, value: u64) -> Option<u64> {
        (self.calls > 0).then(|| value / self.calls)
    }

    fn json(&self) -> serde_json::Value {
        let stage_us_avg: Vec<Option<u64>> = self
            .stage_us_total
            .iter()
            .map(|value| self.avg_per_call(*value))
            .collect();
        serde_json::json!({
            "calls": self.calls,
            "overlapped_calls": self.overlapped_calls,
            "rows": self.rows,
            "max_batch": self.max_batch,
            "last_batch": self.last_batch,
            "microbatch_count_max": self.microbatch_count_max,
            "microbatch_count_last": self.microbatch_count_last,
            "microbatch_size_max": self.microbatch_size_max,
            "microbatch_size_last": self.microbatch_size_last,
            "in_flight_stage_count_max": self.in_flight_stage_count_max,
            "in_flight_stage_count_last": self.in_flight_stage_count_last,
            "queue_depth_max": self.queue_depth_max,
            "queue_depth_last": self.queue_depth_last,
            "host_bridge_bytes_total": self.host_bridge_bytes_total,
            "host_bridge_bytes_last": self.host_bridge_bytes_last,
            "host_bridge_bytes_avg": self.avg_per_call(self.host_bridge_bytes_total),
            "bridge_us_total": self.bridge_us_total,
            "bridge_us_last": self.bridge_us_last,
            "bridge_us_avg": self.avg_per_call(self.bridge_us_total),
            "host_copy_us_total": self.host_copy_us_total,
            "host_copy_us_last": self.host_copy_us_last,
            "host_copy_us_avg": self.avg_per_call(self.host_copy_us_total),
            "device_copy_us_total": self.device_copy_us_total,
            "device_copy_us_last": self.device_copy_us_last,
            "device_copy_us_avg": self.avg_per_call(self.device_copy_us_total),
            "stage_us_total": self.stage_us_total,
            "stage_us_last": self.stage_us_last,
            "stage_us_avg": stage_us_avg,
            "logits_us_total": self.logits_us_total,
            "logits_us_last": self.logits_us_last,
            "logits_us_avg": self.avg_per_call(self.logits_us_total),
            "total_us_total": self.total_us_total,
            "total_us_last": self.total_us_last,
            "total_us_avg": self.avg_per_call(self.total_us_total),
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LlamaPipelineStagePlacement {
    pub backend_device_ordinal: Option<usize>,
}

impl LlamaPipelineStagePlacement {
    pub fn default_backend_device() -> Self {
        Self {
            backend_device_ordinal: None,
        }
    }

    pub fn backend_device(ordinal: usize) -> Self {
        Self {
            backend_device_ordinal: Some(ordinal),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LlamaPipelinePlacement {
    stages: Vec<LlamaPipelineStagePlacement>,
    transport: LlamaPipelineTransport,
    pipeline_mode: LlamaPipelineMode,
}

impl LlamaPipelinePlacement {
    pub fn unplaced(stage_count: usize) -> Self {
        Self {
            stages: vec![LlamaPipelineStagePlacement::default_backend_device(); stage_count],
            transport: LlamaPipelineTransport::HostHiddenBridge,
            pipeline_mode: LlamaPipelineMode::default_for_stage_count(stage_count),
        }
    }

    pub fn from_backend_device_ordinals(stage_device_ordinals: Vec<Option<usize>>) -> Self {
        let stage_count = stage_device_ordinals.len();
        Self {
            stages: stage_device_ordinals
                .into_iter()
                .map(|backend_device_ordinal| LlamaPipelineStagePlacement {
                    backend_device_ordinal,
                })
                .collect(),
            transport: LlamaPipelineTransport::HostHiddenBridge,
            pipeline_mode: LlamaPipelineMode::default_for_stage_count(stage_count),
        }
    }

    pub fn with_pipeline_mode(mut self, pipeline_mode: LlamaPipelineMode) -> Self {
        self.pipeline_mode = pipeline_mode;
        self
    }

    pub fn len(&self) -> usize {
        self.stages.len()
    }

    pub fn is_empty(&self) -> bool {
        self.stages.is_empty()
    }

    pub fn stage(&self, idx: usize) -> LlamaPipelineStagePlacement {
        self.stages[idx]
    }

    pub fn stages(&self) -> &[LlamaPipelineStagePlacement] {
        &self.stages
    }

    pub fn transport(&self) -> LlamaPipelineTransport {
        self.transport
    }

    pub fn stage_bridge(&self) -> LlamaPipelineStageBridge {
        self.transport.stage_bridge()
    }

    pub fn pipeline_mode(&self) -> LlamaPipelineMode {
        self.pipeline_mode
    }

    pub fn stage_device_ordinals(&self) -> Vec<Option<usize>> {
        self.stages
            .iter()
            .map(|stage| stage.backend_device_ordinal)
            .collect()
    }

    fn has_explicit_device_ordinals(&self) -> bool {
        self.stages
            .iter()
            .any(|stage| stage.backend_device_ordinal.is_some())
    }
}

pub struct LlamaFamilyPipelineModel<B: MoeLlmBackend, K: KvLayer<B>> {
    stages: Vec<LlamaFamilyModel<B, K>>,
    placement: LlamaPipelinePlacement,
    runtime_cfg: LlmRuntimeConfig,
    decode_stats: PipelineDecodeStats,
}

impl<B: MoeLlmBackend, K: KvLayer<B>> LlamaFamilyPipelineModel<B, K> {
    pub fn new(stages: Vec<LlamaFamilyModel<B, K>>) -> Result<Self> {
        let placement = LlamaPipelinePlacement::unplaced(stages.len());
        Self::new_with_placement(stages, placement)
    }

    pub fn new_with_backend_device_ordinals(
        stages: Vec<LlamaFamilyModel<B, K>>,
        stage_device_ordinals: Vec<Option<usize>>,
    ) -> Result<Self> {
        Self::new_with_placement(
            stages,
            LlamaPipelinePlacement::from_backend_device_ordinals(stage_device_ordinals),
        )
    }

    pub fn new_with_placement(
        stages: Vec<LlamaFamilyModel<B, K>>,
        placement: LlamaPipelinePlacement,
    ) -> Result<Self> {
        if stages.is_empty() {
            return Err(FerrumError::model(
                "LlamaFamilyPipelineModel requires at least one stage",
            ));
        }
        if placement.len() != stages.len() {
            return Err(FerrumError::model(format!(
                "Llama pipeline stage device count {} must match stage count {}",
                placement.len(),
                stages.len()
            )));
        }
        if placement.has_explicit_device_ordinals() && !B::supports_device_ordinal_scope() {
            return Err(FerrumError::unsupported(
                "Llama layer-split pipeline requested explicit backend device ordinals, \
                 but the selected backend does not support device-scoped execution",
            ));
        }
        if placement.pipeline_mode() == LlamaPipelineMode::Overlapped && placement.len() != 2 {
            return Err(FerrumError::model(
                "Llama layer-split overlapped pipeline mode requires exactly two stages",
            ));
        }
        if stages.first().is_some_and(|stage| stage.embed.is_none()) {
            return Err(FerrumError::model(
                "first Llama pipeline stage must load embedding weights",
            ));
        }
        if stages.last().is_some_and(|stage| stage.lm_head.is_none()) {
            return Err(FerrumError::model(
                "last Llama pipeline stage must load lm_head weights",
            ));
        }

        let runtime_cfg = stages[0].runtime_cfg.clone();
        let mut expected_start = 0usize;
        for stage in &stages {
            if stage.runtime_cfg.hidden_size != runtime_cfg.hidden_size
                || stage.runtime_cfg.vocab_size != runtime_cfg.vocab_size
                || stage.runtime_cfg.num_kv_heads != runtime_cfg.num_kv_heads
                || stage.runtime_cfg.head_dim != runtime_cfg.head_dim
                || stage.runtime_cfg.max_seq_len != runtime_cfg.max_seq_len
            {
                return Err(FerrumError::model(
                    "Llama pipeline stages must share runtime dimensions",
                ));
            }
            let range = stage.source_layer_range();
            if range.start != expected_start {
                return Err(FerrumError::model(format!(
                    "Llama pipeline stage range starts at {}, expected {expected_start}",
                    range.start
                )));
            }
            expected_start = range.end;
        }
        if expected_start != runtime_cfg.num_layers {
            return Err(FerrumError::model(format!(
                "Llama pipeline stages cover {expected_start} layers but model has {}",
                runtime_cfg.num_layers
            )));
        }

        let stage_count = placement.len();
        Ok(Self {
            stages,
            placement,
            runtime_cfg,
            decode_stats: PipelineDecodeStats::new(stage_count),
        })
    }

    pub fn stages(&self) -> &[LlamaFamilyModel<B, K>] {
        &self.stages
    }

    pub fn placement(&self) -> &LlamaPipelinePlacement {
        &self.placement
    }

    fn pipeline_mode(&self) -> LlamaPipelineMode {
        self.placement.pipeline_mode()
    }

    fn decode_microbatch_size(&self, batch_len: usize) -> usize {
        match self.pipeline_mode() {
            LlamaPipelineMode::Overlapped if batch_len < MIN_OVERLAPPED_DECODE_BATCH => {
                batch_len.max(1)
            }
            LlamaPipelineMode::Overlapped => ((batch_len + 1) / 2).max(1),
            LlamaPipelineMode::Batch => batch_len.max(1),
        }
    }

    fn last_hidden_row<'a>(&self, hidden: &'a [f32], seq_len: usize) -> &'a [f32] {
        let h = self.runtime_cfg.hidden_size;
        &hidden[(seq_len - 1) * h..seq_len * h]
    }
}

#[allow(private_bounds)]
impl<B, K> LlamaFamilyPipelineModel<B, K>
where
    B: MoeLlmBackend,
    K: KvLayer<B>,
    LlamaFamilyModel<B, K>: DecoderOnlyLLM + LlamaPipelineStageBatchOps<B> + Send,
{
    fn decode_batch_sequential_internal(
        &mut self,
        batch: &[(String, u32, u32)],
        force_full_logits: bool,
    ) -> Vec<Vec<f32>> {
        let profile = pipeline_decode_profile_enabled();
        let total_t0 = std::time::Instant::now();
        let mut stage_us: Vec<u64> = Vec::with_capacity(self.stages.len());
        let mut host_bridge_bytes = 0usize;
        let mut bridge_timing = LlamaStageHiddenBridgeTiming::default();
        let stage_bridge = self.placement.stage_bridge();

        let stage_t0 = std::time::Instant::now();
        let mut hidden =
            B::with_device_ordinal(self.placement.stage(0).backend_device_ordinal, || {
                self.stages[0].decode_stage_tokens_to_hidden_batch(batch)
            });
        stage_us.push(elapsed_micros_u64(stage_t0));
        for idx in 1..self.stages.len() {
            let device = self.placement.stage(idx).backend_device_ordinal;
            host_bridge_bytes = host_bridge_bytes.saturating_add(hidden.len_bytes());
            let stage_t0 = std::time::Instant::now();
            let (next_hidden, stage_bridge_timing) = B::with_device_ordinal(device, || {
                self.stages[idx].decode_stage_hidden_from_host_batch(batch, &hidden)
            });
            hidden = next_hidden;
            bridge_timing = bridge_timing.add(stage_bridge_timing);
            stage_us.push(elapsed_micros_u64(stage_t0));
        }
        let last_idx = self.stages.len() - 1;
        let logits_t0 = std::time::Instant::now();
        let logits = B::with_device_ordinal(
            self.placement.stage(last_idx).backend_device_ordinal,
            || self.stages[last_idx].logits_from_hidden_batch(&hidden, force_full_logits),
        );
        let logits_us = elapsed_micros_u64(logits_t0);
        let total_us = elapsed_micros_u64(total_t0);
        self.decode_stats.record(
            batch.len(),
            1,
            batch.len().max(1),
            1,
            0,
            false,
            host_bridge_bytes,
            bridge_timing,
            &stage_us,
            logits_us,
            total_us,
        );
        if profile {
            static PIPELINE_PROFILE_CALLS: std::sync::atomic::AtomicU64 =
                std::sync::atomic::AtomicU64::new(0);
            let n = PIPELINE_PROFILE_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if n.is_multiple_of(8) {
                eprintln!(
                    "[pipeline-decode-profile] call#{} m={} hidden={:?} mode=batch bridge={} host_bridge_bytes={} bridge_us={} host_copy_us={} device_copy_us={} stage_us={:?} logits_us={} total_us={}",
                    n,
                    batch.len(),
                    hidden.metadata_json(),
                    stage_bridge.as_str(),
                    host_bridge_bytes,
                    bridge_timing.bridge_us,
                    bridge_timing.host_copy_us,
                    bridge_timing.device_copy_us,
                    stage_us,
                    logits_us,
                    total_us,
                );
            }
        }
        logits
    }

    fn decode_batch_overlapped_two_stage(
        &mut self,
        batch: &[(String, u32, u32)],
        force_full_logits: bool,
    ) -> Vec<Vec<f32>> {
        debug_assert_eq!(self.stages.len(), 2);
        let microbatch_size = self.decode_microbatch_size(batch.len());
        let chunks: Vec<Vec<(String, u32, u32)>> = batch
            .chunks(microbatch_size)
            .map(|chunk| chunk.to_vec())
            .collect();
        let chunk_count = chunks.len();
        if chunk_count <= 1 {
            return self.decode_batch_sequential_internal(batch, force_full_logits);
        }

        let profile = pipeline_decode_profile_enabled();
        let total_t0 = std::time::Instant::now();
        let stage0_device = self.placement.stage(0).backend_device_ordinal;
        let stage1_device = self.placement.stage(1).backend_device_ordinal;
        let stage_bridge = self.placement.stage_bridge();
        let (stage0_slice, stage1_slice) = self.stages.split_at_mut(1);
        let stage0 = &mut stage0_slice[0];
        let stage1 = &mut stage1_slice[0];
        let (tx, rx) = std::sync::mpsc::sync_channel::<(
            usize,
            Vec<(String, u32, u32)>,
            PipelineHidden<B>,
            u64,
        )>(1);
        let mut ordered_logits: Vec<Option<Vec<Vec<f32>>>> = vec![None; chunk_count];
        let mut stage0_us_total = 0u64;
        let mut stage1_us_total = 0u64;
        let mut logits_us_total = 0u64;
        let mut host_bridge_bytes = 0usize;
        let mut bridge_timing = LlamaStageHiddenBridgeTiming::default();

        std::thread::scope(|scope| {
            let worker = scope.spawn(move || {
                for (idx, chunk) in chunks.into_iter().enumerate() {
                    let stage_t0 = std::time::Instant::now();
                    let hidden = B::with_device_ordinal(stage0_device, || {
                        stage0.decode_stage_tokens_to_hidden_batch(&chunk)
                    });
                    let stage_us = elapsed_micros_u64(stage_t0);
                    if tx.send((idx, chunk, hidden, stage_us)).is_err() {
                        break;
                    }
                }
            });

            for _ in 0..chunk_count {
                let (idx, chunk, hidden, stage0_us) = rx
                    .recv()
                    .expect("pipeline stage0 worker ended before sending all microbatches");
                stage0_us_total = stage0_us_total.saturating_add(stage0_us);
                host_bridge_bytes = host_bridge_bytes.saturating_add(hidden.len_bytes());

                let stage_t0 = std::time::Instant::now();
                let (hidden, stage_bridge_timing) = B::with_device_ordinal(stage1_device, || {
                    stage1.decode_stage_hidden_from_host_batch(&chunk, &hidden)
                });
                bridge_timing = bridge_timing.add(stage_bridge_timing);
                stage1_us_total = stage1_us_total.saturating_add(elapsed_micros_u64(stage_t0));

                let logits_t0 = std::time::Instant::now();
                let logits = B::with_device_ordinal(stage1_device, || {
                    stage1.logits_from_hidden_batch(&hidden, force_full_logits)
                });
                logits_us_total = logits_us_total.saturating_add(elapsed_micros_u64(logits_t0));
                ordered_logits[idx] = Some(logits);
            }

            worker
                .join()
                .expect("pipeline stage0 worker panicked during overlapped decode");
        });

        let total_us = elapsed_micros_u64(total_t0);
        let stage_us = vec![stage0_us_total, stage1_us_total];
        self.decode_stats.record(
            batch.len(),
            chunk_count,
            microbatch_size,
            2,
            1,
            true,
            host_bridge_bytes,
            bridge_timing,
            &stage_us,
            logits_us_total,
            total_us,
        );
        if profile {
            static PIPELINE_PROFILE_CALLS: std::sync::atomic::AtomicU64 =
                std::sync::atomic::AtomicU64::new(0);
            let n = PIPELINE_PROFILE_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if n.is_multiple_of(8) {
                eprintln!(
                    "[pipeline-decode-profile] call#{} m={} mode=overlapped microbatch_size={} microbatch_count={} bridge={} host_bridge_bytes={} bridge_us={} host_copy_us={} device_copy_us={} stage_us={:?} logits_us={} total_us={}",
                    n,
                    batch.len(),
                    microbatch_size,
                    chunk_count,
                    stage_bridge.as_str(),
                    host_bridge_bytes,
                    bridge_timing.bridge_us,
                    bridge_timing.host_copy_us,
                    bridge_timing.device_copy_us,
                    stage_us,
                    logits_us_total,
                    total_us,
                );
            }
        }

        let mut logits = Vec::with_capacity(batch.len());
        for chunk_logits in ordered_logits {
            logits.extend(
                chunk_logits.expect("pipeline overlapped decode missing logits for microbatch"),
            );
        }
        logits
    }
}

impl<B, K> DecoderOnlyLLM for LlamaFamilyPipelineModel<B, K>
where
    B: MoeLlmBackend,
    K: KvLayer<B>,
    LlamaFamilyModel<B, K>: DecoderOnlyLLM + LlamaPipelineStageBatchOps<B> + Send,
{
    fn config(&self) -> &LlmRuntimeConfig {
        &self.runtime_cfg
    }

    fn cache_metrics_snapshot(&self) -> Option<serde_json::Value> {
        let stage_bridge = self.placement.stage_bridge();
        let pipeline_mode = self.pipeline_mode();
        Some(serde_json::json!({
            "position": "llama-layer-split-pipeline",
            "stage_count": self.stages.len() as u64,
            "stage_device_ordinals": self.placement.stage_device_ordinals(),
            "transport": self.placement.transport().as_str(),
            "selected_pipeline_mode": pipeline_mode.as_str(),
            "selected_stage_bridge": stage_bridge.as_str(),
            "pipeline_hidden": {
                "dtype": PipelineHiddenDtype::F32.as_str(),
                "device": PipelineHiddenDevice::Host.as_str(),
                "layout": PipelineHiddenLayout::RowMajor.as_str(),
            },
            "pipeline_decode": self.decode_stats.json(),
        }))
    }

    fn prepare(&mut self, cache_id: &str, max_tokens: usize) {
        for (idx, stage) in self.stages.iter_mut().enumerate() {
            let device = self.placement.stage(idx).backend_device_ordinal;
            B::with_device_ordinal(device, || {
                stage.ensure_scratch(max_tokens);
                stage.ensure_kv(cache_id);
            });
        }
    }

    fn kv_capacity(&self) -> usize {
        self.stages
            .iter()
            .map(|stage| stage.kv_capacity())
            .min()
            .unwrap_or(self.runtime_cfg.max_seq_len)
    }

    fn prefill(&mut self, cache_id: &str, tokens: &[u32]) -> Vec<f32> {
        assert!(!tokens.is_empty(), "pipeline prefill called with no tokens");
        let pos_offset = self.stages[0].cache_len(cache_id);
        let mut hidden =
            B::with_device_ordinal(self.placement.stage(0).backend_device_ordinal, || {
                self.stages[0].prefill_stage_tokens_to_hidden(cache_id, tokens, pos_offset)
            });
        for idx in 1..self.stages.len() {
            let device = self.placement.stage(idx).backend_device_ordinal;
            hidden = B::with_device_ordinal(device, || {
                self.stages[idx].prefill_stage_hidden_from_host(
                    cache_id,
                    &hidden,
                    tokens.len(),
                    pos_offset,
                )
            });
        }
        let last_hidden = self.last_hidden_row(&hidden, tokens.len()).to_vec();
        let last_idx = self.stages.len() - 1;
        B::with_device_ordinal(
            self.placement.stage(last_idx).backend_device_ordinal,
            || self.stages[last_idx].logits_from_hidden(&last_hidden),
        )
    }

    fn decode(&mut self, cache_id: &str, token: u32, pos: u32) -> Vec<f32> {
        let total_t0 = std::time::Instant::now();
        let mut stage_us: Vec<u64> = Vec::with_capacity(self.stages.len());
        let mut host_bridge_bytes = 0usize;
        let mut bridge_timing = LlamaStageHiddenBridgeTiming::default();

        let stage_t0 = std::time::Instant::now();
        let mut hidden =
            B::with_device_ordinal(self.placement.stage(0).backend_device_ordinal, || {
                self.stages[0].decode_stage_token_to_hidden(cache_id, token, pos)
            });
        stage_us.push(elapsed_micros_u64(stage_t0));
        for idx in 1..self.stages.len() {
            let device = self.placement.stage(idx).backend_device_ordinal;
            host_bridge_bytes = host_bridge_bytes.saturating_add(hidden.len() * size_of::<f32>());
            let stage_t0 = std::time::Instant::now();
            let (next_hidden, stage_bridge_timing) = B::with_device_ordinal(device, || {
                self.stages[idx].decode_stage_hidden_from_host_with_timing(cache_id, &hidden, pos)
            });
            hidden = next_hidden;
            bridge_timing = bridge_timing.add(stage_bridge_timing);
            stage_us.push(elapsed_micros_u64(stage_t0));
        }
        let last_idx = self.stages.len() - 1;
        let logits_t0 = std::time::Instant::now();
        let logits = B::with_device_ordinal(
            self.placement.stage(last_idx).backend_device_ordinal,
            || self.stages[last_idx].logits_from_hidden(&hidden),
        );
        self.decode_stats.record(
            1,
            1,
            1,
            1,
            0,
            false,
            host_bridge_bytes,
            bridge_timing,
            &stage_us,
            elapsed_micros_u64(logits_t0),
            elapsed_micros_u64(total_t0),
        );
        logits
    }

    fn decode_batch(&mut self, batch: &[(String, u32, u32)]) -> Vec<Vec<f32>> {
        self.decode_batch_with_full_logits(batch, false)
    }

    fn decode_batch_with_full_logits(
        &mut self,
        batch: &[(String, u32, u32)],
        force_full_logits: bool,
    ) -> Vec<Vec<f32>> {
        if batch.is_empty() {
            return Vec::new();
        }
        if batch.len() == 1 && !force_full_logits {
            let (cache_id, token, pos) = &batch[0];
            return vec![self.decode(cache_id, *token, *pos)];
        }

        if self.pipeline_mode() == LlamaPipelineMode::Overlapped {
            self.decode_batch_overlapped_two_stage(batch, force_full_logits)
        } else {
            self.decode_batch_sequential_internal(batch, force_full_logits)
        }
    }

    fn release(&mut self, cache_id: &str) {
        for (idx, stage) in self.stages.iter_mut().enumerate() {
            B::with_device_ordinal(self.placement.stage(idx).backend_device_ordinal, || {
                stage.release(cache_id);
            });
        }
    }

    fn truncate_kv(&mut self, cache_id: &str, new_len: usize) {
        for (idx, stage) in self.stages.iter_mut().enumerate() {
            B::with_device_ordinal(self.placement.stage(idx).backend_device_ordinal, || {
                stage.truncate_kv(cache_id, new_len);
            });
        }
    }

    fn reset(&mut self) {
        for (idx, stage) in self.stages.iter_mut().enumerate() {
            B::with_device_ordinal(self.placement.stage(idx).backend_device_ordinal, || {
                stage.reset();
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use ferrum_interfaces::kv_dtype::KvFp16;
    use ferrum_kernels::backend::cpu::CpuBackend;
    use ferrum_quantization::{DenseLinear, QuantConfig, WeightLoader};
    use ferrum_types::{FerrumError, Result};

    use super::*;
    use crate::models::llama_family::{LlamaFamilyConfig, LlamaFamilyLayerStageConfig};

    struct ParityLoader {
        cfg: LlamaFamilyConfig,
    }

    impl ParityLoader {
        fn new(cfg: LlamaFamilyConfig) -> Self {
            Self { cfg }
        }

        fn deterministic_values(name: &str, len: usize, base: f32, scale: f32) -> Vec<f32> {
            let mut hash = 0x811c9dc5u32;
            for byte in name.bytes() {
                hash ^= byte as u32;
                hash = hash.wrapping_mul(0x01000193);
            }
            (0..len)
                .map(|idx| {
                    let mixed = hash
                        .wrapping_add((idx as u32).wrapping_mul(0x9e3779b9))
                        .rotate_left((idx % 17) as u32);
                    let centered = (mixed % 23) as f32 - 11.0;
                    base + centered * scale
                })
                .collect()
        }

        fn layer_norm_values(&self, name: &str) -> Vec<f32> {
            Self::deterministic_values(name, self.cfg.hidden_size, 1.0, 0.005)
        }

        fn linear_dims(&self, name: &str) -> Result<(usize, usize)> {
            let q_dim = self.cfg.num_heads * self.cfg.head_dim;
            let kv_dim = self.cfg.num_kv_heads * self.cfg.head_dim;
            if name.ends_with(".self_attn.qkv_proj") {
                Ok((q_dim + 2 * kv_dim, self.cfg.hidden_size))
            } else if name.ends_with(".self_attn.o_proj") {
                Ok((self.cfg.hidden_size, q_dim))
            } else if name.ends_with(".mlp.gate_up_proj") {
                Ok((2 * self.cfg.intermediate_size, self.cfg.hidden_size))
            } else if name.ends_with(".mlp.down_proj") {
                Ok((self.cfg.hidden_size, self.cfg.intermediate_size))
            } else if name == "lm_head" || name == "model.embed_tokens" {
                Ok((self.cfg.vocab_size, self.cfg.hidden_size))
            } else {
                Err(FerrumError::model(format!(
                    "unexpected linear requested by parity loader: {name}"
                )))
            }
        }
    }

    impl WeightLoader<CpuBackend> for ParityLoader {
        fn load_tensor(&self, name: &str) -> Result<Vec<f32>> {
            if name == "model.embed_tokens.weight" {
                return Ok(Self::deterministic_values(
                    name,
                    self.cfg.vocab_size * self.cfg.hidden_size,
                    0.0,
                    0.02,
                ));
            }
            if name == "model.norm.weight"
                || name.ends_with(".input_layernorm.weight")
                || name.ends_with(".post_attention_layernorm.weight")
            {
                return Ok(self.layer_norm_values(name));
            }
            Err(FerrumError::model(format!(
                "unexpected tensor requested by parity loader: {name}"
            )))
        }

        fn load_linear(
            &self,
            name: &str,
        ) -> Result<Box<dyn ferrum_quantization::Linear<CpuBackend>>> {
            let (out_features, in_features) = self.linear_dims(name)?;
            let weights = Self::deterministic_values(name, out_features * in_features, 0.0, 0.015);
            Ok(Box::new(DenseLinear::<CpuBackend>::from_rows(
                &weights,
                out_features,
                in_features,
            )))
        }

        fn has_tensor(&self, name: &str) -> bool {
            name == "lm_head.weight"
        }

        fn quant_config(&self) -> Option<&QuantConfig> {
            None
        }
    }

    fn parity_config(num_layers: usize) -> LlamaFamilyConfig {
        LlamaFamilyConfig {
            hidden_size: 4,
            intermediate_size: 8,
            num_heads: 2,
            num_kv_heads: 2,
            head_dim: 2,
            num_layers,
            vocab_size: 7,
            max_seq_len: 16,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.0,
            rope_scaling: None,
            rope_interleaved: false,
            has_qk_norm: false,
            sliding_window: 0,
        }
    }

    fn build_full_and_pipeline() -> (
        LlamaFamilyModel<CpuBackend, KvFp16>,
        LlamaFamilyPipelineModel<CpuBackend, KvFp16>,
    ) {
        build_full_and_pipeline_with_mode(LlamaPipelineMode::default_for_stage_count(2))
    }

    fn build_full_and_pipeline_with_mode(
        pipeline_mode: LlamaPipelineMode,
    ) -> (
        LlamaFamilyModel<CpuBackend, KvFp16>,
        LlamaFamilyPipelineModel<CpuBackend, KvFp16>,
    ) {
        let cfg = parity_config(3);
        let loader = ParityLoader::new(cfg.clone());
        let full = LlamaFamilyModel::<CpuBackend, KvFp16>::new(cfg.clone(), &loader).unwrap();
        let stage0 = LlamaFamilyModel::<CpuBackend, KvFp16>::new_layer_stage(
            cfg.clone(),
            &loader,
            LlamaFamilyLayerStageConfig::pipeline_stage(0..1, true, false),
        )
        .unwrap();
        let stage1 = LlamaFamilyModel::<CpuBackend, KvFp16>::new_layer_stage(
            cfg,
            &loader,
            LlamaFamilyLayerStageConfig::pipeline_stage(1..3, false, true),
        )
        .unwrap();
        let pipeline = LlamaFamilyPipelineModel::new_with_placement(
            vec![stage0, stage1],
            LlamaPipelinePlacement::unplaced(2).with_pipeline_mode(pipeline_mode),
        )
        .unwrap();
        (full, pipeline)
    }

    fn assert_logits_close(label: &str, expected: &[f32], actual: &[f32]) {
        assert_eq!(expected.len(), actual.len(), "{label} length mismatch");
        let max_diff = expected
            .iter()
            .zip(actual)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 1e-5,
            "{label} logits diverged: max_diff={max_diff} expected={expected:?} actual={actual:?}"
        );
    }

    #[test]
    fn pipeline_prefill_matches_full_model_on_multi_token_cpu_model() {
        let (mut full, mut pipeline) = build_full_and_pipeline();

        let full_logits = full.prefill("full", &[0, 1, 2, 3]);
        let pipeline_logits = pipeline.prefill("pipe", &[0, 1, 2, 3]);

        assert_eq!(pipeline.config().num_layers, 3);
        assert_eq!(pipeline.stages().len(), 2);
        assert_eq!(
            pipeline.placement().stage_device_ordinals(),
            vec![None, None]
        );
        assert_eq!(
            pipeline.placement().transport(),
            LlamaPipelineTransport::HostHiddenBridge
        );
        let metrics = pipeline.cache_metrics_snapshot().unwrap();
        assert_eq!(metrics["selected_pipeline_mode"], "overlapped");
        assert_eq!(metrics["selected_stage_bridge"], "host");
        assert_eq!(metrics["pipeline_decode"]["calls"], 0);
        assert_logits_close("multi-token prefill", &full_logits, &pipeline_logits);
    }

    #[test]
    fn pipeline_decode_after_multi_token_prefill_matches_full_model() {
        let (mut full, mut pipeline) = build_full_and_pipeline();

        let _ = full.prefill("full", &[0, 1, 2]);
        let _ = pipeline.prefill("pipe", &[0, 1, 2]);
        let full_logits_3 = full.decode("full", 3, 3);
        let pipeline_logits_3 = pipeline.decode("pipe", 3, 3);
        assert_logits_close("decode pos 3", &full_logits_3, &pipeline_logits_3);

        let full_logits_4 = full.decode("full", 4, 4);
        let pipeline_logits_4 = pipeline.decode("pipe", 4, 4);
        assert_logits_close("decode pos 4", &full_logits_4, &pipeline_logits_4);

        let metrics = pipeline.cache_metrics_snapshot().unwrap();
        assert_eq!(metrics["pipeline_decode"]["calls"], 2);
        assert_eq!(metrics["pipeline_decode"]["overlapped_calls"], 0);
        assert_eq!(metrics["pipeline_decode"]["rows"], 2);
        assert_eq!(metrics["pipeline_decode"]["max_batch"], 1);
        assert_eq!(
            metrics["pipeline_decode"]["stage_us_last"]
                .as_array()
                .unwrap()
                .len(),
            pipeline.stages().len()
        );
        assert!(
            metrics["pipeline_decode"]["host_bridge_bytes_total"]
                .as_u64()
                .unwrap()
                > 0
        );
        assert!(
            metrics["pipeline_decode"]["bridge_us_total"]
                .as_u64()
                .unwrap()
                > 0
        );
        assert!(
            metrics["pipeline_decode"]["host_copy_us_total"]
                .as_u64()
                .unwrap()
                > 0
        );
        assert!(
            metrics["pipeline_decode"]["host_copy_us_total"]
                .as_u64()
                .unwrap()
                + metrics["pipeline_decode"]["device_copy_us_total"]
                    .as_u64()
                    .unwrap()
                > 0
        );
    }

    #[test]
    fn pipeline_decode_batch_matches_full_model_and_preserves_order() {
        let (mut full, mut pipeline) = build_full_and_pipeline();

        let prefills: Vec<Vec<u32>> = (0..16)
            .map(|idx| {
                let len = 1 + (idx % 4);
                (0..len).map(|offset| ((idx + offset) % 7) as u32).collect()
            })
            .collect();
        for (idx, tokens) in prefills.iter().enumerate() {
            let _ = full.prefill(&format!("full_{idx}"), tokens);
            let _ = pipeline.prefill(&format!("pipe_{idx}"), tokens);
        }

        let first_tokens: Vec<u32> = (0..prefills.len())
            .map(|idx| ((idx + 5) % 7) as u32)
            .collect();
        let second_tokens: Vec<u32> = (0..prefills.len())
            .map(|idx| ((idx + 1) % 7) as u32)
            .collect();
        let full_first: Vec<_> = prefills
            .iter()
            .enumerate()
            .map(|(idx, tokens)| {
                (
                    format!("full_{idx}"),
                    first_tokens[idx],
                    tokens.len() as u32,
                )
            })
            .collect();
        let pipe_first: Vec<_> = prefills
            .iter()
            .enumerate()
            .map(|(idx, tokens)| {
                (
                    format!("pipe_{idx}"),
                    first_tokens[idx],
                    tokens.len() as u32,
                )
            })
            .collect();

        let expected = full.decode_batch(&full_first);
        let actual = pipeline.decode_batch(&pipe_first);

        assert_eq!(actual.len(), prefills.len());
        for row in 0..prefills.len() {
            assert_logits_close(
                &format!("decode batch row {row}"),
                &expected[row],
                &actual[row],
            );
        }

        let full_next: Vec<_> = prefills
            .iter()
            .enumerate()
            .map(|(idx, tokens)| {
                (
                    format!("full_{idx}"),
                    second_tokens[idx],
                    tokens.len() as u32 + 1,
                )
            })
            .collect();
        let pipe_next: Vec<_> = prefills
            .iter()
            .enumerate()
            .map(|(idx, tokens)| {
                (
                    format!("pipe_{idx}"),
                    second_tokens[idx],
                    tokens.len() as u32 + 1,
                )
            })
            .collect();
        let expected_next = full.decode_batch(&full_next);
        let actual_next = pipeline.decode_batch(&pipe_next);

        for row in 0..prefills.len() {
            assert_logits_close(
                &format!("follow-up decode batch row {row}"),
                &expected_next[row],
                &actual_next[row],
            );
        }

        let metrics = pipeline.cache_metrics_snapshot().unwrap();
        assert_eq!(metrics["pipeline_decode"]["calls"], 2);
        assert_eq!(metrics["pipeline_decode"]["overlapped_calls"], 2);
        assert_eq!(metrics["pipeline_decode"]["rows"], 32);
        assert_eq!(metrics["pipeline_decode"]["max_batch"], 16);
        assert_eq!(metrics["pipeline_decode"]["last_batch"], 16);
        assert_eq!(metrics["pipeline_decode"]["microbatch_count_max"], 2);
        assert_eq!(metrics["pipeline_decode"]["microbatch_size_max"], 8);
        assert_eq!(metrics["pipeline_decode"]["in_flight_stage_count_max"], 2);
        assert_eq!(metrics["pipeline_decode"]["queue_depth_max"], 1);
        assert_eq!(
            metrics["pipeline_decode"]["stage_us_last"]
                .as_array()
                .unwrap()
                .len(),
            pipeline.stages().len()
        );
        assert!(
            metrics["pipeline_decode"]["host_bridge_bytes_total"]
                .as_u64()
                .unwrap()
                > 0
        );
        assert!(
            metrics["pipeline_decode"]["bridge_us_total"]
                .as_u64()
                .unwrap()
                > 0
        );
        assert!(
            metrics["pipeline_decode"]["host_copy_us_total"]
                .as_u64()
                .unwrap()
                > 0
        );
        assert!(
            metrics["pipeline_decode"]["host_copy_us_total"]
                .as_u64()
                .unwrap()
                + metrics["pipeline_decode"]["device_copy_us_total"]
                    .as_u64()
                    .unwrap()
                > 0
        );
    }

    #[test]
    fn pipeline_overlapped_mode_keeps_small_batches_whole() {
        let (mut full, mut pipeline) = build_full_and_pipeline();

        let prefills: Vec<Vec<u32>> = (0..8)
            .map(|idx| {
                let len = 1 + (idx % 3);
                (0..len).map(|offset| ((idx + offset) % 7) as u32).collect()
            })
            .collect();
        for (idx, tokens) in prefills.iter().enumerate() {
            let _ = full.prefill(&format!("full_{idx}"), tokens);
            let _ = pipeline.prefill(&format!("pipe_{idx}"), tokens);
        }

        let full_batch: Vec<_> = prefills
            .iter()
            .enumerate()
            .map(|(idx, tokens)| {
                (
                    format!("full_{idx}"),
                    ((idx + 5) % 7) as u32,
                    tokens.len() as u32,
                )
            })
            .collect();
        let pipe_batch: Vec<_> = prefills
            .iter()
            .enumerate()
            .map(|(idx, tokens)| {
                (
                    format!("pipe_{idx}"),
                    ((idx + 5) % 7) as u32,
                    tokens.len() as u32,
                )
            })
            .collect();

        let expected = full.decode_batch(&full_batch);
        let actual = pipeline.decode_batch(&pipe_batch);

        assert_eq!(actual.len(), prefills.len());
        for row in 0..prefills.len() {
            assert_logits_close(
                &format!("small batch row {row}"),
                &expected[row],
                &actual[row],
            );
        }

        let metrics = pipeline.cache_metrics_snapshot().unwrap();
        assert_eq!(metrics["selected_pipeline_mode"], "overlapped");
        assert_eq!(metrics["pipeline_decode"]["calls"], 1);
        assert_eq!(metrics["pipeline_decode"]["overlapped_calls"], 0);
        assert_eq!(metrics["pipeline_decode"]["max_batch"], 8);
        assert_eq!(metrics["pipeline_decode"]["microbatch_count_max"], 1);
        assert_eq!(metrics["pipeline_decode"]["microbatch_size_max"], 8);
        assert_eq!(metrics["pipeline_decode"]["in_flight_stage_count_max"], 1);
        assert_eq!(metrics["pipeline_decode"]["queue_depth_max"], 0);
    }

    #[test]
    fn pipeline_batch_mode_uses_stage_batch_without_overlap() {
        let (mut full, mut pipeline) = build_full_and_pipeline_with_mode(LlamaPipelineMode::Batch);

        let _ = full.prefill("full_a", &[0, 1]);
        let _ = full.prefill("full_b", &[2, 3, 4]);
        let _ = pipeline.prefill("pipe_a", &[0, 1]);
        let _ = pipeline.prefill("pipe_b", &[2, 3, 4]);

        let expected =
            full.decode_batch(&[("full_a".to_string(), 5, 2), ("full_b".to_string(), 6, 3)]);
        let actual =
            pipeline.decode_batch(&[("pipe_a".to_string(), 5, 2), ("pipe_b".to_string(), 6, 3)]);

        assert_logits_close("batch mode row 0", &expected[0], &actual[0]);
        assert_logits_close("batch mode row 1", &expected[1], &actual[1]);

        let metrics = pipeline.cache_metrics_snapshot().unwrap();
        assert_eq!(metrics["selected_pipeline_mode"], "batch");
        assert_eq!(metrics["pipeline_decode"]["calls"], 1);
        assert_eq!(metrics["pipeline_decode"]["overlapped_calls"], 0);
        assert_eq!(metrics["pipeline_decode"]["microbatch_count_max"], 1);
        assert_eq!(metrics["pipeline_decode"]["microbatch_size_max"], 2);
        assert_eq!(metrics["pipeline_decode"]["in_flight_stage_count_max"], 1);
        assert_eq!(metrics["pipeline_decode"]["queue_depth_max"], 0);
    }

    #[test]
    fn pipeline_incremental_prefill_matches_full_model_position_offset() {
        let (mut full, mut pipeline) = build_full_and_pipeline();

        let _ = full.prefill("full", &[0, 1]);
        let _ = pipeline.prefill("pipe", &[0, 1]);
        let full_logits = full.prefill("full", &[2, 3]);
        let pipeline_logits = pipeline.prefill("pipe", &[2, 3]);

        assert_logits_close("incremental prefill", &full_logits, &pipeline_logits);
    }

    #[test]
    fn pipeline_rejects_device_ordinals_without_backend_scope_support() {
        let cfg = parity_config(2);
        let loader = ParityLoader::new(cfg.clone());
        let stage0 = LlamaFamilyModel::<CpuBackend, KvFp16>::new_layer_stage(
            cfg.clone(),
            &loader,
            LlamaFamilyLayerStageConfig::pipeline_stage(0..1, true, false),
        )
        .unwrap();
        let stage1 = LlamaFamilyModel::<CpuBackend, KvFp16>::new_layer_stage(
            cfg,
            &loader,
            LlamaFamilyLayerStageConfig::pipeline_stage(1..2, false, true),
        )
        .unwrap();

        let err = match LlamaFamilyPipelineModel::new_with_backend_device_ordinals(
            vec![stage0, stage1],
            vec![Some(0), Some(1)],
        ) {
            Ok(_) => panic!("pipeline unexpectedly accepted unsupported device ordinals"),
            Err(err) => err.to_string(),
        };

        assert!(err.contains("does not support device-scoped execution"));
    }
}
