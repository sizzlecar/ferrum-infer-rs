//! Original context custody, backed by real CPU numerical operations.
use super::*;
use crate::common::DecoderOnlyLLM;
use crate::models::llama_family::LlamaFamilyConfig;
use crate::test_support::SyntheticLlamaLoader;
use ferrum_interfaces::kv_dtype::KvFp16;
use ferrum_kernels::backend::cpu::{CpuBackend, CpuContext};
use ferrum_kernels::backend::timer::BackendTimer;
use ferrum_kernels::backend::{
    AttnConfig, Backend, BackendGraph, BackendKvDtype, BackendPagedKv, BackendQuantGguf,
    BackendQuantMarlin, Dtype, HostDtype,
};
use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    Arc,
};

struct TrackedCpu;
#[derive(Clone, Copy, PartialEq, Eq)]
enum GraphFault {
    SecondLayer,
    SecondLayerUnresolved,
    EndCapture,
    ReplayMissing,
    ReplayError,
    ReplayUnresolved,
}
struct Context {
    cpu: CpuContext,
    identity: Arc<()>,
    unresolved: Arc<AtomicBool>,
    calls: Arc<AtomicUsize>,
    begins: Arc<AtomicUsize>,
    retired: Arc<AtomicUsize>,
    storage_retirement_blocked: Arc<AtomicBool>,
    retirement_blocked: Arc<AtomicBool>,
    terminal_error: Arc<AtomicBool>,
    // Armed only by the LoRA allocation-failure test on its original context.
    successful_allocations_before_failure: Option<usize>,
    // Fault injection exercises the real model's bookkeeping/custody only;
    // CPU arithmetic remains eager and does not emulate CUDA graph numerics.
    graph_fault: Option<GraphFault>,
    capture_active: bool,
    capture_gemms: usize,
    graph_installed: bool,
    replay_attempts: usize,
}
struct Timer;
impl BackendTimer<TrackedCpu> for Timer {
    fn new() -> Self {
        Self
    }
    fn record_start(&mut self, _: &mut Context) {}
    fn record_end(&mut self, _: &mut Context) {}
    fn elapsed_ms(&self) -> f64 {
        0.0
    }
}
impl BackendGraph for TrackedCpu {
    unsafe fn begin_graph_capture(ctx: &mut Context) -> Result<()> {
        ctx.capture_active = true;
        ctx.capture_gemms = 0;
        Ok(())
    }
    unsafe fn end_graph_capture(ctx: &mut Context, _: u64) -> Result<()> {
        assert!(ctx.capture_active);
        ctx.capture_active = false;
        if ctx.graph_fault == Some(GraphFault::EndCapture) {
            return Err(FerrumError::backend(
                "controlled graph installation failure",
            ));
        }
        ctx.graph_installed = true;
        Ok(())
    }
    unsafe fn replay_graph(ctx: &mut Context, _: u64) -> Result<bool> {
        ctx.replay_attempts += 1;
        assert!(
            ctx.graph_installed,
            "only an installed original graph may replay"
        );
        match ctx.graph_fault {
            Some(GraphFault::ReplayMissing) => Ok(false),
            Some(GraphFault::ReplayUnresolved) => {
                ctx.unresolved.store(true, Ordering::SeqCst);
                Err(FerrumError::backend("controlled unresolved graph launch"))
            }
            _ => Err(FerrumError::backend("controlled graph launch failure")),
        }
    }
    fn abort_graph_capture(ctx: &mut Context) -> Result<()> {
        ctx.capture_active = false;
        Ok(())
    }
    fn retire_graphs_for_storage_mutation(ctx: &mut Context) -> Result<()> {
        if ctx.storage_retirement_blocked.load(Ordering::SeqCst) {
            return Err(FerrumError::backend("controlled graph retirement failure"));
        }
        Self::observe_completion(ctx).into_result()?;
        ctx.graph_installed = false;
        Ok(())
    }
}
impl BackendPagedKv for TrackedCpu {}
impl BackendQuantMarlin for TrackedCpu {}
impl BackendQuantGguf for TrackedCpu {}
impl BackendMoeFused for TrackedCpu {}
impl BackendKvDtype<KvFp16> for TrackedCpu {
    type KvBuffer = Vec<f32>;
    type KvScales = ();
}
impl Backend for TrackedCpu {
    type Buffer = Vec<f32>;
    type Context = Context;
    type Timer = Timer;
    fn make_timer() -> Timer {
        Timer
    }
    fn new_context(config: &ferrum_types::BackendContextConfig) -> Result<Context> {
        Ok(Context {
            cpu: CpuBackend::new_context(config)?,
            identity: Arc::new(()),
            unresolved: Arc::new(AtomicBool::new(false)),
            calls: Arc::new(AtomicUsize::new(0)),
            begins: Arc::new(AtomicUsize::new(0)),
            retired: Arc::new(AtomicUsize::new(0)),
            storage_retirement_blocked: Arc::new(AtomicBool::new(false)),
            retirement_blocked: Arc::new(AtomicBool::new(false)),
            terminal_error: Arc::new(AtomicBool::new(false)),
            successful_allocations_before_failure: None,
            graph_fault: None,
            capture_active: false,
            capture_gemms: 0,
            graph_installed: false,
            replay_attempts: 0,
        })
    }
    fn context_config(ctx: &Context) -> &ferrum_types::BackendContextConfig {
        CpuBackend::context_config(&ctx.cpu)
    }
    fn host_execution_is_synchronous() -> bool {
        true
    }
    fn begin_execution(ctx: &mut Context) -> Result<()> {
        assert_eq!(ctx.retired.load(Ordering::SeqCst), 0);
        if ctx.unresolved.load(Ordering::SeqCst) {
            return Err(FerrumError::backend("original context unresolved"));
        }
        ctx.begins.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
    fn observe_completion(ctx: &mut Context) -> BackendCompletion {
        if ctx.capture_active || ctx.unresolved.load(Ordering::SeqCst) {
            BackendCompletion::Unresolved(FerrumError::backend(
                "controlled original context remains unresolved",
            ))
        } else if ctx.terminal_error.load(Ordering::SeqCst) {
            BackendCompletion::Failed(FerrumError::backend("controlled original terminal failure"))
        } else {
            BackendCompletion::Succeeded
        }
    }
    fn retire_context(ctx: &mut Context) -> Result<()> {
        if let BackendCompletion::Unresolved(error) = Self::observe_completion(ctx) {
            return Err(error);
        }
        if ctx.retirement_blocked.load(Ordering::SeqCst) {
            return Err(FerrumError::backend("controlled original cleanup failure"));
        }
        ctx.retired.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
    fn sync(ctx: &mut Context) {
        Self::observe_completion(ctx).into_result().unwrap();
    }
    fn graph_capture_in_flight(ctx: &Context) -> bool {
        ctx.capture_active
    }
    fn alloc(n: usize) -> Vec<f32> {
        CpuBackend::alloc(n)
    }
    fn alloc_typed(dtype: Dtype, n: usize) -> Vec<f32> {
        CpuBackend::alloc_typed(dtype, n)
    }
    fn from_slice(x: &[f32]) -> Vec<f32> {
        CpuBackend::from_slice(x)
    }
    fn from_slice_typed<T: HostDtype>(x: &[T]) -> Vec<f32> {
        CpuBackend::from_slice_typed(x)
    }
    fn to_vec(x: &Vec<f32>, n: usize) -> Vec<f32> {
        CpuBackend::to_vec(x, n)
    }
    fn alloc_in_context(ctx: &mut Context, n: usize) -> Vec<f32> {
        ctx.calls.fetch_add(1, Ordering::SeqCst);
        if let Some(remaining) = ctx.successful_allocations_before_failure {
            if remaining == 0 {
                ctx.successful_allocations_before_failure = None;
                ctx.unresolved.store(true, Ordering::SeqCst);
                std::panic::panic_any(FerrumError::backend(
                    "controlled second LoRA workspace allocation failure",
                ));
            }
            ctx.successful_allocations_before_failure = Some(remaining - 1);
        }
        Self::alloc(n)
    }
    fn alloc_typed_in_context(ctx: &mut Context, dtype: Dtype, n: usize) -> Vec<f32> {
        ctx.calls.fetch_add(1, Ordering::SeqCst);
        Self::alloc_typed(dtype, n)
    }
    fn from_slice_in_context(ctx: &mut Context, x: &[f32]) -> Vec<f32> {
        ctx.calls.fetch_add(1, Ordering::SeqCst);
        Self::from_slice(x)
    }
    fn from_slice_typed_in_context<T: HostDtype>(ctx: &mut Context, x: &[T]) -> Vec<f32> {
        ctx.calls.fetch_add(1, Ordering::SeqCst);
        Self::from_slice_typed(x)
    }
    fn zero_buffer(ctx: &mut Self::Context, _buf: &mut Self::Buffer, _len: usize) -> Result<()> {
        CpuBackend::zero_buffer(&mut ctx.cpu, _buf, _len)
    }
    fn write_typed<T: HostDtype>(ctx: &mut Self::Context, dst: &mut Self::Buffer, data: &[T]) {
        ctx.calls.fetch_add(1, Ordering::SeqCst);
        CpuBackend::write_typed(&mut ctx.cpu, dst, data)
    }
    fn gemm(
        ctx: &mut Self::Context,
        a: &Self::Buffer,
        b: &Self::Buffer,
        out: &mut Self::Buffer,
        m: usize,
        n: usize,
        k: usize,
    ) {
        ctx.calls.fetch_add(1, Ordering::SeqCst);
        if ctx.capture_active {
            ctx.capture_gemms += 1;
            if ctx.capture_gemms == 5
                && matches!(
                    ctx.graph_fault,
                    Some(GraphFault::SecondLayer | GraphFault::SecondLayerUnresolved)
                )
            {
                if ctx.graph_fault == Some(GraphFault::SecondLayerUnresolved) {
                    ctx.unresolved.store(true, Ordering::SeqCst);
                }
                std::panic::panic_any(FerrumError::backend(
                    "controlled second layer capture failure",
                ));
            }
        }
        CpuBackend::gemm(&mut ctx.cpu, a, b, out, m, n, k)
    }
    fn rms_norm(
        ctx: &mut Self::Context,
        x: &Self::Buffer,
        w: &Self::Buffer,
        eps: f32,
        out: &mut Self::Buffer,
        tokens: usize,
        dim: usize,
    ) {
        ctx.calls.fetch_add(1, Ordering::SeqCst);
        CpuBackend::rms_norm(&mut ctx.cpu, x, w, eps, out, tokens, dim)
    }
    fn fused_add_rms_norm(
        ctx: &mut Self::Context,
        residual: &mut Self::Buffer,
        x: &Self::Buffer,
        w: &Self::Buffer,
        eps: f32,
        out: &mut Self::Buffer,
        tokens: usize,
        dim: usize,
    ) {
        ctx.calls.fetch_add(1, Ordering::SeqCst);
        CpuBackend::fused_add_rms_norm(&mut ctx.cpu, residual, x, w, eps, out, tokens, dim)
    }
    fn flash_attention(
        ctx: &mut Self::Context,
        q: &Self::Buffer,
        k: &Self::Buffer,
        v: &Self::Buffer,
        out: &mut Self::Buffer,
        batch: usize,
        q_len: usize,
        kv_len: usize,
        pos_offset: usize,
        cfg: &AttnConfig,
    ) {
        ctx.calls.fetch_add(1, Ordering::SeqCst);
        CpuBackend::flash_attention(
            &mut ctx.cpu,
            q,
            k,
            v,
            out,
            batch,
            q_len,
            kv_len,
            pos_offset,
            cfg,
        )
    }
    fn copy_slice(
        ctx: &mut Self::Context,
        src: &Self::Buffer,
        src_offset: usize,
        dst: &mut Self::Buffer,
        dst_offset: usize,
        len: usize,
    ) {
        ctx.calls.fetch_add(1, Ordering::SeqCst);
        CpuBackend::copy_slice(&mut ctx.cpu, src, src_offset, dst, dst_offset, len)
    }
    fn embedding_lookup(
        ctx: &mut Self::Context,
        table: &Self::Buffer,
        ids: &[u32],
        out: &mut Self::Buffer,
        dim: usize,
    ) {
        ctx.calls.fetch_add(1, Ordering::SeqCst);
        CpuBackend::embedding_lookup(&mut ctx.cpu, table, ids, out, dim)
    }
    fn split_qkv(
        ctx: &mut Self::Context,
        qkv: &Self::Buffer,
        q: &mut Self::Buffer,
        k: &mut Self::Buffer,
        v: &mut Self::Buffer,
        tokens: usize,
        q_dim: usize,
        kv_dim: usize,
    ) {
        ctx.calls.fetch_add(1, Ordering::SeqCst);
        CpuBackend::split_qkv(&mut ctx.cpu, qkv, q, k, v, tokens, q_dim, kv_dim)
    }
    fn fused_silu_mul_split(
        ctx: &mut Self::Context,
        gate_up: &Self::Buffer,
        out: &mut Self::Buffer,
        tokens: usize,
        im: usize,
    ) {
        ctx.calls.fetch_add(1, Ordering::SeqCst);
        CpuBackend::fused_silu_mul_split(&mut ctx.cpu, gate_up, out, tokens, im)
    }
    fn scale_inplace(ctx: &mut Self::Context, buf: &mut Self::Buffer, scale: f32, len: usize) {
        ctx.calls.fetch_add(1, Ordering::SeqCst);
        CpuBackend::scale_inplace(&mut ctx.cpu, buf, scale, len)
    }
    fn scaled_add_inplace(
        ctx: &mut Self::Context,
        dst: &mut Self::Buffer,
        src: &Self::Buffer,
        scale: f32,
        len: usize,
    ) {
        ctx.calls.fetch_add(1, Ordering::SeqCst);
        CpuBackend::scaled_add_inplace(&mut ctx.cpu, dst, src, scale, len)
    }
    fn qk_norm_rope(
        ctx: &mut Self::Context,
        input: &Self::Buffer,
        norm_w: &Self::Buffer,
        cos: &Self::Buffer,
        sin: &Self::Buffer,
        output: &mut Self::Buffer,
        tokens: usize,
        heads: usize,
        head_dim: usize,
        pos_offset: usize,
        eps: f32,
        mode: i32,
    ) {
        ctx.calls.fetch_add(1, Ordering::SeqCst);
        CpuBackend::qk_norm_rope(
            &mut ctx.cpu,
            input,
            norm_w,
            cos,
            sin,
            output,
            tokens,
            heads,
            head_dim,
            pos_offset,
            eps,
            mode,
        )
    }
    fn qk_norm_rope_partial(
        ctx: &mut Self::Context,
        input: &Self::Buffer,
        norm_w: &Self::Buffer,
        cos: &Self::Buffer,
        sin: &Self::Buffer,
        output: &mut Self::Buffer,
        tokens: usize,
        heads: usize,
        head_dim: usize,
        rope_dim: usize,
        input_stride: usize,
        input_offset: usize,
        input_head_stride: usize,
        pos_offset: usize,
        eps: f32,
        mode: i32,
    ) -> Result<()> {
        ctx.calls.fetch_add(1, Ordering::SeqCst);
        CpuBackend::qk_norm_rope_partial(
            &mut ctx.cpu,
            input,
            norm_w,
            cos,
            sin,
            output,
            tokens,
            heads,
            head_dim,
            rope_dim,
            input_stride,
            input_offset,
            input_head_stride,
            pos_offset,
            eps,
            mode,
        )
    }
    fn split_qkv_norm_rope(
        ctx: &mut Self::Context,
        _qkv: &Self::Buffer,
        _q_norm_w: &Self::Buffer,
        _k_norm_w: &Self::Buffer,
        _cos: &Self::Buffer,
        _sin: &Self::Buffer,
        _q_out: &mut Self::Buffer,
        _k_out: &mut Self::Buffer,
        _v_out: &mut Self::Buffer,
        _tokens: usize,
        _q_heads: usize,
        _kv_heads: usize,
        _head_dim: usize,
        _pos_offset: usize,
        _eps: f32,
        _qk_mode: i32,
    ) -> Result<()> {
        CpuBackend::split_qkv_norm_rope(
            &mut ctx.cpu,
            _qkv,
            _q_norm_w,
            _k_norm_w,
            _cos,
            _sin,
            _q_out,
            _k_out,
            _v_out,
            _tokens,
            _q_heads,
            _kv_heads,
            _head_dim,
            _pos_offset,
            _eps,
            _qk_mode,
        )
    }
    fn split_qkv_norm_rope_into_cache(
        ctx: &mut Self::Context,
        _qkv: &Self::Buffer,
        _q_norm_w: &Self::Buffer,
        _k_norm_w: &Self::Buffer,
        _cos: &Self::Buffer,
        _sin: &Self::Buffer,
        _q_out: &mut Self::Buffer,
        _cache_k: &mut Self::Buffer,
        _cache_v: &mut Self::Buffer,
        _tokens: usize,
        _q_heads: usize,
        _kv_heads: usize,
        _head_dim: usize,
        _pos_offset: usize,
        _eps: f32,
        _qk_mode: i32,
        _cache_len: usize,
        _cache_capacity: usize,
    ) -> Result<()> {
        CpuBackend::split_qkv_norm_rope_into_cache(
            &mut ctx.cpu,
            _qkv,
            _q_norm_w,
            _k_norm_w,
            _cos,
            _sin,
            _q_out,
            _cache_k,
            _cache_v,
            _tokens,
            _q_heads,
            _kv_heads,
            _head_dim,
            _pos_offset,
            _eps,
            _qk_mode,
            _cache_len,
            _cache_capacity,
        )
    }
    fn kv_cache_append_head_major(
        ctx: &mut Self::Context,
        cache_k: &mut Self::Buffer,
        cache_v: &mut Self::Buffer,
        cache_len: usize,
        cache_capacity: usize,
        new_k_head_major: &Self::Buffer,
        new_v_head_major: &Self::Buffer,
        new_tokens: usize,
        nkv: usize,
        hd: usize,
    ) {
        ctx.calls.fetch_add(1, Ordering::SeqCst);
        CpuBackend::kv_cache_append_head_major(
            &mut ctx.cpu,
            cache_k,
            cache_v,
            cache_len,
            cache_capacity,
            new_k_head_major,
            new_v_head_major,
            new_tokens,
            nkv,
            hd,
        )
    }
    fn transpose_head_to_token(
        ctx: &mut Self::Context,
        src: &Self::Buffer,
        dst: &mut Self::Buffer,
        tokens: usize,
        heads: usize,
        dim: usize,
    ) {
        ctx.calls.fetch_add(1, Ordering::SeqCst);
        CpuBackend::transpose_head_to_token(&mut ctx.cpu, src, dst, tokens, heads, dim)
    }
    fn transpose_token_to_head(
        ctx: &mut Self::Context,
        _src: &Self::Buffer,
        _dst: &mut Self::Buffer,
        _tokens: usize,
        _heads: usize,
        _dim: usize,
    ) {
        CpuBackend::transpose_token_to_head(&mut ctx.cpu, _src, _dst, _tokens, _heads, _dim)
    }
    fn add_inplace(
        ctx: &mut Self::Context,
        residual: &mut Self::Buffer,
        x: &Self::Buffer,
        len: usize,
    ) {
        ctx.calls.fetch_add(1, Ordering::SeqCst);
        CpuBackend::add_inplace(&mut ctx.cpu, residual, x, len)
    }
    fn add_bias(
        ctx: &mut Self::Context,
        data: &mut Self::Buffer,
        bias: &Self::Buffer,
        rows: usize,
        cols: usize,
    ) {
        ctx.calls.fetch_add(1, Ordering::SeqCst);
        CpuBackend::add_bias(&mut ctx.cpu, data, bias, rows, cols)
    }
    fn layer_norm(
        ctx: &mut Self::Context,
        x: &Self::Buffer,
        gamma: &Self::Buffer,
        beta: &Self::Buffer,
        eps: f32,
        out: &mut Self::Buffer,
        tokens: usize,
        dim: usize,
    ) {
        ctx.calls.fetch_add(1, Ordering::SeqCst);
        CpuBackend::layer_norm(&mut ctx.cpu, x, gamma, beta, eps, out, tokens, dim)
    }
    fn gelu(ctx: &mut Self::Context, x: &Self::Buffer, out: &mut Self::Buffer, len: usize) {
        ctx.calls.fetch_add(1, Ordering::SeqCst);
        CpuBackend::gelu(&mut ctx.cpu, x, out, len)
    }
    fn write_f32_to_activation(ctx: &mut Self::Context, dst: &mut Self::Buffer, data: &[f32]) {
        ctx.calls.fetch_add(1, Ordering::SeqCst);
        CpuBackend::write_f32_to_activation(&mut ctx.cpu, dst, data)
    }
}

fn config() -> LlamaFamilyConfig {
    LlamaFamilyConfig {
        hidden_size: 16,
        intermediate_size: 32,
        num_heads: 2,
        num_kv_heads: 2,
        head_dim: 8,
        num_layers: 2,
        vocab_size: 32,
        max_seq_len: 32,
        ..Default::default()
    }
}

fn tracked_model() -> LlamaFamilyModel<TrackedCpu, KvFp16> {
    let cfg = config();
    let loader = SyntheticLlamaLoader::new(cfg.clone());
    LlamaFamilyModel::new(cfg, &loader).unwrap()
}

#[test]
fn instance_context_reuses_original_owner_after_growth_and_peer_close() {
    let loader = SyntheticLlamaLoader::new(config());
    let mut first_execution = ferrum_types::LlamaExecutionConfig::default();
    first_execution.backend_context.cuda.graph_skip_upload = true;
    first_execution.backend_context.moe.moe_block_size = Some(8);
    first_execution.backend_context.moe.moe_streams = 2;
    let mut second_execution = ferrum_types::LlamaExecutionConfig::default();
    second_execution.backend_context.cuda.graph_skip_sync = true;
    second_execution.backend_context.moe.moe_block_size = Some(64);
    second_execution.backend_context.moe.moe_streams = 6;
    let first_config = first_execution.backend_context.clone();
    let second_config = second_execution.backend_context.clone();
    let mut first = LlamaFamilyModel::<TrackedCpu>::new_with_execution_config(
        config(),
        &loader,
        first_execution,
    )
    .unwrap();
    let mut second = LlamaFamilyModel::<TrackedCpu>::new_with_execution_config(
        config(),
        &loader,
        second_execution,
    )
    .unwrap();
    assert_eq!(
        TrackedCpu::context_config(first.execution_context.as_ref().unwrap()),
        &first_config
    );
    assert_eq!(
        TrackedCpu::context_config(second.execution_context.as_ref().unwrap()),
        &second_config
    );
    let mut reference = LlamaFamilyModel::<CpuBackend>::new(config(), &loader).unwrap();
    let owner = first.execution_context.as_ref().unwrap().identity.clone();
    let calls = first.execution_context.as_ref().unwrap().calls.clone();
    let begins = first.execution_context.as_ref().unwrap().begins.clone();
    let retired = first.execution_context.as_ref().unwrap().retired.clone();
    assert!(!Arc::ptr_eq(
        &owner,
        &second.execution_context.as_ref().unwrap().identity
    ));
    first.prepare_kv_capacity("first", 1);
    assert_eq!(first.kv_caches["first"][0].capacity, 1);
    let original_calls = calls.load(Ordering::SeqCst);
    let prefill = first.prefill_internal("first", &[1, 2, 3]);
    assert_eq!(prefill, reference.prefill_internal("reference", &[1, 2, 3]));
    assert!(first.kv_caches["first"][0].capacity >= 3);
    assert!(first.scratch.max_tokens >= 3);
    assert!(calls.load(Ordering::SeqCst) > original_calls);
    assert!(Arc::ptr_eq(
        &owner,
        &first.execution_context.as_ref().unwrap().identity
    ));
    assert_eq!(second.prefill_internal("second", &[1, 2, 3]), prefill);
    assert_eq!(
        TrackedCpu::context_config(first.execution_context.as_ref().unwrap()),
        &first_config
    );
    assert_eq!(
        TrackedCpu::context_config(second.execution_context.as_ref().unwrap()),
        &second_config
    );
    second.close().unwrap();
    drop(second);
    assert_eq!(
        first.decode_internal("first", 4, 3),
        reference.decode_internal("reference", 4, 3)
    );
    first.truncate_kv("first", 3);
    reference.truncate_kv("reference", 3);
    assert_eq!(
        first.decode_internal("first", 5, 3),
        reference.decode_internal("reference", 5, 3)
    );
    first.release("first");
    first.reset();
    assert!(Arc::ptr_eq(
        &owner,
        &first.execution_context.as_ref().unwrap().identity
    ));
    assert_eq!(
        TrackedCpu::context_config(first.execution_context.as_ref().unwrap()),
        &first_config
    );
    assert!(begins.load(Ordering::SeqCst) > 1);
    assert_eq!(retired.load(Ordering::SeqCst), 0);
    first.close().unwrap();
    assert_eq!(retired.load(Ordering::SeqCst), 1);
    assert!(first.execution_context.is_none());
    assert!(matches!(
        first.run_device_frame(|_, _| ()),
        DeviceFrameResult::Rejected(_)
    ));
    drop(first);
    assert_eq!(retired.load(Ordering::SeqCst), 1);
}

#[test]
fn lora_instance_workspace_second_allocation_failure_retains_original_owner() {
    use crate::lora::workspace_tests::buffers;
    use crate::models::llama_family::lora_tests::adapter_fixture;

    let cfg = config();
    let (_directory, adapter) = adapter_fixture(cfg.hidden_size);
    let mut model = tracked_model();
    let mut reference =
        LlamaFamilyModel::<CpuBackend>::new(cfg.clone(), &SyntheticLlamaLoader::new(cfg.clone()))
            .unwrap();
    model
        .set_lora_adapter_for_cache("adapted", Some(adapter.clone()))
        .unwrap();
    reference
        .set_lora_adapter_for_cache("adapted", Some(adapter))
        .unwrap();
    // Pregrow unrelated model scratch before arming the fault. The next
    // two allocations in a real prefill are exactly the LoRA pair's growth.
    model.ensure_scratch(2);
    assert_eq!(
        model.prefill_internal("adapted", &[1]),
        reference.prefill_internal("adapted", &[1]),
    );
    assert_eq!(
        model.lora_workspace.capacities_for_test(),
        (1, cfg.hidden_size)
    );
    let (rank, delta) = buffers(&model.lora_workspace);
    let old_rank = rank.unwrap().as_ptr();
    let old_delta = delta.unwrap().as_ptr();
    let old_delta_values = delta.unwrap().clone();
    let cache = model.kv_caches["adapted"][0].k.as_ptr();
    let original_scratch = model.scratch.residual.as_ref().unwrap().as_ptr();
    let context = model.execution_context.as_ref().unwrap();
    let owner = Arc::clone(&context.identity);
    let unresolved = Arc::clone(&context.unresolved);
    let calls = Arc::clone(&context.calls);
    let begins = Arc::clone(&context.begins);
    let retired = Arc::clone(&context.retired);

    // If an assertion fails while the controlled observation is unresolved,
    // reset only the test fault before the model's real checked Drop runs.
    struct ClearFault(Arc<AtomicBool>);
    impl Drop for ClearFault {
        fn drop(&mut self) {
            self.0.store(false, Ordering::SeqCst);
        }
    }
    let _clear_fault = ClearFault(Arc::clone(&unresolved));
    let calls_before = calls.load(Ordering::SeqCst);
    let begins_before = begins.load(Ordering::SeqCst);
    model
        .execution_context
        .as_mut()
        .unwrap()
        .successful_allocations_before_failure = Some(1);
    let result =
        model.run_device_frame(|model, frame| model.prefill_in_frame(frame, "adapted", &[2, 3]));
    let DeviceFrameResult::Held(error) = result else {
        panic!("the failed second allocation must retain the original frame");
    };
    assert!(matches!(error, FerrumError::Backend { .. }));
    assert!(error
        .to_string()
        .contains("controlled second LoRA workspace allocation failure"));
    assert_eq!(calls.load(Ordering::SeqCst), calls_before + 2);
    assert_eq!(begins.load(Ordering::SeqCst), begins_before + 1);
    assert!(unresolved.load(Ordering::SeqCst));
    assert!(model.execution_context.is_none());
    assert!(Arc::ptr_eq(
        &owner,
        &model
            .pending_device_frame
            .as_ref()
            .unwrap()
            .context
            .identity
    ));
    assert_eq!(
        model.lora_workspace.capacities_for_test(),
        (2, cfg.hidden_size)
    );
    let (rank, delta) = buffers(&model.lora_workspace);
    let retained_rank = rank.unwrap().as_ptr();
    assert_ne!(retained_rank, old_rank);
    assert_eq!(rank.unwrap().len(), 2);
    assert_eq!(delta.unwrap().as_ptr(), old_delta);
    assert_eq!(delta.unwrap(), &old_delta_values);
    assert_eq!(model.cache_len("adapted"), 1);
    assert_eq!(model.kv_caches["adapted"][0].k.as_ptr(), cache);
    assert_eq!(
        model.scratch.residual.as_ref().unwrap().as_ptr(),
        original_scratch
    );
    assert_eq!(model.lora_adapters.len(), 1);
    assert_eq!(retired.load(Ordering::SeqCst), 0);

    assert!(matches!(
        model.run_device_frame(|_, _| -> () { panic!("a Held model must reject new work") }),
        DeviceFrameResult::Rejected(_)
    ));
    assert!(model.close().is_err());
    assert!(Arc::ptr_eq(
        &owner,
        &model
            .pending_device_frame
            .as_ref()
            .unwrap()
            .context
            .identity
    ));
    assert_eq!(begins.load(Ordering::SeqCst), begins_before + 1);
    assert_eq!(calls.load(Ordering::SeqCst), calls_before + 2);
    assert_eq!(retired.load(Ordering::SeqCst), 0);
    assert_eq!(
        buffers(&model.lora_workspace).0.unwrap().as_ptr(),
        retained_rank
    );
    assert_eq!(
        buffers(&model.lora_workspace).1.unwrap().as_ptr(),
        old_delta
    );

    unresolved.store(false, Ordering::SeqCst);
    assert!(model.recover_device_frame().unwrap());
    assert!(model.pending_device_frame.is_none());
    assert!(Arc::ptr_eq(
        &owner,
        &model.execution_context.as_ref().unwrap().identity
    ));
    assert_eq!(
        buffers(&model.lora_workspace).0.unwrap().as_ptr(),
        retained_rank
    );
    assert_eq!(
        buffers(&model.lora_workspace).1.unwrap().as_ptr(),
        old_delta
    );
    // The interrupted preparation ran no token math; completing the same
    // original cache now must match an independent complete CPU execution.
    let logits = model.prefill_internal("adapted", &[2, 3]);
    let expected = reference.prefill_internal("adapted", &[2, 3]);
    assert_eq!(logits.len(), cfg.vocab_size);
    assert!(logits.iter().all(|value| value.is_finite()));
    assert_eq!(logits, expected);
    assert_eq!(model.cache_len("adapted"), 3);
    assert_eq!(
        model.lora_workspace.capacities_for_test(),
        (2, 2 * cfg.hidden_size)
    );
    assert_eq!(
        buffers(&model.lora_workspace).0.unwrap().as_ptr(),
        retained_rank
    );
    assert_ne!(
        buffers(&model.lora_workspace).1.unwrap().as_ptr(),
        old_delta
    );
    let logits = model.decode_internal("adapted", 4, 3);
    assert_eq!(logits.len(), cfg.vocab_size);
    assert_eq!(logits, reference.decode_internal("adapted", 4, 3));
    assert!(Arc::ptr_eq(
        &owner,
        &model.execution_context.as_ref().unwrap().identity
    ));
    model.close().unwrap();
    reference.close().unwrap();
    assert_eq!(retired.load(Ordering::SeqCst), 1);
}

#[test]
fn instance_context_storage_retirement_failure_preserves_original_allocations() {
    let mut model = tracked_model();
    model.prepare_kv_capacity("original", 1);
    let blocked = model
        .execution_context
        .as_ref()
        .unwrap()
        .storage_retirement_blocked
        .clone();
    let scratch = model.scratch.residual.as_ref().unwrap().as_ptr();
    let logits = model.scratch.batch_logits.as_ptr();
    let cache = model.kv_caches["original"][0].k.as_ptr();
    let capacity = model.scratch.max_tokens;
    blocked.store(true, Ordering::SeqCst);

    let scratch_growth = model.run_preparation_frame(|model, frame| {
        model.ensure_scratch_in_context(&mut frame.context, capacity + 1);
    });
    assert!(matches!(
        scratch_growth,
        DeviceFrameResult::Terminal(Err(_))
    ));
    assert_eq!(model.scratch.residual.as_ref().unwrap().as_ptr(), scratch);
    assert_eq!(model.scratch.max_tokens, capacity);

    let logits_growth = model.run_preparation_frame(|model, frame| {
        model
            .scratch
            .ensure_batch_logits(&mut frame.context, &model.cfg, 2);
    });
    assert!(matches!(logits_growth, DeviceFrameResult::Terminal(Err(_))));
    assert_eq!(model.scratch.batch_logits.as_ptr(), logits);

    let kv_growth = model.run_preparation_frame(|model, frame| {
        model.ensure_paged_kv_capacity_for_cache_id(&mut frame.context, "original", 2)
    });
    assert!(matches!(kv_growth, DeviceFrameResult::Terminal(Ok(Err(_)))));
    assert_eq!(model.kv_caches["original"][0].k.as_ptr(), cache);
    assert_eq!(model.kv_caches["original"][0].capacity, 1);

    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| model.reset())).is_err());
    assert_eq!(model.kv_caches["original"][0].k.as_ptr(), cache);
    model.release("original");
    assert_eq!(model.kv_free_pool.len(), 1);
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        model.prepare_kv_capacity("replacement", 2);
    }))
    .is_err());
    assert!(!model.kv_caches.contains_key("replacement"));
    assert_eq!(model.kv_free_pool.len(), 1);
    assert_eq!(model.kv_free_pool[0][0].k.as_ptr(), cache);

    blocked.store(false, Ordering::SeqCst);
    let loader = SyntheticLlamaLoader::new(config());
    let mut reference = LlamaFamilyModel::<CpuBackend>::new(config(), &loader).unwrap();
    assert_eq!(
        model.prefill_internal("replacement", &[1, 2, 3]),
        reference.prefill_internal("reference", &[1, 2, 3])
    );
    model.reset();
    assert!(model.kv_caches.is_empty());
    assert!(model.kv_free_pool.is_empty());
    model.close().unwrap();
}

#[test]
fn instance_context_unresolved_retains_original_owner_until_recovery() {
    let mut model = tracked_model();
    let owner = model.execution_context.as_ref().unwrap().identity.clone();
    let unresolved = model.execution_context.as_ref().unwrap().unresolved.clone();
    let retired = model.execution_context.as_ref().unwrap().retired.clone();
    let original = model.scratch.residual.as_ref().unwrap().as_ptr();
    let result: DeviceFrameResult<()> = model.run_device_frame(|model, frame| {
        frame
            .take_scratch(&mut model.scratch, FrameLayout::Single, false)
            .unwrap();
        let residual = frame.residual.as_mut().unwrap();
        TrackedCpu::write_f32_to_activation(&mut frame.context, residual, &[3.0; 16]);
        TrackedCpu::scale_inplace(&mut frame.context, residual, 2.0, 16);
        frame.context.unresolved.store(true, Ordering::SeqCst);
        std::panic::panic_any(FerrumError::backend("controlled encoded failure"));
    });
    assert!(matches!(result, DeviceFrameResult::Held(_)));
    assert!(model.execution_context.is_none());
    let frame = model.pending_device_frame.as_ref().unwrap();
    assert!(Arc::ptr_eq(&owner, &frame.context.identity));
    assert_eq!(frame.residual.as_ref().unwrap().as_ptr(), original);
    assert_eq!(frame.residual.as_ref().unwrap(), &[6.0; 16]);
    assert!(model.close().is_err());
    assert_eq!(retired.load(Ordering::SeqCst), 0);
    assert!(matches!(
        model.run_preparation_frame(|_, _| ()),
        DeviceFrameResult::Rejected(_)
    ));
    let mut peer = tracked_model();
    assert!(peer
        .prefill_internal("peer", &[1, 2])
        .iter()
        .all(|v| v.is_finite()));
    peer.close().unwrap();
    unresolved.store(false, Ordering::SeqCst);
    assert!(model.recover_device_frame().unwrap());
    assert!(model.pending_device_frame.is_none());
    assert!(Arc::ptr_eq(
        &owner,
        &model.execution_context.as_ref().unwrap().identity
    ));
    assert_eq!(model.scratch.residual.as_ref().unwrap().as_ptr(), original);
    model.close().unwrap();
    assert_eq!(retired.load(Ordering::SeqCst), 1);
}

#[test]
fn instance_context_whole_retirement_preserves_failed_cleanup_owner_for_retry() {
    use crate::common::ModelRetirementOutcome;
    let mut model = tracked_model();
    let loader = SyntheticLlamaLoader::new(config());
    let mut reference = LlamaFamilyModel::<CpuBackend>::new(config(), &loader).unwrap();
    let actual = model.prefill_internal("actual", &[1, 2, 3]);
    let expected = reference.prefill_internal("reference", &[1, 2, 3]);
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.into_iter().zip(expected) {
        assert!(actual.is_finite() && expected.is_finite());
        assert_eq!(actual, expected);
    }
    let context = model.execution_context.as_ref().unwrap();
    let original = Arc::downgrade(&context.identity);
    let retired = Arc::clone(&context.retired);
    let blocked = Arc::clone(&context.retirement_blocked);
    struct Restore(Arc<AtomicBool>);
    impl Drop for Restore {
        fn drop(&mut self) {
            self.0.store(false, Ordering::SeqCst);
        }
    }
    let _restore = Restore(Arc::clone(&blocked));
    blocked.store(true, Ordering::SeqCst);
    let scratch = model.scratch.residual.as_ref().unwrap().as_ptr();
    let ModelRetirementOutcome::Retained { error } = model.retire_model() else {
        panic!("failed cleanup must retain original context")
    };
    assert!(error
        .to_string()
        .contains("controlled original cleanup failure"));
    assert!(original.upgrade().is_some());
    assert_eq!(retired.load(Ordering::SeqCst), 0);
    assert_eq!(model.scratch.residual.as_ref().unwrap().as_ptr(), scratch);
    blocked.store(false, Ordering::SeqCst);
    assert!(matches!(
        model.retire_model(),
        ModelRetirementOutcome::Retired { error: None }
    ));
    assert!(model.execution_context.is_none());
    assert!(original.upgrade().is_none());
    assert_eq!(retired.load(Ordering::SeqCst), 1);
    reference.close().unwrap();
}

#[test]
fn instance_context_whole_retirement_reports_terminal_error_with_released_owner() {
    use crate::common::ModelRetirementOutcome;
    let mut model = tracked_model();
    let context = model.execution_context.as_ref().unwrap();
    let original = Arc::downgrade(&context.identity);
    let unresolved = Arc::clone(&context.unresolved);
    let terminal_error = Arc::clone(&context.terminal_error);
    let retired = Arc::clone(&context.retired);
    struct Restore(Arc<AtomicBool>);
    impl Drop for Restore {
        fn drop(&mut self) {
            self.0.store(false, Ordering::SeqCst);
        }
    }
    let _restore = Restore(Arc::clone(&unresolved));
    let result: DeviceFrameResult<()> = model.run_device_frame(|model, frame| {
        frame
            .take_scratch(&mut model.scratch, FrameLayout::Single, false)
            .unwrap();
        let residual = frame.residual.as_mut().unwrap();
        TrackedCpu::write_f32_to_activation(&mut frame.context, residual, &[3.0; 16]);
        TrackedCpu::scale_inplace(&mut frame.context, residual, 2.0, 16);
        frame.context.unresolved.store(true, Ordering::SeqCst);
        std::panic::panic_any(FerrumError::backend("controlled original encoded failure"));
    });
    assert!(matches!(result, DeviceFrameResult::Held(_)));
    assert_eq!(
        model
            .pending_device_frame
            .as_ref()
            .unwrap()
            .residual
            .as_ref()
            .unwrap(),
        &[6.0; 16]
    );
    unresolved.store(false, Ordering::SeqCst);
    terminal_error.store(true, Ordering::SeqCst);
    let ModelRetirementOutcome::Retired { error: Some(error) } = model.retire_model() else {
        panic!("terminal execution error must remain distinct from cleanup custody")
    };
    assert!(error
        .to_string()
        .contains("controlled original terminal failure"));
    assert!(model.execution_context.is_none());
    assert!(model.pending_device_frame.is_none());
    assert!(original.upgrade().is_none());
    assert_eq!(retired.load(Ordering::SeqCst), 1);
    assert_eq!(model.scratch.residual.as_ref().unwrap(), &[6.0; 16]);
}

#[test]
fn instance_context_constructor_error_retires_original_before_return() {
    use ferrum_quantization::{QuantConfig, WeightLoader};
    type Captured = (std::sync::Weak<()>, Arc<AtomicUsize>);
    struct FailingLoader(std::sync::Mutex<Option<Captured>>);
    impl WeightLoader<TrackedCpu> for FailingLoader {
        fn load_tensor(&self, ctx: &mut Context, _: &str) -> Result<Vec<f32>> {
            assert!(
                ctx.calls.load(Ordering::SeqCst) > 0,
                "real rope and scratch already allocated"
            );
            *self.0.lock().unwrap() = Some((Arc::downgrade(&ctx.identity), ctx.retired.clone()));
            Err(FerrumError::model("controlled first weight failure"))
        }
        fn load_linear(
            &self,
            _: &mut Context,
            _: &str,
        ) -> Result<ferrum_kernels::LoadedLinear<TrackedCpu>> {
            unreachable!()
        }
        fn has_tensor(&self, _: &str) -> bool {
            false
        }
        fn quant_config(&self) -> Option<&QuantConfig> {
            None
        }
    }
    let loader = FailingLoader(std::sync::Mutex::new(None));
    let result = LlamaFamilyModel::<TrackedCpu>::new(config(), &loader);
    assert!(matches!(result, Err(FerrumError::Model { .. })));
    let (owner, retired) = loader.0.lock().unwrap().take().unwrap();
    assert!(owner.upgrade().is_none());
    assert_eq!(retired.load(Ordering::SeqCst), 1);
}

#[test]
fn instance_context_unresolved_bare_drop_fails_stop() {
    const CHILD: &str = "FERRUM_TEST_UNRESOLVED_MODEL_DROP_CHILD";
    if std::env::var_os(CHILD).is_some() {
        let mut model = tracked_model();
        assert!(matches!(
            model.run_device_frame(|_, frame| {
                frame.context.unresolved.store(true, Ordering::SeqCst);
            }),
            DeviceFrameResult::Held(_)
        ));
        drop(model);
        panic!("unresolved owner was released");
    }
    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .arg("--exact")
        .arg("models::llama_family::device_execution::instance_context_tests::instance_context_unresolved_bare_drop_fails_stop")
        .arg("--nocapture")
        .env(CHILD, "1")
        .output().unwrap();
    assert!(!output.status.success());
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt;
        assert_eq!(output.status.signal(), Some(6));
    }
    assert!(String::from_utf8_lossy(&output.stderr)
        .contains("cannot drop model with unretired original device context"));
}

#[test]
fn instance_context_single_graph_close_failure_retains_exact_cache_until_retry() {
    use crate::common::llm::ModelTokenCloseOutcome;
    let mut model = tracked_model();
    model.prefill_internal("captured", &[1, 2, 3]);
    // Fault injection exercises the retirement boundary, not a simulated CUDA
    // graph result. Actual capture/replay mathematics has separate GPU tests.
    model.single_graph_cache_id = Some("captured".to_owned());
    let blocked = model
        .execution_context
        .as_ref()
        .unwrap()
        .storage_retirement_blocked
        .clone();
    let original = model.kv_caches["captured"][0].k.as_ptr();
    blocked.store(true, Ordering::SeqCst);
    assert!(matches!(
        model.close_managed_token_cache("captured"),
        ModelTokenCloseOutcome::Held { .. }
    ));
    assert_eq!(model.kv_caches["captured"][0].k.as_ptr(), original);
    assert_eq!(model.cache_len("captured"), 3);
    assert!(model.kv_free_pool.is_empty());
    assert_eq!(model.single_graph_cache_id.as_deref(), Some("captured"));
    assert!(model.execution_context.is_some());
    blocked.store(false, Ordering::SeqCst);
    assert!(matches!(
        model.close_managed_token_cache("captured"),
        ModelTokenCloseOutcome::Ended { error: None }
    ));
    assert!(model.single_graph_cache_id.is_none());
    assert!(!model.kv_caches.contains_key("captured"));
    assert_eq!(model.kv_free_pool[0][0].k.as_ptr(), original);
    let loader = SyntheticLlamaLoader::new(config());
    let mut reference = LlamaFamilyModel::<CpuBackend>::new(config(), &loader).unwrap();
    let actual = model.prefill_internal("captured", &[5, 6, 7]);
    let expected = reference.prefill_internal("reference", &[5, 6, 7]);
    assert!(actual.iter().all(|value| value.is_finite()));
    assert_eq!(actual, expected);
    model.close().unwrap();
    reference.close().unwrap();
}

#[test]
fn instance_context_logits_growth_failure_keeps_single_graph_owner_until_retirement() {
    use crate::common::llm::ModelTokenCloseOutcome;
    let mut model = tracked_model();
    model.prefill_internal("captured", &[1, 2, 3]);
    assert!(model.scratch.max_tokens >= 2);
    assert_eq!(model.scratch.batch_logits_capacity, 1);
    // Inject only the graph-retirement boundary. The actual CUDA graph and
    // full-logit equivalence are exercised by the separate GPU regressions.
    model.single_graph_cache_id = Some("captured".to_owned());
    let context = model.execution_context.as_ref().unwrap();
    let owner = context.identity.clone();
    let blocked = context.storage_retirement_blocked.clone();
    let original_kv: Vec<_> = model.kv_caches["captured"]
        .iter()
        .map(|cache| (cache.k.as_ptr(), cache.v.as_ptr(), cache.capacity))
        .collect();
    let original_logits = model.scratch.batch_logits.as_ptr();
    blocked.store(true, Ordering::SeqCst);
    let failed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        model.forward_verify("captured", &[4, 5]);
    }));
    let error = failed.expect_err("logits growth requires graph retirement");
    assert!(panic_error(error)
        .to_string()
        .contains("controlled graph retirement failure"));
    assert_eq!(model.single_graph_cache_id.as_deref(), Some("captured"));
    assert_eq!(model.scratch.batch_logits.as_ptr(), original_logits);
    assert_eq!(model.scratch.batch_logits_capacity, 1);
    assert_eq!(model.cache_len("captured"), 3);
    assert!(Arc::ptr_eq(
        &owner,
        &model.execution_context.as_ref().unwrap().identity
    ));
    assert!(matches!(
        model.close_managed_token_cache("captured"),
        ModelTokenCloseOutcome::Held { .. }
    ));
    assert!(model.kv_free_pool.is_empty());
    assert_eq!(
        model.kv_caches["captured"]
            .iter()
            .map(|cache| (cache.k.as_ptr(), cache.v.as_ptr(), cache.capacity))
            .collect::<Vec<_>>(),
        original_kv
    );
    blocked.store(false, Ordering::SeqCst);
    let loader = SyntheticLlamaLoader::new(config());
    let mut reference = LlamaFamilyModel::<CpuBackend>::new(config(), &loader).unwrap();
    reference.prefill_internal("reference", &[1, 2, 3]);
    let actual = model.forward_verify("captured", &[4, 5]);
    let expected = reference.forward_verify("reference", &[4, 5]);
    assert_eq!(actual.len(), 2 * config().vocab_size);
    assert!(actual
        .iter()
        .chain(&expected)
        .all(|value| value.is_finite()));
    assert_eq!(actual, expected);
    assert_eq!(model.cache_len("captured"), 5);
    assert!(model.single_graph_cache_id.is_none());
    assert!(Arc::ptr_eq(
        &owner,
        &model.execution_context.as_ref().unwrap().identity
    ));
    assert!(matches!(
        model.close_managed_token_cache("captured"),
        ModelTokenCloseOutcome::Ended { error: None }
    ));
    assert_eq!(model.kv_free_pool[0][0].k.as_ptr(), original_kv[0].0);
    reference.release("reference");
    let actual = model.prefill_internal("captured", &[6, 7, 8]);
    let expected = reference.prefill_internal("reference", &[6, 7, 8]);
    assert_eq!(actual.len(), config().vocab_size);
    assert!(actual
        .iter()
        .chain(&expected)
        .all(|value| value.is_finite()));
    assert_eq!(actual, expected);
    model.close().unwrap();
    reference.close().unwrap();
}

fn prepared_graph_fault_models() -> (
    LlamaFamilyModel<TrackedCpu, KvFp16>,
    LlamaFamilyModel<CpuBackend, KvFp16>,
) {
    let loader = SyntheticLlamaLoader::new(config());
    let mut execution = ferrum_types::LlamaExecutionConfig::default();
    execution.options.cuda_graph = true;
    execution.prefix_cache = false;
    execution.options.metal_paged_kv = Some(false);
    let mut model = LlamaFamilyModel::<TrackedCpu>::new_with_execution_config(
        config(),
        &loader,
        execution.clone(),
    )
    .unwrap();
    execution.options.cuda_graph = false;
    let mut reference =
        LlamaFamilyModel::<CpuBackend>::new_with_execution_config(config(), &loader, execution)
            .unwrap();
    for id in ["captured", "peer"] {
        model.prepare_kv_capacity(id, 32);
        reference.prepare_kv_capacity(id, 32);
        assert_eq!(
            model.prefill_internal(id, &[1, 2, 3]),
            reference.prefill_internal(id, &[1, 2, 3]),
        );
    }
    for position in 3..6 {
        assert_eq!(
            model.decode_internal("captured", position + 4, position),
            reference.decode_internal("captured", position + 4, position),
        );
    }
    assert_eq!(model.graph_warmup, 3);
    assert!(model.single_graph_cache_id.is_none());
    (model, reference)
}

fn assert_graph_failure_frontier(model: &LlamaFamilyModel<TrackedCpu, KvFp16>) {
    assert_eq!(model.cache_len("captured"), 6);
    assert!(model.kv_caches["captured"]
        .iter()
        .all(|cache| cache.len == 6));
    assert!(model.graph_capture_failed);
    assert!(!model.single_decode_graph_enabled());
    assert_eq!(model.single_graph_cache_id.as_deref(), Some("captured"));
}

#[test]
fn instance_context_single_capture_partial_failure_never_publishes_layer_frontiers() {
    use crate::common::llm::{ModelExecutionRecovery, ModelTokenCloseOutcome};

    for unresolved in [false, true] {
        let (mut model, mut reference) = prepared_graph_fault_models();
        let context = model.execution_context.as_mut().unwrap();
        let identity = context.identity.clone();
        let pending = context.unresolved.clone();
        context.graph_fault = Some(if unresolved {
            GraphFault::SecondLayerUnresolved
        } else {
            GraphFault::SecondLayer
        });
        let original_kv: Vec<_> = model.kv_caches["captured"]
            .iter()
            .map(|cache| (cache.k.as_ptr(), cache.v.as_ptr()))
            .collect();
        let residual = model.scratch.residual.as_ref().unwrap().as_ptr();
        let failed =
            model.run_device_frame(|model, frame| model.decode_in_frame(frame, "captured", 10, 6));
        if unresolved {
            assert!(matches!(failed, DeviceFrameResult::Held(_)));
            assert!(model.execution_context.is_none());
            let frame = model.pending_device_frame.as_ref().unwrap();
            assert!(Arc::ptr_eq(&identity, &frame.context.identity));
            assert_eq!(frame.residual.as_ref().unwrap().as_ptr(), residual);
            assert!(model.scratch.residual.is_none());
            assert!(matches!(
                model.run_device_frame(
                    |model, frame| model.decode_in_frame(frame, "captured", 10, 6)
                ),
                DeviceFrameResult::Rejected(_)
            ));
            assert!(matches!(
                model.close_managed_token_cache("captured"),
                ModelTokenCloseOutcome::Held { .. }
            ));
            assert!(model.kv_free_pool.is_empty());
            assert_graph_failure_frontier(&model);
            pending.store(false, Ordering::SeqCst);
            assert!(matches!(
                model.recover_token_frame(),
                ModelExecutionRecovery::Quiescent { error: None }
            ));
        } else {
            let DeviceFrameResult::Terminal(Err(error)) = failed else {
                panic!("capture must fail");
            };
            assert!(error
                .to_string()
                .contains("controlled second layer capture failure"));
        }
        assert_graph_failure_frontier(&model);
        assert_eq!(model.execution_context.as_ref().unwrap().capture_gemms, 5);
        assert_eq!(model.execution_context.as_ref().unwrap().replay_attempts, 0);
        assert_eq!(model.scratch.residual.as_ref().unwrap().as_ptr(), residual);
        assert_eq!(
            model.kv_caches["captured"]
                .iter()
                .map(|cache| (cache.k.as_ptr(), cache.v.as_ptr()))
                .collect::<Vec<_>>(),
            original_kv
        );
        // This is CPU numerical recovery after an injected contract failure,
        // not a claim that CPU simulated CUDA graph execution.
        let expected = reference.decode_internal("captured", 10, 6);
        let actual = model.decode_internal("captured", 10, 6);
        assert_eq!(actual.len(), config().vocab_size);
        assert!(actual
            .iter()
            .chain(&expected)
            .all(|value| value.is_finite()));
        assert_eq!(actual, expected);
        assert!(model.kv_caches["captured"]
            .iter()
            .all(|cache| cache.len == 7));
        assert!(model.single_graph_cache_id.is_none());
        model.close().unwrap();
        reference.close().unwrap();
    }
}

#[test]
fn instance_context_single_graph_first_replay_failure_keeps_custody_but_denies_replay() {
    use crate::common::llm::{ModelExecutionRecovery, ModelTokenCloseOutcome};

    for fault in [
        GraphFault::EndCapture,
        GraphFault::ReplayMissing,
        GraphFault::ReplayError,
        GraphFault::ReplayUnresolved,
    ] {
        let (mut model, mut reference) = prepared_graph_fault_models();
        let context = model.execution_context.as_mut().unwrap();
        let identity = context.identity.clone();
        let pending = context.unresolved.clone();
        let retirement_blocked = context.storage_retirement_blocked.clone();
        context.graph_fault = Some(fault);
        let original_kv: Vec<_> = model.kv_caches["captured"]
            .iter()
            .map(|cache| (cache.k.as_ptr(), cache.v.as_ptr()))
            .collect();
        let result =
            model.run_device_frame(|model, frame| model.decode_in_frame(frame, "captured", 10, 6));
        if fault == GraphFault::ReplayUnresolved {
            assert!(matches!(result, DeviceFrameResult::Held(_)));
            assert!(model.execution_context.is_none());
            assert!(Arc::ptr_eq(
                &identity,
                &model
                    .pending_device_frame
                    .as_ref()
                    .unwrap()
                    .context
                    .identity
            ));
            assert!(matches!(
                model.close_managed_token_cache("captured"),
                ModelTokenCloseOutcome::Held { .. }
            ));
            assert!(model.kv_free_pool.is_empty());
            assert_graph_failure_frontier(&model);
            pending.store(false, Ordering::SeqCst);
            assert!(matches!(
                model.recover_token_frame(),
                ModelExecutionRecovery::Quiescent { error: None }
            ));
        } else {
            assert!(matches!(result, DeviceFrameResult::Terminal(Err(_))));
        }
        assert_graph_failure_frontier(&model);
        let attempts = usize::from(fault != GraphFault::EndCapture);
        assert_eq!(
            model.execution_context.as_ref().unwrap().replay_attempts,
            attempts
        );
        retirement_blocked.store(true, Ordering::SeqCst);
        let blocked =
            model.run_device_frame(|model, frame| model.decode_in_frame(frame, "captured", 10, 6));
        let DeviceFrameResult::Terminal(Err(error)) = blocked else {
            panic!("checked retirement must block reuse");
        };
        assert!(error
            .to_string()
            .contains("controlled graph retirement failure"));
        assert_eq!(
            model.execution_context.as_ref().unwrap().replay_attempts,
            attempts
        );
        assert_graph_failure_frontier(&model);
        assert!(matches!(
            model.close_managed_token_cache("captured"),
            ModelTokenCloseOutcome::Held { .. }
        ));
        assert_eq!(
            model.kv_caches["captured"]
                .iter()
                .map(|cache| (cache.k.as_ptr(), cache.v.as_ptr()))
                .collect::<Vec<_>>(),
            original_kv
        );
        // Unrelated cache operations cannot restore failed graph eligibility.
        model.truncate_kv("peer", 2);
        assert!(model.graph_capture_failed);
        model.release("peer");
        assert_graph_failure_frontier(&model);
        retirement_blocked.store(false, Ordering::SeqCst);
        let expected = reference.decode_internal("captured", 10, 6);
        let actual = model.decode_internal("captured", 10, 6);
        assert_eq!(actual.len(), config().vocab_size);
        assert!(actual
            .iter()
            .chain(&expected)
            .all(|value| value.is_finite()));
        assert_eq!(actual, expected);
        assert_eq!(
            model.execution_context.as_ref().unwrap().replay_attempts,
            attempts
        );
        assert!(Arc::ptr_eq(
            &identity,
            &model.execution_context.as_ref().unwrap().identity
        ));
        assert!(model.kv_caches["captured"]
            .iter()
            .all(|cache| cache.len == 7));
        assert!(model.single_graph_cache_id.is_none());
        model.close().unwrap();
        reference.close().unwrap();
    }
}

#[test]
fn instance_context_batch_graph_members_retire_before_close_or_truncate() {
    use crate::common::llm::ModelTokenCloseOutcome;
    let mut model = tracked_model();
    model.prefill_internal("batch-a", &[1, 2, 3]);
    model.prefill_internal("batch-b", &[4, 5, 6]);
    // Only the checked-retirement failure is injected. The actual CPU cache,
    // original context and subsequent numerical execution remain real; CUDA
    // capture/replay is covered by the separate backend regressions.
    model.batched_pointers_for = Some(vec!["batch-a".into(), "batch-b".into()]);
    let context = model.execution_context.as_ref().unwrap();
    let owner = context.identity.clone();
    let blocked = context.storage_retirement_blocked.clone();
    let original_a = model.kv_caches["batch-a"][0].k.as_ptr();
    let original_b = model.kv_caches["batch-b"][0].k.as_ptr();
    blocked.store(true, Ordering::SeqCst);
    assert!(matches!(
        model.close_managed_token_cache("batch-a"),
        ModelTokenCloseOutcome::Held { .. }
    ));
    let failure = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        model.truncate_token_cache("batch-b", 2);
    }))
    .expect_err("a graph-bound truncation must retire the original graph");
    assert!(panic_error(failure)
        .to_string()
        .contains("controlled graph retirement failure"));
    assert_eq!(model.cache_len("batch-a"), 3);
    assert_eq!(model.cache_len("batch-b"), 3);
    assert_eq!(model.kv_caches["batch-a"][0].k.as_ptr(), original_a);
    assert_eq!(model.kv_caches["batch-b"][0].k.as_ptr(), original_b);
    assert!(model.kv_free_pool.is_empty());
    assert_eq!(
        model.batched_pointers_for.as_deref(),
        Some(["batch-a".to_owned(), "batch-b".to_owned()].as_slice())
    );
    assert!(Arc::ptr_eq(
        &owner,
        &model.execution_context.as_ref().unwrap().identity
    ));

    blocked.store(false, Ordering::SeqCst);
    model.truncate_token_cache("batch-b", 2);
    assert!(model.batched_pointers_for.is_none());
    assert_eq!(model.cache_len("batch-b"), 2);
    let loader = SyntheticLlamaLoader::new(config());
    let mut reference = LlamaFamilyModel::<CpuBackend>::new(config(), &loader).unwrap();
    reference.prefill_internal("a", &[1, 2, 3]);
    reference.prefill_internal("b", &[4, 5]);
    for (actual_id, reference_id, token, position) in
        [("batch-b", "b", 7, 2), ("batch-a", "a", 8, 3)]
    {
        let actual = model.decode_internal(actual_id, token, position);
        let expected = reference.decode_internal(reference_id, token, position);
        assert_eq!(actual.len(), config().vocab_size);
        assert!(actual
            .iter()
            .chain(&expected)
            .all(|value| value.is_finite()));
        assert_eq!(actual, expected);
    }
    assert!(matches!(
        model.close_managed_token_cache("batch-a"),
        ModelTokenCloseOutcome::Ended { error: None }
    ));
    assert_eq!(model.kv_free_pool[0][0].k.as_ptr(), original_a);
    reference.release("a");
    let actual = model.prefill_internal("batch-a", &[9, 10, 11]);
    let expected = reference.prefill_internal("a", &[9, 10, 11]);
    assert!(actual
        .iter()
        .chain(&expected)
        .all(|value| value.is_finite()));
    assert_eq!(actual, expected);
    assert!(Arc::ptr_eq(
        &owner,
        &model.execution_context.as_ref().unwrap().identity
    ));
    model.close().unwrap();
    reference.close().unwrap();
}
