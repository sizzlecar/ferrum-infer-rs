//! Original context custody, backed by real CPU numerical operations.
use super::*;
use crate::common::DecoderOnlyLLM;
use crate::models::llama_family::LlamaFamilyConfig;
use crate::test_support::SyntheticLlamaLoader;
use ferrum_interfaces::kv_dtype::KvFp16;
use ferrum_kernels::backend::cpu::CpuBackend;
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
struct Context {
    identity: Arc<()>,
    unresolved: Arc<AtomicBool>,
    calls: Arc<AtomicUsize>,
    begins: Arc<AtomicUsize>,
    retired: Arc<AtomicUsize>,
    storage_retirement_blocked: Arc<AtomicBool>,
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
    fn retire_graphs_for_storage_mutation(ctx: &mut Context) -> Result<()> {
        if ctx.storage_retirement_blocked.load(Ordering::SeqCst) {
            return Err(FerrumError::backend("controlled graph retirement failure"));
        }
        Self::observe_completion(ctx).into_result()
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
    fn new_context() -> Context {
        Context {
            identity: Arc::new(()),
            unresolved: Arc::new(AtomicBool::new(false)),
            calls: Arc::new(AtomicUsize::new(0)),
            begins: Arc::new(AtomicUsize::new(0)),
            retired: Arc::new(AtomicUsize::new(0)),
            storage_retirement_blocked: Arc::new(AtomicBool::new(false)),
        }
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
        if ctx.unresolved.load(Ordering::SeqCst) {
            BackendCompletion::Unresolved(FerrumError::backend(
                "controlled original context remains unresolved",
            ))
        } else {
            BackendCompletion::Succeeded
        }
    }
    fn retire_context(ctx: &mut Context) -> Result<()> {
        Self::observe_completion(ctx).into_result()?;
        ctx.retired.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
    fn sync(ctx: &mut Context) {
        Self::observe_completion(ctx).into_result().unwrap();
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
    fn zero_buffer(_ctx: &mut Self::Context, _buf: &mut Self::Buffer, _len: usize) -> Result<()> {
        CpuBackend::zero_buffer(&mut (), _buf, _len)
    }
    fn write_typed<T: HostDtype>(ctx: &mut Self::Context, dst: &mut Self::Buffer, data: &[T]) {
        ctx.calls.fetch_add(1, Ordering::SeqCst);
        CpuBackend::write_typed(&mut (), dst, data)
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
        CpuBackend::gemm(&mut (), a, b, out, m, n, k)
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
        CpuBackend::rms_norm(&mut (), x, w, eps, out, tokens, dim)
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
        CpuBackend::fused_add_rms_norm(&mut (), residual, x, w, eps, out, tokens, dim)
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
        CpuBackend::flash_attention(&mut (), q, k, v, out, batch, q_len, kv_len, pos_offset, cfg)
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
        CpuBackend::copy_slice(&mut (), src, src_offset, dst, dst_offset, len)
    }
    fn embedding_lookup(
        ctx: &mut Self::Context,
        table: &Self::Buffer,
        ids: &[u32],
        out: &mut Self::Buffer,
        dim: usize,
    ) {
        ctx.calls.fetch_add(1, Ordering::SeqCst);
        CpuBackend::embedding_lookup(&mut (), table, ids, out, dim)
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
        CpuBackend::split_qkv(&mut (), qkv, q, k, v, tokens, q_dim, kv_dim)
    }
    fn fused_silu_mul_split(
        ctx: &mut Self::Context,
        gate_up: &Self::Buffer,
        out: &mut Self::Buffer,
        tokens: usize,
        im: usize,
    ) {
        ctx.calls.fetch_add(1, Ordering::SeqCst);
        CpuBackend::fused_silu_mul_split(&mut (), gate_up, out, tokens, im)
    }
    fn scale_inplace(ctx: &mut Self::Context, buf: &mut Self::Buffer, scale: f32, len: usize) {
        ctx.calls.fetch_add(1, Ordering::SeqCst);
        CpuBackend::scale_inplace(&mut (), buf, scale, len)
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
            &mut (),
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
            &mut (),
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
        _ctx: &mut Self::Context,
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
            &mut (),
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
        _ctx: &mut Self::Context,
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
            &mut (),
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
            &mut (),
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
        CpuBackend::transpose_head_to_token(&mut (), src, dst, tokens, heads, dim)
    }
    fn transpose_token_to_head(
        _ctx: &mut Self::Context,
        _src: &Self::Buffer,
        _dst: &mut Self::Buffer,
        _tokens: usize,
        _heads: usize,
        _dim: usize,
    ) {
        CpuBackend::transpose_token_to_head(&mut (), _src, _dst, _tokens, _heads, _dim)
    }
    fn add_inplace(
        ctx: &mut Self::Context,
        residual: &mut Self::Buffer,
        x: &Self::Buffer,
        len: usize,
    ) {
        ctx.calls.fetch_add(1, Ordering::SeqCst);
        CpuBackend::add_inplace(&mut (), residual, x, len)
    }
    fn add_bias(
        ctx: &mut Self::Context,
        data: &mut Self::Buffer,
        bias: &Self::Buffer,
        rows: usize,
        cols: usize,
    ) {
        ctx.calls.fetch_add(1, Ordering::SeqCst);
        CpuBackend::add_bias(&mut (), data, bias, rows, cols)
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
        CpuBackend::layer_norm(&mut (), x, gamma, beta, eps, out, tokens, dim)
    }
    fn gelu(ctx: &mut Self::Context, x: &Self::Buffer, out: &mut Self::Buffer, len: usize) {
        ctx.calls.fetch_add(1, Ordering::SeqCst);
        CpuBackend::gelu(&mut (), x, out, len)
    }
    fn write_f32_to_activation(ctx: &mut Self::Context, dst: &mut Self::Buffer, data: &[f32]) {
        ctx.calls.fetch_add(1, Ordering::SeqCst);
        CpuBackend::write_f32_to_activation(&mut (), dst, data)
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
    let mut first = tracked_model();
    let mut second = tracked_model();
    let loader = SyntheticLlamaLoader::new(config());
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
fn instance_context_constructor_error_retires_original_before_return() {
    use ferrum_quantization::{Linear, QuantConfig, WeightLoader};
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
        fn load_linear(&self, _: &mut Context, _: &str) -> Result<Box<dyn Linear<TrackedCpu>>> {
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
