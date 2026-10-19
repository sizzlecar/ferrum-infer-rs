use super::*;
use crate::common::DecoderOnlyLLM;
use crate::models::llama_family::LlamaFamilyConfig;
use crate::test_support::SyntheticLlamaLoader;
use ferrum_interfaces::kv_dtype::KvFp16;
use ferrum_kernels::backend::cpu::CpuBackend;

fn real_forward_unwind_restores_original_scratch<B: MoeLlmBackend>(
    identity: impl Fn(&B::Buffer) -> usize,
) {
    let config = LlamaFamilyConfig {
        hidden_size: 128,
        intermediate_size: 256,
        num_heads: 2,
        num_kv_heads: 1,
        head_dim: 64,
        num_layers: 1,
        vocab_size: 128,
        max_seq_len: 32,
        ..Default::default()
    };
    let loader = SyntheticLlamaLoader::new(config.clone());
    let mut model = LlamaFamilyModel::<B, KvFp16>::new(config.clone(), &loader).unwrap();
    let mut reference = LlamaFamilyModel::<B, KvFp16>::new(config, &loader).unwrap();
    for current in [&mut model, &mut reference] {
        current.runtime_env.metal_paged_kv = Some(false);
        current.runtime_env.prefix_cache = false;
        current.runtime_env.cuda_graph = false;
        current.runtime_env.kv_capacity = Some(32);
        current.ensure_scratch(3);
    }
    let original = identity(model.scratch.residual.as_ref().unwrap());
    let reference_logits = reference.prefill_internal("oracle", &[1, 2, 3]);
    assert!(reference_logits.iter().all(|value| value.is_finite()));
    let result: DeviceFrameResult<()> = model.run_device_frame(|model, frame| {
        let logits = model.prefill_in_frame(frame, "failed", &[1, 2, 3]);
        assert_eq!(logits, reference_logits);
        assert!(model.scratch.residual.is_none());
        assert_eq!(identity(frame.residual.as_ref().unwrap()), original);
        // The numerical core really finished, including actual device
        // readback. Unwinding the caller must return the same scratch owner.
        std::panic::panic_any(FerrumError::backend("injected post-forward failure"));
    });
    let DeviceFrameResult::Terminal(Err(error)) = result else {
        panic!("actual completed forward must retain its terminal failure");
    };
    assert!(error.to_string().contains("injected post-forward failure"));
    assert!(model.pending_device_frame.is_none());
    assert_eq!(identity(model.scratch.residual.as_ref().unwrap()), original);
    assert_eq!(model.cache_len("failed"), 3);
    assert!(model.kv_caches["failed"].iter().all(|cache| cache.len == 3));
    // Releasing only the failed incarnation does not reset the model or
    // replay its input. A distinct sequence still produces complete logits.
    model.release("failed");
    assert_eq!(model.prefill_internal("peer", &[1, 2, 3]), reference_logits);
    assert_eq!(model.cache_len("peer"), 3);

    let count = 3 * model.cfg.hidden_size;
    let before = B::to_vec(model.scratch.residual.as_ref().unwrap(), count);
    let encoded: DeviceFrameResult<()> = model.run_device_frame(|model, frame| {
        frame
            .take_scratch(&mut model.scratch, FrameLayout::Single, false)
            .unwrap();
        B::scale_inplace(
            &mut frame.context,
            frame.residual.as_mut().unwrap(),
            2.0,
            count,
        );
        // Unlike the first failure, this one happens before synchronization.
        // The outer owner must observe this same encoded context before
        // restoring its borrowed buffer, including on the actual Metal test.
        std::panic::panic_any(FerrumError::backend("injected encoded-work failure"));
    });
    let DeviceFrameResult::Terminal(Err(error)) = encoded else {
        panic!("actual encoded work must be observed before restoring scratch");
    };
    assert!(error.to_string().contains("injected encoded-work failure"));
    assert!(model.pending_device_frame.is_none());
    assert_eq!(identity(model.scratch.residual.as_ref().unwrap()), original);
    assert_eq!(
        B::to_vec(model.scratch.residual.as_ref().unwrap(), count),
        before
            .into_iter()
            .map(|value| 2.0 * value)
            .collect::<Vec<_>>(),
    );
    model.release("peer");
    assert!(model.kv_caches.is_empty());
}

#[test]
fn single_device_frame_restores_original_scratch_after_real_cpu_forward_failure() {
    real_forward_unwind_restores_original_scratch::<CpuBackend>(|buffer| buffer.as_ptr() as usize);
}

#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
#[test]
#[ignore = "requires an actual Metal device"]
fn single_device_frame_restores_original_scratch_after_real_metal_forward_failure() {
    use ferrum_kernels::backend::metal::MetalBackend;
    real_forward_unwind_restores_original_scratch::<MetalBackend>(|buffer| {
        buffer.raw().contents() as usize
    });
}

#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
#[test]
#[ignore = "requires an actual Metal model and original host-only context"]
fn metal_frame_failure_before_first_encode_restores_owner_and_full_numerical_execution() {
    use ferrum_kernels::backend::{metal::MetalBackend, Backend};

    let config = LlamaFamilyConfig {
        hidden_size: 128,
        intermediate_size: 256,
        num_heads: 2,
        num_kv_heads: 1,
        head_dim: 64,
        num_layers: 1,
        vocab_size: 128,
        max_seq_len: 32,
        ..Default::default()
    };
    let loader = SyntheticLlamaLoader::new(config.clone());
    let mut model = LlamaFamilyModel::<MetalBackend, KvFp16>::new(config.clone(), &loader).unwrap();
    let mut reference = LlamaFamilyModel::<MetalBackend, KvFp16>::new(config, &loader).unwrap();
    for current in [&mut model, &mut reference] {
        current.runtime_env.metal_paged_kv = Some(false);
        current.runtime_env.prefix_cache = false;
        current.runtime_env.cuda_graph = false;
        current.runtime_env.kv_capacity = Some(32);
        current.ensure_scratch(3);
    }
    let original = model.scratch.residual.as_ref().unwrap().raw().contents();
    let failed: DeviceFrameResult<()> = model.run_device_frame(|model, frame| {
        frame
            .take_scratch(&mut model.scratch, FrameLayout::Single, false)
            .unwrap();
        assert_eq!(frame.residual.as_ref().unwrap().raw().contents(), original);
        assert!(model.scratch.residual.is_none());
        // The real Metal context proves that no encoder, command, submission
        // or terminal observation exists in this successfully begun invocation.
        assert!(MetalBackend::preparation_is_host_only(&frame.context));
        std::panic::panic_any(FerrumError::backend(
            "injected failure before first Metal encode",
        ));
    });
    let DeviceFrameResult::Terminal(Err(error)) = failed else {
        panic!("a proven host-only failure must not leave a permanently Held model");
    };
    assert!(error
        .to_string()
        .contains("injected failure before first Metal encode"));
    assert!(model.pending_device_frame.is_none());
    assert_eq!(
        model.scratch.residual.as_ref().unwrap().raw().contents(),
        original
    );
    assert!(model.kv_caches.is_empty());
    // Resource retirement of a failed host-only operation is not a certificate
    // that a successful numerical operation or an empty checked sync completed.
    assert!(MetalBackend::sync_checked(model.execution_context.as_mut().unwrap()).is_err());
    assert!(matches!(
        model.recover_token_frame(),
        crate::common::llm::ModelExecutionRecovery::Quiescent { error: None }
    ));

    let expected_prefill = reference.prefill_internal("oracle", &[1, 2, 3]);
    let actual_prefill = model.prefill_internal("continued", &[1, 2, 3]);
    assert_eq!(actual_prefill.len(), 128);
    assert!(actual_prefill.iter().all(|value| value.is_finite()));
    assert_eq!(actual_prefill, expected_prefill);
    let expected_decode = reference.decode_internal("oracle", 4, 3);
    let actual_decode = model.decode_internal("continued", 4, 3);
    assert_eq!(actual_decode.len(), 128);
    assert!(actual_decode.iter().all(|value| value.is_finite()));
    assert_eq!(actual_decode, expected_decode);
    assert_eq!(model.cache_len("continued"), 4);
    assert_eq!(
        model.scratch.residual.as_ref().unwrap().raw().contents(),
        original
    );
    model.release("continued");
    reference.release("oracle");
    model.close().unwrap();
    reference.close().unwrap();
    assert!(model.execution_context.is_none() && model.pending_device_frame.is_none());
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires an actual CUDA device"]
fn single_device_frame_restores_original_scratch_after_real_cuda_forward_failure() {
    use cudarc::driver::DevicePtr;
    use ferrum_kernels::backend::cuda::CudaBackend;

    real_forward_unwind_restores_original_scratch::<CudaBackend>(|buffer| {
        let (pointer, _record) = buffer.as_f16().device_ptr(buffer.as_f16().stream());
        usize::try_from(pointer).expect("CUDA allocation address must fit the host pointer width")
    });
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires an actual CUDA device and model capture cancellation"]
fn model_frame_cancels_failed_cuda_capture_without_executing_prefix() {
    use cudarc::driver::DevicePtr;
    use ferrum_kernels::backend::{cuda::CudaBackend, Backend, BackendGraph};
    use std::sync::Arc;

    fn assert_original_stream(
        context: &mut ferrum_kernels::backend::cuda::CudaState,
        original_stream: &Arc<cudarc::driver::CudaStream>,
    ) {
        // An actual allocation through the recovered context must still use
        // the original instance's stream. Inspect the existing buffer API,
        // without opening a native-owner getter on the production context.
        let probe = CudaBackend::alloc_in_context(context, 1);
        assert!(Arc::ptr_eq(probe.as_f16().stream(), original_stream));
        assert!(matches!(
            CudaBackend::observe_completion(context),
            ferrum_kernels::backend::BackendCompletion::Succeeded
        ));
    }

    let config = LlamaFamilyConfig {
        hidden_size: 128,
        intermediate_size: 256,
        num_heads: 2,
        num_kv_heads: 1,
        head_dim: 64,
        num_layers: 1,
        vocab_size: 128,
        max_seq_len: 32,
        ..Default::default()
    };
    let loader = SyntheticLlamaLoader::new(config.clone());
    let mut model = LlamaFamilyModel::<CudaBackend, KvFp16>::new(config.clone(), &loader).unwrap();
    let mut reference = LlamaFamilyModel::<CudaBackend, KvFp16>::new(config, &loader).unwrap();
    for current in [&mut model, &mut reference] {
        current.runtime_env.metal_paged_kv = Some(false);
        current.runtime_env.prefix_cache = false;
        current.runtime_env.cuda_graph = false;
        current.runtime_env.kv_capacity = Some(32);
        current.ensure_scratch(3);
    }
    let expected_prefill = reference.prefill_internal("oracle", &[1, 2, 3]);
    assert_eq!(
        model.prefill_internal("warmup", &[1, 2, 3]),
        expected_prefill
    );
    assert_eq!(expected_prefill.len(), 128);
    assert!(expected_prefill.iter().all(|value| value.is_finite()));
    model.release("warmup");

    let count = 3 * model.cfg.hidden_size;
    let sentinel = vec![0.5; count];
    let prepared: DeviceFrameResult<()> = model.run_device_frame(|model, frame| {
        frame
            .take_scratch(&mut model.scratch, FrameLayout::Single, false)
            .unwrap();
        let residual = frame.residual.as_mut().unwrap();
        // Load the real zero kernel eagerly; only its captured execution is
        // under test below, not a first-use module load during capture.
        CudaBackend::zero_buffer(&mut frame.context, residual, count).unwrap();
        CudaBackend::write_f32_to_activation(&mut frame.context, residual, &sentinel);
    });
    assert!(matches!(prepared, DeviceFrameResult::Terminal(Ok(()))));
    let original_stream = model
        .scratch
        .residual
        .as_ref()
        .unwrap()
        .as_f16()
        .stream()
        .clone();
    let original_buffer = model
        .scratch
        .residual
        .as_ref()
        .unwrap()
        .as_f16()
        .device_ptr(&original_stream)
        .0;

    let failed: DeviceFrameResult<()> = model.run_device_frame(|model, frame| {
        frame
            .take_scratch(&mut model.scratch, FrameLayout::Single, false)
            .unwrap();
        CudaBackend::begin_graph_capture(&mut frame.context).unwrap();
        CudaBackend::zero_buffer(&mut frame.context, frame.residual.as_mut().unwrap(), count)
            .unwrap();
        // The model's real unwind boundary must cancel this recorded prefix.
        // This test intentionally never calls abort_graph_capture itself.
        std::panic::panic_any(FerrumError::backend("model capture prefix failure"));
    });
    let DeviceFrameResult::Terminal(Err(error)) = failed else {
        panic!("cancelled model capture must preserve its terminal encoding error");
    };
    assert_eq!(
        error.to_string(),
        FerrumError::backend("model capture prefix failure").to_string()
    );
    assert!(model.pending_device_frame.is_none());
    let context = model.execution_context.as_mut().unwrap();
    assert!(!CudaBackend::graph_capture_in_flight(context));
    assert_original_stream(context, &original_stream);
    let residual = model.scratch.residual.as_ref().unwrap();
    assert_eq!(
        residual.as_f16().device_ptr(&original_stream).0,
        original_buffer
    );
    assert_eq!(
        CudaBackend::to_vec_in_context(context, residual, count),
        sentinel
    );

    // The same model/context remains usable. Both complete logit vectors and
    // real per-layer KV progress must agree with an independent model owner.
    assert_eq!(
        model.prefill_internal("after", &[1, 2, 3]),
        expected_prefill
    );
    assert!(model.kv_caches["after"].iter().all(|cache| cache.len == 3));
    let expected_decode = reference.decode_internal("oracle", 4, 3);
    let actual_decode = model.decode_internal("after", 4, 3);
    assert_eq!(actual_decode.len(), 128);
    assert!(actual_decode.iter().all(|value| value.is_finite()));
    assert_eq!(actual_decode, expected_decode);
    assert!(model.kv_caches["after"].iter().all(|cache| cache.len == 4));
    assert_original_stream(model.execution_context.as_mut().unwrap(), &original_stream);
    model.release("after");
    reference.release("oracle");
    assert!(model.kv_caches.is_empty());
    assert!(reference.kv_caches.is_empty());
    model.close().unwrap();
    reference.close().unwrap();
}
