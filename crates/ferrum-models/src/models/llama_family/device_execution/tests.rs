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

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires an actual CUDA device"]
fn single_device_frame_restores_original_scratch_after_real_cuda_forward_failure() {
    use cudarc::driver::DevicePtr;
    use ferrum_kernels::backend::{cuda::CudaBackend, Backend};

    // Initialization failure is a failed selected test, never a skipped case.
    let context = CudaBackend::new_context();
    real_forward_unwind_restores_original_scratch::<CudaBackend>(|buffer| {
        let (pointer, _record) = buffer.as_f16().device_ptr(&context.stream);
        usize::try_from(pointer).expect("CUDA allocation address must fit the host pointer width")
    });
}
