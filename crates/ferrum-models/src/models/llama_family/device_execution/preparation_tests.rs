use super::*;
use crate::common::llm::{ModelKvPreparation, ModelKvReservation};
use crate::common::DecoderOnlyLLM;
use crate::models::llama_family::LlamaFamilyConfig;
use crate::test_support::SyntheticLlamaLoader;
use ferrum_execution::model_executor::KvSlotRequest;
use ferrum_interfaces::kv_dtype::KvFp16;
use ferrum_kernels::backend::cpu::CpuBackend;

fn model<B: MoeLlmBackend>(paged: bool) -> LlamaFamilyModel<B, KvFp16> {
    let config = LlamaFamilyConfig {
        hidden_size: 128,
        intermediate_size: 256,
        num_heads: 2,
        num_kv_heads: 2,
        head_dim: 64,
        num_layers: 2,
        vocab_size: 128,
        max_seq_len: 32,
        ..Default::default()
    };
    let loader = SyntheticLlamaLoader::new(config.clone());
    let mut model = LlamaFamilyModel::new(config, &loader).unwrap();
    model.runtime_env.metal_paged_kv = Some(paged);
    model.runtime_env.prefix_cache = false;
    model.runtime_env.cuda_graph = false;
    model.runtime_env.kv_capacity = Some(32);
    model.runtime_env.kv_max_blocks = Some(3);
    model.runtime_env.paged_max_seqs = 2;
    model
}

fn request(cache_id: &str, target_len: usize) -> KvSlotRequest {
    KvSlotRequest {
        cache_id: cache_id.to_owned(),
        target_len,
        admission_target_len: Some(target_len),
    }
}

fn preparation<'a>(
    cache_id: &'a str,
    capacity_hint: usize,
    prompt: &'a std::sync::Arc<[u32]>,
) -> ModelKvPreparation<'a> {
    ModelKvPreparation {
        cache_id,
        capacity_hint,
        prompt,
        adapter: None,
    }
}

fn free_blocks<B: MoeLlmBackend>(model: &LlamaFamilyModel<B, KvFp16>) -> usize {
    model
        .paged_block_alloc
        .as_ref()
        .unwrap()
        .lock()
        .unwrap()
        .free_count()
}

/// The CPU case exercises the real allocator and table preparation, not an
/// unsupported CPU paged attention kernel. Numerical continuation below uses
/// the real contiguous path on every backend.
fn real_preparation_preserves_allocator_and_numerical_state<B: MoeLlmBackend>(
    read_u32: impl Fn(&B::Buffer, usize) -> Vec<u32>,
) {
    let mut paged = model::<B>(true);
    assert!(matches!(
        paged
            .prepare_token_kv(preparation("first", 32, &std::sync::Arc::from([1; 17])))
            .unwrap(),
        ModelKvReservation::Reserved(_)
    ));
    assert!(paged.pending_device_frame.is_none());
    assert_eq!(free_blocks(&paged), 1);
    for cache in &paged.kv_caches["first"] {
        assert_eq!(cache.len, 0);
        assert_eq!(cache.paged_block_indices.len(), 2);
        assert_eq!(
            read_u32(cache.block_table.as_ref().unwrap(), 2),
            cache.paged_block_indices
        );
        assert_eq!(read_u32(cache.context_lens.as_ref().unwrap(), 1), [0]);
    }
    let ModelKvReservation::CapacityDeferred(deferral) = paged
        .prepare_token_kv(preparation("waiter", 32, &std::sync::Arc::from([1; 17])))
        .unwrap()
    else {
        panic!("real three-block pool must reject the second two-block reservation");
    };
    assert_eq!(deferral.required_blocks(), 2);
    assert_eq!(deferral.snapshot().free_blocks, 1);
    assert!(paged.kv_caches["waiter"]
        .iter()
        .all(|cache| cache.paged_block_indices.is_empty()));
    paged.release("waiter");
    assert_eq!(free_blocks(&paged), 1);
    paged.release("first");
    assert_eq!(free_blocks(&paged), 3);

    let failed: DeviceFrameResult<()> = paged.run_preparation_frame(|model, frame| {
        model.ensure_kv_in_context(&mut frame.context, "failed", Some(32));
        let result = model
            .reserve_paged_token_kv_slots_in_context(
                Some(&mut frame.context),
                &[request("failed", 17)],
            )
            .unwrap();
        assert!(matches!(result, ModelKvReservation::Reserved(_)));
        // Table writes have really happened/enqueued, but no model forward or
        // inner sync has certified their readiness. The outer frame must do it.
        std::panic::panic_any(FerrumError::backend("injected table preparation failure"));
    });
    let DeviceFrameResult::Terminal(Err(error)) = failed else {
        panic!("actual preparation must be observed before its owners can retire");
    };
    assert!(error
        .to_string()
        .contains("injected table preparation failure"));
    assert!(paged.pending_device_frame.is_none());
    assert_eq!(free_blocks(&paged), 1);
    for cache in &paged.kv_caches["failed"] {
        assert_eq!(cache.len, 0);
        assert_eq!(
            read_u32(cache.block_table.as_ref().unwrap(), 2),
            cache.paged_block_indices
        );
    }
    paged.release("failed");
    assert_eq!(free_blocks(&paged), 3);
    paged.release("failed");
    assert_eq!(free_blocks(&paged), 3);
    assert!(paged.kv_caches.is_empty());

    let mut actual = model::<B>(false);
    let mut reference = model::<B>(false);
    actual
        .prepare_token_kv(preparation("sequence", 3, &std::sync::Arc::from([1, 2, 3])))
        .unwrap();
    reference
        .prepare_token_kv(preparation(
            "reference",
            32,
            &std::sync::Arc::from([1, 2, 3]),
        ))
        .unwrap();
    let first = actual.prefill_internal("sequence", &[1, 2, 3]);
    assert_eq!(first, reference.prefill_internal("reference", &[1, 2, 3]));
    assert!(first.iter().all(|value| value.is_finite()));
    let heads = actual.cfg.num_kv_heads;
    let head_dim = actual.cfg.head_dim;
    assert_eq!(heads, 2);
    let old_kv: Vec<_> = actual.kv_caches["sequence"]
        .iter()
        .map(|cache| {
            assert_eq!(cache.capacity, 3);
            let count = heads * cache.capacity * head_dim;
            (B::to_vec(&cache.k, count), B::to_vec(&cache.v, count))
        })
        .collect();
    let failed: DeviceFrameResult<()> = actual.run_preparation_frame(|model, frame| {
        for cache in model.kv_caches.get_mut("sequence").unwrap() {
            <KvFp16 as KvLayer<B>>::ensure_contig_capacity(
                &mut frame.context,
                cache,
                7,
                heads,
                head_dim,
            )
            .unwrap();
        }
        std::panic::panic_any(FerrumError::backend("injected contiguous growth failure"));
    });
    let DeviceFrameResult::Terminal(Err(error)) = failed else {
        panic!("real growth copies require the caller's terminal observation");
    };
    assert!(error
        .to_string()
        .contains("injected contiguous growth failure"));
    assert!(actual.pending_device_frame.is_none());
    for (cache, (expected_k, expected_v)) in actual.kv_caches["sequence"].iter().zip(old_kv) {
        assert_eq!(cache.len, 3);
        assert_eq!(cache.capacity, 7);
        let count = heads * cache.capacity * head_dim;
        let copied_k = B::to_vec(&cache.k, count);
        let copied_v = B::to_vec(&cache.v, count);
        // Each head occupies its own capacity stride. A flat prefix copy can
        // preserve head zero while corrupting every subsequent head.
        for head in 0..heads {
            let old_start = head * 3 * head_dim;
            let new_start = head * 7 * head_dim;
            let live = cache.len * head_dim;
            assert_eq!(
                &copied_k[new_start..new_start + live],
                &expected_k[old_start..old_start + live]
            );
            assert_eq!(
                &copied_v[new_start..new_start + live],
                &expected_v[old_start..old_start + live]
            );
        }
    }
    let actual_logits = actual.decode_internal("sequence", 4, 3);
    let reference_logits = reference.decode_internal("reference", 4, 3);
    assert_eq!(actual_logits, reference_logits);
    assert!(actual_logits.iter().all(|value| value.is_finite()));
    actual.release("sequence");
    reference.release("reference");
    assert!(actual.kv_caches.is_empty());
    assert!(reference.kv_caches.is_empty());
}

#[test]
fn preparation_frame_real_cpu_allocator_and_contiguous_growth() {
    real_preparation_preserves_allocator_and_numerical_state::<CpuBackend>(|buffer, count| {
        buffer[..count].iter().map(|word| word.to_bits()).collect()
    });
}

#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
#[test]
#[ignore = "requires an actual Metal device"]
fn preparation_frame_real_metal_allocator_and_contiguous_growth() {
    use ferrum_kernels::backend::metal::MetalBackend;
    real_preparation_preserves_allocator_and_numerical_state::<MetalBackend>(|buffer, count| {
        // The preparation frame must have completed before host shared-memory
        // inspection. This read adds no synchronization that could hide a bug.
        unsafe { std::slice::from_raw_parts(buffer.raw().contents().cast::<u32>(), count).to_vec() }
    });
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires an actual CUDA device"]
fn preparation_frame_real_cuda_allocator_and_contiguous_growth() {
    use ferrum_kernels::backend::{cuda::CudaBackend, Backend};
    let context = CudaBackend::new_context();
    real_preparation_preserves_allocator_and_numerical_state::<CudaBackend>(|buffer, count| {
        context.stream.clone_dtoh(buffer.as_u32()).unwrap()[..count].to_vec()
    });
}

#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
#[test]
#[ignore = "requires an actual Metal device"]
fn preparation_frame_real_metal_pending_context_blocks_allocation() {
    use ferrum_kernels::backend::{metal::MetalBackend, Backend};
    let mut model = model::<MetalBackend>(true);
    let mut empty = MetalBackend::new_context();
    assert!(MetalBackend::preparation_is_host_only(&empty));
    assert!(MetalBackend::sync_checked(&mut empty).is_err());
    // Ordinary numerical observation cannot claim completion without a
    // submission. Its real unresolved context must block even host-only prep.
    assert!(matches!(
        model.run_device_frame(|_, _| ()),
        DeviceFrameResult::Held(_)
    ));
    assert!(model
        .prepare_token_kv(preparation("blocked", 32, &std::sync::Arc::from([1])))
        .is_err());
    assert!(model.kv_caches.is_empty());
    assert!(model.paged_block_alloc.is_none());
    let mut frame = model.pending_device_frame.take().unwrap();
    let count = model.cfg.hidden_size;
    let buffer = &mut model.scratch.last_normed;
    MetalBackend::write_f32_to_activation(&mut frame.context, buffer, &vec![1.0; count]);
    MetalBackend::scale_inplace(&mut frame.context, buffer, 2.0, count);
    assert!(!MetalBackend::preparation_is_host_only(&frame.context));
    let FrameObservation::Terminal { owners, error } = frame.observe() else {
        panic!("the same context's actual encoded work must reach terminal");
    };
    assert!(error.is_none());
    model.execution_context = Some(owners.restore(&mut model.scratch));
    assert_eq!(
        MetalBackend::to_vec(&model.scratch.last_normed, count),
        vec![2.0; count]
    );
    assert!(matches!(
        model
            .prepare_token_kv(preparation("ready", 32, &std::sync::Arc::from([1])))
            .unwrap(),
        ModelKvReservation::Reserved(_)
    ));
    model.release("ready");
    assert_eq!(free_blocks(&model), 3);
}
