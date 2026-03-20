//! Integration tests for paged KV cache lifecycle:
//! multi-request block sharing, isolation, deallocation, and reuse.

use ferrum_interfaces::kv_cache::AllocationRequest;
use ferrum_interfaces::KvCacheManager;
use ferrum_kv::attention::paged_attention;
use ferrum_kv::managers::paged::{PagedKvCacheConfig, PagedKvCacheHandle, PagedKvCacheManager};
use ferrum_types::{DataType, Device, Priority, RequestId};
use std::sync::Arc;

fn small_config(block_size: usize, max_blocks: usize) -> PagedKvCacheConfig {
    PagedKvCacheConfig {
        block_size,
        max_gpu_blocks: max_blocks,
        max_cpu_blocks: 0,
        enable_cow: false,
        enable_swapping: false,
        num_layers: 1,
        num_heads: 2,
        head_dim: 4,
        enable_prefix_cache: false,
        ..Default::default()
    }
}

fn make_request(initial_tokens: usize) -> AllocationRequest {
    AllocationRequest {
        request_id: RequestId::new(),
        initial_tokens,
        max_sequence_length: 256,
        num_layers: 1,
        num_heads: 2,
        head_dim: 4,
        device: Device::CPU,
        dtype: DataType::FP16,
        priority: Priority::Normal,
    }
}

fn get_handle(
    manager: &PagedKvCacheManager,
    rid: &RequestId,
) -> Arc<dyn ferrum_interfaces::KvCacheHandle> {
    manager.get_handle(rid.clone()).unwrap()
}

const KV_SIZE: usize = 2 * 4; // num_heads * head_dim

#[tokio::test]
async fn two_requests_share_block_pool_without_corruption() {
    let config = small_config(4, 8);
    let manager = PagedKvCacheManager::new(Device::CPU, config).unwrap();

    // Request A: 6 tokens → 2 blocks
    let req_a = make_request(6);
    let rid_a = req_a.request_id.clone();
    let _ = manager.allocate(&req_a).await.unwrap();

    // Request B: 5 tokens → 2 blocks
    let req_b = make_request(5);
    let rid_b = req_b.request_id.clone();
    let _ = manager.allocate(&req_b).await.unwrap();

    // Write distinct data to each request
    let arc_a = get_handle(&manager, &rid_a);
    let handle_a = arc_a.as_any().downcast_ref::<PagedKvCacheHandle>().unwrap();
    for pos in 0..6 {
        let key: Vec<f32> = (0..KV_SIZE).map(|i| (pos * 10 + i) as f32).collect();
        let val: Vec<f32> = (0..KV_SIZE).map(|i| (pos * 10 + i + 100) as f32).collect();
        manager.write_kv(handle_a, 0, pos, &key, &val).unwrap();
    }

    let arc_b = get_handle(&manager, &rid_b);
    let handle_b = arc_b.as_any().downcast_ref::<PagedKvCacheHandle>().unwrap();
    for pos in 0..5 {
        let key: Vec<f32> = (0..KV_SIZE).map(|i| (pos * 20 + i + 1000) as f32).collect();
        let val: Vec<f32> = (0..KV_SIZE).map(|i| (pos * 20 + i + 2000) as f32).collect();
        manager.write_kv(handle_b, 0, pos, &key, &val).unwrap();
    }

    // Read back request A and verify no contamination from B
    let (keys_a, _) = manager.read_kv(handle_a, 0, 0, 6).unwrap();
    assert_eq!(keys_a[0], 0.0);
    assert_eq!(keys_a[KV_SIZE], 10.0);

    // Read back request B
    let (keys_b, _) = manager.read_kv(handle_b, 0, 0, 5).unwrap();
    assert_eq!(keys_b[0], 1000.0);
    assert_eq!(keys_b[KV_SIZE], 1020.0);

    // Run attention on both — should not interfere
    let q = vec![1.0f32; KV_SIZE];
    let out_a = paged_attention(&q, 1, 2, 2, 4, &manager, handle_a, 0, 6).unwrap();
    let out_b = paged_attention(&q, 1, 2, 2, 4, &manager, handle_b, 0, 5).unwrap();
    assert!(out_a != out_b, "Outputs from different requests should differ");

    manager.deallocate(rid_a).await.unwrap();
    manager.deallocate(rid_b).await.unwrap();
}

#[tokio::test]
async fn deallocated_blocks_are_reused() {
    let config = small_config(4, 4);
    let manager = PagedKvCacheManager::new(Device::CPU, config).unwrap();

    // Request 1: 8 tokens → 2 blocks
    let req1 = make_request(8);
    let rid1 = req1.request_id.clone();
    let _ = manager.allocate(&req1).await.unwrap();

    // Request 2: 8 tokens → 2 blocks (fills remaining budget)
    let req2 = make_request(8);
    let rid2 = req2.request_id.clone();
    let _ = manager.allocate(&req2).await.unwrap();

    // Pool is now full
    let stats = manager.stats();
    assert_eq!(stats.free_blocks, 0);

    // Request 3 should fail
    let req3_fail = make_request(4);
    assert!(manager.allocate(&req3_fail).await.is_err());

    // Deallocate request 1 → frees 2 blocks
    manager.deallocate(rid1).await.unwrap();

    // Now request 3 should succeed (reuses freed blocks)
    let req3 = make_request(4);
    let rid3 = req3.request_id.clone();
    assert!(manager.allocate(&req3).await.is_ok());

    // Verify the reused blocks work
    let arc3 = get_handle(&manager, &rid3);
    let handle3 = arc3.as_any().downcast_ref::<PagedKvCacheHandle>().unwrap();
    let key = vec![42.0f32; KV_SIZE];
    let val = vec![99.0f32; KV_SIZE];
    manager.write_kv(handle3, 0, 0, &key, &val).unwrap();

    let (k, v) = manager.read_kv(handle3, 0, 0, 1).unwrap();
    assert_eq!(k, key);
    assert_eq!(v, val);

    manager.deallocate(rid2).await.unwrap();
    manager.deallocate(rid3).await.unwrap();
}

#[tokio::test]
async fn extend_allocates_additional_blocks() {
    let config = small_config(4, 16);
    let manager = PagedKvCacheManager::new(Device::CPU, config).unwrap();

    let req = make_request(4); // 1 block
    let rid = req.request_id.clone();
    let _ = manager.allocate(&req).await.unwrap();

    let arc = get_handle(&manager, &rid);
    let handle = arc.as_any().downcast_ref::<PagedKvCacheHandle>().unwrap();
    assert_eq!(handle.num_blocks(), 1);

    // Write 4 initial tokens
    for pos in 0..4 {
        let key = vec![(pos + 1) as f32; KV_SIZE];
        let val = vec![(pos + 1) as f32 * 10.0; KV_SIZE];
        manager.write_kv(handle, 0, pos, &key, &val).unwrap();
    }

    // Extend: allocate 1 more block for tokens 4-7
    manager.allocate_blocks(handle, 1).unwrap();
    handle.set_num_tokens(8);
    assert_eq!(handle.num_blocks(), 2);

    // Write tokens 4-7 in the new block
    for pos in 4..8 {
        let key = vec![(pos + 1) as f32; KV_SIZE];
        let val = vec![(pos + 1) as f32 * 10.0; KV_SIZE];
        manager.write_kv(handle, 0, pos, &key, &val).unwrap();
    }

    // Read all 8 tokens spanning both blocks
    let (keys, _) = manager.read_kv(handle, 0, 0, 8).unwrap();
    assert_eq!(keys.len(), 8 * KV_SIZE);
    assert_eq!(keys[0], 1.0); // token 0
    assert_eq!(keys[4 * KV_SIZE], 5.0); // token 4
    assert_eq!(keys[7 * KV_SIZE], 8.0); // token 7

    // Attention over all 8 tokens
    let query = vec![1.0f32; KV_SIZE];
    let output = paged_attention(&query, 1, 2, 2, 4, &manager, handle, 0, 8).unwrap();
    assert_eq!(output.len(), KV_SIZE);

    manager.deallocate(rid).await.unwrap();
}

#[tokio::test]
async fn no_block_leak_after_full_lifecycle() {
    let config = small_config(4, 8);
    let manager = PagedKvCacheManager::new(Device::CPU, config).unwrap();

    // Run 10 allocate-use-deallocate cycles
    for _ in 0..10 {
        let req = make_request(6); // 2 blocks each
        let rid = req.request_id.clone();
        let _ = manager.allocate(&req).await.unwrap();

        let arc = get_handle(&manager, &rid);
        let handle = arc.as_any().downcast_ref::<PagedKvCacheHandle>().unwrap();
        for pos in 0..6 {
            let key = vec![1.0f32; KV_SIZE];
            let val = vec![2.0f32; KV_SIZE];
            manager.write_kv(handle, 0, pos, &key, &val).unwrap();
        }

        manager.deallocate(rid).await.unwrap();
    }

    let stats = manager.stats();
    assert_eq!(stats.active_caches, 0);
}
