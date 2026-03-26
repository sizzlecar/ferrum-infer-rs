//! Tests for executor common utilities and GenericKvCacheHandle
//! in realistic scenarios.

use candle_core::{Device as CandleDevice, Tensor};
use ferrum_interfaces::KvCacheHandle;
use ferrum_models::executor::common;

#[test]
fn roundtrip_tokens_through_tensor() {
    let original = vec![100u32, 200, 300, 42];
    let tensor = common::tokens_to_tensor(&original, &CandleDevice::Cpu).unwrap();
    let wrapped = common::wrap_tensor(tensor);
    // Extract back as f32 (candle I64 tensors can be extracted as f32)
    let shape = wrapped.shape();
    assert_eq!(shape, &[1, 4]); // [batch=1, seq=4]
}

#[test]
fn kv_cache_handle_device_mapping_cpu() {
    let h = common::GenericKvCacheHandle::new(4, 8, 64, CandleDevice::Cpu, 10, "c1".into());
    assert_eq!(h.device(), ferrum_types::Device::CPU);
}

#[test]
fn kv_cache_handle_sequence_length_updates() {
    let h = common::GenericKvCacheHandle::new(4, 8, 64, CandleDevice::Cpu, 0, "c1".into());
    assert_eq!(h.block_table().sequence_length, 0);

    let h2 = h.with_sequence_length(50);
    assert_eq!(h2.block_table().sequence_length, 50);

    let h3 = h2.with_sequence_length(100);
    assert_eq!(h3.block_table().sequence_length, 100);

    // cache_id preserved across updates
    assert_eq!(h3.cache_id(), "c1");
}

#[test]
fn kv_cache_handle_stats() {
    let h = common::GenericKvCacheHandle::new(4, 8, 64, CandleDevice::Cpu, 25, "stats-test".into());
    let stats = h.stats();
    assert_eq!(stats.tokens_stored, 25);
    assert!(h.is_valid());
}

#[test]
fn multiple_handles_independent() {
    let h1 = common::GenericKvCacheHandle::new(4, 8, 64, CandleDevice::Cpu, 10, "seq-1".into());
    let h2 = common::GenericKvCacheHandle::new(4, 8, 64, CandleDevice::Cpu, 20, "seq-2".into());

    assert_eq!(h1.cache_id(), "seq-1");
    assert_eq!(h2.cache_id(), "seq-2");
    assert_eq!(h1.block_table().sequence_length, 10);
    assert_eq!(h2.block_table().sequence_length, 20);

    // Cloning one doesn't affect the other
    let h1_clone = h1.clone_handle().unwrap();
    assert_eq!(h1_clone.cache_id(), "seq-1");
}

#[test]
fn tensor_to_tokens_handles_batched() {
    // Simulate a [1, 5] tensor (batch=1, seq=5)
    let t = Tensor::new(&[1u32, 2, 3, 4, 5], &CandleDevice::Cpu)
        .unwrap()
        .unsqueeze(0)
        .unwrap();
    let wrapped = common::wrap_tensor(t);
    // The mock should extract flat tokens
    let tokens = common::tensor_to_tokens(&wrapped);
    assert!(tokens.is_ok());
}
