use super::*;
use crate::qwen_image21::{QwenImage21Config, QwenImage21T2iLayout, StructureLimits};
use ferrum_runtime::{HostMemoryClassLimits, HostMemoryLimits};
use serde_json::json;

fn config(causal: bool, valid: Vec<u32>) -> QwenImage21TransformerConfig {
    let model = QwenImage21Config::from_hf_json(
        &serde_json::to_vec(&json!({
            "_class_name":"QwenImage21Transformer2DModel", "num_layers":2,
            "num_attention_heads":2,"attention_head_dim":8,"context_in_dim":8,
            "in_channels":3,"out_channels":4,"patch_size":1,"mlp_ratio":3,
            "axes_dims_rope":[2,2,4],"causal_condition":causal,"eps":0.000001
        }))
        .unwrap(),
        StructureLimits::default(),
    )
    .unwrap();
    QwenImage21TransformerConfig::new(model, QwenImage21T2iLayout::new(3, 2, 4, valid).unwrap())
        .unwrap()
}

fn account(bytes: usize) -> HostMemoryAccount {
    HostMemoryAccount::new(
        HostMemoryLimits::new(
            bytes,
            HostMemoryClassLimits {
                input_preprocess_bytes: bytes,
                output_bytes: 0,
                transfer_staging_bytes: 0,
                weight_backing_bytes: 0,
            },
        )
        .unwrap(),
    )
}

fn bits(input: &ManagedHostTensor) -> Vec<u16> {
    input
        .bytes()
        .bytes()
        .chunks_exact(2)
        .map(|x| u16::from_le_bytes([x[0], x[1]]))
        .collect()
}

fn floats(input: &ManagedHostTensor) -> Vec<f32> {
    input
        .bytes()
        .bytes()
        .chunks_exact(4)
        .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
        .collect()
}

#[test]
fn time_features_keep_early_bf16_bridge_batch_rows_and_one_shared_t0() {
    let config = config(true, vec![3, 1]);
    let host = account(3 * 256 * 2);
    let input = config.managed_time_features(&host, 499.015625).unwrap();
    assert_eq!(input.value_id().as_str(), Family::TIME_FEATURES_ID);
    assert_eq!(input.tensor().dimensions(), &[3, 256]);
    assert_eq!(input.tensor().element_type(), ElementType::Bf16);
    let values = bits(&input);
    assert_eq!(&values[..256], &values[256..512]);
    // Early B(499+1/64)=500, then normalized B(.5). cos(500) is within
    // [-.883850,-.883848], wholly in BF16 -113/128's rounding cell.
    assert_eq!(values[0], 0xbf62);
    assert_eq!(&values[512..640], &[0x3f80; 128]);
    assert_eq!(&values[640..], &[0; 128]);
    assert_eq!(input.bytes().class(), HostMemoryClass::InputPreprocess);
    assert!(host.owns(input.bytes()));
    assert!(host.clone().owns(input.bytes()));
    assert!(!account(1536).owns(input.bytes()));
    assert_eq!(host.usage().unwrap().allocated_bytes(), 1536);
    drop(input);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn noncausal_rows_and_negative_zero_keep_the_existing_adapter_bits() {
    let checked = config(false, vec![3, 1]);
    let host = account(1024);
    let output = checked.managed_time_features(&host, -0.0).unwrap();
    assert_eq!(output.tensor().dimensions(), &[2, 256]);
    let actual = bits(&output);
    for row in actual.chunks_exact(256) {
        assert_eq!(&row[..128], &[0x3f80; 128]);
        assert_eq!(&row[128..], &[0x8000; 128]);
    }
    drop(output);
    let causal = config(true, vec![3]);
    let output = causal.managed_time_features(&host, -0.0).unwrap();
    let actual = bits(&output);
    assert_eq!(&actual[128..256], &[0x8000; 128]);
    assert_eq!(&actual[384..512], &[0; 128]);
}

#[test]
fn managed_frequencies_preserve_padded_text_axes_and_are_not_repeated_over_batch() {
    let config = config(true, vec![3, 1]);
    let host = account(11 * 8 * 4);
    let input = config.managed_rotary_frequencies(&host).unwrap();
    assert_eq!(input.value_id().as_str(), Family::FREQUENCIES_ID);
    assert_eq!(input.tensor().dimensions(), &[11, 4, 2]);
    assert_eq!(input.tensor().element_type(), ElementType::F32);
    let actual = floats(&input);
    assert_eq!(&actual[..8], &[1., 0., 1., 0., 1., 0., 1., 0.]);
    let near = |actual: f32, expected: f64| assert!((f64::from(actual) - expected).abs() < 1e-7);
    // First target is [padded T, centered H, centered W] = [3,-1,-2],
    // even though sample 1's valid text length is only one.
    let row = &actual[24..32];
    near(row[0], -0.9899924966004454);
    near(row[1], 0.1411200080598672);
    near(row[2], 0.5403023058681398);
    near(row[3], -0.8414709848078965);
    near(row[4], -0.4161468365471424);
    near(row[5], -0.9092974268256817);
    near(row[7], -0.0199986666933331);
    assert!(host.owns(input.bytes()));
    assert_eq!(host.usage().unwrap().allocated_bytes(), 352);
    drop(input);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn invalid_time_and_capacity_failures_publish_no_input_and_preserve_existing_owners() {
    let config = config(true, vec![3, 1]);
    let host = account(1536);
    for time in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -1., 1001.] {
        assert!(config.managed_time_features(&host, time).is_err());
        assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    }
    assert!(config.managed_time_features(&account(1535), 500.).is_err());
    assert!(config.managed_rotary_frequencies(&account(351)).is_err());
    let frequencies = config.managed_rotary_frequencies(&host).unwrap();
    let before = frequencies.bytes().bytes().to_vec();
    assert!(config.managed_time_features(&host, 500.).is_err());
    assert_eq!(host.usage().unwrap().committed_bytes(), 352);
    assert_eq!(frequencies.bytes().bytes(), before);
    drop(frequencies);
    assert_eq!(host.usage().unwrap().allocated_bytes(), 0);
}

#[test]
fn wrong_extent_alignment_and_time_arity_fail_before_caller_buffer_writes() {
    let config = config(true, vec![3, 1]);
    let adapter = QwenImage21Timestep256::new().unwrap();
    let time = QwenImage21ModelTimestep::from_scheduler_timestep(500.).unwrap();
    let mut storage = [0x1234_u16; 3 * 256];
    assert!(write_time_bytes(
        &config,
        &adapter,
        &[time],
        bytemuck::cast_slice_mut(&mut storage)
    )
    .is_err());
    assert_eq!(storage, [0x1234; 3 * 256]);
    assert!(write_time_bytes(
        &config,
        &adapter,
        &[time; 2],
        bytemuck::cast_slice_mut(&mut storage[..767])
    )
    .is_err());
    assert_eq!(storage, [0x1234; 3 * 256]);
    let mut rotary = [123_f32; 88];
    assert!(write_rotary_bytes(&config, bytemuck::cast_slice_mut(&mut rotary[..87])).is_err());
    assert_eq!(rotary, [123.; 88]);
    // Slice from a known aligned element array at byte offset one. This proves
    // a checked error, rather than depending on a platform allocator accident.
    let mut misaligned = [0x1234_u32; 89];
    let bytes = bytemuck::cast_slice_mut::<u32, u8>(&mut misaligned);
    let before = bytes.to_vec();
    assert!(write_rotary_bytes(&config, &mut bytes[1..353]).is_err());
    assert_eq!(bytes, before);
}
