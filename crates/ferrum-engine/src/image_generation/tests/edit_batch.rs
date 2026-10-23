use super::*;
use half::bf16;
use serde_json::json;

fn host(bytes: usize) -> HostMemoryAccount {
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

fn packed(host: &HostMemoryAccount, shape: QwenImage21LatentShape, bits: u16) -> ManagedHostTensor {
    let mut bytes = host
        .allocate_zeroed(HostMemoryClass::InputPreprocess, shape.byte_len())
        .unwrap();
    for value in bytes.as_mut().chunks_exact_mut(2) {
        value.copy_from_slice(&bits.to_le_bytes());
    }
    ManagedHostTensor::new(
        id(QwenImage21TransformerFamily::REFERENCE_LATENTS_ID).unwrap(),
        shape.tensor(QwenImage21LatentLayout::Packed).unwrap(),
        bytes.freeze(),
    )
    .unwrap()
}

fn normalized(
    host: &HostMemoryAccount,
    shape: QwenImage21LatentShape,
    offset: f32,
) -> ManagedHostTensor {
    let mut bytes = host
        .allocate_zeroed(HostMemoryClass::InputPreprocess, shape.byte_len())
        .unwrap();
    for (index, value) in bytes.as_mut().chunks_exact_mut(2).enumerate() {
        // Distinct channels and positions test packed order as well as sample order.
        value.copy_from_slice(
            &bf16::from_f32(offset + (index % 11) as f32 / 8.0)
                .to_bits()
                .to_le_bytes(),
        );
    }
    let tensor = ManagedHostTensor::new(
        id("value.edit.test.posterior").unwrap(),
        shape.tensor(QwenImage21LatentLayout::Nchw).unwrap(),
        bytes.freeze(),
    )
    .unwrap();
    let posterior = QwenImage21ManagedLatents::new(
        host,
        tensor,
        shape,
        QwenImage21LatentLayout::Nchw,
        QwenImage21LatentDomain::Denormalized,
    )
    .unwrap();
    let statistics = QwenImage21LatentStatistics::from_vae_config(&json!({
        "latents_mean": vec![0.25; 64], "latents_std": vec![0.75; 64],
    }))
    .unwrap();
    posterior
        .normalize_from_vae(
            host,
            &statistics,
            id(QwenImage21TransformerFamily::REFERENCE_LATENTS_ID).unwrap(),
        )
        .unwrap()
        .into_input_tensor()
        .unwrap()
}

#[test]
fn packed_reference_batch_preserves_independent_normalized_samples_and_host_claims() {
    let shape = QwenImage21LatentShape::new(1, 2, 2).unwrap();
    let host = host(shape.byte_len() * 4);
    let mut batch = ReferenceLatentBatch::new(&host, 2, shape).unwrap();
    let mut expected = Vec::new();
    for offset in [1.0, -2.0] {
        let sample = normalized(&host, shape, offset);
        expected.extend_from_slice(sample.bytes().bytes());
        batch.push(sample).unwrap();
        assert_eq!(
            host.usage().unwrap().committed_bytes(),
            shape.byte_len() * 2
        );
    }
    let output = batch.finish().unwrap();
    assert_eq!(output.tensor().dimensions(), [8, 64]);
    assert_eq!(output.bytes().bytes(), expected);
    assert_ne!(&expected[..shape.byte_len()], &expected[shape.byte_len()..]);
    let retained = output.clone();
    drop(output);
    assert_eq!(
        host.usage().unwrap().committed_bytes(),
        shape.byte_len() * 2
    );
    drop(retained);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn one_reference_reuses_original_normalized_owner_without_an_extra_allocation() {
    let shape = QwenImage21LatentShape::new(1, 2, 2).unwrap();
    let host = host(shape.byte_len());
    let source = packed(&host, shape, 0x3f80);
    let original = source.bytes().clone();
    let mut batch = ReferenceLatentBatch::new(&host, 1, shape).unwrap();
    batch.push(source).unwrap();
    let output = batch.finish().unwrap();
    assert!(output.bytes().same_region(&original));
    assert_eq!(host.usage().unwrap().committed_bytes(), shape.byte_len());
    drop((output, original));
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn packed_reference_batch_enforces_capacity_identity_shape_and_complete_population() {
    let shape = QwenImage21LatentShape::new(1, 2, 2).unwrap();
    let too_small = host(shape.byte_len() * 2 - 1);
    assert!(ReferenceLatentBatch::new(&too_small, 2, shape).is_err());
    assert_eq!(too_small.usage().unwrap().committed_bytes(), 0);
    let host = host(shape.byte_len() * 3);
    let mut batch = ReferenceLatentBatch::new(&host, 2, shape).unwrap();
    let foreign = self::host(shape.byte_len());
    assert!(batch.push(packed(&foreign, shape, 0x3f80)).is_err());
    assert_eq!(foreign.usage().unwrap().committed_bytes(), 0);
    let source = packed(&host, shape, 0x3f80);
    let wrong_shape = ManagedHostTensor::new(
        source.value_id().clone(),
        ResolvedTensorSpec::new(
            vec![2, 128],
            ElementType::Bf16,
            ResolvedTensorLayout::Contiguous,
        )
        .unwrap(),
        source.bytes().clone(),
    )
    .unwrap();
    assert!(batch.push(wrong_shape).is_err());
    let wrong_id = ManagedHostTensor::new(
        id("value.wrong.reference").unwrap(),
        source.tensor().clone(),
        source.bytes().clone(),
    )
    .unwrap();
    assert!(batch.push(wrong_id).is_err());
    batch.push(source).unwrap();
    assert_eq!(
        host.usage().unwrap().committed_bytes(),
        shape.byte_len() * 2
    );
    assert!(batch.finish().is_err());
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    for count in [0, STRUCTURED_ATTENTION_BF16_MAX_BATCH + 1] {
        assert!(ReferenceLatentBatch::new(&host, count, shape).is_err());
    }
    assert!(
        ReferenceLatentBatch::new(&host, 2, QwenImage21LatentShape::new(2, 2, 2).unwrap()).is_err()
    );
}

#[test]
fn processed_reference_geometry_rejects_same_area_different_axes_grid_or_count() {
    let square = ConditionGeometry {
        height: 1024,
        width: 1024,
        grid: [1, 64, 64],
    };
    assert_eq!(
        shared_condition_geometry([square, square], 2).unwrap(),
        square
    );
    for changed in [
        ConditionGeometry {
            height: 512,
            width: 2048,
            ..square
        },
        ConditionGeometry {
            grid: [1, 32, 128],
            ..square
        },
        ConditionGeometry {
            grid: [2, 64, 64],
            ..square
        },
    ] {
        assert!(shared_condition_geometry([square, changed], 2).is_err());
    }
    assert!(shared_condition_geometry([], 1).is_err());
    assert!(shared_condition_geometry([square], 2).is_err());
    assert!(shared_condition_geometry([square, square], 1).is_err());
    assert!(shared_condition_geometry([square], 0).is_err());
}

#[test]
fn editing_identity_batch_order_keeps_distinct_prompts_and_seeds_in_caller_order() {
    let requests = vec![
        ImageGenerationRequest {
            prompt: "long prompt first".into(),
            width: 64,
            height: 64,
            seed: 19,
            n: 1,
        },
        ImageGenerationRequest {
            prompt: "short".into(),
            width: 64,
            height: 64,
            seed: 3,
            n: 1,
        },
    ];
    let order =
        super::super::prepare::BatchOrder::new((0..requests.len()).collect(), requests.len())
            .unwrap();
    let actual: Vec<_> = order
        .requests(&requests)
        .unwrap()
        .map(|r| (r.prompt.as_str(), r.seed))
        .collect();
    assert_eq!(actual, vec![("long prompt first", 19), ("short", 3)]);
}
