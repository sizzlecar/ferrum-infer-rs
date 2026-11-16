use super::*;
use std::sync::Arc;
type F = FastH3VideoClipFamily;
#[cfg(feature = "cuda-cutlass-nvfp4")]
mod cuda;
fn transport_host(bytes: usize) -> HostMemoryAccount {
    HostMemoryAccount::new(
        HostMemoryLimits::new(
            bytes,
            HostMemoryClassLimits {
                input_preprocess_bytes: bytes,
                transfer_staging_bytes: 0,
                output_bytes: bytes,
                weight_backing_bytes: 0,
            },
        )
        .unwrap(),
    )
}
fn clip_limits() -> FastH3VideoClipLimits {
    FastH3VideoClipLimits {
        blocks: FastH3VideoDecoderBlocksLimits {
            maximum_rows: 128,
            maximum_blocks: 36,
            maximum_stored_bytes: 1 << 20,
            maximum_metadata_bytes: 1 << 20,
        },
        maximum_plan_nodes: 1200,
        maximum_activation_bytes: 1 << 20,
    }
}

#[test]
fn full_clip_composes_original_outer_rounding_full_depth_and_checked_sources() {
    let mut f = Fixture::new();
    f.tensors.get_mut("post_quant_conv.weight").unwrap().2 =
        [1.00390625f32, 1.01171875, -0., f32::from_bits(1)]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect();
    let (_dir, package) = f.open(limits());
    let account = host(1 << 20);
    let (c, source) = FastH3VideoClipSource::prepare(
        Arc::new(package.unwrap()),
        [2, 2, 1, 2],
        &account,
        clip_limits(),
    )
    .unwrap();
    assert_eq!(
        (
            c.patch_rows().unwrap(),
            c.sequence_rows().unwrap(),
            c.rows().unwrap()
        ),
        (8, 9, 18)
    );
    let prepared = TypedFamilyRegistration::new(F::new(clip_limits()).unwrap())
        .prepare_with_profile(
            &serde_json::to_value(&c).unwrap(),
            &NumericalProfileId::new(F::PROFILE_ID).unwrap(),
        )
        .unwrap();
    let nodes = &prepared.program().blocks()[0].nodes;
    assert_eq!(
        nodes[..6]
            .iter()
            .map(|n| n.operation_id.as_str())
            .collect::<Vec<_>>(),
        [
            CAST_F32_TO_BF16_OPERATION_ID,
            DENSE_LINEAR_BF16_OPERATION_ID,
            CHANNEL_BIAS_ADD_BF16_OPERATION_ID,
            AFFINE_LINEAR_BF16_OPERATION_ID,
            CAST_BF16_TO_F32_OPERATION_ID,
            BATCHED_SUFFIX_APPEND_F32_OPERATION_ID
        ]
    );
    assert_eq!(
        nodes[nodes.len() - 4].operation_id.as_str(),
        LAYER_NORM_F32_OPERATION_ID
    );
    let ModelWeights::Stored(schema) = prepared.weights() else {
        panic!("Stored")
    };
    assert_eq!(schema.components.len(), 39);
    let component = schema
        .components
        .iter()
        .find(|c| c.external_names == ["post_quant_conv.weight"])
        .unwrap();
    assert_eq!(component.dimensions, [2, 2]);
    assert_eq!(component.physical_element_type(), ElementType::Bf16);
    let before = account.usage().unwrap().allocated_bytes();
    let first = source.component(component).unwrap();
    let second = source.component(component).unwrap();
    assert_eq!(first.bytes().as_ptr(), second.bytes().as_ptr());
    assert!(first.retained_host_memory().is_some());
    assert_eq!(first.bytes(), [0x80, 0x3f, 0x82, 0x3f, 0, 0x80, 0, 0]);
    for c in &schema.components {
        assert_eq!(
            source.component(c).unwrap().bytes().len() as u64,
            c.physical_bytes().unwrap()
        );
    }
    assert_eq!(account.usage().unwrap().allocated_bytes(), before);
    let retained = first.retained_host_memory().unwrap().clone();
    drop((first, second));
    drop(source);
    assert_eq!(account.usage().unwrap().allocated_bytes(), 8);
    assert_eq!(retained.bytes().len(), 8);
    drop(retained);
    assert_eq!(account.usage().unwrap().allocated_bytes(), 0);

    for field in ["decoder_num_layers", "decoder_num_attention_heads"] {
        let mut wire = serde_json::to_value(&c).unwrap();
        wire["architecture"][field] = serde_json::json!(2 + config_json()[field].as_u64().unwrap());
        assert!(F::new(clip_limits()).unwrap().parse_config(&wire).is_err());
    }
    let mut wire = serde_json::to_value(&c).unwrap();
    wire["outer"][0]["component"]["dimensions"] = serde_json::json!([1, 4]);
    assert!(F::new(clip_limits()).unwrap().parse_config(&wire).is_err());
    let tiny = FastH3VideoClipLimits {
        maximum_plan_nodes: nodes.len() - 1,
        ..clip_limits()
    };
    assert!(c.compile_options(tiny).is_err());
    assert!(c
        .compile_options(FastH3VideoClipLimits {
            maximum_plan_nodes: nodes.len(),
            ..clip_limits()
        })
        .is_ok());
}

#[test]
fn clip_transport_preserves_ncthw_axes_and_rotary_suffix_with_admitted_rollback() {
    let (_dir, package) = Fixture::new().open(limits());
    let weights = host(1 << 20);
    let (c, _source) = FastH3VideoClipSource::prepare(
        Arc::new(package.unwrap()),
        [2, 2, 1, 2],
        &weights,
        clip_limits(),
    )
    .unwrap();
    let inputs = transport_host(1 << 20);
    let dense: Vec<f32> = (0..16).map(|n| n as f32).collect();
    let packed = c
        .managed_input_from_decoder_ncthw(&dense, &inputs, clip_limits())
        .unwrap();
    let expected = [
        0f32, 4., 1., 5., 2., 6., 3., 7., 8., 12., 9., 13., 10., 14., 11., 15.,
    ];
    assert_eq!(
        packed.bytes().bytes(),
        expected
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>()
    );
    let (cos, sin) = c.managed_bf16_frequencies(&inputs, clip_limits()).unwrap();
    assert_eq!(cos.tensor().dimensions(), [18, 48]);
    for base in [0usize, 9] {
        for row in 4..9 {
            for j in 0..48 {
                let i = 2 * ((base + row) * 48 + j);
                assert_eq!(&cos.bytes().bytes()[i..i + 2], &0x3f80u16.to_le_bytes());
                assert_eq!(&sin.bytes().bytes()[i..i + 2], &0u16.to_le_bytes());
            }
        }
    }
    assert_eq!(
        &cos.bytes().bytes()[..9 * 48 * 2],
        &cos.bytes().bytes()[9 * 48 * 2..]
    );
    let too_small = transport_host(18 * 48 * 2); // admits cosine, refuses sine; cosine rolls back
    assert!(c
        .managed_bf16_frequencies(&too_small, clip_limits())
        .is_err());
    assert_eq!(too_small.usage().unwrap().allocated_bytes(), 0);
    assert!(c
        .managed_input_from_decoder_ncthw(&dense[..15], &inputs, clip_limits())
        .is_err());
    drop((packed, cos, sin));
    assert_eq!(inputs.usage().unwrap().allocated_bytes(), 0);
}

#[test]
fn clip_source_admission_failure_releases_earlier_block_and_outer_backing() {
    let (_dir, package) = Fixture::new().open(limits());
    let account = host(200); // block constants156, postweights8+bias4 then projection refuses
    assert!(FastH3VideoClipSource::prepare(
        Arc::new(package.unwrap()),
        [1, 1, 1, 1],
        &account,
        clip_limits()
    )
    .is_err());
    assert_eq!(account.usage().unwrap().allocated_bytes(), 0);
}

#[test]
fn grouped_clip_unpack_copies_complete_batch_items_without_pinning_other_tiles() {
    let (_dir, package) = Fixture::new().open(limits());
    let (config, _source) = FastH3VideoClipSource::prepare(
        Arc::new(package.unwrap()),
        [4, 2, 1, 2],
        &host(1 << 20),
        clip_limits(),
    )
    .unwrap();
    let shape =
        ferrum_diffusion::SpatiotemporalLatentShape::new([4, 3, 8, 2, 4], [4, 2, 2]).unwrap();
    let dense: Vec<u16> = (0..shape.element_count())
        .map(|i| half::bf16::from_f32(i as f32 / 16.).to_bits())
        .collect();
    let mut packed = vec![0u16; dense.len()];
    ferrum_diffusion::pack_spatiotemporal_into(shape, &dense, &mut packed).unwrap();
    let bytes: Vec<u8> = packed.into_iter().flat_map(u16::to_le_bytes).collect();
    let item_bytes = bytes.len() / 4;
    let host = transport_host(item_bytes);
    for item in 0..4 {
        let raw = config
            .unpack_bytes_batch_range(&bytes, item..item + 1, &host, clip_limits())
            .unwrap();
        assert_eq!(raw.ncthw_shape(), [1, 3, 8, 2, 4]);
        let expected: Vec<u8> = dense[item * item_bytes / 2..(item + 1) * item_bytes / 2]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        assert_eq!(raw.bytes().bytes(), expected);
        assert_eq!(host.usage().unwrap().committed_bytes(), item_bytes);
        assert!(config
            .unpack_bytes_batch_range(&bytes, item..item + 1, &host, clip_limits())
            .is_err());
        drop(raw);
        assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    }
    // Empty, out-of-bounds and inverted inputs are rejection fixtures.
    #[allow(clippy::reversed_empty_ranges)]
    for range in [0..0, 4..5, 3..2] {
        assert!(config
            .unpack_bytes_batch_range(&bytes, range, &host, clip_limits())
            .is_err());
    }
    assert!(config
        .unpack_bytes_batch_range(&bytes, 0..1, &transport_host(item_bytes - 1), clip_limits())
        .is_err());
}
