use super::*;
use std::sync::Arc;
type F = FastH3VideoDecoderBlocksFamily;
#[cfg(feature = "cuda-cutlass-nvfp4")]
mod cuda;
fn block_limits() -> FastH3VideoDecoderBlocksLimits {
    FastH3VideoDecoderBlocksLimits {
        maximum_rows: 64,
        maximum_blocks: 36,
        maximum_stored_bytes: 1 << 20,
        maximum_metadata_bytes: 65536,
    }
}
fn fixture() -> Fixture {
    let mut f = Fixture::new();
    let prefix = "decoder.transformer_blocks.0";
    for suffix in ["norm1.weight", "norm2.weight"] {
        f.tensors.get_mut(&format!("{prefix}.{suffix}")).unwrap().2 =
            6f32.to_le_bytes().repeat(128);
    }
    for (suffix, number) in [("scale1", 0.25f32), ("scale2", 0.125)] {
        f.tensors.get_mut(&format!("{prefix}.{suffix}")).unwrap().2 =
            number.to_le_bytes().repeat(128);
    }
    for suffix in [
        "attn.to_q",
        "attn.to_k",
        "attn.to_v",
        "attn.to_out.0",
        "ff.net.0.proj",
        "ff.net.2",
    ] {
        let p = format!("{prefix}.{suffix}");
        f.tensors
            .get_mut(&format!("{p}.weight_global_sf"))
            .unwrap()
            .2 = 1f32.to_le_bytes().to_vec();
        let (_, shape, bytes) = f.tensors.get_mut(&format!("{p}.packed_weight")).unwrap();
        bytes.fill(0);
        let k = 2 * shape[1];
        for row in 0..shape[0] {
            if suffix == "ff.net.0.proj" && row >= 256 {
                continue;
            }
            let column = row % k;
            bytes[row * (k / 2) + column / 2] |= 2 << (4 * (column % 2)); // E2M1 1.0
        }
        let (_, _, bias) = f.tensors.get_mut(&format!("{p}.bias")).unwrap();
        let constant = if suffix == "attn.to_out.0" {
            0.5
        } else if suffix == "ff.net.2" {
            0.25
        } else {
            0.0
        };
        for (row, word) in bias.chunks_exact_mut(2).enumerate() {
            let value = if suffix == "ff.net.0.proj" && row >= 256 {
                16.0
            } else {
                constant
            };
            word.copy_from_slice(&half::bf16::from_f32(value).to_bits().to_le_bytes());
        }
    }
    f
}

#[test]
fn complete_decoder_blocks_keep_staged_rounding_and_original_source_roles() {
    let (_directory, p) = fixture().open(limits());
    let package = Arc::new(p.unwrap());
    let account = host(156); // one unit, D64 gamma, six F32 reciprocal scalars
    let (c, source) = FastH3VideoDecoderBlocksSource::prepare(
        package,
        0..1,
        vec![2, 3],
        &account,
        block_limits(),
    )
    .unwrap();
    assert!(c.covers_full_decoder());
    assert_eq!(c.rows().unwrap(), 5);
    assert_eq!(account.usage().unwrap().allocated_bytes(), 156);
    let prepared = TypedFamilyRegistration::new(F::new(block_limits()).unwrap())
        .prepare_with_profile(
            &serde_json::to_value(&c).unwrap(),
            &NumericalProfileId::new(F::PROFILE_ID).unwrap(),
        )
        .unwrap();
    let ModelWeights::Stored(schema) = prepared.weights() else {
        panic!("Stored")
    };
    assert_eq!(schema.components.len(), 30);
    let nodes = &prepared.program().blocks()[0].nodes;
    assert_eq!(
        nodes
            .iter()
            .filter(|n| n.operation_id.as_str() == NVFP4_MATMUL_BF16_OPERATION_ID)
            .count(),
        6
    );
    assert_eq!(
        nodes
            .iter()
            .filter(|n| n.operation_id.as_str() == CHANNEL_BIAS_ADD_BF16_OPERATION_ID)
            .count(),
        6
    );
    assert_eq!(nodes.iter().filter(|n|n.operation_id.as_str()==CHANNEL_SCALED_RESIDUAL_SEPARATE_F32_OPERATION_ID).count(),2);
    assert!(nodes.iter().all(|n| !matches!(
        n.operation_id.as_str(),
        CHANNEL_SCALED_RESIDUAL_F32_OPERATION_ID | FUSED_HEAD_RMS_PARTIAL_ROPE_BF16_OPERATION_ID
    )));
    let ff = nodes
        .iter()
        .find(|n| n.operation_id.as_str() == PACKED_SWIGLU_BF16_OPERATION_ID)
        .unwrap();
    assert_eq!(
        PackedSwiGluBf16Profile::from_attributes(&ff.attributes).unwrap(),
        PackedSwiGluBf16Profile::SiluBf16ThenMultiply
    );
    let attention = nodes
        .iter()
        .find(|n| n.operation_id.as_str() == SEGMENTED_ATTENTION_BF16_OPERATION_ID)
        .unwrap();
    assert_eq!(
        attention.attributes[&AttributeId::new("segment_lengths").unwrap()],
        SemanticValue::Integers(vec![2, 3])
    );
    let specs = c.compile_options(block_limits()).unwrap();
    assert_eq!(
        specs.tensor_specs()[&c.output_id().unwrap()].element_type,
        ElementType::F32
    );
    let mut retained = None;
    for component in &schema.components {
        let p = source.component(component).unwrap();
        assert_eq!(p.bytes().len() as u64, component.physical_bytes().unwrap());
        assert!(p.retained_host_memory().is_some());
        if component
            .id
            .as_str()
            .starts_with("derived.fast_h3.decoder_alpha.")
        {
            assert_eq!(p.bytes(), 1f32.to_le_bytes());
            retained = p.retained_host_memory().cloned();
        }
    }
    drop(source);
    assert_eq!(account.usage().unwrap().allocated_bytes(), 4);
    drop(retained);
    assert_eq!(account.usage().unwrap().allocated_bytes(), 0);
}

#[test]
fn decoder_block_source_rejects_wire_slot_swaps_and_rolls_back_constants() {
    let (_dir, p) = fixture().open(limits());
    let package = Arc::new(p.unwrap());
    let account = host(155);
    assert!(FastH3VideoDecoderBlocksSource::prepare(
        package.clone(),
        0..1,
        vec![2, 3],
        &account,
        block_limits()
    )
    .is_err());
    assert_eq!(account.usage().unwrap().allocated_bytes(), 0);
    let (c, _s) = FastH3VideoDecoderBlocksSource::prepare(
        package.clone(),
        0..1,
        vec![2, 3],
        &host(156),
        block_limits(),
    )
    .unwrap();
    let family = F::new(block_limits()).unwrap();
    let mut wire = serde_json::to_value(c).unwrap();
    wire["blocks"][0]["projections"]
        .as_array_mut()
        .unwrap()
        .swap(0, 1);
    assert!(family.parse_config(&wire).is_err());
    for range in [0..0, 1..2, 0..2] {
        assert!(FastH3VideoDecoderBlocksSource::prepare(
            package.clone(),
            range,
            vec![2, 3],
            &account,
            block_limits()
        )
        .is_err());
    }
    assert!(FastH3VideoDecoderBlocksSource::prepare(
        package.clone(),
        0..1,
        vec![0],
        &account,
        block_limits()
    )
    .is_err());
    assert!(FastH3VideoDecoderBlocksSource::prepare(
        package,
        0..1,
        vec![65],
        &account,
        block_limits()
    )
    .is_err());
    assert_eq!(account.usage().unwrap().allocated_bytes(), 0);
}

#[test]
fn full_block_dyadic_fixture_has_independent_nonzero_attention_and_ffn_answer() {
    // Whole rows have one sign, identical inside each independent sequence.
    // RMS gamma6 produces BF16 +/-6 for all three actual residual magnitudes.
    // NVFP4 groups are exact with scales1 (norm/attention) and16 (FFN).
    for magnitude in [1.0f64, 2.625, 2.375] {
        let normalized = (magnitude / (magnitude * magnitude + 1e-5).sqrt()) * 6.0;
        assert_eq!(half::bf16::from_f64(normalized).to_f32(), 6.0);
    }
    assert_eq!(
        half::bf16::from_f64(16.0 / (1.0 + (-16.0f64).exp())).to_f32(),
        16.0
    );
    for (sign, answer) in [(1f32, 14.625f32), (-1., -14.375)] {
        let attention = sign * 6.; // every value row in this sequence is equal
        let biased = half::bf16::from_f32(attention + 0.5).to_f32();
        let hidden = sign + biased * 0.25;
        let ff = half::bf16::from_f32(sign * 6. * 16. + 0.25).to_f32();
        assert_eq!(ff, sign * 96.);
        assert_eq!(hidden + ff * 0.125, answer);
    }
}
