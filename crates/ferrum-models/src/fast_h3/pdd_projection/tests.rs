use super::*;

fn original(
    dtype: ElementType,
    heads: usize,
    width: usize,
    values: &[f32],
) -> (WeightComponentSpec, Vec<u8>) {
    assert_eq!(values.len(), heads * width);
    let component = WeightComponentSpec {
        id: WeightId::new("source.pdd.weight").unwrap(),
        role: WeightComponentRole::Values,
        external_names: vec!["proj_out.weight".into()],
        dimensions: vec![heads as u64, width as u64],
        encoding: WeightEncoding::Dense {
            element_type: dtype,
        },
        required: true,
    };
    let mut bytes = Vec::new();
    for &value in values {
        match dtype {
            ElementType::F32 => bytes.extend(value.to_le_bytes()),
            ElementType::F16 => bytes.extend(f16::from_f32(value).to_bits().to_le_bytes()),
            ElementType::Bf16 => bytes.extend(bf16::from_f32(value).to_bits().to_le_bytes()),
            _ => panic!("floating fixture"),
        }
    }
    (component, bytes)
}

#[test]
fn fusion_selects_channel_major_heads_and_keeps_f32_products_separate() {
    // The first/last head are sentinels and must not enter the selected block.
    for dtype in [ElementType::F32, ElementType::F16, ElementType::Bf16] {
        let (c, bytes) = original(
            dtype,
            4,
            3,
            &[
                4096., 4096., 4096., 2., -8., 4., 10., 4., -12., -4096., -4096., -4096.,
            ],
        );
        let p = WeightComponentPayload::new(
            &c,
            "proj_out.weight",
            "weights.safetensors",
            c.dimensions.clone(),
            dtype,
            bytes,
        )
        .unwrap();
        let mut output = [0u8; 12];
        fuse_parameter(
            &p,
            &(1..3),
            &[0.25f32.to_bits(), 0.75f32.to_bits()],
            &mut output,
        )
        .unwrap();
        let values: Vec<_> = output
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        assert_eq!(values, [8., 1., -8.]);
    }
    // Exact cancellation after rounded products differs from a fused FMA.
    // (1+2^-23)*(1-2^-23) rounds to 1, so the declared profile emits +0.
    let coefficient = f32::from_bits(0x3f7ffffe);
    let value = f32::from_bits(0x3f800001);
    let (c, bytes) = original(ElementType::F32, 2, 1, &[-1., value]);
    let p = WeightComponentPayload::new(
        &c,
        "proj_out.weight",
        "weights.safetensors",
        c.dimensions.clone(),
        ElementType::F32,
        bytes,
    )
    .unwrap();
    let mut out = [0u8; 4];
    fuse_parameter(
        &p,
        &(0..2),
        &[1f32.to_bits(), coefficient.to_bits()],
        &mut out,
    )
    .unwrap();
    assert_eq!(f32::from_le_bytes(out).to_bits(), 0f32.to_bits());
    assert_ne!(value.mul_add(coefficient, -1.), 0.);
}

#[test]
fn fusion_rejects_nonfinite_results_and_malformed_extents() {
    for values in [
        vec![1., f32::NAN],
        vec![1., f32::INFINITY],
        vec![f32::MAX, f32::MAX],
    ] {
        let (c, bytes) = original(ElementType::F32, 2, 1, &values);
        let p = WeightComponentPayload::new(
            &c,
            "proj_out.weight",
            "weights.safetensors",
            c.dimensions.clone(),
            ElementType::F32,
            bytes,
        )
        .unwrap();
        assert!(fuse_parameter(&p, &(0..2), &[1f32.to_bits(); 2], &mut [0u8; 4]).is_err());
    }
    let (c, bytes) = original(ElementType::Bf16, 2, 1, &[1., 2.]);
    let p = WeightComponentPayload::new(
        &c,
        "proj_out.weight",
        "weights.safetensors",
        c.dimensions.clone(),
        ElementType::Bf16,
        bytes,
    )
    .unwrap();
    assert!(fuse_parameter(&p, &(0..3), &[1f32.to_bits(); 3], &mut [0u8; 4]).is_err());
    assert!(fuse_parameter(&p, &(1..2), &[], &mut [0u8; 4]).is_err());
    assert!(fuse_parameter(&p, &(0..2), &[1f32.to_bits(); 2], &mut [0u8; 3]).is_err());
    assert!(fuse_parameter(&p, &(0..2), &[1f32.to_bits(); 2], &mut []).is_err());
}
