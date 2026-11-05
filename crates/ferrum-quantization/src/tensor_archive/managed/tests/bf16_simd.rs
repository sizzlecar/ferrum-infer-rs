use super::*;

fn raw(name: &'static str, rows: usize, words: &[u16]) -> Tensor {
    Tensor {
        name,
        shape: vec![rows, 8],
        dtype: Dtype::BF16,
        bytes: words.iter().flat_map(|v| v.to_le_bytes()).collect(),
    }
}
fn oracle_bytes(input: &[u8]) -> Vec<u8> {
    input
        .chunks_exact(2)
        .flat_map(|word| {
            f16::from_f32(bf16::from_le_bytes(word.try_into().unwrap()).to_f32()).to_le_bytes()
        })
        .collect()
}

#[test]
fn bf16_to_f16_simd_managed_single_mlp_and_gdn_have_one_exact_backing_and_retry() {
    let a = [
        0x3f80u16, 0xbf80, 0, 0x8000, 0x3880, 0x477f, 0x3e80, 0xbe80, 0x3f00, 0xbf00, 0x4000,
        0xc000, 0x0001, 0x387f, 0x7f80, 0x7fc1,
    ];
    let b = [0x3f00u16; 16];
    let c = [0xbf00u16; 8];
    let d = [0x4000u16; 8];
    let (_directory, original) = archive(&[
        raw("a", 2, &a),
        raw("b", 2, &b),
        raw("c", 1, &c),
        raw("d", 1, &d),
    ]);
    let h = host();
    for (names, rows) in [
        (vec!["a"], 2),
        (vec!["b", "a"], 4),
        (vec!["d", "a", "c", "b"], 6),
    ] {
        let component = component(&names, &[rows, 8], ElementType::F16);
        let expected = names
            .iter()
            .flat_map(|name| oracle_bytes(original.tensor(name).unwrap().bytes()))
            .collect::<Vec<_>>();
        let physical = h.backing_bytes(expected.len()).unwrap();
        let short = h
            .reserve_scope(&[(HostMemoryClass::WeightBacking, physical - 1)])
            .unwrap();
        assert!(ManagedDenseArchiveSource::new(&original, &short.account())
            .component(&component)
            .is_err());
        assert_eq!(h.usage().unwrap().allocated_bytes(), 0);
        short.close().unwrap();
        let exact = h
            .reserve_scope(&[(HostMemoryClass::WeightBacking, physical)])
            .unwrap();
        let actual_host = exact.account();
        let source = ManagedDenseArchiveSource::new(&original, &actual_host);
        let payload = source.component(&component).unwrap();
        assert_eq!(payload.bytes(), expected);
        assert_eq!(h.usage().unwrap().allocated_bytes(), physical);
        assert_eq!(payload.external_names(), names);
        let retained = payload.retained_host_memory().unwrap().clone();
        exact.close().unwrap();
        assert!(source.component(&component).is_err());
        drop(payload);
        assert_eq!(h.usage().unwrap().committed_bytes(), physical);
        drop(retained);
        assert_eq!(h.usage().unwrap().committed_bytes(), 0);
    }
}

#[test]
fn bf16_to_f16_simd_late_finite_overflow_refunds_and_preserves_exact_scope_retry() {
    let mut invalid = [0x3f80u16; 32];
    invalid[31] = 0x4780; // Finite BF16 65536 cannot be represented as finite F16.
    let (_directory, original) =
        archive(&[raw("invalid", 4, &invalid), raw("valid", 4, &[0x3f80; 32])]);
    let h = host();
    let physical = h.backing_bytes(64).unwrap();
    let scope = h
        .reserve_scope(&[(HostMemoryClass::WeightBacking, physical)])
        .unwrap();
    let view = scope.account();
    let source = ManagedDenseArchiveSource::new(&original, &view);
    let invalid = component(&["invalid"], &[4, 8], ElementType::F16);
    let error = source
        .component(&invalid)
        .err()
        .expect("finite overflow must reject the whole payload");
    assert!(error.to_string().contains("finite dense value overflows"));
    assert_eq!(h.usage().unwrap().allocated_bytes(), 0);
    let valid = component(&["valid"], &[4, 8], ElementType::F16);
    let payload = source.component(&valid).unwrap();
    assert_eq!(payload.bytes(), f16_bytes(&[1.; 32]));
    assert_eq!(h.usage().unwrap().allocated_bytes(), physical);
    drop(payload);
    scope.close().unwrap();
    assert_eq!(h.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn bf16_to_f16_simd_never_bypasses_affine_or_its_f64_then_f32_rounding() {
    let (_directory, original) = archive(&[raw("ones", 1, &[0x3f80; 8])]);
    let h = host();
    let source = ManagedDenseArchiveSource::new(&original, &h);
    for (bias, expected) in [
        ((3i64, 2048u64), 0x3c02u16),
        // Just above an F16 halfway boundary; the contract first rounds to
        // F32's exact halfway value, then rounds F16 ties to even (1.0).
        (((1i64 << 29) + 1, 1u64 << 40), 0x3c00u16),
    ] {
        let mut c = component(&["ones"], &[1, 8], ElementType::F16);
        c.encoding = WeightEncoding::DenseAffine {
            element_type: ElementType::F16,
            scale: CanonicalRational::new(1, 1).unwrap(),
            bias: CanonicalRational::new(bias.0, bias.1).unwrap(),
        };
        let payload = source.component(&c).unwrap();
        assert_eq!(payload.bytes(), expected.to_le_bytes().repeat(8));
        assert_eq!(payload.bytes(), original.component(&c).unwrap().bytes());
        assert_eq!(
            h.usage().unwrap().allocated_bytes(),
            h.backing_bytes(16).unwrap()
        );
        drop(payload);
        assert_eq!(h.usage().unwrap().committed_bytes(), 0);
    }
    // Do not use half::from_f64 as an independent single-rounding oracle: its
    // selected platform/fallback conversion may itself discard low F64 bits.
    // Exact dyadic distances to the adjacent F16 values prove which side of
    // the F16 midpoint the original F64 result occupies.
    let lower = 1.0f64;
    let upper = 1.0 + 1.0 / 1024.0; // F16 bits 0x3c01.
    let midpoint = 1.0 + 1.0 / 2048.0;
    let value = midpoint + 1.0 / 1099511627776.0;
    assert!(value > midpoint);
    assert!(
        upper - value < value - lower,
        "single rounding is closer to 0x3c01"
    );
    assert_eq!(f64::from(value as f32), midpoint);
    assert_eq!(
        f16::from_f32(value as f32).to_bits(),
        0x3c00,
        "the declared F64→F32→F16 path rounds the F32 midpoint to even"
    );
}
