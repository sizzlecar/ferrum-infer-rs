use super::*;
use std::ops::Range;

fn rows(precision: TokenPrecision, count: usize) -> Vec<NativeProjectionRow> {
    let bytes = precision.element().size_bytes();
    (0..count)
        .map(|index| NativeProjectionRow {
            input: 0x10000 + index as u64 * 33 * bytes,
            input_bytes: 33 * bytes,
            output: 0x100000 + index as u64 * 11 * bytes,
            output_bytes: 11 * bytes,
        })
        .collect()
}

fn select(
    precision: TokenPrecision,
    rows: &[NativeProjectionRow],
    spans: &[Range<u64>],
) -> Option<u32> {
    packed_projection_rows(
        precision,
        true,
        33,
        11,
        &[part(MatrixFormat::DenseF16, 11, 33)],
        spans.iter().cloned(),
        rows.iter().copied(),
        &[0x200000..0x201000],
    )
}

#[test]
fn native_projection_packing_requires_exact_physical_rows_and_nonaliasing() {
    let spans = [0..1, 1..2, 2..3];
    for precision in [TokenPrecision::F16, TokenPrecision::F32] {
        let valid = rows(precision, 3);
        assert_eq!(select(precision, &valid, &spans), Some(3));
        let mut invalid = valid.clone();
        invalid[1].input += precision.element().size_bytes();
        assert_eq!(select(precision, &invalid, &spans), None);
        invalid = valid.clone();
        invalid[1].output += precision.element().size_bytes();
        assert_eq!(select(precision, &invalid, &spans), None);
        invalid = valid.clone();
        invalid.swap(0, 1);
        assert_eq!(select(precision, &invalid, &spans), None);
        invalid = valid.clone();
        invalid[0].input_bytes -= 1;
        assert_eq!(select(precision, &invalid, &spans), None);
        invalid = valid.clone();
        invalid[0].output_bytes += 1;
        assert_eq!(select(precision, &invalid, &spans), None);
        invalid = valid.clone();
        invalid[0].input = 0;
        assert_eq!(select(precision, &invalid, &spans), None);
        invalid = valid.clone();
        invalid[0].output |= 1;
        assert_eq!(select(precision, &invalid, &spans), None);
        invalid = valid.clone();
        invalid[0].input = u64::MAX - 3;
        assert_eq!(select(precision, &invalid, &spans), None);
        for start in [valid[0].input, 0x200000] {
            invalid = valid.clone();
            for (index, row) in invalid.iter_mut().enumerate() {
                row.output = start + index as u64 * row.output_bytes;
            }
            assert_eq!(select(precision, &invalid, &spans), None);
        }
        assert_eq!(select(precision, &valid, &[0..2, 2..3, 3..6]), None);
        assert_eq!(select(precision, &valid, &[2..3, 0..1, 1..2]), None);
        assert_eq!(select(precision, &valid[..2], &spans), None);
        assert_eq!(select(precision, &valid[..1], &spans[..1]), None);
    }
}

#[test]
fn native_projection_packing_preserves_capacity_and_transform_fallback() {
    use ferrum_interfaces::vnext::{
        HadamardApplication, HadamardSigns, HadamardTransformSpec, PhysicalWeightComponentBinding,
    };
    let table = part(MatrixFormat::DenseF16, 11, 33);
    let capacity = |count: usize, input_packed, parts: &[MatrixPart]| {
        packed_projection_rows(
            TokenPrecision::F32,
            input_packed,
            33,
            11,
            parts,
            (0..count).map(|i| i as u64..i as u64 + 1),
            (0..count).map(|i| NativeProjectionRow {
                input: 0x10000 + i as u64 * 132,
                input_bytes: 132,
                output: 0x10000000 + i as u64 * 44,
                output_bytes: 44,
            }),
            &[0x20000000..0x20001000],
        )
    };
    assert_eq!(
        capacity(65535, true, std::slice::from_ref(&table)),
        Some(65535)
    );
    assert_eq!(capacity(65536, true, std::slice::from_ref(&table)), None);
    assert_eq!(capacity(8, false, std::slice::from_ref(&table)), None);
    let transformed = MatrixPart {
        transform: Some(HadamardTransformSpec {
            block_size: std::num::NonZeroU32::new(1).unwrap(),
            signs: HadamardSigns::Explicit(PhysicalWeightComponentBinding::exact_contiguous(
                WeightId::new("component.shared-signs").unwrap(),
            )),
            application: HadamardApplication::BeforeMatmul {
                input_permutation: None,
            },
        }),
        ..table
    };
    assert_eq!(capacity(8, true, &[transformed]), None);
}
