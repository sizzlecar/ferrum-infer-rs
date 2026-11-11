use serde::Serialize;

/// The actual B1 and rejected B3 packed row counts in the sealed V17 trace.
/// The other dimensions are the unchanged H3 MLP fc1 projection.
#[derive(Clone, Copy, Debug, Serialize)]
#[repr(C)]
pub struct Shape {
    pub rows: u32,
    pub outputs: u32,
    pub inputs: u32,
    pub reserved: u32,
}
impl Shape {
    pub const fn fc1(rows: u32) -> Self {
        Self {
            rows,
            outputs: 28672,
            inputs: 5376,
            reserved: 0,
        }
    }
    pub fn output_elements(self) -> usize {
        self.rows as usize * self.outputs as usize
    }
}

/// Hash each coordinate independently; no repeated-row shortcut in the oracle.
fn mixed(mut x: u64) -> u64 {
    x ^= x >> 30;
    x = x.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}
pub fn activation(index: usize) -> i8 {
    (mixed(index as u64 ^ 0x37ab_c981) as u8) as i8
}
pub fn weight(index: usize) -> i8 {
    (mixed(index as u64 ^ 0x09e1_7243) as u8) as i8
}
pub fn row_scale(row: usize) -> f32 {
    let mantissa = 1.0 + (mixed(row as u64 ^ 17) % 4096) as f32 / 4096.0;
    mantissa * 2.0_f32.powi((row % 11) as i32 - 14)
}
pub fn column_scale(column: usize) -> f32 {
    let mantissa = 1.0 + (mixed(column as u64 ^ 91) % 4096) as f32 / 4096.0;
    mantissa * 2.0_f32.powi((column % 13) as i32 - 13)
}
pub fn reference(shape: Shape, row: usize, column: usize) -> (i32, u16) {
    let k = shape.inputs as usize;
    let dot: i64 = (0..k)
        .map(|i| i64::from(activation(row * k + i)) * i64::from(weight(column * k + i)))
        .sum();
    let dot = i32::try_from(dot).expect("declared K excludes even worst-case I8 overflow");
    let row_scaled = dot as f32 * row_scale(row);
    let value = row_scaled * column_scale(column);
    (dot, half::bf16::from_f32(value).to_bits())
}
pub fn samples(shape: Shape) -> Vec<(usize, usize)> {
    let m = shape.rows as usize;
    let n = shape.outputs as usize;
    let mut result = vec![(0, 0), (0, n - 1), (m - 1, 0), (m - 1, n - 1)];
    result.extend((0..28).map(|i| (mixed(i ^ 0xb1) as usize % m, mixed(i ^ 0xb3) as usize % n)));
    result.sort_unstable();
    result.dedup();
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn actual_fc1_geometry_covers_tail_rows_and_exact_scratch_savings() {
        for (m, bytes) in [(6296, 722075648_u64), (19389, 2223685632)] {
            let shape = Shape::fc1(m);
            assert_eq!(shape.output_elements() as u64 * 4, bytes);
            assert!(u64::from(shape.inputs) * 128 * 128 < i32::MAX as u64);
            let indices = samples(shape);
            assert!(indices.contains(&(m as usize - 1, 28671)));
            for (row, column) in indices {
                let (_, value) = reference(shape, row, column);
                assert!(half::bf16::from_bits(value).is_finite());
            }
        }
    }

    #[test]
    fn integer_fixture_has_full_signed_range_and_nonconstant_scales() {
        let values = (0..4096)
            .map(activation)
            .collect::<std::collections::BTreeSet<_>>();
        assert!(values.contains(&i8::MIN) && values.contains(&i8::MAX));
        assert_ne!(row_scale(0), row_scale(1));
        assert_ne!(column_scale(0), column_scale(1));
        let (dot, value) = reference(Shape::fc1(6296), 17, 31);
        let expected = half::bf16::from_f32((dot as f32 * row_scale(17)) * column_scale(31));
        assert_eq!(value, expected.to_bits());
    }
}
