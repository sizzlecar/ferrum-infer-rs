//! Test-only independent byte oracle. No production decoder or ideal Q8 codes.
use half::f16;

pub(crate) fn weight(block: &[u8], index: usize) -> f64 {
    assert_eq!(block.len(), 210);
    assert!(index < 256);
    let half = index / 128;
    let quarter = index % 128 / 32;
    let lane = index % 32;
    let low = (block[half * 64 + (quarter & 1) * 32 + lane] >> (4 * (quarter / 2))) & 15;
    let high = (block[128 + half * 32 + lane] >> (2 * quarter)) & 3;
    let code = i32::from(low | high << 4) - 32;
    let scale = block[192 + index / 16] as i8;
    let d = f16::from_bits(u16::from_le_bytes([block[208], block[209]])).to_f64();
    d * f64::from(scale) * f64::from(code)
}

pub(crate) fn activation(packed: &[u8], rows: usize, row: usize, column: usize) -> f64 {
    assert!(row < rows);
    let block = ((column / 128) * rows + row) * 144;
    let group = column % 128 / 32;
    let d = f32::from_le_bytes(
        packed[block + 4 * group..block + 4 * group + 4]
            .try_into()
            .unwrap(),
    );
    let q = packed[block + 16 + column % 128] as i8;
    f64::from(d) * f64::from(q)
}

/// (actual-pack target, expanded magnitude, original F32 target, original magnitude,
///  conservative measured activation-quantization error bound).
pub(crate) fn dot(w: &[u8], x: &[f32], packed: &[u8], rows: usize, row: usize) -> [f64; 5] {
    assert_eq!(x.len() % 256, 0);
    assert_eq!(w.len(), x.len() / 256 * 210);
    let mut out = [0.0; 5];
    for (i, &original) in x.iter().enumerate() {
        let a = activation(packed, rows, row, i);
        let b = weight(&w[i / 256 * 210..(i / 256 + 1) * 210], i % 256);
        out[0] += a * b;
        out[1] += (a * b).abs();
        out[2] += f64::from(original) * b;
        out[3] += (f64::from(original) * b).abs();
        out[4] += ((a - f64::from(original)) * b).abs();
    }
    out
}

#[test]
fn q6_literal_signed_codes_scales_and_half_metadata() {
    let mut b = [0u8; 210];
    b[192..208].fill(1);
    b[208..].copy_from_slice(&f16::from_f32(0.5).to_bits().to_le_bytes());
    assert!((0..256).all(|i| weight(&b, i) == -16.0));
    b[..192].fill(255);
    b[193] = (-2i8) as u8;
    assert_eq!(weight(&b, 0), 15.5);
    assert_eq!(weight(&b, 16), -31.0);
    assert_eq!(weight(&b, 255), 15.5);
    b[208..].copy_from_slice(&1u16.to_le_bytes());
    assert_eq!(weight(&b, 0), 31.0 * 2f64.powi(-24));
    b[208..].copy_from_slice(&0x7c00u16.to_le_bytes());
    assert!(!weight(&b, 0).is_finite());
}

#[test]
fn observed_d4_is_k_block_major_and_has_no_affine_sum() {
    let mut p = vec![0u8; 2 * 2 * 144];
    for b in 0..4 {
        for g in 0..4 {
            p[b * 144 + g * 4..b * 144 + g * 4 + 4]
                .copy_from_slice(&((b + g + 1) as f32).to_le_bytes());
        }
        p[b * 144 + 16..(b + 1) * 144].fill((-3i8) as u8);
    }
    assert_eq!(activation(&p, 2, 0, 0), -3.0);
    assert_eq!(activation(&p, 2, 1, 0), -6.0);
    assert_eq!(activation(&p, 2, 0, 128), -9.0);
    assert_eq!(activation(&p, 2, 1, 224), -21.0);
}
