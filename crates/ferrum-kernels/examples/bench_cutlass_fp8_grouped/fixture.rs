//! Finite, deterministic, branch-dependent fixtures; no checkpoint payloads.
use half::bf16;

fn mix(mut value: u64) -> u64 {
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58476d1ce4e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d049bb133111eb);
    value ^ (value >> 31)
}

pub fn input(m: u32, k: u32) -> Vec<u8> {
    let mut bytes = vec![0; m as usize * k as usize * 2];
    for (index, dst) in bytes.chunks_exact_mut(2).enumerate() {
        let row = index / k as usize;
        let column = index % k as usize;
        let amplitude = [1.0 / 64.0, 1.0 / 8.0, 1.0, 4.0][row % 4];
        let value = if row % 257 == 0 {
            0.0
        } else {
            let integer = (mix((row as u64) << 32 | column as u64) % 2049) as i32 - 1024;
            amplitude * integer as f32 / 128.0
        };
        dst.copy_from_slice(&bf16::from_f32(value).to_bits().to_le_bytes());
    }
    bytes
}

pub fn weight(n: u32, k: u32, branch: usize) -> Vec<u8> {
    let mut bytes = vec![0; n as usize * k as usize];
    // Exact finite E4M3 encodings, positive/negative and several magnitudes.
    const VALUES: [u8; 16] = [
        0x00, 0x08, 0x10, 0x18, 0x20, 0x28, 0x30, 0x38, 0x80, 0x88, 0x90, 0x98, 0xa0, 0xa8, 0xb0,
        0xb8,
    ];
    for (index, value) in bytes.iter_mut().enumerate() {
        let h = mix((index as u64).wrapping_add((branch as u64 + 1) * 0x1_0000_0001));
        *value = VALUES[h as usize % VALUES.len()];
    }
    bytes
}

pub fn scale(branch: usize) -> [u8; 4] {
    ((branch as f32 + 1.0) / 128.0).to_le_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn finite_fixture_has_distinct_branches_and_zero_row() {
        let input = input(8, 32);
        assert!(input[..64].iter().all(|&x| x == 0));
        let values = input
            .chunks_exact(2)
            .map(|v| bf16::from_bits(u16::from_le_bytes([v[0], v[1]])).to_f32())
            .collect::<Vec<_>>();
        assert!(values.iter().all(|v| v.is_finite()));
        assert!(values.iter().any(|&v| v < 0.0));
        assert!(values.iter().any(|&v| v > 0.0));
        assert_ne!(weight(16, 32, 0), weight(16, 32, 1));
        assert_ne!(scale(0), scale(1));
    }
}
