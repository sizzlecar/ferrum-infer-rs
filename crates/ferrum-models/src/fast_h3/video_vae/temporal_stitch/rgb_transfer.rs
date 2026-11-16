//! Exact finite BF16 input transfer, tabulated at compile time. The original
//! F32 multiply/add/clamp/truncate recipe has only 65536 possible input words.
//! This is 192 KiB of immutable code data, with no lazy allocation or per-image
//! cache. The caller rejects nonfinite words before lookup.
pub(super) static TABLE: [[u8; 65536]; 3] = table();

const fn table() -> [[u8; 65536]; 3] {
    let mut result = [[0; 65536]; 3];
    let mut channel = 0;
    while channel < 3 {
        let mut word = 0;
        while word < 65536 {
            if word & 0x7f80 != 0x7f80 {
                let x = f32::from_bits((word as u32) << 16);
                let scaled = x * [0.229f32, 0.224, 0.225][channel];
                let shifted = scaled + [0.485f32, 0.456, 0.406][channel];
                result[channel][word] = if shifted <= 0. {
                    0
                } else if shifted >= 1. {
                    255
                } else {
                    (shifted * 255f32) as u8
                };
            }
            word += 1;
        }
        channel += 1;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finite_bf16_transfer_matches_independently_staged_f64_oracle() {
        for (channel, (scale, offset)) in [
            (0.229f32, 0.485f32),
            (0.224f32, 0.456f32),
            (0.225f32, 0.406f32),
        ]
        .into_iter()
        .enumerate()
        {
            for word in 0..=u16::MAX {
                let input = f32::from_bits(u32::from(word) << 16);
                if !input.is_finite() {
                    continue;
                }
                // Compute in F64 and cast at each original F32 boundary;
                // never replace the two stages with a fused multiply/add.
                let scaled = (f64::from(input) * f64::from(scale)) as f32;
                let shifted = (f64::from(scaled) + f64::from(offset)) as f32;
                let clamped = f64::from(shifted).clamp(0., 1.);
                let byte = (clamped * 255.) as f32 as u8;
                assert_eq!(
                    TABLE[channel][usize::from(word)],
                    byte,
                    "{channel}/{word:04x}"
                );
            }
        }
    }
}
