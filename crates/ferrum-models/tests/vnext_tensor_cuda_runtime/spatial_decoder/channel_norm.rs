//! Independent replay of the original level-by-level padded FP32 tree. This
//! deliberately does not use the new kernel's binary-carry accumulation.
use super::*;
use half::bf16;

fn input_bits(channel: usize, position: usize, channels: usize) -> u16 {
    let sign = if (channel + position) % 2 == 0 {
        0
    } else {
        0x8000
    };
    match position % 13 {
        0 => sign,
        1 => 0x3f80 | sign,
        2 => {
            if channel < 4 {
                0x3f81 | sign
            } else {
                sign
            }
        }
        3 => {
            if channel == 0 {
                0x7f80
            } else {
                0x3f80 | sign
            }
        }
        4 => {
            if channel + 1 == channels {
                0x7fc1
            } else {
                0x3f80 | sign
            }
        }
        5 => 0x0001 | sign,
        6 => {
            if channel == 0 {
                0x7f7f
            } else {
                0x3f80 | sign
            }
        }
        // Square leaves [2^24,1,1,1,0,...] discriminate adjacent and linear
        // reduction before sqrt: the original tree is exactly 2^24 + 2.
        7 => match channel {
            0 => 0x4580,
            1..=3 => 0x3f80,
            _ => 0,
        },
        8 => {
            if channel == 0 {
                0x4580
            } else {
                0x3f80
            }
        }
        9 => {
            if channel + 1 == channels {
                0x4040 | sign
            } else {
                sign
            }
        }
        10 => bf16::from_f32((channel % 17) as f32 / 4. - 2.).to_bits(),
        11 => {
            if channel < 2 {
                0x7f7f | sign
            } else {
                0x3f80 | sign
            }
        }
        _ => 0x0080 | sign,
    }
}

fn old_tree_sum(input: &[u16], batch: usize, pixel: usize, channels: usize, width: usize) -> f32 {
    let mut squares = vec![0_f32; channels.next_power_of_two()];
    for (channel, square) in squares[..channels].iter_mut().enumerate() {
        let x = bf16::from_bits(input[(batch * channels + channel) * width + pixel]).to_f32();
        *square = x * x;
    }
    let mut span = 1;
    while span < squares.len() {
        for start in (0..squares.len()).step_by(2 * span) {
            let left = squares[start];
            let right = squares[start + span];
            squares[start] = left + right;
        }
        span *= 2;
    }
    squares[0]
}

pub(super) fn prepared(
    channels: usize,
    batch: usize,
    width: usize,
) -> (
    BTreeMap<&'static str, Vec<u16>>,
    BTreeMap<&'static str, Vec<u16>>,
) {
    let mut input = vec![0; batch * channels * width];
    let gamma: Vec<_> = (0..channels)
        .map(|channel| [0x3f85, 0xbf80, 0x3f00, 0x4000, 0][channel % 5])
        .collect();
    for n in 0..batch {
        for channel in 0..channels {
            for pixel in 0..width {
                input[(n * channels + channel) * width + pixel] =
                    input_bits(channel, n * width + pixel, channels);
            }
        }
    }
    let mut expected = vec![0; input.len()];
    for n in 0..batch {
        for pixel in 0..width {
            let sum = old_tree_sum(&input, n, pixel, channels, width);
            // The original profile specifies RN32 sqrt followed by RN32 div.
            // Compute sqrt at F64 before rounding, rather than copying a
            // candidate rsqrt or carry helper. NaN is checked by class below.
            let norm = f64::from(sum).sqrt() as f32;
            let denominator = if norm.is_nan() {
                norm
            } else {
                norm.max(1e-12_f32)
            };
            for channel in 0..channels {
                let i = (n * channels + channel) * width + pixel;
                let x = bf16::from_bits(input[i]).to_f32();
                let normalized = bf16::from_f32(x / denominator);
                let scaled = bf16::from_f32(normalized.to_f32() * 1.5);
                let weighted =
                    bf16::from_f32(scaled.to_f32() * bf16::from_bits(gamma[channel]).to_f32());
                expected[i] = bf16::from_f32(weighted.to_f32() + 0.0).to_bits();
            }
        }
    }
    (
        BTreeMap::from([(NORM, input), (GAMMA, gamma)]),
        BTreeMap::from([(NORM_OUT, expected)]),
    )
}

pub(super) fn check(bytes: &[u8], expected: &[u16]) {
    assert_eq!(bytes.len(), 2 * expected.len());
    for (i, (bytes, &expected)) in bytes.chunks_exact(2).zip(expected).enumerate() {
        let actual = u16::from_le_bytes([bytes[0], bytes[1]]);
        if expected & 0x7fff > 0x7f80 {
            assert!(
                actual & 0x7fff > 0x7f80,
                "channel norm[{i}] expected NaN, got {actual:04x}"
            );
        } else {
            assert_eq!(actual, expected, "channel norm[{i}]");
        }
    }
}

#[test]
fn old_tree_fixture_covers_nonassociativity_padding_ieee_and_nchw_batch_tail() {
    for channels in [96, 144, 288, 4096] {
        let width = if channels == 4096 { 33 } else { 131 };
        let (inputs, expected) = prepared(channels, 2, width);
        assert_eq!(
            old_tree_sum(&inputs[NORM], 0, 7, channels, width),
            16_777_218.
        );
        let x = &inputs[NORM];
        let output = &expected[NORM_OUT];
        assert_eq!(x.len(), 2 * channels * width);
        // All-zero input retains the explicit final +0 boundary, including
        // negative input zeros and negative/zero gamma values.
        for channel in 0..channels {
            assert_eq!(output[channel * width], 0);
            assert!(output[channel * width + 4] & 0x7fff > 0x7f80);
        }
        assert_ne!(x[0], x[channels * width]);
        let probe = Probe::ChannelNorm {
            channels: channels as u64,
            batch: 2,
            width: width as u64,
        };
        TypedFamilyRegistration::new(DecoderFamily)
            .prepare_with_profile(
                &serde_json::to_value(probe).unwrap(),
                &NumericalProfileId::new(PROFILE).unwrap(),
            )
            .unwrap();
    }
}
