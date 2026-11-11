use super::*;
use std::{hint::black_box, time::Duration, time::Instant};

// Preserve the original runtime arithmetic independently of the const table.
// Both intermediate BF16 roundings and final F32 ties-to-even are observable.
fn arithmetic_oracle(value: bf16) -> u8 {
    let clamped = value.to_f32().clamp(-1.0, 1.0);
    let scaled = bf16::from_f32(clamped * 0.5);
    let shifted = bf16::from_f32(scaled.to_f32() + 0.5)
        .to_f32()
        .clamp(0.0, 1.0);
    (shifted * 255.0).round_ties_even() as u8
}

#[test]
fn lookup_matches_staged_arithmetic_for_every_finite_bf16() {
    for bits in u16::MIN..=u16::MAX {
        let value = bf16::from_bits(bits);
        if value.is_finite() {
            assert_eq!(
                rgba8_finite(value),
                arithmetic_oracle(value),
                "BF16 bits {bits:#06x}"
            );
        }
    }
}

#[test]
fn lookup_preserves_saturation_signed_zero_and_rounding_boundaries() {
    for (bits, expected) in [
        (0x0000, 128),
        (0x8000, 128),
        (0x0001, 128),
        (0x8001, 128),
        (0x3f80, 255),
        (0xbf80, 0),
        (0x7f7f, 255),
        (0xff7f, 0),
        // BF16 shift makes x=5/512 map to128; fused F32 arithmetic gives129.
        (0x3c20, 128),
        // The BF16 result 0.5 gives127.5, which rounds to the even128.
        (0x3b00, 128),
        (0xbb00, 128),
    ] {
        assert_eq!(rgba8_finite(bf16::from_bits(bits)), expected, "{bits:#06x}");
    }
}

fn conversion_time(source: &[u16], destination: &mut [u8], lookup: bool) -> Duration {
    let source = black_box(source);
    let destination = black_box(destination);
    let started = Instant::now();
    if lookup {
        for (&bits, output) in source.iter().zip(destination.iter_mut()) {
            *output = rgba8_finite(bf16::from_bits(bits));
        }
    } else {
        for (&bits, output) in source.iter().zip(destination.iter_mut()) {
            *output = arithmetic_oracle(bf16::from_bits(bits));
        }
    }
    let elapsed = started.elapsed();
    black_box(destination);
    elapsed
}

/// Reusable CPU diagnostic, not an image/HTTP throughput acceptance test.
/// Run with an optimized profile and --ignored --nocapture. Inputs and output
/// backing are prepared before timing; this excludes NCHW interleaving, finite
/// validation, PNG/base64, scheduler delays, and device work. No speed threshold
/// is asserted, and both orders are measured to expose order effects.
#[test]
#[ignore = "CPU conversion microbenchmark; use an optimized profile and --nocapture"]
fn benchmark_bf16_rgba8_conversion() {
    assert!(
        !cfg!(debug_assertions),
        "benchmark requires an optimized profile"
    );
    for edge in [1024, 2048] {
        let elements = edge * edge * 4;
        // A deterministic finite ramp covers [-1.25,1.25], including clipping
        // and pixel boundaries. It contains no model output or prompt cache.
        let source: Vec<_> = (0..elements)
            .map(|index| {
                let value = ((index % 8193) as f32 - 4096.0) * (1.25 / 4096.0);
                bf16::from_f32(value).to_bits()
            })
            .collect();
        let mut arithmetic = vec![0; elements];
        let mut lookup = vec![0; elements];
        conversion_time(&source, &mut arithmetic, false);
        conversion_time(&source, &mut lookup, true);
        assert_eq!(arithmetic, lookup);
        for repetition in 0..4 {
            let (arithmetic_time, lookup_time) = if repetition % 2 == 0 {
                (
                    conversion_time(&source, &mut arithmetic, false),
                    conversion_time(&source, &mut lookup, true),
                )
            } else {
                let lookup_time = conversion_time(&source, &mut lookup, true);
                let arithmetic_time = conversion_time(&source, &mut arithmetic, false);
                (arithmetic_time, lookup_time)
            };
            assert_eq!(arithmetic, lookup);
            eprintln!(
                "bf16_rgba8_conversion edge={edge} elements={elements} repetition={repetition} first={} arithmetic_us={} lookup_us={}",
                if repetition % 2 == 0 { "arithmetic" } else { "lookup" },
                arithmetic_time.as_micros(),
                lookup_time.as_micros(),
            );
        }
    }
}
