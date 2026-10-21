//! Exhaustive finite BF16 grid qualification and classified special inputs.
//! Timing and set-size histograms are observations only, never pass criteria.

use super::*;
use crate::backend::reference::runtime::{silu_bf16_bits, silu_mul_bf16_bits, tanh_bf16_bits};
use std::hint::black_box;
use std::time::Instant;

#[derive(Default, Debug)]
struct PrimitiveObservations {
    checked: usize,
    special_results: usize,
    reference_cell_ambiguities: usize,
    step_bound_histogram: [usize; 5],
    allowed_branch_histogram: [usize; 10],
}

impl PrimitiveObservations {
    fn record(&mut self, expected: &AllowedF32, actual: u32) {
        self.checked += 1;
        self.reference_cell_ambiguities += usize::from(expected.reference_is_ambiguous());
        self.allowed_branch_histogram[expected.len()] += 1;
        match expected.observed_step_bound(actual) {
            Some(distance) => self.step_bound_histogram[distance as usize] += 1,
            None => self.special_results += 1,
        }
    }
}

fn context(stage: &str, input: u16, error: String) -> String {
    format!("{stage}, BF16 input {input:04x}: {error}")
}

#[test]
fn arithmetic_environment_supports_rne_and_gradual_underflow() {
    // These runtime probes reject an unsupported environment; they do not
    // change process rounding state or replace the declared platform contract.
    let tiny32 = black_box(f32::from_bits(1));
    assert_eq!((tiny32 * black_box(2.0)).to_bits(), 2);
    assert_eq!((tiny32 / black_box(2.0)).to_bits(), 0);
    assert_eq!((black_box(f32::from_bits(3)) / black_box(2.0)).to_bits(), 2);
    let tiny64 = black_box(f64::from_bits(1));
    assert_eq!((tiny64 * black_box(2.0)).to_bits(), 2);
    assert_eq!(
        (black_box(1.0_f32) + black_box(f32::from_bits(0x3380_0000))).to_bits(),
        0x3f80_0000
    );
    assert_eq!(
        (black_box(f32::from_bits(0x3f80_0001)) + black_box(f32::from_bits(0x3380_0000))).to_bits(),
        0x3f80_0002
    );
}

#[test]
fn all_bf16_encodings_qualify_expf_and_tanhf_against_prepared_intervals() -> Result<(), String> {
    let started = Instant::now();
    let mut exponential = PrimitiveObservations::default();
    let mut hyperbolic = PrimitiveObservations::default();
    for input in 0..=u16::MAX {
        // Both independent objects exist before either candidate is evaluated.
        let expected_exp = prepare_primitive(Primitive::Exp, input)
            .map_err(|e| context("prepare exp", input, e))?;
        let expected_tanh = prepare_primitive(Primitive::Tanh, input)
            .map_err(|e| context("prepare tanh", input, e))?;
        let value = decode_bf16(input);
        let actual_exp = libm::expf(value).to_bits();
        let actual_tanh = libm::tanhf(value).to_bits();
        expected_exp
            .check(actual_exp)
            .map_err(|e| context("check exp", input, e))?;
        expected_tanh
            .check(actual_tanh)
            .map_err(|e| context("check tanh", input, e))?;
        exponential.record(&expected_exp, actual_exp);
        hyperbolic.record(&expected_tanh, actual_tanh);
    }
    eprintln!(
        "BF16 primitive qualification elapsed={:?}; exp={exponential:?}; tanh={hyperbolic:?}",
        started.elapsed()
    );
    Ok(())
}

#[test]
fn all_bf16_activations_and_selected_multiplier_domains_use_independent_sets() -> Result<(), String>
{
    // Each entry exercises a distinct arithmetic/special branch, not a
    // throughput workload or an empirical substitute for all 2^32 input pairs.
    const MULTIPLIERS: [u16; 11] = [
        0, 0x8000, 0x3f80, 0xbf80, 0x3f81, 0x3f00, 0x0001, 0x7f7f, 0x7f80, 0xff80, 0x7fc1,
    ];
    let started = Instant::now();
    let mut silu_sizes = [0_usize; 10];
    let mut tanh_sizes = [0_usize; 10];
    let mut product_sizes = [0_usize; 10];
    for input in 0..=u16::MAX {
        let expected_silu = prepare_silu(input).map_err(|e| context("prepare silu", input, e))?;
        let expected_tanh =
            prepare_tanh(input).map_err(|e| context("prepare tanh BF16", input, e))?;
        expected_silu
            .check(silu_bf16_bits(input))
            .map_err(|e| context("check silu", input, e))?;
        expected_tanh
            .check(tanh_bf16_bits(input))
            .map_err(|e| context("check tanh BF16", input, e))?;
        silu_sizes[expected_silu.len()] += 1;
        tanh_sizes[expected_tanh.len()] += 1;
        for up in MULTIPLIERS {
            // Propagate EVERY allowed activation, never the observed scalar.
            let expected_product = expected_silu
                .multiplied_by(up)
                .map_err(|e| context("prepare silu-mul", input, e))?;
            expected_product
                .check(silu_mul_bf16_bits(input, up))
                .map_err(|e| format!("check silu-mul, gate={input:04x}, up={up:04x}: {e}"))?;
            product_sizes[expected_product.len()] += 1;
        }
    }
    eprintln!("BF16 composition qualification elapsed={:?}; silu-set-sizes={silu_sizes:?}; tanh-set-sizes={tanh_sizes:?}; selected-product-set-sizes={product_sizes:?}", started.elapsed());
    Ok(())
}
