//! Exact analytic probes for the adjacent-pair FMA profile. Coefficients are
//! exclusively 0 or +/-1: each output has one exact BF16 term and one signed
//! zero term, so neither FP32 multiplication nor FMA loses precision. This is
//! not an oracle for arbitrary trigonometric frequency tensors.

use anyhow::{bail, Result};
use serde::Serialize;
use std::collections::BTreeSet;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RotaryProbe {
    Identity,
    Quadrants,
}

impl RotaryProbe {
    fn coefficients(self, pair: usize) -> (i8, i8) {
        match self {
            Self::Identity => (1, 0),
            Self::Quadrants => [(1, 0), (0, 1), (-1, 0), (0, -1)][pair % 4],
        }
    }

    pub fn frequencies(self, head_dim: usize) -> Result<Vec<u8>> {
        if head_dim == 0 || head_dim % 2 != 0 {
            bail!("rotary probe requires a positive even head dimension");
        }
        let bytes = head_dim
            .checked_mul(4)
            .ok_or_else(|| anyhow::anyhow!("frequency size overflows"))?;
        let mut result = Vec::with_capacity(bytes);
        for pair in 0..head_dim / 2 {
            let (cosine, sine) = self.coefficients(pair);
            result.extend_from_slice(&(cosine as f32).to_le_bytes());
            result.extend_from_slice(&(sine as f32).to_le_bytes());
        }
        Ok(result)
    }

    pub fn choices(self, normalized: &[Vec<u16>], head_dim: usize) -> Result<Vec<Vec<u16>>> {
        if head_dim == 0
            || head_dim % 2 != 0
            || normalized.is_empty()
            || normalized.len() % head_dim != 0
        {
            bail!("invalid rotary probe geometry");
        }
        if normalized
            .iter()
            .any(|choices| choices.is_empty() || choices.iter().any(|bits| bits & 0x7fff >= 0x7f80))
        {
            bail!("rotary probes require nonempty finite BF16 allowed sets");
        }
        let mut outputs = Vec::with_capacity(normalized.len());
        for (pair, inputs) in normalized.chunks_exact(2).enumerate() {
            let (cosine, sine) = self.coefficients(pair % (head_dim / 2));
            let mut real = BTreeSet::new();
            let mut imag = BTreeSet::new();
            // Keep every independent normalization possibility. Correlation
            // is conservatively relaxed; no actual intermediate is inspected.
            for &a in &inputs[0] {
                for &b in &inputs[1] {
                    real.insert(add_with_zero(scale(a, cosine), scale(b ^ 0x8000, sine))?);
                    imag.insert(add_with_zero(scale(b, cosine), scale(a, sine))?);
                }
            }
            outputs.push(real.into_iter().collect());
            outputs.push(imag.into_iter().collect());
        }
        Ok(outputs)
    }
}

/// Coefficient zero is positive zero; multiplication preserves XOR zero sign.
fn scale(value: u16, coefficient: i8) -> u16 {
    match coefficient {
        0 => value & 0x8000,
        1 => value,
        -1 => value ^ 0x8000,
        _ => unreachable!("probe coefficient is 0 or +/-1"),
    }
}

/// Exact RN-even addition with at least one zero term, including FMA's rule
/// that opposing zero signs produce +0 and two negative zeros produce -0.
fn add_with_zero(left: u16, right: u16) -> Result<u16> {
    match (left & 0x7fff == 0, right & 0x7fff == 0) {
        (true, true) => Ok(left & right & 0x8000),
        (true, false) => Ok(right),
        (false, true) => Ok(left),
        (false, false) => bail!("quadrant proof requires at least one zero product"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quadrants_rotate_adjacent_pairs_and_restart_at_each_head() {
        let inputs = [0x3f80, 0x4000]
            .repeat(8)
            .into_iter()
            .map(|bits| vec![bits])
            .collect::<Vec<_>>();
        let expected = [
            0x3f80, 0x4000, 0xc000, 0x3f80, 0xbf80, 0xc000, 0x4000, 0xbf80,
        ]
        .repeat(2);
        assert_eq!(
            RotaryProbe::Quadrants.choices(&inputs, 8).unwrap(),
            expected
                .into_iter()
                .map(|bits| vec![bits])
                .collect::<Vec<_>>()
        );
        let frequencies = RotaryProbe::Quadrants.frequencies(8).unwrap();
        let floats = frequencies
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
            .collect::<Vec<_>>();
        assert_eq!(floats, [1., 0., 0., 1., -1., 0., 0., -1.]);
    }

    #[test]
    fn identity_keeps_ieee_signed_zero_fma_semantics() {
        let inputs = [vec![0x8000], vec![0]];
        assert_eq!(
            RotaryProbe::Identity.choices(&inputs, 2).unwrap(),
            [vec![0x8000], vec![0]]
        );
        let inputs = [vec![0x8000], vec![0x8000]];
        assert_eq!(
            RotaryProbe::Identity.choices(&inputs, 2).unwrap(),
            [vec![0], vec![0x8000]]
        );
        let inputs = [vec![0, 0x8000], vec![0, 0x8000]];
        assert_eq!(
            RotaryProbe::Identity.choices(&inputs, 2).unwrap(),
            [vec![0, 0x8000], vec![0, 0x8000]]
        );
    }

    #[test]
    fn allowed_sets_propagate_without_selecting_one_normalization_candidate() {
        let inputs = vec![
            vec![0x3f80, 0x3f81],
            vec![0x4000, 0x4001],
            vec![0x3f80, 0x3f81],
            vec![0x4000, 0x4001],
        ];
        let outputs = RotaryProbe::Quadrants.choices(&inputs, 4).unwrap();
        assert_eq!(
            outputs,
            [
                vec![0x3f80, 0x3f81],
                vec![0x4000, 0x4001],
                vec![0xc000, 0xc001],
                vec![0x3f80, 0x3f81]
            ]
        );
        assert!(!outputs[2].contains(&0xbf80));
        assert!(RotaryProbe::Quadrants
            .choices(&[vec![0x7f80], vec![0]], 2)
            .is_err());
        assert!(RotaryProbe::Identity.frequencies(3).is_err());
        assert!(RotaryProbe::Identity
            .choices(&[vec![], vec![0]], 2)
            .is_err());
    }
}
