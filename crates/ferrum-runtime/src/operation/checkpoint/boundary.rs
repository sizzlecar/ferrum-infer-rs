//! Boundary arithmetic shared by checkpoint layout and product range planning.

use super::CheckpointBoundaryConstraint;

impl CheckpointBoundaryConstraint {
    /// Greatest legal boundary before the prompt's end. This pure arithmetic
    /// does not establish checkpoint capability or reserve any resources.
    pub fn prompt_tail_boundary(&self, processed: u64, prompt: u64) -> Option<u64> {
        self.latest_reusable_boundary(processed, prompt, prompt, &[])
    }

    pub(crate) fn latest_reusable_boundary(
        &self,
        processed: u64,
        source_prompt: u64,
        common_prefix: u64,
        follower_prompts: &[u64],
    ) -> Option<u64> {
        let prefix = self.prefix();
        let suffix = self.suffix();
        let a = prefix.alignment().get();
        let b = suffix.alignment().get();
        let suffix_residue = source_prompt % b;
        let mut upper =
            common_prefix.min(source_prompt.checked_sub(suffix.minimum_tokens().get())?);
        for &prompt in follower_prompts {
            if prompt % b != suffix_residue {
                return None;
            }
            upper = upper.min(prompt.checked_sub(suffix.minimum_tokens().get())?);
        }
        let lower = processed.checked_add(prefix.minimum_tokens().get())?;
        if lower > upper {
            return None;
        }
        let divisor = gcd(a, b);
        let prefix_residue = processed % a;
        if prefix_residue % divisor != suffix_residue % divisor {
            return None;
        }

        // Solve x = processed (mod a), x = source_prompt (mod b). The
        // coefficients and period fit u128 even when their product exceeds
        // u64. Euclid takes logarithmic steps; no token-position scan occurs.
        let reduced_b = b / divisor;
        let inverse = modular_inverse(a / divisor, reduced_b)?;
        let delta = (u128::from(suffix_residue) + u128::from(b) - u128::from(prefix_residue % b))
            % u128::from(b);
        let multiplier =
            (delta / u128::from(divisor) * u128::from(inverse)) % u128::from(reduced_b);
        let period = u128::from(a) * u128::from(reduced_b);
        let residue = u128::from(prefix_residue) + u128::from(a) * multiplier;
        let available = u128::from(upper).checked_sub(residue)?;
        let boundary = residue + available / period * period;
        (boundary >= u128::from(lower))
            .then(|| u64::try_from(boundary).ok())
            .flatten()
    }
}

fn gcd(mut left: u64, mut right: u64) -> u64 {
    while right != 0 {
        (left, right) = (right, left % right);
    }
    left
}

fn modular_inverse(value: u64, modulus: u64) -> Option<u64> {
    if modulus == 1 {
        return Some(0);
    }
    let (mut previous, mut remainder) = (modulus, value % modulus);
    let (mut previous_coefficient, mut coefficient) = (0_u128, 1_u128);
    let wide_modulus = u128::from(modulus);
    while remainder != 0 {
        let quotient = previous / remainder;
        let next = (previous_coefficient + wide_modulus
            - (u128::from(quotient) * coefficient) % wide_modulus)
            % wide_modulus;
        (previous, remainder) = (remainder, previous % remainder);
        (previous_coefficient, coefficient) = (coefficient, next);
    }
    (previous == 1)
        .then(|| u64::try_from(previous_coefficient).ok())
        .flatten()
}

#[cfg(test)]
mod tests;
