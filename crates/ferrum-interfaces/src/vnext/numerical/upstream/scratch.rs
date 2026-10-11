//! Checked admission envelope for fixed-J32 MMQ prefill, including MarkerV2.
//! The native planner remains authoritative for each actual launch. This
//! monotone bound avoids enumerating every admitted row count during planning.

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct UpstreamScratchEstimate {
    pub fixed_bytes: u64,
    pub bytes_per_row: u64,
}

impl UpstreamScratchEstimate {
    pub fn bytes(self, rows: u64) -> Result<u64, String> {
        self.bytes_per_row
            .checked_mul(rows)
            .and_then(|n| n.checked_add(self.fixed_bytes))
            .ok_or_else(|| "upstream scratch extent overflows".into())
    }

    /// Sequential leaves reuse storage: the maximum intercept and slope bound
    /// each leaf without assuming that native fixup size is monotone in M.
    pub fn include(&mut self, other: Self) {
        self.fixed_bytes = self.fixed_bytes.max(other.fixed_bytes);
        self.bytes_per_row = self.bytes_per_row.max(other.bytes_per_row);
    }

    pub fn mmq_prefill(inputs: u64, outputs: u64, multiprocessors: u32) -> Result<Self, String> {
        if inputs == 0 || inputs % 256 != 0 || outputs == 0 || multiprocessors == 0 {
            return Err("invalid fixed-J32 scratch geometry".into());
        }
        let overflow = || "fixed-J32 scratch arithmetic overflows".to_owned();
        let padded_k = inputs.checked_add(511).ok_or_else(overflow)? / 512 * 512;
        // converted F32 + 144B per 128 activations + F32 output + row flag.
        let bytes_per_row = padded_k
            .checked_mul(41)
            .map(|n| n / 8)
            .and_then(|n| outputs.checked_mul(4).and_then(|o| n.checked_add(o)))
            .and_then(|n| n.checked_add(4))
            .ok_or_else(overflow)?;
        // J_max <= 512 guard blocks. When fixup exists, blocks == SM and each
        // block stores J32*I128 F32. Five 16-aligned regions need four gaps.
        let fixed_bytes = u64::from(multiprocessors)
            .checked_mul(32 * 128 * 4)
            .and_then(|n| n.checked_add(512 * 144 + 4 * 15))
            .ok_or_else(overflow)?;
        Ok(Self {
            fixed_bytes,
            bytes_per_row,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fixed_j32_envelope_bounds_guard_fixup_alignment_and_partial_k() {
        for (k, n, sm) in [(5120, 17408, 20), (17408, 5120, 170), (768, 49, 1)] {
            let estimate = UpstreamScratchEstimate::mmq_prefill(k, n, sm).unwrap();
            let kp = k.div_ceil(512) * 512;
            let mut previous = 0;
            for m in [33, 64, 128, 256, 2048] {
                // Deliberately alternate zero / maximum fixup: the bound must
                // not rely on the actual planner's discontinuous fixup choice.
                for fixup in [0, u64::from(sm) * 32 * 128 * 4] {
                    let sizes = [
                        4 * m * kp,
                        m * (kp / 128) * 144 + 512 * 144,
                        4 * m * n,
                        fixup,
                        4 * m,
                    ];
                    let exact = sizes
                        .into_iter()
                        .fold(0_u64, |offset, size| offset.div_ceil(16) * 16 + size);
                    assert!(estimate.bytes(m).unwrap() >= exact);
                }
                let bytes = estimate.bytes(m).unwrap();
                assert!(bytes > previous);
                previous = bytes;
            }
        }
        assert!(UpstreamScratchEstimate::mmq_prefill(u64::MAX - 255, 1, 1).is_err());
        assert!(UpstreamScratchEstimate::mmq_prefill(256, u64::MAX, 1).is_err());
        assert!(UpstreamScratchEstimate::mmq_prefill(256, 1, 0).is_err());
        assert!(UpstreamScratchEstimate {
            fixed_bytes: 1,
            bytes_per_row: u64::MAX
        }
        .bytes(2)
        .is_err());
    }
}
