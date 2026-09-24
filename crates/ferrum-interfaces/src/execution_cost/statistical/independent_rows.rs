//! Passive empirical family hypothesis, never an executable reordering proof.
//! An explicitly selected provider may group whole independent attention rows;
//! the original ordered V1 stream is recorded unconditionally by its builder.
use super::*;

pub(super) enum IndependentRowsDigest {
    Ordered,
    Rows {
        prefix: [u8; 32],
        expected: usize,
        completed: Vec<[u8; 32]>,
        current: Option<(Sha256, usize)>,
    },
    Suffix {
        prefix: [u8; 32],
        rows: [u8; 32],
        suffix: Sha256,
    },
    Unavailable,
}

impl IndependentRowsDigest {
    pub(super) fn begin(&mut self, prefix: &Sha256, expected: usize) {
        if !matches!(self, Self::Ordered) || !(2..=MAX_COST_ROWS).contains(&expected) {
            *self = Self::Unavailable;
            return;
        }
        let mut completed = Vec::new();
        if completed.try_reserve_exact(expected).is_err() {
            *self = Self::Unavailable;
            return;
        }
        *self = Self::Rows {
            prefix: prefix.clone().finalize().into(),
            expected,
            completed,
            current: None,
        };
    }

    pub(super) fn begin_row(&mut self) {
        match self {
            Self::Rows {
                expected,
                completed,
                current,
                ..
            } if current.is_none() && completed.len() < *expected => {
                let mut hash = Sha256::new();
                bytes(&mut hash, b"ferrum.independent-attention-row.v2");
                *current = Some((hash, 0));
            }
            _ => *self = Self::Unavailable,
        }
    }

    pub(super) fn kernel(&mut self, algorithm: SelectedAlgorithmClassV1) {
        match self {
            Self::Rows {
                current: Some((hash, count)),
                ..
            } => {
                number(hash, 0);
                hash.update(algorithm.0);
                *count += 1;
            }
            Self::Suffix { suffix, .. } => {
                number(suffix, 0);
                suffix.update(algorithm.0);
            }
            Self::Rows { .. } => *self = Self::Unavailable,
            Self::Ordered | Self::Unavailable => {}
        }
    }

    pub(super) fn transfer(&mut self, algorithm: SelectedAlgorithmClassV1, kind: u64) {
        match self {
            Self::Suffix { suffix, .. } => {
                number(suffix, 1);
                suffix.update(algorithm.0);
                number(suffix, kind);
            }
            // The initial hypothesis covers compute-only row blocks. Transfers
            // have their own ownership/lifetime and cannot be sorted into them.
            Self::Rows { .. } => *self = Self::Unavailable,
            Self::Ordered | Self::Unavailable => {}
        }
    }

    pub(super) fn end_row(&mut self) {
        if let Self::Rows {
            current, completed, ..
        } = self
        {
            if let Some((hash, count)) = current.take() {
                // At least prepare and the selected attention kernel. This
                // checks structural completeness, not provider independence.
                if count >= 2 {
                    completed.push(hash.finalize().into());
                    return;
                }
            }
        }
        *self = Self::Unavailable;
    }

    pub(super) fn end(&mut self) {
        let prior = std::mem::replace(self, Self::Unavailable);
        if let Self::Rows {
            prefix,
            expected,
            mut completed,
            current: None,
        } = prior
        {
            if completed.len() == expected {
                completed.sort_unstable();
                let mut hash = Sha256::new();
                bytes(&mut hash, b"ferrum.independent-attention-row-multiset.v2");
                number(&mut hash, expected as u64);
                for row in completed {
                    hash.update(row);
                }
                let mut suffix = Sha256::new();
                bytes(&mut suffix, b"ferrum.independent-attention-suffix.v2");
                *self = Self::Suffix {
                    prefix,
                    rows: hash.finalize().into(),
                    suffix,
                };
            }
        }
    }

    pub(super) fn finish(self, ordered: [u8; 32]) -> Option<[u8; 32]> {
        match self {
            // Ordinary commands reuse the already checked ordered digest:
            // no second traversal of kernel/transfer sub-work or numeric sums.
            Self::Ordered => Some(ordered),
            Self::Suffix {
                prefix,
                rows,
                suffix,
            } => {
                let mut hash = Sha256::new();
                bytes(&mut hash, b"ferrum.independent-attention-command.v2");
                hash.update(prefix);
                hash.update(rows);
                hash.update(suffix.finalize());
                Some(hash.finalize().into())
            }
            Self::Rows { .. } | Self::Unavailable => None,
        }
    }
}
