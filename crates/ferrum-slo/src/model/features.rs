use super::ModelError;
use ferrum_execution::execution_attempt::{AttemptPrefixRestore, AttemptWorkKind};
use ferrum_scheduler::cost_prediction::CostWork;
use serde::{Deserialize, Serialize};

/// Changing feature meaning or units requires a new schema, including persisted fits.
pub const FEATURE_SCHEMA: u32 = 1;
pub const FEATURE_COUNT: usize = 9;
const MAX_EXACT_INTEGER: u128 = 1u128 << 53;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Phase {
    Prefill,
    Decode,
    /// One actual mixed invocation; never inferred by adding pure-phase samples.
    Mixed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ReuseKnowledge {
    Known,
    Unknown,
    Partial,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct FeatureDomain {
    pub phase: Phase,
    pub reuse: ReuseKnowledge,
}

/// Product work, not measured kernel work. All inputs exist before execution.
/// Numeric entries are, in order: participants, requested tokens, decode context
/// sum, prefill causal attention pairs, confirmed restored tokens, confirmed
/// internal reuse, unknown-reuse participants, prefill participants, max context.
/// Restore and internal reuse remain separate; neither is subtracted from work.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct WorkFeatures {
    domain: FeatureDomain,
    values: [f64; FEATURE_COUNT],
}

impl WorkFeatures {
    pub fn extract(work: &[CostWork], max_participants: usize) -> Result<Self, ModelError> {
        Self::extract_iter(work.iter().copied(), max_participants)
    }

    pub(crate) fn extract_iter(
        work: impl ExactSizeIterator<Item = CostWork>,
        max_participants: usize,
    ) -> Result<Self, ModelError> {
        let count = work.len();
        if count == 0 {
            return Err(ModelError::UnsupportedDomain);
        }
        if count > max_participants {
            return Err(ModelError::ResourceLimit("query participants"));
        }
        let mut values = [0.0; FEATURE_COUNT];
        values[0] = exact(count)?;
        let mut prefill = 0;
        let mut unknown = 0;
        for item in work {
            let start = item.requested.start();
            let end = item.requested.end();
            let count = end - start;
            let n = exact(count)?;
            let position = exact(start)?;
            let context = exact(item.context_tokens)?;
            values[1] += n;
            values[8] = values[8].max(context);
            match item.kind {
                AttemptWorkKind::Prefill => {
                    if end > item.context_tokens {
                        return Err(ModelError::UnsupportedDomain);
                    }
                    prefill += 1;
                    // Sum (start + 1)..=end, without overflowing an integer product.
                    values[3] += n * (2.0 * position + n + 1.0) * 0.5;
                }
                AttemptWorkKind::Decode => {
                    if count != 1 || end != item.context_tokens {
                        return Err(ModelError::UnsupportedDomain);
                    }
                    values[2] += context;
                }
            }
            if let AttemptPrefixRestore::Restored { tokens } = item.prefix_restore {
                if tokens > start {
                    return Err(ModelError::UnsupportedDomain);
                }
                values[4] += exact(tokens)?;
            }
            match item.internally_reused_tokens {
                Some(tokens) if tokens <= end => values[5] += exact(tokens)?,
                Some(_) => return Err(ModelError::UnsupportedDomain),
                None => unknown += 1,
            }
        }
        values[6] = exact(unknown)?;
        values[7] = exact(prefill)?;
        let result = Self {
            domain: FeatureDomain {
                phase: if prefill == 0 {
                    Phase::Decode
                } else if prefill == count {
                    Phase::Prefill
                } else {
                    Phase::Mixed
                },
                reuse: if unknown == 0 {
                    ReuseKnowledge::Known
                } else if unknown == count {
                    ReuseKnowledge::Unknown
                } else {
                    ReuseKnowledge::Partial
                },
            },
            values,
        };
        result.validate()?;
        Ok(result)
    }

    pub fn domain(self) -> FeatureDomain {
        self.domain
    }

    pub fn values(&self) -> &[f64; FEATURE_COUNT] {
        &self.values
    }

    pub fn work_tokens(&self) -> f64 {
        self.values[1]
    }

    pub fn batch_size(&self) -> f64 {
        self.values[0]
    }

    pub(super) fn validate(&self) -> Result<(), ModelError> {
        if self.values.iter().any(|x| !x.is_finite() || *x < 0.0)
            || self.values[0] < 1.0
            || self.values[1] < self.values[0]
        {
            return Err(ModelError::InvalidData(
                "nonfinite or invalid work features",
            ));
        }
        Ok(())
    }
}

fn exact(value: usize) -> Result<f64, ModelError> {
    if value as u128 > MAX_EXACT_INTEGER {
        Err(ModelError::InvalidData(
            "token extent exceeds exact floating-point range",
        ))
    } else {
        Ok(value as f64)
    }
}
