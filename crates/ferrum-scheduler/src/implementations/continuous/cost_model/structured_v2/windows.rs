//! Outcome-independent, pre-execution population windows. These numerical DTOs
//! are not live route/settlement receipts or permission to execute a request.
use super::{StructuredOwnerKeyV2, StructuredUnknownV2};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

type Result<T> = std::result::Result<T, StructuredUnknownV2>;
pub const MEMBERSHIP_RULE_REVISION_V2: &str = "ferrum.structured-frontier-windows.v2";
const MAX_WINDOWS: usize = 64;
const MAX_ROWS: usize = 128;

mod cohorts;
pub use cohorts::{CohortPlanV2, CohortRequestV2, CohortV2};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClosedRangeV2 {
    pub minimum: u64,
    pub maximum: u64,
}
impl ClosedRangeV2 {
    pub const ALL: Self = Self {
        minimum: 0,
        maximum: u64::MAX,
    };
    fn validate(self) -> Result<()> {
        if self.minimum > self.maximum {
            Err(StructuredUnknownV2::InvalidSettings)
        } else {
            Ok(())
        }
    }
    fn contains(self, value: u64) -> bool {
        self.minimum <= value && value <= self.maximum
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum PreparedWorkV2 {
    Decode {
        kv_tokens: u32,
    },
    Prefill {
        offset: u32,
        count: u32,
        total_prompt_tokens: u32,
    },
}
impl PreparedWorkV2 {
    pub fn emits_token(self) -> Result<bool> {
        match self {
            Self::Decode { kv_tokens } if kv_tokens > 0 => Ok(true),
            Self::Prefill {
                offset,
                count,
                total_prompt_tokens,
            } if count > 0 => {
                let end = offset
                    .checked_add(count)
                    .ok_or(StructuredUnknownV2::InvalidInput)?;
                if end > total_prompt_tokens {
                    Err(StructuredUnknownV2::InvalidInput)
                } else {
                    Ok(end == total_prompt_tokens)
                }
            }
            _ => Err(StructuredUnknownV2::InvalidInput),
        }
    }
}

/// Captured from the checked actual physical row before execute. This public
/// numerical value is deliberately insufficient to construct a live receipt.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PreparedRowFactsV2 {
    pub physical_position: u32,
    pub work: PreparedWorkV2,
    pub generated_before: u64,
    pub maximum_output: u64,
    pub context_before: u64,
}
impl PreparedRowFactsV2 {
    pub fn validate(&self) -> Result<()> {
        self.work.emits_token()?;
        if self.maximum_output == 0
            || self.generated_before >= self.maximum_output
            || self.context_before > u32::MAX as u64
            || matches!(self.work, PreparedWorkV2::Decode { kv_tokens }
                if self.context_before != u64::from(kv_tokens))
        {
            return Err(StructuredUnknownV2::InvalidInput);
        }
        Ok(())
    }
    pub fn remaining_output(&self) -> Result<u64> {
        self.validate()?;
        self.maximum_output
            .checked_sub(self.generated_before)
            .ok_or(StructuredUnknownV2::InvalidInput)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum WorkWindowV2 {
    Decode {
        kv_tokens: ClosedRangeV2,
    },
    Prefill {
        offset: ClosedRangeV2,
        count: ClosedRangeV2,
        total_prompt_tokens: ClosedRangeV2,
        /// None includes both physically non-emitting and final prefill.
        emits_token: Option<bool>,
    },
}
impl WorkWindowV2 {
    fn validate(&self) -> Result<()> {
        match self {
            Self::Decode { kv_tokens } => kv_tokens.validate(),
            Self::Prefill {
                offset,
                count,
                total_prompt_tokens,
                ..
            } => {
                offset.validate()?;
                count.validate()?;
                total_prompt_tokens.validate()
            }
        }
    }
    fn matches(&self, work: PreparedWorkV2) -> Result<bool> {
        Ok(match (self, work) {
            (Self::Decode { kv_tokens: range }, PreparedWorkV2::Decode { kv_tokens }) => {
                range.contains(u64::from(kv_tokens))
            }
            (
                Self::Prefill {
                    offset: a,
                    count: b,
                    total_prompt_tokens: c,
                    emits_token,
                },
                PreparedWorkV2::Prefill {
                    offset,
                    count,
                    total_prompt_tokens,
                },
            ) => {
                a.contains(u64::from(offset))
                    && b.contains(u64::from(count))
                    && c.contains(u64::from(total_prompt_tokens))
                    && emits_token.is_none_or(|expected| work.emits_token() == Ok(expected))
            }
            _ => false,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RowWindowV2 {
    pub generated_before: ClosedRangeV2,
    pub remaining_output: ClosedRangeV2,
    pub context_before: ClosedRangeV2,
    pub work: WorkWindowV2,
}
impl RowWindowV2 {
    fn validate(&self) -> Result<()> {
        self.generated_before.validate()?;
        self.remaining_output.validate()?;
        self.context_before.validate()?;
        self.work.validate()
    }
    fn matches(&self, row: &PreparedRowFactsV2) -> Result<bool> {
        Ok(self.generated_before.contains(row.generated_before)
            && self.remaining_output.contains(row.remaining_output()?)
            && self.context_before.contains(row.context_before)
            && self.work.matches(row.work)?)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrontierWindowV2 {
    /// Physical order, never a sorted multiset or a post-execution host order.
    pub rows: Vec<RowWindowV2>,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MembershipRuleV2 {
    /// Owner selection is a separate, unique A classification. Unknown owner
    /// must be rejected before this numerical predicate is called.
    pub owner: StructuredOwnerKeyV2,
    pub windows: Vec<FrontierWindowV2>,
}
impl MembershipRuleV2 {
    pub fn validate(&self) -> Result<()> {
        if self.owner.rows == 0
            || self.owner.rows as usize > MAX_ROWS
            || self.windows.is_empty()
            || self.windows.len() > MAX_WINDOWS
        {
            return Err(StructuredUnknownV2::InvalidSettings);
        }
        for window in &self.windows {
            if window.rows.len() != self.owner.rows as usize {
                return Err(StructuredUnknownV2::InvalidSettings);
            }
            for row in &window.rows {
                row.validate()?;
            }
        }
        Ok(())
    }
    /// Return the first matching frozen window index, or a proved nonmember.
    /// Overlapping windows reserve only one member. Invalid facts never become
    /// a nonmember, including when the observed owner differs from this rule.
    pub fn classify(
        &self,
        owner: &StructuredOwnerKeyV2,
        rows: &[PreparedRowFactsV2],
    ) -> Result<Option<u32>> {
        self.validate()?;
        if rows.is_empty() || rows.len() > MAX_ROWS || rows.len() != owner.rows as usize {
            return Err(StructuredUnknownV2::InvalidInput);
        }
        for (index, row) in rows.iter().enumerate() {
            if row.physical_position as usize != index {
                return Err(StructuredUnknownV2::InvalidInput);
            }
            row.validate()?;
        }
        if owner != &self.owner {
            return Ok(None);
        }
        for (index, window) in self.windows.iter().enumerate() {
            let mut matches = true;
            for (criterion, row) in window.rows.iter().zip(rows) {
                matches &= criterion.matches(row)?;
            }
            if matches {
                return Ok(Some(index as u32));
            }
        }
        Ok(None)
    }
    pub fn signature(&self) -> Result<[u8; 32]> {
        self.validate()?;
        let mut hash = Sha256::new();
        hash.update(MEMBERSHIP_RULE_REVISION_V2.as_bytes());
        hash.update([0]);
        // Typed struct/enum field order is the versioned canonical encoding.
        // Deserialization must reserialize this DTO, never hash caller JSON order.
        hash.update(serde_json::to_vec(self).map_err(|_| StructuredUnknownV2::InvalidInput)?);
        Ok(hash.finalize().into())
    }
}

#[cfg(test)]
mod tests;
