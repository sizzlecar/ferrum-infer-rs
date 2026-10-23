//! Static visual row selection for the three post-layer DeepStack additions.
use super::{invalid, Qwen3VLTextLayout};
use ferrum_runtime::*;
use serde::{Deserialize, Deserializer, Serialize};
use std::collections::BTreeMap;

pub(super) const SCATTER_ADD_OPERATION_ID: &str = "operation.row_scatter_add.bf16";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Qwen3VLTextVisualRun {
    start: u64,
    length: u64,
}
impl Qwen3VLTextVisualRun {
    pub fn new(start: u64, length: u64) -> Result<Self, VNextError> {
        if length == 0 || start.checked_add(length).is_none() {
            return Err(invalid(
                "visual_run",
                "requires a nonempty checked row interval",
            ));
        }
        Ok(Self { start, length })
    }
    pub const fn start(&self) -> u64 {
        self.start
    }
    pub const fn length(&self) -> u64 {
        self.length
    }
}

/// Rows are absolute in the flattened `[batch * sequence, hidden]` tensor.
/// Updates contain these rows in run order; padding cannot receive an update.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Qwen3VLTextDeepStack {
    runs: Vec<Qwen3VLTextVisualRun>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Wire {
    runs: Vec<Qwen3VLTextVisualRun>,
}
impl<'de> Deserialize<'de> for Qwen3VLTextDeepStack {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        Self::new(Wire::deserialize(d)?.runs).map_err(serde::de::Error::custom)
    }
}
impl Qwen3VLTextDeepStack {
    pub fn new(runs: Vec<Qwen3VLTextVisualRun>) -> Result<Self, VNextError> {
        if runs.is_empty() || runs.len() > 32 {
            return Err(invalid("deepstack", "requires 1..=32 visual runs"));
        }
        let mut end = 0;
        let mut total = 0_u64;
        for run in &runs {
            if run.length == 0 || run.start < end {
                return Err(invalid(
                    "deepstack",
                    "runs must be positive, ordered and nonoverlapping",
                ));
            }
            end = run
                .start
                .checked_add(run.length)
                .ok_or_else(|| invalid("deepstack", "row extent overflows"))?;
            total = total
                .checked_add(run.length)
                .ok_or_else(|| invalid("deepstack", "update rows overflow"))?;
            i64::try_from(end)
                .map_err(|_| invalid("deepstack", "row extent exceeds attribute range"))?;
        }
        Ok(Self { runs })
    }
    pub fn runs(&self) -> &[Qwen3VLTextVisualRun] {
        &self.runs
    }
    pub fn rows(&self) -> Result<u64, VNextError> {
        self.runs.iter().try_fold(0_u64, |n, r| {
            n.checked_add(r.length)
                .ok_or_else(|| invalid("deepstack", "update rows overflow"))
        })
    }
    pub(super) fn validate(
        &self,
        layout: &Qwen3VLTextLayout,
        hidden: u64,
    ) -> Result<(), VNextError> {
        Self::new(self.runs.clone())?;
        let sequence = u64::from(layout.sequence_length());
        for run in &self.runs {
            let end = run.start + run.length;
            if end > layout.rows() || run.start / sequence != (end - 1) / sequence {
                return Err(invalid(
                    "deepstack",
                    "a visual run must lie within one sample",
                ));
            }
            let batch = (run.start / sequence) as usize;
            let local_start = run.start % sequence;
            let local_end = local_start + run.length;
            let mut cursor = 0_u64;
            for key_run in &layout.key_runs()[batch] {
                let next = cursor + u64::from(key_run.length());
                if !key_run.is_valid() && local_start < next && cursor < local_end {
                    return Err(invalid(
                        "deepstack",
                        "visual updates cannot target padding rows",
                    ));
                }
                cursor = next;
            }
        }
        ResolvedTensorSpec::new(
            vec![self.rows()?, hidden],
            ElementType::Bf16,
            ResolvedTensorLayout::Contiguous,
        )?;
        Ok(())
    }
    pub(super) fn attributes(&self) -> Result<BTreeMap<AttributeId, SemanticValue>, VNextError> {
        Ok(BTreeMap::from([
            (
                AttributeId::new("destination_starts")?,
                SemanticValue::Integers(self.runs.iter().map(|r| r.start as i64).collect()),
            ),
            (
                AttributeId::new("run_lengths")?,
                SemanticValue::Integers(self.runs.iter().map(|r| r.length as i64).collect()),
            ),
        ]))
    }
}
