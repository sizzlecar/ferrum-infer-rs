use super::*;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StructuredAttentionBf16Mask {
    sequence: u32,
    segment_ends: Vec<u32>,
    bidirectional: Vec<bool>,
    key_ends: Vec<Vec<u32>>,
    key_valid: Vec<Vec<bool>>,
    pairs: u64,
    nonempty: u64,
}
impl StructuredAttentionBf16Mask {
    pub(in crate::standard_operations) fn parse(
        sequence: u32,
        batch: usize,
        attrs: &BTreeMap<AttributeId, SemanticValue>,
    ) -> Result<Self, VNextError> {
        let lengths = integers(attrs, "segment_lengths")?;
        let flags = integers(attrs, "segment_bidirectional")?;
        if lengths.is_empty()
            || lengths.len() > STRUCTURED_ATTENTION_BF16_MAX_SEGMENTS
            || flags.len() != lengths.len()
        {
            return Err(invalid(
                "segment lengths/flags must have matching bounded nonempty lengths",
            ));
        }
        let segment_ends = ends(lengths, sequence)?;
        let bidirectional = booleans(flags)?;
        let counts = integers(attrs, "key_run_counts")?;
        let key_lengths = integers(attrs, "key_run_lengths")?;
        let key_flags = integers(attrs, "key_run_valid")?;
        if counts.len() != batch || key_lengths.len() != key_flags.len() {
            return Err(invalid(
                "key runs must specify every batch and matching lengths/flags",
            ));
        }
        let mut offset = 0_usize;
        let mut key_ends = Vec::with_capacity(batch);
        let mut key_valid = Vec::with_capacity(batch);
        for &count in counts {
            let count = usize::try_from(count).map_err(|_| invalid("negative key run count"))?;
            if !(1..=STRUCTURED_ATTENTION_BF16_MAX_KEY_RUNS).contains(&count) {
                return Err(invalid("key run count must be in 1..=32"));
            }
            let next = offset
                .checked_add(count)
                .filter(|&n| n <= key_lengths.len())
                .ok_or_else(|| invalid("key run counts exceed flattened lists"))?;
            let valid = booleans(&key_flags[offset..next])?;
            if valid.windows(2).any(|pair| pair[0] == pair[1]) {
                return Err(invalid(
                    "adjacent key runs must alternate valid and invalid",
                ));
            }
            key_ends.push(ends(&key_lengths[offset..next], sequence)?);
            key_valid.push(valid);
            offset = next;
        }
        if offset != key_lengths.len() {
            return Err(invalid("unused flattened key runs"));
        }
        let mut result = Self {
            sequence,
            segment_ends,
            bidirectional,
            key_ends,
            key_valid,
            pairs: 0,
            nonempty: 0,
        };
        let (pairs, nonempty) = result.count_work()?;
        result.pairs = pairs;
        result.nonempty = nonempty;
        Ok(result)
    }
    pub fn segment_ends(&self) -> &[u32] {
        &self.segment_ends
    }
    pub fn segment_bidirectional(&self) -> &[bool] {
        &self.bidirectional
    }
    pub fn key_run_ends(&self, batch: usize) -> Option<&[u32]> {
        self.key_ends.get(batch).map(Vec::as_slice)
    }
    pub fn key_run_valid(&self, batch: usize) -> Option<&[bool]> {
        self.key_valid.get(batch).map(Vec::as_slice)
    }
    pub fn batch(&self) -> usize {
        self.key_ends.len()
    }
    pub fn visible_end(&self, query: u32) -> Option<u32> {
        if query >= self.sequence {
            return None;
        }
        let index = self.segment_ends.partition_point(|&end| end <= query);
        Some(if self.bidirectional[index] {
            self.segment_ends[index]
        } else {
            query + 1
        })
    }
    pub fn key_is_valid(&self, batch: usize, key: u32) -> Option<bool> {
        if key >= self.sequence {
            return None;
        }
        let ends = self.key_ends.get(batch)?;
        self.key_valid
            .get(batch)?
            .get(ends.partition_point(|&end| end <= key))
            .copied()
    }
    pub const fn visible_pairs(&self) -> u64 {
        self.pairs
    }
    pub const fn nonempty_rows(&self) -> u64 {
        self.nonempty
    }
    // O(batch * segments * runs), never O(sequence) in plan preparation.
    fn count_work(&self) -> Result<(u64, u64), VNextError> {
        let mut pairs = 0_u128;
        let mut nonempty = 0_u128;
        for (key_ends, key_valid) in self.key_ends.iter().zip(&self.key_valid) {
            let first =
                key_valid
                    .iter()
                    .position(|&v| v)
                    .map(|i| if i == 0 { 0 } else { key_ends[i - 1] });
            let mut start = 0_u32;
            for (&end, &bidir) in self.segment_ends.iter().zip(&self.bidirectional) {
                if let Some(first) = first {
                    nonempty += if bidir {
                        if first < end {
                            u128::from(end - start)
                        } else {
                            0
                        }
                    } else {
                        u128::from(end.saturating_sub(start.max(first)))
                    };
                }
                let mut key_start = 0_u32;
                for (&key_end, &valid) in key_ends.iter().zip(key_valid) {
                    if valid {
                        if bidir {
                            pairs += u128::from(end - start)
                                * u128::from(key_end.min(end).saturating_sub(key_start));
                        } else {
                            // Keys before this causal segment occur in every query.
                            pairs += u128::from(end - start)
                                * u128::from(key_end.min(start).saturating_sub(key_start));
                            let low = key_start.max(start);
                            let high = key_end.min(end);
                            if low < high {
                                let n = u128::from(high - low);
                                pairs += n * u128::from(end)
                                    - (u128::from(low) + u128::from(high) - 1) * n / 2;
                            }
                        }
                    }
                    key_start = key_end;
                }
                start = end;
            }
        }
        Ok((
            u64::try_from(pairs).map_err(|_| invalid("visible pair count overflows u64"))?,
            u64::try_from(nonempty).map_err(|_| invalid("nonempty row count overflows u64"))?,
        ))
    }
}
fn ends(lengths: &[i64], sequence: u32) -> Result<Vec<u32>, VNextError> {
    let mut end = 0_u32;
    let mut result = Vec::with_capacity(lengths.len());
    for &length in lengths {
        let length = u32::try_from(length)
            .ok()
            .filter(|&n| n > 0)
            .ok_or_else(|| invalid("run/segment lengths must be positive u32"))?;
        end = end
            .checked_add(length)
            .filter(|&n| n <= sequence)
            .ok_or_else(|| invalid("run/segment total exceeds sequence"))?;
        result.push(end);
    }
    if end != sequence {
        return Err(invalid(
            "run/segment lengths must cover the entire sequence",
        ));
    }
    Ok(result)
}
fn booleans(flags: &[i64]) -> Result<Vec<bool>, VNextError> {
    flags
        .iter()
        .map(|&value| match value {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(invalid("mask flags must be zero or one")),
        })
        .collect()
}
