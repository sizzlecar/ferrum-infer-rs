use super::*;
fn same(a: &impl Serialize, b: &impl Serialize) -> Result<bool, CostProfileError> {
    Ok(canonical_value_v7(a)? == canonical_value_v7(b)?)
}
impl StructuredServiceCollectorV7 {
    pub(super) fn push_inner(
        &mut self,
        r: &StructuredServiceRecordV7,
    ) -> Result<(), CostProfileError> {
        if self.closed {
            return Err(invalid("source7 is closed"));
        }
        if self.poisoned && !matches!(r, StructuredServiceRecordV7::Footer { .. }) {
            return Err(invalid("source7 poisoned original population"));
        }
        match r {
            StructuredServiceRecordV7::BlockOpen {
                opened_at_ns,
                fifo_cutoff,
                ..
            } => {
                let expected = self.open_inner(*opened_at_ns, *fifo_cutoff)?;
                if !same(&expected, r)? {
                    return Err(invalid("source7 frozen block assignments differ"));
                }
            }
            StructuredServiceRecordV7::Completed { wave } => self.wave(wave)?,
            StructuredServiceRecordV7::OutsideDeclaredRoute { wave } => self.outside(wave)?,
            StructuredServiceRecordV7::NotSubmitted { attempt } => self.not_submitted(attempt)?,
            StructuredServiceRecordV7::BlockClose {
                closing, freezes, ..
            } => {
                let expected = self.close_inner(*closing, Some(freezes))?;
                if !same(&expected, r)? {
                    return Err(invalid("source7 original block/freeze/certificate differs"));
                }
            }
            StructuredServiceRecordV7::Checkpoint { closing, .. } => {
                if !same(&self.checkpoint_record(*closing)?, r)? {
                    return Err(invalid("source7 checkpoint differs"));
                }
            }
            StructuredServiceRecordV7::Failed {
                source_prefix_bytes,
                source_prefix_sha256,
                reason,
                ..
            } => {
                if (*source_prefix_bytes, *source_prefix_sha256) != self.source_receipt()
                    || reason.len() > 4096
                {
                    return Err(invalid("source7 failed population prefix differs"));
                }
                self.poisoned = true;
            }
            StructuredServiceRecordV7::Footer {
                closing,
                offered,
                accepted_fifo_cutoff,
                incomplete_block,
                source_prefix_bytes,
                source_prefix_sha256,
            } => {
                if *offered != self.offered
                    || *accepted_fifo_cutoff != self.last_fifo
                    || *incomplete_block != self.opened.is_some()
                    || (*source_prefix_bytes, *source_prefix_sha256) != self.source_receipt()
                    || closing.monotonic_ns
                        < self.last_observed.max(self.header.opening.monotonic_ns)
                {
                    return Err(invalid("source7 footer population/clock differs"));
                }
                self.closed = true;
            }
        }
        self.append(r)
    }
}
/// Replay a canonical immutable source prefix ending at a complete Checkpoint.
/// Every original offered record is validated exactly once before membership.
/// Later records in a journal are not silently interpreted as this checkpoint.
pub fn replay_structured_source_v7(
    bytes: &[u8],
    limits: &CostProfileLoadLimits,
) -> Result<StructuredServiceCheckpointV7, CostProfileError> {
    limits.validate()?;
    if bytes.is_empty() || bytes.len() > limits.max_file_bytes.get() || bytes.last() != Some(&b'\n')
    {
        return Err(invalid("source7 incomplete/oversized checkpoint prefix"));
    }
    let mut lines = bytes.split_inclusive(|b| *b == b'\n');
    let first = lines
        .next()
        .ok_or_else(|| invalid("source7 missing header"))?;
    if first.len() > 8 * 1024 * 1024 {
        return Err(CostProfileError::Limit("source7 header line limit"));
    }
    let header: StructuredServiceHeaderV7 = serde_json::from_slice(first)?;
    if record_bytes_v7(&header)? != first {
        return Err(invalid("source7 requires canonical header"));
    }
    let mut collector = StructuredServiceCollectorV7::new(header, limits.clone())?;
    let mut checkpoint = None;
    for line in lines {
        if line.len() > 8 * 1024 * 1024 {
            return Err(CostProfileError::Limit("source7 record line limit"));
        }
        let record: StructuredServiceRecordV7 = serde_json::from_slice(line)?;
        if record_bytes_v7(&record)? != line {
            return Err(invalid("source7 requires canonical records"));
        }
        collector.push(&record)?;
        checkpoint = match record {
            StructuredServiceRecordV7::Checkpoint { closing, .. } => Some(closing),
            _ => None,
        };
    }
    StructuredServiceCheckpointV7::from_collector(
        &collector,
        checkpoint.ok_or_else(|| invalid("source7 prefix lacks final complete checkpoint"))?,
    )
}
