//! Immutable numerical rows validated for one cost query. No live authority.
use super::*;

pub(crate) struct ValidatedCostRows<'a> {
    rows: &'a [OperationCostWorkRow],
    packed_starts: Vec<u64>,
    immediate_tokens: u64,
}

impl<'a> ValidatedCostRows<'a> {
    pub(crate) fn new(rows: &'a [OperationCostWorkRow]) -> Result<Self, VNextError> {
        let mut builder = ValidatedCostRowsBuilder::new(rows)?;
        for _ in rows {
            builder.push_next()?;
        }
        builder.finish()
    }

    pub(crate) fn rows(&self) -> &'a [OperationCostWorkRow] {
        self.rows
    }
    pub(crate) fn immediate_tokens(&self) -> u64 {
        self.immediate_tokens
    }
    pub(super) fn packed_starts(&self) -> &[u64] {
        &self.packed_starts
    }
    pub(crate) fn packed_start(&self, index: usize) -> Option<u64> {
        self.packed_starts.get(index).copied()
    }
}

/// The core projection interleaves authority/frontier and row validation.
/// Advancing at its original validation point preserves that error and budget
/// order. A partial builder cannot be passed to a provider or transfer query.
pub(crate) struct ValidatedCostRowsBuilder<'a> {
    validated: ValidatedCostRows<'a>,
}

impl<'a> ValidatedCostRowsBuilder<'a> {
    pub(crate) fn new(rows: &'a [OperationCostWorkRow]) -> Result<Self, VNextError> {
        if rows.is_empty() || rows.len() > MAX_COST_ROWS {
            return Err(invalid_operation(
                "cost route requires a bounded non-empty work shape",
            ));
        }
        let mut packed_starts = Vec::new();
        packed_starts.try_reserve_exact(rows.len()).map_err(|_| {
            invalid_operation("cost route row-prefix allocation capacity unavailable")
        })?;
        Ok(Self {
            validated: ValidatedCostRows {
                rows,
                packed_starts,
                immediate_tokens: 0,
            },
        })
    }

    pub(crate) fn push_next(&mut self) -> Result<u64, VNextError> {
        let rows = &mut self.validated;
        let row = rows
            .rows
            .get(rows.packed_starts.len())
            .ok_or_else(|| invalid_operation("cost route row validation exceeds its work shape"))?;
        let end = row
            .offset
            .checked_add(row.count.get())
            .filter(|&end| end <= row.full_input_tokens.get())
            .ok_or_else(|| invalid_operation("cost route work exceeds its full input"))?;
        let total = rows
            .immediate_tokens
            .checked_add(row.count.get())
            .ok_or_else(|| invalid_operation("cost route immediate token count overflows"))?;
        rows.packed_starts.push(rows.immediate_tokens);
        rows.immediate_tokens = total;
        Ok(end)
    }

    pub(crate) fn finish(self) -> Result<ValidatedCostRows<'a>, VNextError> {
        if self.validated.packed_starts.len() != self.validated.rows.len() {
            return Err(invalid_operation("cost route row validation is incomplete"));
        }
        Ok(self.validated)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn row(offset: u64, count: u64, full: u64) -> OperationCostWorkRow {
        OperationCostWorkRow {
            offset,
            count: NonZeroU64::new(count).unwrap(),
            full_input_tokens: NonZeroU64::new(full).unwrap(),
        }
    }

    #[test]
    fn validated_cost_rows_keep_original_order_checked_total_and_prefixes() {
        let raw = [row(9, 2, 11), row(0, 5, 8), row(7, 1, 8)];
        let rows = ValidatedCostRows::new(&raw).unwrap();
        assert!(std::ptr::eq(rows.rows().as_ptr(), raw.as_ptr()));
        assert_eq!(rows.rows(), &raw);
        assert_eq!(rows.immediate_tokens(), 8);
        assert_eq!(
            (0..4).map(|i| rows.packed_start(i)).collect::<Vec<_>>(),
            vec![Some(0), Some(2), Some(7), None]
        );
        let max = [row(0, u64::MAX, u64::MAX)];
        let rows = ValidatedCostRows::new(&max).unwrap();
        assert_eq!(rows.immediate_tokens(), u64::MAX);
        assert_eq!(rows.packed_start(0), Some(0));
    }

    #[test]
    fn validated_cost_rows_cannot_publish_partial_or_overflowed_work() {
        let raw = [row(0, u64::MAX, u64::MAX), row(0, 1, 1)];
        let mut builder = ValidatedCostRowsBuilder::new(&raw).unwrap();
        assert_eq!(builder.push_next().unwrap(), u64::MAX);
        assert!(builder
            .push_next()
            .unwrap_err()
            .to_string()
            .contains("immediate token count overflows"));
        assert!(builder.finish().is_err());
        let raw = [row(1, 1, 2)];
        assert!(ValidatedCostRowsBuilder::new(&raw)
            .unwrap()
            .finish()
            .is_err());
        let mut builder = ValidatedCostRowsBuilder::new(&raw).unwrap();
        builder.push_next().unwrap();
        assert!(builder.push_next().is_err());
        assert_eq!(builder.finish().unwrap().immediate_tokens(), 1);
        for raw in [vec![row(2, 4, 5)], vec![row(u64::MAX, 1, u64::MAX)]] {
            assert!(ValidatedCostRows::new(&raw)
                .err()
                .unwrap()
                .to_string()
                .contains("work exceeds its full input"));
        }
    }

    #[test]
    fn validated_cost_rows_match_actual_step_upload_and_readback_coordinates() {
        use super::super::super::buffer_view::{
            translate_step_participant_numeric_range, translate_step_participant_readback_range,
            translate_step_participant_upload_range, StepParticipantRangeCoordinates,
        };
        use crate::vnext::{BatchWorkShape, DynamicResourceDemand, TokenSpanWork};
        let first = vec![1_u32; 20];
        let second = vec![2_u32; 9];
        let actual = BatchWorkShape::test_only(vec![
            TokenSpanWork::from_token_ids(&first, 17..18).unwrap(),
            TokenSpanWork::from_token_ids(&second, 7..9).unwrap(),
        ])
        .unwrap();
        let raw = [row(17, 1, 20), row(7, 2, 9)];
        let validated = ValidatedCostRows::new(&raw).unwrap();
        for demand in [
            DynamicResourceDemand::tokens(4, 32).unwrap(),
            DynamicResourceDemand::actual_sequences(16, 4).unwrap(),
            DynamicResourceDemand::fixed(16).unwrap(),
        ] {
            for (index, row) in raw.iter().enumerate() {
                for range in [0..4, 4..12, 28..36, 68..72, 0..0, u64::MAX - 1..u64::MAX] {
                    for (coordinates, old) in [
                        (
                            StepParticipantRangeCoordinates::SourceToken,
                            translate_step_participant_upload_range(
                                &demand,
                                &actual,
                                index,
                                range.clone(),
                            ),
                        ),
                        (
                            StepParticipantRangeCoordinates::ParticipantLocal,
                            translate_step_participant_readback_range(
                                &demand,
                                &actual,
                                index,
                                range.clone(),
                            ),
                        ),
                    ] {
                        let new = translate_step_participant_numeric_range(
                            &demand,
                            raw.len() as u32,
                            validated.immediate_tokens(),
                            index,
                            row.offset..row.offset + row.count.get(),
                            validated.packed_start(index).unwrap(),
                            range.clone(),
                            coordinates,
                        );
                        assert_eq!(
                            new.map_err(|e| e.to_string()),
                            old.map_err(|e| e.to_string())
                        );
                    }
                }
            }
        }
    }
}
