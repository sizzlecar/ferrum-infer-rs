use super::*;

impl StructuredServiceCollectorV7 {
    pub(in super::super) fn prepared_tail_audit(&self) -> Option<StructuredPreparedPartialTailV8> {
        self.prepared_tail.clone()
    }

    pub(in super::super) fn prepared_tail_record(
        &self,
        closing: StructuredServiceClockV7,
    ) -> Result<Option<StructuredPreparedTailRecordV8>, CostProfileError> {
        if self.header.source_kind != PopulationSource::PreparedOwnerBlocksV8
            || self.poisoned
            || self.closed
            || self.prepared_tail.is_some()
        {
            return Err(invalid("source8 partial tail state/protocol differs"));
        }
        if self.opened.is_none() {
            if self.last_close != Some(closing) {
                return Err(invalid("source8 completed block closing differs"));
            }
            return Ok(None);
        }
        if self.block_count == 0
            || self.block_count >= self.header.declaration.schedule.block_offered
            || closing.monotonic_ns < self.last_observed.max(self.opened.unwrap_or(0))
            || closing.wall_unix_ns == 0
            || closing.monotonic_ns > self.epoch_deadline()?
        {
            return Err(invalid(
                "source8 tail is empty/full or closing clock differs",
            ));
        }
        let (source_prefix_bytes, source_prefix_sha256) = self.source_receipt();
        Ok(Some(StructuredPreparedTailRecordV8::PartialTailClosed {
            tail: StructuredPreparedPartialTailV8 {
                block: self.block,
                closing,
                offered: self.offered,
                tail_offered: self.block_count,
                accepted_fifo_cutoff: self.last_fifo,
                source_prefix_bytes,
                source_prefix_sha256,
                route_population: self.block_routes,
            },
        }))
    }

    pub(in super::super) fn push_prepared_tail(
        &mut self,
        record: &StructuredPreparedTailRecordV8,
    ) -> Result<(), CostProfileError> {
        let StructuredPreparedTailRecordV8::PartialTailClosed { tail } = record;
        let expected = self
            .prepared_tail_record(tail.closing)?
            .ok_or_else(|| invalid("source8 tail has no original open block"))?;
        if canonical_value_v7(&expected)? != canonical_value_v7(record)? {
            return Err(invalid("source8 original tail population/hash differs"));
        }
        // The complete original samples remain charged and auditable. We drop
        // only unfinished discovery, without publishing any new scope, fitting
        // any parameter, or changing the owner phase. No later block can open.
        self.discovery = None;
        self.opened = None;
        self.last_close = Some(tail.closing);
        self.prepared_tail = Some(tail.clone());
        self.check_retained()
    }
}
