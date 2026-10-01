use super::*;

/// Exact original tail population, not a numerical phase or a complete block.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuredPreparedPartialTailV8 {
    pub block: u64,
    pub closing: StructuredServiceClockV7,
    pub offered: u64,
    pub tail_offered: usize,
    pub accepted_fifo_cutoff: u64,
    pub source_prefix_bytes: u64,
    pub source_prefix_sha256: [u8; 32],
    pub route_population: StructuredServiceRouteCountsV1,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum StructuredPreparedTailRecordV8 {
    PartialTailClosed {
        tail: StructuredPreparedPartialTailV8,
    },
}

impl StructuredPreparedOwnerBlockCollectorV8 {
    /// Seal only after every original declared cohort has fully completed.
    /// An incomplete tail is kept in the source, but neither numerical state
    /// nor discovery may be frozen from it or continued after this seal.
    pub fn seal_complete_cohorts_with_partial_tail(
        &mut self,
        closing: StructuredServiceClockV7,
    ) -> Result<Option<StructuredPreparedOwnerBlockRecordV8>, CostProfileError> {
        let result = (|| {
            self.population.ensure_active()?;
            self.lifecycle.complete()?;
            let Some(tail) = self.population.prepared_tail_record(closing)? else {
                return Ok(None);
            };
            let record = StructuredPreparedOwnerBlockRecordV8::Tail(tail);
            self.push(&record)?;
            Ok(Some(record))
        })();
        if result.is_err() {
            self.population.poison();
        }
        result
    }
}
