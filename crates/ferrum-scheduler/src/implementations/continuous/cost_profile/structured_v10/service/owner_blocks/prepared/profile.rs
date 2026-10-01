//! Profile15 preserves the original source8 preparation proof and clocks.
//! Numerical assembly is shared with profile14 behind a private typed protocol
//! check; neither public loader can reinterpret the other source format.
use super::super::collector::PopulationSource;
use super::super::profile::{
    metadata_bytes, streaming_metadata_bytes, validate_domain_selection, ImportedCatalogParts,
    OwnerCatalogEnvelope, MAX_METADATA,
};
use super::collector::StructuredPreparedOwnerBlockCheckpointV8;
use super::*;

mod persisted;
pub use persisted::{
    export_structured_profile_v15, export_structured_profile_v15_same_boot,
    export_structured_profile_v15_same_boot_selected,
    export_structured_profile_v15_same_boot_selected_from_original_bytes,
    load_structured_profile_v15, load_structured_profile_v15_same_boot,
    load_structured_profile_v15_same_boot_selected,
    load_structured_profile_v15_same_boot_selected_from_original_bytes,
};
pub type StructuredProfileExportReceiptV15 = StructuredProfileExportReceiptV13;
const KIND: PopulationSource = PopulationSource::PreparedOwnerBlocksV8;
#[cfg(test)]
mod tests;

#[derive(Debug, Clone)]
pub struct ImportedPreparedOwnerBlockCatalogV15 {
    workload_domain: Option<ferrum_interfaces::execution_cost::CostWorkloadDomainV1>,
    pub children: Vec<ImportedStructuredModelV2>,
    pub file_sha256: [u8; 32],
    pub source_sha256: [u8; 32],
    pub capture_protocol: [u8; 32],
    pub file_bytes: u64,
    pub source_bytes: u64,
    pub journal_bytes: u64,
    pub total_shape_rows: u64,
    pub offered_attempts: u64,
}
impl ImportedPreparedOwnerBlockCatalogV15 {
    pub fn workload_domain(
        &self,
    ) -> Option<&ferrum_interfaces::execution_cost::CostWorkloadDomainV1> {
        self.workload_domain.as_ref()
    }
}
impl From<ImportedCatalogParts> for ImportedPreparedOwnerBlockCatalogV15 {
    fn from(parts: ImportedCatalogParts) -> Self {
        Self {
            workload_domain: parts.workload_domain,
            children: parts.children,
            file_sha256: parts.file_sha256,
            source_sha256: parts.source_sha256,
            capture_protocol: parts.capture_protocol,
            file_bytes: parts.file_bytes,
            source_bytes: parts.source_bytes,
            journal_bytes: parts.journal_bytes,
            total_shape_rows: parts.total_shape_rows,
            offered_attempts: parts.offered_attempts,
        }
    }
}

impl StructuredPreparedOwnerBlockCheckpointV8 {
    pub fn source_receipt(&self) -> (u64, [u8; 32]) {
        self.population.source_receipt()
    }
    pub fn qualified_children(&self) -> usize {
        self.population.qualified_children()
    }
    /// Runtime activation must drain and match this original accepted cut.
    pub fn accepted_fifo_cutoff(&self) -> u64 {
        self.population.accepted_fifo_cutoff
    }

    fn validate_original_source(&self) -> Result<(), CostProfileError> {
        self.header.validate()?;
        let original = &self.header;
        let numerical = &self.population.header;
        if numerical.source_kind != KIND
            || numerical.capture_identity != original.capture_identity
            || numerical.protocol != original.protocol
            || numerical.fingerprint != original.fingerprint
            || numerical.monotonic_domain != original.monotonic_domain
            || numerical.opening.monotonic_ns != original.opening.monotonic_ns
            || numerical.opening.wall_unix_ns != original.opening.wall_unix_ns
            || numerical.declaration_sha256 != original.declaration_sha256
            || numerical.maximum_file_bytes != original.maximum_file_bytes
            || record_bytes_v7(&numerical.producer)? != record_bytes_v7(&original.producer)?
            || record_bytes_v7(&numerical.declaration)?
                != record_bytes_v7(&original.declaration.population)?
        {
            return Err(invalid(
                "profile15 numerical checkpoint lost its source8 preparation binding",
            ));
        }
        Ok(())
    }

    pub fn activate_same_process_memory(
        self,
        now: StructuredServiceClockV7,
        limits: &CostProfileLoadLimits,
    ) -> Result<ImportedPreparedOwnerBlockCatalogV15, CostProfileError> {
        self.activate_memory_with_budget(now, limits, None, None)
    }
    /// Activate the original source8 checkpoint using its explicit incremental
    /// encoding budget, independently of an offline file allocation allowance.
    pub fn activate_same_process_memory_streaming(
        self,
        now: StructuredServiceClockV7,
        limits: &CostProfileLoadLimits,
        maximum_encoded_source_bytes: std::num::NonZeroU64,
    ) -> Result<ImportedPreparedOwnerBlockCatalogV15, CostProfileError> {
        self.activate_memory_with_budget(now, limits, Some(maximum_encoded_source_bytes), None)
    }
    /// Reuse only an originally bound boot-relative checkpoint. No timestamp is renewed.
    pub fn activate_same_boot_memory(
        self,
        now: StructuredServiceClockV7,
        domain: &ferrum_interfaces::execution_cost::CostMonotonicDomainV1,
        limits: &CostProfileLoadLimits,
    ) -> Result<ImportedPreparedOwnerBlockCatalogV15, CostProfileError> {
        self.activate_memory_with_budget(now, limits, None, Some(domain))
    }
    pub fn activate_same_boot_memory_streaming(
        self,
        now: StructuredServiceClockV7,
        domain: &ferrum_interfaces::execution_cost::CostMonotonicDomainV1,
        limits: &CostProfileLoadLimits,
        maximum_encoded_source_bytes: std::num::NonZeroU64,
    ) -> Result<ImportedPreparedOwnerBlockCatalogV15, CostProfileError> {
        self.activate_memory_with_budget(
            now,
            limits,
            Some(maximum_encoded_source_bytes),
            Some(domain),
        )
    }
    fn activate_memory_with_budget(
        self,
        now: StructuredServiceClockV7,
        limits: &CostProfileLoadLimits,
        maximum_encoded_source_bytes: Option<std::num::NonZeroU64>,
        monotonic_domain: Option<&ferrum_interfaces::execution_cost::CostMonotonicDomainV1>,
    ) -> Result<ImportedPreparedOwnerBlockCatalogV15, CostProfileError> {
        limits.validate()?;
        self.validate_original_source()?;
        let mapped = self
            .population
            .children
            .iter()
            .map(|child| {
                let mapped = match monotonic_domain {
                    Some(domain) => clock::same_boot(
                        self.population.evidence(child),
                        self.header.monotonic_domain.as_ref(),
                        domain,
                        now.monotonic_ns,
                        now.wall_unix_ns,
                        limits,
                    )?,
                    None => clock::same_process(
                        self.population.evidence(child),
                        now.monotonic_ns,
                        now.wall_unix_ns,
                        limits,
                    )?,
                };
                validate_model_clock(child, &mapped)?;
                Ok::<_, CostProfileError>(mapped)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let declared = self.population.envelope_for(KIND)?;
        let bytes = match maximum_encoded_source_bytes {
            Some(budget) => {
                streaming_metadata_bytes(&declared, self.header.maximum_file_bytes, budget)?
            }
            None => metadata_bytes(&declared, limits)?,
        };
        self.population
            .assemble_for(
                declared,
                None,
                &bytes,
                mapped,
                if monotonic_domain.is_some() {
                    ferrum_types::SloCostProfileClockBasis::SameBootMonotonic
                } else {
                    ferrum_types::SloCostProfileClockBasis::SameProcessMonotonic
                },
                KIND,
            )
            .map(Into::into)
    }
}

// The owner epoch can expire before its oldest numerical sample does. Both
// in-memory activation and persisted import must retain that original limit.
fn validate_model_clock(
    child: &super::super::profile::FrozenChild,
    mapped: &clock::Mapped,
) -> Result<(), CostProfileError> {
    child
        .model
        .validate_runtime_at(mapped.model_now)
        .map_err(|_| CostProfileError::Clock("source8 original qualified owner epoch expired"))
}
