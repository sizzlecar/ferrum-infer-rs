use super::*;
mod persisted;
pub use persisted::{
    export_structured_profile_v14, export_structured_profile_v14_same_boot,
    export_structured_profile_v14_same_boot_selected,
    export_structured_profile_v14_same_boot_selected_from_original_bytes,
    load_structured_profile_v14, load_structured_profile_v14_same_boot,
    load_structured_profile_v14_same_boot_selected,
    load_structured_profile_v14_same_boot_selected_from_original_bytes,
};
pub type StructuredProfileExportReceiptV14 = StructuredProfileExportReceiptV13;
pub(super) const MAX_METADATA: usize = 2 * 1024 * 1024;
/// Encoder-enforced metadata bound shared by profile14 and profile15.
/// Allows a cold caller to reserve output storage without reserving its entire
/// independent source-input limit a second time.
pub fn structured_owner_block_metadata_maximum_bytes() -> usize {
    MAX_METADATA
}
const ARTIFACT: &str = "ferrum.structured-owner-block-catalog";
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Child {
    pub(super) owner_attempt_id: u64,
    pub(super) owner: StructuredOwnerKeyV2,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) numerical_family:
        Option<crate::implementations::continuous::cost_model::structured_v2::NumericalFamilyKeyV1>,
    pub(super) domain_signature: [u8; 32],
    pub(super) parameters_sha256: [u8; 32],
    pub(super) phases: [StructuredPhaseProvenanceV10; 3],
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct OwnerCatalogEnvelope {
    pub(super) artifact_type: String,
    pub(super) schema_version: u32,
    pub(super) model_revision: String,
    pub(super) fingerprint: ProfileFingerprint,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) source_path: Option<PathBuf>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) journal_bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) journal_sha256: Option<[u8; 32]>,
    pub(super) source_bytes: u64,
    pub(super) source_sha256: [u8; 32],
    pub(super) capture_protocol: [u8; 32],
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) monotonic_domain: Option<ferrum_interfaces::execution_cost::CostMonotonicDomainV1>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) source_clock_max_error_ns: Option<u64>,
    #[serde(deserialize_with = "bounded_rows")]
    pub(super) children: Vec<Child>,
}
#[derive(Debug, Clone)]
pub struct ImportedStructuredCatalogV14 {
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

pub(super) struct ImportedCatalogParts {
    pub(super) workload_domain: Option<ferrum_interfaces::execution_cost::CostWorkloadDomainV1>,
    pub(super) children: Vec<ImportedStructuredModelV2>,
    pub(super) file_sha256: [u8; 32],
    pub(super) source_sha256: [u8; 32],
    pub(super) capture_protocol: [u8; 32],
    pub(super) file_bytes: u64,
    pub(super) source_bytes: u64,
    pub(super) journal_bytes: u64,
    pub(super) total_shape_rows: u64,
    pub(super) offered_attempts: u64,
}
impl From<ImportedCatalogParts> for ImportedStructuredCatalogV14 {
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

impl ImportedStructuredCatalogV14 {
    pub fn workload_domain(
        &self,
    ) -> Option<&ferrum_interfaces::execution_cost::CostWorkloadDomainV1> {
        self.workload_domain.as_ref()
    }
}
pub(super) struct FrozenChild {
    pub(super) model: Arc<QualifiedStructuredModelV2>,
    pub(super) metadata: Child,
    pub(super) oldest: u64,
    pub(super) newest: u64,
}

/// Selection never changes the original replayed population or its budget.
/// It chooses which already-qualified children require validity at import time.
pub(super) fn validate_domain_selection(
    children: &[FrozenChild],
    selected: Option<&[[u8; 32]]>,
) -> Result<(), CostProfileError> {
    if let Some(domains) = selected {
        if domains.is_empty()
            || domains.len() > 128
            || domains.windows(2).any(|pair| pair[0] >= pair[1])
            || domains.iter().any(|domain| {
                !children
                    .iter()
                    .any(|child| child.metadata.domain_signature == *domain)
            })
        {
            return Err(invalid(
                "selected domains differ from original qualified population",
            ));
        }
    }
    Ok(())
}

/// Frozen complete original prefix. Pending owners remain pending in the source;
/// only already qualified children are shared with this immutable snapshot.
pub struct StructuredServiceCheckpointV7 {
    pub(super) header: collector::PopulationHeader,
    pub(super) closing: StructuredServiceClockV7,
    pub(super) children: Vec<FrozenChild>,
    pub(super) source_bytes: u64,
    pub(super) source_sha256: [u8; 32],
    pub(super) offered: u64,
    pub(super) total_rows: u64,
    pub(super) accepted_fifo_cutoff: u64,
}
impl StructuredServiceCheckpointV7 {
    pub(super) fn from_collector(
        c: &StructuredServiceCollectorV7,
        closing: StructuredServiceClockV7,
    ) -> Result<Self, CostProfileError> {
        let mut children = Vec::new();
        for o in &c.owners {
            if let collector::State::Qualified(model) = &o.state {
                children.push(FrozenChild {
                    model: model.clone(),
                    oldest: o.oldest,
                    newest: o.newest,
                    metadata: Child {
                        owner_attempt_id: o.contract.owner_attempt_id,
                        owner: model.owner().clone(),
                        numerical_family: model.numerical_family_key().copied(),
                        domain_signature: *model.domain_signature(),
                        parameters_sha256: model.parameters_signature(),
                        phases: o.phases.clone().try_into().map_err(|_| {
                            invalid("source7 child missing independent three freezes")
                        })?,
                    },
                });
            }
        }
        let (source_bytes, source_sha256) = c.source_receipt();
        Ok(Self {
            header: c.header.clone(),
            closing,
            children,
            source_bytes,
            source_sha256,
            offered: c.offered,
            total_rows: c.total_rows,
            accepted_fifo_cutoff: c.last_fifo(),
        })
    }
    pub fn source_receipt(&self) -> (u64, [u8; 32]) {
        (self.source_bytes, self.source_sha256)
    }
    pub fn qualified_children(&self) -> usize {
        self.children.len()
    }
    pub(super) fn evidence(&self, c: &FrozenChild) -> clock::Evidence {
        clock::Evidence {
            opening: self.header.opening.into(),
            closing: self.closing.into(),
            oldest_observed: c.oldest,
            newest_observed: c.newest,
            max_age_ns: self.header.declaration.settings.max_sample_age_ns,
        }
    }
    fn envelope(&self) -> OwnerCatalogEnvelope {
        OwnerCatalogEnvelope {
            artifact_type: ARTIFACT.into(),
            schema_version: 14,
            model_revision: MODEL_REVISION_V2.into(),
            fingerprint: self.header.fingerprint.clone(),
            source_path: None,
            journal_bytes: None,
            journal_sha256: None,
            source_bytes: self.source_bytes,
            source_sha256: self.source_sha256,
            capture_protocol: self.header.protocol,
            monotonic_domain: self.header.monotonic_domain.clone(),
            source_clock_max_error_ns: None,
            children: self.children.iter().map(|c| c.metadata.clone()).collect(),
        }
    }
    pub(super) fn envelope_for(
        &self,
        kind: collector::PopulationSource,
    ) -> Result<OwnerCatalogEnvelope, CostProfileError> {
        if self.header.source_kind != kind {
            return Err(invalid("catalog schema differs from original source"));
        }
        let mut envelope = self.envelope();
        envelope.schema_version = kind.profile_schema();
        envelope.artifact_type = kind.profile_artifact().into();
        Ok(envelope)
    }
    pub fn activate_same_process_memory(
        self,
        now: StructuredServiceClockV7,
        limits: &CostProfileLoadLimits,
    ) -> Result<ImportedStructuredCatalogV14, CostProfileError> {
        self.activate_memory_with_budget(now, limits, None, None)
    }
    /// Consume a genuine checkpoint without retaining or rereading its source
    /// stream. Its original cumulative work budget remains independently bound.
    pub fn activate_same_process_memory_streaming(
        self,
        now: StructuredServiceClockV7,
        limits: &CostProfileLoadLimits,
        maximum_encoded_source_bytes: std::num::NonZeroU64,
    ) -> Result<ImportedStructuredCatalogV14, CostProfileError> {
        self.activate_memory_with_budget(now, limits, Some(maximum_encoded_source_bytes), None)
    }
    /// Reuse only an originally bound boot-relative checkpoint. No timestamp is renewed.
    pub fn activate_same_boot_memory(
        self,
        now: StructuredServiceClockV7,
        domain: &ferrum_interfaces::execution_cost::CostMonotonicDomainV1,
        limits: &CostProfileLoadLimits,
    ) -> Result<ImportedStructuredCatalogV14, CostProfileError> {
        self.activate_memory_with_budget(now, limits, None, Some(domain))
    }
    pub fn activate_same_boot_memory_streaming(
        self,
        now: StructuredServiceClockV7,
        domain: &ferrum_interfaces::execution_cost::CostMonotonicDomainV1,
        limits: &CostProfileLoadLimits,
        maximum_encoded_source_bytes: std::num::NonZeroU64,
    ) -> Result<ImportedStructuredCatalogV14, CostProfileError> {
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
    ) -> Result<ImportedStructuredCatalogV14, CostProfileError> {
        if self.header.source_kind != collector::PopulationSource::OwnerBlocksV7 {
            return Err(invalid("profile14 requires original source7"));
        }
        limits.validate()?;
        let mapped = self
            .children
            .iter()
            .map(|c| match monotonic_domain {
                Some(domain) => {
                    let mapped = clock::same_boot(
                        self.evidence(c),
                        self.header.monotonic_domain.as_ref(),
                        domain,
                        now.monotonic_ns,
                        now.wall_unix_ns,
                        limits,
                    )?;
                    c.model.validate_runtime_at(mapped.model_now).map_err(|_| {
                        CostProfileError::Clock("original source7 owner epoch expired")
                    })?;
                    Ok(mapped)
                }
                None => clock::same_process(
                    self.evidence(c),
                    now.monotonic_ns,
                    now.wall_unix_ns,
                    limits,
                ),
            })
            .collect::<Result<Vec<_>, _>>()?;
        let envelope = self.envelope();
        let bytes = match maximum_encoded_source_bytes {
            Some(budget) => {
                streaming_metadata_bytes(&envelope, self.header.maximum_file_bytes, budget)?
            }
            None => metadata_bytes(&envelope, limits)?,
        };
        self.assemble(
            envelope,
            None,
            &bytes,
            mapped,
            if monotonic_domain.is_some() {
                ferrum_types::SloCostProfileClockBasis::SameBootMonotonic
            } else {
                ferrum_types::SloCostProfileClockBasis::SameProcessMonotonic
            },
        )
    }
    fn assemble(
        self,
        declared: OwnerCatalogEnvelope,
        path: Option<&Path>,
        bytes: &[u8],
        mapped: Vec<clock::Mapped>,
        clock_basis: ferrum_types::SloCostProfileClockBasis,
    ) -> Result<ImportedStructuredCatalogV14, CostProfileError> {
        self.assemble_for(
            declared,
            path,
            bytes,
            mapped,
            clock_basis,
            collector::PopulationSource::OwnerBlocksV7,
        )
        .map(Into::into)
    }

    pub(super) fn validate_original_metadata(
        &self,
        declared: &OwnerCatalogEnvelope,
        kind: collector::PopulationSource,
    ) -> Result<(), CostProfileError> {
        if self.header.source_kind != kind
            || declared.schema_version != kind.profile_schema()
            || declared.artifact_type != kind.profile_artifact()
            || declared.fingerprint != self.header.fingerprint
            || declared.capture_protocol != self.header.protocol
            || declared.monotonic_domain != self.header.monotonic_domain
            || declared.source_sha256 != self.source_sha256
            || declared.source_bytes != self.source_bytes
        {
            return Err(invalid(
                "owner catalog differs from its original source protocol",
            ));
        }
        if self.children.is_empty()
            || declared.children.len() != self.children.len()
            || self
                .children
                .iter()
                .zip(&declared.children)
                .any(|(child, declared)| &child.metadata != declared)
        {
            return Err(invalid("profile frozen child inventory differs"));
        }
        Ok(())
    }

    pub(super) fn assemble_for(
        self,
        declared: OwnerCatalogEnvelope,
        path: Option<&Path>,
        bytes: &[u8],
        mapped: Vec<clock::Mapped>,
        clock_basis: ferrum_types::SloCostProfileClockBasis,
        kind: collector::PopulationSource,
    ) -> Result<ImportedCatalogParts, CostProfileError> {
        self.validate_original_metadata(&declared, kind)?;
        if self.children.is_empty()
            || self.children.len() != mapped.len()
            || declared.children.len() != self.children.len()
        {
            return Err(invalid("profile14 requires qualified children"));
        }
        let file_sha256 = Sha256::digest(bytes).into();
        let file_bytes = bytes.len() as u64;
        let mut children = Vec::with_capacity(self.children.len());
        for ((child, mapped), d) in self.children.into_iter().zip(mapped).zip(declared.children) {
            if child.metadata != d {
                return Err(invalid("profile14 frozen child differs"));
            }
            let model = child.model;
            let contract = model
                .owner_block_contract()
                .ok_or_else(|| invalid("profile14 child lacks owner block population"))?;
            let rule_signature = contract.membership_rule;
            children.push(ImportedStructuredModelV2 {
                fingerprint: self.header.fingerprint.clone().into(),
                scope: model.scope().clone(),
                domain: *model.domain_signature(),
                model,
                provenance: StructuredImportProvenanceV10 {
                    monotonic_domain: self.header.monotonic_domain.clone(),
                    storage: if path.is_some() {
                        ferrum_types::SloCostProfileStorage::File
                    } else {
                        ferrum_types::SloCostProfileStorage::Memory
                    },
                    clock_basis,
                    schema_version: kind.profile_schema(),
                    loaded_from: path.map(Path::to_path_buf),
                    source_path: declared.source_path.clone(),
                    file_sha256,
                    source_sha256: self.source_sha256,
                    parameters_sha256: d.parameters_sha256,
                    capture_identity: self.header.capture_identity,
                    protocol: self.header.protocol,
                    rule_signature,
                    cohort_manifest_sha256: self.header.declaration_sha256,
                    offered_attempts: self.offered,
                    reserved_members: d.phases.iter().map(|p| p.members as u64).sum(),
                    total_shape_rows: self.total_rows,
                    file_bytes,
                    source_bytes: self.source_bytes,
                    conservative_clock_error_ns: mapped.error,
                    oldest_imported_age_ns: mapped.oldest_age,
                    newest_imported_age_ns: mapped.newest_age,
                    loaded_unix_ns: mapped.wall,
                    generated_unix_ns: self.closing.wall_unix_ns,
                    clock: mapped.clock,
                    producer: serde_json::value::to_raw_value(&self.header.producer)?,
                    phases: d.phases,
                },
            });
        }
        let journal_bytes = declared.journal_bytes.unwrap_or(self.source_bytes);
        Ok(ImportedCatalogParts {
            workload_domain: self
                .header
                .declaration
                .nonnegative_envelope
                .map(|c| c.workload_domain),
            children,
            file_sha256,
            source_sha256: self.source_sha256,
            capture_protocol: self.header.protocol,
            file_bytes,
            source_bytes: self.source_bytes,
            journal_bytes,
            total_shape_rows: self.total_rows,
            offered_attempts: self.offered,
        })
    }
}
pub(super) fn metadata_bytes(
    declared: &OwnerCatalogEnvelope,
    limits: &CostProfileLoadLimits,
) -> Result<Vec<u8>, CostProfileError> {
    let mut bytes = serde_json::to_vec_pretty(declared)?;
    bytes.push(b'\n');
    if bytes.len() > MAX_METADATA
        || declared
            .journal_bytes
            .unwrap_or(declared.source_bytes)
            .checked_add(bytes.len() as u64)
            .is_none_or(|n| n > limits.max_file_bytes.get() as u64)
    {
        return Err(CostProfileError::Limit(
            "profile14 metadata and journal capacity",
        ));
    }
    Ok(bytes)
}

/// Memory provenance still describes the full canonical source hash/length;
/// only the bounded catalog metadata is materialized here. Offline callers
/// continue to account for the actual source file together with its metadata.
pub(super) fn streaming_metadata_bytes(
    declared: &OwnerCatalogEnvelope,
    original_source_limit: u64,
    maximum_encoded_source_bytes: std::num::NonZeroU64,
) -> Result<Vec<u8>, CostProfileError> {
    if original_source_limit != maximum_encoded_source_bytes.get()
        || declared.source_bytes > original_source_limit
        || declared.source_path.is_some()
        || declared.journal_bytes.is_some()
        || declared.journal_sha256.is_some()
    {
        return Err(invalid(
            "memory source budget or original journal binding differs",
        ));
    }
    let mut bytes = serde_json::to_vec_pretty(declared)?;
    bytes.push(b'\n');
    if bytes.len() > MAX_METADATA {
        return Err(CostProfileError::Limit(
            "profile14 memory metadata capacity",
        ));
    }
    Ok(bytes)
}

impl collector::PopulationSource {
    pub(super) fn profile_schema(self) -> u32 {
        match self {
            Self::OwnerBlocksV7 => 14,
            Self::PreparedOwnerBlocksV8 => 15,
        }
    }
    pub(super) fn profile_artifact(self) -> &'static str {
        match self {
            Self::OwnerBlocksV7 => ARTIFACT,
            Self::PreparedOwnerBlocksV8 => "ferrum.structured-prepared-owner-block-catalog",
        }
    }
}
