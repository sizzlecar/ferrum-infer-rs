//! Structured schema 10 imports replay-only numbers from a complete original
//! source. No wire DTO can mint a live settlement receipt or execution permit.
use super::super::cost_model::structured_v2::windows::{
    CohortPlanV2, MembershipRuleV2, PreparedRowFactsV2, PreparedWorkV2,
};
use super::super::cost_model::structured_v2::*;
use super::*;
use ferrum_interfaces::execution_cost::IndependentAttentionWaveEvidenceWireV2;
use std::io::Write;
mod clock;
mod lifecycle;
mod observation;
mod prepared;
mod replay;
mod shared;
pub use shared::{
    export_structured_profile_v11, load_structured_profile_v11, structured_shared_source_header_v4,
    ImportedStructuredCatalogV11, StructuredProfileExportReceiptV11,
};
mod wire;
use wire::*;
pub use wire::{StructuredPhaseProvenanceV10, StructuredProfilePhaseV10};
#[cfg(test)]
mod tests;
pub const COST_PROFILE_SCHEMA_VERSION_V10: u32 = 10;

#[derive(Debug, Clone, Serialize)]
pub struct StructuredImportProvenanceV10 {
    pub schema_version: u32,
    pub loaded_from: PathBuf,
    pub source_path: PathBuf,
    pub file_sha256: [u8; 32],
    pub source_sha256: [u8; 32],
    pub parameters_sha256: [u8; 32],
    pub capture_identity: [u8; 32],
    pub protocol: [u8; 32],
    pub rule_signature: [u8; 32],
    pub cohort_manifest_sha256: [u8; 32],
    pub offered_attempts: u64,
    pub reserved_members: u64,
    /// Original completed rows, including all outside-window FIFO evidence.
    pub total_shape_rows: u64,
    pub file_bytes: u64,
    pub source_bytes: u64,
    pub conservative_clock_error_ns: u64,
    pub oldest_imported_age_ns: u64,
    pub newest_imported_age_ns: u64,
    pub loaded_unix_ns: u64,
    pub generated_unix_ns: u64,
    pub clock: ProfileObservationClock,
    pub producer: serde_json::Value,
    pub phases: [StructuredPhaseProvenanceV10; 3],
}
#[derive(Debug, Clone, Serialize)]
pub struct StructuredProfileExportReceiptV10 {
    pub model_revision: &'static str,
    pub uncertainty: StructuredUncertaintyV2,
    pub path: PathBuf,
    pub source_path: PathBuf,
    pub file_sha256: [u8; 32],
    pub source_sha256: [u8; 32],
    pub parameters_sha256: [u8; 32],
    pub file_bytes: u64,
    pub source_bytes: u64,
    pub phases: [StructuredPhaseProvenanceV10; 3],
}
#[derive(Clone)]
pub struct ImportedStructuredModelV2 {
    model: Arc<QualifiedStructuredModelV2>,
    fingerprint: ExecutionFingerprint,
    scope: StructuredScopeV2,
    domain: [u8; 32],
    provenance: StructuredImportProvenanceV10,
}
impl std::fmt::Debug for ImportedStructuredModelV2 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ImportedStructuredModelV2")
            .field("provenance", &self.provenance)
            .finish_non_exhaustive()
    }
}
impl ImportedStructuredModelV2 {
    pub fn runtime_limits(&self) -> (u64, u64) {
        self.model.runtime_limits()
    }
    pub fn predict_query_local(
        &self,
        fingerprint: &ExecutionFingerprint,
        input: &StructuredQueryV2,
        local_now_ns: u64,
    ) -> Result<StructuredPredictionV2, StructuredUnknownV2> {
        self.predict_query_local_with_clock(fingerprint, input, local_now_ns)
            .map(|(value, _)| value)
    }
    /// Return the same converted epoch used for prediction so adapters can
    /// compute remaining TTL without reading or converting a second clock.
    pub fn predict_query_local_with_clock(
        &self,
        fingerprint: &ExecutionFingerprint,
        input: &StructuredQueryV2,
        local_now_ns: u64,
    ) -> Result<(StructuredPredictionV2, u64), StructuredUnknownV2> {
        let model_now_ns = self.model_now_ns(local_now_ns)?;
        let value = self.model.predict_query(fingerprint, input, model_now_ns)?;
        Ok((value, model_now_ns))
    }
    pub fn model_now_ns(&self, local_now_ns: u64) -> Result<u64, StructuredUnknownV2> {
        self.provenance
            .clock
            .model_now_ns(local_now_ns)
            .map_err(|_| StructuredUnknownV2::Clock)
    }
    pub fn scope(&self) -> &StructuredScopeV2 {
        &self.scope
    }
    pub fn owner(&self) -> &StructuredOwnerKeyV2 {
        &self.scope.owner
    }
    pub fn source_path(&self) -> &PathBuf {
        &self.provenance.source_path
    }
    pub fn domain_signature(&self) -> &[u8; 32] {
        &self.domain
    }
    pub fn parameters_signature(&self) -> [u8; 32] {
        self.provenance.parameters_sha256
    }
    pub fn fingerprint(&self) -> &ExecutionFingerprint {
        &self.fingerprint
    }
    pub fn provenance(&self) -> &StructuredImportProvenanceV10 {
        &self.provenance
    }
}
fn invalid(reason: &'static str) -> CostProfileError {
    CostProfileError::Metadata(reason)
}
fn numeric_error(_: StructuredUnknownV2) -> CostProfileError {
    invalid("invalid structured numerical replay")
}
fn read_bounded(path: &Path, maximum: usize) -> Result<Vec<u8>, CostProfileError> {
    let mut f = File::open(path)?;
    let m = f.metadata()?;
    if !m.is_file() {
        return Err(invalid("structured input must be a regular file"));
    }
    let n = usize::try_from(m.len())
        .ok()
        .filter(|n| *n > 0 && *n <= maximum)
        .ok_or(CostProfileError::Limit("structured file byte limit"))?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(n)
        .map_err(|_| CostProfileError::Limit("structured file allocation"))?;
    bytes.resize(n, 0);
    f.read_exact(&mut bytes)?;
    if f.read(&mut [0u8; 1])? != 0 {
        return Err(invalid("structured file grew while reading"));
    }
    Ok(bytes)
}
fn envelope(
    replayed: &replay::Replayed,
    source_path: PathBuf,
    source_bytes: u64,
    source_sha256: [u8; 32],
    error: u64,
) -> Envelope {
    Envelope {
        schema_version: 10,
        model_revision: MODEL_REVISION_V2.into(),
        fingerprint: replayed.header.fingerprint.clone(),
        source_path,
        source_bytes,
        source_sha256,
        source_clock_max_error_ns: error,
        capture_identity: replayed.header.capture_identity,
        protocol: replayed.header.protocol,
        rule_signature: replayed.header.rule_signature,
        cohort_manifest_sha256: replayed.header.cohort_manifest_sha256,
        parameters_sha256: replayed.model.parameters_signature(),
        phases: replayed.phases.clone(),
    }
}
/// Both export and load replay every original population and its phase digest.
/// Atomic new-name publication refuses to overwrite an earlier artifact.
pub fn export_structured_profile_v10(
    source_path: &Path,
    expected_source_sha256: [u8; 32],
    destination: &Path,
    declared_source_clock_error_ns: u64,
    limits: &CostProfileLoadLimits,
) -> Result<StructuredProfileExportReceiptV10, CostProfileError> {
    limits.validate()?;
    if declared_source_clock_error_ns > limits.max_clock_error_ns {
        return Err(CostProfileError::Clock(
            "source clock error exceeds import policy",
        ));
    }
    let source = read_bounded(source_path, limits.max_file_bytes.get())?;
    let digest: [u8; 32] = Sha256::digest(&source).into();
    if digest != expected_source_sha256 {
        return Err(invalid("structured source digest mismatch"));
    }
    let replayed = replay::replay_source(&source, limits)?;
    clock::validate_source(&replayed, declared_source_clock_error_ns)?;
    // Preserve absolute original source identity even if envelope is relocated.
    let source_path = source_path.canonicalize()?;
    let value = envelope(
        &replayed,
        source_path.clone(),
        source.len() as u64,
        digest,
        declared_source_clock_error_ns,
    );
    let mut bytes = serde_json::to_vec_pretty(&value)?;
    bytes.push(b'\n');
    if bytes
        .len()
        .checked_add(source.len())
        .is_none_or(|n| n > limits.max_file_bytes.get())
    {
        return Err(CostProfileError::Limit(
            "structured envelope and source exceed byte limit",
        ));
    }
    publish_new(destination, &bytes)?;
    Ok(StructuredProfileExportReceiptV10 {
        model_revision: MODEL_REVISION_V2,
        uncertainty: replayed.model.uncertainty(),
        path: destination.into(),
        source_path,
        file_sha256: Sha256::digest(&bytes).into(),
        source_sha256: digest,
        parameters_sha256: value.parameters_sha256,
        file_bytes: bytes.len() as u64,
        source_bytes: source.len() as u64,
        phases: value.phases,
    })
}
fn publish_new(destination: &Path, bytes: &[u8]) -> Result<(), CostProfileError> {
    let parent = destination
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let temporary = parent.join(format!(".ferrum-structured-{}.tmp", uuid::Uuid::new_v4()));
    let result = (|| {
        let mut file = std::fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&temporary)?;
        file.write_all(bytes)?;
        file.sync_all()?;
        // A hard link publishes only a completely written inode and cannot
        // replace an existing name, unlike a platform-dependent rename.
        std::fs::hard_link(&temporary, destination)?;
        Ok::<(), std::io::Error>(())
    })();
    let _ = std::fs::remove_file(&temporary);
    result.map_err(Into::into)
}

pub fn load_structured_profile_v10(
    path: &Path,
    expected_fingerprint: &ExecutionFingerprint,
    limits: &CostProfileLoadLimits,
    load_clock: ProfileLoadClock,
) -> Result<ImportedStructuredModelV2, CostProfileError> {
    limits.validate()?;
    let bytes = read_bounded(path, limits.max_file_bytes.get())?;
    #[derive(Deserialize)]
    struct Version {
        schema_version: u32,
    }
    let version: Version = serde_json::from_slice(&bytes)?;
    if version.schema_version != 10 {
        return Err(CostProfileError::UnsupportedVersion(version.schema_version));
    }
    let declared: Envelope = serde_json::from_slice(&bytes)?;
    if declared.model_revision != MODEL_REVISION_V2 {
        return Err(invalid("unsupported structured revision"));
    }
    if declared.fingerprint != ProfileFingerprint::from(expected_fingerprint) {
        return Err(CostProfileError::FingerprintMismatch);
    }
    let source_path = if declared.source_path.is_absolute() {
        declared.source_path.clone()
    } else {
        path.parent()
            .unwrap_or(Path::new("."))
            .join(&declared.source_path)
    };
    let remaining = limits
        .max_file_bytes
        .get()
        .checked_sub(bytes.len())
        .ok_or(CostProfileError::Limit("structured envelope byte limit"))?;
    let source = read_bounded(&source_path, remaining)?;
    if source.len() as u64 != declared.source_bytes
        || <[u8; 32]>::from(Sha256::digest(&source)) != declared.source_sha256
    {
        return Err(invalid("structured source bytes/hash mismatch"));
    }
    let replayed = replay::replay_source(&source, limits)?;
    if replayed.header.fingerprint != declared.fingerprint
        || replayed.header.capture_identity != declared.capture_identity
        || replayed.header.protocol != declared.protocol
        || replayed.header.rule_signature != declared.rule_signature
        || replayed.header.cohort_manifest_sha256 != declared.cohort_manifest_sha256
        || replayed.phases != declared.phases
        || replayed.model.parameters_signature() != declared.parameters_sha256
    {
        return Err(invalid("structured envelope differs from original replay"));
    }
    let mapped = clock::map_clock(
        &replayed,
        declared.source_clock_max_error_ns,
        limits,
        load_clock,
    )?;
    let domain = *replayed.model.domain_signature();
    let model = ImportedStructuredModelV2 {
        model: Arc::new(replayed.model),
        fingerprint: expected_fingerprint.clone(),
        scope: replayed.header.scope.clone(),
        domain,
        provenance: StructuredImportProvenanceV10 {
            schema_version: 10,
            loaded_from: path.into(),
            source_path,
            file_sha256: Sha256::digest(&bytes).into(),
            source_sha256: declared.source_sha256,
            parameters_sha256: declared.parameters_sha256,
            capture_identity: declared.capture_identity,
            protocol: declared.protocol,
            rule_signature: declared.rule_signature,
            cohort_manifest_sha256: declared.cohort_manifest_sha256,
            offered_attempts: replayed.offered_attempts,
            reserved_members: replayed.reserved_members,
            total_shape_rows: replayed.total_shape_rows,
            file_bytes: bytes.len() as u64,
            source_bytes: source.len() as u64,
            conservative_clock_error_ns: mapped.error,
            oldest_imported_age_ns: mapped.oldest_age,
            newest_imported_age_ns: mapped.newest_age,
            loaded_unix_ns: mapped.wall,
            generated_unix_ns: replayed.closing.wall_unix_ns,
            clock: mapped.clock,
            producer: replayed.header.producer,
            phases: declared.phases,
        },
    };
    Ok(model)
}
