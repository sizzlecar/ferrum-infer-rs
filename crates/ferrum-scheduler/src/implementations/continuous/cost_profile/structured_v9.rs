//! Structured schema 9 imports replay-only numbers from a complete original
//! source. No wire DTO can mint a live settlement receipt or execution permit.
use super::super::cost_model::structured::{
    CalibratedStructuredModelV1, FittedStructuredModelV1, QualifiedStructuredModelV1,
    StructuredInputV1, StructuredMemberBindingV1, StructuredNumericObservationV1,
    StructuredPartitionV1, StructuredPopulationV1, StructuredPredictionV1, StructuredScopeV1,
    StructuredSettingsV1, StructuredUnknown, MODEL_REVISION, POPULATION_REVISION,
};
use super::*;
use std::io::Write;
mod clock;
mod observation;
mod replay;
mod wire;
use wire::*;
pub use wire::{StructuredPhaseProvenanceV9, StructuredProfilePhaseV9};
#[cfg(test)]
mod tests;
pub const COST_PROFILE_SCHEMA_VERSION_V9: u32 = 9;

#[derive(Debug, Clone, Serialize)]
pub struct StructuredImportProvenanceV9 {
    pub schema_version: u32,
    pub loaded_from: PathBuf,
    pub source_path: PathBuf,
    pub file_sha256: [u8; 32],
    pub source_sha256: [u8; 32],
    pub parameters_sha256: [u8; 32],
    pub capture_identity: [u8; 32],
    pub protocol: [u8; 32],
    pub rule_signature: [u8; 32],
    pub offered_attempts: u64,
    pub reserved_members: u64,
    pub file_bytes: u64,
    pub source_bytes: u64,
    pub conservative_clock_error_ns: u64,
    pub oldest_imported_age_ns: u64,
    pub newest_imported_age_ns: u64,
    pub loaded_unix_ns: u64,
    pub generated_unix_ns: u64,
    pub clock: ProfileObservationClock,
    pub producer: serde_json::Value,
    pub phases: [StructuredPhaseProvenanceV9; 3],
}
#[derive(Debug, Clone, Serialize)]
pub struct StructuredProfileExportReceiptV9 {
    pub path: PathBuf,
    pub source_path: PathBuf,
    pub file_sha256: [u8; 32],
    pub source_sha256: [u8; 32],
    pub parameters_sha256: [u8; 32],
    pub file_bytes: u64,
    pub source_bytes: u64,
    pub phases: [StructuredPhaseProvenanceV9; 3],
}
#[derive(Clone)]
pub struct ImportedStructuredModelV1 {
    model: Arc<QualifiedStructuredModelV1>,
    fingerprint: ExecutionFingerprint,
    scope: StructuredScopeV1,
    domain: [u8; 32],
    provenance: StructuredImportProvenanceV9,
}
impl std::fmt::Debug for ImportedStructuredModelV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ImportedStructuredModelV1")
            .field("provenance", &self.provenance)
            .finish_non_exhaustive()
    }
}
impl ImportedStructuredModelV1 {
    pub fn predict_input(
        &self,
        fingerprint: &ExecutionFingerprint,
        input: &StructuredInputV1,
        local_now_ns: u64,
    ) -> Result<StructuredPredictionV1, StructuredUnknown> {
        self.model
            .predict(fingerprint, input, self.model_now_ns(local_now_ns)?)
    }
    pub fn model_now_ns(&self, local_now_ns: u64) -> Result<u64, StructuredUnknown> {
        self.provenance
            .clock
            .model_now_ns(local_now_ns)
            .map_err(|_| StructuredUnknown::Clock)
    }
    pub fn scope(&self) -> StructuredScopeV1 {
        self.scope
    }
    pub fn domain_signature(&self) -> &[u8; 32] {
        &self.domain
    }
    pub fn parameters_signature(&self) -> &[u8; 32] {
        &self.provenance.parameters_sha256
    }
    pub fn fingerprint(&self) -> &ExecutionFingerprint {
        &self.fingerprint
    }
    pub fn provenance(&self) -> &StructuredImportProvenanceV9 {
        &self.provenance
    }
}
fn invalid(reason: &'static str) -> CostProfileError {
    CostProfileError::Metadata(reason)
}
fn numeric(_: StructuredUnknown) -> CostProfileError {
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
        schema_version: 9,
        model_revision: MODEL_REVISION.into(),
        population_revision: POPULATION_REVISION.into(),
        fingerprint: replayed.header.fingerprint.clone(),
        source_path,
        source_bytes,
        source_sha256,
        source_clock_max_error_ns: error,
        capture_identity: replayed.header.capture_identity,
        protocol: replayed.header.protocol,
        rule_signature: replayed.header.rule_signature,
        parameters_sha256: replayed.model.parameters_signature(),
        phases: replayed.phases.clone(),
    }
}
/// Both export and load replay every original population and its phase digest.
/// Atomic new-name publication refuses to overwrite an earlier artifact.
pub fn export_structured_profile_v9(
    source_path: &Path,
    expected_source_sha256: [u8; 32],
    destination: &Path,
    declared_source_clock_error_ns: u64,
    limits: &CostProfileLoadLimits,
) -> Result<StructuredProfileExportReceiptV9, CostProfileError> {
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
    Ok(StructuredProfileExportReceiptV9 {
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

pub fn load_structured_profile_v9(
    path: &Path,
    expected_fingerprint: &ExecutionFingerprint,
    limits: &CostProfileLoadLimits,
    load_clock: ProfileLoadClock,
) -> Result<ImportedStructuredModelV1, CostProfileError> {
    limits.validate()?;
    let bytes = read_bounded(path, limits.max_file_bytes.get())?;
    #[derive(Deserialize)]
    struct Version {
        schema_version: u32,
    }
    let version: Version = serde_json::from_slice(&bytes)?;
    if version.schema_version != 9 {
        return Err(CostProfileError::UnsupportedVersion(version.schema_version));
    }
    let declared: Envelope = serde_json::from_slice(&bytes)?;
    if declared.model_revision != MODEL_REVISION
        || declared.population_revision != POPULATION_REVISION
    {
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
    let model = ImportedStructuredModelV1 {
        model: Arc::new(replayed.model),
        fingerprint: expected_fingerprint.clone(),
        scope: StructuredScopeV1::OrdinaryDecodeSingleLength {
            rows: replayed.header.scope.rows,
        },
        domain: replayed.header.scope.domain,
        provenance: StructuredImportProvenanceV9 {
            schema_version: 9,
            loaded_from: path.into(),
            source_path,
            file_sha256: Sha256::digest(&bytes).into(),
            source_sha256: declared.source_sha256,
            parameters_sha256: declared.parameters_sha256,
            capture_identity: declared.capture_identity,
            protocol: declared.protocol,
            rule_signature: declared.rule_signature,
            offered_attempts: replayed.offered_attempts,
            reserved_members: replayed.reserved_members,
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
