use super::*;
const ARTIFACT: &str = "ferrum.structured-shared-v2-catalog";
const MAX_METADATA: usize = 2 * 1024 * 1024;
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Child {
    owner: StructuredOwnerKeyV2,
    domain_signature: [u8; 32],
    source_clock_max_error_ns: u64,
    capture_identity: [u8; 32],
    protocol: [u8; 32],
    rule_signature: [u8; 32],
    cohort_manifest_sha256: [u8; 32],
    parameters_sha256: [u8; 32],
    phases: [StructuredPhaseProvenanceV10; 3],
}
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct EnvelopeV11 {
    artifact_type: String,
    schema_version: u32,
    model_revision: String,
    fingerprint: ProfileFingerprint,
    source_path: PathBuf,
    source_bytes: u64,
    source_sha256: [u8; 32],
    capture_protocol: [u8; 32],
    #[serde(deserialize_with = "bounded_entries")]
    children: Vec<Child>,
}
#[derive(Debug, Clone, Serialize)]
pub struct StructuredProfileExportReceiptV11 {
    pub schema_version: u32,
    pub path: PathBuf,
    pub source_path: PathBuf,
    pub file_sha256: [u8; 32],
    pub source_sha256: [u8; 32],
    pub capture_protocol: [u8; 32],
    pub file_bytes: u64,
    pub source_bytes: u64,
    /// Child diagnostics refer to the same common profile/source bytes.
    pub children: Vec<StructuredProfileExportReceiptV10>,
}
#[derive(Debug, Clone)]
pub struct ImportedStructuredCatalogV11 {
    pub children: Vec<ImportedStructuredModelV2>,
    pub file_sha256: [u8; 32],
    pub source_sha256: [u8; 32],
    pub capture_protocol: [u8; 32],
    pub file_bytes: u64,
    pub source_bytes: u64,
    pub total_shape_rows: u64,
    pub offered_attempts: u64,
}
/// Export replays one physical stream and all independent frozen child models.
/// Publication uses the existing synced, new-name-only atomic file path.
pub fn export_structured_profile_v11(
    source_path: &Path,
    expected_source_sha256: [u8; 32],
    destination: &Path,
    declared_source_clock_errors_ns: &[u64],
    limits: &CostProfileLoadLimits,
) -> Result<StructuredProfileExportReceiptV11, CostProfileError> {
    limits.validate()?;
    let source = read_bounded(source_path, limits.max_file_bytes.get())?;
    let digest: [u8; 32] = Sha256::digest(&source).into();
    if digest != expected_source_sha256 {
        return Err(invalid("shared source digest mismatch"));
    }
    let replayed = replay::replay_source(&source, limits)?;
    if replayed.children.len() != declared_source_clock_errors_ns.len() {
        return Err(invalid("shared clock declarations differ from children"));
    }
    let mut children = Vec::with_capacity(replayed.children.len());
    for (r, &error) in replayed
        .children
        .iter()
        .zip(declared_source_clock_errors_ns)
    {
        if error > limits.max_clock_error_ns {
            return Err(CostProfileError::Clock(
                "source clock error exceeds import policy",
            ));
        }
        clock::validate_source(clock_evidence(r), error)?;
        children.push(Child {
            owner: r.model.owner().clone(),
            domain_signature: *r.model.domain_signature(),
            source_clock_max_error_ns: error,
            capture_identity: r.header.capture_identity,
            protocol: r.header.protocol,
            rule_signature: r.header.rule_signature,
            cohort_manifest_sha256: r.header.common.cohort_manifest_sha256,
            parameters_sha256: r.model.parameters_signature(),
            phases: r.phases.clone(),
        });
    }
    let source_path = source_path.canonicalize()?;
    let value = EnvelopeV11 {
        artifact_type: ARTIFACT.into(),
        schema_version: 11,
        model_revision: MODEL_REVISION_V2.into(),
        fingerprint: replayed.children[0].header.common.fingerprint.clone(),
        source_path: source_path.clone(),
        source_bytes: source.len() as u64,
        source_sha256: digest,
        capture_protocol: replayed.capture_protocol,
        children,
    };
    let mut bytes = serde_json::to_vec_pretty(&value)?;
    bytes.push(b'\n');
    if bytes.len() > MAX_METADATA
        || bytes
            .len()
            .checked_add(source.len())
            .is_none_or(|n| n > limits.max_file_bytes.get())
    {
        return Err(CostProfileError::Limit(
            "shared envelope and source exceed byte limit",
        ));
    }
    publish_new(destination, &bytes)?;
    let file_sha256 = Sha256::digest(&bytes).into();
    Ok(StructuredProfileExportReceiptV11 {
        schema_version: 11,
        path: destination.into(),
        source_path: source_path.clone(),
        file_sha256,
        source_sha256: digest,
        capture_protocol: replayed.capture_protocol,
        file_bytes: bytes.len() as u64,
        source_bytes: source.len() as u64,
        children: replayed
            .children
            .iter()
            .map(|r| StructuredProfileExportReceiptV10 {
                model_revision: MODEL_REVISION_V2,
                uncertainty: r.model.uncertainty(),
                path: destination.into(),
                source_path: source_path.clone(),
                file_sha256,
                source_sha256: digest,
                parameters_sha256: r.model.parameters_signature(),
                file_bytes: bytes.len() as u64,
                source_bytes: source.len() as u64,
                phases: r.phases.clone(),
            })
            .collect(),
    })
}
/// Return all children atomically. No usable child escapes if any source proof,
/// independent qualification, binding or original clock fails.
pub fn load_structured_profile_v11(
    path: &Path,
    expected_fingerprint: &ExecutionFingerprint,
    limits: &CostProfileLoadLimits,
    load_clock: ProfileLoadClock,
) -> Result<ImportedStructuredCatalogV11, CostProfileError> {
    limits.validate()?;
    let bytes = read_bounded(path, limits.max_file_bytes.get().min(MAX_METADATA))?;
    let declared: EnvelopeV11 = serde_json::from_slice(&bytes)?;
    if declared.artifact_type != ARTIFACT
        || declared.schema_version != 11
        || declared.model_revision != MODEL_REVISION_V2
    {
        return Err(invalid("unsupported shared profile identity"));
    }
    if declared.fingerprint != ProfileFingerprint::from(expected_fingerprint) {
        return Err(CostProfileError::FingerprintMismatch);
    }
    if declared.children.is_empty() || declared.children.len() > 128 {
        return Err(invalid("shared profile child count"));
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
        .ok_or(CostProfileError::Limit("shared metadata byte budget"))?;
    let source = read_bounded(&source_path, remaining)?;
    if source.len() as u64 != declared.source_bytes
        || <[u8; 32]>::from(Sha256::digest(&source)) != declared.source_sha256
    {
        return Err(invalid("shared source bytes/hash mismatch"));
    }
    let replayed = replay::replay_source(&source, limits)?;
    if replayed.capture_protocol != declared.capture_protocol
        || replayed.children.len() != declared.children.len()
    {
        return Err(invalid("shared declaration differs from source"));
    }
    let file_sha256: [u8; 32] = Sha256::digest(&bytes).into();
    let total_shape_rows = replayed.children[0].total_shape_rows;
    let offered_attempts = replayed.children[0].offered_attempts;
    let mut children = Vec::with_capacity(replayed.children.len());
    for (r, d) in replayed.children.into_iter().zip(declared.children) {
        if r.header.common.fingerprint != declared.fingerprint
            || r.header.capture_identity != d.capture_identity
            || r.header.protocol != d.protocol
            || r.header.rule_signature != d.rule_signature
            || r.header.common.cohort_manifest_sha256 != d.cohort_manifest_sha256
            || r.phases != d.phases
            || r.model.parameters_signature() != d.parameters_sha256
            || r.model.owner() != &d.owner
            || r.model.domain_signature() != &d.domain_signature
        {
            return Err(invalid("shared child differs from original replay"));
        }
        let mapped = clock::map_clock(
            clock_evidence(&r),
            d.source_clock_max_error_ns,
            limits,
            load_clock,
        )?;
        let domain = *r.model.domain_signature();
        children.push(ImportedStructuredModelV2 {
            model: Arc::new(r.model),
            fingerprint: expected_fingerprint.clone(),
            scope: r.header.scope.clone(),
            domain,
            provenance: StructuredImportProvenanceV10 {
                schema_version: 11,
                loaded_from: path.into(),
                source_path: source_path.clone(),
                file_sha256,
                source_sha256: declared.source_sha256,
                parameters_sha256: d.parameters_sha256,
                capture_identity: d.capture_identity,
                protocol: d.protocol,
                rule_signature: d.rule_signature,
                cohort_manifest_sha256: d.cohort_manifest_sha256,
                offered_attempts: r.offered_attempts,
                reserved_members: r.reserved_members,
                total_shape_rows: r.total_shape_rows,
                file_bytes: bytes.len() as u64,
                source_bytes: source.len() as u64,
                conservative_clock_error_ns: mapped.error,
                oldest_imported_age_ns: mapped.oldest_age,
                newest_imported_age_ns: mapped.newest_age,
                loaded_unix_ns: mapped.wall,
                generated_unix_ns: r.closing.wall_unix_ns,
                clock: mapped.clock,
                producer: r.header.common.producer.clone(),
                phases: d.phases,
            },
        });
    }
    Ok(ImportedStructuredCatalogV11 {
        children,
        file_sha256,
        source_sha256: declared.source_sha256,
        capture_protocol: declared.capture_protocol,
        file_bytes: bytes.len() as u64,
        source_bytes: source.len() as u64,
        total_shape_rows,
        offered_attempts,
    })
}

pub(super) fn clock_evidence(r: &replay::ReplayedChildV4) -> clock::Evidence {
    clock::Evidence {
        opening: r.header.common.opening,
        closing: r.closing,
        oldest_observed: r.oldest_observed,
        newest_observed: r.newest_observed,
        max_age_ns: r.header.settings.max_age_ns,
    }
}
