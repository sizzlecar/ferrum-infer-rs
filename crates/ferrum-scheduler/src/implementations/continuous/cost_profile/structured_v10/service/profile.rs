use super::*;
mod live;
const ARTIFACT: &str = "ferrum.structured-service-window-catalog";
const MAX_METADATA: usize = 2 * 1024 * 1024;
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Child {
    declaration_index: usize,
    owner: StructuredOwnerKeyV2,
    domain_signature: [u8; 32],
    parameters_sha256: [u8; 32],
    phases: [StructuredPhaseProvenanceV10; 3],
}
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct EnvelopeV13 {
    artifact_type: String,
    schema_version: u32,
    model_revision: String,
    fingerprint: ProfileFingerprint,
    #[serde(skip_serializing_if = "Option::is_none")]
    source_path: Option<PathBuf>,
    source_bytes: u64,
    source_sha256: [u8; 32],
    capture_protocol: [u8; 32],
    // A live same-process model needs no wall-clock accuracy declaration.
    // Such a persisted artifact cannot be imported into another clock epoch.
    #[serde(skip_serializing_if = "Option::is_none")]
    source_clock_max_error_ns: Option<u64>,
    #[serde(deserialize_with = "bounded_rows")]
    children: Vec<Child>,
}
#[derive(Debug, Clone)]
pub struct ImportedStructuredCatalogV13 {
    workload_domain: Option<ferrum_interfaces::execution_cost::CostWorkloadDomainV1>,
    pub children: Vec<ImportedStructuredModelV2>,
    pub file_sha256: [u8; 32],
    pub source_sha256: [u8; 32],
    pub capture_protocol: [u8; 32],
    pub file_bytes: u64,
    pub source_bytes: u64,
    pub total_shape_rows: u64,
    pub offered_attempts: u64,
}
impl ImportedStructuredCatalogV13 {
    /// The source-frozen domain, for equality with the current executor before
    /// activation. Matching the four identity digests alone is insufficient.
    pub fn workload_domain(
        &self,
    ) -> Option<&ferrum_interfaces::execution_cost::CostWorkloadDomainV1> {
        self.workload_domain.as_ref()
    }
}
#[derive(Debug, Clone, Serialize)]
pub struct StructuredProfileExportReceiptV13 {
    pub schema_version: u32,
    pub path: PathBuf,
    pub source_path: PathBuf,
    pub file_sha256: [u8; 32],
    pub source_sha256: [u8; 32],
    pub capture_protocol: [u8; 32],
    pub file_bytes: u64,
    pub source_bytes: u64,
    pub children: Vec<StructuredProfileExportReceiptV10>,
}
fn clock_evidence(r: &StructuredServiceCollectorV6, i: usize) -> clock::Evidence {
    clock::Evidence {
        opening: r.header.opening.into(),
        closing: r.closing.expect("verified footer").into(),
        oldest_observed: r.child_ages[i].0,
        newest_observed: r.child_ages[i].1,
        max_age_ns: r.header.declaration.settings.max_sample_age_ns,
    }
}
pub fn export_structured_profile_v13(
    source_path: &Path,
    expected_source_sha256: [u8; 32],
    destination: &Path,
    declared_source_clock_error_ns: u64,
    limits: &CostProfileLoadLimits,
) -> Result<StructuredProfileExportReceiptV13, CostProfileError> {
    limits.validate()?;
    if declared_source_clock_error_ns > limits.max_clock_error_ns {
        return Err(CostProfileError::Clock(
            "source6 source clock error exceeds policy",
        ));
    }
    let source = read_bounded(source_path, limits.max_file_bytes.get())?;
    let digest: [u8; 32] = Sha256::digest(&source).into();
    if digest != expected_source_sha256 {
        return Err(invalid("source6 source digest mismatch"));
    }
    let r = replay::replay_source(&source, limits)?;
    let mut children = Vec::new();
    for (i, m) in r.models() {
        clock::validate_source(clock_evidence(&r, i), declared_source_clock_error_ns)?;
        children.push(Child {
            declaration_index: i,
            owner: m.owner().clone(),
            domain_signature: *m.domain_signature(),
            parameters_sha256: m.parameters_signature(),
            phases: r.phases[i]
                .clone()
                .try_into()
                .map_err(|_| invalid("source6 missing three freezes"))?,
        });
    }
    if children.is_empty() {
        return Err(invalid("source6 has no qualified child"));
    }
    let source_path = source_path.canonicalize()?;
    let envelope = EnvelopeV13 {
        artifact_type: ARTIFACT.into(),
        schema_version: 13,
        model_revision: MODEL_REVISION_V2.into(),
        fingerprint: r.header.fingerprint.clone(),
        source_path: Some(source_path.clone()),
        source_bytes: source.len() as u64,
        source_sha256: digest,
        capture_protocol: r.header.protocol,
        source_clock_max_error_ns: Some(declared_source_clock_error_ns),
        children,
    };
    let mut bytes = serde_json::to_vec_pretty(&envelope)?;
    bytes.push(b'\n');
    if bytes.len() > MAX_METADATA
        || bytes
            .len()
            .checked_add(source.len())
            .is_none_or(|n| n > limits.max_file_bytes.get())
    {
        return Err(CostProfileError::Limit(
            "profile13 and source6 byte capacity",
        ));
    }
    publish_new(destination, &bytes)?;
    let file_sha256 = Sha256::digest(&bytes).into();
    Ok(StructuredProfileExportReceiptV13 {
        schema_version: 13,
        path: destination.into(),
        source_path: source_path.clone(),
        file_sha256,
        source_sha256: digest,
        capture_protocol: r.header.protocol,
        file_bytes: bytes.len() as u64,
        source_bytes: source.len() as u64,
        children: r
            .models()
            .map(|(i, m)| StructuredProfileExportReceiptV10 {
                model_revision: MODEL_REVISION_V2,
                uncertainty: m.uncertainty(),
                path: destination.into(),
                source_path: source_path.clone(),
                file_sha256,
                source_sha256: digest,
                parameters_sha256: m.parameters_signature(),
                file_bytes: bytes.len() as u64,
                source_bytes: source.len() as u64,
                phases: r.phases[i]
                    .clone()
                    .try_into()
                    .expect("validated three freezes"),
            })
            .collect(),
    })
}
pub fn load_structured_profile_v13(
    path: &Path,
    expected_fingerprint: &ExecutionFingerprint,
    limits: &CostProfileLoadLimits,
    load_clock: ProfileLoadClock,
) -> Result<ImportedStructuredCatalogV13, CostProfileError> {
    limits.validate()?;
    let bytes = read_bounded(path, limits.max_file_bytes.get().min(MAX_METADATA))?;
    let mut declared: EnvelopeV13 = serde_json::from_slice(&bytes)?;
    if declared.artifact_type != ARTIFACT
        || declared.schema_version != 13
        || declared.model_revision != MODEL_REVISION_V2
        || declared.children.is_empty()
        || declared.children.len() > 128
    {
        return Err(invalid("unsupported profile13 declaration"));
    }
    if declared.fingerprint != ProfileFingerprint::from(expected_fingerprint) {
        return Err(CostProfileError::FingerprintMismatch);
    }
    let source_error = declared
        .source_clock_max_error_ns
        .ok_or(CostProfileError::Clock(
            "profile13 has no declared source wall-clock accuracy for cross-process import",
        ))?;
    let source_location = declared
        .source_path
        .as_ref()
        .ok_or_else(|| invalid("profile13 has no persisted source"))?;
    let source_path = if source_location.is_absolute() {
        source_location.clone()
    } else {
        path.parent()
            .unwrap_or(Path::new("."))
            .join(source_location)
    };
    let remaining = limits
        .max_file_bytes
        .get()
        .checked_sub(bytes.len())
        .ok_or(CostProfileError::Limit("profile13 metadata capacity"))?;
    let source = read_bounded(&source_path, remaining)?;
    if source.len() as u64 != declared.source_bytes
        || <[u8; 32]>::from(Sha256::digest(&source)) != declared.source_sha256
    {
        return Err(invalid("profile13 source bytes/hash differ"));
    }
    let r = replay::replay_source(&source, limits)?;
    if r.header.fingerprint != declared.fingerprint
        || r.header.protocol != declared.capture_protocol
        || r.qualified_children() != declared.children.len()
    {
        return Err(invalid("profile13 differs from complete source6"));
    }
    let mut metadata = Vec::new();
    for ((i, m), d) in r.models().zip(&declared.children) {
        if d.declaration_index != i
            || m.owner() != &d.owner
            || m.domain_signature() != &d.domain_signature
            || m.parameters_signature() != d.parameters_sha256
            || r.phases[i].as_slice() != d.phases.as_slice()
        {
            return Err(invalid("profile13 child differs from replay"));
        }
        metadata.push((
            i,
            clock::map_clock(clock_evidence(&r, i), source_error, limits, load_clock)?,
        ));
    }
    // Preserve the loader-resolved location in provenance, including profiles
    // whose declared source was relative to the metadata file.
    declared.source_path = Some(source_path);
    assemble_catalog(
        r,
        declared,
        Some(path),
        bytes.len() as u64,
        Sha256::digest(&bytes).into(),
        metadata,
        ferrum_types::SloCostProfileClockBasis::ImportedWallClock,
    )
}

fn assemble_catalog(
    r: StructuredServiceCollectorV6,
    declared: EnvelopeV13,
    path: Option<&Path>,
    file_bytes: u64,
    file_sha256: [u8; 32],
    metadata: Vec<(usize, clock::Mapped)>,
    clock_basis: ferrum_types::SloCostProfileClockBasis,
) -> Result<ImportedStructuredCatalogV13, CostProfileError> {
    if metadata.len() != declared.children.len() || metadata.len() != r.qualified_children() {
        return Err(invalid("profile13 mapped child count differs"));
    }
    let header = r.header.clone();
    let offered = r.offered();
    let total_rows = r.total_rows;
    let closing = r.closing.unwrap();
    let source_bytes = declared.source_bytes;
    let mut children = Vec::with_capacity(metadata.len());
    for (((i, model), (mi, mapped)), d) in r
        .into_models()?
        .into_iter()
        .zip(metadata)
        .zip(declared.children)
    {
        if i != mi {
            return Err(invalid("profile13 child order differs"));
        }
        let contract = header.child_contract(i)?;
        children.push(ImportedStructuredModelV2 {
            fingerprint: header.fingerprint.clone().into(),
            scope: model.scope().clone(),
            domain: *model.domain_signature(),
            model: Arc::new(model),
            provenance: StructuredImportProvenanceV10 {
                monotonic_domain: None,
                storage: if path.is_some() {
                    ferrum_types::SloCostProfileStorage::File
                } else {
                    ferrum_types::SloCostProfileStorage::Memory
                },
                clock_basis,
                schema_version: 13,
                loaded_from: path.map(Path::to_path_buf),
                source_path: declared.source_path.clone(),
                file_sha256,
                source_sha256: declared.source_sha256,
                parameters_sha256: d.parameters_sha256,
                capture_identity: header.capture_identity,
                protocol: header.protocol,
                rule_signature: contract.membership_rule,
                cohort_manifest_sha256: header.declaration_sha256,
                offered_attempts: offered,
                reserved_members: d.phases.iter().map(|p| p.members as u64).sum(),
                total_shape_rows: total_rows,
                file_bytes,
                source_bytes,
                conservative_clock_error_ns: mapped.error,
                oldest_imported_age_ns: mapped.oldest_age,
                newest_imported_age_ns: mapped.newest_age,
                loaded_unix_ns: mapped.wall,
                generated_unix_ns: closing.wall_unix_ns,
                clock: mapped.clock,
                producer: serde_json::value::to_raw_value(&header.producer)?,
                phases: d.phases,
            },
        });
    }
    Ok(ImportedStructuredCatalogV13 {
        workload_domain: header
            .declaration
            .nonnegative_envelope
            .map(|contract| contract.workload_domain),
        children,
        file_sha256,
        source_sha256: declared.source_sha256,
        capture_protocol: declared.capture_protocol,
        file_bytes,
        source_bytes,
        total_shape_rows: total_rows,
        offered_attempts: offered,
    })
}
