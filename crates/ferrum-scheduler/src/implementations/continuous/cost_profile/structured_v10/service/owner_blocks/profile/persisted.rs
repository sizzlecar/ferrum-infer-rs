use super::*;
/// Bind a final journal (including later pending/failed attempts) and replay the
/// chosen immutable prefix. Neither a later footer nor import renews sample age.
pub fn export_structured_profile_v14(
    source_path: &Path,
    expected_journal_sha256: [u8; 32],
    checkpoint_source_bytes: u64,
    destination: &Path,
    declared_source_clock_error_ns: u64,
    limits: &CostProfileLoadLimits,
) -> Result<StructuredProfileExportReceiptV14, CostProfileError> {
    export_with_clock(
        source_path,
        expected_journal_sha256,
        checkpoint_source_bytes,
        destination,
        Some(declared_source_clock_error_ns),
        limits,
        None,
        None,
        None,
    )
}
/// Replay an originally clock-bound source; wall timestamps remain informational.
#[allow(clippy::too_many_arguments)]
pub fn export_structured_profile_v14_same_boot(
    source_path: &Path,
    expected_journal_sha256: [u8; 32],
    checkpoint_source_bytes: u64,
    destination: &Path,
    domain: &ferrum_interfaces::execution_cost::CostMonotonicDomainV1,
    now: StructuredServiceClockV7,
    limits: &CostProfileLoadLimits,
) -> Result<StructuredProfileExportReceiptV14, CostProfileError> {
    export_with_clock(
        source_path,
        expected_journal_sha256,
        checkpoint_source_bytes,
        destination,
        None,
        limits,
        Some((domain, now)),
        None,
        None,
    )
}
/// Preserve the complete original source and metadata, requiring current
/// freshness only for the explicitly selected qualified domains.
#[allow(clippy::too_many_arguments)]
pub fn export_structured_profile_v14_same_boot_selected(
    source_path: &Path,
    expected_journal_sha256: [u8; 32],
    checkpoint_source_bytes: u64,
    destination: &Path,
    domain: &ferrum_interfaces::execution_cost::CostMonotonicDomainV1,
    now: StructuredServiceClockV7,
    limits: &CostProfileLoadLimits,
    selected: &[[u8; 32]],
) -> Result<StructuredProfileExportReceiptV14, CostProfileError> {
    export_with_clock(
        source_path,
        expected_journal_sha256,
        checkpoint_source_bytes,
        destination,
        None,
        limits,
        Some((domain, now)),
        Some(selected),
        None,
    )
}

/// Replay caller-provided ORIGINAL journal bytes, retaining the managed source
/// path as provenance. Storage decoding grants no profile/model authority:
/// full raw digest, exact checkpoint, original population and clocks are checked.
#[allow(clippy::too_many_arguments)]
pub fn export_structured_profile_v14_same_boot_selected_from_original_bytes(
    source_path: &Path,
    source: &[u8],
    expected_journal_sha256: [u8; 32],
    checkpoint_source_bytes: u64,
    destination: &Path,
    domain: &ferrum_interfaces::execution_cost::CostMonotonicDomainV1,
    now: StructuredServiceClockV7,
    limits: &CostProfileLoadLimits,
    selected: &[[u8; 32]],
) -> Result<StructuredProfileExportReceiptV14, CostProfileError> {
    export_with_clock(
        source_path,
        expected_journal_sha256,
        checkpoint_source_bytes,
        destination,
        None,
        limits,
        Some((domain, now)),
        Some(selected),
        Some(source),
    )
}
#[allow(clippy::too_many_arguments)]
fn export_with_clock(
    source_path: &Path,
    expected_journal_sha256: [u8; 32],
    checkpoint_source_bytes: u64,
    destination: &Path,
    declared_source_clock_error_ns: Option<u64>,
    limits: &CostProfileLoadLimits,
    same_boot: Option<(
        &ferrum_interfaces::execution_cost::CostMonotonicDomainV1,
        StructuredServiceClockV7,
    )>,
    selected: Option<&[[u8; 32]]>,
    original_source: Option<&[u8]>,
) -> Result<StructuredProfileExportReceiptV14, CostProfileError> {
    limits.validate()?;
    if declared_source_clock_error_ns.is_some_and(|error| error > limits.max_clock_error_ns) {
        return Err(CostProfileError::Clock(
            "source7 source clock error exceeds policy",
        ));
    }
    let owned_source;
    let source = if let Some(source) = original_source {
        if source.len() > limits.max_file_bytes.get() {
            return Err(CostProfileError::Limit(
                "source7 original journal byte capacity",
            ));
        }
        source
    } else {
        owned_source = read_bounded(source_path, limits.max_file_bytes.get())?;
        owned_source.as_slice()
    };
    if <[u8; 32]>::from(Sha256::digest(source)) != expected_journal_sha256 {
        return Err(invalid("source7 journal digest differs"));
    }
    let prefix = source
        .get(
            ..usize::try_from(checkpoint_source_bytes)
                .map_err(|_| invalid("source7 checkpoint byte overflow"))?,
        )
        .ok_or_else(|| invalid("source7 checkpoint exceeds journal"))?;
    let checkpoint = replay_structured_source_v7(prefix, limits)?;
    if checkpoint.children.is_empty() {
        return Err(invalid("source7 checkpoint has no qualified child"));
    }
    validate_domain_selection(&checkpoint.children, selected)?;
    for c in &checkpoint.children {
        if selected
            .is_some_and(|domains| domains.binary_search(&c.metadata.domain_signature).is_err())
        {
            continue;
        }
        if let Some((domain, now)) = same_boot {
            let mapped = clock::same_boot(
                checkpoint.evidence(c),
                checkpoint.header.monotonic_domain.as_ref(),
                domain,
                now.monotonic_ns,
                now.wall_unix_ns,
                limits,
            )?;
            c.model
                .validate_runtime_at(mapped.model_now)
                .map_err(|_| CostProfileError::Clock("original source7 owner epoch expired"))?;
        } else {
            clock::validate_source(
                checkpoint.evidence(c),
                declared_source_clock_error_ns.ok_or(CostProfileError::Clock(
                    "missing source wall-clock accuracy",
                ))?,
            )?;
        }
    }
    let source_path = source_path.canonicalize()?;
    let mut declared = checkpoint.envelope();
    declared.source_path = Some(source_path.clone());
    declared.journal_bytes = Some(source.len() as u64);
    declared.journal_sha256 = Some(expected_journal_sha256);
    declared.source_clock_max_error_ns = declared_source_clock_error_ns;
    let bytes = metadata_bytes(&declared, limits)?;
    publish_new(destination, &bytes)?;
    let file_sha256 = Sha256::digest(&bytes).into();
    Ok(StructuredProfileExportReceiptV13 {
        schema_version: 14,
        path: destination.into(),
        source_path: source_path.clone(),
        file_sha256,
        source_sha256: checkpoint.source_sha256,
        capture_protocol: checkpoint.header.protocol,
        file_bytes: bytes.len() as u64,
        source_bytes: checkpoint.source_bytes,
        children: checkpoint
            .children
            .iter()
            .map(|c| StructuredProfileExportReceiptV10 {
                model_revision: MODEL_REVISION_V2,
                uncertainty: c.model.uncertainty(),
                path: destination.into(),
                source_path: source_path.clone(),
                file_sha256,
                source_sha256: checkpoint.source_sha256,
                parameters_sha256: c.metadata.parameters_sha256,
                file_bytes: bytes.len() as u64,
                source_bytes: checkpoint.source_bytes,
                phases: c.metadata.phases.clone(),
            })
            .collect(),
    })
}
pub fn load_structured_profile_v14(
    path: &Path,
    expected_fingerprint: &ExecutionFingerprint,
    limits: &CostProfileLoadLimits,
    load_clock: ProfileLoadClock,
) -> Result<ImportedStructuredCatalogV14, CostProfileError> {
    load_with_clock(
        path,
        expected_fingerprint,
        limits,
        Some(load_clock),
        None,
        None,
        None,
    )
}
pub fn load_structured_profile_v14_same_boot(
    path: &Path,
    expected_fingerprint: &ExecutionFingerprint,
    limits: &CostProfileLoadLimits,
    domain: &ferrum_interfaces::execution_cost::CostMonotonicDomainV1,
    now: StructuredServiceClockV7,
) -> Result<ImportedStructuredCatalogV14, CostProfileError> {
    load_with_clock(
        path,
        expected_fingerprint,
        limits,
        None,
        Some((domain, now)),
        None,
        None,
    )
}
/// Replay and bind every original child before returning only the chosen,
/// currently fresh domains. Original source work/row totals are unchanged.
pub fn load_structured_profile_v14_same_boot_selected(
    path: &Path,
    expected_fingerprint: &ExecutionFingerprint,
    limits: &CostProfileLoadLimits,
    domain: &ferrum_interfaces::execution_cost::CostMonotonicDomainV1,
    now: StructuredServiceClockV7,
    selected: &[[u8; 32]],
) -> Result<ImportedStructuredCatalogV14, CostProfileError> {
    load_with_clock(
        path,
        expected_fingerprint,
        limits,
        None,
        Some((domain, now)),
        Some(selected),
        None,
    )
}

/// Independently replay decoded original bytes using the same strict kernel as
/// path loading. The expected managed path must still match profile provenance.
/// Raw bytes, including a later journal tail, remain subject to import limits.
#[allow(clippy::too_many_arguments)]
pub fn load_structured_profile_v14_same_boot_selected_from_original_bytes(
    path: &Path,
    source: &[u8],
    expected_source_path: &Path,
    expected_fingerprint: &ExecutionFingerprint,
    limits: &CostProfileLoadLimits,
    domain: &ferrum_interfaces::execution_cost::CostMonotonicDomainV1,
    now: StructuredServiceClockV7,
    selected: &[[u8; 32]],
) -> Result<ImportedStructuredCatalogV14, CostProfileError> {
    load_with_clock(
        path,
        expected_fingerprint,
        limits,
        None,
        Some((domain, now)),
        Some(selected),
        Some((source, expected_source_path)),
    )
}
#[allow(clippy::too_many_arguments)]
fn load_with_clock(
    path: &Path,
    expected_fingerprint: &ExecutionFingerprint,
    limits: &CostProfileLoadLimits,
    load_clock: Option<ProfileLoadClock>,
    same_boot: Option<(
        &ferrum_interfaces::execution_cost::CostMonotonicDomainV1,
        StructuredServiceClockV7,
    )>,
    selected: Option<&[[u8; 32]]>,
    original_source: Option<(&[u8], &Path)>,
) -> Result<ImportedStructuredCatalogV14, CostProfileError> {
    limits.validate()?;
    let bytes = read_bounded(path, limits.max_file_bytes.get().min(MAX_METADATA))?;
    let mut declared: OwnerCatalogEnvelope = serde_json::from_slice(&bytes)?;
    if declared.artifact_type != ARTIFACT
        || declared.schema_version != 14
        || declared.model_revision != MODEL_REVISION_V2
        || declared.children.is_empty()
        || declared.children.len() > 128
    {
        return Err(invalid("unsupported profile14 declaration"));
    }
    if declared.fingerprint != ProfileFingerprint::from(expected_fingerprint) {
        return Err(CostProfileError::FingerprintMismatch);
    }
    let error = if same_boot.is_some() {
        None
    } else {
        Some(
            declared
                .source_clock_max_error_ns
                .ok_or(CostProfileError::Clock(
                    "profile has no declared source wall-clock accuracy",
                ))?,
        )
    };
    let location = declared
        .source_path
        .as_ref()
        .ok_or_else(|| invalid("profile14 has no persisted journal"))?;
    let source_path = if location.is_absolute() {
        location.clone()
    } else {
        path.parent().unwrap_or(Path::new(".")).join(location)
    };
    let remaining = limits
        .max_file_bytes
        .get()
        .checked_sub(bytes.len())
        .ok_or(CostProfileError::Limit("profile14 metadata capacity"))?;
    let owned_source;
    let source = if let Some((source, expected_source_path)) = original_source {
        // An identical byte stream at another path is not the declared source.
        // The exporter writes the canonical managed path into the profile.
        if source_path != expected_source_path.canonicalize()? {
            return Err(invalid("profile14 original journal path differs"));
        }
        if source.len() > remaining {
            return Err(CostProfileError::Limit(
                "profile14 original journal byte capacity",
            ));
        }
        source
    } else {
        owned_source = read_bounded(&source_path, remaining)?;
        owned_source.as_slice()
    };
    if declared.journal_bytes != Some(source.len() as u64)
        || declared.journal_sha256 != Some(Sha256::digest(source).into())
    {
        return Err(invalid("profile14 journal bytes/hash differ"));
    }
    let prefix = source
        .get(
            ..usize::try_from(declared.source_bytes)
                .map_err(|_| invalid("profile14 prefix byte overflow"))?,
        )
        .ok_or_else(|| invalid("profile14 prefix exceeds journal"))?;
    if <[u8; 32]>::from(Sha256::digest(prefix)) != declared.source_sha256 {
        return Err(invalid("profile14 checkpoint prefix hash differs"));
    }
    let mut checkpoint = replay_structured_source_v7(prefix, limits)?;
    if checkpoint.header.fingerprint != declared.fingerprint
        || checkpoint.header.protocol != declared.capture_protocol
        || checkpoint.header.monotonic_domain != declared.monotonic_domain
        || checkpoint.children.len() != declared.children.len()
    {
        return Err(invalid("profile14 differs from replayed checkpoint"));
    }
    validate_domain_selection(&checkpoint.children, selected)?;
    // Validate ALL metadata against independently replayed ALL original work
    // before selecting any child. Dropping an expired domain cannot hide
    // corruption or reduce the original offered/row budget.
    checkpoint.validate_original_metadata(&declared, collector::PopulationSource::OwnerBlocksV7)?;
    if let Some(domains) = selected {
        checkpoint.children.retain(|child| {
            domains
                .binary_search(&child.metadata.domain_signature)
                .is_ok()
        });
        declared
            .children
            .retain(|child| domains.binary_search(&child.domain_signature).is_ok());
    }
    let mapped = checkpoint
        .children
        .iter()
        .map(|c| {
            if let Some((domain, now)) = same_boot {
                let mapped = clock::same_boot(
                    checkpoint.evidence(c),
                    checkpoint.header.monotonic_domain.as_ref(),
                    domain,
                    now.monotonic_ns,
                    now.wall_unix_ns,
                    limits,
                )?;
                c.model
                    .validate_runtime_at(mapped.model_now)
                    .map_err(|_| CostProfileError::Clock("original source7 owner epoch expired"))?;
                Ok(mapped)
            } else {
                clock::map_clock(
                    checkpoint.evidence(c),
                    error.ok_or(CostProfileError::Clock(
                        "missing source wall-clock accuracy",
                    ))?,
                    limits,
                    load_clock.ok_or(CostProfileError::Clock("missing load wall clock"))?,
                )
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    declared.source_path = Some(source_path);
    let mut imported: ImportedStructuredCatalogV14 = checkpoint.assemble(
        declared,
        Some(path),
        &bytes,
        mapped,
        if same_boot.is_some() {
            ferrum_types::SloCostProfileClockBasis::SameBootMonotonic
        } else {
            ferrum_types::SloCostProfileClockBasis::ImportedWallClock
        },
    )?;
    if let Some(domains) = selected {
        imported
            .children
            .retain(|child| domains.binary_search(child.domain_signature()).is_ok());
    }
    Ok(imported)
}
