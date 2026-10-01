use super::*;
fn size(n: u64) -> Result<usize, FerrumError> {
    usize::try_from(n).map_err(|_| FerrumError::config("structured V2 receipt size overflow"))
}
fn hex(hash: &[u8; 32]) -> String {
    hash.iter().map(|b| format!("{b:02x}")).collect()
}

fn catalog_inventory_sha256(children: &[ferrum_types::SloStructuredChildReceiptV2]) -> [u8; 32] {
    let mut inventory = Sha256::new();
    inventory.update(b"ferrum.structured-v2-catalog-content-inventory.v1\0");
    inventory.update((children.len() as u64).to_le_bytes());
    for child in children {
        inventory.update(child.domain_signature);
        inventory.update(child.owner_sha256);
        inventory.update(child.scope_sha256);
        inventory.update(child.profile_sha256);
        inventory.update(child.source_sha256);
        inventory.update(child.parameters_sha256);
        inventory.update(child.profile_bytes.to_le_bytes());
        inventory.update(child.source_bytes.to_le_bytes());
    }
    inventory.finalize().into()
}

/// Select the installed subset of an already verified complete source8 import.
/// Source bytes, all offered work, original clocks and child provenance stay
/// intact. Only the selected child inventory is narrowed, before publication.
pub(super) fn retain_startup_catalog_children(
    receipt: &mut SloCostProfileReceipt,
    original: &[ImportedStructuredModelV2],
    keep: &[bool],
) -> Result<(), FerrumError> {
    let declared = receipt
        .structured_whole_wave_v2
        .as_mut()
        .ok_or_else(|| FerrumError::config("startup subset lacks structured receipt"))?;
    if receipt.schema_version != 15
        || declared.artifact_kind
            != ferrum_types::SloStructuredArtifactKindV2::PreparedOwnerBlockCatalogV15
        || original.is_empty()
        || original.len() != keep.len()
        || declared.child_count != original.len()
        || declared.children.len() != original.len()
        || declared
            .children
            .windows(2)
            .any(|pair| pair[0].domain_signature >= pair[1].domain_signature)
        || declared.source_inventory_sha256 != Some(catalog_inventory_sha256(&declared.children))
    {
        return Err(FerrumError::config(
            "startup subset original inventory differs",
        ));
    }
    // Validate even the children that structural selection will omit. A bad
    // original receipt must never become acceptable by filtering that record.
    for record in &declared.children {
        let model = original
            .iter()
            .find(|child| child.domain_signature() == &record.domain_signature)
            .ok_or_else(|| FerrumError::config("startup subset original owner missing"))?;
        if record != &child(model)? {
            return Err(FerrumError::config(
                "startup subset original source binding differs",
            ));
        }
    }
    if !keep.iter().any(|keep| *keep) {
        return Err(FerrumError::config(
            "startup source has no effective catalog extension; original structural coverage retained",
        ));
    }
    declared.children.retain(|record| {
        original
            .iter()
            .zip(keep)
            .any(|(model, keep)| *keep && model.domain_signature() == &record.domain_signature)
    });
    declared.child_count = declared.children.len();
    declared.source_inventory_sha256 = Some(catalog_inventory_sha256(&declared.children));
    receipt.bucket_count = declared.child_count;
    // Physical source totals above intentionally remain totals of the complete
    // original capture. Reserved members describe the selected model inventory.
    receipt.recorded_samples = size(declared.children.iter().try_fold(0u64, |sum, child| {
        sum.checked_add(child.reserved_members)
            .ok_or_else(|| FerrumError::config("startup subset member count overflow"))
    })?)?;
    Ok(())
}
pub(super) fn child(
    imported: &ImportedStructuredModelV2,
) -> Result<ferrum_types::SloStructuredChildReceiptV2, FerrumError> {
    let p = imported.provenance();
    let phases = p
        .phases
        .each_ref()
        .map(|v| ferrum_types::SloStructuredPhaseReceiptV1 {
            phase: match v.phase {
                StructuredProfilePhaseV10::Fit => ferrum_types::SloStructuredProfilePhaseV1::Fit,
                StructuredProfilePhaseV10::Residual => {
                    ferrum_types::SloStructuredProfilePhaseV1::Residual
                }
                StructuredProfilePhaseV10::Qualification => {
                    ferrum_types::SloStructuredProfilePhaseV1::Qualification
                }
            },
            members: v.members,
            member_cutoff: v.member_cutoff,
            accepted_fifo_cutoff: v.accepted_fifo_cutoff,
            frozen_at_ns: v.frozen_at_ns,
            source_prefix_bytes: v.source_prefix_bytes,
            source_prefix_sha256: v.source_prefix_sha256,
            parameters_sha256: v.parameters_sha256,
        });
    Ok(ferrum_types::SloStructuredChildReceiptV2 {
        storage: p.storage,
        clock_basis: p.clock_basis,
        profile_path: p.loaded_from.clone(),
        profile_sha256: p.file_sha256,
        profile_bytes: p.file_bytes,
        domain_signature: *imported.domain_signature(),
        owner_rows: imported.owner().rows,
        owner_sha256: Sha256::digest(serde_json::to_vec(imported.owner()).map_err(profile_error)?)
            .into(),
        scope_sha256: Sha256::digest(serde_json::to_vec(imported.scope()).map_err(profile_error)?)
            .into(),
        capture_identity_sha256: p.capture_identity,
        protocol_sha256: p.protocol,
        rule_signature: p.rule_signature,
        cohort_manifest_sha256: p.cohort_manifest_sha256,
        parameters_sha256: p.parameters_sha256,
        source_path: p.source_path.clone(),
        source_sha256: p.source_sha256,
        source_bytes: p.source_bytes,
        offered_attempts: p.offered_attempts,
        reserved_members: p.reserved_members,
        total_shape_rows: p.total_shape_rows,
        source_monotonic_anchor_ns: p.clock.source_monotonic_anchor_ns,
        model_anchor_ns: p.clock.model_anchor_ns,
        conservative_clock_error_ns: p.conservative_clock_error_ns,
        oldest_imported_age_ns: p.oldest_imported_age_ns,
        newest_imported_age_ns: p.newest_imported_age_ns,
        phases,
    })
}
pub(super) fn single(
    imported: &ImportedStructuredModelV2,
    declared: u64,
) -> Result<SloCostProfileReceipt, FerrumError> {
    let p = imported.provenance();
    let producer: serde_json::Value =
        serde_json::from_str(p.producer.get()).map_err(profile_error)?;
    Ok(SloCostProfileReceipt {
        storage: ferrum_types::SloCostProfileStorage::File,
        clock_basis: ferrum_types::SloCostProfileClockBasis::ImportedWallClock,
        selected_whole_wave: None,
        structured_whole_wave: None,
        structured_whole_wave_v2: Some(ferrum_types::SloStructuredWholeWaveReceiptV2 {
            model_revision: MODEL_REVISION_V2.into(),
            artifact_kind: ferrum_types::SloStructuredArtifactKindV2::SingleChild,
            child_count: 1,
            total_imported_bytes: p
                .file_bytes
                .checked_add(p.source_bytes)
                .ok_or_else(|| FerrumError::config("structured V2 byte count overflow"))?,
            total_shape_rows: p.total_shape_rows,
            source_inventory_sha256: None,
            children: vec![child(imported)?],
        }),
        schema_version: 10,
        path: p.loaded_from.clone(),
        file_sha256: format!("sha256:{}", hex(&p.file_sha256)),
        file_bytes: size(p.file_bytes)?,
        generated_unix_ns: p.generated_unix_ns,
        loaded_unix_ns: p.loaded_unix_ns,
        conservative_clock_error_ns: p.conservative_clock_error_ns,
        declared_local_clock_max_error_ns: Some(declared),
        oldest_imported_age_ns: Some(p.oldest_imported_age_ns),
        newest_imported_age_ns: Some(p.newest_imported_age_ns),
        offered_samples: size(p.offered_attempts)?,
        recorded_samples: size(p.reserved_members)?,
        stale_samples: 0,
        skipped_samples: Default::default(),
        model_version: 1,
        bucket_count: 1,
        source_generator: producer["executable_path"]
            .as_str()
            .ok_or_else(|| FerrumError::config("source3 validated producer missing"))?
            .into(),
        source_generator_revision: producer["source_revision"]
            .as_str()
            .map(str::to_owned)
            .unwrap_or_else(|| {
                format!(
                    "package:{}",
                    producer["package_version"]
                        .as_str()
                        .unwrap_or("unspecified")
                )
            }),
        source_measurement_protocol: format!("source3:sha256:{}", hex(&p.protocol)),
        source_observation_artifact_sha256: p.source_sha256,
    })
}

/// Outer source hash identifies the sorted child-content inventory. It is not
/// a fabricated raw observation source; every original source is in children.
pub(super) fn catalog(
    path: Option<&Path>,
    digest: [u8; 32],
    metadata_bytes: usize,
    snapshot: &StructuredSnapshot,
    declared: u64,
) -> Result<SloCostProfileReceipt, FerrumError> {
    if snapshot.children.is_empty() {
        return Err(FerrumError::config("empty structured catalog receipt"));
    }
    let mut children = Vec::with_capacity(snapshot.len());
    let mut total_bytes = metadata_bytes as u64;
    let mut total_rows = 0u64;
    let mut offered = 0u64;
    let mut members = 0u64;
    let mut generated = 0u64;
    let mut loaded = None;
    let mut error = 0u64;
    let mut oldest = 0u64;
    let mut newest = u64::MAX;
    let add = |a: u64, b: u64| {
        a.checked_add(b)
            .ok_or_else(|| FerrumError::config("catalog receipt total overflow"))
    };
    // BTreeMap canonical domain order; paths locate artifacts but are not an
    // alternative model identity. Profile bytes already bind original source path.
    for model in snapshot.children.values() {
        let p = model.provenance();
        let r = child(model)?;
        total_bytes = add(add(total_bytes, p.file_bytes)?, p.source_bytes)?;
        total_rows = add(total_rows, p.total_shape_rows)?;
        offered = add(offered, p.offered_attempts)?;
        members = add(members, p.reserved_members)?;
        generated = generated.max(p.generated_unix_ns);
        error = error.max(p.conservative_clock_error_ns);
        oldest = oldest.max(p.oldest_imported_age_ns);
        newest = newest.min(p.newest_imported_age_ns);
        if loaded.is_some_and(|at| at != p.loaded_unix_ns) {
            return Err(FerrumError::config(
                "catalog children use different startup wall clocks",
            ));
        }
        loaded = Some(p.loaded_unix_ns);
        children.push(r);
    }
    let source_inventory = catalog_inventory_sha256(&children);
    Ok(SloCostProfileReceipt {
        storage: if path.is_some() {
            ferrum_types::SloCostProfileStorage::File
        } else {
            ferrum_types::SloCostProfileStorage::Memory
        },
        clock_basis: ferrum_types::SloCostProfileClockBasis::ImportedWallClock,
        selected_whole_wave: None,
        structured_whole_wave: None,
        structured_whole_wave_v2: Some(ferrum_types::SloStructuredWholeWaveReceiptV2 {
            model_revision: MODEL_REVISION_V2.into(),
            artifact_kind: ferrum_types::SloStructuredArtifactKindV2::CatalogV1,
            child_count: children.len(),
            total_imported_bytes: total_bytes,
            total_shape_rows: total_rows,
            source_inventory_sha256: Some(source_inventory),
            children,
        }),
        schema_version: 1,
        path: path.map(Path::to_path_buf),
        file_sha256: format!("sha256:{}", hex(&digest)),
        file_bytes: metadata_bytes,
        // Catalog has no new measurement clock; report latest original close.
        generated_unix_ns: generated,
        loaded_unix_ns: loaded.unwrap(),
        conservative_clock_error_ns: error,
        declared_local_clock_max_error_ns: Some(declared),
        oldest_imported_age_ns: Some(oldest),
        newest_imported_age_ns: Some(newest),
        offered_samples: size(offered)?,
        recorded_samples: size(members)?,
        stale_samples: 0,
        skipped_samples: Default::default(),
        model_version: 1,
        bucket_count: snapshot.len(),
        source_generator: "ferrum.structured-v2-catalog".into(),
        source_generator_revision: "1".into(),
        source_measurement_protocol:
            "ferrum.structured-v2-catalog-content-inventory.v1 (original protocols in children)"
                .into(),
        source_observation_artifact_sha256: source_inventory,
    })
}

/// The physical file/row accounting is explicit, never path/hash deduplication
/// applied opportunistically to unrelated legacy catalogs.
pub(super) fn shared(
    path: &Path,
    imported: &file::ImportedStructuredCatalogV11,
    snapshot: &StructuredSnapshot,
    declared: u64,
) -> Result<SloCostProfileReceipt, FerrumError> {
    let mut receipt = catalog(
        Some(path),
        imported.file_sha256,
        size(imported.file_bytes)?,
        snapshot,
        declared,
    )?;
    let v2 = receipt.structured_whole_wave_v2.as_mut().unwrap();
    v2.artifact_kind = ferrum_types::SloStructuredArtifactKindV2::SharedCatalogV11;
    v2.total_imported_bytes = imported
        .file_bytes
        .checked_add(imported.source_bytes)
        .ok_or_else(|| FerrumError::config("shared imported byte count overflow"))?;
    v2.total_shape_rows = imported.total_shape_rows;
    // Inventory continues to bind every child independently; the actual source
    // digest and physical accounting are separately unambiguous.
    receipt.schema_version = 11;
    receipt.offered_samples = size(imported.offered_attempts)?;
    receipt.source_generator = "ferrum.structured-shared-v2-catalog".into();
    receipt.source_generator_revision = "11".into();
    receipt.source_measurement_protocol =
        format!("source4:sha256:{}", hex(&imported.capture_protocol));
    receipt.source_observation_artifact_sha256 = imported.source_sha256;
    Ok(receipt)
}

// Source5 keeps a distinct receipt identity and counts the common source once.
pub(super) fn prefix(
    path: &Path,
    imported: &file::ImportedStructuredCatalogV12,
    snapshot: &StructuredSnapshot,
    declared: u64,
) -> Result<SloCostProfileReceipt, FerrumError> {
    let mut receipt = catalog(
        Some(path),
        imported.file_sha256,
        size(imported.file_bytes)?,
        snapshot,
        declared,
    )?;
    let v2 = receipt.structured_whole_wave_v2.as_mut().unwrap();
    v2.artifact_kind = ferrum_types::SloStructuredArtifactKindV2::PrefixCatalogV12;
    v2.total_imported_bytes = imported
        .file_bytes
        .checked_add(imported.source_bytes)
        .ok_or_else(|| FerrumError::config("shared imported byte count overflow"))?;
    v2.total_shape_rows = imported.total_shape_rows;
    // Inventory continues to bind every child independently; the actual source
    // digest and physical accounting are separately unambiguous.
    receipt.schema_version = 12;
    receipt.offered_samples = size(imported.offered_attempts)?;
    receipt.source_generator = "ferrum.structured-prefix-v2-catalog".into();
    receipt.source_generator_revision = "12".into();
    receipt.source_measurement_protocol =
        format!("source5:sha256:{}", hex(&imported.capture_protocol));
    receipt.source_observation_artifact_sha256 = imported.source_sha256;
    Ok(receipt)
}

pub(super) fn service(
    path: Option<&Path>,
    imported: &file::ImportedStructuredCatalogV13,
    snapshot: &StructuredSnapshot,
    declared: Option<u64>,
) -> Result<SloCostProfileReceipt, FerrumError> {
    let mut receipt = catalog(
        path,
        imported.file_sha256,
        size(imported.file_bytes)?,
        snapshot,
        declared.unwrap_or(0),
    )?;
    receipt.declared_local_clock_max_error_ns = declared;
    receipt.clock_basis = imported.children[0].provenance().clock_basis;
    if imported.children.iter().any(|c| {
        c.provenance().clock_basis != receipt.clock_basis
            || c.provenance().storage != receipt.storage
            || c.provenance().loaded_from.as_deref() != path
    }) {
        return Err(FerrumError::config(
            "service catalog mixes clock bases or storage locations",
        ));
    }
    let v2 = receipt.structured_whole_wave_v2.as_mut().unwrap();
    v2.artifact_kind = ferrum_types::SloStructuredArtifactKindV2::ServiceWindowCatalogV13;
    v2.total_imported_bytes = imported
        .file_bytes
        .checked_add(imported.source_bytes)
        .ok_or_else(|| FerrumError::config("service imported byte count overflow"))?;
    v2.total_shape_rows = imported.total_shape_rows;
    receipt.schema_version = 13;
    receipt.offered_samples = size(imported.offered_attempts)?;
    receipt.source_generator = "ferrum.structured-service-window-catalog".into();
    receipt.source_generator_revision = "13".into();
    receipt.source_measurement_protocol =
        format!("source6:sha256:{}", hex(&imported.capture_protocol));
    receipt.source_observation_artifact_sha256 = imported.source_sha256;
    Ok(receipt)
}

// Source7 preserves its original global block prefix and per-owner phases.
pub(super) fn owner_blocks(
    path: Option<&Path>,
    imported: &file::ImportedStructuredCatalogV14,
    snapshot: &StructuredSnapshot,
    declared: Option<u64>,
) -> Result<SloCostProfileReceipt, FerrumError> {
    if imported.journal_bytes < imported.source_bytes
        || (path.is_none() && imported.journal_bytes != imported.source_bytes)
    {
        return Err(FerrumError::config(
            "owner block catalog journal and checkpoint byte accounting differ",
        ));
    }
    let mut receipt = catalog(
        path,
        imported.file_sha256,
        size(imported.file_bytes)?,
        snapshot,
        declared.unwrap_or(0),
    )?;
    receipt.declared_local_clock_max_error_ns = declared;
    receipt.clock_basis = imported.children[0].provenance().clock_basis;
    if imported.children.iter().any(|c| {
        c.provenance().clock_basis != receipt.clock_basis
            || c.provenance().storage != receipt.storage
            || c.provenance().loaded_from.as_deref() != path
    }) {
        return Err(FerrumError::config(
            "owner block catalog mixes clock bases or storage locations",
        ));
    }
    let v2 = receipt.structured_whole_wave_v2.as_mut().unwrap();
    v2.artifact_kind = ferrum_types::SloStructuredArtifactKindV2::OwnerBlockCatalogV14;
    v2.total_imported_bytes = imported
        .file_bytes
        .checked_add(imported.journal_bytes)
        .ok_or_else(|| FerrumError::config("owner block imported byte count overflow"))?;
    v2.total_shape_rows = imported.total_shape_rows;
    receipt.schema_version = 14;
    receipt.offered_samples = size(imported.offered_attempts)?;
    receipt.source_generator = "ferrum.structured-owner-block-catalog".into();
    receipt.source_generator_revision = "14".into();
    receipt.source_measurement_protocol =
        format!("source7:sha256:{}", hex(&imported.capture_protocol));
    receipt.source_observation_artifact_sha256 = imported.source_sha256;
    Ok(receipt)
}

// Source8 binds preparation and ordinary suffixes to the same original journal.
pub(super) fn prepared_owner_blocks(
    path: Option<&Path>,
    imported: &file::ImportedPreparedOwnerBlockCatalogV15,
    snapshot: &StructuredSnapshot,
    declared: Option<u64>,
) -> Result<SloCostProfileReceipt, FerrumError> {
    if imported.journal_bytes < imported.source_bytes
        || (path.is_none() && imported.journal_bytes != imported.source_bytes)
    {
        return Err(FerrumError::config(
            "prepared owner block catalog journal and checkpoint byte accounting differ",
        ));
    }
    let mut receipt = catalog(
        path,
        imported.file_sha256,
        size(imported.file_bytes)?,
        snapshot,
        declared.unwrap_or(0),
    )?;
    receipt.declared_local_clock_max_error_ns = declared;
    receipt.clock_basis = imported.children[0].provenance().clock_basis;
    if imported.children.iter().any(|c| {
        c.provenance().clock_basis != receipt.clock_basis
            || c.provenance().storage != receipt.storage
            || c.provenance().loaded_from.as_deref() != path
    }) {
        return Err(FerrumError::config(
            "prepared owner block catalog mixes clock bases or storage locations",
        ));
    }
    let v2 = receipt.structured_whole_wave_v2.as_mut().unwrap();
    v2.artifact_kind = ferrum_types::SloStructuredArtifactKindV2::PreparedOwnerBlockCatalogV15;
    v2.total_imported_bytes = imported
        .file_bytes
        .checked_add(imported.journal_bytes)
        .ok_or_else(|| FerrumError::config("prepared owner block imported byte count overflow"))?;
    v2.total_shape_rows = imported.total_shape_rows;
    receipt.schema_version = 15;
    receipt.offered_samples = size(imported.offered_attempts)?;
    receipt.source_generator = "ferrum.structured-prepared-owner-block-catalog".into();
    receipt.source_generator_revision = "15".into();
    receipt.source_measurement_protocol =
        format!("source8:sha256:{}", hex(&imported.capture_protocol));
    receipt.source_observation_artifact_sha256 = imported.source_sha256;
    Ok(receipt)
}
