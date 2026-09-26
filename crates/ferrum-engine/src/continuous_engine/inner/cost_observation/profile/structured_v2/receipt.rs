use super::*;
fn size(n: u64) -> Result<usize, FerrumError> {
    usize::try_from(n).map_err(|_| FerrumError::config("structured V2 receipt size overflow"))
}
fn hex(hash: &[u8; 32]) -> String {
    hash.iter().map(|b| format!("{b:02x}")).collect()
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
    Ok(SloCostProfileReceipt {
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
        declared_local_clock_max_error_ns: declared,
        oldest_imported_age_ns: Some(p.oldest_imported_age_ns),
        newest_imported_age_ns: Some(p.newest_imported_age_ns),
        offered_samples: size(p.offered_attempts)?,
        recorded_samples: size(p.reserved_members)?,
        stale_samples: 0,
        skipped_samples: Default::default(),
        model_version: 1,
        bucket_count: 1,
        source_generator: p.producer["executable_path"]
            .as_str()
            .ok_or_else(|| FerrumError::config("source3 validated producer missing"))?
            .into(),
        source_generator_revision: p.producer["source_revision"]
            .as_str()
            .map(str::to_owned)
            .unwrap_or_else(|| {
                format!(
                    "package:{}",
                    p.producer["package_version"]
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
    path: &Path,
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
    let mut inventory = Sha256::new();
    inventory.update(b"ferrum.structured-v2-catalog-content-inventory.v1\0");
    inventory.update((snapshot.len() as u64).to_le_bytes());
    // BTreeMap canonical domain order; paths locate artifacts but are not an
    // alternative model identity. Profile bytes already bind original source path.
    for (domain, model) in snapshot.children.iter() {
        let p = model.provenance();
        let r = child(model)?;
        inventory.update(domain);
        inventory.update(r.owner_sha256);
        inventory.update(r.scope_sha256);
        inventory.update(p.file_sha256);
        inventory.update(p.source_sha256);
        inventory.update(p.parameters_sha256);
        inventory.update(p.file_bytes.to_le_bytes());
        inventory.update(p.source_bytes.to_le_bytes());
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
    let source_inventory: [u8; 32] = inventory.finalize().into();
    Ok(SloCostProfileReceipt {
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
        path: path.into(),
        file_sha256: format!("sha256:{}", hex(&digest)),
        file_bytes: metadata_bytes,
        // Catalog has no new measurement clock; report latest original close.
        generated_unix_ns: generated,
        loaded_unix_ns: loaded.unwrap(),
        conservative_clock_error_ns: error,
        declared_local_clock_max_error_ns: declared,
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
        path,
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
