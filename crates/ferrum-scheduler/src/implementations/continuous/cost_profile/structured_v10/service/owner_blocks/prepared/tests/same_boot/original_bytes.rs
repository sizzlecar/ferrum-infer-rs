//! The storage codec is outside scheduler authority; raw receipts remain exact.
use super::*;

#[test]
fn source8_original_journal_bytes_match_path_and_preserve_every_gate() {
    let domain = domain(1);
    let (mut bytes, mut collector, checkpoint) = collector::collected_with_domain(domain.clone());
    let closing = checkpoint.population.closing.monotonic_ns;
    let limits = CostProfileLoadLimits::default();
    let cutoff = bytes.len() as u64;
    let memory = checkpoint
        .activate_same_boot_memory(now(closing + 1), &domain, &limits)
        .unwrap();
    let mut selected: Vec<_> = memory
        .children
        .iter()
        .map(|c| *c.domain_signature())
        .collect();
    selected.sort();
    let stop = collector.stop(now(closing + 2)).unwrap();
    bytes.extend(record_bytes_v7(&stop).unwrap());
    assert!(bytes.len() as u64 > cutoff);
    let full_sha: [u8; 32] = Sha256::digest(&bytes).into();
    let files = Files::new(&bytes);
    let export_now = now(closing + 3);
    export_structured_profile_v15_same_boot_selected(
        &files.source,
        full_sha,
        cutoff,
        &files.profile,
        &domain,
        export_now,
        &limits,
        &selected,
    )
    .unwrap();
    let original_profile = std::fs::read(&files.profile).unwrap();
    let bytes_profile = files.profile.with_file_name("bytes-profile.json");
    export_structured_profile_v15_same_boot_selected_from_original_bytes(
        &files.source,
        &bytes,
        full_sha,
        cutoff,
        &bytes_profile,
        &domain,
        export_now,
        &limits,
        &selected,
    )
    .unwrap();
    assert_eq!(std::fs::read(&bytes_profile).unwrap(), original_profile);
    let load_now = now(closing + 101);
    let path = load_structured_profile_v15_same_boot_selected(
        &files.profile,
        &old::fingerprint(),
        &limits,
        &domain,
        load_now,
        &selected,
    )
    .unwrap();
    let from_bytes = load_structured_profile_v15_same_boot_selected_from_original_bytes(
        &bytes_profile,
        &bytes,
        &files.source,
        &old::fingerprint(),
        &limits,
        &domain,
        load_now,
        &selected,
    )
    .unwrap();
    assert_eq!(path.file_sha256, from_bytes.file_sha256);
    assert_eq!(path.source_sha256, from_bytes.source_sha256);
    assert_eq!(path.capture_protocol, from_bytes.capture_protocol);
    assert_eq!(path.file_bytes, from_bytes.file_bytes);
    assert_eq!(path.source_bytes, cutoff);
    assert_eq!(path.source_bytes, from_bytes.source_bytes);
    assert_eq!(path.journal_bytes, bytes.len() as u64);
    assert_eq!(path.journal_bytes, from_bytes.journal_bytes);
    assert_eq!(path.offered_attempts, from_bytes.offered_attempts);
    assert_eq!(path.total_shape_rows, from_bytes.total_shape_rows);
    assert_eq!(path.children.len(), from_bytes.children.len());
    for (before, after) in path.children.iter().zip(&from_bytes.children) {
        assert_eq!(before.parameters_signature(), after.parameters_signature());
        assert_eq!(before.domain_signature(), after.domain_signature());
        assert_eq!(before.provenance().phases, after.provenance().phases);
        assert_eq!(before.provenance().clock, after.provenance().clock);
        assert_eq!(
            before.provenance().oldest_imported_age_ns,
            after.provenance().oldest_imported_age_ns
        );
        assert_eq!(
            before.provenance().newest_imported_age_ns,
            after.provenance().newest_imported_age_ns
        );
    }

    // A storage layer may retain encoded bytes at this exact managed identity.
    // The byte API never falls back to reopening it as the original stream.
    std::fs::write(&files.source, b"encoded storage is opaque to scheduler").unwrap();
    assert!(load_structured_profile_v15_same_boot_selected(
        &files.profile,
        &old::fingerprint(),
        &limits,
        &domain,
        load_now,
        &selected,
    )
    .is_err());
    assert!(
        load_structured_profile_v15_same_boot_selected_from_original_bytes(
            &files.profile,
            &bytes,
            &files.source,
            &old::fingerprint(),
            &limits,
            &domain,
            load_now,
            &selected,
        )
        .is_ok()
    );
    let other = files.source.with_file_name("other-source.bin");
    std::fs::write(&other, &bytes).unwrap();
    assert!(
        load_structured_profile_v15_same_boot_selected_from_original_bytes(
            &files.profile,
            &bytes,
            &other,
            &old::fingerprint(),
            &limits,
            &domain,
            load_now,
            &selected,
        )
        .is_err()
    );

    // Later tail is outside the qualified prefix but remains fully hash-bound.
    let mut corrupt = bytes.clone();
    *corrupt.last_mut().unwrap() ^= 1;
    assert!(
        load_structured_profile_v15_same_boot_selected_from_original_bytes(
            &files.profile,
            &corrupt,
            &files.source,
            &old::fingerprint(),
            &limits,
            &domain,
            load_now,
            &selected,
        )
        .is_err()
    );
    assert!(
        load_structured_profile_v15_same_boot_selected_from_original_bytes(
            &files.profile,
            &bytes[..bytes.len() - 1],
            &files.source,
            &old::fingerprint(),
            &limits,
            &domain,
            load_now,
            &selected,
        )
        .is_err()
    );
    let mut changed: serde_json::Value = serde_json::from_slice(&original_profile).unwrap();
    changed["source_sha256"] = serde_json::json!(vec![9u8; 32]);
    std::fs::write(&files.profile, serde_json::to_vec(&changed).unwrap()).unwrap();
    assert!(
        load_structured_profile_v15_same_boot_selected_from_original_bytes(
            &files.profile,
            &bytes,
            &files.source,
            &old::fingerprint(),
            &limits,
            &domain,
            load_now,
            &selected,
        )
        .is_err()
    );
    std::fs::write(&files.profile, &original_profile).unwrap();

    // Decoded source + metadata share the original byte cap; compressed size
    // cannot buy more raw records, rows, or a renewed sample lifetime.
    let mut bounded = limits.clone();
    bounded.max_file_bytes =
        std::num::NonZeroUsize::new(bytes.len() + original_profile.len() - 1).unwrap();
    assert!(matches!(
        load_structured_profile_v15_same_boot_selected_from_original_bytes(
            &files.profile,
            &bytes,
            &files.source,
            &old::fingerprint(),
            &bounded,
            &domain,
            load_now,
            &selected,
        ),
        Err(CostProfileError::Limit(_))
    ));
    for rows in [false, true] {
        let mut bounded = limits.clone();
        if rows {
            bounded.max_total_shape_rows = std::num::NonZeroUsize::new(1).unwrap();
        } else {
            bounded.max_samples = std::num::NonZeroUsize::new(1).unwrap();
        }
        assert!(
            load_structured_profile_v15_same_boot_selected_from_original_bytes(
                &files.profile,
                &bytes,
                &files.source,
                &old::fingerprint(),
                &bounded,
                &domain,
                load_now,
                &selected,
            )
            .is_err()
        );
    }
    let mut stale = limits.clone();
    stale.max_profile_age_ns = std::num::NonZeroU64::new(1).unwrap();
    assert!(matches!(
        load_structured_profile_v15_same_boot_selected_from_original_bytes(
            &files.profile,
            &bytes,
            &files.source,
            &old::fingerprint(),
            &stale,
            &domain,
            load_now,
            &selected,
        ),
        Err(CostProfileError::Clock(_))
    ));
    let rejected = files.profile.with_file_name("rejected.json");
    assert!(
        export_structured_profile_v15_same_boot_selected_from_original_bytes(
            &files.source,
            &bytes,
            [0; 32],
            cutoff,
            &rejected,
            &domain,
            export_now,
            &limits,
            &selected,
        )
        .is_err()
    );
    bounded.max_file_bytes = std::num::NonZeroUsize::new(bytes.len() - 1).unwrap();
    assert!(matches!(
        export_structured_profile_v15_same_boot_selected_from_original_bytes(
            &files.source,
            &bytes,
            full_sha,
            cutoff,
            &rejected,
            &domain,
            export_now,
            &bounded,
            &selected,
        ),
        Err(CostProfileError::Limit(_))
    ));
    assert!(!rejected.exists());
}
