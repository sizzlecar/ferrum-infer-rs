use super::*;

#[test]
fn source8_selected_domains_preserve_full_preparation_source_after_mixed_expiry() {
    let boot = domain(1);
    let (bytes, _, checkpoint) = collector::collected_with_domain_routes(boot.clone());
    let closing = checkpoint.population.closing;
    let limits = CostProfileLoadLimits::default();
    let original = checkpoint
        .activate_same_boot_memory(closing, &boot, &limits)
        .unwrap();
    assert_eq!(
        original.children.len(),
        2,
        "two independently qualified routes"
    );
    let oldest = original
        .children
        .iter()
        .min_by_key(|child| closing.monotonic_ns - child.provenance().oldest_imported_age_ns)
        .unwrap();
    let at = now(
        closing.monotonic_ns - oldest.provenance().oldest_imported_age_ns
            + oldest.runtime_limits().1
            + 1,
    );
    let selected: Vec<_> = original
        .children
        .iter()
        .filter(|child| child.is_current_local(at.monotonic_ns).is_ok())
        .map(|child| *child.domain_signature())
        .collect();
    assert_eq!(selected.len(), 1);
    let files = Files::new(&bytes);
    assert!(export_structured_profile_v15_same_boot(
        &files.source,
        Sha256::digest(&bytes).into(),
        bytes.len() as u64,
        &files.profile,
        &boot,
        at,
        &limits,
    )
    .is_err());
    export_structured_profile_v15_same_boot_selected(
        &files.source,
        Sha256::digest(&bytes).into(),
        bytes.len() as u64,
        &files.profile,
        &boot,
        at,
        &limits,
        &selected,
    )
    .unwrap();
    let loaded = load_structured_profile_v15_same_boot_selected(
        &files.profile,
        &old::fingerprint(),
        &limits,
        &boot,
        at,
        &selected,
    )
    .unwrap();
    assert_eq!(loaded.children.len(), 1);
    assert_eq!(loaded.offered_attempts, original.offered_attempts);
    assert_eq!(loaded.total_shape_rows, original.total_shape_rows);
    assert_eq!(loaded.source_bytes, original.source_bytes);
    assert_eq!(loaded.source_sha256, original.source_sha256);
    let survivor = original
        .children
        .iter()
        .find(|child| child.domain_signature() == loaded.children[0].domain_signature())
        .unwrap();
    assert_eq!(
        loaded.children[0].parameters_signature(),
        survivor.parameters_signature()
    );
    assert_eq!(
        loaded.children[0].provenance().clock,
        survivor.provenance().clock
    );
    assert_eq!(
        loaded.children[0].provenance().phases,
        survivor.provenance().phases
    );
    assert!(loaded.children[0].is_current_local(at.monotonic_ns).is_ok());
    assert!(load_structured_profile_v15_same_boot_selected(
        &files.profile,
        &old::fingerprint(),
        &limits,
        &boot,
        at,
        &[*oldest.domain_signature()],
    )
    .is_err());
    assert!(load_structured_profile_v15_same_boot_selected(
        &files.profile,
        &old::fingerprint(),
        &limits,
        &domain(2),
        at,
        &selected,
    )
    .is_err());
    let mut short = limits.clone();
    short.max_samples =
        std::num::NonZeroUsize::new(original.offered_attempts as usize - 1).unwrap();
    assert!(
        load_structured_profile_v15_same_boot_selected(
            &files.profile,
            &old::fingerprint(),
            &short,
            &boot,
            at,
            &selected,
        )
        .is_err(),
        "preparation and omitted owner work still consume full replay budget"
    );
    let mut metadata: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&files.profile).unwrap()).unwrap();
    let omitted = metadata["children"]
        .as_array_mut()
        .unwrap()
        .iter_mut()
        .find(|child| {
            serde_json::from_value::<[u8; 32]>(child["domain_signature"].clone()).unwrap()
                == *oldest.domain_signature()
        })
        .unwrap();
    let value = omitted["parameters_sha256"][0].as_u64().unwrap();
    omitted["parameters_sha256"][0] = serde_json::json!((value + 1) % 256);
    std::fs::write(&files.profile, serde_json::to_vec(&metadata).unwrap()).unwrap();
    assert!(
        load_structured_profile_v15_same_boot_selected(
            &files.profile,
            &old::fingerprint(),
            &limits,
            &boot,
            at,
            &selected,
        )
        .is_err(),
        "omitted metadata tampering must be detected before filtering"
    );
}
