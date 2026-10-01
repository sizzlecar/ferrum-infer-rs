//! Replay original cache bytes without refreshing any clock or fitted state.
use super::*;
use crate::implementations::continuous::cost_model::structured_v2::StructuredUnknownV2;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ExpirySpec {
    source: PathBuf,
    source_sha256: [u8; 32],
    profile: PathBuf,
    profile_sha256: [u8; 32],
    final_cost_now_ns: u64,
}

#[test]
#[ignore = "requires original source8/cache profile via FERRUM_ARCHIVED_EXPIRY_SPEC"]
fn archived_cache_source8_original_sample_expiry() {
    let spec: ExpirySpec = serde_json::from_slice(
        &std::fs::read(std::env::var_os("FERRUM_ARCHIVED_EXPIRY_SPEC").expect("expiry spec"))
            .unwrap(),
    )
    .unwrap();
    let source = std::fs::read(&spec.source).unwrap();
    assert_eq!(
        <[u8; 32]>::from(Sha256::digest(&source)),
        spec.source_sha256
    );
    let profile_bytes = std::fs::read(&spec.profile).unwrap();
    assert_eq!(
        <[u8; 32]>::from(Sha256::digest(&profile_bytes)),
        spec.profile_sha256
    );
    let profile: Value = serde_json::from_slice(&profile_bytes).unwrap();
    assert_eq!(profile["schema_version"], 15);
    let prefix_bytes = profile["source_bytes"].as_u64().unwrap() as usize;
    let prefix = &source[..prefix_bytes];
    let prefix_sha: [u8; 32] = serde_json::from_value(profile["source_sha256"].clone()).unwrap();
    assert_eq!(<[u8; 32]>::from(Sha256::digest(prefix)), prefix_sha);
    let limits = CostProfileLoadLimits::default();
    let checkpoint = replay_structured_source_v8(prefix, &limits).unwrap();
    let original_closing = checkpoint.population.closing;
    let catalog = checkpoint
        .activate_same_process_memory(original_closing, &limits)
        .unwrap();
    assert_eq!(
        catalog.children.len(),
        profile["children"].as_array().unwrap().len()
    );
    let mut expired = 0;
    for child in &catalog.children {
        let declared = profile["children"]
            .as_array()
            .unwrap()
            .iter()
            .find(|p| {
                serde_json::from_value::<[u8; 32]>(p["parameters_sha256"].clone()).unwrap()
                    == child.parameters_signature()
            })
            .expect("original child parameters must match exact archived profile");
        assert_eq!(
            declared["domain_signature"],
            serde_json::to_value(child.domain_signature()).unwrap()
        );
        let p = child.provenance();
        let oldest_original = p
            .clock
            .model_anchor_ns
            .checked_sub(p.oldest_imported_age_ns)
            .unwrap();
        let expires_at_ns = oldest_original
            .checked_add(child.runtime_limits().1)
            .unwrap();
        assert!(child.is_current_local(expires_at_ns).is_ok());
        assert_eq!(
            child.is_current_local(expires_at_ns + 1),
            Err(StructuredUnknownV2::Stale)
        );
        let final_result = child.is_current_local(spec.final_cost_now_ns);
        expired += usize::from(final_result == Err(StructuredUnknownV2::Stale));
        eprintln!(
            "ARCHIVED_CACHE_EXPIRY {}",
            json!({
                "parameters_sha256": child.parameters_signature(),
                "domain_signature": child.domain_signature(),
                "original_closing_ns": original_closing.monotonic_ns,
                "model_anchor_ns": p.clock.model_anchor_ns,
                "oldest_imported_age_ns": p.oldest_imported_age_ns,
                "oldest_original_sample_ns": oldest_original,
                "max_sample_age_ns": child.runtime_limits().1,
                "expires_at_ns": expires_at_ns,
                "final_cost_now_ns": spec.final_cost_now_ns,
                "past_expiry_ns": spec.final_cost_now_ns.saturating_sub(expires_at_ns),
                "final_result": format!("{final_result:?}"),
            })
        );
    }
    assert!(
        expired > 0,
        "claimed expiry must be established from actual original sample clock"
    );
}
