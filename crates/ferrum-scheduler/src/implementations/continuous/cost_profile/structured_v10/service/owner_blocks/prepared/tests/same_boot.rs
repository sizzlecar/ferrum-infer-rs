use super::*;
mod expiry;
mod original_bytes;
use ferrum_interfaces::execution_cost::CostMonotonicDomainV1;
use ferrum_types::SloCostProfileClockBasis;
use std::sync::atomic::{AtomicU64, Ordering};

fn domain(boot: u8) -> CostMonotonicDomainV1 {
    CostMonotonicDomainV1::new_macos_continuous([boot; 16]).unwrap()
}
fn now(monotonic_ns: u64) -> StructuredServiceClockV7 {
    StructuredServiceClockV7 {
        monotonic_ns,
        wall_unix_ns: 1,
    }
}
struct Files {
    directory: PathBuf,
    source: PathBuf,
    profile: PathBuf,
}
impl Files {
    fn new(bytes: &[u8]) -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let directory = std::env::temp_dir().join(format!(
            "ferrum-source8-same-boot-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&directory).unwrap();
        let source = directory.join("source.jsonl");
        let profile = directory.join("profile.json");
        std::fs::write(&source, bytes).unwrap();
        Self {
            directory,
            source,
            profile,
        }
    }
}
impl Drop for Files {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.directory);
    }
}

#[test]
fn source8_same_boot_header_none_keeps_original_wire_and_protocol() {
    let h = header();
    let bytes = record_bytes_v7(&h).unwrap();
    assert!(serde_json::from_slice::<serde_json::Value>(&bytes)
        .unwrap()
        .get("monotonic_domain")
        .is_none());
    let mut legacy = Sha256::new();
    legacy.update(PREPARED_OWNER_BLOCK_SOURCE_PROTOCOL_V8.as_bytes());
    legacy.update([0]);
    legacy.update(MODEL_REVISION_V2.as_bytes());
    legacy.update(h.declaration_sha256);
    legacy.update(record_bytes_v7(&h.fingerprint).unwrap());
    legacy.update(h.maximum_file_bytes.to_le_bytes());
    legacy.update(record_bytes_v7(&h.cohort_phase_policy()).unwrap());
    legacy.update(record_bytes_v7(&h.tail_policy()).unwrap());
    assert_eq!(h.protocol, <[u8; 32]>::from(legacy.finalize()));
    let parsed: StructuredPreparedOwnerBlockHeaderV8 = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(record_bytes_v7(&parsed).unwrap(), bytes);
    let bound = StructuredPreparedOwnerBlockHeaderV8::new_with_monotonic_domain(
        h.capture_identity,
        h.generation,
        h.fingerprint,
        h.producer,
        h.opening,
        h.declaration,
        h.maximum_file_bytes,
        domain(1),
    )
    .unwrap();
    assert_ne!(bound.protocol, h.protocol);
    let mut forged = bound;
    forged.monotonic_domain = Some(domain(2));
    assert!(forged.validate().is_err());
}

#[test]
fn source8_same_boot_replays_real_prefix_and_terminal_cohorts_without_new_ttl() {
    let original = domain(1);
    let (mut bytes, mut collector, checkpoint) = collector::collected_with_domain(original.clone());
    let limits = CostProfileLoadLimits::default();
    assert_eq!(
        collector.source_receipt(),
        (bytes.len() as u64, Sha256::digest(&bytes).into())
    );
    let closing_clock = checkpoint.population.closing;
    let closing = closing_clock.monotonic_ns;
    let ttl = checkpoint
        .population
        .header
        .declaration
        .settings
        .max_sample_age_ns;
    let memory = checkpoint
        .activate_same_boot_memory(now(closing + 1), &original, &limits)
        .unwrap();
    let checkpoint_bytes = bytes.len() as u64;
    let stop = collector
        .stop(StructuredServiceClockV7 {
            monotonic_ns: closing + 2,
            wall_unix_ns: closing_clock.wall_unix_ns + 2,
        })
        .unwrap();
    bytes.extend(record_bytes_v7(&stop).unwrap());
    assert_eq!(
        collector.source_receipt(),
        (bytes.len() as u64, Sha256::digest(&bytes).into())
    );
    // The final journal includes its footer; only the original checkpoint is replayed.
    assert!(replay_structured_source_v8(&bytes, &limits).is_err());
    let files = Files::new(&bytes);
    export_structured_profile_v15_same_boot(
        &files.source,
        Sha256::digest(&bytes).into(),
        checkpoint_bytes,
        &files.profile,
        &original,
        now(closing + 3),
        &limits,
    )
    .unwrap();
    let metadata: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&files.profile).unwrap()).unwrap();
    assert!(metadata.get("source_clock_max_error_ns").is_none());
    let restarted = domain(1);
    let imported = load_structured_profile_v15_same_boot(
        &files.profile,
        &old::fingerprint(),
        &limits,
        &restarted,
        now(closing + 101),
    )
    .unwrap();
    assert_eq!(memory.children.len(), imported.children.len());
    for (before, after) in memory.children.iter().zip(&imported.children) {
        assert_eq!(before.parameters_signature(), after.parameters_signature());
        assert_eq!(before.monotonic_domain(), Some(&original));
        assert_eq!(after.monotonic_domain(), Some(&original));
        assert_eq!(
            before.provenance().clock.source_monotonic_anchor_ns,
            after.provenance().clock.source_monotonic_anchor_ns
        );
        assert_eq!(
            before.provenance().clock.model_anchor_ns,
            after.provenance().clock.model_anchor_ns
        );
        assert_eq!(before.provenance().protocol, after.provenance().protocol);
        assert_eq!(before.provenance().phases, after.provenance().phases);
        assert_eq!(
            after.provenance().clock_basis,
            SloCostProfileClockBasis::SameBootMonotonic
        );
        assert_eq!(
            after.provenance().oldest_imported_age_ns,
            before.provenance().oldest_imported_age_ns + 100
        );
        assert_eq!(
            after.provenance().newest_imported_age_ns,
            before.provenance().newest_imported_age_ns + 100
        );
    }
    assert_eq!(memory.source_sha256, imported.source_sha256);
    assert_eq!(memory.offered_attempts, imported.offered_attempts);
    assert!(matches!(
        load_structured_profile_v15_same_boot(
            &files.profile,
            &old::fingerprint(),
            &limits,
            &domain(2),
            now(closing + 101)
        ),
        Err(CostProfileError::Clock(_))
    ));
    assert!(load_structured_profile_v15_same_boot(
        &files.profile,
        &old::fingerprint(),
        &limits,
        &restarted,
        now(closing + ttl + 1)
    )
    .is_err());
    let mut short = limits.clone();
    short.max_profile_age_ns = NonZeroU64::new(1).unwrap();
    assert!(load_structured_profile_v15_same_boot(
        &files.profile,
        &old::fingerprint(),
        &short,
        &restarted,
        now(closing + 2)
    )
    .is_err());
}

#[test]
fn source8_same_boot_metadata_cannot_upgrade_unbound_preparation_source() {
    let (bytes, _, checkpoint) = collector::collected();
    let closing = checkpoint.population.closing.monotonic_ns;
    let files = Files::new(&bytes);
    let limits = CostProfileLoadLimits::default();
    let current = domain(1);
    assert!(checkpoint
        .activate_same_boot_memory(now(closing), &current, &limits)
        .is_err());
    assert!(export_structured_profile_v15_same_boot(
        &files.source,
        Sha256::digest(&bytes).into(),
        bytes.len() as u64,
        &files.profile,
        &current,
        now(closing),
        &limits,
    )
    .is_err());
    export_structured_profile_v15(
        &files.source,
        Sha256::digest(&bytes).into(),
        bytes.len() as u64,
        &files.profile,
        0,
        &limits,
    )
    .unwrap();
    let mut metadata: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&files.profile).unwrap()).unwrap();
    metadata["monotonic_domain"] = serde_json::to_value(&current).unwrap();
    std::fs::write(&files.profile, serde_json::to_vec(&metadata).unwrap()).unwrap();
    assert!(load_structured_profile_v15_same_boot(
        &files.profile,
        &old::fingerprint(),
        &limits,
        &current,
        now(closing),
    )
    .is_err());
}
