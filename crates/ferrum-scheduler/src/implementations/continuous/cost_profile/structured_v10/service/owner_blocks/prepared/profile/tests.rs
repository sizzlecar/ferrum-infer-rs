//! Original source8 CPU receipts, including preparation and complete cohorts.
//! These test profile validation; the engine tests own private activation/FIFO.
use super::super::tests::collector::collected;
use super::*;
use crate::implementations::continuous::cost_profile::structured_v10::{
    service::{physical, tests::nonnegative::owner_blocks as original_source7},
    tests::fixture as old,
};
use ferrum_types::{SloCostProfileClockBasis, SloCostProfileStorage};

mod archived_bound;
mod numerical_family;

#[test]
fn profile15_streaming_budget_preserves_bounded_file_interoperability() {
    use super::super::tests::collector::collected_streaming;
    let budget = std::num::NonZeroU64::new(1024 * 1024 * 1024).unwrap();
    let tiny_file_limit = CostProfileLoadLimits {
        max_file_bytes: std::num::NonZeroUsize::new(1024).unwrap(),
        ..Default::default()
    };
    // Declare the streaming budget before the first genuine record. Every
    // preparation/ordinary receipt and checkpoint uses that original header.
    let (bytes, collector, checkpoint) = collected_streaming(budget, tiny_file_limit.clone());
    let h = header(&bytes);
    let now = at(&h, checkpoint.population.closing.monotonic_ns + 100);
    assert_eq!(h.maximum_file_bytes, budget.get());
    assert!(bytes.len() > tiny_file_limit.max_file_bytes.get());
    assert_eq!(
        collector.source_receipt(),
        (bytes.len() as u64, Sha256::digest(&bytes).into())
    );
    assert!(replay_structured_source_v8(&bytes, &tiny_file_limit).is_err());
    let original_receipt = checkpoint.source_receipt();
    let memory = checkpoint
        .activate_same_process_memory_streaming(now, &tiny_file_limit, budget)
        .unwrap();
    assert_eq!(memory.source_bytes, original_receipt.0);
    assert_eq!(memory.source_sha256, original_receipt.1);
    assert!(memory
        .children
        .iter()
        .all(|c| c.provenance().storage == SloCostProfileStorage::Memory));
    let limits = CostProfileLoadLimits::default();
    assert!(budget.get() > limits.max_file_bytes.get() as u64);
    let files = Files::new(&bytes);
    let exported = export(&files, &bytes, original_receipt.0);
    assert_eq!(std::fs::read(&files.source).unwrap(), bytes);
    let imported = load_structured_profile_v15(
        &files.profile,
        &old::fingerprint(),
        &limits,
        load_at(now, 17),
    )
    .unwrap();
    assert_eq!(imported.source_sha256, memory.source_sha256);
    assert_eq!(imported.source_bytes, memory.source_bytes);
    assert_eq!(exported.schema_version, 15);
    let q = query(&bytes);
    for (live, file) in memory.children.iter().zip(&imported.children) {
        assert_eq!(live.parameters_signature(), file.parameters_signature());
        assert_eq!(
            live.predict_query_local(&old::fingerprint(), &q, now.monotonic_ns)
                .unwrap()
                .planning_ns,
            file.predict_query_local(&old::fingerprint(), &q, 17)
                .unwrap()
                .planning_ns,
        );
    }
    // The import still charges the actual complete journal plus metadata.
    let total = bytes.len() + std::fs::metadata(&files.profile).unwrap().len() as usize;
    let insufficient = CostProfileLoadLimits {
        max_file_bytes: std::num::NonZeroUsize::new(total - 1).unwrap(),
        ..limits.clone()
    };
    assert!(load_structured_profile_v15(
        &files.profile,
        &old::fingerprint(),
        &insufficient,
        load_at(now, 17)
    )
    .is_err());
    // An opaque checkpoint cannot be activated under a different work budget.
    let (_, _, checkpoint) = collected_streaming(budget, tiny_file_limit.clone());
    assert!(checkpoint
        .activate_same_process_memory_streaming(
            now,
            &tiny_file_limit,
            std::num::NonZeroU64::new(budget.get() - 1).unwrap()
        )
        .is_err());
    let (_, _, checkpoint) = collected_streaming(budget, tiny_file_limit.clone());
    assert!(checkpoint
        .activate_same_process_memory(now, &tiny_file_limit)
        .is_err());
}

struct Files {
    dir: PathBuf,
    source: PathBuf,
    profile: PathBuf,
}
impl Files {
    fn new(source: &[u8]) -> Self {
        let dir = std::env::temp_dir().join(format!("ferrum-source8-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir(&dir).unwrap();
        let source_path = dir.join("source.jsonl");
        std::fs::write(&source_path, source).unwrap();
        Self {
            profile: dir.join("profile.json"),
            source: source_path,
            dir,
        }
    }
}
impl Drop for Files {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

fn header(bytes: &[u8]) -> StructuredPreparedOwnerBlockHeaderV8 {
    serde_json::from_slice(bytes.split_inclusive(|b| *b == b'\n').next().unwrap()).unwrap()
}
fn at(h: &StructuredPreparedOwnerBlockHeaderV8, monotonic_ns: u64) -> StructuredServiceClockV7 {
    StructuredServiceClockV7 {
        monotonic_ns,
        wall_unix_ns: h.opening.wall_unix_ns + monotonic_ns - h.opening.monotonic_ns,
    }
}
fn load_at(now: StructuredServiceClockV7, local: u64) -> ProfileLoadClock {
    ProfileLoadClock {
        wall_unix_ns: Some(now.wall_unix_ns),
        wall_max_error_ns: Some(0),
        monotonic_now_ns: local,
    }
}
fn query(bytes: &[u8]) -> StructuredQueryV2 {
    let h = header(bytes);
    // Reuse the complete original physical/host validator. No raw support
    // vectors, provider identities or query-domain stamps are constructed here.
    for line in bytes.split_inclusive(|b| *b == b'\n').skip(1) {
        let record: StructuredPreparedOwnerBlockRecordV8 = serde_json::from_slice(line).unwrap();
        if let StructuredPreparedOwnerBlockRecordV8::Population(
            StructuredServiceRecordV7::Completed { wave },
        ) = record
        {
            let (input, _, _) = physical::validate_parts(
                &h.fingerprint,
                h.opening.monotonic_ns,
                h.declaration.population.nonnegative_envelope.as_ref(),
                h.opening.monotonic_ns,
                wave.ticket,
                wave.fifo,
                wave.issued_at_ns,
                &wave.host_stages,
                wave.independent.as_ref(),
                &mut physical::Frontiers::default(),
            )
            .unwrap();
            return StructuredQueryV2::exact(input);
        }
    }
    panic!("fixture has no original ordinary settlement");
}
fn export(files: &Files, bytes: &[u8], cutoff: u64) -> StructuredProfileExportReceiptV15 {
    export_structured_profile_v15(
        &files.source,
        Sha256::digest(bytes).into(),
        cutoff,
        &files.profile,
        0,
        &CostProfileLoadLimits::default(),
    )
    .unwrap()
}

#[test]
fn profile15_original_preparation_memory_replay_and_file_predict_identically() {
    let (bytes, collector, checkpoint) = collected();
    let limits = CostProfileLoadLimits::default();
    let h = header(&bytes);
    let closing = checkpoint.population.closing;
    let now = at(&h, closing.monotonic_ns + 100);
    let cutoff = checkpoint.source_receipt();
    assert_eq!(cutoff, (bytes.len() as u64, Sha256::digest(&bytes).into()));
    assert_eq!(checkpoint.accepted_fifo_cutoff(), collector.last_fifo());
    let memory = checkpoint
        .activate_same_process_memory(now, &limits)
        .unwrap();
    let replayed = replay_structured_source_v8(&bytes, &limits)
        .unwrap()
        .activate_same_process_memory(now, &limits)
        .unwrap();
    assert!(!memory.children.is_empty());
    assert_eq!(memory.children.len(), replayed.children.len());
    assert_eq!(memory.offered_attempts, collector.offered());
    assert!(collector.preparation_attempts() > 0);
    let files = Files::new(&bytes);
    let receipt = export(&files, &bytes, cutoff.0);
    assert_eq!(receipt.schema_version, 15);
    let local = 17;
    let imported = load_structured_profile_v15(
        &files.profile,
        &old::fingerprint(),
        &limits,
        load_at(now, local),
    )
    .unwrap();
    assert_eq!(imported.source_sha256, cutoff.1);
    assert_eq!(imported.source_bytes, cutoff.0);
    assert_eq!(imported.journal_bytes, bytes.len() as u64);
    assert_eq!(imported.offered_attempts, memory.offered_attempts);
    assert_eq!(imported.total_shape_rows, memory.total_shape_rows);
    assert_eq!(imported.workload_domain(), memory.workload_domain());
    let q = query(&bytes);
    for ((live, replay), file) in memory
        .children
        .iter()
        .zip(&replayed.children)
        .zip(&imported.children)
    {
        let expected = live
            .predict_query_local(&old::fingerprint(), &q, now.monotonic_ns)
            .unwrap();
        let actual = file
            .predict_query_local(&old::fingerprint(), &q, local)
            .unwrap();
        let again = replay
            .predict_query_local(&old::fingerprint(), &q, now.monotonic_ns)
            .unwrap();
        assert_eq!(
            (actual.planning_ns, actual.valid_until_ns),
            (expected.planning_ns, expected.valid_until_ns)
        );
        assert_eq!(
            (again.planning_ns, again.valid_until_ns),
            (expected.planning_ns, expected.valid_until_ns)
        );
        assert_eq!(live.parameters_signature(), file.parameters_signature());
        assert_eq!(live.domain_signature(), file.domain_signature());
        let l = live.provenance();
        let f = file.provenance();
        assert_eq!(l.schema_version, 15);
        assert_eq!(f.schema_version, 15);
        assert_eq!(l.storage, SloCostProfileStorage::Memory);
        assert_eq!(f.storage, SloCostProfileStorage::File);
        assert_eq!(
            l.clock_basis,
            SloCostProfileClockBasis::SameProcessMonotonic
        );
        assert_eq!(f.clock_basis, SloCostProfileClockBasis::ImportedWallClock);
        assert!(l.loaded_from.is_none() && l.source_path.is_none());
        assert!(f.loaded_from.is_some() && f.source_path.is_some());
        assert_eq!(
            (l.oldest_imported_age_ns, l.newest_imported_age_ns),
            (f.oldest_imported_age_ns, f.newest_imported_age_ns)
        );
        assert_eq!(l.cohort_manifest_sha256, h.declaration_sha256);
        assert_eq!(f.cohort_manifest_sha256, h.declaration_sha256);
        assert_eq!(
            serde_json::to_value(&l.phases).unwrap(),
            serde_json::to_value(&f.phases).unwrap()
        );
    }
    assert!(load_structured_profile_v14(
        &files.profile,
        &old::fingerprint(),
        &limits,
        load_at(now, local)
    )
    .is_err());
    assert!(replay_structured_source_v7(&bytes, &limits).is_err());
}

#[test]
fn profile15_later_import_or_memory_activation_never_renews_original_age() {
    let (bytes, _, checkpoint) = collected();
    let h = header(&bytes);
    let limits = CostProfileLoadLimits::default();
    let now = at(&h, checkpoint.population.closing.monotonic_ns + 100);
    let cutoff = checkpoint.source_receipt().0;
    let memory = checkpoint
        .activate_same_process_memory(now, &limits)
        .unwrap();
    let files = Files::new(&bytes);
    export(&files, &bytes, cutoff);
    let q = query(&bytes);
    let live = &memory.children[0];
    let first = live
        .predict_query_local(&old::fingerprint(), &q, now.monotonic_ns)
        .unwrap();
    let delay = 50;
    let later = at(&h, now.monotonic_ns + delay);
    let local = 71;
    let imported = load_structured_profile_v15(
        &files.profile,
        &old::fingerprint(),
        &limits,
        load_at(later, local),
    )
    .unwrap();
    let after = &imported.children[0];
    let (prediction, model_now) = after
        .predict_query_local_with_clock(&old::fingerprint(), &q, local)
        .unwrap();
    assert_eq!(prediction.valid_until_ns, first.valid_until_ns);
    assert_eq!(model_now, later.monotonic_ns);
    assert_eq!(
        after.provenance().oldest_imported_age_ns,
        live.provenance().oldest_imported_age_ns + delay
    );
    assert_eq!(
        after.provenance().newest_imported_age_ns,
        live.provenance().newest_imported_age_ns + delay
    );
    let expires_local = local + prediction.valid_until_ns - model_now;
    assert!(after
        .predict_query_local(&old::fingerprint(), &q, expires_local + 1)
        .is_err());
    let expired = at(&h, first.valid_until_ns + 1);
    assert!(replay_structured_source_v8(&bytes, &limits)
        .unwrap()
        .activate_same_process_memory(expired, &limits)
        .is_err());
    assert!(load_structured_profile_v15(
        &files.profile,
        &old::fingerprint(),
        &limits,
        load_at(expired, 3)
    )
    .is_err());
}

#[test]
fn profile15_rejects_real_source7_and_relabelled_profile14() {
    let (bytes, _, _, closing) = original_source7::collected();
    let files = Files::new(&bytes);
    let limits = CostProfileLoadLimits::default();
    let hash = Sha256::digest(&bytes).into();
    export_structured_profile_v14(
        &files.source,
        hash,
        bytes.len() as u64,
        &files.profile,
        0,
        &limits,
    )
    .unwrap();
    let load = load_at(closing, 13);
    let valid =
        load_structured_profile_v14(&files.profile, &old::fingerprint(), &limits, load).unwrap();
    assert!(!valid.children.is_empty());
    assert!(
        load_structured_profile_v15(&files.profile, &old::fingerprint(), &limits, load).is_err()
    );
    let mut envelope: OwnerCatalogEnvelope =
        serde_json::from_slice(&std::fs::read(&files.profile).unwrap()).unwrap();
    envelope.schema_version = KIND.profile_schema();
    envelope.artifact_type = KIND.profile_artifact().into();
    std::fs::write(&files.profile, metadata_bytes(&envelope, &limits).unwrap()).unwrap();
    assert!(
        load_structured_profile_v15(&files.profile, &old::fingerprint(), &limits, load).is_err()
    );
    let destination = files.dir.join("cannot-export-as-15.json");
    assert!(export_structured_profile_v15(
        &files.source,
        hash,
        bytes.len() as u64,
        &destination,
        0,
        &limits
    )
    .is_err());
    assert!(!destination.exists());
    assert!(replay_structured_source_v8(&bytes, &limits).is_err());
}

#[test]
fn profile15_rejects_changed_prefix_or_unclosed_cohort_despite_matching_file_hashes() {
    let (bytes, _, checkpoint) = collected();
    let h = header(&bytes);
    let limits = CostProfileLoadLimits::default();
    let now = at(&h, checkpoint.population.closing.monotonic_ns + 100);
    let original = Files::new(&bytes);
    export(&original, &bytes, bytes.len() as u64);
    let valid: OwnerCatalogEnvelope =
        serde_json::from_slice(&std::fs::read(&original.profile).unwrap()).unwrap();
    // The outer journal/prefix hashes are recomputed for every mutation: the
    // original preparation/cohort replay must independently reject it.
    for remove in [false, true] {
        let mut bad = Vec::new();
        let mut changed = false;
        for (index, line) in bytes.split_inclusive(|b| *b == b'\n').enumerate() {
            if index == 0 {
                bad.extend_from_slice(line);
                continue;
            }
            let mut value: serde_json::Value = serde_json::from_slice(line).unwrap();
            if !changed && remove && value["kind"] == "cohort_end" {
                changed = true;
                continue;
            }
            if !changed && !remove && value["kind"] == "preparation_released" {
                value["receipt"]["generated_prefix_sha256"] =
                    serde_json::to_value([99u8; 32]).unwrap();
                changed = true;
            }
            let typed: StructuredPreparedOwnerBlockRecordV8 =
                serde_json::from_value(value).unwrap();
            bad.extend(record_bytes_v7(&typed).unwrap());
        }
        assert!(changed);
        assert!(replay_structured_source_v8(&bad, &limits).is_err());
        let files = Files::new(&bad);
        let hash: [u8; 32] = Sha256::digest(&bad).into();
        let mut envelope = valid.clone();
        envelope.source_path = Some(files.source.canonicalize().unwrap());
        envelope.source_bytes = bad.len() as u64;
        envelope.source_sha256 = hash;
        envelope.journal_bytes = Some(bad.len() as u64);
        envelope.journal_sha256 = Some(hash);
        std::fs::write(&files.profile, metadata_bytes(&envelope, &limits).unwrap()).unwrap();
        let failure = load_structured_profile_v15(
            &files.profile,
            &old::fingerprint(),
            &limits,
            load_at(now, 19),
        )
        .unwrap_err();
        assert!(!failure.to_string().contains("journal bytes/hash differ"));
        assert!(!failure
            .to_string()
            .contains("checkpoint prefix hash differs"));
        let destination = files.dir.join("cannot-export.json");
        assert!(export_structured_profile_v15(
            &files.source,
            hash,
            bad.len() as u64,
            &destination,
            0,
            &limits
        )
        .is_err());
        assert!(!destination.exists());
    }
    // A valid journal alone cannot certify an earlier incomplete cohort prefix.
    let cut = bytes
        .split_inclusive(|b| *b == b'\n')
        .scan(0usize, |offset, line| {
            *offset += line.len();
            Some((*offset, line))
        })
        .find_map(|(offset, line)| {
            let value: serde_json::Value = serde_json::from_slice(line).unwrap();
            (value["kind"] == "preparation_released").then_some(offset as u64)
        })
        .unwrap();
    let destination = original.dir.join("unclosed-prefix.json");
    assert!(export_structured_profile_v15(
        &original.source,
        Sha256::digest(&bytes).into(),
        cut,
        &destination,
        0,
        &limits
    )
    .is_err());
    assert!(!destination.exists());
}
