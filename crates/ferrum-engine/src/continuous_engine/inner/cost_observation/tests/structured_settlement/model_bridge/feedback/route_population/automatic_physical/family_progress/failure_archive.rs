//! Failed source7 journals retain settled raw evidence after a missing ticket.
//! This uses the same private CPU submission and FIFO as ordinary run/serve.
use super::*;
use sha2::{Digest, Sha256};

struct ArchiveDirectory(PathBuf);
impl ArchiveDirectory {
    fn new() -> Self {
        Self(std::env::temp_dir().join(format!(
            "ferrum-owner-block-failed-archive-{}",
            uuid::Uuid::new_v4()
        )))
    }
    fn policy(&self) -> SloAutomaticCalibrationDiagnosticsV1 {
        SloAutomaticCalibrationDiagnosticsV1::Directory {
            directory: self.0.clone(),
            maximum_source_bytes: NonZeroU64::new(16 * 1024 * 1024).unwrap(),
            maximum_total_bytes: NonZeroU64::new(128 * 1024 * 1024).unwrap(),
            maximum_retained_generations: NonZeroUsize::new(4).unwrap(),
        }
    }
}
impl Drop for ArchiveDirectory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[tokio::test]
async fn automatic_family_progress_failed_directory_keeps_all_settled_tickets_after_gap() {
    let directory = ArchiveDirectory::new();
    let mut f = Families::new_with_diagnostics(directory.policy());
    f.block(A);
    f.assert_unpublished();
    f.runtime.consume_samples();
    f.record(A);
    f.record(A);
    let at = f.clock.now_ns().unwrap() + 100;
    f.clock.set(at);
    // Keep the third original offer alive while later real calls finish.
    // Dropping it earlier would close issuance and would not exercise retained
    // successful records after the actual original-ticket gap.
    let missing = f
        .runtime
        .reserve_live_ticket(Some(at))
        .expect("third original Fit-block ticket");
    for _ in 3..OFFERS {
        f.record(A);
    }
    let before = f.live().audit();
    assert_eq!(before.failed_generations, 0);
    assert_eq!(before.population.issued, OFFERS);
    assert_eq!(before.population.retired, OFFERS - 1);
    assert!(!before.population.failed);
    assert_eq!(f.recorded, (2 * OFFERS - 1) as u64);
    assert_eq!(f.runtime.audit_snapshot().sink.raw_lost, 0);
    drop(missing);
    f.runtime.consume_samples();

    let audit = f.live().audit();
    assert_eq!(audit.failed_generations, 1, "{audit:#?}");
    assert_eq!(audit.population.issued, OFFERS);
    assert_eq!(audit.population.retired, OFFERS);
    assert!(audit.population.failed);
    f.assert_unpublished();
    assert!(f.runtime.training.published_catalog_receipt().is_none());
    let failed = audit
        .last_failed_source
        .as_ref()
        .expect("failed original block remains auditable");
    assert!(failed.footer_complete, "{failed:#?}");
    assert!(!failed.retained_unpublished, "{failed:#?}");
    let path = failed
        .path
        .as_ref()
        .expect("published failed journal retains its actual receipt path");
    let bytes = fs::read(path).unwrap();
    assert_eq!(failed.bytes, bytes.len() as u64);
    let digest: [u8; 32] = Sha256::digest(&bytes).into();
    let digest_hex = format!("{:x}", Sha256::digest(&bytes));
    assert_eq!(failed.sha256.as_deref(), Some(digest_hex.as_str()));
    let records: Vec<serde_json::Value> = bytes
        .split(|byte| *byte == b'\n')
        .filter(|line| !line.is_empty())
        .map(|line| serde_json::from_slice(line).unwrap())
        .collect();
    assert_eq!(records[0]["schema_version"], 7);
    let settled: Vec<u64> = records
        .iter()
        .filter(|record| record["kind"] == "completed")
        .map(|record| record["wave"]["ticket"].as_u64().unwrap())
        .collect();
    let missing_global = OFFERS as u64 + 3;
    let expected: Vec<_> = (1..=2 * OFFERS as u64)
        .filter(|ticket| *ticket != missing_global)
        .collect();
    assert_eq!(
        settled, expected,
        "every settled original ticket is retained, including all later tickets after the gap"
    );
    let failures: Vec<_> = records
        .iter()
        .filter(|record| record["kind"] == "failed")
        .collect();
    assert_eq!(failures.len(), 1);
    assert_eq!(failures[0]["ticket"].as_u64(), Some(missing_global));
    assert_eq!(records.last().unwrap()["kind"], "footer");
    assert!(records.last().unwrap()["incomplete_block"]
        .as_bool()
        .unwrap());
    assert!(records.iter().all(|record| record["kind"] != "checkpoint"));
    assert_eq!(
        records
            .iter()
            .filter(|record| record["kind"] == "block_close")
            .count(),
        1,
        "only the independent Discovery block completed; the broken Fit block never freezes"
    );

    // The valid filesystem receipt authenticates failure evidence, never a
    // complete calibration prefix or an importable model.
    let limits = file::CostProfileLoadLimits::default();
    assert!(file::replay_structured_source_v7(&bytes, &limits).is_err());
    let output = directory.0.join("must-not-publish-profile14.json");
    assert!(file::export_structured_profile_v14(
        path,
        digest,
        bytes.len() as u64,
        &output,
        0,
        &limits,
    )
    .is_err());
    assert!(!output.exists());
    f.runtime.shutdown().await.unwrap();
    f.assert_unpublished();
    assert!(f.runtime.training.published_catalog_receipt().is_none());
}
