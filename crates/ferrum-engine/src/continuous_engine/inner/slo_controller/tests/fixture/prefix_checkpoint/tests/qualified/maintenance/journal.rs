//! Read original diagnostic bytes; never manufacture source or live receipts.
use super::*;
use crate::continuous_engine::inner::cost_observation::{
    ProspectiveCapture, ProspectiveCaptureOutcomeV1,
};
use ferrum_scheduler::implementations::continuous::cost_profile::{
    self as file, ImportedStructuredModelV2, StructuredPreparedOwnerBlockCollectorV8 as Collector,
    StructuredPreparedOwnerBlockHeaderV8 as Header, StructuredPreparedOwnerBlockRecordV8 as Record,
    StructuredServiceRecordV7,
};
use std::{fs, path::PathBuf};

pub(super) struct Directory(PathBuf);
impl Directory {
    pub(super) fn new() -> Self {
        let path =
            std::env::temp_dir().join(format!("ferrum-maintained-source-{}", RequestId::new()));
        fs::create_dir(&path).unwrap();
        Self(path)
    }

    pub(super) fn policy(&self) -> ferrum_types::SloAutomaticCalibrationDiagnosticsV1 {
        ferrum_types::SloAutomaticCalibrationDiagnosticsV1::Directory {
            directory: self.0.clone(),
            maximum_source_bytes: NonZeroU64::new(64 * 1024 * 1024).unwrap(),
            maximum_total_bytes: NonZeroU64::new(512 * 1024 * 1024).unwrap(),
            maximum_retained_generations: NonZeroUsize::new(16).unwrap(),
        }
    }

    pub(super) fn completed_source(
        &self,
        captured: &vnext::NativeCheckpointTransferIdentity,
    ) -> CompletedSource {
        let mut matched = None;
        let limits = file::CostProfileLoadLimits::default();
        let expected = serde_json::json!({
            "slot": captured.slot_id().get(),
            "checkpoint_coordinator": captured.checkpoint_authority().coordinator_id().get(),
            "checkpoint_serial": captured.checkpoint_authority().serial(),
            "sequence_sparse": captured.sequence_authority().sparse_id(),
            "sequence_generation": captured.sequence_authority().generation(),
            "request_sparse": captured.request_authority().sparse_id(),
            "request_generation": captured.request_authority().generation(),
            "boundary_tokens": captured.boundary_tokens(),
            "kind": "capture",
            "plan_hash": captured.plan_hash().as_str(),
            "layout_fingerprint": captured.layout_fingerprint(),
            "runtime_implementation_fingerprint": captured.runtime_implementation_fingerprint(),
            "device_id": captured.device_id().as_str(),
        });
        for entry in fs::read_dir(self.0.join("ferrum-automatic-v1")).unwrap() {
            let path = entry.unwrap().path().join("source.jsonl");
            if !path.is_file() {
                continue;
            }
            let bytes = fs::read(&path).unwrap();
            let mut lines = bytes.split_inclusive(|b| *b == b'\n');
            let first = lines.next().unwrap();
            // The same owned store also contains reference/source7 diagnostics.
            let Ok(header) = serde_json::from_slice::<Header>(first) else {
                continue;
            };
            let capture_identity = header.capture_identity;
            let mut collector = Collector::new(header, limits.clone()).unwrap();
            let mut restored_seed = false;
            let mut consumed = first.len();
            let mut checkpoint_end = None;
            for line in lines {
                let original: Record = serde_json::from_slice(line).unwrap();
                collector.push(&original).unwrap();
                consumed += line.len();
                if matches!(original, Record::Preparation(_)) {
                    // Public event is opaque. Inspect only its original typed
                    // serialization after the full collector validated it.
                    let wire = serde_json::to_value(&original).unwrap();
                    if wire["kind"] == "native_prefix_restored" && wire["capture"] == expected {
                        restored_seed = true;
                    }
                }
                if matches!(
                    original,
                    Record::Population(StructuredServiceRecordV7::Checkpoint { .. })
                ) {
                    checkpoint_end = Some(consumed);
                }
            }
            if !restored_seed {
                continue;
            }
            assert!(
                collector.audit().closed,
                "maintained source has no original footer: {path:?}"
            );
            let checkpoint_end =
                checkpoint_end.expect("maintained source has no complete checkpoint");
            // Requires the complete declared lifecycle, original F/R/Q and
            // final checkpoint; a partially published/skipped source fails.
            let replayed = file::replay_structured_source_v8(&bytes[..checkpoint_end], &limits)
                .unwrap_or_else(|error| {
                    panic!("maintained source did not complete: {path:?}: {error}")
                });
            assert!(replayed.qualified_children() > 0);
            assert!(
                matched
                    .replace(CompletedSource {
                        capture_identity,
                        source_sha256: replayed.source_receipt().1,
                    })
                    .is_none(),
                "one private checkpoint crossed distinct source journals"
            );
        }
        matched.expect("no completed original source restored the actual deferred seed checkpoint")
    }
}

impl Drop for Directory {
    fn drop(&mut self) {
        if std::thread::panicking() {
            eprintln!(
                "maintained-source failure journals retained at {:?}",
                self.0
            );
        } else {
            let _ = fs::remove_dir_all(&self.0);
        }
    }
}

pub(super) struct CompletedSource {
    pub(super) capture_identity: [u8; 32],
    pub(super) source_sha256: [u8; 32],
}
impl CompletedSource {
    pub(super) fn matches_adopted(
        &self,
        capture: &ProspectiveCapture,
        children: &[ImportedStructuredModelV2],
        epoch: u64,
    ) -> bool {
        let receipt = capture
            .settled_receipt_for_test()
            .expect("normal worker did not produce the original issued-source receipt");
        let identity = serde_json::to_value(receipt).unwrap();
        // Matched is produced only after original private host settlement,
        // actual canonical/recipe/participant equality and once-only finish.
        // This read cannot re-run any of those producers or change the outcome.
        assert_eq!(receipt.outcome(), ProspectiveCaptureOutcomeV1::Matched);
        assert_eq!(identity["model_epoch"], epoch);
        if identity["source"]["source_sha256"] != serde_json::json!(self.source_sha256)
            || identity["source"]["capture_identity"] != serde_json::json!(self.capture_identity)
        {
            return false;
        }
        let matching: Vec<_> = children
            .iter()
            .filter(|child| {
                child.provenance().source_sha256 == self.source_sha256
                    && child.provenance().capture_identity == self.capture_identity
                    && identity["source"]["domain"] == serde_json::json!(child.domain_signature())
                    && identity["source"]["parameters_sha256"]
                        == serde_json::json!(child.parameters_signature())
            })
            .collect();
        assert_eq!(
            matching.len(),
            1,
            "issued identity must identify one actually installed child"
        );
        true
    }
}
