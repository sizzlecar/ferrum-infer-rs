//! Reuses the one original live source/profile; startup imports do not retrain.
use super::*;
use crate::continuous_engine::inner::cost_observation::EngineCostRuntime;
use ferrum_interfaces::execution_cost::ExecutorCostIdentityAvailability;
use ferrum_scheduler::implementations::continuous::cost_profile::{
    ImportedStructuredModelV2, ProfileFingerprint, StructuredProfileExportReceiptV10,
};
use ferrum_types::{SloCostObservationConfig, SloStructuredArtifactKindV2};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

#[path = "catalog/planner.rs"]
mod planner;

#[derive(Clone, Serialize)]
struct Catalog {
    artifact_type: &'static str,
    schema_version: u32,
    model_revision: &'static str,
    fingerprint: ProfileFingerprint,
    children: Vec<Child>,
}
#[derive(Clone, Serialize)]
struct Child {
    profile_path: PathBuf,
    profile_sha256: [u8; 32],
    owner: StructuredOwnerKeyV2,
    domain_signature: [u8; 32],
}
fn write_manifest(path: &Path, manifest: &Catalog) -> Vec<u8> {
    let bytes = serde_json::to_vec(manifest).unwrap();
    std::fs::write(path, &bytes).unwrap();
    bytes
}
pub(super) async fn check_real_child_catalog(
    session: &mut CalibrationSession,
    directory: &Path,
    exported: &StructuredProfileExportReceiptV10,
    imported: &ImportedStructuredModelV2,
    identity: &ExecutorCostIdentityAvailability,
    config: &SloCostObservationConfig,
    direct_runtime: &EngineCostRuntime,
    ttl: u64,
) {
    let direct = direct_runtime.profile_receipt().unwrap();
    let manifest = Catalog {
        artifact_type: "ferrum.structured-v2-catalog",
        schema_version: 1,
        model_revision: MODEL_REVISION_V2,
        fingerprint: ProfileFingerprint::from(imported.fingerprint()),
        children: vec![Child {
            // Exercise documented resolution against the catalog directory.
            profile_path: exported.path.file_name().unwrap().into(),
            profile_sha256: exported.file_sha256,
            owner: imported.owner().clone(),
            domain_signature: *imported.domain_signature(),
        }],
    };
    let path = directory.join("catalog.json");
    let bytes = write_manifest(&path, &manifest);
    let runtime = EngineCostRuntime::new(identity.clone(), config, Some(&path)).unwrap();
    let receipt = runtime.profile_receipt().unwrap();
    let old = direct.structured_whole_wave_v2.as_ref().unwrap();
    let catalog = receipt.structured_whole_wave_v2.as_ref().unwrap();
    assert_eq!(old.artifact_kind, SloStructuredArtifactKindV2::SingleChild);
    assert_eq!(
        catalog.artifact_kind,
        SloStructuredArtifactKindV2::CatalogV1
    );
    assert_eq!(receipt.schema_version, 1);
    assert_eq!(catalog.child_count, 1);
    assert!(receipt.selected_whole_wave.is_none());
    assert!(receipt.structured_whole_wave.is_none());
    assert_eq!(receipt.generated_unix_ns, direct.generated_unix_ns);
    assert_eq!(receipt.recorded_samples, direct.recorded_samples);
    assert_eq!(receipt.offered_samples, direct.offered_samples);
    assert_eq!(catalog.total_shape_rows, old.total_shape_rows);
    assert_eq!(
        catalog.total_imported_bytes,
        old.total_imported_bytes + bytes.len() as u64
    );
    let digest: [u8; 32] = Sha256::digest(&bytes).into();
    assert_eq!(
        receipt.file_sha256,
        format!(
            "sha256:{}",
            digest
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect::<String>()
        )
    );
    assert_eq!(
        receipt.source_observation_artifact_sha256,
        catalog.source_inventory_sha256.unwrap()
    );
    let before = &old.children[0];
    let after = &catalog.children[0];
    // Separate startup clocks legitimately map to different local anchors.
    // Every source identity, original phase cut and original clock stays fixed.
    assert_eq!(after.profile_sha256, before.profile_sha256);
    assert_eq!(after.source_sha256, before.source_sha256);
    // Despite its historical field name, source_monotonic_anchor_ns is
    // the NEW local runtime clock anchor. Recover the original observation
    // instants from each model epoch instead: import must not shift samples.
    assert_eq!(
        after
            .model_anchor_ns
            .checked_sub(after.oldest_imported_age_ns)
            .unwrap(),
        before
            .model_anchor_ns
            .checked_sub(before.oldest_imported_age_ns)
            .unwrap(),
    );
    assert_eq!(
        after
            .model_anchor_ns
            .checked_sub(after.newest_imported_age_ns)
            .unwrap(),
        before
            .model_anchor_ns
            .checked_sub(before.newest_imported_age_ns)
            .unwrap(),
    );
    assert_eq!(after.parameters_sha256, before.parameters_sha256);
    assert_eq!(after.owner_sha256, before.owner_sha256);
    assert_eq!(after.scope_sha256, before.scope_sha256);
    assert_eq!(after.domain_signature, before.domain_signature);
    assert_eq!(
        after.capture_identity_sha256,
        before.capture_identity_sha256
    );
    assert_eq!(after.protocol_sha256, before.protocol_sha256);
    assert_eq!(after.rule_signature, before.rule_signature);
    assert_eq!(after.cohort_manifest_sha256, before.cohort_manifest_sha256);
    assert_eq!(after.phases, before.phases);
    assert_eq!(after.reserved_members, before.reserved_members);
    assert_eq!(after.total_shape_rows, before.total_shape_rows);
    assert!(after.oldest_imported_age_ns >= before.oldest_imported_age_ns);
    assert!(after.newest_imported_age_ns >= before.newest_imported_age_ns);

    // Failures target declarations against this valid real child. No missing
    // file, fabricated source, text matching or fake private receipt is used.
    for mutation in 0..3 {
        let mut changed = manifest.clone();
        match mutation {
            0 => changed.children[0].profile_sha256[0] ^= 1,
            1 => changed.children[0].owner.rows += 1,
            _ => changed.children[0].domain_signature[0] ^= 1,
        }
        let bad = directory.join(format!("catalog-mismatch-{mutation}.json"));
        write_manifest(&bad, &changed);
        assert!(EngineCostRuntime::new(identity.clone(), config, Some(&bad)).is_err());
    }
    // Each global budget fails independently at one less than real consumption;
    // original outside FIFO rows count even though they never train the model.
    for dimension in 0..3 {
        let mut bounded = config.clone();
        match dimension {
            0 => {
                bounded.profile_import.max_file_bytes =
                    NonZeroUsize::new(usize::try_from(catalog.total_imported_bytes).unwrap() - 1)
                        .unwrap()
            }
            1 => {
                bounded.profile_import.max_samples =
                    NonZeroUsize::new(receipt.recorded_samples - 1).unwrap()
            }
            _ => {
                bounded.profile_import.max_total_shape_rows =
                    NonZeroUsize::new(usize::try_from(catalog.total_shape_rows).unwrap() - 1)
                        .unwrap()
            }
        }
        assert!(EngineCostRuntime::new(identity.clone(), &bounded, Some(&path)).is_err());
    }
    planner::compare_real_queries(session, direct_runtime, &runtime, ttl).await;
    tokio::time::timeout(Duration::from_secs(30), runtime.shutdown())
        .await
        .unwrap()
        .unwrap();
}
