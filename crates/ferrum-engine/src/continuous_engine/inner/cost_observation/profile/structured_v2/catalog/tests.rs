use super::*;
use model::structured_v2::{StructuredProductV2, StructuredTemplateV2, StructuredWaveRoleV2};
fn fp() -> model::ExecutionFingerprint {
    model::ExecutionFingerprint {
        model_weights: [1; 32],
        numerical_policy: [2; 32],
        device_runtime: [3; 32],
        execution_config: [4; 32],
    }
}
fn child() -> Child {
    Child {
        profile_path: "child.json".into(),
        profile_sha256: [8; 32],
        domain_signature: [9; 32],
        owner: StructuredOwnerKeyV2 {
            rows: 1,
            role: StructuredWaveRoleV2::OrdinaryDecode,
            product: StructuredProductV2::GreedyToken,
            readback: ferrum_interfaces::execution_cost::CoreReadbackRoute::HostSynchronized,
            provider_template: StructuredTemplateV2::Ordered([5; 32]),
            algorithm_domain: [6; 32],
            installed_policy: [7; 32],
        },
    }
}
fn manifest() -> Manifest {
    Manifest {
        artifact_type: ARTIFACT_TYPE.into(),
        schema_version: 1,
        model_revision: MODEL_REVISION_V2.into(),
        fingerprint: file::ProfileFingerprint::from(&fp()),
        children: vec![child()],
    }
}
#[test]
fn catalog_owner_and_domain_are_unique_before_any_child_io() {
    let limits = file::CostProfileLoadLimits::default();
    let good = manifest();
    good.validate(&fp(), &limits).unwrap();
    let mut same_owner = manifest();
    let mut changed = child();
    changed.domain_signature = [10; 32];
    changed.profile_path = "other-window.json".into();
    same_owner.children.push(changed);
    assert!(same_owner.validate(&fp(), &limits).is_err());
    let mut same_domain = manifest();
    let mut changed = child();
    changed.owner.rows = 2;
    same_domain.children.push(changed);
    assert!(same_domain.validate(&fp(), &limits).is_err());
    let mut wrong = fp();
    wrong.numerical_policy = [99; 32];
    assert!(good.validate(&wrong, &limits).is_err());
    let mut empty = manifest();
    empty.children.clear();
    assert!(empty.validate(&fp(), &limits).is_err());
}
#[test]
fn catalog_total_budget_counts_manifest_sources_all_samples_and_outside_rows() {
    let limits = file::CostProfileLoadLimits {
        max_file_bytes: NonZeroUsize::new(100).unwrap(),
        max_samples: NonZeroUsize::new(30).unwrap(),
        max_total_shape_rows: NonZeroUsize::new(80).unwrap(),
        ..Default::default()
    };
    let mut budget = Budget::new(&limits, 10).unwrap();
    assert_eq!(budget.limits(&limits).unwrap().max_file_bytes.get(), 90);
    budget.consume(50, 24, 72).unwrap();
    let remaining = budget.limits(&limits).unwrap();
    assert_eq!(
        (
            remaining.max_file_bytes.get(),
            remaining.max_samples.get(),
            remaining.max_total_shape_rows.get()
        ),
        (40, 6, 8)
    );
    for (bytes, samples, rows) in [(41, 1, 1), (1, 7, 1), (1, 1, 9)] {
        assert!(budget.consume(bytes, samples, rows).is_err());
    }
    budget.consume(40, 6, 8).unwrap();
    assert!(budget.limits(&limits).is_err()); // never manufacture NonZero(1)
    assert!(Budget::new(&limits, 101).is_err());
}
#[test]
fn catalog_children_deserializer_and_metadata_are_bounded() {
    let entry = serde_json::json!({"profile_path":"x","profile_sha256":vec![1u8;32],"domain_signature":vec![2u8;32],"owner":child().owner});
    let value = serde_json::json!({"artifact_type":ARTIFACT_TYPE,"schema_version":1,"model_revision":MODEL_REVISION_V2,"fingerprint":file::ProfileFingerprint::from(&fp()),"children":vec![entry;MAX_CHILDREN+1]});
    assert!(serde_json::from_value::<Manifest>(value).is_err());
    let mut m = manifest();
    m.children[0].profile_path = "bad\npath".into();
    assert!(m
        .validate(&fp(), &file::CostProfileLoadLimits::default())
        .is_err());
}

fn wire_manifest() -> serde_json::Value {
    let c = child();
    serde_json::json!({
        "artifact_type": ARTIFACT_TYPE,
        "schema_version": 1,
        "model_revision": MODEL_REVISION_V2,
        "fingerprint": file::ProfileFingerprint::from(&fp()),
        "children": [{
            "profile_path": c.profile_path,
            "profile_sha256": c.profile_sha256,
            "owner": c.owner,
            "domain_signature": c.domain_signature,
        }],
    })
}

#[test]
fn catalog_rejects_unknown_fields_at_every_identity_boundary() {
    let original = wire_manifest();
    serde_json::from_value::<Manifest>(original.clone()).unwrap();
    for pointer in ["", "/fingerprint", "/children/0", "/children/0/owner"] {
        let mut changed = original.clone();
        changed
            .pointer_mut(pointer)
            .unwrap()
            .as_object_mut()
            .unwrap()
            .insert("undeclared_override".into(), serde_json::json!(true));
        assert!(serde_json::from_value::<Manifest>(changed).is_err());
    }
    let serialized = serde_json::to_string(&original).unwrap();
    let duplicate = serialized.replacen(
        "\"schema_version\":1",
        "\"schema_version\":1,\"schema_version\":1",
        1,
    );
    assert_ne!(duplicate, serialized);
    assert!(serde_json::from_str::<Manifest>(&duplicate).is_err());
}

struct TestDirectory(PathBuf);
impl TestDirectory {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "ferrum-structured-catalog-{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }
}
impl Drop for TestDirectory {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn catalog_metadata_checks_real_file_size_before_allocating_or_parsing() {
    let dir = TestDirectory::new();
    let path = dir.0.join("metadata.json");
    let limits = file::CostProfileLoadLimits::default();
    assert!(read_metadata(&dir.0, &limits).is_err());
    let file = File::create(&path).unwrap();
    assert!(read_metadata(&path, &limits).is_err());
    // Sparse file exercises the pre-allocation bound without a large fixture.
    file.set_len(MAX_METADATA_BYTES as u64 + 1).unwrap();
    assert!(read_metadata(&path, &limits).is_err());
    std::fs::write(&path, b"12345678").unwrap();
    let small = file::CostProfileLoadLimits {
        max_file_bytes: NonZeroUsize::new(7).unwrap(),
        ..limits.clone()
    };
    assert!(read_metadata(&path, &small).is_err());
    assert_eq!(read_metadata(&path, &limits).unwrap(), b"12345678");
}

#[cfg(unix)]
#[test]
fn catalog_symlink_paths_are_locators_and_the_opened_bytes_are_authoritative() {
    let dir = TestDirectory::new();
    let first = dir.0.join("first.json");
    let second = dir.0.join("second.json");
    let link = dir.0.join("child.json");
    std::fs::write(&first, b"first!").unwrap();
    std::fs::write(&second, b"second").unwrap();
    std::os::unix::fs::symlink(&first, &link).unwrap();
    let limits = file::CostProfileLoadLimits::default();
    let before = read_metadata(&link, &limits).unwrap();
    std::fs::remove_file(&link).unwrap();
    std::os::unix::fs::symlink(&second, &link).unwrap();
    let after = read_metadata(&link, &limits).unwrap();
    assert_eq!(before, b"first!");
    assert_eq!(after, b"second");
    assert_ne!(Sha256::digest(before), Sha256::digest(after));
}

#[test]
fn structured_v2_fit_floor_catalog_rejects_legacy_revision_before_child_io() {
    let mut value = wire_manifest();
    let limits = file::CostProfileLoadLimits::default();
    serde_json::from_value::<Manifest>(value.clone())
        .unwrap()
        .validate(&fp(), &limits)
        .unwrap();
    value["model_revision"] = "structured_whole_wave_pending_envelope_v2".into();
    let legacy = serde_json::from_value::<Manifest>(value).unwrap();
    // No child file is present: rejection belongs to algorithm identity, before IO.
    assert!(legacy.validate(&fp(), &limits).is_err());
}
