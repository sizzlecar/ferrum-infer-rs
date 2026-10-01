//! Runtime provenance follows completed publication, using real qualified
//! source3 children. A saved diagnostic value owns no worker lock or authority.
use super::*;

#[tokio::test]
async fn runtime_profile_receipt_follows_publication_and_preserves_owned_startup_value() {
    let mut f = Fixture::new();
    f.config.structured_feedback = SloStructuredFeedbackPolicy::Disabled;
    let runtime = f.build();
    let initial = runtime.profile_receipt().unwrap();
    assert_eq!(Some(&initial), runtime.training.receipt.as_ref());
    let now = f.clock.now_ns().unwrap();
    let children = runtime.training.live_catalog_children(now).unwrap();
    let version = runtime
        .training
        .publish_live_catalog(children, initial.clone(), now)
        .unwrap();

    let current = runtime.profile_receipt().unwrap();
    assert_eq!(current.model_version, version);
    assert!(current.model_version > initial.model_version);
    assert_eq!(Some(&initial), runtime.training.receipt.as_ref());
    assert_eq!(
        current.structured_whole_wave_v2,
        initial.structured_whole_wave_v2
    );
    assert_eq!(current.clock_basis, initial.clock_basis);
    assert_eq!(
        current.source_observation_artifact_sha256,
        initial.source_observation_artifact_sha256
    );

    let mut caller_copy = current.clone();
    caller_copy.source_observation_artifact_sha256[0] ^= 1;
    assert_ne!(caller_copy, current);
    assert_eq!(runtime.profile_receipt(), Some(current.clone()));

    let children = runtime.training.live_catalog_children(now).unwrap();
    let mut invalid = current.clone();
    invalid.structured_whole_wave_v2.as_mut().unwrap().children[0].source_sha256[0] ^= 1;
    assert!(runtime
        .training
        .publish_live_catalog(children, invalid, now)
        .is_err());
    assert_eq!(runtime.profile_receipt(), Some(current));
    f.unchanged();
    runtime.shutdown().await.unwrap();
}

#[tokio::test]
async fn runtime_profile_receipt_becomes_available_without_an_initial_import() {
    let mut f = Fixture::new();
    f.config.structured_feedback = SloStructuredFeedbackPolicy::Disabled;
    let imported = f.build();
    let now = f.clock.now_ns().unwrap();
    let children = imported.training.live_catalog_children(now).unwrap();
    let initial_source = imported.profile_receipt().unwrap();
    let runtime = EngineCostRuntime::build(identity(), f.clock.clone(), &f.config, false).unwrap();
    assert!(runtime.training.receipt.is_none());
    assert!(runtime.profile_receipt().is_none());

    let version = runtime
        .training
        .publish_live_catalog(children, initial_source.clone(), now)
        .unwrap();
    let current = runtime.profile_receipt().unwrap();
    assert_eq!(current.model_version, version);
    assert_eq!(
        current.structured_whole_wave_v2,
        initial_source.structured_whole_wave_v2
    );
    assert_eq!(current.clock_basis, initial_source.clock_basis);
    assert!(runtime.training.receipt.is_none());
    runtime.shutdown().await.unwrap();
    imported.shutdown().await.unwrap();
    f.unchanged();
}
