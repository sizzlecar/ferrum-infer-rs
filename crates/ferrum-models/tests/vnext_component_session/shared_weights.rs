//! One real installed matrix shared across different exact-shape plans.
use super::*;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

#[path = "shared_weights/release_failure.rs"]
mod release_failure;

fn closed(session: TensorComponentSession<ReferenceDeviceRuntime>) -> PlanRuntimeCloseReceipt {
    match session.close() {
        Ok(PlanRuntimeCloseOutcome::Closed(receipt)) => receipt,
        Ok(_) => panic!("completed session remains referenced"),
        Err(error) => panic!("close failed: {:?}", error.failure()),
    }
}

fn close_weights(
    weights: SharedComponentWeights<ReferenceDeviceRuntime>,
) -> SharedComponentWeightsCloseReceipt {
    match weights.close() {
        Ok(SharedComponentWeightsCloseOutcome::Closed(receipt)) => receipt,
        Ok(_) => panic!("all weight leases must have retired"),
        Err(error) => panic!("weight release failed: {:?}", error.failure()),
    }
}

#[test]
fn shared_weights_survive_changed_plan_without_reupload_or_second_claim() {
    let fixture = Fixture::new("device.component.shared-shape");
    let full = fixture.compile_rows(2, &fixture.family);
    let small = fixture.compile_rows(1, &fixture.family);
    assert_ne!(
        full.executable().execution_plan().plan_hash(),
        small.executable().execution_plan().plan_hash()
    );
    let (mut first, weights) = full
        .initialize_shared_weights(
            fixture.composition.runtime().clone(),
            fixture.host.clone(),
            ComponentWeightSource::Stored(&fixture.source),
            Fixture::init_config("shared-first"),
        )
        .unwrap();
    let first_output = execute(&mut first, &fixture, "shared-first", 1.);
    let static_bytes = weights.evidence().physical_bytes();
    let snapshot = weights.evidence().clone();
    assert_eq!(closed(first).released_static_resources(), 0);
    assert_eq!(fixture.composition.runtime().snapshot().live_allocations, 1);
    let after_cold = fixture.composition.runtime().snapshot();
    let mut next = small
        .initialize_with_shared_weights(
            fixture.composition.runtime().clone(),
            fixture.host.clone(),
            &weights,
            Fixture::init_config("shared-small"),
        )
        .unwrap();
    assert_eq!(
        fixture.composition.runtime().snapshot().uploaded_bytes,
        after_cold.uploaded_bytes
    );
    assert_eq!(
        fixture.composition.runtime().snapshot().live_allocations,
        1,
        "attach creates no physical buffer"
    );
    // Independent budget claim proves exactly one physical matrix remains.
    drop(
        fixture
            .capacity
            .reserve(
                DeviceOpaqueMemoryKind::Bootstrap,
                DEVICE_BYTES - static_bytes,
            )
            .unwrap(),
    );
    assert!(fixture
        .capacity
        .reserve(
            DeviceOpaqueMemoryKind::Bootstrap,
            DEVICE_BYTES - static_bytes + 1
        )
        .is_err());
    let ComponentInitializationReceipt::SharedWeights { evidence } = next.initialization_receipt()
    else {
        panic!("expected snapshot receipt")
    };
    assert_eq!(evidence, &snapshot);
    let source = fixture.input(1.);
    let input = ManagedHostTensor::new(
        id(IMAGE_PROJECTION_INPUT_ID),
        ResolvedTensorSpec::new(
            vec![1, 3],
            ElementType::Bf16,
            ResolvedTensorLayout::Contiguous,
        )
        .unwrap(),
        source.bytes().slice(0..6).unwrap(),
    )
    .unwrap();
    drop(source);
    let second_output = next
        .execute(
            id("run.shared.small"),
            id("request.shared.small"),
            vec![input],
        )
        .unwrap();
    assert_eq!(second_output[0].tensor().dimensions(), &[1, 6]);
    assert_eq!(second_output[0].bytes().bytes(), bytes(&EXPECTED[..6]));
    assert_eq!(
        fixture.composition.runtime().snapshot().uploaded_bytes - after_cold.uploaded_bytes,
        6
    );
    let close = closed(next);
    assert_eq!(
        close.evidence().plan_hash(),
        small.executable().execution_plan().plan_hash()
    );
    assert_eq!(close.evidence().shared_component_weights(), Some(&snapshot));
    assert_eq!(
        close.evidence().static_provisioning_identity(),
        Some(snapshot.source_identity())
    );
    assert_eq!(close_weights(weights).released_static_resources(), 1);
    fixture.assert_device_reusable();
    assert_eq!(first_output[0].bytes().bytes(), bytes(&EXPECTED));
    drop((first_output, second_output));
    assert_eq!(fixture.host.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn shared_weights_reject_different_runtime_and_weight_contract_before_allocation() {
    let fixture = Fixture::new("device.component.shared-reject");
    let compiled = fixture.compile_rows(2, &fixture.family);
    let (session, weights) = compiled
        .initialize_shared_weights(
            fixture.composition.runtime().clone(),
            fixture.host.clone(),
            ComponentWeightSource::Stored(&fixture.source),
            Fixture::init_config("reject"),
        )
        .unwrap();
    closed(session);
    // Even identical device metadata is not the actual context that owns storage.
    let other = ReferenceVNextComposition::create_with_capacity(
        fixture.composition.runtime().descriptor().id.clone(),
        DEVICE_BYTES,
    )
    .unwrap();
    assert_eq!(
        other
            .runtime()
            .descriptor()
            .runtime_implementation_fingerprint,
        fixture
            .composition
            .runtime()
            .descriptor()
            .runtime_implementation_fingerprint
    );
    assert!(weights
        .validate_for(compiled.executable().execution_plan(), other.runtime())
        .is_err());
    let mut config = fixture.family.canonical_config().clone();
    config["in_channels"] = json!(4);
    let different = TypedFamilyRegistration::new(QwenImage21ImageProjectionFamily::new().unwrap())
        .prepare_with_profile(
            &config,
            &NumericalProfileId::new(IMAGE_PROJECTION_PROFILE_ID).unwrap(),
        )
        .unwrap();
    let different = fixture.compile_rows(1, &different);
    let before = fixture.composition.runtime().snapshot();
    assert!(different
        .initialize_with_shared_weights(
            fixture.composition.runtime().clone(),
            fixture.host.clone(),
            &weights,
            Fixture::init_config("wrong-weights")
        )
        .is_err());
    assert_eq!(
        fixture.composition.runtime().snapshot().allocations,
        before.allocations
    );
    assert_eq!(
        fixture.composition.runtime().snapshot().uploaded_bytes,
        before.uploaded_bytes
    );
    close_weights(weights);
    fixture.assert_device_reusable();
}

#[test]
fn snapshot_close_retains_live_plan_and_later_last_session_releases_once() {
    let fixture = Fixture::new("device.component.shared-close");
    let compiled = fixture.compile_rows(2, &fixture.family);
    let (mut session, weights) = compiled
        .initialize_shared_weights(
            fixture.composition.runtime().clone(),
            fixture.host.clone(),
            ComponentWeightSource::Stored(&fixture.source),
            Fixture::init_config("close"),
        )
        .unwrap();
    let weights = match weights.close() {
        Ok(SharedComponentWeightsCloseOutcome::Referenced { weights }) => weights,
        Ok(_) => panic!("live session must retain weights"),
        Err(error) => panic!("unexpected release: {:?}", error.failure()),
    };
    assert!(weights
        .validate_for(
            compiled.executable().execution_plan(),
            fixture.composition.runtime()
        )
        .is_err());
    // Closing the cache does not revoke a previously admitted plan's reads.
    let output = execute(&mut session, &fixture, "closing-cache", 1.);
    drop(weights);
    assert_eq!(closed(session).released_static_resources(), 1);
    fixture.assert_device_reusable();
    assert_eq!(output[0].bytes().bytes(), bytes(&EXPECTED));
    drop(output);
    assert_eq!(fixture.host.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn active_root_authority_keeps_shared_weights_charged_during_close() {
    let fixture = Fixture::new("device.component.shared-authority");
    let compiled = fixture.compile_rows(2, &fixture.family);
    let (session, weights) = compiled
        .initialize_shared_weights(
            fixture.composition.runtime().clone(),
            fixture.host.clone(),
            ComponentWeightSource::Stored(&fixture.source),
            Fixture::init_config("authority"),
        )
        .unwrap();
    closed(session);
    let root = weights
        .bind_plan(
            compiled.executable().execution_plan(),
            fixture.composition.runtime().clone(),
            id("request.shared.authority"),
        )
        .unwrap();
    let binding = root.trusted_runtime_binding().unwrap();
    drop(weights);
    let root = match PlanRuntimeResources::close(root) {
        Ok(PlanRuntimeCloseOutcome::Referenced { resources, .. }) => resources,
        _ => panic!("durable execution authority must prevent final retirement"),
    };
    assert!(fixture
        .capacity
        .reserve(DeviceOpaqueMemoryKind::Bootstrap, DEVICE_BYTES)
        .is_err());
    drop(binding);
    match PlanRuntimeResources::close(root) {
        Ok(PlanRuntimeCloseOutcome::Closed(receipt)) => {
            assert_eq!(receipt.released_static_resources(), 1)
        }
        _ => panic!("last root must now release the one installation"),
    }
    fixture.assert_device_reusable();
}
