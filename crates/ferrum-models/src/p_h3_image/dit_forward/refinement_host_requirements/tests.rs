use super::*;
use crate::p_h3_image::dit_forward::tests::{config as fixture_config, limits};
fn account(alignment: usize) -> HostMemoryAccount {
    HostMemoryAccount::with_allocation_alignment(
        HostMemoryLimits::new(
            1 << 20,
            HostMemoryClassLimits {
                input_preprocess_bytes: 1 << 20,
                output_bytes: 1 << 20,
                transfer_staging_bytes: 0,
                weight_backing_bytes: 0,
            },
        )
        .unwrap(),
        alignment,
    )
    .unwrap()
}
fn prepared(c: &PH3ImageDiTConfig) -> (PreparedModelFamily, ProgramPlanCompileOptions) {
    let provider = PH3ImageDiTFamily::new(limits()).unwrap();
    let options = provider.compile_options(c).unwrap();
    let family = TypedFamilyRegistration::new(provider)
        .prepare_with_profile(
            &serde_json::to_value(c).unwrap(),
            &NumericalProfileId::new(PH3ImageDiTFamily::cuda_profile_id(c)).unwrap(),
        )
        .unwrap();
    (family, options)
}
fn ids(count: usize) -> Vec<RequestIdentity> {
    (0..count)
        .map(|i| RequestIdentity::new(format!("request.dit.{i}")).unwrap())
        .collect()
}
fn connector(c: &PH3ImageDiTConfig, host: &HostMemoryAccount) -> ManagedHostTensor {
    let spec = config::tensor(vec![c.text_rows().unwrap(), c.text_dim], ElementType::F32);
    let mut buffer = host
        .allocate_zeroed(
            HostMemoryClass::InputPreprocess,
            usize::try_from(spec.byte_len().unwrap()).unwrap(),
        )
        .unwrap();
    for bytes in buffer.as_mut().chunks_exact_mut(4) {
        bytes.copy_from_slice(&0.25_f32.to_le_bytes());
    }
    inputs::managed("value.connector.borrowed", spec, buffer.freeze()).unwrap()
}

#[test]
fn actual_pending_cache_destinations_and_work_backings_match_separate_aligned_grants() {
    let c = fixture_config(2).with_phase(PH3ImageDiTPhase::RefineOnly);
    let (family, options) = prepared(&c);
    let ids = ids(2);
    let host = account(4096);
    let requirement = query(&c, &ids, &family, &options, &host).unwrap();
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    let connector = connector(&c, &host);
    let old_charge = connector.bytes().backing_bytes();
    assert_eq!(connector.tensor(), &requirement.borrowed_conditioning);
    let keys = cache::fixture_keys(&c, &connector);
    let a = host
        .reserve_scope(&[requirement.persistent_cache_by_request[0].scope_claim()])
        .unwrap();
    let b = host
        .reserve_scope(&[requirement.persistent_cache_by_request[1].scope_claim()])
        .unwrap();
    let destinations = [
        PH3ImageRefinementDestination::new(&ids[0], &a),
        PH3ImageRefinementDestination::new(&ids[1], &b),
    ];
    // This is the real consumer's pending-cache allocation path. It creates
    // the persistent payloads before submission, not after a readback exists.
    let pending = cache::PreparedRefinedTextCaches::new(
        &c,
        &host,
        keys,
        &destinations,
        limits().max_metadata_bytes,
    )
    .unwrap();
    assert_eq!(
        host.usage().unwrap().allocated_bytes(),
        old_charge + requirement.persistent_cache_bytes
    );
    assert!(
        requirement.persistent_cache_bytes > requirement.work_output_bytes,
        "independent request backing padding cannot be replaced by one batched backing"
    );
    let work = host
        .reserve_scope(&requirement.work_scope_claims())
        .unwrap();
    let input = work
        .account()
        .allocate_zeroed(
            HostMemoryClass::InputPreprocess,
            requirement.work_input.minimum_storage_bytes().unwrap() as usize,
        )
        .unwrap()
        .freeze();
    let output = work
        .account()
        .allocate_zeroed(
            HostMemoryClass::Output,
            requirement.work_output.minimum_storage_bytes().unwrap() as usize,
        )
        .unwrap()
        .freeze();
    assert_eq!(input.backing_bytes(), requirement.work_input_bytes);
    assert_eq!(output.backing_bytes(), requirement.work_output_bytes);
    assert_eq!(
        host.usage().unwrap().allocated_bytes(),
        old_charge + requirement.new_live_peak_bytes
    );
    assert_eq!(
        host.usage()
            .unwrap()
            .class_allocated_bytes(HostMemoryClass::InputPreprocess),
        old_charge + requirement.work_input_bytes + requirement.persistent_cache_bytes
    );
    // The borrowed F32 connector's real charge is additional caller state,
    // not an allocation of the new work scope or a second persistent grant.
    drop(pending);
    assert_eq!(
        host.usage().unwrap().allocated_bytes(),
        old_charge + requirement.work_scope_bytes
    );
    drop((input, output));
    work.close().unwrap();
    a.close().unwrap();
    b.close().unwrap();
    assert_eq!(host.usage().unwrap().committed_bytes(), old_charge);
    drop(connector);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn actual_late_cache_scope_failure_refunds_earlier_request_before_work_is_admitted() {
    let c = fixture_config(2).with_phase(PH3ImageDiTPhase::RefineOnly);
    let (family, options) = prepared(&c);
    let ids = ids(2);
    let host = account(4096);
    let report = query(&c, &ids, &family, &options, &host).unwrap();
    let connector = connector(&c, &host);
    let keys = cache::fixture_keys(&c, &connector);
    let a = host
        .reserve_scope(&[report.persistent_cache_by_request[0].scope_claim()])
        .unwrap();
    let b = host
        .reserve_scope(&[(
            HostMemoryClass::InputPreprocess,
            report.persistent_cache_by_request[1].input_bytes - 1,
        )])
        .unwrap();
    let destinations = [
        PH3ImageRefinementDestination::new(&ids[0], &a),
        PH3ImageRefinementDestination::new(&ids[1], &b),
    ];
    let error = cache::PreparedRefinedTextCaches::new(
        &c,
        &host,
        keys,
        &destinations,
        limits().max_metadata_bytes,
    )
    .err()
    .unwrap();
    assert!(error.chain().any(|e| matches!(
        e.downcast_ref::<HostMemoryError>(),
        Some(HostMemoryError::ScopeCapacity {
            class: HostMemoryClass::InputPreprocess,
            ..
        })
    )));
    assert_eq!(
        host.usage().unwrap().allocated_bytes(),
        connector.bytes().backing_bytes()
    );
    // The first destination is reusable after the later request rejects;
    // spacious parent capacity cannot bypass the second scoped limit.
    drop(
        a.account()
            .allocate_zeroed(
                HostMemoryClass::InputPreprocess,
                report.persistent_cache_by_request[0]
                    .tensor
                    .minimum_storage_bytes()
                    .unwrap() as usize,
            )
            .unwrap(),
    );
    a.close().unwrap();
    b.close().unwrap();
    drop(connector);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn wrong_phase_mismatched_actual_output_and_padding_sum_overflow_are_rejected() {
    let c = fixture_config(2).with_phase(PH3ImageDiTPhase::RefineOnly);
    let (family, options) = prepared(&c);
    let ids = ids(2);
    let host = account(1);
    let mut specs = options.tensor_specs().clone();
    specs
        .get_mut(&ProgramValueId::new(PH3ImageDiTFamily::REFINED_TEXT_ID).unwrap())
        .unwrap()
        .dimensions[0] += 1;
    let wrong_options = ProgramPlanCompileOptions::new(specs).unwrap();
    assert!(query(&c, &ids, &family, &wrong_options, &host).is_err());
    let wrong = c.clone().with_phase(PH3ImageDiTPhase::RefinedText);
    assert!(query(&wrong, &ids, &family, &options, &host).is_err());
    assert!(query(
        &c,
        &[ids[0].clone(), ids[0].clone()],
        &family,
        &options,
        &host
    )
    .is_err());
    let huge_padding = account(1usize << (usize::BITS - 2));
    assert!(query(&c, &ids, &family, &options, &huge_padding).is_err());
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    assert_eq!(huge_padding.usage().unwrap().committed_bytes(), 0);
}
