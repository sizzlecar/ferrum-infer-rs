//! Original canonical inputs and complete blocks exercise the opt-in policy.
//! Synthetic durations test the protocol, not backend latency.
use super::*;
use crate::implementations::continuous::cost_model::structured_v2::{
    DeclaredAlgorithmUniverseV1, OwnerPhaseSupportPolicyV1,
};

fn support_header() -> StructuredServiceHeaderV7 {
    let mut h = header();
    let q = query(&[B]);
    h.declaration
        .nonnegative_envelope
        .as_mut()
        .unwrap()
        .algorithm_universe = Some(
        DeclaredAlgorithmUniverseV1::from_inputs([q.input()], h.declaration.settings.max_axes)
            .unwrap(),
    );
    h.declaration.schedule.algorithm_universe =
        Some(OwnerAlgorithmUniversePolicyV1::SeededFirstOrdinaryDiscoveryBlockSubsetV1);
    h.declaration.schedule.phase_support =
        Some(OwnerPhaseSupportPolicyV1::FrozenInputIntersectionV1);
    rebuild(h).unwrap()
}

#[test]
fn source7_phase_support_outside_fit_keeps_all_offers_and_waits_for_real_members() {
    let h = support_header();
    let mut bytes = record_bytes_v7(&h).unwrap();
    let mut c = StructuredServiceCollectorV7::new(h, CostProfileLoadLimits::default()).unwrap();
    for n in 1..=6 {
        let r = block(
            &mut c,
            &mut bytes,
            n,
            |ticket| if n <= 2 { vec![A] } else { alternating(ticket) },
            1,
        );
        let StructuredServiceRecordV7::BlockClose { freezes, .. } = r else {
            unreachable!()
        };
        if n == 1 {
            assert!(freezes.is_empty());
        } else if n == 3 || n == 5 {
            assert!(
                freezes.is_empty(),
                "eight original offers contain only four members"
            );
            let audit = c.audit();
            assert_eq!(audit.owners[0].eligible, 4);
            assert_eq!(audit.owners[0].owner_offered, 8);
            assert!(audit.owners[0].failure.is_none());
        } else {
            assert_eq!(freezes.len(), 1);
            let frozen = &freezes[0];
            assert_eq!(frozen.failure, None, "{frozen:?}");
            assert_eq!(frozen.close.member_count, 8);
            let index = if n == 2 {
                0
            } else if n == 4 {
                1
            } else {
                2
            };
            assert_eq!(frozen.close.phase, phase_at(index));
            assert_eq!(frozen.domain.eligible, 8);
            assert_eq!(frozen.domain.owner_offered, if n == 2 { 8 } else { 16 });
            assert_eq!(
                frozen.domain.outside_fit_support,
                if n == 2 { 0 } else { 8 }
            );
            assert_eq!(frozen.domain.outside_residual_support, 0);
            assert_eq!(
                frozen.close.boundary.first_block,
                if n == 2 { 2 } else { n - 1 }
            );
            assert_eq!(frozen.close.boundary.last_block, n);
        }
    }
    replay_and_check(c, bytes, 6);
}

#[test]
fn source7_phase_support_qualification_uses_frozen_residual_subdomain() {
    let h = support_header();
    let mut bytes = record_bytes_v7(&h).unwrap();
    let mut c = StructuredServiceCollectorV7::new(h, CostProfileLoadLimits::default()).unwrap();
    for n in 1..=5 {
        let r = block(
            &mut c,
            &mut bytes,
            n,
            |ticket| if n == 3 { vec![A] } else { alternating(ticket) },
            1,
        );
        let StructuredServiceRecordV7::BlockClose { freezes, .. } = r else {
            unreachable!()
        };
        if n == 1 || n == 4 {
            assert!(freezes.is_empty());
            if n == 4 {
                assert_eq!(c.audit().owners[0].eligible, 4);
                assert_eq!(c.audit().owners[0].owner_offered, 8);
            }
        } else {
            assert_eq!(freezes.len(), 1);
            let frozen = &freezes[0];
            assert_eq!(frozen.failure, None, "{frozen:?}");
            assert_eq!(frozen.close.member_count, 8);
            assert_eq!(
                frozen.close.phase,
                phase_at(if n == 5 { 2 } else { (n - 2) as usize })
            );
            assert_eq!(frozen.domain.outside_fit_support, 0);
            assert_eq!(
                frozen.domain.outside_residual_support,
                if n == 5 { 8 } else { 0 }
            );
            assert_eq!(frozen.domain.owner_offered, if n == 5 { 16 } else { 8 });
        }
    }
    replay_and_check(c, bytes, 5);
}

fn replay_and_check(mut c: StructuredServiceCollectorV7, mut bytes: Vec<u8>, blocks: u64) {
    assert_eq!(c.offered(), blocks * 8);
    assert_eq!(c.qualified_children(), 1);
    let now = paired(blocks * 8 * 2_000 + 1_101);
    let (r, checkpoint) = c.checkpoint(now).unwrap();
    append(&mut bytes, &r);
    let limits = CostProfileLoadLimits::default();
    let original = checkpoint
        .activate_same_process_memory(now, &limits)
        .unwrap();
    let replay = replay_structured_source_v7(&bytes, &limits)
        .unwrap()
        .activate_same_process_memory(now, &limits)
        .unwrap();
    assert_eq!(original.source_sha256, replay.source_sha256);
    assert_eq!(original.total_shape_rows, blocks * 8);
    assert_eq!(original.children.len(), 1);
    assert_eq!(
        original.children[0].parameters_signature(),
        replay.children[0].parameters_signature()
    );
    for child in [&original.children[0], &replay.children[0]] {
        assert_eq!(
            child.provenance().phases.each_ref().map(|p| p.members),
            [8; 3]
        );
        assert_eq!(
            child
                .provenance()
                .phases
                .each_ref()
                .map(|p| p.member_cutoff),
            [8, 16, 24]
        );
        child
            .predict_query_local(&old::fingerprint(), &query(&[A]), now.monotonic_ns)
            .unwrap();
        assert!(matches!(
            child.predict_query_local(&old::fingerprint(), &query(&[B]), now.monotonic_ns),
            Err(StructuredUnknownV2::QualificationCoverage)
        ));
    }
    // Original replay must reject a false eligible denominator even if the
    // excluded records themselves were preserved without modification.
    let mut forged = Vec::new();
    let mut changed = false;
    for line in bytes.split_inclusive(|b| *b == b'\n') {
        let mut value: serde_json::Value = serde_json::from_slice(line).unwrap();
        if !changed {
            if let Some(domain) = value
                .get_mut("freezes")
                .and_then(|v| v.get_mut(0))
                .and_then(|v| v.get_mut("domain"))
            {
                let excluded = domain["outside_fit_support"].as_u64().unwrap()
                    + domain["outside_residual_support"].as_u64().unwrap();
                if excluded > 0 {
                    domain["eligible"] = serde_json::json!(domain["owner_offered"]);
                    changed = true;
                }
            }
        }
        forged.extend(record_bytes_v7(&value).unwrap());
    }
    assert!(changed);
    assert!(replay_structured_source_v7(&forged, &limits).is_err());
}

#[test]
fn source7_phase_support_requires_explicit_physical_contract_and_preserves_none_wire() {
    let old = header();
    let bytes = record_bytes_v7(&old).unwrap();
    assert!(!String::from_utf8(bytes.clone())
        .unwrap()
        .contains("phase_support"));
    let decoded: StructuredServiceHeaderV7 = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(record_bytes_v7(&rebuild(decoded).unwrap()).unwrap(), bytes);
    let mut physical = old.clone();
    physical.declaration.schedule.phase_support =
        Some(OwnerPhaseSupportPolicyV1::FrozenInputIntersectionV1);
    let physical = rebuild(physical).unwrap();
    assert_ne!(old.protocol, physical.protocol);
    assert_ne!(old.declaration_sha256, physical.declaration_sha256);
    let mut legacy = old;
    legacy.declaration.domain_policy = StructuredServiceDomainPolicyV1::FrozenFitSupportV1;
    legacy.declaration.nonnegative_envelope = None;
    // Rebuild the old schedule's derived member capacity; simply removing
    // input_readiness would leave its different maximum_phase_members bound.
    legacy.declaration.schedule = OwnerBlockScheduleV1::new(8, [8; 3], [8; 3]).unwrap();
    let mut legacy = rebuild(legacy).unwrap();
    legacy.declaration.schedule.phase_support =
        Some(OwnerPhaseSupportPolicyV1::FrozenInputIntersectionV1);
    assert!(rebuild(legacy).is_err());
}

#[test]
fn source7_phase_support_v2_sparse_discovery_algorithm_stays_unknown_and_replays() {
    let mut h = support_header();
    let v1_protocol = h.protocol;
    h.declaration.schedule.phase_support =
        Some(OwnerPhaseSupportPolicyV1::FrozenInputIntersectionV2);
    let h = rebuild(h).unwrap();
    assert_ne!(h.protocol, v1_protocol);
    let mut bytes = record_bytes_v7(&h).unwrap();
    let mut c = StructuredServiceCollectorV7::new(h, CostProfileLoadLimits::default()).unwrap();
    for n in 1..=6 {
        // Discovery contains both original algorithms. Fit contains only A.
        // Later whole blocks keep offering B; pre-result support excludes it.
        let r = block(
            &mut c,
            &mut bytes,
            n,
            |ticket| if n == 2 { vec![A] } else { alternating(ticket) },
            1,
        );
        let StructuredServiceRecordV7::BlockClose { freezes, .. } = r else {
            unreachable!()
        };
        if n == 1 || n == 3 || n == 5 {
            assert!(freezes.is_empty());
        } else {
            assert_eq!(freezes.len(), 1);
            let frozen = &freezes[0];
            assert_eq!(frozen.failure, None, "{frozen:?}");
            assert_eq!(frozen.close.phase, phase_at((n / 2 - 1) as usize));
            assert_eq!(frozen.close.member_count, 8);
            assert_eq!(frozen.domain.owner_offered, if n == 2 { 8 } else { 16 });
            assert_eq!(
                frozen.domain.outside_fit_support,
                if n == 2 { 0 } else { 8 }
            );
        }
    }
    // This checks source replay, equal parameters, exact disjoint member cuts,
    // A prediction, B Unknown, and rejection of a forged eligible denominator.
    replay_and_check(c, bytes, 6);
}

#[test]
fn source7_phase_support_v1_still_waits_for_discovery_algorithm_in_fit() {
    let h = support_header();
    let mut bytes = record_bytes_v7(&h).unwrap();
    let mut c = StructuredServiceCollectorV7::new(h, CostProfileLoadLimits::default()).unwrap();
    block(&mut c, &mut bytes, 1, alternating, 1);
    let r = block(&mut c, &mut bytes, 2, |_| vec![A], 1);
    let StructuredServiceRecordV7::BlockClose { freezes, .. } = r else {
        unreachable!()
    };
    assert!(freezes.is_empty());
    assert_eq!(c.qualified_children(), 0);
    assert_eq!(c.audit().owners[0].eligible, 8);
    assert!(c.audit().owners[0].failure.is_none());
}
