use super::*;
use crate::implementations::continuous::cost_model::structured_v2::DeclaredAlgorithmUniverseV1;

fn seeded_header() -> StructuredServiceHeaderV7 {
    let mut h = header();
    let b = query(&[B]);
    let seed =
        DeclaredAlgorithmUniverseV1::from_inputs([b.input()], h.declaration.settings.max_axes)
            .unwrap();
    h.declaration.schedule.algorithm_universe =
        Some(OwnerAlgorithmUniversePolicyV1::SeededFirstOrdinaryDiscoveryBlockSubsetV1);
    h.declaration
        .nonnegative_envelope
        .as_mut()
        .unwrap()
        .algorithm_universe = Some(seed);
    rebuild(h).unwrap()
}

#[test]
fn source7_cold_seed_unions_first_discovery_and_keeps_later_unknown_original_offers() {
    let h = seeded_header();
    let original_seed = h
        .declaration
        .nonnegative_envelope
        .as_ref()
        .unwrap()
        .algorithm_universe
        .as_ref()
        .unwrap();
    assert!(!original_seed
        .contains_checked_algorithms(query(&[A]).input())
        .unwrap());
    let mut bytes = record_bytes_v7(&h).unwrap();
    let mut collector =
        StructuredServiceCollectorV7::new(h.clone(), CostProfileLoadLimits::default()).unwrap();
    let first = block(&mut collector, &mut bytes, 1, |_| vec![A], 1);
    let StructuredServiceRecordV7::BlockClose {
        discoveries,
        freezes,
        ..
    } = first
    else {
        unreachable!()
    };
    assert!(
        freezes.is_empty(),
        "declaration and discovery are not Fit samples"
    );
    assert_eq!(discoveries.len(), 1);
    let union = discoveries[0]
        .contract
        .nonnegative_envelope
        .as_ref()
        .unwrap()
        .algorithm_universe
        .as_ref()
        .unwrap();
    assert!(union
        .contains_checked_algorithms(query(&[A]).input())
        .unwrap());
    assert!(union
        .contains_checked_algorithms(query(&[B]).input())
        .unwrap());
    assert!(!union
        .contains_checked_algorithms(query(&[C]).input())
        .unwrap());
    assert_ne!(union.signature(), original_seed.signature());

    // B was absent from discovery but present in checked cold inventory. It is
    // accepted as a real member, not silently removed before classification.
    append(&mut bytes, &collector.open_block(17_999, 24).unwrap());
    for ticket in 9..=16 {
        let r = record(ticket, if ticket == 9 { &[B] } else { &[C] }, 1);
        collector.push(&r).unwrap();
        append(&mut bytes, &r);
    }
    let audit = collector.audit();
    assert_eq!(audit.offered, 16);
    assert_eq!(audit.owners.len(), 1);
    assert_eq!(audit.owners[0].eligible, 1);
    assert_eq!(audit.owners[0].owner_offered, 1);
    assert!(!audit.owners[0].qualified);
    let close = collector.close_block(paired(33_101)).unwrap();
    append(&mut bytes, &close);
    let stop = collector.stop(paired(33_102)).unwrap();
    append(&mut bytes, &stop);
    // A fresh replay independently derives the same union and membership from
    // original seed + records. Neither old closes nor fitted state are patched.
    let mut lines = bytes.split(|b| *b == b'\n').filter(|line| !line.is_empty());
    let header: StructuredServiceHeaderV7 = serde_json::from_slice(lines.next().unwrap()).unwrap();
    let mut replay =
        StructuredServiceCollectorV7::new(header, CostProfileLoadLimits::default()).unwrap();
    for line in lines {
        let record: StructuredServiceRecordV7 = serde_json::from_slice(line).unwrap();
        replay.push(&record).unwrap();
    }
    assert_eq!(replay.source_receipt(), collector.source_receipt());
    assert!(!replay.audit().owners[0].qualified);
}

#[test]
fn source7_cold_seed_is_explicit_domain_and_capacity_bound() {
    let seeded = seeded_header();
    let mut absent = seeded.clone();
    absent
        .declaration
        .nonnegative_envelope
        .as_mut()
        .unwrap()
        .algorithm_universe = None;
    assert!(rebuild(absent).is_err());
    let mut unbound = seeded.clone();
    unbound.declaration.schedule.algorithm_universe =
        Some(OwnerAlgorithmUniversePolicyV1::FirstOrdinaryDiscoveryBlockSubsetV1);
    assert!(
        rebuild(unbound).is_err(),
        "old discovered policy must not reinterpret a seed"
    );
    let mut too_small = seeded.clone();
    let required = too_small
        .declaration
        .nonnegative_envelope
        .as_ref()
        .unwrap()
        .algorithm_universe
        .as_ref()
        .unwrap()
        .retained_payload_bytes()
        .unwrap();
    too_small.declaration.maximum_discovery_bytes = required - 1;
    assert!(rebuild(too_small).is_err());
    let mut wrong_domain = seeded.clone();
    let mut wire = serde_json::to_value(
        wrong_domain
            .declaration
            .nonnegative_envelope
            .as_ref()
            .unwrap()
            .algorithm_universe
            .as_ref()
            .unwrap(),
    )
    .unwrap();
    wire["workload_domain"] = serde_json::to_value([99u8; 32]).unwrap();
    wrong_domain
        .declaration
        .nonnegative_envelope
        .as_mut()
        .unwrap()
        .algorithm_universe = Some(serde_json::from_value(wire).unwrap());
    assert!(
        rebuild(wrong_domain).is_err(),
        "algorithm declaration cannot cross physical D/fingerprint"
    );
    let old = header();
    let encoded = record_bytes_v7(&old).unwrap();
    assert!(!String::from_utf8(encoded.clone())
        .unwrap()
        .contains("seeded_first"));
    let decoded: StructuredServiceHeaderV7 = serde_json::from_slice(&encoded).unwrap();
    assert_eq!(
        record_bytes_v7(&rebuild(decoded).unwrap()).unwrap(),
        encoded
    );
}
