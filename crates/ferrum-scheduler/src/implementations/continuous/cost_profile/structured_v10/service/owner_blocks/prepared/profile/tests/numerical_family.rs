//! Complete original source8 populations, memory activation and independent
//! profile replay. Timings come from the canonical CPU fixture, not a GPU.
use super::super::super::tests::numerical_family as family_fixture;
use super::*;
use ferrum_interfaces::execution_cost::{ActualWaveGraphState, CostWorkloadDomainV1};

fn future(
    domain: &CostWorkloadDomainV1,
    rows: usize,
    graph: ActualWaveGraphState,
) -> StructuredQueryV2 {
    let names: Vec<_> = (0..rows)
        .map(|i| format!("independent-family-future-{i}"))
        .collect();
    let ids: Vec<_> = names.iter().map(String::as_str).collect();
    old::prepared_batch_route_policy_bounds_and_future(
        &ids,
        2,
        1,
        None,
        graph,
        Some([44; 32]),
        64,
        &vec![4; rows],
        Some(domain),
    )
    .3
    .unwrap()
}

#[test]
fn profile15_homogeneous_family_original_memory_replay_and_future_queries_agree() {
    let (bytes, collector, checkpoint) = family_fixture::collected();
    let h = header(&bytes);
    let limits = CostProfileLoadLimits::default();
    let now = at(&h, checkpoint.population.closing.monotonic_ns + 100);
    let cutoff = checkpoint.source_receipt();
    assert_eq!(cutoff, collector.source_receipt());
    let memory = checkpoint
        .activate_same_process_memory(now, &limits)
        .unwrap();
    let replay = replay_structured_source_v8(&bytes, &limits)
        .unwrap()
        .activate_same_process_memory(now, &limits)
        .unwrap();
    let files = Files::new(&bytes);
    export(&files, &bytes, cutoff.0);
    let file = load_structured_profile_v15(
        &files.profile,
        &old::fingerprint(),
        &limits,
        load_at(now, 17),
    )
    .unwrap();
    let family = memory
        .children
        .iter()
        .filter(|child| child.numerical_family_key().is_some())
        .collect::<Vec<_>>();
    assert_eq!(
        family.len(),
        1,
        "one common statistical family, original owners remain distinct"
    );
    let live = family[0];
    let imported = file
        .children
        .iter()
        .find(|child| child.domain_signature() == live.domain_signature())
        .unwrap();
    let replayed = replay
        .children
        .iter()
        .find(|child| child.domain_signature() == live.domain_signature())
        .unwrap();
    assert_eq!(imported.numerical_family_key(), live.numerical_family_key());
    assert!(live.same_population(imported));
    assert_eq!(imported.parameters_signature(), live.parameters_signature());
    assert_eq!(replayed.parameters_signature(), live.parameters_signature());

    let domain = h
        .declaration
        .population
        .nonnegative_envelope
        .as_ref()
        .unwrap()
        .workload_domain
        .clone();
    let a = future(&domain, 1, ActualWaveGraphState::Disabled);
    let b = future(&domain, 2, ActualWaveGraphState::Disabled);
    assert_ne!(a.owner(), b.owner());
    assert_ne!(a.input().domain_signature(), b.input().domain_signature());
    assert_eq!(
        a.input().numerical_family_key().unwrap(),
        b.input().numerical_family_key().unwrap()
    );
    for query in [&a, &b] {
        let actual = live
            .predict_query_local(&old::fingerprint(), query, now.monotonic_ns)
            .unwrap();
        let from_file = imported
            .predict_query_local(&old::fingerprint(), query, 17)
            .unwrap();
        let from_replay = replayed
            .predict_query_local(&old::fingerprint(), query, now.monotonic_ns)
            .unwrap();
        assert_eq!(
            (actual.planning_ns, actual.valid_until_ns),
            (from_file.planning_ns, from_file.valid_until_ns)
        );
        assert_eq!(
            (actual.planning_ns, actual.valid_until_ns),
            (from_replay.planning_ns, from_replay.valid_until_ns)
        );
    }
    let other_route = future(&domain, 2, ActualWaveGraphState::ConfiguredEager);
    assert!(matches!(
        live.predict_query_local(&old::fingerprint(), &other_route, now.monotonic_ns),
        Err(StructuredUnknownV2::WrongDomain)
    ));

    // The metadata key is a checked replay claim, never trusted authority.
    let original = std::fs::read(&files.profile).unwrap();
    let mut altered: serde_json::Value = serde_json::from_slice(&original).unwrap();
    let child = altered["children"]
        .as_array_mut()
        .unwrap()
        .iter_mut()
        .find(|c| c.get("numerical_family").is_some())
        .unwrap();
    child["numerical_family"]["route"][4] = serde_json::json!(7);
    std::fs::write(&files.profile, serde_json::to_vec(&altered).unwrap()).unwrap();
    assert!(load_structured_profile_v15(
        &files.profile,
        &old::fingerprint(),
        &limits,
        load_at(now, 17),
    )
    .is_err());
}
