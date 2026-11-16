use super::*;

/// Enumerate complete assignments independently of the solver's frontier,
/// component partitioning and partial-state deduplication.
fn exhaustive_assignment(
    sets: &[Vec<JointProviderCandidate>],
    order: &[DynamicStorageProfile],
) -> Option<(Vec<ProviderId>, BTreeMap<ResourceId, DynamicStorageProfile>)> {
    let count = sets.iter().map(Vec::len).product();
    let mut best = None;
    for mut encoded in 0..count {
        let selected = sets
            .iter()
            .map(|set| {
                let candidate = &set[encoded % set.len()];
                encoded /= set.len();
                candidate
            })
            .collect::<Vec<_>>();
        let resources = selected
            .iter()
            .flat_map(|candidate| candidate.allowed_profiles.keys().cloned())
            .collect::<BTreeSet<_>>();
        let profiles = resources
            .into_iter()
            .map(|resource| {
                order
                    .iter()
                    .copied()
                    .find(|profile| {
                        selected.iter().all(|candidate| {
                            candidate
                                .allowed_profiles
                                .get(&resource)
                                .is_none_or(|allowed| allowed.contains(profile))
                        })
                    })
                    .map(|profile| (resource, profile))
            })
            .collect::<Option<BTreeMap<_, _>>>();
        let Some(profiles) = profiles else { continue };
        let preference = selected
            .iter()
            .map(|candidate| !candidate.is_preferred)
            .collect::<Vec<_>>();
        let penalties = order
            .iter()
            .skip(1)
            .rev()
            .map(|profile| profiles.values().filter(|value| *value == profile).count())
            .collect::<Vec<_>>();
        let providers = selected
            .iter()
            .map(|candidate| candidate.resources.provider_id().clone())
            .collect::<Vec<_>>();
        let objective = (preference, penalties, providers.clone());
        if best
            .as_ref()
            .is_none_or(|(current, _)| objective < *current)
        {
            best = Some((objective, (providers, profiles)));
        }
    }
    best.map(|(_, assignment)| assignment)
}

#[test]
fn joint_frontier_matches_exhaustive_assignments_with_merges_and_rejected_branches() {
    let profiles = [linear_profile(), paged_profile()];
    let shared = ResourceId::new("resource/shared").unwrap();
    let linked = ResourceId::new("resource/linked").unwrap();
    // Every nonempty profile subset for six candidates, with a second shared
    // resource and changing preference bits. Reversals exercise either branch
    // being last, including a rejected last branch after a viable sibling.
    for case in 0..3_usize.pow(6) {
        let mut encoded = case;
        let sets = (0..3)
            .map(|node| {
                let mut candidates = (0..2)
                    .map(|candidate| {
                        let mask = encoded % 3 + 1;
                        encoded /= 3;
                        let accepted = profiles
                            .iter()
                            .enumerate()
                            .filter_map(|(bit, &profile)| {
                                (mask & (1 << bit) != 0).then_some(profile)
                            })
                            .collect::<Vec<_>>();
                        let mut value = joint_candidate(
                            &format!("provider/{node}/{candidate}"),
                            &shared,
                            &accepted,
                        );
                        value.is_preferred = (case >> node) & 1 == candidate;
                        if node != 1 {
                            value.allowed_profiles.insert(
                                linked.clone(),
                                BTreeSet::from([profiles[(node + candidate) % 2]]),
                            );
                        }
                        value
                    })
                    .collect::<Vec<_>>();
                if case & 1 != 0 {
                    candidates.reverse();
                }
                candidates
            })
            .collect::<Vec<_>>();
        for order in [profiles, [profiles[1], profiles[0]]] {
            let expected = exhaustive_assignment(&sets, &order);
            let actual = ExecutionPlan::solve_joint_provider_candidates(&sets, &order)
                .ok()
                .map(|(chosen, profiles)| {
                    (
                        chosen.iter().map(|p| p.provider_id().clone()).collect(),
                        profiles,
                    )
                });
            assert_eq!(actual, expected, "candidate case {case}");
        }
    }
}

#[test]
#[ignore = "CPU planning diagnostic; run explicitly with --ignored --nocapture"]
fn profile_joint_storage_on_single_candidate_chain() {
    let profile = linear_profile();
    let nodes = 4096;
    let sets = (0..nodes)
        .map(|node| {
            let mut candidate = joint_candidate(
                "provider/only",
                &ResourceId::new(format!("resource/{node}")).unwrap(),
                &[profile],
            );
            candidate.allowed_profiles.insert(
                ResourceId::new(format!("resource/{}", node + 1)).unwrap(),
                BTreeSet::from([profile]),
            );
            vec![candidate]
        })
        .collect::<Vec<_>>();
    let started = std::time::Instant::now();
    let (chosen, profiles) =
        ExecutionPlan::solve_joint_provider_candidates(&sets, &[profile]).unwrap();
    let elapsed_seconds = started.elapsed().as_secs_f64();
    assert_eq!(chosen.len(), nodes);
    assert_eq!(profiles.len(), nodes + 1);
    let providers = chosen.iter().map(|p| p.provider_id()).collect::<Vec<_>>();
    let fingerprint = canonical_fingerprint(&(providers, profiles), "joint selection").unwrap();
    eprintln!("joint_storage_profile nodes={nodes} elapsed_seconds={elapsed_seconds:.6} selection_fingerprint={fingerprint}");
}
