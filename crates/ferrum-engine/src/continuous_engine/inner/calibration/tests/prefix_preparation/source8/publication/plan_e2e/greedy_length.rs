//! Bind an adopted ordinary replay to the new independently qualified family.
//! The retained query journal is passive evidence, never execution authority.
use super::*;
use ferrum_scheduler::implementations::continuous::{
    cost_profile::ImportedStructuredModelV2, slo_planner::PlanningQueryPhase,
};

pub(super) struct PackedFamily {
    family: NumericalFamilyKeyV1,
    domain: [u8; 32],
    capture: [u8; 32],
}

fn host_policy(child: &ImportedStructuredModelV2) -> Option<HostCostPolicyV2> {
    let family = serde_json::to_value(child.numerical_family_key()?).unwrap();
    Some(serde_json::from_value(family["host_policy"].clone()).unwrap())
}

impl PackedFamily {
    pub(super) fn from_children(children: &[ImportedStructuredModelV2]) -> Self {
        let targets: Vec<_> = children
            .iter()
            .filter(|child| {
                child.owner().role == StructuredWaveRoleV2::OrdinaryDecode
                    && child.owner().product == StructuredProductV2::FullLogits
                    && host_policy(child).is_some_and(|host| {
                        host.empirical_content_domain
                            == Some(HostContentDomainV1::PlainTextGreedyV1)
                    })
            })
            .collect();
        assert_eq!(
            targets.len(),
            1,
            "one original FullLogits/GreedyLength numerical family"
        );
        let target = targets[0];
        // This CPU fixture packs two original raw interpretations. An invented
        // local universe would no longer test exact-scope preservation.
        assert!(target.algorithm_universe().is_none());
        let family = *target.numerical_family_key().unwrap();
        assert_eq!(
            children
                .iter()
                .filter(|child| child.numerical_family_key() == Some(&family))
                .count(),
            1,
            "the original catalog's raw-family lookup has exactly this child"
        );
        let configured = children.iter().find(|child| {
            child.provenance().capture_identity == target.provenance().capture_identity
                && child.owner().role == StructuredWaveRoleV2::OrdinaryDecode
                && child.owner().product == StructuredProductV2::FullLogits
                && host_policy(child).is_some_and(|host| matches!(
                    host.empirical_content_domain,
                    Some(HostContentDomainV1::PlainTextInstalledV2(PlainTextPolicyCapabilityV2 {
                        sampling: PlainTextSamplingRouteV2::FullLogits,
                        ..
                    }))
                ))
        }).expect("the newly retained family must share a real source8 capture with Configured FullLogits");
        assert!(configured.algorithm_universe().is_none());
        assert_ne!(configured.numerical_family_key(), Some(&family));
        assert_ne!(configured.domain_signature(), target.domain_signature());
        // The caller checks each child's independent original F/R/Q minimum.
        // Sharing a journal cannot turn the Configured coefficients into this
        // GreedyLength family's evidence.
        Self {
            family,
            domain: *target.domain_signature(),
            capture: target.provenance().capture_identity,
        }
    }

    pub(super) fn assert_selected_replay(&self, path: &std::path::Path, adopted: &[u64]) {
        let records: Vec<serde_json::Value> = std::fs::read_to_string(path)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert!(!adopted.is_empty());
        let expected_family = serde_json::to_value(self.family).unwrap();
        let mut matched = Vec::new();
        for &transaction in adopted {
            let events: Vec<_> = records
                .iter()
                .filter(|record| record["transaction"].as_u64() == Some(transaction))
                .collect();
            assert_eq!(
                events
                    .iter()
                    .filter(|r| r["event"] == "transaction_end")
                    .count(),
                1
            );
            let selected: Vec<_> = events
                .iter()
                .filter(|r| r["event"] == "selected_replay")
                .collect();
            assert_eq!(
                selected.len(),
                1,
                "adopted transaction {transaction} must identify its original replay"
            );
            let replay = selected[0]["data"]["replay"].as_u64().unwrap();
            assert!(events.iter().any(|record| {
                record["event"] == "replay_end"
                    && record["data"]["replay"].as_u64() == Some(replay)
                    && record["data"]["reason"] == "Completed"
            }));
            let phase = format!("{:?}", PlanningQueryPhase::IndependentReplay { replay });
            for attempt in events.iter().filter(|record| {
                record["event"] == "attempt_begin"
                    && record["data"]["phase"] == phase
                    && record["data"]["depth"].as_u64().unwrap() > 0
            }) {
                let attempt_id = attempt["data"]["attempt"].as_u64().unwrap();
                for query in events.iter().filter(|record| {
                    record["event"] == "query_constructed"
                        && record["data"]["attempt"].as_u64() == Some(attempt_id)
                        && record["data"]["identity"]["numerical_family"] == expected_family
                }) {
                    assert!(query["data"]["input_unknown"].is_null());
                    assert!(query["data"]["demand_error"].is_null());
                    assert_eq!(
                        query["data"]["identity"]["owner"]["product"],
                        serde_json::to_value(StructuredProductV2::FullLogits).unwrap()
                    );
                    let alternative = query["data"]["alternative"].as_u64().unwrap();
                    let lookup: Vec<_> = events
                        .iter()
                        .filter(|record| {
                            record["event"] == "query_lookup"
                                && record["data"]["attempt"].as_u64() == Some(attempt_id)
                                && record["data"]["alternative"].as_u64() == Some(alternative)
                        })
                        .collect();
                    assert_eq!(lookup.len(), 1);
                    assert_eq!(
                        lookup[0]["data"]["outcome"]["kind"], "known",
                        "selected replay must use the original catalog: {:?}",
                        lookup[0]
                    );
                    assert!(
                        lookup[0]["data"]["outcome"]["cost"]["planning_ns"]
                            .as_u64()
                            .unwrap()
                            > 0
                    );
                    matched.push((transaction, replay, attempt_id, alternative));
                }
            }
        }
        assert!(
            !matched.is_empty(),
            "no adopted future replay used the newly packed FullLogits/GreedyLength child; family={:?} domain={:?} capture={:?} adopted={adopted:?}; journal={}",
            self.family, self.domain, self.capture, path.display()
        );
        eprintln!(
            "packed GreedyLength family supports adopted future replay: family={:?} domain={:?} capture={:?} matches={matched:?}; Prefill uses normal admission and subsequent actual Decode waves remain GreedyToken",
            self.family, self.domain, self.capture
        );
    }
}
