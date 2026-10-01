//! Counterfactual timing feasibility over immutable original source7 blocks.
//! This is not a producer ticket, execution result, or proof of live enrollment.
use super::*;
use crate::implementations::continuous::cost_model::structured_v2::*;
use crate::implementations::continuous::cost_profile::structured_v10::service::owner_blocks::collector::State as OwnerState;
use std::{collections::BTreeMap, num::NonZeroU64};

mod capacity;
mod coverage;
use coverage::{InputAudit, OriginalCoverage};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct OriginalSpec {
    source: PathBuf,
    sha256: [u8; 32],
    /// Additional immutable records audit future queries only. They are never
    /// spliced into a collector or used to complete a missing phase block.
    #[serde(default)]
    future_sources: Vec<FutureSource>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct FutureSource {
    source: PathBuf,
    sha256: [u8; 32],
}

#[derive(Clone, Copy, Debug, serde::Serialize)]
#[serde(rename_all = "snake_case")]
enum TriggerPolicy {
    FitInputReady,
    DiscoveryFrozen,
}

struct Trigger {
    parent_attempt: u64,
    parent_scope: StructuredScopeV2,
    frozen_universe: DeclaredAlgorithmUniverseV1,
    block: u64,
    cutoff: u64,
    at_ns: u64,
    closing: StructuredServiceClockV7,
}

struct Successor {
    trigger: Trigger,
    collector: StructuredServiceCollectorV7,
    original_offset: u64,
    failure: Option<String>,
    activation: Option<u64>,
    readiness_visits: BTreeMap<u64, u64>,
    coverage: OriginalCoverage,
    /// First checked original input in the declared parent lineage, captured
    /// before any numerical close. It binds successor lineage even when its
    /// independent Discovery changes the frozen algorithm universe.
    lineage_input: Option<StructuredInputV2>,
}

fn discovery_triggers(
    original: &StructuredServiceCollectorV7,
    block: u64,
    closing: StructuredServiceClockV7,
) -> Vec<Trigger> {
    original
        .owners
        .iter()
        .filter(|owner| {
            owner.contract.discovery_block == block
                && owner.scope.owner.role == StructuredWaveRoleV2::OrdinaryDecode
                && owner.scope.owner.product == StructuredProductV2::GreedyToken
                && owner.scope.numerical_family.is_some()
        })
        .map(|owner| {
            // This attempt has only completed Discovery: no Fit samples, wall
            // residuals or qualification result can select this launch boundary.
            assert!(matches!(owner.state, OwnerState::Empty));
            assert!(owner.samples.is_empty());
            eprintln!(
                "ROLLING_ORIGINAL_DISCOVERY_FROZEN {}",
                json!({
                    "block":block,"parent_attempt":owner.contract.owner_attempt_id,
                    "closed_at_ns":closing.monotonic_ns,
                })
            );
            Trigger {
                parent_attempt: owner.contract.owner_attempt_id,
                parent_scope: owner.scope.clone(),
                frozen_universe: owner
                    .contract
                    .nonnegative_envelope
                    .as_ref()
                    .unwrap()
                    .algorithm_universe
                    .as_ref()
                    .unwrap()
                    .clone(),
                block,
                cutoff: original.audit().last_fifo,
                at_ns: closing.monotonic_ns,
                closing,
            }
        })
        .collect()
}

fn input_triggers(
    original: &StructuredServiceCollectorV7,
    block: u64,
    closing: StructuredServiceClockV7,
    visits: &mut BTreeMap<u64, u64>,
) -> Vec<Trigger> {
    let mut result = Vec::new();
    for owner in &original.owners {
        // A declared product/role, never the identity of the eventual winner.
        if !matches!(owner.state, OwnerState::Empty)
            || owner.scope.owner.role != StructuredWaveRoleV2::OrdinaryDecode
            || owner.scope.owner.product != StructuredProductV2::GreedyToken
            || owner.scope.numerical_family.is_none()
        {
            continue;
        }
        let boundary = owner.boundary.as_ref().unwrap();
        let offered = original.offered() - boundary.first_offered + 1;
        let decision = owner
            .contract
            .schedule
            .assess_inputs(
                StructuredPhaseV2::Fit,
                offered,
                &owner.samples,
                owner.contract.input_target.as_ref(),
                &original.header.declaration.settings,
                visits.entry(owner.contract.owner_attempt_id).or_default(),
            )
            .unwrap();
        eprintln!(
            "ROLLING_ORIGINAL_INPUT_READY {}",
            json!({
                "block":block,"parent_attempt":owner.contract.owner_attempt_id,
                "decision":format!("{decision:?}"),"members":owner.samples.len(),
                "geometry_visits":visits[&owner.contract.owner_attempt_id],
                "closed_at_ns":closing.monotonic_ns,
            })
        );
        if decision == OwnerInputReadinessDecisionV1::Freeze {
            result.push(Trigger {
                parent_attempt: owner.contract.owner_attempt_id,
                parent_scope: owner.scope.clone(),
                frozen_universe: owner
                    .contract
                    .nonnegative_envelope
                    .as_ref()
                    .unwrap()
                    .algorithm_universe
                    .as_ref()
                    .unwrap()
                    .clone(),
                block,
                cutoff: original.audit().last_fifo,
                at_ns: closing.monotonic_ns,
                closing,
            });
        }
    }
    result
}

impl Successor {
    fn open(
        original: &StructuredServiceHeaderV7,
        trigger: Trigger,
        opened_at_ns: u64,
        fifo: u64,
        original_offset: u64,
        phase_support: Option<OwnerPhaseSupportPolicyV1>,
    ) -> Self {
        assert!(opened_at_ns >= trigger.at_ns && fifo >= trigger.cutoff);
        let mut declaration = original.declaration.clone();
        if let Some(policy) = phase_support {
            declaration.schedule.phase_support = Some(policy);
        }
        let envelope = declaration.nonnegative_envelope.as_mut().unwrap();
        let old_seed_bytes = envelope
            .algorithm_universe
            .as_ref()
            .unwrap()
            .retained_payload_bytes()
            .unwrap();
        let new_seed_bytes = trigger.frozen_universe.retained_payload_bytes().unwrap();
        declaration.maximum_discovery_bytes = declaration
            .maximum_discovery_bytes
            .checked_add(old_seed_bytes)
            .unwrap()
            .checked_sub(new_seed_bytes)
            .unwrap();
        declaration.maximum_retained_numeric_bytes = declaration
            .maximum_retained_numeric_bytes
            .checked_add(old_seed_bytes)
            .unwrap()
            .checked_sub(new_seed_bytes)
            .unwrap();
        envelope.algorithm_universe = Some(trigger.frozen_universe.clone());
        let mut identity = Sha256::new();
        identity.update(b"ferrum.offline.original-block-successor.v1\0");
        identity.update(original.capture_identity);
        identity.update(trigger.parent_attempt.to_le_bytes());
        identity.update(trigger.cutoff.to_le_bytes());
        // The prior original BlockClose supplies a real paired reading. No
        // wall-clock interpolation or publication-time TTL reset is permitted.
        let opening = trigger.closing;
        let header = StructuredServiceHeaderV7::new(
            identity.finalize().into(),
            original.generation,
            original.fingerprint.clone(),
            original.producer.clone(),
            opening,
            declaration,
            original.maximum_file_bytes,
        )
        .unwrap();
        let header = match &original.monotonic_domain {
            Some(domain) => header.with_monotonic_domain(domain.clone()).unwrap(),
            None => header,
        };
        let collector = StructuredServiceCollectorV7::new_streaming(
            header,
            CostProfileLoadLimits::default(),
            NonZeroU64::new(original.maximum_file_bytes).unwrap(),
        )
        .unwrap();
        Self {
            trigger,
            collector,
            original_offset,
            failure: None,
            activation: None,
            readiness_visits: BTreeMap::new(),
            coverage: OriginalCoverage::default(),
            lineage_input: None,
        }
    }

    fn matches_lineage(
        &self,
        scope: &StructuredScopeV2,
        universe: &DeclaredAlgorithmUniverseV1,
    ) -> bool {
        self.lineage_input.as_ref().is_some_and(|input| {
            scope.numerical_family.as_ref().is_some_and(|key| {
                input
                    .numerical_family_key_for_universe(universe)
                    .ok()
                    .as_ref()
                    == Some(key)
            })
        })
    }

    fn observe_input(&mut self, input: &InputAudit, future_only: bool) {
        let raw = input.query.input();
        if !future_only
            && self.lineage_input.is_none()
            && raw
                .numerical_family_key_for_universe(&self.trigger.frozen_universe)
                .ok()
                .as_ref()
                == self.trigger.parent_scope.numerical_family.as_ref()
        {
            self.lineage_input = Some(raw.clone());
        }
        let owner = self.collector.owners.iter().find(|owner| {
            owner
                .contract
                .nonnegative_envelope
                .as_ref()
                .and_then(|c| c.algorithm_universe.as_ref())
                .is_some_and(|u| self.matches_lineage(&owner.scope, u))
        });
        self.coverage.observe(
            input,
            owner,
            self.activation,
            future_only,
            &self.collector.header.fingerprint,
        );
    }

    fn feed(&mut self, original: &StructuredServiceRecordV7) {
        if self.failure.is_some() {
            return;
        }
        let result = match original {
            StructuredServiceRecordV7::BlockOpen {
                opened_at_ns,
                fifo_cutoff,
                ..
            } => self
                .collector
                .open_block(*opened_at_ns, *fifo_cutoff)
                .map(|_| ()),
            StructuredServiceRecordV7::Completed { wave } => {
                let mut wave = wave.clone();
                wave.ticket = wave.ticket.checked_sub(self.original_offset).unwrap();
                self.collector
                    .push(&StructuredServiceRecordV7::Completed { wave })
            }
            StructuredServiceRecordV7::OutsideDeclaredRoute { wave } => {
                let mut wave = wave.clone();
                wave.ticket = wave.ticket.checked_sub(self.original_offset).unwrap();
                self.collector
                    .push(&StructuredServiceRecordV7::OutsideDeclaredRoute { wave })
            }
            StructuredServiceRecordV7::NotSubmitted { attempt } => {
                let ticket = attempt.ticket.checked_sub(self.original_offset).unwrap();
                self.collector
                    .push(&StructuredServiceRecordV7::NotSubmitted {
                        attempt: attempt.clone().with_source_position(ticket),
                    })
            }
            StructuredServiceRecordV7::BlockClose { block, closing, .. } => {
                self.diagnose_wait(*block);
                self.collector.close_block(*closing).map(|record| {
                    let StructuredServiceRecordV7::BlockClose { freezes, .. } = record else {
                        unreachable!()
                    };
                    eprintln!(
                        "ROLLING_ORIGINAL_SUCCESSOR_BLOCK {}",
                        json!({
                            "parent_attempt":self.trigger.parent_attempt,"original_block":block,
                            "closing":closing,"audit":self.collector.audit(),
                            "freezes":freezes.iter().map(|f|json!({"attempt":f.owner_attempt_id,
                                "phase":f.close.phase,"members":f.close.member_count,
                                "failure":f.failure})).collect::<Vec<_>>(),
                        })
                    );
                    for owner in &self.collector.owners {
                        if owner
                            .contract
                            .nonnegative_envelope
                            .as_ref()
                            .and_then(|c| c.algorithm_universe.as_ref())
                            .is_some_and(|u| self.matches_lineage(&owner.scope, u))
                            && matches!(owner.state, OwnerState::Qualified(_))
                            && self.activation.is_none()
                        {
                            self.activation = Some(closing.monotonic_ns);
                        }
                    }
                })
            }
            _ => Ok(()),
        };
        if let Err(error) = result {
            self.failure = Some(format!("{error:?}"));
        }
    }

    fn diagnose_wait(&mut self, original_block: u64) {
        for owner in &self.collector.owners {
            if !matches!(owner.state, OwnerState::Empty)
                || owner.scope.owner.role != StructuredWaveRoleV2::OrdinaryDecode
                || owner.scope.owner.product != StructuredProductV2::GreedyToken
            {
                continue;
            }
            let boundary = owner.boundary.as_ref().unwrap();
            let offered = self.collector.offered() - boundary.first_offered + 1;
            let counts_ready = owner
                .contract
                .schedule
                .is_ready(StructuredPhaseV2::Fit, offered, owner.samples.len())
                .unwrap();
            let target = owner.contract.input_target.as_ref();
            let facts = (!owner.samples.is_empty())
                .then(|| OwnerInputTargetV1::from_samples(&owner.samples).unwrap());
            let gap = seeded::gap(facts.as_ref(), target);
            let visits = self
                .readiness_visits
                .entry(owner.contract.owner_attempt_id)
                .or_default();
            let before = *visits;
            let decision = owner
                .contract
                .schedule
                .assess_inputs(
                    StructuredPhaseV2::Fit,
                    offered,
                    &owner.samples,
                    target,
                    &self.collector.header.declaration.settings,
                    visits,
                )
                .unwrap();
            let missing_input = gap["missing_positive_axes"]
                .as_array()
                .is_some_and(|v| !v.is_empty())
                || gap["missing_branches"]
                    .as_array()
                    .is_some_and(|v| !v.is_empty());
            let reason = if !counts_ready {
                "offered_or_member_minimum"
            } else if missing_input
                && owner.contract.schedule.phase_support
                    != Some(OwnerPhaseSupportPolicyV1::FrozenInputIntersectionV2)
            {
                "missing_input_coverage"
            } else if decision == OwnerInputReadinessDecisionV1::Wait {
                "insufficient_input_geometry"
            } else {
                "see_typed_decision"
            };
            let universe = owner
                .contract
                .nonnegative_envelope
                .as_ref()
                .and_then(|c| c.algorithm_universe.as_ref());
            let missing_axes = gap["missing_positive_axes"].as_array().map(|axes| {
                axes.iter()
                    .map(|axis| {
                        json!({"axis":axis,"algorithm":seeded::algorithm_axis(
                    universe, axis.as_u64().unwrap() as usize)})
                    })
                    .collect::<Vec<_>>()
            });
            eprintln!(
                "ROLLING_ORIGINAL_SUCCESSOR_READINESS {}",
                json!({
                    "original_block":original_block,"attempt":owner.contract.owner_attempt_id,
                    "parent_attempt":self.trigger.parent_attempt,"members":owner.samples.len(),
                    "offered":offered,"decision":format!("{decision:?}"),"reason":reason,
                    "geometry_visits_before":before,"geometry_visits_after":visits,
                    "gap":gap,"missing_axis_algorithms":missing_axes,
                    "first_sample_at_ns":owner.samples.first().map(|s|s.observed_at_ns),
                    "last_sample_at_ns":owner.samples.last().map(|s|s.observed_at_ns),
                    "sample_rows_min":owner.samples.iter().map(|s|s.input.owner().rows).min(),
                    "sample_rows_max":owner.samples.iter().map(|s|s.input.owner().rows).max(),
                    "same_original_numerical_family":owner.scope.numerical_family == self.trigger.parent_scope.numerical_family,
                })
            );
        }
    }
}

#[test]
#[ignore = "requires pinned original source7; counterfactual schedule feasibility, not live tickets"]
fn archived_source7_fit_ready_successor_has_original_time_to_qualify() {
    assert!(
        original_feasibility(None, TriggerPolicy::FitInputReady),
        "earliest Fit-ready schedule has not proven an on-time qualified successor"
    );
}

#[test]
#[ignore = "requires pinned original source7; counterfactual schedule feasibility, not live tickets"]
fn archived_source7_fit_support_v2_successor_has_original_time_to_qualify() {
    assert!(
        original_feasibility(
            Some(OwnerPhaseSupportPolicyV1::FrozenInputIntersectionV2),
            TriggerPolicy::FitInputReady
        ),
        "declared Fit support has not proven an on-time qualified successor"
    );
}

#[test]
#[ignore = "requires pinned original source7; counterfactual schedule feasibility, not live tickets"]
fn archived_source7_discovery_successor_v2_has_original_time_and_reports_query_coverage() {
    assert!(
        original_feasibility(
            Some(OwnerPhaseSupportPolicyV1::FrozenInputIntersectionV2),
            TriggerPolicy::DiscoveryFrozen
        ),
        "Discovery-triggered successor has not qualified before original parent expiry"
    );
}

fn original_feasibility(
    phase_support: Option<OwnerPhaseSupportPolicyV1>,
    trigger_policy: TriggerPolicy,
) -> bool {
    let spec: OriginalSpec = serde_json::from_slice(
        &std::fs::read(
            std::env::var_os("FERRUM_ROLLING_ORIGINAL_SPEC").expect("pinned original source spec"),
        )
        .unwrap(),
    )
    .unwrap();
    assert!(
        spec.future_sources.len() <= 4,
        "bounded original future audit sources"
    );
    let header = seeded::verify_source_file(&spec.source, spec.sha256);
    let mut original = StructuredServiceCollectorV7::new_streaming(
        header.clone(),
        CostProfileLoadLimits::default(),
        NonZeroU64::new(header.maximum_file_bytes).unwrap(),
    )
    .unwrap();
    let mut visits = BTreeMap::new();
    let mut pending = Vec::new();
    let mut successors = Vec::<Successor>::new();
    let mut accepted_blocks = 0;
    let mut original_rejection = None;
    let mut maximum_simultaneous_numeric_bytes = 0;
    for line in BufReader::new(std::fs::File::open(&spec.source).unwrap())
        .lines()
        .skip(1)
    {
        let record: StructuredServiceRecordV7 = serde_json::from_str(&line.unwrap()).unwrap();
        // The policy decision is computed before numerical close/replay, using
        // only current original input rows and persistent geometry-work usage.
        let triggers = match &record {
            StructuredServiceRecordV7::BlockClose { block, closing, .. }
                if matches!(trigger_policy, TriggerPolicy::FitInputReady) =>
            {
                input_triggers(&original, *block, *closing, &mut visits)
            }
            _ => Vec::new(),
        };
        if let Err(error) = original.push(&record) {
            let StructuredServiceRecordV7::Completed { wave } = &record else {
                panic!("unexpected original rejection: {error:?}");
            };
            assert!(matches!(&error, CostProfileError::Metadata(reason)
                if *reason == "source7 observation clock differs"));
            let deadline = header.opening.monotonic_ns + header.declaration.maximum_window_ns;
            assert!(
                wave.issued_at_ns <= deadline
                    && wave.host_stages.finalized_at_ns.unwrap() > deadline
            );
            original_rejection = Some(json!({"ticket":wave.ticket,"fifo":wave.fifo,
                "issued_at_ns":wave.issued_at_ns,"finalized_at_ns":wave.host_stages.finalized_at_ns,
                "deadline_ns":deadline,"error":format!("{error:?}")}));
            // No synthetic complete block or splicing across the failed tail.
            break;
        }
        if matches!(trigger_policy, TriggerPolicy::DiscoveryFrozen) {
            if let StructuredServiceRecordV7::BlockClose { block, closing, .. } = &record {
                pending.extend(discovery_triggers(&original, *block, *closing));
            }
        }
        if let StructuredServiceRecordV7::BlockOpen {
            block,
            opened_at_ns,
            fifo_cutoff,
            ..
        } = &record
        {
            for trigger in pending.drain(..) {
                let trigger: Trigger = trigger;
                assert_eq!(*block, trigger.block + 1);
                // The experiment has an explicit bounded number of collectors;
                // exhaustion fails, never selects the eventual best result.
                assert!(
                    successors.len() < 4,
                    "counterfactual retained-source capacity"
                );
                successors.push(Successor::open(
                    &header,
                    trigger,
                    *opened_at_ns,
                    *fifo_cutoff,
                    original.offered(),
                    phase_support,
                ));
            }
        }
        for successor in &mut successors {
            if let StructuredServiceRecordV7::Completed { wave } = &record {
                let input = InputAudit::checked(&header, wave, accepted_blocks + 1);
                successor.observe_input(&input, false);
            }
            successor.feed(&record);
        }
        pending.extend(triggers);
        if matches!(record, StructuredServiceRecordV7::BlockClose { .. }) {
            accepted_blocks += 1;
        }
        let bytes = original.audit().retained_numeric_bytes
            + successors
                .iter()
                .map(|s| s.collector.audit().retained_numeric_bytes)
                .sum::<usize>();
        maximum_simultaneous_numeric_bytes = maximum_simultaneous_numeric_bytes.max(bytes);
    }
    for source in &spec.future_sources {
        let future = seeded::verify_source_file(&source.source, source.sha256);
        assert_eq!(future.fingerprint, header.fingerprint);
        assert_eq!(
            serde_json::to_value(&future.monotonic_domain).unwrap(),
            serde_json::to_value(&header.monotonic_domain).unwrap()
        );
        assert!(future.opening.monotonic_ns > header.opening.monotonic_ns);
        let mut block = 0;
        for line in BufReader::new(std::fs::File::open(&source.source).unwrap())
            .lines()
            .skip(1)
        {
            let record: StructuredServiceRecordV7 = serde_json::from_str(&line.unwrap()).unwrap();
            match record {
                StructuredServiceRecordV7::BlockOpen { block: n, .. } => block = n,
                StructuredServiceRecordV7::Completed { wave } => {
                    let input = InputAudit::checked(&future, &wave, block);
                    for successor in &mut successors {
                        successor.observe_input(&input, true);
                    }
                }
                _ => {}
            }
        }
    }
    let summaries: Vec<_> = successors.iter().map(|s| {
        let parent = original.owners.iter().find(|o|o.contract.owner_attempt_id == s.trigger.parent_attempt).unwrap();
        let parent_expiry = parent.oldest.checked_add(header.declaration.settings.max_sample_age_ns);
        json!({"parent_attempt":s.trigger.parent_attempt,"trigger_block":s.trigger.block,
            "trigger_ns":s.trigger.at_ns,"parent_sample_expiry_ns":parent_expiry,
            "successor_qualified_at_ns":s.activation,
            "before_parent_expiry":s.activation.zip(parent_expiry).is_some_and(|(at, expiry)|at<=expiry),
            "failure":s.failure,"audit":s.collector.audit()})
    }).collect();
    let feasible = summaries.iter().any(|s| s["before_parent_expiry"] == true);
    eprintln!(
        "ROLLING_ORIGINAL_FEASIBILITY {}",
        json!({"original_source_sha256":spec.sha256,
        "successor_phase_support_override":phase_support,
        "trigger_policy":trigger_policy,
        "query_coverage":successors.iter().map(|s|json!({"parent_attempt":s.trigger.parent_attempt,
            "coverage":s.coverage,"lineage_witness_retained":s.lineage_input.is_some()})).collect::<Vec<_>>(),
        "future_query_sources":spec.future_sources.iter().map(|s|json!({"source":s.source,"sha256":s.sha256})).collect::<Vec<_>>(),
        "accepted_complete_blocks":accepted_blocks,"original_rejection":original_rejection,
        "successors":summaries,"maximum_simultaneous_numeric_bytes":maximum_simultaneous_numeric_bytes,
        "feasible":feasible,
        "limits":"Original source7 collectors and 300s windows; 4 counterfactual successors maximum. Sum reports original plus successors, not a completed production resource-ledger proof.",
        "scope":"Input-triggered counterfactual independent source7 windows over original complete blocks. Source-local ticket coordinates are remapped by the declared opening offset; no live pre-execution enrollment or issued-prediction claim. Original failed tail is not promoted."})
    );
    feasible
}
