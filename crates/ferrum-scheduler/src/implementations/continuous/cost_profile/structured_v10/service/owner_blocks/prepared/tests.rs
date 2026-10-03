use super::*;
mod native_acquisition;
use crate::implementations::continuous::cost_model::structured_v2::{
    prefixes::{StructuredPrefixCohortV5, StructuredPrefixPlanV5, StructuredPrefixSlotV5},
    windows::{CohortRequestV2, CohortV2},
};
use crate::implementations::continuous::cost_profile::structured_v10::tests::fixture as old;
use ferrum_interfaces::execution_cost::{
    CostWorkloadDomainV1, CostWorkloadLimitsV1, ExecutorCostIdentity, EXECUTOR_COST_IDENTITY_SCHEMA,
};
use std::num::{NonZeroU32, NonZeroU64};

fn header() -> StructuredPreparedOwnerBlockHeaderV8 {
    let h = old::header();
    let f = old::fingerprint();
    let domain = CostWorkloadDomainV1::new_vnext(
        &ExecutorCostIdentity {
            schema_version: EXECUTOR_COST_IDENTITY_SCHEMA,
            model_weights: f.model_weights,
            numerical_policy: f.numerical_policy,
            device_runtime: f.device_runtime,
            execution_config: f.execution_config,
        },
        CostWorkloadLimitsV1 {
            maximum_rows: NonZeroU32::new(1).unwrap(),
            maximum_context_tokens: NonZeroU32::new(128).unwrap(),
            maximum_scheduled_tokens_per_wave: NonZeroU64::new(64).unwrap(),
            output_vocabulary_elements: NonZeroU64::new(16).unwrap(),
            repetition_slot_capacity: 3,
            fixed_state_bytes_per_row: 0,
        },
    )
    .unwrap();
    StructuredPreparedOwnerBlockHeaderV8::new(
        [81; 32],
        1,
        h.fingerprint,
        h.producer,
        StructuredServiceClockV7 {
            wall_unix_ns: 1_000_000,
            monotonic_ns: 1,
        },
        StructuredPreparedOwnerBlockDeclarationV8 {
            population: StructuredServiceDeclarationV7 {
                schedule: OwnerBlockScheduleV1::new(8, [8; 3], [8; 3]).unwrap(),
                route_population: ferrum_types::SloCalibrationRoutePopulationV1::AllAttempts,
                domain_policy: StructuredServiceDomainPolicyV1::NonNegativePhysicalEnvelopeV1,
                nonnegative_envelope: Some(NonNegativeEnvelopeContractV1 {
                    algorithm_universe: None,
                    population_policy: Default::default(),
                    planning_estimator: Default::default(),
                    workload_domain: domain,
                    settings: Default::default(),
                    challenge: WorkAxisAndBranchChallengesV1::WorkAxesAndPlainTextBranchesV1,
                    template_policy: StructuredCostTemplatePolicyV1::InstalledAlgorithmSetV1,
                }),
                settings: h.settings.native(),
                maximum_window_ns: 1_000_000_000,
                maximum_owners: 8,
                maximum_retained_numeric_bytes: 32 * 1024 * 1024,
                maximum_discovery_bytes: 1024 * 1024,
            },
            cohort_plan: CohortPlanV2 {
                phases: std::array::from_fn(|_| {
                    vec![CohortV2 {
                        manifest_case: 0,
                        repetition: 0,
                        requests: vec![CohortRequestV2 {
                            manifest_prompt: 0,
                            maximum_output: 3,
                        }],
                    }]
                }),
            },
            native_prefix_acquisition: None,
            prefix_plan: StructuredPrefixPlanV5 {
                phases: std::array::from_fn(|_| {
                    vec![Some(StructuredPrefixCohortV5 {
                        release_generated: 1,
                        slots: vec![StructuredPrefixSlotV5 {
                            tokenizer_policy_sha256: [33; 32],
                            token_ids: vec![ferrum_types::TokenId::new(2)],
                            token_bytes: vec![vec![0xc3]],
                        }],
                    })]
                }),
            },
            cohort_manifest_payload: serde_json::value::to_raw_value(
                &serde_json::json!({"case":0}),
            )
            .unwrap(),
            maximum_offered_waves: 128,
        },
        16 * 1024 * 1024,
    )
    .unwrap()
}

#[test]
fn source8_rejects_source7_phase_support_policy_in_original_header() {
    let mut h = header();
    let bytes = record_bytes_v7(&h).unwrap();
    assert!(!String::from_utf8(bytes).unwrap().contains("phase_support"));
    h.declaration.population.schedule.phase_support = Some(
        crate::implementations::continuous::cost_model::structured_v2::OwnerPhaseSupportPolicyV1::FrozenInputIntersectionV1,
    );
    // Recompute all identities through the producer API: the policy itself,
    // rather than a stale hash from editing a header, must be rejected.
    assert!(StructuredPreparedOwnerBlockHeaderV8::new(
        h.capture_identity,
        h.generation,
        h.fingerprint,
        h.producer,
        h.opening,
        h.declaration,
        h.maximum_file_bytes,
    )
    .is_err());
}

#[test]
fn source8_contract_binds_real_prefix_plan_and_keeps_source7_identity_strict() {
    let h = header();
    h.validate().unwrap();
    let bytes = record_bytes_v7(&h).unwrap();
    let parsed: StructuredPreparedOwnerBlockHeaderV8 = serde_json::from_slice(&bytes).unwrap();
    parsed.validate().unwrap();
    assert!(serde_json::from_slice::<StructuredServiceHeaderV7>(&bytes).is_err());
    let mut bad = h.clone();
    bad.declaration.prefix_plan.phases[0][0]
        .as_mut()
        .unwrap()
        .slots[0]
        .token_ids[0] = ferrum_types::TokenId::new(3);
    assert!(bad.validate().is_err());
    let changed = StructuredPreparedOwnerBlockHeaderV8::new(
        bad.capture_identity,
        bad.generation,
        bad.fingerprint,
        bad.producer,
        bad.opening,
        bad.declaration,
        bad.maximum_file_bytes,
    )
    .unwrap();
    assert_ne!(changed.protocol, h.protocol);
}

#[test]
fn source8_contract_runs_original_utf8_and_release_frontier_validation() {
    let mut h = header();
    let slot = &mut h.declaration.prefix_plan.phases[0][0]
        .as_mut()
        .unwrap()
        .slots[0];
    assert_eq!(slot.expected_pending().unwrap(), [0xc3]);
    slot.token_bytes[0] = vec![0x80];
    assert!(h.declaration.validate().is_err());
    let mut h = header();
    h.declaration.prefix_plan.phases[0][0]
        .as_mut()
        .unwrap()
        .release_generated = 3;
    assert!(h.declaration.validate().is_err());
    let mut h = header();
    h.declaration.cohort_plan.phases[0][0].requests[0].maximum_output = 1;
    assert!(h.declaration.validate().is_err());
}

#[test]
fn source8_contract_cannot_relabel_legacy_rowspace_or_wrong_runtime_domain() {
    let mut h = header();
    h.declaration
        .population
        .nonnegative_envelope
        .as_mut()
        .unwrap()
        .template_policy = StructuredCostTemplatePolicyV1::OrderedV1;
    assert!(h.declaration.validate().is_err());
    let mut h = header();
    h.declaration.population.domain_policy = StructuredServiceDomainPolicyV1::AllOffered;
    assert!(h.declaration.validate().is_err());
    let h = header();
    let mut fingerprint = h.fingerprint.clone();
    fingerprint.device_runtime = [99; 32];
    assert!(StructuredPreparedOwnerBlockHeaderV8::new(
        h.capture_identity,
        h.generation,
        fingerprint,
        h.producer,
        h.opening,
        h.declaration,
        h.maximum_file_bytes,
    )
    .is_err());
}

#[test]
fn source8_declaration_charges_actual_vector_capacity_not_encoded_bytes() {
    let mut d = header().declaration;
    let encoded = record_bytes_v7(&d).unwrap();
    let before = d.retained_payload_bytes().unwrap();
    d.prefix_plan.phases[0][0].as_mut().unwrap().slots[0].token_bytes[0].reserve_exact(4096);
    assert_eq!(record_bytes_v7(&d).unwrap(), encoded);
    assert!(d.retained_payload_bytes().unwrap() >= before + 4096);
    d.population.maximum_retained_numeric_bytes = before;
    assert!(matches!(d.validate(), Err(CostProfileError::Limit(_))));
}

#[test]
fn source8_cannot_shrink_population_or_forge_complete_block_member_capacity() {
    let mut h = header();
    h.declaration.population.schedule.maximum_phase_members[0] -= 1;
    assert!(StructuredPreparedOwnerBlockHeaderV8::new(
        h.capture_identity,
        h.generation,
        h.fingerprint,
        h.producer,
        h.opening,
        h.declaration,
        h.maximum_file_bytes,
    )
    .is_err());
    let mut d = header().declaration;
    d.maximum_offered_waves = d.population.schedule.block_offered - 1;
    assert!(d.validate().is_err());
}

pub(super) mod collector;

mod retained;

mod no_submission;

mod diagnostics;
pub(super) mod numerical_family;
mod partial_tail;
mod phase_exclusion;

mod same_boot;
