//! Actual private CalibrationSession source8 capture. These short cohorts
//! prove original execution/lifecycle facts, not numerical qualification.
use super::*;
use ferrum_interfaces::{execution_cost::*, ModelExecutor};
use ferrum_scheduler::implementations::continuous::{
    cost_model::structured_v2::{prefixes::*, windows::*, *},
    cost_profile::{
        CostProfileLoadLimits, StructuredPreparedOwnerBlockDeclarationV8,
        StructuredServiceDeclarationV7,
    },
};

fn declaration(
    session: &CalibrationSession,
    maximum: usize,
    no_submission: bool,
) -> StructuredPreparedOwnerBlockDeclarationV8 {
    let runtime = session.engine.inner.cost_runtime.as_ref().unwrap();
    let ExecutorCostIdentityAvailability::Known(identity) = &runtime.identity else {
        panic!("original controlled executor identity");
    };
    // The real fixture executes two rows, a one-token prompt, vocabulary 64,
    // and no recurrent/repetition state. This finite numerical declaration
    // confers no device permission; every original recipe still validates.
    let domain = CostWorkloadDomainV1::new_vnext(
        identity,
        CostWorkloadLimitsV1 {
            maximum_rows: NonZeroU32::new(2).unwrap(),
            maximum_context_tokens: NonZeroU32::new(session.context_capacity().try_into().unwrap())
                .unwrap(),
            maximum_scheduled_tokens_per_wave: NonZeroU64::new(2).unwrap(),
            output_vocabulary_elements: NonZeroU64::new(
                session.engine.inner.model_executor.info().vocab_size as u64,
            )
            .unwrap(),
            repetition_slot_capacity: 0,
            fixed_state_bytes_per_row: 0,
        },
    )
    .unwrap();
    let tokenizer = session
        .engine
        .inner
        .tokenizer
        .host_output_policy_identity()
        .unwrap();
    StructuredPreparedOwnerBlockDeclarationV8 {
        population: StructuredServiceDeclarationV7 {
            schedule: OwnerBlockScheduleV1::new(8, [8; 3], [8; 3]).unwrap(),
            route_population: if no_submission {
                ferrum_types::SloCalibrationRoutePopulationV1::WarmOrGraphDisabledWithNoSubmissionV2
            } else {
                ferrum_types::SloCalibrationRoutePopulationV1::AllAttempts
            },
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
            settings: Default::default(),
            maximum_window_ns: 300_000_000_000,
            maximum_owners: 8,
            maximum_retained_numeric_bytes: 32 * 1024 * 1024,
            maximum_discovery_bytes: 1024 * 1024,
        },
        cohort_plan: CohortPlanV2 {
            phases: std::array::from_fn(|_| {
                vec![CohortV2 {
                    manifest_case: 0,
                    repetition: 0,
                    requests: vec![
                        CohortRequestV2 {
                            manifest_prompt: 0,
                            maximum_output: maximum as u64
                        };
                        2
                    ],
                }]
            }),
        },
        native_prefix_acquisition: None,
        prefix_plan: StructuredPrefixPlanV5 {
            phases: std::array::from_fn(|_| {
                vec![Some(StructuredPrefixCohortV5 {
                    release_generated: 1,
                    slots: vec![
                        StructuredPrefixSlotV5 {
                            tokenizer_policy_sha256: tokenizer,
                            token_ids: vec![TokenId::new(11)],
                            token_bytes: vec![vec![0xc3]],
                        };
                        2
                    ],
                })]
            }),
        },
        cohort_manifest_payload: serde_json::value::to_raw_value(&serde_json::json!({
            "prompt": "test", "prefix": "actual-bytelevel-C3", "maximum_output": maximum,
            "outputs": ["cli_text", "completions_sse"]
        }))
        .unwrap(),
        maximum_offered_waves: 32,
    }
}

async fn source8_session(
    maximum: usize,
    no_submission: bool,
) -> (CalibrationSession, Arc<ControlledExecutor>) {
    let (mut session, executor) = prepared_session_with_width(2).await;
    Arc::get_mut(&mut session.engine.inner)
        .unwrap()
        .config
        .scheduler
        .slo
        .output
        .max_queued_events_per_request =
        ferrum_types::SloOutputConfig::default().max_queued_events_per_request;
    executor
        .emit_structured_cost_observations
        .store(true, Ordering::Release);
    let declared = declaration(&session, maximum, no_submission);
    bounded(session.begin_prepared_owner_source(declared, CostProfileLoadLimits::default()))
        .await
        .unwrap();
    session.begin_prepared_owner_cohort(0, 0).unwrap();
    assert_eq!(executor.physical.load(Ordering::Acquire), 0);
    (session, executor)
}

fn requests(
    session: &CalibrationSession,
    maximum: usize,
    early_stop: bool,
) -> Vec<(ferrum_types::InferenceRequest, OutputProjectionContract)> {
    let mut cli = request(session, maximum);
    cli.sampling_params.stop_sequences = vec![if early_stop { "ok" } else { "not-present" }.into()];
    let mut sse = request(session, maximum);
    sse.sampling_params.stop_sequences = cli.sampling_params.stop_sequences.clone();
    sse.api_request = Some(ferrum_types::ApiRequest::Completion(
        ferrum_types::ApiCompletionRequest {
            prompt: sse.prompt.clone(),
            response_format: None,
        },
    ));
    vec![
        (cli, OutputProjectionContract::cli_text()),
        (
            sse,
            OutputProjectionContract::completions_sse(
                "source8-probe".into(),
                session.configuration().model.model_id.to_string(),
                true,
            ),
        ),
    ]
}

async fn admit_cohort(
    session: &mut CalibrationSession,
    maximum: usize,
    early_stop: bool,
) -> (Vec<RequestId>, Vec<CreditedOutputSession>) {
    let mut ids = Vec::new();
    let mut outputs = Vec::new();
    for (request, contract) in requests(session, maximum, early_stop) {
        ids.push(request.id.clone());
        outputs.push(
            session
                .add_request(
                    request,
                    InferenceRequestContext::capture(),
                    Arc::new(contract),
                )
                .await
                .unwrap(),
        );
    }
    for id in &ids {
        ready(session, id, false).await;
    }
    for _ in &ids {
        admit(session).await;
    }
    (ids, outputs)
}

fn assert_collecting(session: &CalibrationSession, offers: u64, preparations: u64) {
    let source = session.prepared_owner_capture.as_ref().unwrap();
    assert!(
        source.collecting(),
        "actual source8 failure: {:?}",
        source.failure()
    );
    let audit = source.prepared_audit();
    assert!(!audit.population.poisoned, "{audit:?}");
    assert_eq!(audit.preparation_attempts, preparations, "{audit:?}");
    assert_eq!(source.offered(), offers, "{audit:?}");
    assert_eq!(audit.population.offered, offers, "{audit:?}");
    assert_eq!(source.last_fifo(), offers);
}

async fn release(session: &mut CalibrationSession, ids: &[RequestId], fifo: u64) {
    for id in ids {
        ready(session, id, false).await;
    }
    let PrefixReleaseProgressV5::Released { receipts } =
        session.advance_prepared_owner_prefix_release().unwrap()
    else {
        panic!("all original actors must acknowledge the prefix");
    };
    assert_eq!(receipts.len(), ids.len());
    for receipt in receipts {
        assert_eq!(receipt.frontier.pending_utf8, [0xc3]);
        assert_eq!(receipt.frontier.generated_tokens, 1);
        assert_eq!(receipt.frontier.kv_tokens, 1);
        assert_eq!(receipt.through_fifo_ordinal, fifo);
        assert_eq!(
            receipt.actor_applied_output_ordinal,
            receipt.frontier.output_accepted_ordinal
        );
        let sequences = session.engine.inner.sequences.read();
        let sequence = &sequences[&receipt.frontier.request_id];
        assert!(sequence.calibration_prefix.is_none());
        assert_eq!(
            sequence.cost_policy_signature,
            Some(receipt.original_policy_signature)
        );
        assert_eq!(
            sequence.cost_numeric_policy,
            Some(receipt.original_numeric_policy)
        );
        assert!(matches!(
            receipt.original_numeric_policy.empirical_content_domain,
            Some(HostContentDomainV1::PlainTextInstalledV2(_))
        ));
    }
}

mod lifecycle;
mod no_submission;

mod publication;
