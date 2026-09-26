//! Real manual driver, sampler, KV, output actor, FIFO and Length completion.
//! The controlled device is a CPU protocol fixture, never numerical evidence.
use super::*;
use crate::continuous_engine::inner::cost_observation::EngineCostRuntime;
use crate::continuous_engine::inner::slo_controller::tests::fixture::fixture_with_custom_config;
use ferrum_interfaces::{output_flow::OutputCompletion, Tokenizer};
use ferrum_tokenizer::implementations::HuggingFaceTokenizer;
use ferrum_types::{SamplingParams, TokenId};

async fn prepared_session() -> (CalibrationSession, Arc<ControlledExecutor>) {
    let (mut engine, _, executor) = fixture_with_custom_config(1, |config| {
        let cost = &mut config.scheduler.slo.cost_observation;
        cost.predictor = ferrum_types::SloCostPredictor::StructuredWholeWaveV2;
        cost.structured_capture = ferrum_types::SloStructuredCostCapture::HostSettledV1;
    })
    .await;
    // ByteLevel's real byte alphabet: Ã maps to C3, © maps to A9.
    let vocab = (0..64)
        .map(|id| {
            (
                match id {
                    5 => "test".into(),
                    6 => "ok".into(),
                    10 => "a".into(),
                    11 => "Ã".into(),
                    12 => "©".into(),
                    _ => format!("v{id}"),
                },
                id,
            )
        })
        .collect();
    let mut raw = tokenizers::Tokenizer::new(
        tokenizers::models::wordlevel::WordLevel::builder()
            .vocab(vocab)
            .unk_token("v0".into())
            .build()
            .unwrap(),
    );
    raw.with_decoder(Some(tokenizers::decoders::byte_level::ByteLevel::default()));
    raw.with_pre_tokenizer(Some(
        tokenizers::pre_tokenizers::whitespace::Whitespace::default(),
    ));
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> =
        Arc::new(HuggingFaceTokenizer::new(raw).await.unwrap());
    let inner = Arc::get_mut(&mut engine.inner).unwrap();
    inner.tokenizer = tokenizer;
    let identity = inner.cost_runtime.as_ref().unwrap().identity.clone();
    inner.cost_runtime = Some(Arc::new(
        EngineCostRuntime::new(identity, &inner.config.scheduler.slo.cost_observation, None)
            .unwrap(),
    ));
    inner.bg_loop_spawned.store(false, Ordering::Release);
    executor
        .emit_cost_observations
        .store(true, Ordering::Release);
    executor
        .completion_work_known
        .store(true, Ordering::Release);
    (
        CalibrationSession::from_fresh_engine(
            engine,
            CalibrationLimits::new(NonZeroUsize::MIN).unwrap(),
        )
        .unwrap(),
        executor,
    )
}

fn request(session: &CalibrationSession, maximum: usize) -> ferrum_types::InferenceRequest {
    let mut request =
        ferrum_types::InferenceRequest::new("test", session.configuration().model.model_id.clone());
    request.stream = true;
    request.sampling_params = SamplingParams {
        max_tokens: maximum,
        ..SamplingParams::greedy()
    };
    request
        .metadata
        .insert("ferrum_ignore_eos".into(), true.into());
    request
}

fn plan(session: &CalibrationSession, tokens: &[u32]) -> CalibrationPrefixTokensV1 {
    CalibrationPrefixTokensV1 {
        tokenizer_policy_sha256: session
            .engine
            .inner
            .tokenizer
            .host_output_policy_identity()
            .unwrap(),
        token_ids: tokens.iter().copied().map(TokenId::new).collect(),
        release_generated: tokens.len(),
    }
}

struct LegacyV1Source(std::path::PathBuf);

impl LegacyV1Source {
    fn new() -> Self {
        Self(std::env::temp_dir().join(format!(
            "ferrum-prefix-legacy-v1-{}.jsonl",
            uuid::Uuid::new_v4()
        )))
    }

    fn options(&self) -> StructuredCalibrationOptions {
        StructuredCalibrationOptions {
            observations_path: self.0.clone(),
            protocol_sha256: [1; 32],
            scope: StructuredCalibrationScopeV1 {
                rows: NonZeroUsize::MIN,
                domain_signature: [2; 32],
            },
            settings: Default::default(),
            fit_members: NonZeroUsize::new(16).unwrap(),
            residual_members: NonZeroUsize::new(16).unwrap(),
            qualification_members: NonZeroUsize::new(8).unwrap(),
            maximum_offered_waves: NonZeroUsize::new(128).unwrap(),
            maximum_file_bytes: NonZeroU64::new(1 << 20).unwrap(),
        }
    }
}

impl Drop for LegacyV1Source {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

async fn legacy_v1_begin_rejects_prefix_session(session: &mut CalibrationSession) {
    let source = LegacyV1Source::new();
    let error = bounded(session.begin_structured_cost_calibration(source.options()))
        .await
        .expect_err("the actual old collector entry must reject this prefix session");
    assert!(
        error.to_string().contains("prefix preparation"),
        "an unrelated options/identity error cannot prove session isolation: {error}"
    );
    assert!(session.structured_capture.is_none());
    assert!(!source.0.exists(), "isolation must precede source creation");
}

#[tokio::test]
async fn prefix_preparation_legacy_v1_options_work_without_preparation() {
    let (mut session, executor) = prepared_session().await;
    let source = LegacyV1Source::new();
    bounded(session.begin_structured_cost_calibration(source.options()))
        .await
        .unwrap();
    assert!(session.structured_capture.is_some());
    assert!(source.0.exists());
    assert_eq!(executor.physical.load(Ordering::Acquire), 0);
    session.shutdown().await.unwrap();
}

#[tokio::test]
async fn prefix_preparation_real_pending_frontier_releases_into_normal_sampling_and_length() {
    let (mut session, executor) = prepared_session().await;
    let request = request(&session, 7);
    let id = request.id.clone();
    let declaration = plan(&session, &[10, 11, 12, 11]);
    let output = session
        .add_request_with_prefix_preparation(
            request,
            InferenceRequestContext::from_ingress(slo_clock_now()),
            Arc::new(OutputProjectionContract::cli_text()),
            declaration,
        )
        .await
        .unwrap();
    // Registration itself cannot dispatch, even after yielding to the actor.
    ready(&session, &id, false).await;
    assert_eq!(executor.physical.load(Ordering::Acquire), 0);
    assert!(!session.engine.inner.bg_loop_spawned.load(Ordering::Acquire));
    assert!(session.selected_phase_boundary().is_err());
    legacy_v1_begin_rejects_prefix_session(&mut session).await;
    let consumer = tokio::spawn(async move {
        let mut output = output;
        let mut frames = Vec::new();
        while let Some(frame) = output.frames.next().await {
            frames.push(frame.metadata());
        }
        let completion = output.completion.await.unwrap();
        match completion.payload() {
            OutputCompletion::Succeeded { reason, usage, .. } => {
                eprintln!("prefix actual completion reason={reason:?} usage={usage:?}");
            }
            OutputCompletion::Failed(error) => {
                eprintln!("prefix actual output failure={error:?}");
            }
        }
        assert!(matches!(
            completion.payload(),
            OutputCompletion::Succeeded {
                reason: ferrum_types::FinishReason::Length,
                ..
            }
        ));
        frames
    });
    admit(&mut session).await;
    let mut call = 0;
    let mut fifo = 0;
    for generated in 1..=7 {
        ready(&session, &id, false).await;
        if generated == 3 {
            let sequences = session.engine.inner.sequences.read();
            let sequence = &sequences[&id];
            assert_eq!(sequence.pending_decoded_utf8_bytes, [0xc3]);
            assert!(!sequence.can_use_model_greedy_argmax());
            // A9 is forbidden only to context-free backend Greedy. The real
            // Full sampler below must select it for the actual C3 pending.
            assert_eq!(
                sequence
                    .argmax_token_mask
                    .as_ref()
                    .unwrap()
                    .valid_token_mask[12],
                0
            );
        }
        let current = frontier(&session, &id);
        let work = if generated == 1 {
            current.prefill_work(NonZeroU32::MIN).unwrap()
        } else {
            current.decode_work().unwrap()
        };
        let report = wave(&mut session, &executor, vec![work]).await;
        assert!(report.error.is_none(), "{:?}", report.error);
        let evidence = session.take_prefix_wave_evidence().unwrap();
        if evidence.chain_error.is_some() {
            eprintln!(
                "prefix actual generated={generated} evidence={evidence:#?} report={report:#?}"
            );
        }
        assert!(evidence.chain_error.is_none(), "{:?}", evidence.chain_error);
        let stages = evidence.host_stages.as_ref().unwrap();
        let ordinal = evidence.host_stage_queue.unwrap().accepted_ordinal.unwrap();
        assert!(stages.call_id > call);
        assert_eq!(ordinal, fifo + 1);
        call = stages.call_id;
        fifo = ordinal;
        let row = &evidence.rows[0];
        assert_eq!(row.before.generated_tokens, generated - 1);
        if generated <= 4 {
            assert!(
                matches!(&report.observation, CalibrationObservation::Rejected { reason } if reason == "CalibrationPreparation")
            );
            let commit = row.preparation_commit.as_ref().unwrap();
            assert_eq!(
                commit.committed_token.get(),
                [10, 11, 12, 11][generated - 1]
            );
            assert_eq!(
                commit.original_candidate.get(),
                if generated == 3 { 12 } else { 6 }
            );
            assert_eq!(
                commit.route,
                if generated == 1 || generated == 3 {
                    PrefixCandidateRouteV1::FullLogitsSampler
                } else {
                    PrefixCandidateRouteV1::ModelGreedyArgmax
                }
            );
            assert_eq!(commit.pending_before, row.before.pending_utf8);
            let after = row.after.as_ref().unwrap();
            assert_eq!(commit.pending_after, after.pending_utf8);
            assert_eq!(after.generated_tokens, generated);
            assert_eq!(after.kv_tokens, generated);
        } else {
            assert!(row.preparation_commit.is_none());
        }
        if generated == 4 {
            let before = executor.physical.load(Ordering::Acquire);
            let work = frontier(&session, &id).decode_work().unwrap();
            assert!(session
                .step(CalibrationAction::Wave(vec![work]))
                .await
                .is_err());
            assert_eq!(executor.physical.load(Ordering::Acquire), before);
            ready(&session, &id, false).await;
            let release = session.release_prefix_preparation(&id).unwrap();
            assert_eq!(release.frontier.pending_utf8, [0xc3]);
            assert_eq!(release.frontier.generated_tokens, 4);
            assert_eq!(release.frontier.kv_tokens, 4);
            assert_eq!(release.through_fifo_ordinal, fifo);
            assert_eq!(
                release.actor_applied_output_ordinal,
                release.frontier.output_accepted_ordinal
            );
            assert!(session.release_prefix_preparation(&id).is_err());
            legacy_v1_begin_rejects_prefix_session(&mut session).await;
            let sequences = session.engine.inner.sequences.read();
            let sequence = &sequences[&id];
            assert!(sequence.calibration_prefix.is_none());
            assert_eq!(
                sequence.cost_policy_signature,
                Some(release.original_policy_signature)
            );
            assert_eq!(
                sequence.cost_numeric_policy,
                Some(release.original_numeric_policy)
            );
        }
        if generated == 5 {
            let sequences = session.engine.inner.sequences.read();
            assert_eq!(sequences[&id].generated_tokens[4], TokenId::new(12));
            assert!(sequences[&id].pending_decoded_utf8_bytes.is_empty());
        }
    }
    let frames = bounded(consumer).await.unwrap();
    // A real incomplete UTF-8 token may have no visible wire frame. Every
    // visible token must still identify the actual committed frontier.
    let expected = [10, 11, 12, 11, 12, 6, 6];
    for frame in &frames {
        if let Some(token) = frame.token {
            assert_eq!(token.get(), expected[frame.generated_tokens - 1]);
        }
    }
    assert!(frames.last().unwrap().terminal);
    assert_eq!(frames.last().unwrap().generated_tokens, 7);
    session.finish_prefix_preparation_run().await.unwrap();
    assert!(
        session.selected_phase_boundary().is_err(),
        "completed prepared sessions still cannot feed source3/source4"
    );
    legacy_v1_begin_rejects_prefix_session(&mut session).await;
    session.shutdown().await.unwrap();
}

#[tokio::test]
async fn prefix_preparation_cannot_override_actual_hard_constraint_or_certify_failure() {
    let (mut session, executor) = prepared_session().await;
    let request = request(&session, 3);
    let id = request.id.clone();
    let declaration = plan(&session, &[10]);
    let output = session
        .add_request_with_prefix_preparation(
            request,
            InferenceRequestContext::from_ingress(slo_clock_now()),
            Arc::new(OutputProjectionContract::cli_text()),
            declaration,
        )
        .await
        .unwrap();
    ready(&session, &id, false).await;
    // This is a real current constraint, not an injected successful receipt.
    session
        .engine
        .inner
        .sequences
        .write()
        .get_mut(&id)
        .unwrap()
        .forbidden_token_ids
        .insert(10);
    admit(&mut session).await;
    let work = frontier(&session, &id)
        .prefill_work(NonZeroU32::MIN)
        .unwrap();
    let submitted = executor.physical.load(Ordering::Acquire);
    let report = wave(&mut session, &executor, vec![work]).await;
    let evidence = session.take_prefix_wave_evidence().unwrap();
    // The device submitted successfully. The actual prefix constraint rejects
    // the subsequent host commit, before dispatch can mark HostReconciled.
    assert_eq!(report.submission, CalibrationSubmissionState::Submitted);
    assert_eq!(
        report
            .error
            .as_ref()
            .expect("actual host commit failed")
            .to_string(),
        FerrumError::invalid_request(
            "declared prefix token violates actual sampling/output constraints"
        )
        .to_string()
    );
    assert_eq!(executor.physical.load(Ordering::Acquire), submitted + 1);
    let stages = evidence
        .host_stages
        .as_ref()
        .expect("real failed host receipt");
    assert!(
        stages.actual_shape.is_some(),
        "actual device wave must be recorded"
    );
    assert_eq!(stages.rows.len(), 1);
    assert_eq!(stages.rows[0].request_id, id);
    assert_eq!(evidence.rows[0].before.generated_tokens, 0);
    assert!(evidence.rows[0].preparation_commit.is_none());
    let mut output = output;
    bounded(async {
        while let Some(frame) = output.frames.next().await {
            assert!(frame.metadata().token.is_none(), "no token was committed");
        }
    })
    .await;
    let completion = bounded(output.completion).await.unwrap();
    let OutputCompletion::Failed(error) = completion.payload() else {
        panic!("hard constraint must fail the actual output session");
    };
    // This error is produced only inside commit_selected_token_with_prefix,
    // after the original Full sampler has returned its validated candidate.
    // Resource/clock/recorder errors therefore cannot satisfy this assertion.
    assert_eq!(
        error.message(),
        FerrumError::backend(report.error.as_ref().unwrap().to_string()).to_string()
    );
    assert!(session.prefix_preparation_failure().is_some());
    assert!(session.release_prefix_preparation(&id).is_err());
    assert!(session.finish_prefix_preparation_run().await.is_err());
    drop(completion);
    session.shutdown().await.unwrap();
}

#[tokio::test]
async fn prefix_preparation_rejects_bad_declaration_before_request_registration() {
    let (mut session, executor) = prepared_session().await;
    let request = request(&session, 2);
    let mut declaration = plan(&session, &[10]);
    declaration.tokenizer_policy_sha256[0] ^= 1;
    assert!(session
        .add_request_with_prefix_preparation(
            request,
            InferenceRequestContext::from_ingress(slo_clock_now()),
            Arc::new(OutputProjectionContract::cli_text()),
            declaration
        )
        .await
        .is_err());
    assert!(session.frontiers().unwrap().is_empty());
    assert_eq!(executor.physical.load(Ordering::Acquire), 0);
    assert!(session.prefix_preparation.is_none());
    assert!(session.selected_phase_boundary().is_ok());
    session.shutdown().await.unwrap();
}
