//! Commit-leaf tests only. The V8 capability below is constructed under cfg(test),
//! never by a public request API. These do not certify source8 collection.
use super::*;
use crate::continuous_engine::inner::slo_controller::tests::fixture::fixture_with_custom_config;
use ferrum_interfaces::output_flow::{CreditedOutputSession, OutputProjectionContract};
use ferrum_interfaces::{InferenceRequestContext, Tokenizer};
use ferrum_tokenizer::implementations::HuggingFaceTokenizer;
use ferrum_types::{FinishReason, InferenceRequest, SamplingParams};
use std::{num::NonZeroUsize, sync::atomic::Ordering, sync::Arc};

async fn fixture(
    prefix: &[u32],
    stops: &[&str],
) -> (CalibrationSession, CreditedOutputSession, RequestId) {
    let (mut engine, _, _) = fixture_with_custom_config(1, |config| {
        config.scheduler.slo.cost_observation.predictor =
            ferrum_types::SloCostPredictor::StructuredWholeWaveV2;
        config.scheduler.slo.cost_observation.structured_capture =
            ferrum_types::SloStructuredCostCapture::HostSettledV1;
    })
    .await;
    let vocab = ["unk", "test", "a", "b", "Ã", "©", "</s>", "<pad>", "other"]
        .into_iter()
        .enumerate()
        .map(|(id, token)| (token.to_owned(), id as u32))
        .collect();
    let mut raw = tokenizers::Tokenizer::new(
        tokenizers::models::wordlevel::WordLevel::builder()
            .vocab(vocab)
            .unk_token("unk".into())
            .build()
            .unwrap(),
    );
    raw.add_special_tokens(&[
        tokenizers::AddedToken::from("</s>", true),
        tokenizers::AddedToken::from("<pad>", true),
    ]);
    raw.with_decoder(Some(tokenizers::decoders::byte_level::ByteLevel::default()));
    raw.with_pre_tokenizer(Some(
        tokenizers::pre_tokenizers::whitespace::Whitespace::default(),
    ));
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> =
        Arc::new(HuggingFaceTokenizer::new(raw).await.unwrap());
    let inner = Arc::get_mut(&mut engine.inner).unwrap();
    inner.tokenizer = tokenizer.clone();
    inner.bg_loop_spawned.store(false, Ordering::Release);
    let mut session = CalibrationSession::from_fresh_engine(
        engine,
        super::super::super::CalibrationLimits::new(NonZeroUsize::MIN).unwrap(),
    )
    .unwrap();
    let mut request = InferenceRequest::new("test", session.configuration().model.model_id.clone());
    request.stream = true;
    request.sampling_params = SamplingParams {
        max_tokens: 8,
        repetition_penalty: 1.1,
        stop_sequences: stops.iter().map(|s| (*s).to_owned()).collect(),
        ..SamplingParams::greedy()
    };
    let id = request.id.clone();
    let output = session
        .add_request_inner(
            request,
            InferenceRequestContext::from_ingress(crate::continuous_engine::inner::slo_clock_now()),
            Arc::new(OutputProjectionContract::cli_text()),
        )
        .await
        .unwrap();
    let declaration = CalibrationPrefixTokensV1 {
        tokenizer_policy_sha256: tokenizer.host_output_policy_identity().unwrap(),
        token_ids: prefix.iter().copied().map(TokenId::new).collect(),
        release_generated: prefix.len(),
    };
    let mut states = session.engine.inner.sequences.write();
    let state = states.get_mut(&id).unwrap();
    let (original_policy, original_numeric) =
        state.current_host_cost_policy(tokenizer.as_ref()).unwrap();
    assert!(matches!(
        original_numeric.empirical_content_domain,
        Some(ferrum_interfaces::execution_cost::HostContentDomainV1::PlainTextInstalledV2(_))
    ));
    state.calibration_prefix = Some(InstalledPrefix {
        plan: declaration
            .validate_for_tokenizer(tokenizer.as_ref(), NonZeroUsize::new(8).unwrap())
            .unwrap(),
        authority: PrefixPreparationAuthority::InstalledPlainTextV8 {
            prepared_policy: None,
        },
        owner: state.cost_frontier.unwrap().owner_incarnation.get(),
        original_policy,
        original_numeric,
        pending_commit: None,
    });
    session.engine.inner.refresh_sequence_cost_policy(state);
    let prepared_policy = state.current_host_cost_policy(tokenizer.as_ref()).unwrap();
    assert!(prepared_policy.1.empirical_content_domain.is_none());
    assert_ne!(prepared_policy.0, original_policy);
    state.calibration_prefix.as_mut().unwrap().authority =
        PrefixPreparationAuthority::InstalledPlainTextV8 {
            prepared_policy: Some(prepared_policy),
        };
    drop(states);
    (session, output, id)
}

fn commit(session: &CalibrationSession, id: &RequestId, original: u32) -> Result<TokenId> {
    let mut states = session.engine.inner.sequences.write();
    let state = states.get_mut(id).unwrap();
    let logits = vec![0.0; session.engine.inner.tokenizer.vocab_size()];
    state.commit_selected_token_with_prefix(
        Some(session.engine.inner.tokenizer.as_ref()),
        TokenId::new(original),
        PrefixCandidateRouteV1::FullLogitsSampler,
        Some(&logits),
    )
}

async fn close(session: CalibrationSession, output: CreditedOutputSession) {
    drop(output);
    session.shutdown().await.unwrap();
}

#[tokio::test]
async fn installed_prefix_v8_preserves_natural_policy_and_records_original_candidate() {
    let (session, output, id) = fixture(&[2], &["ab"]).await;
    let before =
        serde_json::to_value(&session.engine.inner.sequences.read()[&id].sampling_params).unwrap();
    assert_eq!(commit(&session, &id, 6).unwrap(), TokenId::new(2));
    {
        let states = session.engine.inner.sequences.read();
        let state = &states[&id];
        assert_eq!(
            serde_json::to_value(&state.sampling_params).unwrap(),
            before
        );
        assert!(state.model_eos_token_ids.contains(&6));
        assert_eq!(state.stop_text_seqs, ["ab"]);
        assert_eq!(
            state.stop_reason(Some(session.engine.inner.tokenizer.as_ref())),
            None
        );
        let receipt = state
            .calibration_prefix
            .as_ref()
            .unwrap()
            .pending_commit
            .as_ref()
            .unwrap();
        assert_eq!(receipt.original_candidate, TokenId::new(6));
        assert_eq!(receipt.committed_token, TokenId::new(2));
        assert_eq!(state.generated_tokens, [TokenId::new(2)]);
    }
    close(session, output).await;
}

#[tokio::test]
async fn installed_prefix_v8_rejects_eos_special_and_decoded_cross_token_stop() {
    for token in [6, 7] {
        let (session, output, id) = fixture(&[token], &[]).await;
        assert!(commit(&session, &id, 2).is_err());
        assert!(session.engine.inner.sequences.read()[&id]
            .generated_tokens
            .is_empty());
        close(session, output).await;
    }
    let (session, output, id) = fixture(&[2, 3], &["ab"]).await;
    commit(&session, &id, 8).unwrap();
    session
        .engine
        .inner
        .sequences
        .write()
        .get_mut(&id)
        .unwrap()
        .calibration_prefix
        .as_mut()
        .unwrap()
        .pending_commit
        .take();
    assert!(commit(&session, &id, 8).is_err());
    {
        let states = session.engine.inner.sequences.read();
        let state = &states[&id];
        assert_eq!(state.generated_tokens, [TokenId::new(2)]);
        let tok = session.engine.inner.tokenizer.as_ref();
        assert!(state
            .decoded_tokens_match_stop(tok, &[TokenId::new(2), TokenId::new(3)])
            .unwrap());
        assert_eq!(state.stop_reason(Some(tok)), None);
    }
    close(session, output).await;
}

#[tokio::test]
async fn installed_prefix_v8_checks_fresh_policy_and_processed_logits_before_commit() {
    let changes: [fn(&mut SequenceState); 4] = [
        |s| s.sampling_params.repetition_penalty = 1.2,
        |s| {
            s.forbidden_token_ids.insert(2);
        },
        |s| s.stop_text_seqs.push("other".into()),
        |s| s.model_eos_token_ids.push(8),
    ];
    for change in changes {
        let (session, output, id) = fixture(&[2], &[]).await;
        change(session.engine.inner.sequences.write().get_mut(&id).unwrap());
        let error = commit(&session, &id, 8).unwrap_err();
        assert!(error
            .to_string()
            .contains("prefix installed ordinary policy changed"));
        assert!(session.engine.inner.sequences.read()[&id]
            .generated_tokens
            .is_empty());
        close(session, output).await;
    }
    let (session, output, id) = fixture(&[2], &[]).await;
    {
        let mut states = session.engine.inner.sequences.write();
        let state = states.get_mut(&id).unwrap();
        let mut logits = vec![0.0; session.engine.inner.tokenizer.vocab_size()];
        logits[2] = f32::NEG_INFINITY;
        assert!(state
            .commit_selected_token_with_prefix(
                Some(session.engine.inner.tokenizer.as_ref()),
                TokenId::new(8),
                PrefixCandidateRouteV1::FullLogitsSampler,
                Some(&logits)
            )
            .is_err());
        assert!(state.generated_tokens.is_empty());
    }
    close(session, output).await;
}

#[tokio::test]
async fn installed_prefix_v8_retains_real_pending_utf8_and_v5_still_rejects_natural_policy() {
    let (session, output, id) = fixture(&[4], &[]).await;
    assert_eq!(commit(&session, &id, 2).unwrap(), TokenId::new(4));
    assert_eq!(
        session.engine.inner.sequences.read()[&id].pending_decoded_utf8_bytes,
        [0xc3]
    );
    close(session, output).await;
    let (session, output, id) = fixture(&[2], &[]).await;
    session
        .engine
        .inner
        .sequences
        .write()
        .get_mut(&id)
        .unwrap()
        .calibration_prefix
        .as_mut()
        .unwrap()
        .authority = PrefixPreparationAuthority::LengthV5;
    assert!(commit(&session, &id, 2).is_err());
    assert!(session.engine.inner.sequences.read()[&id]
        .generated_tokens
        .is_empty());
    close(session, output).await;
}

#[tokio::test]
async fn installed_prefix_v8_stop_predicate_is_the_normal_postcommit_predicate() {
    let (session, output, id) = fixture(&[2], &["ab"]).await;
    {
        let mut states = session.engine.inner.sequences.write();
        let state = states.get_mut(&id).unwrap();
        // This tests the shared predicate on the original ordinary mutation
        // path, not a source8 release or a fabricated terminal receipt.
        state.calibration_prefix.take();
        state
            .commit_generated_token(
                Some(session.engine.inner.tokenizer.as_ref()),
                TokenId::new(2),
            )
            .unwrap();
        state
            .commit_generated_token(
                Some(session.engine.inner.tokenizer.as_ref()),
                TokenId::new(3),
            )
            .unwrap();
        assert_eq!(
            state.stop_reason(Some(session.engine.inner.tokenizer.as_ref())),
            Some(FinishReason::Stop)
        );
    }
    close(session, output).await;
}
