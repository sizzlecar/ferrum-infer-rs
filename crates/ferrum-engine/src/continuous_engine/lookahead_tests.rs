use super::*;

#[test]
fn decode_lookahead_requires_pure_greedy_work_and_two_remaining_outputs_with_context() {
    for (enabled, pure_decode, max_output, max_context, temperature, expected) in [
        (false, true, 3, 8, 0.0, false),
        (true, false, 3, 8, 0.0, false),
        (true, true, 2, 8, 0.0, false),
        (true, true, 3, 2, 0.0, false),
        (true, true, 3, 8, 0.5, false),
        (true, true, 3, 3, 0.0, true),
    ] {
        let mut config = EngineConfig::default();
        config.batching.decode_lookahead = enabled;
        config.runtime.max_model_len = Some(max_context);
        let engine = test_continuous_engine_with_config(config);
        let mut request = policy_request();
        request.sampling_params.max_tokens = max_output;
        request.sampling_params.temperature = temperature;
        let request_id = request.id.clone();
        let mut sequence = SequenceState::new(request, vec![TokenId::new(7)]);
        sequence.generated_tokens.push(TokenId::new(11));
        sequence.prefill_complete = true;
        sequence.install_runtime_managed_model_kv(Arc::new(
            ferrum_testkit::MockKvCacheHandle::new(request_id.clone(), 1, 1),
        ));
        engine
            .inner
            .sequences
            .write()
            .insert(request_id.clone(), sequence);
        let inputs = engine
            .inner
            .prepare_plan_runtime_decodes(&[request_id], pure_decode);
        assert_eq!(inputs.len(), 1);
        assert_eq!(inputs[0].lookahead.is_some(), expected,
            "enabled={enabled} pure={pure_decode} max_output={max_output} ctx={max_context} temperature={temperature}");
        if let Some(grant) = &inputs[0].lookahead {
            assert!(grant.matches_input(&inputs[0]));
            assert_eq!(grant.successor_input_tokens(), 3);
        }
    }
}
