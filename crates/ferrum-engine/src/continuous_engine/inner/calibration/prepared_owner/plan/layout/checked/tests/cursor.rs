use super::*;

#[tokio::test]
async fn template_cursor_keeps_declared_length_opportunities_before_smaller_completion_inventory() {
    let (mut session, _) = fixture(2).await;
    Arc::get_mut(&mut session.engine.inner).unwrap().tokenizer = Arc::new(tokenizer().await);
    let model = session.configuration().model.model_id.clone();
    let templates = [
        serve_template("test", &model, AutomaticCostProbeOutput::ApiCompletionSse).unwrap(),
        serve_template(
            "test",
            &model,
            AutomaticCostProbeOutput::ApiChatSse {
                include_usage: false,
            },
        )
        .unwrap(),
        serve_template(
            "test",
            &model,
            AutomaticCostProbeOutput::ApiChatSse {
                include_usage: true,
            },
        )
        .unwrap(),
    ];
    let settings = SloAutomaticCalibrationSettingsV1::default();
    let inputs = Box::pin(PreparedProbeInputs::new(
        &mut session,
        &settings,
        &templates,
    ))
    .await
    .unwrap();
    let (cases, _, _, _) = prepare_cases(&inputs).unwrap();
    assert!(
        cases.iter().filter(|case| case.template == 0).count()
            < cases.iter().filter(|case| case.template == 1).count()
    );
    assert!(!cases.iter().any(|case| case.template == 0
        && case.preset == SloAutomaticCostProbeSamplingPresetV1::GreedyLength));
    for template in [1, 2] {
        assert!(cases.iter().any(|case| case.template == template
            && case.preset == SloAutomaticCostProbeSamplingPresetV1::GreedyLength));
    }
    let cursor = inputs.into_cursor().unwrap();
    assert_eq!(cursor.declared_template_order(), &[1, 2, 0]);
}

mod global;
mod recipe_sharing;
