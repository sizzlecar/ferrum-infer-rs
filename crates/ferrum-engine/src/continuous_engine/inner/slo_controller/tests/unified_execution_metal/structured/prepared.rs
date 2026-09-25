//! Exercise the production pre-execution getter against the actual native
//! route and original host settlement, including the real final Length wave.
use super::*;
use crate::continuous_engine::inner::slo_controller::calibration::CalibrationPreparation;

#[tokio::test]
async fn structured_prepared_metal_route_is_read_only_and_matches_real_length_settlement() {
    let (mut session, directory) = fixture::fixture_with_structured_capture(true).await;
    let inner = session.test_engine_inner();
    let (id, output) = add(&mut session).await;
    admit(&mut session, &id).await;
    let prefill = frontier(&session, &id).prefill_work(n32(4)).unwrap();
    wave(&mut session, vec![prefill]).await;
    ready(&inner, &id).await;
    let first_decode = frontier(&session, &id).decode_work().unwrap();
    // Warm the actual decode workspace through its real maintenance protocol.
    wave(&mut session, vec![first_decode]).await;

    let mut terminals = 0;
    while !session.frontiers().unwrap().is_empty() {
        ready(&inner, &id).await;
        let before = frontier(&session, &id);
        let work = before.decode_work().unwrap();
        let prepared = match inner
            .prepare_calibration_wave(&[work], NonZeroUsize::new(8).unwrap())
            .unwrap()
        {
            CalibrationPreparation::Selected(prepared) => prepared,
            CalibrationPreparation::Blocked(reason) => {
                panic!("native Prepared wave unavailable: {reason:?}")
            }
        };
        let counters = inner.model_executor.cache_metrics_snapshot().unwrap()["counters"].clone();
        let facts = prepared.structured_prepared_facts(&inner).unwrap();
        facts.validate().unwrap();
        assert_eq!(facts.rows.len(), 1);
        assert_eq!(facts.rows[0].request_id, id);
        assert_eq!(
            facts.rows[0].frontier.generated_before,
            before.generated_tokens() as u64
        );
        assert_eq!(
            inner.model_executor.cache_metrics_snapshot().unwrap()["counters"],
            counters,
            "membership projection must not encode, allocate, or mutate execution caches"
        );
        assert_eq!(
            frontier(&session, &id).generated_tokens(),
            before.generated_tokens()
        );
        assert_eq!(
            frontier(&session, &id).work_generation(),
            before.work_generation()
        );
        let receipt = prepared.calibration_receipt().unwrap();
        tokio::time::timeout(
            Duration::from_secs(30),
            inner.execute_slo_controller_wave(prepared),
        )
        .await
        .unwrap()
        .unwrap();
        let stages = receipt
            .capture()
            .host_stages()
            .expect("original native host settlement");
        let qualified = stages
            .structured_evidence
            .as_ref()
            .unwrap()
            .as_ref()
            .unwrap();
        qualified.validate_host_stages(&stages).unwrap();
        assert_eq!(qualified.recipe(), facts.recipe.as_ref());
        assert_eq!(
            qualified.recipe().algorithm_work().unwrap(),
            facts.recipe.algorithm_work().unwrap(),
        );
        assert_eq!(
            stages.actual_shape.as_ref(),
            Some(&canonical_cost_shape(&facts.exact).unwrap())
        );
        assert_eq!(stages.rows[0].request_id, id);
        terminals += usize::from(stages.rows[0].terminal.is_some());
    }
    assert_eq!(terminals, 1);
    output.await.unwrap();
    session.shutdown().await.unwrap();
    drop(directory);
}
