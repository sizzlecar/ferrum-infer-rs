use super::*;
use ferrum_scheduler::implementations::continuous::prefill_reference::PiecewiseReferenceSpec;

fn spec() -> PiecewiseReferenceSpec {
    PiecewiseReferenceSpec {
        minimum_prompt_tokens: NonZeroU32::new(2).unwrap(),
        maximum_prompt_tokens: NonZeroU32::new(4).unwrap(),
        body_endpoints: vec![NonZeroU32::MIN, NonZeroU32::new(3).unwrap()],
    }
}
async fn reference_wave(
    session: &mut CalibrationSession,
    executor: &ControlledExecutor,
    id: &RequestId,
    output: &mut CreditedOutputSession,
) -> (CalibrationFrontier, CalibrationWaveReport) {
    let before = frontier(session, id);
    let row = match before.prefill_progress() {
        Some((offset, total)) => before
            .prefill_work(spec().next_count(total as u32, offset as u32).unwrap())
            .unwrap(),
        None => before.decode_work().unwrap(),
    };
    let report = wave(session, executor, vec![row]).await;
    assert!(
        matches!(report.observation, CalibrationObservation::Observed { .. }),
        "{:?}",
        report.observation
    );
    if frontier(session, id).generated_tokens() > before.generated_tokens() {
        drop(bounded(output.frames.next()).await.unwrap());
        ready(session, id, false).await;
    }
    (before, report)
}

#[tokio::test]
async fn piecewise_actual_nonuniform_trials_cut_and_product_loader_cover_unmeasured_n() {
    let directory = Directory::new();
    // This controlled executor permanently binds each observed owner to a
    // distinct real core session. Two discovery owners plus three independent
    // fresh trials require five slots even though execution is singleton.
    let (mut session, executor) = observed_fixture_with_owner_capacity(&directory, 5).await;
    let (mut plan, mut evidence, _) = discovery(&mut session, &executor).await;
    plan.piecewise = Some(spec());
    let (id, mut output) = add(&mut session, 4).await;
    admit(&mut session).await;
    let input = *frontier(&session, &id).request_evidence();
    let mut partition = Vec::new();
    for _ in 0..3 {
        let (before, report) = reference_wave(&mut session, &executor, &id, &mut output).await;
        let row = session
            .capture_reference_discovery(&before, &report)
            .unwrap();
        partition.push(row.shape());
        evidence.push(row);
    }
    assert_eq!(
        partition
            .iter()
            .map(|s| s.exact.prefill_chunks[0].count.get())
            .collect::<Vec<_>>(),
        vec![1, 2, 1]
    );
    plan.curves.push(CalibrationReferenceCurve {
        total_prompt_tokens: NonZeroU32::new(4).unwrap(),
        input_tokens_sha256: input.original_input_tokens_sha256,
        partition,
    });
    drop(output);
    let fingerprint = match &session.engine.inner.cost_runtime.as_ref().unwrap().identity {
        ferrum_interfaces::execution_cost::ExecutorCostIdentityAvailability::Known(identity) => {
            identity.clone()
        }
        _ => panic!("controlled identity required"),
    };
    let mut collector = freeze(&mut session, plan, evidence).await.unwrap();
    for (n, key) in [
        (
            2,
            CalibrationReferenceTrial::Prefill {
                curve: 0,
                repetition: 0,
            },
        ),
        (
            4,
            CalibrationReferenceTrial::Prefill {
                curve: 1,
                repetition: 0,
            },
        ),
        (2, CalibrationReferenceTrial::Decode { repetition: 0 }),
    ] {
        let (id, mut output) = add(&mut session, n).await;
        admit(&mut session).await;
        session
            .begin_reference_trial(&mut collector, key, &frontier(&session, &id))
            .unwrap();
        loop {
            let (_, report) = reference_wave(&mut session, &executor, &id, &mut output).await;
            if collector.observe(key, &report).unwrap() {
                break;
            }
        }
        drop(output);
    }
    let model = export(&mut session, directory.cut()).await.unwrap();
    let receipt = collector
        .finish(model.artifact(), &directory.0.join("reference-v2.json"))
        .unwrap();
    assert_eq!(receipt.schema_version, 2);
    assert_eq!(
        receipt.reference_domain,
        Some((NonZeroU32::new(2).unwrap(), NonZeroU32::new(4).unwrap()))
    );
    let expected =
        ferrum_scheduler::implementations::continuous::cost_model::ExecutionFingerprint {
            model_weights: fingerprint.model_weights,
            numerical_policy: fingerprint.numerical_policy,
            device_runtime: fingerprint.device_runtime,
            execution_config: fingerprint.execution_config,
        };
    let loaded = load_prefill_reference(
        &receipt.path,
        &expected,
        receipt.protocol_sha256,
        &Default::default(),
    )
    .unwrap();
    assert_eq!(loaded.supported_lengths().collect::<Vec<_>>(), vec![2, 4]);
    let curve = loaded.curve(NonZeroU32::new(3).unwrap()).unwrap();
    assert!(curve.work_at(2).unwrap() > curve.work_at(1).unwrap());
    assert!(curve.work_at(3).unwrap() > curve.work_at(2).unwrap());
    assert_eq!(receipt.tau_ref_ns, loaded.tau_ref_ns().get());
    assert!(loaded.curve(NonZeroU32::new(5).unwrap()).is_err());
    session.shutdown().await.unwrap();
}
