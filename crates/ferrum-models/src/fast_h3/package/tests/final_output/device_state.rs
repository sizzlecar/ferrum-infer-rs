use super::*;
use ferrum_execution::TensorSerialActor;

fn compile<F: ModelFamilyProvider>(
    family: F,
    config: &F::Config,
    profile: &str,
    options: ProgramPlanCompileOptions,
    c: &ReferenceVNextComposition,
) -> CompiledTensorComponent<ferrum_kernels::backend::reference::ReferenceDeviceRuntime> {
    let prepared = TypedFamilyRegistration::new(family)
        .prepare_with_profile(&serde_json::to_value(config).unwrap(), &id(profile))
        .unwrap();
    let d = c.runtime().descriptor();
    CompiledTensorComponent::compile(
        prepared,
        c.registry(),
        c.catalog(),
        &TensorComponentPolicy::new(
            d.total_memory_bytes,
            d.dynamic_storage_profiles.iter().copied().collect(),
        )
        .unwrap(),
        c.weight_materializers(),
        &options,
    )
    .unwrap()
}
fn initialization(name: &str) -> ComponentInitializationConfig {
    ComponentInitializationConfig {
        run_id: id(format!("run.device-head.{name}")),
        request_id: id(format!("request.device-head.{name}")),
        transaction_id: id(format!("transaction.device-head.{name}")),
        static_policy: StaticInitializationPolicy::new(1 << 20, 64).unwrap(),
        executor: TensorExecutorConfig::default(),
    }
}

#[test]
fn device_state_head_matches_host_head_after_real_producer_retirement() {
    let (_dir, package) = fixture().open(limits());
    let package = package.unwrap();
    let plan = pdd_projection::plan();
    let layout = layout();
    let c =
        ReferenceVNextComposition::create_with_capacity(id("device.reference.head-state"), 8 << 20)
            .unwrap();
    for evaluation in [1, 7] {
        let account = host(8 << 20);
        let (config, source) = FastH3FinalOutputSource::prepare(
            &package,
            &plan,
            &layout,
            evaluation,
            &account,
            final_limits(),
        )
        .unwrap();
        let state_config = config
            .clone()
            .with_input_residency(FastH3PredictionInput::RequestState);
        let plain = compile(
            F::new(final_limits()).unwrap(),
            &config,
            F::PROFILE_ID,
            config.compile_options(final_limits()).unwrap(),
            &c,
        );
        let state = compile(
            F::new(final_limits()).unwrap(),
            &state_config,
            F::PROFILE_ID,
            state_config.compile_options(final_limits()).unwrap(),
            &c,
        );
        assert_eq!(plain.family().weights(), state.family().weights());
        let mut plain = plain
            .initialize(
                c.runtime().clone(),
                account.clone(),
                ComponentWeightSource::Stored(&source),
                initialization("host"),
            )
            .unwrap();
        let state = state
            .initialize(
                c.runtime().clone(),
                account.clone(),
                ComponentWeightSource::Stored(&source),
                initialization("state"),
            )
            .unwrap();
        let norm = config.normalization();
        let norm_source = FastH3OutputConditioningSource::for_normalization(
            &package,
            norm,
            final_limits().conditioning,
        )
        .unwrap();
        let producer = compile(
            FastH3OutputNormFamily::new(final_limits().conditioning).unwrap(),
            norm,
            FastH3OutputNormFamily::PROFILE_ID,
            norm.compile_options().unwrap(),
            &c,
        );
        let mut producer = producer
            .initialize(
                c.runtime().clone(),
                account.clone(),
                ComponentWeightSource::Stored(&norm_source),
                initialization("producer"),
            )
            .unwrap();
        let h = norm.hidden_size();
        let rows = norm.rows();
        let table = managed(
            &account,
            F::TABLE_ID,
            vec![norm.time_values().len() as u64, 2 * h],
            &vec![0.; norm.time_values().len() * 2 * h as usize],
        );
        let x = (0..rows * h)
            .map(|i| ((i % 13) as f32 - 6.) / 8.)
            .collect::<Vec<_>>();
        let inputs = vec![
            managed(&account, F::INPUT_ID, vec![rows, h], &x),
            table.clone(),
        ];
        let mut actor = TensorSerialActor::new(c.runtime().clone()).unwrap();
        let hidden = producer
            .execute_on_actor(
                &mut actor,
                id("run.head.producer.host"),
                id("request.head.producer.host"),
                inputs.clone(),
            )
            .unwrap();
        let values = hidden[0]
            .bytes()
            .bytes()
            .chunks_exact(2)
            .map(|b| bf16::from_bits(u16::from_le_bytes(b.try_into().unwrap())).to_f32())
            .collect::<Vec<_>>();
        let expected = plain
            .execute_on_actor(
                &mut actor,
                id("run.head.host"),
                id("request.head.host"),
                vec![
                    managed(&account, F::INPUT_ID, vec![rows, h], &values),
                    table.clone(),
                ],
            )
            .unwrap();
        drop(hidden);
        let mut request = state
            .begin_request(
                id("run.head.state"),
                id("request.head.state"),
                vec![table.clone()],
            )
            .unwrap();
        assert!(request.execute(vec![table.clone()]).is_err());
        let before = c.runtime().snapshot();
        let result = producer
            .execute_initializing_request_on_actor(
                &mut actor,
                id("run.head.producer.device"),
                id("request.head.producer.device"),
                inputs,
                &mut request,
                &[TensorRequestStateMapping {
                    source_output: id(FastH3OutputNormFamily::OUTPUT_ID),
                    destination_state: id(F::INPUT_ID),
                    selection: TensorOutputSelection::Whole,
                }],
            )
            .unwrap();
        assert!(result.is_empty());
        assert_eq!(before.readback_bytes, c.runtime().snapshot().readback_bytes);
        assert!(matches!(
            producer
                .close()
                .unwrap_or_else(|e| panic!("producer close: {:?}", e.failure())),
            PlanRuntimeCloseOutcome::Closed(_)
        ));
        let actual = request.execute_on_actor(&mut actor, vec![table]).unwrap();
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(&expected) {
            assert_eq!(actual.value_id(), expected.value_id());
            assert_eq!(actual.tensor(), expected.tensor());
            assert_eq!(
                actual.bytes().bytes(),
                expected.bytes().bytes(),
                "evaluation {evaluation}"
            );
        }
        request.close().unwrap();
        for component in [state, plain] {
            assert!(matches!(
                component
                    .close()
                    .unwrap_or_else(|e| panic!("head close: {:?}", e.failure())),
                PlanRuntimeCloseOutcome::Closed(_)
            ));
        }
        drop(actor);
        assert_eq!(c.runtime().snapshot().live_allocations, 0);
    }
}
