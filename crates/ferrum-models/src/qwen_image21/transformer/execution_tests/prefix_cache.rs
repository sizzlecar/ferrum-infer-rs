//! Actual compact prefix transfer and target-only execution over one immutable
//! weight installation. The small numerical fixture is defined by the parent.
use super::*;
use ferrum_execution::component::{
    CompiledTensorComponent, ComponentInitializationConfig, ComponentWeightSource,
    TensorComponentPolicy, TensorComponentSession,
};
use ferrum_kernels::backend::reference::ReferenceDeviceRuntime;

fn family(config: &QwenImage21TransformerConfig) -> PreparedModelFamily {
    TypedFamilyRegistration::new(F::new().unwrap())
        .prepare_with_profile(
            &serde_json::to_value(config).unwrap(),
            &profile_id(F::REFERENCE_PROFILE_ID).unwrap(),
        )
        .unwrap()
}

fn compile(
    composition: &ReferenceVNextComposition,
    config: &QwenImage21TransformerConfig,
) -> CompiledTensorComponent<ReferenceDeviceRuntime> {
    let d = composition.runtime().descriptor();
    CompiledTensorComponent::compile(
        family(config),
        composition.registry(),
        composition.catalog(),
        &TensorComponentPolicy::new(
            d.total_memory_bytes,
            d.dynamic_storage_profiles.iter().copied().collect(),
        )
        .unwrap(),
        composition.weight_materializers(),
        &config.compile_options().unwrap(),
    )
    .unwrap()
}

fn init(suffix: &str) -> ComponentInitializationConfig {
    ComponentInitializationConfig {
        request_id: id(format!("request.prefix.init.{suffix}")),
        run_id: id(format!("run.prefix.init.{suffix}")),
        transaction_id: id(format!("transaction.prefix.init.{suffix}")),
        static_policy: StaticInitializationPolicy::new(4096, 64).unwrap(),
        executor: TensorExecutorConfig::default(),
    }
}

fn inputs(
    compiled: &CompiledTensorComponent<ReferenceDeviceRuntime>,
    host: &HostMemoryAccount,
    raw: &BTreeMap<ProgramValueId, Vec<u8>>,
) -> Vec<ManagedHostTensor> {
    compiled
        .executable()
        .execution_plan()
        .payload()
        .tensor_ports()
        .unwrap()
        .inputs()
        .iter()
        .map(|port| {
            let source = &raw[port.value_id()];
            assert_eq!(
                source.len() as u64,
                port.tensor().minimum_storage_bytes().unwrap()
            );
            let mut bytes = host
                .allocate_zeroed(HostMemoryClass::InputPreprocess, source.len())
                .unwrap();
            bytes.as_mut().copy_from_slice(source);
            ManagedHostTensor::new(
                port.value_id().clone(),
                port.tensor().clone(),
                bytes.freeze(),
            )
            .unwrap()
        })
        .collect()
}

fn close(session: TensorComponentSession<ReferenceDeviceRuntime>) {
    assert!(matches!(
        session
            .close()
            .unwrap_or_else(|e| panic!("close failed: {:?}", e.failure())),
        PlanRuntimeCloseOutcome::Closed(_)
    ));
}

#[test]
fn prefix_phases_share_full_weight_installation_and_match_two_full_steps() {
    let fixture = Fixture::new();
    let prefill_config = fixture
        .config
        .clone()
        .with_execution_phase(QwenImage21TransformerPhase::PrefixPrefill)
        .unwrap();
    let decode_config = fixture
        .config
        .clone()
        .with_execution_phase(QwenImage21TransformerPhase::PrefixDecode)
        .unwrap();
    let mut second = fixture.input.clone();
    for value in second
        .get_mut(&id(F::LATENTS_ID))
        .unwrap()
        .chunks_exact_mut(2)
    {
        let x = bf16::from_bits(u16::from_le_bytes([value[0], value[1]])).to_f32();
        value.copy_from_slice(&bits(-x + 0.5).to_le_bytes());
    }
    // Only sample timesteps change. The shared t0 row remains exactly fixed.
    second.get_mut(&id(F::TIME_FEATURES_ID)).unwrap()[..2].copy_from_slice(&bits(3.).to_le_bytes());
    let full_family = family(&fixture.config);
    let expected_first = execute(
        &full_family,
        fixture.config.compile_options().unwrap(),
        &fixture.full.source,
        fixture.input.clone(),
    );
    let expected_second = execute(
        &full_family,
        fixture.config.compile_options().unwrap(),
        &fixture.full.source,
        second.clone(),
    );
    assert_ne!(
        expected_first[&id(F::VELOCITY_ID)],
        expected_second[&id(F::VELOCITY_ID)]
    );

    let c = ReferenceVNextComposition::create(id("device.reference.prefix.two-layer")).unwrap();
    let prefill = compile(&c, &prefill_config);
    let decode = compile(&c, &decode_config);
    let p = prefill.executable().execution_plan();
    let d = decode.executable().execution_plan();
    assert_eq!(p.payload().execution_weights(), d.payload().execution_weights(),
        "source schema, materializer, quality approval, ordered sources and physical schema stay identical");
    assert_ne!(p.plan_hash(), d.plan_hash());
    let pa = p.payload().memory().static_allocations();
    let da = d.payload().memory().static_allocations();
    assert_eq!(pa.len(), 1);
    assert_eq!(da.len(), 1);
    assert_eq!(pa[0].resource_id(), da[0].resource_id());
    assert!(
        da[0].size_bytes() < pa[0].size_bytes(),
        "decode omits the actual unused text weights"
    );
    let placements = |plan: &ExecutionPlan| {
        plan.payload()
            .nodes()
            .iter()
            .flat_map(|node| node.values())
            .filter(|v| v.usage() == BufferUsage::Weights)
            .flat_map(|v| v.storage().components())
            .map(|v| (v.component_id().unwrap().clone(), v.clone()))
            .collect::<BTreeMap<_, _>>()
    };
    let original = placements(p);
    for (component, placement) in placements(d) {
        assert_eq!(Some(&placement), original.get(&component));
    }
    let host = HostMemoryAccount::new(
        HostMemoryLimits::new(
            12288,
            HostMemoryClassLimits {
                input_preprocess_bytes: 4096,
                output_bytes: 4096,
                transfer_staging_bytes: 4096,
                weight_backing_bytes: 0,
            },
        )
        .unwrap(),
    );
    let (mut producer, weights) = prefill
        .initialize_shared_weights(
            c.runtime().clone(),
            host.clone(),
            ComponentWeightSource::Stored(&fixture.full.source),
            init("prefill"),
        )
        .unwrap();
    assert_eq!(
        weights.evidence().physical_bytes(),
        pa[0].size_bytes(),
        "full source arena stays charged"
    );
    let uploaded = c.runtime().snapshot().uploaded_bytes;
    let consumer = decode
        .initialize_with_shared_weights(c.runtime().clone(), host.clone(), &weights, init("decode"))
        .unwrap();
    assert_eq!(
        c.runtime().snapshot().uploaded_bytes,
        uploaded,
        "attach does not read or upload weights"
    );

    let as_decode = |mut raw: BTreeMap<ProgramValueId, Vec<u8>>| {
        raw.remove(&id(F::TEXT_ID));
        let full = raw.get_mut(&id(F::FREQUENCIES_ID)).unwrap();
        full.drain(..TEXT * H * 4);
        raw
    };
    let first_decode = as_decode(fixture.input.clone());
    let mut request = consumer
        .begin_request(
            id("run.prefix.decode"),
            id("request.prefix.decode"),
            inputs(&decode, &host, &first_decode),
        )
        .unwrap();
    let mappings = decode_config
        .prefix_cache_ports()
        .unwrap()
        .into_iter()
        .map(|port| TensorRequestStateMapping {
            source_output: port.value_id().clone(),
            destination_state: port.value_id().clone(),
        })
        .collect::<Vec<_>>();
    let before = c.runtime().snapshot().readback_bytes;
    let first = producer
        .execute_initializing_request(
            id("run.prefix.prefill"),
            id("request.prefix.prefill"),
            inputs(&prefill, &host, &fixture.input),
            &mut request,
            &mappings,
        )
        .unwrap();
    assert_eq!(
        first.len(),
        1,
        "prefix state transfers never become Host outputs"
    );
    assert_eq!(
        first[0].bytes().bytes(),
        bytes(&expected_first[&id(F::VELOCITY_ID)])
    );
    assert_eq!(
        c.runtime().snapshot().readback_bytes - before,
        (BATCH * IMAGE * H * 2) as u64
    );
    close(producer);
    let repeated = request
        .execute(inputs(&decode, &host, &first_decode))
        .unwrap();
    assert_eq!(repeated[0].bytes().bytes(), first[0].bytes().bytes());
    let second_decode = as_decode(second);
    let next = request
        .execute(inputs(&decode, &host, &second_decode))
        .unwrap();
    assert_eq!(
        next[0].bytes().bytes(),
        bytes(&expected_second[&id(F::VELOCITY_ID)])
    );
    request.close().unwrap();
    close(consumer);
    assert_eq!(
        c.runtime().snapshot().live_allocations,
        1,
        "snapshot still owns the complete original weight arena"
    );
    assert!(matches!(
        weights
            .close()
            .unwrap_or_else(|e| panic!("weight close failed: {:?}", e.failure())),
        SharedComponentWeightsCloseOutcome::Closed(_)
    ));
    assert_eq!(c.runtime().snapshot().live_allocations, 0);
    assert_eq!(
        next[0].bytes().bytes(),
        bytes(&expected_second[&id(F::VELOCITY_ID)])
    );
    drop((first, repeated, next));
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}
