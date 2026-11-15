use super::*;
use ferrum_kernels::backend::reference::ReferenceDeviceRuntime;

fn graph(schedule: TensorProgramSchedule) -> StoredComposition {
    let mut parts = [
        ("side", "value.input"),
        ("main", "value.input"),
        ("late", "value.output.main"),
        ("second", "value.output.late"),
        ("unused", "value.input"),
    ]
    .into_iter()
    .map(|(name, input)| {
        Fragment::family_instance(
            &Family::new(),
            &false,
            PROFILE,
            &geometry(),
            mapping(&[("value.input", input)]).unwrap(),
            Some(name),
        )
        .unwrap()
    })
    .collect::<Vec<_>>();
    // The first cast shrinks the live side branch; the second expands it.
    // Keeping the smaller representation early is the reason for this policy.
    for (name, input, output, operation, dtype) in [
        (
            "compress",
            "value.output.side",
            "value.compressed",
            CAST_F32_TO_BF16_OPERATION_ID,
            ElementType::Bf16,
        ),
        (
            "expand",
            "value.compressed",
            "value.expanded",
            CAST_BF16_TO_F32_OPERATION_ID,
            ElementType::F32,
        ),
    ] {
        parts[0].nodes.push(ProgramNode {
            id: id(&format!("node.{name}")),
            operation_id: id(operation),
            required_version: ContractVersion::new(1, 0),
            work: ProgramNodeWorkSpec::Fixed,
            inputs: vec![id(input)],
            outputs: vec![id(output)],
            attributes: BTreeMap::new(),
        });
        parts[0].specs.insert(
            id(output),
            ProgramTensorSpec {
                element_type: dtype,
                ..tensor()
            },
        );
        parts[0].boundaries.insert(id(output), dtype);
        parts[0].operations.push(NumericalOperationContract {
            operation_id: id(operation),
            version: ContractVersion::new(1, 0),
            multiplication_type: None,
            accumulation_type: None,
            staged_arithmetic: None,
            composite_arithmetic: None,
        });
    }
    parts[3].nodes[0].inputs[1] = id("value.expanded");
    joined(parts, 8).unwrap().with_schedule(schedule).unwrap()
}

fn copy_graph(graph: &StoredComposition) -> StoredComposition {
    StoredComposition {
        schema: graph.schema.clone(),
        program: graph.program.clone(),
        profile: graph.profile.clone(),
        options: graph.options.clone(),
    }
}

fn compile(
    c: &ReferenceVNextComposition,
    graph: &StoredComposition,
    capacity: u64,
) -> Result<CompiledTensorComponent<ReferenceDeviceRuntime>, VNextError> {
    let (family, options) =
        copy_graph(graph).prepare_family(&EmptyConfig {}, id("metadata.joined.test"))?;
    CompiledTensorComponent::compile(
        family,
        c.registry(),
        c.catalog(),
        &TensorComponentPolicy::new(
            capacity,
            c.runtime()
                .descriptor()
                .dynamic_storage_profiles
                .iter()
                .copied()
                .collect(),
        )?
        .with_execution_order(PlanExecutionOrder::Serial),
        c.weight_materializers(),
        &options,
    )
}

#[test]
fn delayed_expansion_schedule_preserves_definitions_and_real_capacity() {
    check_schedule(TensorProgramSchedule::DelayedExpansion);
}

#[test]
fn early_compaction_preserves_definitions_capacity_and_managed_results() {
    check_schedule(TensorProgramSchedule::EarlyCompaction);
}

#[test]
fn consumer_chains_preserve_definitions_capacity_and_managed_results() {
    check_schedule(TensorProgramSchedule::ConsumerChains);
}

fn check_schedule(schedule: TensorProgramSchedule) {
    let before = graph(TensorProgramSchedule::DeclarationOrder);
    let after = graph(schedule);
    let position = |name: &str| {
        after.program.blocks()[0]
            .nodes
            .iter()
            .position(|n| n.id.as_str() == name)
            .unwrap()
    };
    if schedule != TensorProgramSchedule::ConsumerChains {
        assert!(position("node.compress") < position("node.add.main"));
    }
    assert!(position("node.expand") > position("node.add.late"));
    check_graph(before, after, false, |values| {
        [2. * values[0] + 0.75, 2. * values[1] - 1.5]
    });
}

#[test]
fn consumer_chains_retire_branches_before_preparing_all_intermediates() {
    let make = |name: &str, input: &str| {
        Fragment::family_instance(
            &Family::new(),
            &false,
            PROFILE,
            &geometry(),
            mapping(&[("value.input", input)]).unwrap(),
            Some(name),
        )
        .unwrap()
    };
    let mut parts = (0..6)
        .map(|index| make(&format!("branch{index}"), "value.input"))
        .collect::<Vec<_>>();
    let mut previous = "value.output.branch0".to_owned();
    for index in 1..6 {
        let name = if index == 5 {
            "second".to_owned()
        } else {
            format!("accumulate{index}")
        };
        let mut part = make(&name, &previous);
        part.nodes[0].inputs[1] = id(&format!("value.output.branch{index}"));
        previous = format!("value.output.{name}");
        parts.push(part);
    }
    let before = joined_with_node_capacity(parts, 8, 16).unwrap();
    let after = copy_graph(&before)
        .with_schedule(TensorProgramSchedule::ConsumerChains)
        .unwrap();
    check_graph(before, after, true, |values| {
        [6. * (values[0] + 0.25), 6. * (values[1] - 0.5)]
    });
}

fn check_graph(
    before: StoredComposition,
    after: StoredComposition,
    requires_reduction: bool,
    expected_values: impl Fn([f32; 2]) -> [f32; 2],
) {
    assert_eq!(before.schema, after.schema);
    assert_eq!(before.profile, after.profile);
    assert_eq!(before.options, after.options);
    assert_eq!(before.program.inputs(), after.program.inputs());
    assert_eq!(before.program.outputs(), after.program.outputs());
    assert_eq!(before.program.weights(), after.program.weights());
    let definitions = |g: &StoredComposition| {
        g.program
            .blocks()
            .iter()
            .flat_map(|b| &b.nodes)
            .map(|n| (n.id.clone(), n.clone()))
            .collect::<BTreeMap<_, _>>()
    };
    // Includes the unused branch; scheduling must not silently prune it.
    assert_eq!(definitions(&before), definitions(&after));
    let c =
        ReferenceVNextComposition::create_with_capacity(id("device.reference.schedule"), 1 << 20)
            .unwrap();
    let required = |compiled: &CompiledTensorComponent<ReferenceDeviceRuntime>| {
        let m = compiled.executable().execution_plan().payload().memory();
        m.static_bytes() + m.minimum_runnable_request_bytes()
    };
    let old = compile(&c, &before, 1 << 20).unwrap();
    let new = compile(&c, &after, 1 << 20).unwrap();
    let capacity = required(&new);
    // Terminal-output reuse can already remove this fixture's old storage
    // penalty. An equal plan is valid; no schedule is promised to always win.
    assert!(capacity <= required(&old));
    if requires_reduction {
        assert!(capacity < required(&old));
    }
    assert!(compile(&c, &after, capacity).is_ok());
    assert!(matches!(
        compile(&c, &after, capacity - 1),
        Err(VNextError::PlanCapacityExceeded { .. })
    ));
    let account = HostMemoryAccount::new(
        HostMemoryLimits::new(
            1 << 20,
            HostMemoryClassLimits {
                input_preprocess_bytes: 1 << 20,
                transfer_staging_bytes: 1 << 20,
                output_bytes: 1 << 20,
                weight_backing_bytes: 1 << 20,
            },
        )
        .unwrap(),
    );
    for compiled in [old, new] {
        let mut weights = account
            .allocate_zeroed(HostMemoryClass::WeightBacking, 8)
            .unwrap();
        for (bytes, value) in weights.as_mut().chunks_exact_mut(4).zip([0.25f32, -0.5]) {
            bytes.copy_from_slice(&value.to_le_bytes());
        }
        let source = Source {
            component: before.schema.components[0].clone(),
            bytes: weights.freeze(),
            reads: AtomicUsize::new(0),
        };
        let mut session = compiled
            .initialize(
                c.runtime().clone(),
                account.clone(),
                ComponentWeightSource::Stored(&source),
                ComponentInitializationConfig {
                    run_id: id("run.schedule.init"),
                    request_id: id("request.schedule.init"),
                    transaction_id: id("transaction.schedule.init"),
                    static_policy: StaticInitializationPolicy::new(1 << 20, 8).unwrap(),
                    executor: TensorExecutorConfig::default(),
                },
            )
            .unwrap();
        drop(source);
        for values in [[1f32, 2.], [-3., 0.5]] {
            let mut input = account
                .allocate_zeroed(HostMemoryClass::InputPreprocess, 8)
                .unwrap();
            for (bytes, value) in input.as_mut().chunks_exact_mut(4).zip(values) {
                bytes.copy_from_slice(&value.to_le_bytes());
            }
            let input = ManagedHostTensor::new(
                id("value.input"),
                ResolvedTensorSpec::new(
                    vec![1, 2],
                    ElementType::F32,
                    ResolvedTensorLayout::Contiguous,
                )
                .unwrap(),
                input.freeze(),
            )
            .unwrap();
            let out = session
                .execute(
                    id("run.schedule.step"),
                    id("request.schedule.step"),
                    vec![input],
                )
                .unwrap();
            let expected = expected_values(values)
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect::<Vec<_>>();
            assert_eq!(out[0].bytes().bytes(), expected);
        }
        assert!(matches!(
            session.close().unwrap_or_else(|e| {
                let reason = format!("{:?}", e.failure());
                std::mem::forget(e);
                panic!("{reason}")
            }),
            PlanRuntimeCloseOutcome::Closed(_)
        ));
        assert_eq!(c.runtime().snapshot().live_allocations, 0);
        assert_eq!(account.usage().unwrap().committed_bytes(), 0);
    }
}

#[test]
fn delayed_expansion_schedule_does_not_erase_checkpoint_input_roles() {
    for schedule in [
        TensorProgramSchedule::DelayedExpansion,
        TensorProgramSchedule::EarlyCompaction,
        TensorProgramSchedule::ConsumerChains,
    ] {
        let mut built = graph(TensorProgramSchedule::DeclarationOrder);
        built.program = built
            .program
            .with_checkpoint_inputs(
                ProgramCheckpointInputs::new(id("value.input"), BTreeSet::new()).unwrap(),
            )
            .unwrap();
        assert!(built.with_schedule(schedule).is_err());
    }
}
