//! Real selected primitive providers: future declarations versus submitted work.
//! Reuses the native compiler, validated weight source and resource lifecycle.
use super::*;
use std::num::NonZeroU64;

#[path = "primitive_cost_route/family.rs"]
mod family;
use family::{LinearWeight, PrimitiveFamily, LINEAR_OUTPUT};

struct NoTiming;
impl DeviceSubmissionTimingSink for NoTiming {
    const ENABLED: bool = false;
    fn record_device_submission(&self, _: DeviceSubmissionStage, _: std::time::Duration) {
        panic!("cost route parity must not enable profiling");
    }
}
impl SubmissionWaveDispatchTimingSink for NoTiming {
    fn record(&self, _: SubmissionWaveDispatchStage, _: std::time::Duration) {
        panic!("cost route parity must not enable profiling");
    }
}

fn row(offset: u64, count: u64, full: u64) -> OperationCostWorkRow {
    OperationCostWorkRow {
        offset,
        count: NonZeroU64::new(count).unwrap(),
        full_input_tokens: NonZeroU64::new(full).unwrap(),
    }
}

fn fixture(hadamard: bool, vocabulary: u64, linear: LinearWeight) -> (Fixture, WeightSchema) {
    let kind = if hadamard {
        AttentionKind::GatedDeltaHadamardF16
    } else {
        AttentionKind::GatedDelta
    };
    let definition = PrimitiveFamily {
        base: Family::new(kind),
        vocabulary,
        linear,
        graph_attention: false,
    };
    let inputs = definition.additional_inputs();
    let profile_id = definition.base.profile_id();
    let prepared = TypedFamilyRegistration::new(definition)
        .prepare_with_profile(&serde_json::to_value(kind).unwrap(), &id(profile_id))
        .unwrap();
    // Preparation normalizes component order. The deterministic source uses
    // that ordinal to generate values, so the oracle must share exactly the
    // schema supplied to static initialization, including optional components.
    let schema = prepared.weight_schema().clone();
    (
        Fixture::from_prepared_family(kind, prepared, vec![], false, None, 1, inputs),
        schema,
    )
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PrimitiveExecutionStage {
    Eager,
    GraphWarmup,
    GraphCapture,
    DirectReplay,
}

fn compare(hadamard: bool, vocabulary: u64, linear: LinearWeight, row_lengths: [usize; 2]) {
    let (fixture, schema) = fixture(hadamard, vocabulary, linear);
    execute(
        &fixture,
        &schema,
        hadamard,
        vocabulary,
        linear,
        &row_lengths,
        PrimitiveExecutionStage::Eager,
        None,
        0,
    );
}

#[allow(clippy::too_many_arguments)]
fn execute(
    fixture: &Fixture,
    schema: &WeightSchema,
    hadamard: bool,
    vocabulary: u64,
    linear: LinearWeight,
    row_lengths: &[usize],
    stage: PrimitiveExecutionStage,
    require_selected: Option<bool>,
    token_seed: u32,
) -> BTreeMap<(String, u32), Vec<u8>> {
    let replay = stage == PrimitiveExecutionStage::DirectReplay;
    let tokens: Vec<Arc<[u32]>> = row_lengths
        .iter()
        .map(|&length| {
            (0..length)
                .map(|index| ((index as u32 + token_seed) % 31 + 1) as u32)
                .collect()
        })
        .collect();
    let admitted = tokens
        .iter()
        .enumerate()
        .map(|(index, tokens)| fixture.admit(&format!("route.{index}"), Arc::clone(tokens)))
        .collect::<Vec<_>>();
    let batch = ExecutionBatchParticipants::new(admitted.clone()).unwrap();
    let sessions = batch.sessions().to_vec();
    let tokens = sessions
        .iter()
        .map(|session| {
            let source = admitted
                .iter()
                .position(|candidate| Arc::ptr_eq(candidate, session))
                .unwrap();
            Arc::clone(&tokens[source])
        })
        .collect::<Vec<_>>();
    let row_lengths = tokens.iter().map(|row| row.len()).collect::<Vec<_>>();
    let rows = row_lengths
        .iter()
        .map(|&length| row(0, length as u64, length as u64))
        .collect::<Vec<_>>();
    let executable = fixture.compilation.executable();
    // Query BEFORE step admission and resource preparation. This is the
    // selected production provider, not a metadata/mock substitute.
    let routes = fixture
        .providers
        .providers()
        .iter()
        .map(|provider| {
            provider
                .eager_cost_route(executable, &rows)
                .unwrap_or_else(|error| {
                    panic!(
                        "primitive route {} failed: {error}",
                        provider.descriptor().operation_id()
                    )
                })
        })
        .collect::<Vec<_>>();
    let graph_attention = executable
        .execution_plan()
        .payload()
        .nodes()
        .iter()
        .any(|node| node.id().as_str() == "node.attention");
    assert_eq!(routes.len(), if graph_attention { 6 } else { 5 });

    for (provider, route) in fixture.providers.providers().iter().zip(&routes) {
        // CUDA's general DenseLinear provider executes real native work but
        // does not declare an eager cost route. Keep it explicitly opaque;
        // selected embedding/argmax evidence never upgrades the whole program.
        let opaque = cfg!(feature = "cuda")
            && provider.descriptor().operation_id().as_str() == DENSE_LINEAR_OPERATION_ID;
        assert_eq!(
            route.is_none(),
            opaque,
            "unexpected route coverage for operation {}",
            provider.descriptor().operation_id()
        );
    }
    #[cfg(all(feature = "metal", not(feature = "cuda"), target_os = "macos"))]
    {
        let routes = routes
            .iter()
            .map(|route| route.as_ref().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(routes.len(), 5);
        assert_eq!(
            routes[0].commands()[0].compute_dispatch_count(),
            if hadamard { 4 } else { 2 }
        );
        assert_eq!(
            routes[1].commands()[0].batching(),
            DeviceBatchingForm::Packed
        );
        assert_eq!(
            routes[2].commands()[0].batching(),
            DeviceBatchingForm::Packed
        );
        assert_eq!(
            routes[4].commands()[0].compute_dispatch_count(),
            if vocabulary >= 8192 { 4 } else { 2 }
        );

        assert_eq!(
            routes[3].commands()[0].native_operation(),
            "vnext_dense_linear"
        );
        assert_eq!(
            routes[3].commands()[0].batching(),
            DeviceBatchingForm::Packed
        );
        // This shape executes a complete 32-row prefix and a one-row Q4_K tail.
        // Compare to the actual runtime attribution below; no dispatch-count
        // helper is called by this fixture as a substitute for execution evidence.
        let expected_linear_dispatches = if linear == LinearWeight::Q4K { 2 } else { 1 };
        assert_eq!(
            routes[3].commands()[0].compute_dispatch_count(),
            expected_linear_dispatches
        );
    }

    let work = batch
        .bind_work_shape(
            tokens
                .iter()
                .map(|tokens| token_span(Arc::clone(tokens), 0..tokens.len()))
                .collect(),
        )
        .unwrap();
    let mut request = StepResourceAdmissionRequest::new(
        work,
        AdmissionFitPolicy::ImmediateOnly,
        AdmissionPressureAction::WaitForRelease,
    )
    .unwrap();
    if let Some(bucket) = &fixture.reusable_bucket {
        request = request.with_reusable_execution_bucket(bucket.clone());
    }
    let step = loop {
        match batch
            .try_begin_step(request.clone(), &fixture.lane)
            .unwrap()
        {
            StepResourceAdmissionDecision::Admitted(step) => break step,
            StepResourceAdmissionDecision::BackingDeferred(deferred) => {
                require_progress(deferred.maintain().unwrap())
            }
            StepResourceAdmissionDecision::Deferred(reason) => require_progress(
                fixture
                    .resources
                    .maintain_for_admission_deferred(&reason)
                    .unwrap(),
            ),
            StepResourceAdmissionDecision::PermanentRejected(reason) => {
                panic!("primitive batch rejected: {reason:?}")
            }
        }
    };
    let wave = loop {
        match step
            .try_prepare_full_plan_submission_wave(
                Arc::new(step.work_shape().clone()),
                AdmissionFitPolicy::ImmediateOnly,
                AdmissionPressureAction::WaitForRelease,
            )
            .unwrap()
        {
            StepSubmissionWaveAdmissionDecision::Prepared(wave) => break wave,
            StepSubmissionWaveAdmissionDecision::BackingDeferred(deferred) => {
                require_progress(deferred.maintain().unwrap())
            }
            StepSubmissionWaveAdmissionDecision::Deferred(reason) => require_progress(
                fixture
                    .resources
                    .maintain_for_admission_deferred(&reason)
                    .unwrap(),
            ),
            other => panic!(
                "primitive wave could not be prepared: {:?}",
                std::mem::discriminant(&other)
            ),
        }
    };
    let active = sessions
        .iter()
        .map(|session| TrustedActiveSequenceBinding::from_session(session).unwrap())
        .collect::<Vec<_>>();
    let identity = OperationDispatch::bind_submission_wave_identity(
        executable,
        active.iter(),
        &wave,
        &fixture.lane,
    )
    .unwrap();
    let mut uploads = Vec::new();
    for (participant, tokens) in tokens.iter().enumerate() {
        uploads.push(
            SubmissionWaveInputUpload::new(
                id("node.embedding"),
                participant as u32,
                0,
                0,
                HostTransferLayout::new(ElementType::U32, tokens.len() as u64).unwrap(),
                tokens.iter().flat_map(|v| v.to_le_bytes()).collect(),
            )
            .unwrap(),
        );
        let logits = (0..vocabulary)
            .flat_map(|index| {
                f16::from_f32(if index == 7 + participant as u64 + u64::from(token_seed) {
                    10.0
                } else {
                    -1.0
                })
                .to_le_bytes()
            })
            .collect();
        for (ordinal, dtype, length, bytes) in [
            (0, ElementType::F16, vocabulary, logits),
            (1, ElementType::U8, vocabulary, vec![1; vocabulary as usize]),
            (2, ElementType::U32, 1, 0_u32.to_le_bytes().to_vec()),
            (3, ElementType::U32, 2, vec![0; 8]),
            (4, ElementType::F32, 1, 1_f32.to_le_bytes().to_vec()),
        ] {
            uploads.push(
                SubmissionWaveInputUpload::new(
                    id("node.argmax"),
                    participant as u32,
                    ordinal,
                    0,
                    HostTransferLayout::new(dtype, length).unwrap(),
                    bytes,
                )
                .unwrap(),
            );
        }
    }
    if graph_attention {
        let binding_nodes = executable
            .execution_plan()
            .payload()
            .nodes()
            .iter()
            .enumerate()
            .filter(|(_, node)| node.binding_resource().is_some())
            .map(|(index, _)| index)
            .collect::<Vec<_>>();
        assert_eq!(
            binding_nodes,
            vec![5],
            "only the real GDN owns dynamic state bindings"
        );
        assert!(wave.claimed_backing().program_binding_node(5).is_some());
        assert!(
            OperationDispatch::reusable_execution_program_id_for_wave(
                fixture.providers.providers(),
                executable,
                &wave,
                &fixture.lane,
            )
            .unwrap()
            .is_some(),
            "real state binding must authorize every graph phase"
        );
    }
    let catalog = replay.then(|| fixture.lane.reusable_execution_catalog().unwrap());

    let program = if let Some(catalog) = &catalog {
        let id = OperationDispatch::reusable_execution_program_id_for_wave(
            fixture.providers.providers(),
            executable,
            &wave,
            &fixture.lane,
        )
        .unwrap()
        .unwrap();
        Some(
            catalog
                .programs()
                .iter()
                .find(|p| p.program_id() == &id)
                .expect("real primitive graph must be resident, no eager fallback"),
        )
    } else {
        None
    };
    let (handle, attribution) = OperationDispatch::encode_and_submit_wave_with_cost_observation(
        fixture.providers.providers(),
        executable,
        &identity,
        active.iter(),
        DeviceTimingMode::Off,
        &uploads,
        if replay {
            SubmissionExecutionPolicy::determinism_replayed(1)
        } else {
            SubmissionExecutionPolicy::adaptive()
        },
        program,
        &NoTiming,
        wave,
        &fixture.lane,
        &fixture.reaper,
    )
    .unwrap()
    .into_parts();
    let attribution =
        attribution.expect("real runtime must retain actual logical attribution without timing");
    #[cfg(feature = "cuda")]
    {
        let graph = attribution
            .device()
            .graph_evidence()
            .expect("actual CUDA graph state");
        match stage {
            PrimitiveExecutionStage::Eager => assert!(graph.proves_unconfigured_eager()),
            PrimitiveExecutionStage::GraphWarmup => {
                assert!(
                    !graph.proves_unconfigured_eager(),
                    "warmup uses the configured graph lane"
                );
                assert_eq!(
                    graph.replayed_segments(),
                    0,
                    "first warmup must execute eagerly"
                );
            }
            PrimitiveExecutionStage::GraphCapture => {
                assert!(
                    graph.replayed_segments() > 0,
                    "second adaptive wave must execute captured work"
                );
                assert!(
                    !graph.proves_warm_direct_replay(),
                    "capture phase must not masquerade as binding-only replay"
                );
                assert!(
                    attribution.device().replayed_segments().is_empty(),
                    "adaptive capture retains physical commands, not direct-program logical segments"
                );
            }
            PrimitiveExecutionStage::DirectReplay => assert!(graph.proves_warm_direct_replay()),
        }
    }
    if !replay {
        let compute = attribution
            .device()
            .commands()
            .iter()
            .filter(|command| command.command_phase() == DeviceCommandPhase::Compute)
            .collect::<Vec<_>>();
        assert_eq!(
            compute.len(),
            routes.len(),
            "no undeclared physical compute commands"
        );
        for (node, (actual, route)) in compute.iter().zip(&routes).enumerate() {
            assert_eq!(actual.node_index(), Some(node as u32));
            assert_eq!(
                actual.execution_path(),
                if stage == PrimitiveExecutionStage::GraphCapture {
                    DeviceExecutionPath::Replayed
                } else {
                    DeviceExecutionPath::Eager
                },
                "actual primitive path must match its declared fixture phase"
            );
            // Timing Off retains execution paths and aggregate graph evidence,
            // but does not request per-command native graph-node attribution.
            assert_eq!(actual.reusable_graph_node_count(), None);
            let Some(route) = route else {
                // Coverage was checked against the selected operation above.
                // This command still executes and is read back below; it must
                // pass the unchanged independent linear output oracle.
                continue;
            };
            let predicted_compute = route
                .commands()
                .iter()
                .filter(|command| command.phase() == DeviceCommandPhase::Compute)
                .collect::<Vec<_>>();
            let [predicted] = predicted_compute.as_slice() else {
                panic!("one declared physical compute command per node")
            };
            assert_eq!(actual.native_op_id(), predicted.native_operation());
            assert_eq!(actual.command_phase(), predicted.phase());
            assert_eq!(actual.participant_start(), predicted.participant_start());
            assert_eq!(actual.participant_count(), predicted.participant_count());
            assert_eq!(actual.token_count(), predicted.token_count());
            assert_eq!(actual.batching_form(), predicted.batching());
            assert_eq!(
                actual.compute_dispatch_count(),
                predicted.compute_dispatch_count()
            );
            assert_eq!(
                actual.transfer_command_count(),
                predicted.transfer_command_count()
            );
        }
    }
    #[cfg(feature = "cuda")]
    if let Some(required) = require_selected {
        for index in [0_u32, 4] {
            let [projected] = routes[index as usize]
                .as_ref()
                .expect("embedding and argmax must have declared routes")
                .commands()
            else {
                panic!("one primitive command")
            };
            let expected = projected.statistical_evidence();
            assert_eq!(
                expected.is_some(),
                required,
                "future selected evidence: stage={stage:?}, node={index}, hadamard={hadamard}, token_seed={token_seed}"
            );
            let actual = if replay {
                let rows = attribution
                    .device()
                    .replayed_segments()
                    .iter()
                    .flat_map(|s| s.logical_commands())
                    .filter(|c| c.node_index() == index)
                    .collect::<Vec<_>>();
                let [actual] = rows.as_slice() else {
                    panic!("selected IO node must really replay")
                };
                actual.statistical_evidence()
            } else {
                let rows = attribution
                    .device()
                    .commands()
                    .iter()
                    .filter(|c| {
                        c.node_index() == Some(index)
                            && c.command_phase() == DeviceCommandPhase::Compute
                    })
                    .collect::<Vec<_>>();
                let [actual] = rows.as_slice() else {
                    panic!("selected IO node must execute once")
                };
                actual.statistical_evidence()
            };
            // Adaptive capture executes the newly retained graph through the
            // physical-command path. Its Replayed physical rows intentionally
            // cannot borrow eager selected evidence; only a later direct
            // program launch binds current evidence to sealed logical rows.
            // Preserve this capture-stage Unknown rather than calling it Known.
            let actual_required = required && stage != PrimitiveExecutionStage::GraphCapture;
            assert_eq!(
                actual.is_some(),
                actual_required,
                "actual selected evidence: stage={stage:?}, node={index}, hadamard={hadamard}, token_seed={token_seed}"
            );
            if let (Some(actual), Some(expected)) = (actual, expected) {
                assert_eq!(actual, expected);
                assert_eq!(
                    actual.algorithm_work().unwrap().unwrap(),
                    expected.algorithm_work().unwrap().unwrap()
                );
                ferrum_interfaces::execution_cost::SelectedReplayAlgorithmTemplateV1::from_selected(expected,
                    projected.token_count(),projected.compute_dispatch_count(),projected.transfer_command_count()).unwrap()
                    .validate_binding(actual).unwrap();
            }
        }
    }
    #[cfg(all(feature = "metal", not(feature = "cuda"), target_os = "macos"))]
    if let Some(capture_enabled) = require_selected {
        assert_eq!(stage, PrimitiveExecutionStage::Eager);
        let capture = if capture_enabled {
            ferrum_types::SloStructuredCostCapture::HostSettledV1
        } else {
            ferrum_types::SloStructuredCostCapture::Disabled
        };
        for index in [0_u32, 4] {
            let [projected] = routes[index as usize].as_ref().unwrap().commands() else {
                panic!("one actual embedding/argmax operation command");
            };
            let actual = attribution
                .device()
                .commands()
                .iter()
                .filter(|command| {
                    command.node_index() == Some(index)
                        && command.command_phase() == DeviceCommandPhase::Compute
                })
                .collect::<Vec<_>>();
            let [actual] = actual.as_slice() else {
                panic!("embedding/argmax must execute exactly its declared command");
            };
            crate::assert_metal_algorithm_work(
                actual
                    .statistical_evidence()
                    .expect("actual primitive evidence"),
                projected
                    .statistical_evidence()
                    .expect("future primitive evidence"),
                capture,
            );
        }
    }
    let node = executable
        .execution_plan()
        .payload()
        .nodes()
        .iter()
        .find(|node| node.id().as_str() == "node.argmax")
        .unwrap();
    let output = node
        .values()
        .iter()
        .find(|binding| binding.role() == ResolvedValueRole::Output)
        .unwrap();
    let component = &output.storage().components()[0];
    let mut requests = vec![CompletionReadbackBatchRequest::new(
        (0..sessions.len() as u32)
            .map(|participant| {
                CompletionReadbackRequest::new(
                    node.id().clone(),
                    participant,
                    component.resource_id().clone(),
                    component.offset_bytes(),
                    HostTransferLayout::new(ElementType::U32, 1).unwrap(),
                )
                .unwrap()
            })
            .collect(),
    )
    .unwrap()];
    // Retain the real linear input and all output rows, including both sides
    // of a split. CPU dots from the actual source bytes independently verify
    // that the physical launch did not omit or overlap the tail.
    let mut outputs = vec![
        ("node.embedding", HIDDEN),
        ("node.residual", HIDDEN),
        ("node.linear", LINEAR_OUTPUT),
    ];
    if graph_attention {
        outputs.push(("node.attention", HIDDEN));
    }
    for (node_id, width) in outputs {
        let node = executable
            .execution_plan()
            .payload()
            .nodes()
            .iter()
            .find(|node| node.id().as_str() == node_id)
            .unwrap();
        let output = node
            .values()
            .iter()
            .find(|binding| binding.role() == ResolvedValueRole::Output)
            .unwrap();
        let component = &output.storage().components()[0];
        requests.push(
            CompletionReadbackBatchRequest::new(
                row_lengths
                    .iter()
                    .enumerate()
                    .map(|(participant, length)| {
                        CompletionReadbackRequest::new(
                            node.id().clone(),
                            participant as u32,
                            component.resource_id().clone(),
                            component.offset_bytes(),
                            HostTransferLayout::new(ElementType::F16, *length as u64 * width)
                                .unwrap(),
                        )
                        .unwrap()
                    })
                    .collect(),
            )
            .unwrap(),
        );
    }
    let CompletionReadbackBatchObservation::Terminal(receipt) = handle
        .wait_with_readback_collection(CompletionReadbackCollectionRequest::new(requests).unwrap())
        .unwrap()
    else {
        panic!("primitive wave did not reach terminal completion")
    };
    assert_eq!(
        receipt.dispositions().len(),
        sessions.len() * if graph_attention { 5 } else { 4 }
    );
    let data = receipt
        .dispositions()
        .iter()
        .map(|result| {
            let CompletionReadbackDisposition::Succeeded(output) = result else {
                panic!("real primitive readback failed: {result:?}")
            };
            (
                (
                    output.request().node_id().as_str(),
                    output.request().participant_index(),
                ),
                output.bytes(),
            )
        })
        .collect::<BTreeMap<_, _>>();
    for participant in 0..sessions.len() {
        let bytes = data[&("node.argmax", participant as u32)];
        assert_eq!(
            u32::from_le_bytes(bytes.try_into().unwrap()),
            7 + participant as u32 + token_seed
        );
    }
    if graph_attention {
        for participant in 0..sessions.len() {
            let output = half_values(data[&("node.attention", participant as u32)]);
            assert_eq!(output.len(), row_lengths[participant] * HIDDEN as usize);
            assert!(output.iter().all(|value| value.is_finite()));
            assert!(output.iter().any(|value| *value != 0.0));
        }
    }
    // Embedding may apply its inverse Hadamard transform before norm/residual.
    // This linear projection is independently Dense or Q4_K: its oracle uses
    // the actual retained residual values without applying another transform.
    // Rebuilding an unprepared family schema here would change source ordinals
    // and therefore generate different weights for the Hadamard fixture.
    let source = crate::family::Weights::new(&schema);
    let component = schema
        .components
        .iter()
        .find(|component| component.id.as_str() == "component.route_linear")
        .unwrap();
    let payload = source.component(component).unwrap();
    let weights = if linear == LinearWeight::Q4K {
        let mut values = vec![0.0; (LINEAR_OUTPUT * HIDDEN) as usize];
        ferrum_kernels::gguf_blocks::GgufBlockFormat::Q4K
            .decode(payload.bytes(), &mut values)
            .unwrap();
        values
    } else {
        half_values(payload.bytes())
    };
    for participant in 0..sessions.len() {
        let input = half_values(data[&("node.residual", participant as u32)]);
        let output = half_values(data[&("node.linear", participant as u32)]);
        assert_eq!(input.len(), row_lengths[participant] * HIDDEN as usize);
        assert_eq!(
            output.len(),
            row_lengths[participant] * LINEAR_OUTPUT as usize
        );
        for (row, values) in input.chunks_exact(HIDDEN as usize).enumerate() {
            for (column, weights) in weights.chunks_exact(HIDDEN as usize).enumerate() {
                let dot = values.iter().zip(weights).map(|(x, w)| x * w).sum::<f32>();
                let expected = f16::from_f32(dot).to_f32();
                let actual = output[row * LINEAR_OUTPUT as usize + column];
                assert!(actual.is_finite() && (actual - expected).abs() <= 0.003 + 0.003 * expected.abs(),
                    "{linear:?} participant={participant} row={row} column={column}: actual={actual} expected={expected}");
            }
        }
    }
    let result = data
        .iter()
        .map(|((name, participant), bytes)| (((*name).to_owned(), *participant), bytes.to_vec()))
        .collect();
    drop(data);
    drop((receipt, handle, attribution, identity, active));
    step.try_retire_normal().unwrap();
    for session in sessions {
        session.try_abort_if_quiescent().unwrap();
    }
    result
}

fn half_values(bytes: &[u8]) -> Vec<f32> {
    assert!(bytes.len().is_multiple_of(2));
    bytes
        .chunks_exact(2)
        .map(|bytes| f16::from_bits(u16::from_le_bytes(bytes.try_into().unwrap())).to_f32())
        .collect()
}

#[cfg(all(feature = "metal", not(feature = "cuda"), target_os = "macos"))]
#[test]
fn selected_metal_primitive_routes_match_dense_packed_and_serial_argmax_attribution() {
    compare(false, 8191, LinearWeight::Dense, [2, 3]);
}

#[cfg(all(feature = "metal", not(feature = "cuda"), target_os = "macos"))]
#[test]
fn selected_metal_primitive_routes_match_hadamard_packed_and_parallel_argmax_attribution() {
    compare(true, 8192, LinearWeight::Dense, [2, 3]);
}

#[cfg(all(feature = "metal", not(feature = "cuda"), target_os = "macos"))]
#[test]
fn selected_metal_primitive_routes_reject_kernel_parameter_overflow_before_submission() {
    let (fixture, _) = fixture(false, 8192, LinearWeight::Dense);
    let executable = fixture.compilation.executable();
    for (node, count) in [
        (0, u64::from(u32::MAX) + 1),
        (1, u64::from(u32::MAX) + 1),
        (2, u64::from(u32::MAX) / HIDDEN + 1),
    ] {
        assert!(fixture.providers.providers()[node]
            .eager_cost_route(executable, &[row(0, count, count)])
            .is_err());
    }
}

#[cfg(all(feature = "metal", not(feature = "cuda"), target_os = "macos"))]
#[test]
fn selected_metal_dense_linear_route_matches_actual_q4k_packed_split_rows() {
    compare(false, 8192, LinearWeight::Q4K, [16, 17]);
}

#[cfg(feature = "cuda")]
#[path = "primitive_cost_route/cuda_selected.rs"]
mod cuda_selected;

#[cfg(all(feature = "metal", not(feature = "cuda"), target_os = "macos"))]
#[test]
fn selected_metal_embedding_argmax_algorithm_work_matches_actual_and_capture_off_outputs() {
    for vocabulary in [17, 8192] {
        let mut control = None;
        for capture in [
            ferrum_types::SloStructuredCostCapture::Disabled,
            ferrum_types::SloStructuredCostCapture::HostSettledV1,
        ] {
            let kind = AttentionKind::GatedDelta;
            let definition = PrimitiveFamily {
                base: Family::new(kind),
                vocabulary,
                linear: LinearWeight::Dense,
                graph_attention: false,
            };
            let inputs = definition.additional_inputs();
            let profile = definition.base.profile_id();
            let prepared = TypedFamilyRegistration::new(definition)
                .prepare_with_profile(&serde_json::to_value(kind).unwrap(), &id(profile))
                .unwrap();
            let schema = prepared.weight_schema().clone();
            let fixture = Fixture::from_prepared_family_with_composition(
                kind,
                prepared,
                vec![],
                FixtureExecutionMode::Eager,
                None,
                1,
                inputs,
                crate::composition_with_capture(kind, capture),
            );
            let output = execute(
                &fixture,
                &schema,
                false,
                vocabulary,
                LinearWeight::Dense,
                &[2, 3],
                PrimitiveExecutionStage::Eager,
                Some(!capture.is_disabled()),
                7,
            );
            if let Some(expected) = &control {
                assert_eq!(
                    &output, expected,
                    "capture must preserve every original primitive output oracle"
                );
            } else {
                control = Some(output);
            }
        }
    }
}
