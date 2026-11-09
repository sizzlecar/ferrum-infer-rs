//! The production full-generation step, with real stored DiT weights and the
//! Reference runtime. Noise and VAE are outside this boundary; these tests do
//! not claim that the image service admits requests into an active cohort.
use super::*;
use crate::image_generation::{host_budget, tests::adapter};
use ferrum_kernels::backend::reference::ReferenceDeviceRuntime;
use ferrum_models::qwen_image21::ExpectedWeights;
use half::bf16;

pub(in crate::image_generation) fn pipeline() -> (
    QwenImage21Pipeline<ReferenceDeviceRuntime>,
    tempfile::TempDir,
) {
    let (mut pipeline, directory) = adapter::pipeline(2);
    pipeline.execution.host = HostMemoryAccount::with_allocation_alignment(
        HostMemoryLimits::new(
            1 << 20,
            HostMemoryClassLimits {
                input_preprocess_bytes: 1 << 18,
                output_bytes: 1 << 18,
                transfer_staging_bytes: 1 << 18,
                weight_backing_bytes: 0,
            },
        )
        .unwrap(),
        64,
    )
    .unwrap();
    let inventory = ExpectedWeights::from_config(
        &pipeline.execution.sources.assets.transformer,
        StructureLimits::default(),
    )
    .unwrap();
    let tensors = inventory
        .tensors()
        .values()
        .map(|tensor| {
            let shape = tensor
                .dimensions()
                .iter()
                .map(|&value| value as usize)
                .collect::<Vec<_>>();
            let bytes = (0..shape.iter().product())
                .flat_map(|index| {
                    bf16::from_f32(weight(tensor.external_name(), &shape, index))
                        .to_bits()
                        .to_le_bytes()
                })
                .collect::<Vec<_>>();
            (tensor.external_name().to_owned(), shape, bytes)
        })
        .collect::<Vec<_>>();
    let path = directory.path().join("nonzero-dit.safetensors");
    safetensors::tensor::serialize_to_file(
        tensors.iter().map(|(name, shape, bytes)| {
            (
                name.as_str(),
                safetensors::tensor::TensorView::new(
                    safetensors::Dtype::BF16,
                    shape.clone(),
                    bytes,
                )
                .unwrap(),
            )
        }),
        &None,
        &path,
    )
    .unwrap();
    Arc::get_mut(&mut pipeline.execution.sources.assets)
        .unwrap()
        .sources
        .transformer = Arc::new(
        SafetensorsArchive::open_single_file(directory.path(), "nonzero-dit.safetensors").unwrap(),
    );
    (pipeline, directory)
}

fn weight(name: &str, shape: &[usize], index: usize) -> f32 {
    if name.ends_with("norm_q.weight") || name.ends_with("norm_k.weight") {
        return 1.0;
    }
    if name == "txt_in.text_norm.weight" {
        return 0.0;
    }
    let row = index / shape[1];
    let column = index % shape[1];
    let diagonal = if row == column { 1.0 } else { 0.0 };
    match name {
        "img_in.weight" | "proj_out.weight" => diagonal,
        "txt_in.in_layer.weight" => 32.0 * diagonal,
        "txt_in.out_layer.weight" => diagonal / 16.0,
        "time_text_embed.timestep_embedder.linear_1.weight" => {
            if row == 0 && column == 0 {
                8.0
            } else {
                0.0
            }
        }
        "time_text_embed.timestep_embedder.linear_2.weight" => {
            if row == 0 && column == 0 {
                1.0
            } else {
                0.0
            }
        }
        "modulation.1.weight" => {
            if column == 0 {
                let sign = if row % 2 == 0 { 1.0 } else { -1.0 };
                match row / 8 {
                    0 => sign / 64.0,
                    1 => 1.0 / 16.0,
                    2 => -sign / 128.0,
                    _ => 0.25,
                }
            } else {
                0.0
            }
        }
        "norm_out.linear.weight" => 0.0,
        _ if name.ends_with("attn.to_q.weight") || name.ends_with("attn.to_k.weight") => 0.0,
        _ if name.ends_with("attn.to_v.weight") => 0.5 * diagonal,
        _ if name.ends_with("attn.to_out.0.weight") => {
            if column == (row + 1) % 8 {
                1.0
            } else {
                0.0
            }
        }
        _ if name.ends_with("img_mlp.gate_layer.weight") => diagonal,
        _ if name.ends_with("img_mlp.proj.weight") => 0.5 * diagonal,
        _ if name.ends_with("img_mlp.out.weight") => {
            if column == (row + 2) % 8 {
                0.25
            } else {
                0.0
            }
        }
        _ => panic!("unexpected stored weight {name}"),
    }
}

pub(super) fn config(
    pipeline: &QwenImage21Pipeline<ReferenceDeviceRuntime>,
    rows: &[usize],
) -> QwenImage21TransformerConfig {
    QwenImage21TransformerConfig::new(
        pipeline.execution.sources.assets.transformer.clone(),
        QwenImage21T2iLayout::new(
            2,
            2,
            2,
            rows.iter()
                .map(|&row| if row == 0 { 2 } else { 1 })
                .collect(),
        )
        .unwrap(),
    )
    .unwrap()
}

pub(super) fn input(
    host: &HostMemoryAccount,
    config: &QwenImage21TransformerConfig,
    name: &str,
    rows: &[usize],
) -> ManagedHostTensor {
    let spec = config
        .input_tensor_specs()
        .unwrap()
        .remove(&id::<ProgramValueId>(name).unwrap())
        .unwrap();
    let spec = ResolvedTensorSpec::new(spec.dimensions, spec.element_type, spec.layout).unwrap();
    let len = spec.minimum_storage_bytes().unwrap() as usize;
    let stride = len / rows.len();
    let mut bytes = host
        .allocate_zeroed(HostMemoryClass::InputPreprocess, len)
        .unwrap();
    for (destination, &row) in bytes.as_mut().chunks_exact_mut(stride).zip(rows) {
        for (index, value) in destination.chunks_exact_mut(2).enumerate() {
            let number = if name == QwenImage21TransformerFamily::TEXT_ID {
                if (index + row) % 3 == 0 {
                    0.5
                } else {
                    -0.5
                }
            } else {
                ((index * 3 + row * 7) % 19) as f32 * 0.125 - 1.125
            };
            value.copy_from_slice(&bf16::from_f32(number).to_bits().to_le_bytes());
        }
    }
    ManagedHostTensor::new(id(name).unwrap(), spec, bytes.freeze()).unwrap()
}

fn state(
    pipeline: &QwenImage21Pipeline<ReferenceDeviceRuntime>,
    config: QwenImage21TransformerConfig,
    rows: &[usize],
    indices: &[usize],
) -> FullGenerationDenoising {
    let host = pipeline.execution_host();
    let sample = input(
        host,
        &config,
        QwenImage21TransformerFamily::LATENTS_ID,
        rows,
    );
    let text = input(host, &config, QwenImage21TransformerFamily::TEXT_ID, rows);
    let frequencies = config.managed_rotary_frequencies(host).unwrap();
    FullGenerationDenoising::new(
        host,
        41,
        indices,
        config,
        pipeline.execution.sources.assets.schedule.solver(),
        pipeline.profiles.euler,
        sample,
        text,
        frequencies,
    )
    .unwrap()
}

fn grant(
    pipeline: &QwenImage21Pipeline<ReferenceDeviceRuntime>,
    stage: &PreparedStage<ReferenceDeviceRuntime>,
    short: usize,
) -> HostMemoryScope {
    let ports = host_budget::ports(stage).unwrap();
    let input = host_budget::full_generation_input_peak(
        &pipeline.execution.host,
        ports,
        stage.plan.batch_size,
    )
    .unwrap();
    pipeline
        .execution
        .host
        .reserve_scope(
            &host_budget::ExecutionHostBudget {
                input: input - short,
                output: host_budget::sum_ports(&pipeline.execution.host, ports.outputs().iter())
                    .unwrap(),
                staging: host_budget::staging_bytes(
                    &pipeline.execution.host,
                    stage,
                    pipeline.execution.config.static_initialization,
                )
                .unwrap(),
                weights: 0,
            }
            .claims(),
        )
        .unwrap()
}

fn snapshot(state: &FullGenerationDenoising) -> Vec<(RequestId, usize, Vec<u8>)> {
    state
        .requests
        .iter()
        .map(|request| {
            (
                request.id,
                request.solver.step_index(),
                request.sample.bytes().bytes().to_vec(),
            )
        })
        .collect()
}

pub(in crate::image_generation) fn trajectory(
    rows: &[usize],
    indices: &[usize],
) -> Vec<Vec<Vec<u8>>> {
    let (mut pipeline, _directory) = pipeline();
    let config = config(&pipeline, rows);
    let prepared = pipeline.prepare_denoiser(&config, rows.len()).unwrap();
    let scope = grant(&pipeline, &prepared, 0);
    pipeline.execution.execution_host = Some(scope.account());
    let mut state = state(&pipeline, config, rows, indices);
    assert_eq!(
        state
            .requests
            .iter()
            .map(|request| request.id)
            .collect::<Vec<_>>(),
        indices
            .iter()
            .map(|&caller_index| RequestId {
                job: 41,
                caller_index
            })
            .collect::<Vec<_>>()
    );
    let mut stages = Vec::new();
    let mut steps = Vec::new();
    let mut active = pipeline
        .start_stage(prepared, 41, &mut stages)
        .unwrap_or_else(|error| panic!("{error}"));
    let initial = snapshot(&state);
    let mut history = vec![initial
        .iter()
        .map(|(_, _, sample)| sample.clone())
        .collect::<Vec<_>>()];
    while !state.is_complete() {
        active = state
            .step(&mut pipeline, active, &mut steps)
            .unwrap_or_else(|error| panic!("{error}"));
        let current = snapshot(&state);
        assert!(current.iter().all(|(_, index, _)| *index == steps.len()));
        assert_eq!(active.receipt.completed_executions, steps.len());
        history.push(current.into_iter().map(|(_, _, sample)| sample).collect());
    }
    assert_ne!(
        history[0],
        *history.last().unwrap(),
        "nonzero actual DiT must advance the latent"
    );
    assert_eq!(
        steps.len(),
        pipeline
            .execution
            .sources
            .assets
            .schedule
            .evaluation_count()
    );
    // Calling a completed state must retain the real stage in a fatal
    // Processing failure and must not submit another model evaluation.
    let before = pipeline.execution.runtime.snapshot().submissions;
    let failure = state
        .step(&mut pipeline, active, &mut steps)
        .err()
        .expect("completed state rejects another evaluation");
    let ImageGenerationFailure::Processing {
        active: Some(active),
        ..
    } = failure
    else {
        panic!("completed state must retain its real stage owner")
    };
    assert_eq!(pipeline.execution.runtime.snapshot().submissions, before);
    pipeline
        .finish_stage(active, &mut stages)
        .unwrap_or_else(|error| panic!("{error}"));
    let final_sample = state.finish(pipeline.execution_host()).unwrap();
    assert_eq!(
        final_sample.bytes().bytes(),
        history.last().unwrap().concat()
    );
    pipeline.execution.execution_host = None;
    drop(scope);
    assert!(pipeline.execution.host.usage().unwrap().committed_bytes() > 0);
    drop(final_sample);
    pipeline
        .shutdown()
        .unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(
        pipeline.execution.host.usage().unwrap().committed_bytes(),
        0
    );
    assert_eq!(pipeline.execution.runtime.snapshot().live_allocations, 0);
    history
}

#[test]
fn production_step_matches_independent_b1_trajectories_and_preserves_caller_identity() {
    let first = trajectory(&[0], &[0]);
    let second = trajectory(&[1], &[0]);
    // Canonical row zero belongs to caller one, including after each commit.
    let batch = trajectory(&[1, 0], &[1, 0]);
    for ((batch, first), second) in batch.iter().zip(&first).zip(&second) {
        assert_eq!(batch[0], second[0]);
        assert_eq!(batch[1], first[0]);
    }
}

#[test]
fn last_successor_admission_failure_preserves_every_cursor_sample_and_active_stage() {
    let (mut pipeline, _directory) = pipeline();
    let config = config(&pipeline, &[1, 0]);
    let prepared = pipeline.prepare_denoiser(&config, 2).unwrap();
    let ports = host_budget::ports(&prepared).unwrap();
    let latent = ports
        .inputs()
        .iter()
        .find(|port| port.value_id().as_str() == QwenImage21TransformerFamily::LATENTS_ID)
        .unwrap();
    let logical = latent.tensor().minimum_storage_bytes().unwrap() as usize;
    let per_request = pipeline.execution.host.backing_bytes(logical / 2).unwrap();
    assert!(
        2 * per_request > pipeline.execution.host.backing_bytes(logical).unwrap(),
        "each request backing needs its own alignment padding"
    );
    let scope = grant(&pipeline, &prepared, 1);
    pipeline.execution.execution_host = Some(scope.account());
    let mut state = state(&pipeline, config, &[1, 0], &[1, 0]);
    let before = snapshot(&state);
    let host_before = pipeline.execution.host.usage().unwrap().allocated_bytes();
    let mut stages = Vec::new();
    let mut steps = Vec::new();
    let active = pipeline
        .start_stage(prepared, 41, &mut stages)
        .unwrap_or_else(|error| panic!("{error}"));
    let submissions = pipeline.execution.runtime.snapshot().submissions;
    let failure = state
        .step(&mut pipeline, active, &mut steps)
        .err()
        .expect("the final successor exceeds its real Host grant by one byte");
    let ImageGenerationFailure::Processing {
        source,
        active: Some(active),
    } = failure
    else {
        panic!("a completed DiT followed by Host failure must retain its original active stage")
    };
    // prepare_advance preserves its typed Euler error around Host admission;
    // anyhow's top-level downcast does not unwrap that domain error. Requiring
    // this wrapper also proves the failure occurred in successor preparation,
    // rather than earlier while gathering inputs or constructing time features.
    let Some(ferrum_diffusion::Bf16EulerError::Host(HostMemoryError::ScopeCapacity {
        class: HostMemoryClass::InputPreprocess,
        requested,
        available,
    })) = source.downcast_ref::<ferrum_diffusion::Bf16EulerError>()
    else {
        panic!("expected successor Host admission failure; error chain: {source:#}")
    };
    assert_eq!(*requested, per_request);
    assert_eq!(
        *available,
        per_request - 1,
        "the first successor must have spent its admitted credit before the final row fails"
    );
    assert!(pipeline.execution.runtime.snapshot().submissions > submissions);
    assert_eq!(active.receipt.completed_executions, 1);
    assert_eq!(snapshot(&state), before);
    assert!(steps.is_empty());
    assert_eq!(
        pipeline.execution.host.usage().unwrap().allocated_bytes(),
        host_before,
        "all private pending successors and the gathered wave must be released"
    );
    active.close().unwrap_or_else(|error| panic!("{error}"));
    drop(state);
    pipeline.execution.execution_host = None;
    drop(scope);
    pipeline
        .shutdown()
        .unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(
        pipeline.execution.host.usage().unwrap().committed_bytes(),
        0
    );
    assert_eq!(pipeline.execution.runtime.snapshot().live_allocations, 0);
}
