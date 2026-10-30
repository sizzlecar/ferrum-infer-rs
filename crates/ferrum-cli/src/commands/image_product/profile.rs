//! Diagnostic reuse of the actual product factory and image pipeline.
use super::{factory::PreparedPipeline, ImageProductSource};
use anyhow::{ensure, Context, Result};
use ferrum_engine::image_generation::{
    ImageEditRequest, ImageGenerationReceipt, ImageGenerationRequest,
};
use ferrum_models::qwen_image21::pipeline_adapters::QwenImage21RgbaOutput;
use ferrum_runtime::{DeviceTimingMode, HostMemoryClass};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, io::Write, path::PathBuf, time::Instant};

/// Explicit diagnostic inputs. Batches bypass HTTP scheduling but retain the
/// product pipeline, admission, residency, and request-prefix cache policy.
pub struct ImageProfileWorkload {
    pub request: ImageGenerationRequest,
    pub image: Option<PathBuf>,
    pub batches: Vec<usize>,
    pub warm_repetitions: usize,
    pub timing: DeviceTimingMode,
}

/// One warmup per shape followed by measured batches. CUDA events bracket
/// physical commands, including any enqueue gaps inside those intervals;
/// compound FP8 quantization/GEMM/output remains one command. Host intervals
/// overlap GPU work. These measurements are not uninstrumented HTTP throughput.
pub fn profile_image_source(
    source: ImageProductSource,
    workload: ImageProfileWorkload,
    mut writer: impl Write,
) -> Result<()> {
    ensure!(
        !workload.batches.is_empty() && workload.batches.iter().all(|batch| *batch > 0),
        "profile batches must be positive"
    );
    ensure!(
        workload.warm_repetitions > 0,
        "profile requires a measured repetition"
    );
    ensure!(
        matches!(
            workload.timing,
            DeviceTimingMode::Off | DeviceTimingMode::Kernel
        ),
        "image profiling supports Off or Kernel timing"
    );
    let image_file = workload
        .image
        .as_deref()
        .map(super::run::open_image)
        .transpose()?;
    let startup = Instant::now();
    let product = source.prepare()?;
    let (pipeline, composition) = product.backend.create(product.assets)?;
    let pipeline = match pipeline {
        PreparedPipeline::Cuda(pipeline) => pipeline,
        #[cfg(all(feature = "metal", target_os = "macos"))]
        PreparedPipeline::Metal(_) => anyhow::bail!("this diagnostic requires CUDA"),
    };
    let mut pipeline = (*pipeline).with_device_timing_mode(workload.timing)?;
    let image = image_file
        .map(|(file, length)| super::run::read_image(pipeline.host_account(), file, length))
        .transpose()?;
    serde_json::to_writer(
        &mut writer,
        &json!({
            "phase":"startup", "startup_wall_us":startup.elapsed().as_micros(),
            "composition":composition, "batches":workload.batches, "request":workload.request,
            "request_kind":if image.is_some() { "edit" } else { "generate" },
            "reference_sha256":image.as_ref().map(|image| format!("{:x}", Sha256::digest(image.bytes()))),
            "device_timing":workload.timing,
            "measurement":"CUDA physical-command events may include enqueue gaps; host wall overlaps GPU; profiling overhead included; no kernel-active or HTTP throughput claim",
        }),
    )?;
    writeln!(writer)?;
    writer.flush()?;
    for batch in workload.batches.iter().copied() {
        for repetition in 0..=workload.warm_repetitions {
            let phase = if repetition == 0 {
                "shape_warmup"
            } else {
                "warm_profile"
            };
            eprintln!("image profile: {phase}, batch {batch}, repetition {repetition}");
            let requests = vec![workload.request.clone(); batch];
            let started = Instant::now();
            let generated = if let Some(image) = &image {
                pipeline.edit_batch(
                    requests
                        .into_iter()
                        .map(|generation| ImageEditRequest {
                            generation,
                            image: image.clone(),
                        })
                        .collect(),
                )
            } else {
                pipeline.generate_batch(requests)
            };
            let outputs = generated.map_err(|error| anyhow::anyhow!("{phase}: {error}"))?;
            let generation_wall_us = started.elapsed().as_micros();
            ensure!(outputs.len() == batch, "profile output count differs");
            let mut hashes = Vec::with_capacity(outputs.len());
            let mut scratch = pipeline
                .host_account()
                .allocate_zeroed(HostMemoryClass::Output, 16 * 1024)?;
            for output in &outputs {
                let rgba = QwenImage21RgbaOutput::new(
                    pipeline.host_account(),
                    output.rgba.tensor(),
                    output.rgba.bytes(),
                )?;
                // Exercise the same finite-pixel/output adapter without retaining another image backing.
                rgba.write_frame_rgba8(0, scratch.as_mut(), &mut std::io::sink())?;
                hashes.push(format!("{:x}", Sha256::digest(output.rgba.bytes().bytes())));
            }
            let receipt = &outputs[0].receipt;
            // Preserve the completed real execution even if diagnostic aggregation
            // rejects coverage. A summary failure must not discard its raw evidence.
            let stage_summary = summarize(receipt, workload.timing);
            serde_json::to_writer(
                &mut writer,
                &json!({
                    "phase":phase, "batch_size":batch, "repetition":repetition,
                    "generation_wall_us":generation_wall_us,
                    "rgba_bf16_sha256":hashes, "finite_output_adapter_passed":true,
                    "stage_summary":stage_summary.as_ref().ok(),
                    "summary_error":stage_summary.as_ref().err().map(|error| format!("{error:#}")),
                    "generation":receipt,
                }),
            )?;
            writeln!(writer)?;
            writer.flush()?;
            stage_summary?;
        }
    }
    let shutdown = pipeline
        .shutdown()
        .map_err(|error| anyhow::anyhow!("shutdown: {error}"))?;
    serde_json::to_writer(
        &mut writer,
        &json!({"phase":"shutdown", "receipts":shutdown}),
    )?;
    writeln!(writer)?;
    writer.flush()?;
    Ok(())
}

fn summarize(
    receipt: &ImageGenerationReceipt,
    mode: DeviceTimingMode,
) -> Result<Vec<serde_json::Value>> {
    let mut summaries = Vec::new();
    let mut total_profiles = 0;
    for stage in &receipt.stages {
        let execution = &stage.execution;
        ensure!(
            execution.device_profiles.len()
                == if mode == DeviceTimingMode::Off {
                    0
                } else {
                    execution.completed_executions
                },
            "{:?}/{:?} has {} completed executions but {} profiles",
            execution.plan.stage,
            execution.plan.role,
            execution.completed_executions,
            execution.device_profiles.len()
        );
        let mut operations: BTreeMap<String, (u64, u64)> = BTreeMap::new();
        let mut paths: BTreeMap<String, u64> = BTreeMap::new();
        let mut host: BTreeMap<String, u64> = BTreeMap::new();
        let mut device_ns = 0_u64;
        let mut submission_ns = 0_u64;
        for profile in &execution.device_profiles {
            total_profiles += 1;
            let submission = profile.completion.fence_timing().device_execution();
            submission_ns = submission_ns
                .checked_add(
                    submission
                        .measured()
                        .context("submission device events are unavailable")?
                        .elapsed_ns(),
                )
                .context("submission duration overflows")?;
            let attribution = profile
                .attribution
                .as_ref()
                .context("device attribution is missing")?;
            let timing = attribution
                .terminal_timing()
                .measured()
                .context("device events are unavailable")?;
            ensure!(
                !attribution.device().commands().is_empty(),
                "device attribution is empty"
            );
            ensure!(
                timing.spans().len() == attribution.device().commands().len(),
                "event/command coverage differs"
            );
            for command in attribution.device().commands() {
                let span = timing
                    .span_for_command(command.command_index())
                    .context("command event is missing")?;
                ensure!(
                    span.command_count() == 1,
                    "cannot duplicate one physical replay interval across logical commands"
                );
                let elapsed = span
                    .measurement()
                    .elapsed_ns()
                    .context("command event is unavailable")?;
                device_ns = device_ns
                    .checked_add(elapsed)
                    .context("device duration overflows")?;
                let key = format!("{:?}/{}", command.command_phase(), command.native_op_id());
                let aggregate = operations.entry(key).or_default();
                aggregate.0 += 1;
                aggregate.1 = aggregate
                    .1
                    .checked_add(elapsed)
                    .context("operation duration overflows")?;
                *paths
                    .entry(format!("{:?}", command.execution_path()))
                    .or_default() += 1;
            }
            let host_timings = serde_json::to_value(&profile.host_wall_us)?;
            for (key, value) in host_timings
                .as_object()
                .context("host timing is not an object")?
            {
                let value = value
                    .as_u64()
                    .context("host timing is not an integer duration")?;
                let total = host.entry(key.clone()).or_default();
                *total = total
                    .checked_add(value)
                    .context("host duration overflows")?;
            }
        }
        summaries.push(json!({
            "stage":execution.plan.stage, "role":execution.plan.role, "batch_size":execution.plan.batch_size,
            "executions":execution.completed_executions, "profile_count":execution.device_profiles.len(),
            "compile_wall_us":execution.plan.compile_wall_duration_us,
            "initialization_wall_us":execution.initialization_wall_duration_us,
            "execution_wall_us":execution.execution_wall_durations_us,
            "close_wall_us":stage.close_wall_duration_us,
            "device_command_total_ns":(mode != DeviceTimingMode::Off).then_some(device_ns),
            "device_submission_total_ns":(mode != DeviceTimingMode::Off).then_some(submission_ns),
            "commands_by_phase_and_native_operation":operations.into_iter().map(|(operation,(commands,elapsed_ns))| json!({"operation":operation,"commands":commands,"elapsed_ns":elapsed_ns})).collect::<Vec<_>>(),
            "execution_paths":paths, "host_wall_us":host,
            "scope":"CUDA physical command intervals may include enqueue gaps; submission interval also includes inter-command gaps; managed D2H readback follows the submission fence and has no separate device measurement here; host wall overlaps GPU; not kernel-active cycles",
        }));
    }
    ensure!(
        mode == DeviceTimingMode::Off || total_profiles > 0,
        "diagnostic returned no device execution profiles"
    );
    Ok(summaries)
}
