//! Diagnostic reuse of the actual product factory and image pipeline.
use super::dit_input_capture::ImageDitInputCaptureOptions;
use super::final_latent_capture::ImageFinalLatentCaptureOptions;
use super::main_text_capture::ImageMainTextCaptureOptions;
use super::main_vision_capture::ImageMainVisionCaptureOptions;
#[cfg(feature = "cuda")]
use super::{
    dit_input_capture::{complete_dit_png, DitInputCaptureWriter},
    factory::{PreparedBackend, PreparedPipeline},
    final_latent_capture::{complete_png, FinalLatentCaptureWriter},
    main_text_capture::MainTextCaptureWriter,
    main_vision_capture::MainVisionCaptureWriter,
    ImageProductError, ImageProductSource,
};
#[cfg(feature = "cuda")]
use anyhow::Context;
use anyhow::{ensure, Result};
#[cfg(feature = "cuda")]
use ferrum_engine::image_generation::{
    ImageEditRequest, ImageGenerationReceipt, ImageGenerationRequest,
};
use ferrum_runtime::DeviceTimingMode;
#[cfg(feature = "cuda")]
use ferrum_runtime::HostMemoryClass;
#[cfg(feature = "cuda")]
use serde_json::json;
#[cfg(feature = "cuda")]
use sha2::{Digest, Sha256};
use std::path::PathBuf;
#[cfg(feature = "cuda")]
use std::{collections::BTreeMap, io::Write, time::Instant};

/// Explicit diagnostic inputs. Batches bypass HTTP scheduling but retain the
/// product pipeline, admission, residency, and request-prefix cache policy.
pub struct ImageProfileWorkload {
    #[cfg(feature = "cuda")]
    pub request: ImageGenerationRequest,
    pub image: Option<PathBuf>,
    pub batches: Vec<usize>,
    pub warm_repetitions: usize,
    pub timing: DeviceTimingMode,
    /// None leaves capture disabled. The optional limit covers the entire
    /// directory's cumulative raw payload, including shape warmup executions.
    pub main_vision_capture: Option<ImageMainVisionCaptureOptions>,
    /// Independent raw five-tap capture; mutually exclusive with vision capture.
    pub main_text_capture: Option<ImageMainTextCaptureOptions>,
    /// Pair the final normalized target and same-job PNG; raw budget excludes PNG.
    pub final_latent_capture: Option<ImageFinalLatentCaptureOptions>,
    /// Actual B1 DiT inputs/initial state paired with the same completed PNG.
    pub dit_input_capture: Option<ImageDitInputCaptureOptions>,
}
fn validate_workload(workload: &ImageProfileWorkload) -> Result<()> {
    ensure!(
        workload.dit_input_capture.is_none()
            || (workload.batches == [1]
                && workload.main_text_capture.is_none()
                && workload.main_vision_capture.is_none()
                && workload.final_latent_capture.is_none()),
        "DiT input capture requires B1 and excludes other captures"
    );
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
    ensure!(
        workload.main_vision_capture.is_none() || workload.image.is_some(),
        "main vision capture requires an edit image"
    );
    ensure!(
        workload.main_text_capture.is_none() || workload.image.is_some(),
        "main text capture requires an edit image"
    );
    ensure!(
        workload.main_text_capture.is_none() || workload.main_vision_capture.is_none(),
        "select either main text or main vision capture"
    );
    ensure!(
        workload.main_text_capture.is_none() || workload.batches == [1],
        "main text capture requires one B=1 profile shape"
    );
    ensure!(
        workload.final_latent_capture.is_none()
            || (workload.main_text_capture.is_none() && workload.main_vision_capture.is_none()),
        "final latent capture excludes conditioning capture"
    );
    ensure!(
        workload.final_latent_capture.is_none() || workload.batches == [1],
        "final latent capture requires one B=1 profile shape"
    );
    Ok(())
}

#[cfg(feature = "cuda")]
fn text_capture_metadata(
    options: Option<&ImageMainTextCaptureOptions>,
) -> Option<serde_json::Value> {
    options.map(|options| json!({
        "output_directory":options.output_directory,
        "maximum_cumulative_payload_bytes":options.maximum_payload_bytes,
        "budget":"five original F32 taps across all captures; excludes metadata; not RSS",
        "iterations":"all actual main text executions, including shape warmup",
        "measurement":"synchronous file writes and hashes are included in conditioning wall time; numerical diagnostic only",
    }))
}

#[cfg(feature = "cuda")]
fn final_capture_metadata(
    options: Option<&ImageFinalLatentCaptureOptions>,
) -> Option<serde_json::Value> {
    options.map(|options| json!({
        "output_directory":options.output_directory,
        "maximum_cumulative_payload_bytes":options.maximum_payload_bytes,
        "budget":"raw final latent bytes across all captures only; excludes metadata and PNG; not RSS",
        "png_limits":"ordinary PngLimits; existing managed 16 KiB output scratch",
        "iterations":"one shape warmup plus measured repetitions; each job produces its own latent/PNG pair",
        "measurement":"synchronous latent I/O is included in generation wall time; PNG I/O follows that interval; numerical diagnostic only",
    }))
}

#[cfg(feature = "cuda")]
fn dit_capture_metadata(
    options: Option<&ImageDitInputCaptureOptions>,
) -> Option<serde_json::Value> {
    options.map(|o| json!({"output_directory":o.output_directory,
        "maximum_cumulative_payload_bytes":o.maximum_payload_bytes,
        "budget":"raw stored tensors across all steps/jobs, excluding JSON/PNG; existing Host owners",
        "scope":"actual B1 six ports and initial F32 state, original time rows/routing; same-job PNG",
        "iterations":"shape warmup and measured repetitions; synchronous capture, not performance evidence"}))
}

/// One warmup per shape followed by measured batches. CUDA events bracket
/// physical commands, including any enqueue gaps inside those intervals;
/// compound FP8 quantization/GEMM/output remains one command. Host intervals
/// overlap GPU work. These measurements are not uninstrumented HTTP throughput.
#[cfg(feature = "cuda")]
pub fn profile_image_source(
    source: ImageProductSource,
    mut workload: ImageProfileWorkload,
    writer: impl Write,
) -> Result<(), ImageProductError> {
    validate_workload(&workload)?;
    let capture_metadata = workload.main_vision_capture.as_ref().map(|options| {
        json!({
            "output_directory":options.output_directory,
            "maximum_cumulative_payload_bytes":options.maximum_payload_bytes,
            "budget":"RGBA plus four F32 payloads across all captures; excludes metadata; not RSS",
            "iterations":"all actual main vision executions, including shape warmup",
            "measurement":"synchronous file writes and hashes are included in conditioning wall time; numerical diagnostic only",
        })
    });
    let image_file = workload
        .image
        .as_deref()
        .map(super::run::open_image)
        .transpose()?;
    let startup = Instant::now();
    // Profile uses the same optional trace preparation as run/serve.
    let product = source.prepare()?;
    if (workload.main_vision_capture.is_some()
        || workload.main_text_capture.is_some()
        || workload.final_latent_capture.is_some()
        || workload.dit_input_capture.is_some())
        && !matches!(&product.backend, PreparedBackend::PH3Cuda(_))
    {
        return Err(anyhow::anyhow!("diagnostic capture requires the P-H3 CUDA pipeline").into());
    }
    let capture = workload
        .main_vision_capture
        .take()
        .map(|options| -> Result<_> {
            let limit = options.maximum_payload_bytes;
            Ok((limit, MainVisionCaptureWriter::open(options)?))
        })
        .transpose()?;
    let text_capture = workload
        .main_text_capture
        .as_ref()
        .map(|options| -> Result<_> {
            Ok((
                options.maximum_payload_bytes,
                MainTextCaptureWriter::open(options.clone())?,
            ))
        })
        .transpose()?;
    let final_capture = workload
        .final_latent_capture
        .as_ref()
        .map(|options| -> Result<_> {
            Ok((
                options.maximum_payload_bytes,
                FinalLatentCaptureWriter::open(options.clone())?,
            ))
        })
        .transpose()?;
    let dit_capture = workload
        .dit_input_capture
        .as_ref()
        .map(|options| -> Result<_> {
            Ok((
                options.maximum_payload_bytes,
                DitInputCaptureWriter::open(options.clone())?,
            ))
        })
        .transpose()?;
    let (pipeline, composition) = product.backend.create(product.assets)?;
    let pipeline = pipeline.with_device_timing_mode(workload.timing)?;
    let mut pipeline = match (pipeline, capture) {
        (PreparedPipeline::PH3Cuda(pipeline), Some((limit, capture))) => PreparedPipeline::PH3Cuda(
            Box::new((*pipeline).with_main_vision_observer(limit, Box::new(capture))?),
        ),
        (pipeline, None) => pipeline,
        (_, Some(_)) => {
            return Err(
                anyhow::anyhow!("main vision capture requires the P-H3 CUDA pipeline").into(),
            )
        }
    };
    if let Some((limit, capture)) = text_capture {
        pipeline = match pipeline {
            PreparedPipeline::PH3Cuda(pipeline) => PreparedPipeline::PH3Cuda(Box::new(
                (*pipeline).with_main_text_observer(limit, Box::new(capture))?,
            )),
            _ => {
                return Err(
                    anyhow::anyhow!("main text capture requires the P-H3 CUDA pipeline").into(),
                )
            }
        };
    }
    if let Some((limit, capture)) = final_capture {
        pipeline = match pipeline {
            PreparedPipeline::PH3Cuda(pipeline) => PreparedPipeline::PH3Cuda(Box::new(
                (*pipeline).with_final_latent_observer(limit, Box::new(capture))?,
            )),
            _ => {
                return Err(
                    anyhow::anyhow!("final latent capture requires the P-H3 CUDA pipeline").into(),
                )
            }
        };
    }
    if let Some((limit, capture)) = dit_capture {
        pipeline = match pipeline {
            PreparedPipeline::PH3Cuda(pipeline) => PreparedPipeline::PH3Cuda(Box::new(
                (*pipeline).with_dit_input_observer(limit, Box::new(capture))?,
            )),
            _ => return Err(anyhow::anyhow!("DiT capture requires the P-H3 CUDA pipeline").into()),
        };
    }
    let trace = if let Some(prepared) = product.continuous_trace {
        let context = json!({
            "entrypoint":"profile", "composition":&composition,
            "batches":&workload.batches, "warm_repetitions":workload.warm_repetitions,
            "device_timing":workload.timing, "main_vision_capture":&capture_metadata,
            "main_text_capture":text_capture_metadata(workload.main_text_capture.as_ref()),
            "final_latent_capture":final_capture_metadata(workload.final_latent_capture.as_ref()),
            "dit_input_capture":dit_capture_metadata(workload.dit_input_capture.as_ref()),
        });
        match pipeline.enable_continuous_trace_artifact(&prepared, &context) {
            Ok(handle) => Some(prepared.bind(handle)),
            Err(source) => {
                if let Err(failure) = pipeline.shutdown() {
                    return Err(ImageProductError::generation_pipeline(failure, pipeline));
                }
                return Err(source.into());
            }
        }
    } else {
        None
    };
    let result = execute_profile_pipeline(
        pipeline,
        workload,
        composition,
        capture_metadata,
        image_file,
        startup,
        writer,
    );
    finish_profile_result(result, || {
        if let Some(trace) = trace {
            // Original shutdown seals a successful trace. A native failure
            // keeps its pipeline/error owners; export is an unsealed snapshot.
            trace.export_best_effort();
        }
    })
}

/// Finalize diagnostics after the original result exists. This helper neither
/// formats nor drops a failure owner, and it has no authority to seal a trace.
fn finish_profile_result<T, E>(result: Result<T, E>, export: impl FnOnce()) -> Result<T, E> {
    export();
    result
}

#[cfg(feature = "cuda")]
enum ProfileBodyFailure {
    Diagnostic(anyhow::Error),
    Generation(Box<dyn std::error::Error + Send>),
}
#[cfg(feature = "cuda")]
impl From<anyhow::Error> for ProfileBodyFailure {
    fn from(error: anyhow::Error) -> Self {
        Self::Diagnostic(error)
    }
}

#[cfg(feature = "cuda")]
fn execute_profile_pipeline(
    mut pipeline: PreparedPipeline,
    workload: ImageProfileWorkload,
    composition: serde_json::Value,
    capture_metadata: Option<serde_json::Value>,
    image_file: Option<(std::fs::File, usize)>,
    startup: Instant,
    mut writer: impl Write,
) -> Result<(), ImageProductError> {
    let result = profile_iterations(
        &mut pipeline,
        &workload,
        composition,
        capture_metadata,
        image_file,
        startup,
        &mut writer,
    );
    let diagnostic = match result {
        Ok(()) => None,
        Err(ProfileBodyFailure::Diagnostic(error)) => Some(error),
        Err(ProfileBodyFailure::Generation(source)) => {
            return Err(ImageProductError::generation_pipeline(source, pipeline));
        }
    };
    // Pure output/summary errors follow completed native work. Close the
    // original pipeline before reporting them, preserving a failed close.
    let shutdown = match pipeline.shutdown() {
        Ok(receipts) => receipts,
        Err(source) => return Err(ImageProductError::generation_pipeline(source, pipeline)),
    };
    if let Some(error) = diagnostic {
        return Err(error.into());
    }
    serde_json::to_writer(
        &mut writer,
        &json!({"phase":"shutdown", "receipts":shutdown}),
    )
    .map_err(anyhow::Error::from)?;
    writeln!(writer).map_err(anyhow::Error::from)?;
    writer.flush().map_err(anyhow::Error::from)?;
    Ok(())
}

#[cfg(feature = "cuda")]
fn profile_iterations(
    pipeline: &mut PreparedPipeline,
    workload: &ImageProfileWorkload,
    composition: serde_json::Value,
    capture_metadata: Option<serde_json::Value>,
    image_file: Option<(std::fs::File, usize)>,
    startup: Instant,
    mut writer: impl Write,
) -> Result<(), ProfileBodyFailure> {
    let image = image_file
        .map(|(file, length)| super::run::read_image(pipeline.host_account(), file, length))
        .transpose()?;
    (|| -> Result<()> {
        serde_json::to_writer(
            &mut writer,
            &json!({
                "phase":"startup", "startup_wall_us":startup.elapsed().as_micros(),
                "composition":composition, "batches":workload.batches, "request":workload.request,
                "request_kind":if image.is_some() { "edit" } else { "generate" },
                "reference_sha256":image.as_ref().map(|image| format!("{:x}", Sha256::digest(image.bytes()))),
                "device_timing":workload.timing,
                "main_vision_capture":capture_metadata,
                "main_text_capture":text_capture_metadata(workload.main_text_capture.as_ref()),
                "final_latent_capture":final_capture_metadata(workload.final_latent_capture.as_ref()),
                "dit_input_capture":dit_capture_metadata(workload.dit_input_capture.as_ref()),
                "measurement":"CUDA physical-command events may include enqueue gaps; host wall overlaps GPU; profiling overhead included; no kernel-active or HTTP throughput claim",
            }),
        )?;
        writeln!(writer)?;
        writer.flush()?;
        Ok(())
    })()?;
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
                            references: Vec::new(),
                        })
                        .collect(),
                )
            } else {
                pipeline.generate_batch(requests)
            };
            let outputs = generated.map_err(ProfileBodyFailure::Generation)?;
            let generation_wall_us = started.elapsed().as_micros();
            (|| -> Result<()> {
                ensure!(outputs.len() == batch, "profile output count differs");
                let mut hashes = Vec::with_capacity(outputs.len());
                let mut encodings = Vec::with_capacity(outputs.len());
                let mut final_captures = Vec::new();
                let mut scratch = pipeline
                    .host_account()
                    .allocate_zeroed(HostMemoryClass::Output, 16 * 1024)?;
                for output in &outputs {
                    let rgba = output.rgba.view(pipeline.host_account())?;
                    if let Some(options) = &workload.final_latent_capture {
                        let receipt = &output.receipt;
                        ensure!(
                            rgba.rgba_dimensions()
                                == [1, 4, receipt.height as usize, receipt.width as usize],
                            "profile PNG dimensions differ from the completed request"
                        );
                        final_captures.push(complete_png(
                            options,
                            receipt.job,
                            receipt.seed,
                            receipt.steps.len(),
                            ferrum_output::png::PngDimensions::new(
                                receipt.width.into(),
                                receipt.height.into(),
                            )?,
                            ferrum_output::png::PngLimits::default(),
                            |sink| {
                                rgba.write_frame_rgba8(0, scratch.as_mut(), sink)
                                    .map_err(std::io::Error::other)
                            },
                        )?);
                    } else if let Some(options) = &workload.dit_input_capture {
                        let receipt = &output.receipt;
                        ensure!(
                            rgba.rgba_dimensions()
                                == [1, 4, receipt.height as usize, receipt.width as usize],
                            "DiT capture PNG dimensions differ"
                        );
                        final_captures.push(complete_dit_png(
                            options,
                            receipt.job,
                            receipt.seed,
                            receipt.steps.len(),
                            ferrum_output::png::PngDimensions::new(
                                receipt.width.into(),
                                receipt.height.into(),
                            )?,
                            |sink| {
                                rgba.write_frame_rgba8(0, scratch.as_mut(), sink)
                                    .map_err(std::io::Error::other)
                            },
                        )?);
                    } else {
                        // Preserve the ordinary profiler's finite-pixel sink.
                        rgba.write_frame_rgba8(0, scratch.as_mut(), &mut std::io::sink())?;
                    }
                    hashes.push(format!("{:x}", Sha256::digest(output.rgba.bytes().bytes())));
                    encodings.push(match output.rgba.encoding() {
                        ferrum_models::image_pixels::ImagePixelEncoding::Bf16SignedRgbaStaged => {
                            "bf16_signed_rgba_staged"
                        }
                        ferrum_models::image_pixels::ImagePixelEncoding::F32UnitRgbRne => {
                            "f32_unit_rgb_rne"
                        }
                    });
                }
                let receipt = &outputs[0].receipt;
                // Preserve the completed real execution even if diagnostic aggregation
                // rejects coverage. A summary failure must not discard its raw evidence.
                let stage_summary = summarize(receipt, workload.timing);
                let mut report = json!({
                    "phase":phase, "batch_size":batch, "repetition":repetition,
                    "jobs":outputs.iter().map(|output| output.receipt.job).collect::<Vec<_>>(),
                    "generation_wall_us":generation_wall_us,
                    "pixel_storage_sha256":hashes, "pixel_encoding":encodings,
                    "finite_output_adapter_passed":true,
                    "stage_summary":stage_summary.as_ref().ok(),
                    "summary_error":stage_summary.as_ref().err().map(|error| format!("{error:#}")),
                    "generation":receipt,
                });
                // Preserve the existing BF16 diagnostic field only for artifacts
                // whose declared storage really is BF16 RGBA.
                if encodings.iter().all(|e| *e == "bf16_signed_rgba_staged") {
                    report["rgba_bf16_sha256"] = json!(hashes);
                }
                if workload.final_latent_capture.is_some() {
                    report["final_latent_captures"] = json!(final_captures);
                }
                if workload.dit_input_capture.is_some() {
                    report["dit_input_captures"] = json!(final_captures);
                }
                serde_json::to_writer(&mut writer, &report)?;
                writeln!(writer)?;
                writer.flush()?;
                stage_summary?;
                Ok(())
            })()?;
        }
    }
    Ok(())
}

#[cfg(feature = "cuda")]
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        cell::Cell,
        io::Write,
        sync::{
            atomic::{AtomicBool, Ordering},
            Arc,
        },
    };

    #[derive(Debug)]
    struct OriginalFailure {
        dropped: Arc<AtomicBool>,
        // The product's original error is Send-only, not required to be Sync.
        marker: Cell<u8>,
    }
    impl std::fmt::Display for OriginalFailure {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "original failure {}", self.marker.get())
        }
    }
    impl std::error::Error for OriginalFailure {}
    impl Drop for OriginalFailure {
        fn drop(&mut self) {
            self.dropped.store(true, Ordering::Release);
        }
    }

    // These qualify final-result ownership and actual sink failure behavior.
    // They do not construct a PH3 trace collector or prove GPU shutdown/seal.
    #[test]
    fn failed_export_preserves_the_original_send_only_error_until_caller_drop() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("read-only-snapshot.jsonl");
        std::fs::write(&path, b"original artifact\n").unwrap();
        let dropped = Arc::new(AtomicBool::new(false));
        let source: Box<dyn std::error::Error + Send> = Box::new(OriginalFailure {
            dropped: Arc::clone(&dropped),
            marker: Cell::new(71),
        });
        let original_pointer = source.as_ref() as *const dyn std::error::Error;
        let result: Result<(), Box<dyn std::error::Error + Send>> = Err(source);
        let returned = finish_profile_result(result, || {
            assert!(!dropped.load(Ordering::Acquire));
            let mut read_only = std::fs::File::open(&path).unwrap();
            // Like export_best_effort, sink failure must not replace the
            // already-obtained native result with an I/O error or its text.
            assert!(read_only.write_all(b"replacement\n").is_err());
        })
        .unwrap_err();
        assert!(std::ptr::eq(
            original_pointer,
            returned.as_ref() as *const dyn std::error::Error
        ));
        assert_eq!(
            returned
                .downcast_ref::<OriginalFailure>()
                .unwrap()
                .marker
                .get(),
            71
        );
        assert!(!dropped.load(Ordering::Acquire));
        assert_eq!(std::fs::read(path).unwrap(), b"original artifact\n");
        drop(returned);
        assert!(dropped.load(Ordering::Acquire));
    }

    #[test]
    fn success_remains_owned_while_final_export_writes_its_artifact() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("snapshot.txt");
        let dropped = Arc::new(AtomicBool::new(false));
        let value = OriginalFailure {
            dropped: Arc::clone(&dropped),
            marker: Cell::new(23),
        };
        let result: Result<OriginalFailure, ()> = Ok(value);
        let returned = finish_profile_result(result, || {
            assert!(!dropped.load(Ordering::Acquire));
            std::fs::write(&path, b"exported after result\n").unwrap();
        })
        .unwrap();
        assert_eq!(std::fs::read(path).unwrap(), b"exported after result\n");
        assert_eq!(returned.marker.get(), 23);
        assert!(!dropped.load(Ordering::Acquire));
        drop(returned);
        assert!(dropped.load(Ordering::Acquire));
    }

    #[test]
    fn profile_capture_preflight_requires_an_image_without_opening_artifacts() {
        let root = tempfile::tempdir().unwrap();
        let capture_path = root.path().join("captures");
        let mut workload = ImageProfileWorkload {
            #[cfg(feature = "cuda")]
            request: ImageGenerationRequest {
                prompt: "A ceramic teapot".into(),
                width: 1024,
                height: 1024,
                seed: 0,
                n: 1,
            },
            image: None,
            batches: vec![1],
            warm_repetitions: 1,
            timing: DeviceTimingMode::Off,
            main_vision_capture: None,
            main_text_capture: None,
            final_latent_capture: None,
            dit_input_capture: None,
        };
        validate_workload(&workload).unwrap();
        workload.main_vision_capture = Some(ImageMainVisionCaptureOptions {
            output_directory: capture_path.clone(),
            maximum_payload_bytes: std::num::NonZeroUsize::new(72).unwrap(),
        });
        assert!(validate_workload(&workload).is_err());
        assert!(!capture_path.exists());
        workload.image = Some(root.path().join("not-opened.png"));
        validate_workload(&workload).unwrap();
        assert!(!capture_path.exists());
        workload.main_text_capture = Some(ImageMainTextCaptureOptions {
            output_directory: capture_path.clone(),
            maximum_payload_bytes: std::num::NonZeroUsize::new(134217728).unwrap(),
        });
        assert!(
            validate_workload(&workload).is_err(),
            "capture modes conflict"
        );
        workload.main_vision_capture = None;
        validate_workload(&workload).unwrap();
        workload.batches = vec![2];
        assert!(validate_workload(&workload).is_err());
        workload.batches = vec![1];
        workload.image = None;
        assert!(validate_workload(&workload).is_err());
        workload.final_latent_capture = Some(ImageFinalLatentCaptureOptions {
            output_directory: capture_path.clone(),
            maximum_payload_bytes: std::num::NonZeroUsize::new(96).unwrap(),
        });
        assert!(validate_workload(&workload).is_err());
        workload.main_text_capture = None;
        validate_workload(&workload).unwrap();
        workload.batches = vec![2];
        assert!(validate_workload(&workload).is_err());
        workload.batches = vec![1];
        workload.image = Some(root.path().join("not-opened.png"));
        workload.main_vision_capture = Some(ImageMainVisionCaptureOptions {
            output_directory: capture_path.clone(),
            maximum_payload_bytes: std::num::NonZeroUsize::new(72).unwrap(),
        });
        assert!(validate_workload(&workload).is_err());
        workload.dit_input_capture = Some(ImageDitInputCaptureOptions {
            output_directory: capture_path.clone(),
            maximum_payload_bytes: std::num::NonZeroUsize::new(536870912).unwrap(),
        });
        assert!(validate_workload(&workload).is_err());
        workload.main_vision_capture = None;
        workload.final_latent_capture = None;
        validate_workload(&workload).unwrap();
        workload.batches = vec![2];
        assert!(validate_workload(&workload).is_err());
        assert!(!capture_path.exists());
    }
}
