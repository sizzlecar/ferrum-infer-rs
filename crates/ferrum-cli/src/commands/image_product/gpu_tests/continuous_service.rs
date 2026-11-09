//! Opt-in functional evidence from ordinary validated H3 assets and the real
//! product factory/worker. A polling deadline is a test bound, not a latency
//! SLO. Missing mixed progress is a failure, including a too-fast workload.
use super::*;
use crate::commands::image_product::{
    factory::{PreparedAssets, PreparedBackend},
    p_h3::{PH3ImageProductConfig, PH3ImagePromptExpansionConfig},
    ImageProductSource,
};
use ferrum_engine::image_generation::{
    continuous_trace::{ImageContinuousTraceHandle, ImageContinuousTraceLimits},
    service::{ImageJob, ImageService, ImageServiceConfig, ImageServiceErrorKind},
};
use ferrum_types::ImageRequestCorrelation;
use std::time::{Duration, Instant};
use uuid::Uuid;

const DEADLINE: Duration = Duration::from_secs(60 * 60);
const POLL: Duration = Duration::from_millis(10);
const DEFAULT_TRACE_BYTES: usize = 2 << 20;

#[derive(Clone, Copy, PartialEq, Eq)]
enum FirstExit {
    Cancelled,
    Completed,
}

impl FirstExit {
    fn outcome(self) -> &'static str {
        match self {
            Self::Cancelled => "cancelled",
            Self::Completed => "completed",
        }
    }
}

fn snapshot(handle: &ImageContinuousTraceHandle) -> Result<Vec<Value>> {
    let mut bytes = Vec::new();
    let status = handle.write_jsonl(&mut bytes)?;
    ensure!(!status.truncated, "bounded trace lost observations");
    bytes
        .split(|byte| *byte == b'\n')
        .filter(|line| !line.is_empty())
        .map(|line| serde_json::from_slice(line).map_err(Into::into))
        .collect()
}

fn linked_request(records: &[Value], id: Uuid) -> Result<Option<&str>> {
    let mut links = records.iter().filter(|record| {
        record["event"]["kind"] == "request_linked"
            && record["event"]["request_id"] == id.to_string()
    });
    let identity = links
        .next()
        .map(|record| {
            record["event"]["request"]
                .as_str()
                .context("request link has no actual model identity")
        })
        .transpose()?;
    ensure!(
        links.next().is_none(),
        "server request linked more than once"
    );
    Ok(identity)
}

fn member_step(record: &Value, identity: &str) -> Option<u64> {
    record["event"]["members"]
        .as_array()?
        .iter()
        .find(|member| member["request"] == identity)?["step_index"]
        .as_u64()
}

fn checked_commit<'a>(records: &'a [Value], record: &'a Value) -> Result<&'a Value> {
    let event = &record["event"];
    let sequence = event["start_sequence"]
        .as_u64()
        .context("committed step has no successful original start observation")?;
    let start = records
        .iter()
        .find(|candidate| candidate["sequence"] == sequence)
        .context("committed step's start is absent")?;
    ensure!(
        start["event"]["kind"] == "packed_step_started"
            && start["event"]["cohort_job"] == event["cohort_job"]
            && start["event"]["plan_hash"] == event["plan_hash"]
            && start["event"]["program_fingerprint"] == event["program_fingerprint"]
            && start["event_ns"].as_u64().context("start clock absent")?
                <= record["event_ns"].as_u64().context("commit clock absent")?,
        "committed step differs from its actual matched start"
    );
    let members = event["members"]
        .as_array()
        .context("missing step members")?;
    let started = start["event"]["members"]
        .as_array()
        .context("missing start members")?;
    ensure!(
        members.len() == started.len()
            && members.iter().zip(started).all(|(member, started)| {
                member["request"] == started["request"]
                    && member["step_index"] == started["step_index"]
            }),
        "start/commit member order or solver progress differs"
    );
    Ok(record)
}

async fn observe(
    handle: &ImageContinuousTraceHandle,
    service: &ImageService,
    deadline: Instant,
    label: &str,
    ids: &[Uuid],
    accept: impl Fn(&[Value]) -> Result<Option<Value>>,
) -> Result<Value> {
    loop {
        let records = snapshot(handle)?;
        if let Some(event) = accept(&records)? {
            return Ok(event);
        }
        ensure!(
            !records.iter().any(|record| {
                record["event"]["kind"] == "request_service_finished"
                    && ids
                        .iter()
                        .any(|id| record["event"]["request_id"] == id.to_string())
            }),
            "{label} was not observed before a required request terminated"
        );
        ensure!(
            Instant::now() < deadline,
            "deadline waiting for {label}; actual service {:?}",
            service.snapshot()
        );
        tokio::time::sleep(POLL).await;
    }
}

fn mixed(records: &[Value], older: Uuid, newer: Uuid) -> Result<Option<Value>> {
    let (Some(older), Some(newer)) = (
        linked_request(records, older)?,
        linked_request(records, newer)?,
    ) else {
        return Ok(None);
    };
    for record in records {
        if record["event"]["kind"] != "packed_step_committed" {
            continue;
        }
        if let (Some(a), Some(b)) = (member_step(record, older), member_step(record, newer)) {
            if a > b {
                checked_commit(records, record)?;
                ensure!(
                    record["event"]["packed_request_count"]
                        .as_u64()
                        .unwrap_or(0)
                        >= 2,
                    "mixed member list lacks an actual packed forward"
                );
                return Ok(Some(record.clone()));
            }
        }
    }
    Ok(None)
}

fn terminal_evidence(records: &[Value], trajectory: &Value, first_exit: FirstExit) -> Result<()> {
    for name in ["a", "b", "c", "reference_b"] {
        if name == "reference_b" && first_exit == FirstExit::Cancelled {
            continue;
        }
        let outcome = if name == "a" {
            first_exit.outcome()
        } else {
            "completed"
        };
        let output = match name {
            "a" if first_exit == FirstExit::Completed => Some(&trajectory["first_output"]),
            "b" => Some(&trajectory["outputs"][0]),
            "c" => Some(&trajectory["outputs"][1]),
            "reference_b" => Some(&trajectory["reference_b"]),
            _ => None,
        };
        let correlation_value = if name == "reference_b" {
            &trajectory["reference_b"]["correlation"]
        } else {
            &trajectory["correlations"][name]
        };
        let correlation: ImageRequestCorrelation =
            serde_json::from_value(correlation_value.clone())?;
        let identity = linked_request(records, correlation.request_id)?
            .context("delivered/cancelled request has no original model identity")?;
        let terminal = records
            .iter()
            .filter(|record| {
                record["event"]["kind"] == "request_service_finished"
                    && record["event"]["request_id"] == correlation.request_id.to_string()
            })
            .collect::<Vec<_>>();
        let exits = records
            .iter()
            .filter(|record| {
                record["event"]["kind"] == "request_exited_at_fence"
                    && record["event"]["request"] == identity
            })
            .collect::<Vec<_>>();
        ensure!(
            terminal.len() == 1 && terminal[0]["event"]["outcome"] == outcome,
            "{name} lacks exactly one actual service terminal outcome"
        );
        ensure!(
            exits.len() == 1 && exits[0]["event"]["reason"] == outcome,
            "{name} lacks exactly one actual completed-fence exit"
        );
        let exit_sequence = exits[0]["sequence"]
            .as_u64()
            .context("exit sequence absent")?;
        ensure!(
            records
                .iter()
                .filter(|record| record["event"]["kind"] == "packed_step_committed"
                    && member_step(record, identity).is_some())
                .all(|record| record["sequence"]
                    .as_u64()
                    .is_some_and(|sequence| sequence < exit_sequence)),
            "{name} appeared in a committed forward after its actual exit"
        );
        let committed = records
            .iter()
            .filter(|record| record["event"]["kind"] == "packed_step_committed")
            .filter_map(|record| member_step(record, identity))
            .collect::<Vec<_>>();
        ensure!(
            !committed.is_empty() && committed.iter().copied().eq(0..committed.len() as u64),
            "{name} actual committed solver indices are missing or repeated"
        );
        if name == "reference_b" {
            ensure!(
                records
                    .iter()
                    .filter(|record| record["event"]["kind"] == "packed_step_committed"
                        && member_step(record, identity).is_some())
                    .all(|record| record["event"]["packed_request_count"] == 1
                        && record["event"]["members"]
                            .as_array()
                            .is_some_and(|members| members.len() == 1)),
                "B1 reference must execute every real forward alone at unchanged service capacity"
            );
        }
        if let Some(output) = output {
            ensure!(
                output["generation"]["steps"]
                    .as_array()
                    .context("output has no actual step receipts")?
                    .len()
                    == committed.len(),
                "{name} output receipts differ from actual committed steps"
            );
            let initialized = records
                .iter()
                .find(|record| {
                    record["event"]["kind"] == "request_initialized"
                        && record["event"]["request"] == identity
                })
                .context("actual initialization event absent")?;
            ensure!(
                initialized["event"]["job"] == output["generation"]["job"],
                "{name} output receipt job differs from its actual model identity"
            );
        }
        if name == "a" && first_exit == FirstExit::Completed {
            let c: ImageRequestCorrelation =
                serde_json::from_value(trajectory["correlations"]["c"].clone())?;
            let c_identity = linked_request(records, c.request_id)?.context("C identity absent")?;
            let c_first = records
                .iter()
                .find(|record| {
                    record["event"]["kind"] == "packed_step_committed"
                        && member_step(record, c_identity).is_some()
                })
                .context("C never entered a real forward")?;
            let terminal_sequence = terminal[0]["sequence"]
                .as_u64()
                .context("A terminal sequence absent")?;
            let c_sequence = c_first["sequence"]
                .as_u64()
                .context("C commit sequence absent")?;
            ensure!(
                exit_sequence < terminal_sequence && terminal_sequence < c_sequence,
                "C must enter after A's actual completed-fence exit and successful delivery"
            );
        }
    }
    Ok(())
}

fn write_service_output(
    host: &HostMemoryAccount,
    directory: &Path,
    name: &str,
    correlation: ImageRequestCorrelation,
    request: &ImageGenerationRequest,
    output: &ImageGenerationOutput,
) -> Result<Value> {
    ensure!(
        output.receipt.seed == request.seed
            && output.receipt.width == request.width
            && output.receipt.height == request.height,
        "returned receipt differs from the original request"
    );
    let pixels = output.rgba.view(host)?;
    ensure!(
        pixels.rgba_dimensions() == [1, 4, request.height as usize, request.width as usize],
        "pixel view is not the submitted single image"
    );
    let mut scratch = host.allocate_zeroed(HostMemoryClass::Output, 16 * 1024)?;
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(directory.join(format!("{name}.png")))?;
    let mut writer = HashedWriter {
        writer: file,
        sha256: Sha256::new(),
    };
    let mut rgba8_sha256 = None;
    let png = write_rgba8_png(
        &mut writer,
        PngDimensions::new(request.width.into(), request.height.into())?,
        PngLimits::default(),
        |sink| {
            let mut pixel_writer = HashedWriter {
                writer: sink,
                sha256: Sha256::new(),
            };
            pixels
                .write_frame_rgba8(0, scratch.as_mut(), &mut pixel_writer)
                .map_err(io::Error::other)?;
            rgba8_sha256 = Some(format!("{:x}", pixel_writer.sha256.finalize()));
            Ok(())
        },
    )?;
    writer.writer.sync_all()?;
    let record = json!({
        "correlation":correlation, "generation":output.receipt,
        "submitted_prompt_sha256":format!("{:x}",Sha256::digest(request.prompt.as_bytes())),
        "image_pixels":{"encoding":format!("{:?}",output.rgba.encoding()),
            "shape":output.rgba.tensor().dimensions(),"bytes":output.rgba.bytes().len(),
            "finite":true,"sha256":format!("{:x}",Sha256::digest(output.rgba.bytes().bytes())),
            "rgba8_sha256":rgba8_sha256.context("complete RGBA8 stream was not observed")?},
        "png":{"file":format!("{name}.png"),"encoded_bytes":png.encoded_bytes(),
            "color_type":"rgba8","sha256":format!("{:x}",writer.sha256.finalize())}
    });
    write_json_new(&directory.join(format!("{name}.json")), &record)?;
    Ok(record)
}

async fn await_output(job: ImageJob, deadline: Instant) -> Result<ImageGenerationOutput> {
    tokio::time::timeout_at(tokio::time::Instant::from_std(deadline), job.wait())
        .await
        .context("deadline awaiting actual image delivery")?
        .map_err(Into::into)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "requires ordinary H3 CUDA assets/config, sufficient true B2 capacity and an exclusive GPU window"]
async fn cuda_official_h3_service_mixed_progress_cancel_refill_and_retirement(
) -> Result<(), ImageProductError> {
    service_trajectory(FirstExit::Cancelled).await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "requires H3 CUDA assets, explicit expansion Disabled, fixed expanded A/B/C briefs and an exclusive GPU window"]
async fn cuda_official_h3_service_natural_completion_refill_preserves_b1_pixels(
) -> Result<(), ImageProductError> {
    service_trajectory(FirstExit::Completed).await
}

async fn service_trajectory(first_exit: FirstExit) -> Result<(), ImageProductError> {
    let config_path = required_path("FERRUM_TEST_IMAGE_PRODUCT_CONFIG")?;
    let directory = required_path("FERRUM_TEST_IMAGE_ARTIFACT_DIR")?;
    if first_exit == FirstExit::Completed {
        (|| -> Result<()> {
            let config = PH3ImageProductConfig::load(&config_path)?;
            ensure!(matches!(config.prompt_expansion, PH3ImagePromptExpansionConfig::Disabled),
                "natural refill oracle requires explicitly disabled expansion and fixed expanded briefs");
            required_path("FERRUM_TEST_IMAGE_REQUESTS_JSON")?;
            Ok(())
        })()?;
    }
    // Refuse an old evidence directory before mapping assets or creating CUDA.
    fs::create_dir(&directory).map_err(anyhow::Error::from)?;
    let product = ImageProductSource::ExplicitConfig(config_path.clone()).prepare()?;
    let (requests, service_config) = (|| -> Result<_> {
        ensure!(
            matches!(&product.backend, PreparedBackend::PH3Cuda(_)),
            "this test requires the typed H3 CUDA product"
        );
        if first_exit == FirstExit::Completed {
            ensure!(
                matches!(&product.assets, PreparedAssets::PH3Image(assets) if !assets.has_prompt_expansion()),
                "prepared oracle assets unexpectedly enable prompt expansion"
            );
        }
        let serving = product
            .serving
            .as_ref()
            .context("explicit serving config required")?;
        let service_config = ImageServiceConfig {
            batching: serving.batching,
            maximum_pending: serving.maximum_pending,
            maximum_pending_bytes: serving
                .maximum_pending_bytes
                .context("resolved queue bytes required")?
                .get(),
            maximum_prompt_bytes: serving.maximum_prompt_bytes,
            max_num_seqs: serving.max_num_seqs,
            request_batch_max_wait_ms: serving.request_batch_max_wait_ms,
        };
        ensure!(
            service_config.max_num_seqs >= 2,
            "declare at least two service slots"
        );
        ensure!(
            first_exit != FirstExit::Completed || service_config.max_num_seqs == 2,
            "natural refill oracle keeps exactly two declared slots, including the B1 reference"
        );
        let requests = if let Some(path) = std::env::var_os("FERRUM_TEST_IMAGE_REQUESTS_JSON") {
            serde_json::from_slice::<Vec<ImageGenerationRequest>>(&source::read_bounded(
                Path::new(&path),
                3 * 1024 * 1024 + 4096,
            )?)?
        } else {
            [
                "A red kite above the sea",
                "A blue ceramic bowl on a wooden table",
                "A small wooden boat on calm water",
            ]
            .into_iter()
            .enumerate()
            .map(|(index, prompt)| {
                Ok(ImageGenerationRequest {
                    prompt: prompt.into(),
                    width: serving.default_width,
                    height: serving.default_height,
                    seed: serving
                        .default_seed
                        .checked_add(index as u64)
                        .context("fixture seed overflow")?,
                    n: 1,
                })
            })
            .collect::<Result<Vec<_>>>()?
        };
        ensure!(requests.len() == 3, "provide exactly A/B/C requests");
        if first_exit == FirstExit::Completed {
            let prompts = requests
                .iter()
                .map(|r| &r.prompt)
                .collect::<std::collections::BTreeSet<_>>();
            let seeds = requests
                .iter()
                .map(|r| r.seed)
                .collect::<std::collections::BTreeSet<_>>();
            ensure!(
                prompts.len() == 3 && seeds.len() == 3,
                "natural refill oracle requires three distinct fixed briefs and independent seeds"
            );
        }
        for request in &requests {
            ensure!(
                request.n == 1
                    && request.width == requests[0].width
                    && request.height == requests[0].height,
                "A/B/C require compatible single-image geometry"
            );
            PngLimits::default().validate(PngDimensions::new(
                request.width.into(),
                request.height.into(),
            )?)?;
        }
        Ok((requests, service_config))
    })()?;
    let mut trace_file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(directory.join("continuous.trace.jsonl"))
        .map_err(anyhow::Error::from)?;
    write_json_new(
        &directory.join("request.json"),
        &json!({"config":config_path,"requests":requests,
        "first_exit":first_exit.outcome(),
        "fixed_expanded_brief_b1_oracle":first_exit == FirstExit::Completed,
        "service_config":service_config,"deadline_seconds":DEADLINE.as_secs(),
        "qualification":"functional real-asset service trajectory only; no performance, quality or Torch parity claim"}),
    )?;
    let configured_trace = product.continuous_trace;
    let trace_limits = configured_trace
        .as_ref()
        .map(|trace| trace.limits())
        .unwrap_or(ImageContinuousTraceLimits {
            maximum_bytes: DEFAULT_TRACE_BYTES,
        });
    let (mut pipeline, composition) = product.backend.create(product.assets)?;
    let host = pipeline.host_account().clone();
    let context = json!({"entrypoint":"official_h3_cuda_continuous_service_test", "composition":composition,
        "service_config":service_config});
    let handle = match pipeline.enable_continuous_trace(trace_limits, &context) {
        Ok(handle) => handle,
        Err(error) => {
            if let Err(failure) = pipeline.shutdown() {
                return Err(ImageProductError::generation_pipeline(failure, pipeline));
            }
            return Err(error.into());
        }
    };
    let configured_trace = configured_trace.map(|trace| trace.bind(handle.clone()));
    let service = pipeline.into_service(service_config)?;
    let deadline = Instant::now()
        .checked_add(DEADLINE)
        .context("fixture deadline overflow")?;
    let mut jobs: [Option<ImageJob>; 3] = [None, None, None];
    let result: Result<Value> = async {
        let reference_b = if first_exit == FirstExit::Completed {
            let job = service.try_submit_with_client_attempt(requests[1].clone(), Some(Uuid::new_v4()))?;
            let correlation = job.correlation();
            let output = await_output(job, deadline).await?;
            let record = write_service_output(&host, &directory, "reference-b", correlation, &requests[1], &output)?;
            drop(output); // Keep only metadata, never B1's canvas or its Output credit.
            Some(record)
        } else {
            None
        };
        jobs[0] = Some(
            service.try_submit_with_client_attempt(requests[0].clone(), Some(Uuid::new_v4()))?,
        );
        let a = jobs[0].as_ref().unwrap().correlation();
        let first = observe(
            &handle,
            &service,
            deadline,
            "A's first actual committed step",
            &[a.request_id],
            |records| {
                let Some(identity) = linked_request(records, a.request_id)? else {
                    return Ok(None);
                };
                for record in records {
                    if record["event"]["kind"] == "packed_step_committed"
                        && member_step(record, identity).is_some()
                    {
                        return Ok(Some(checked_commit(records, record)?.clone()));
                    }
                }
                Ok(None)
            },
        )
        .await?;
        jobs[1] = Some(
            service.try_submit_with_client_attempt(requests[1].clone(), Some(Uuid::new_v4()))?,
        );
        let b = jobs[1].as_ref().unwrap().correlation();
        let ab = observe(
            &handle,
            &service,
            deadline,
            "A/B actual mixed progress",
            &[a.request_id, b.request_id],
            |records| mixed(records, a.request_id, b.request_id),
        )
        .await?;
        if first_exit == FirstExit::Cancelled {
            jobs[0].as_ref().unwrap().cancel();
        }
        // In the natural case C is queued while both original slots are live.
        // Its admission is driven by real retirement, not an elapsed-time guess.
        jobs[2] = Some(
            service.try_submit_with_client_attempt(requests[2].clone(), Some(Uuid::new_v4()))?,
        );
        let c = jobs[2].as_ref().unwrap().correlation();
        let first_output = if first_exit == FirstExit::Completed {
            let output = await_output(jobs[0].take().unwrap(), deadline).await?;
            let record = write_service_output(&host, &directory, "a", a, &requests[0], &output)?;
            drop(output); // Do not let a delivered canvas backpressure the refill.
            Some(record)
        } else {
            None
        };
        let bc = observe(
            &handle,
            &service,
            deadline,
            "B/C actual replacement mixed progress",
            &[b.request_id, c.request_id],
            |records| mixed(records, b.request_id, c.request_id),
        )
        .await?;
        if first_exit == FirstExit::Cancelled {
            let cancelled = tokio::time::timeout_at(
            tokio::time::Instant::from_std(deadline),
            jobs[0].take().unwrap().wait(),
        )
        .await
        .context("deadline awaiting A's completed-fence cancellation")?;
            ensure!(
            matches!(cancelled,Err(ref error) if error.kind()==ImageServiceErrorKind::Cancelled),
            "A did not retire by the actual cancellation protocol"
            );
        }
        let output_b = await_output(jobs[1].take().unwrap(), deadline).await?;
        let record_b = write_service_output(&host, &directory, "b", b, &requests[1], &output_b)?;
        // Deliver/write/drop one canvas before awaiting the next. Actual
        // consumer-held Output can legitimately backpressure C's decoder.
        drop(output_b);
        let output_c = await_output(jobs[2].take().unwrap(), deadline).await?;
        let record_c = write_service_output(&host, &directory, "c", c, &requests[2], &output_c)?;
        drop(output_c);
        if let Some(reference) = &reference_b {
            let same_prompt = reference["submitted_prompt_sha256"] == record_b["submitted_prompt_sha256"];
            let same_pixels = reference["image_pixels"]["rgba8_sha256"] == record_b["image_pixels"]["rgba8_sha256"];
            let same_geometry = ["width", "height", "seed"].into_iter().all(|key|
                reference["generation"][key] == record_b["generation"][key]);
            let same_steps = reference["generation"]["steps"] == record_b["generation"]["steps"];
            write_json_new(&directory.join("survivor-comparison.json"), &json!({
                "expansion":"disabled", "same_submitted_and_effective_prompt":same_prompt,
                "same_geometry_and_seed":same_geometry, "same_solver_steps":same_steps,
                "same_canonical_rgba8":same_pixels,
                "reference_pixels":reference["image_pixels"], "survivor_pixels":record_b["image_pixels"],
                "scope":"fixed-brief generation B1 versus natural-completion refill; not enabled-expansion parity or an SLO"
            }))?;
            ensure!(same_prompt && same_geometry && same_steps && same_pixels,
                "surviving B differs from its uncancelled B1 reference; see survivor-comparison.json");
        }
        Ok(
            json!({"correlations":{"a":a,"b":b,"c":c},"first_commit":first,
            "mixed_ab":ab,"mixed_bc":bc,"outputs":[record_b,record_c],
            "reference_b":reference_b,"first_output":first_output}),
        )
    }
    .await;
    for job in jobs.iter().flatten() {
        job.cancel();
    }
    drop(jobs);
    // Never time out or abort original native retirement. A failed original
    // shutdown receipt retains/quarantines its recovery owner by its contract.
    let retirement = service.shutdown().await.map_err(anyhow::Error::from)?;
    let terminal = match &result {
        Ok(trajectory) => snapshot(&handle)
            .and_then(|records| terminal_evidence(&records, trajectory, first_exit)),
        Err(_) => Ok(()),
    };
    let status = handle.write_jsonl(&mut trace_file)?;
    trace_file.sync_all().map_err(anyhow::Error::from)?;
    let worker_failed = retirement.has_failure();
    let worker_diagnostic = retirement.failure().map(|failure| format!("{failure:?}"));
    let final_service = service.snapshot();
    if let Some(trace) = configured_trace {
        trace.export_best_effort();
    }
    drop((retirement, service));
    let trace_backing = status.backing_bytes;
    drop(handle);
    let final_host = host.usage().map_err(anyhow::Error::from)?;
    let passed = result.is_ok()
        && terminal.is_ok()
        && !worker_failed
        && status.sealed
        && !status.truncated
        && final_host.committed_bytes() == 0
        && final_host.allocated_bytes() == 0;
    write_json_new(
        &directory.join("result.json"),
        &json!({
        "status":if passed {"passed"} else {"failed"},
        "trajectory":result.as_ref().ok(),"error":result.as_ref().err().map(|error|format!("{error:#}")),
        "terminal_evidence_error":terminal.as_ref().err().map(|error|format!("{error:#}")),
        "worker_failure":worker_diagnostic,"service":final_service,"trace":status,
        "trace_physical_bytes":trace_backing,"host_final_committed_bytes":final_host.committed_bytes(),
        "host_final_allocated_bytes":final_host.allocated_bytes(),
        "qualification":"real asset CUDA packed service functional evidence; no quality/performance/global VRAM-zero claim"}),
    )?;
    result?;
    terminal?;
    (|| -> Result<()> {
        ensure!(
            !worker_failed,
            "original worker retained a native failure; see result and trace"
        );
        ensure!(
            status.sealed && !status.truncated,
            "original producer did not seal a complete trace"
        );
        ensure!(
            final_host.committed_bytes() == 0 && final_host.allocated_bytes() == 0,
            "original Host backing or credit survived outputs, worker and trace"
        );
        Ok(())
    })()
    .map_err(Into::into)
}
