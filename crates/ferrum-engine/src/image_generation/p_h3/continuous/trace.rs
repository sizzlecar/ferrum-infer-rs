//! Borrowed observations of the original published owners and committed work.
//! No secondary membership, solver state, payload buffer or lifetime ledger.
use super::*;
use crate::image_generation::continuous_trace::{
    ImageContinuousTraceEvent as TimedEvent, ImageContinuousTraceRecorder, ImageTracePhase,
    ImageTracePhaseOutcome, ImageTraceStepStartMember,
};
use crate::image_generation::{
    service::ImageDenoisingExitReason, ImagePipelineExecution, ImagePipelineSourceAdapter,
};
use ferrum_models::p_h3_image::{PH3ImageCurveBlockRowRun, PH3ImageSamplingStep};
use ferrum_types::ImageRequestCorrelation;
use serde::{ser::SerializeSeq, Serialize, Serializer};
use sha2::{Digest, Sha256};
use std::time::Instant;

/// Only small metadata survives serialization. Shape and bytes are borrowed
/// from the actual request-persistent latent; no tensor owner is cloned.
#[derive(Serialize)]
struct ReferenceLatentDigest<'a> {
    shape: &'a [u64],
    sha256: String,
}
impl<'a> ReferenceLatentDigest<'a> {
    fn new(latent: &'a ManagedHostTensor) -> Self {
        Self {
            shape: latent.tensor().dimensions(),
            sha256: format!("{:x}", Sha256::digest(latent.bytes().bytes())),
        }
    }
}

#[derive(Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum Event<'a> {
    RequestInitialized {
        request: &'a RequestIdentity,
        job: u64,
        seed: u64,
        width: u32,
        height: u32,
        editing: bool,
        #[serde(skip_serializing_if = "Option::is_none")]
        reference_latent: Option<ReferenceLatentDigest<'a>>,
    },
    CohortPublished {
        cohort_job: u64,
        ordered_members: &'a [RequestIdentity],
        plan_hash: &'a str,
        program_fingerprint: &'a str,
    },
    PackedStepCommitted {
        #[serde(skip_serializing_if = "Option::is_none")]
        start_sequence: Option<u64>,
        cohort_job: u64,
        plan_hash: &'a str,
        program_fingerprint: &'a str,
        packed_request_count: usize,
        time_basis: &'a [f32],
        members: Members<'a>,
    },
    RequestExitedAtFence {
        request: &'a RequestIdentity,
        completed_steps: usize,
        reason: ImageDenoisingExitReason,
    },
}

pub(super) fn initialized<R: DeviceRuntime>(
    pipeline: &PH3ImagePipeline<R>,
    request: &ContinuousRequest<R>,
) {
    if let Some(trace) = &pipeline.execution.continuous_trace {
        trace.record(&Event::RequestInitialized {
            request: &request.geometry.identity,
            job: request.job,
            seed: request.seed,
            width: request.width,
            height: request.height,
            editing: request.reference.is_some(),
            // Hash only with explicit diagnostic tracing enabled. This is
            // additional Host work, not a pure performance measurement.
            reference_latent: request
                .reference
                .as_ref()
                .map(|reference| ReferenceLatentDigest::new(&reference.latent)),
        });
    }
}
pub(super) fn published<R: DeviceRuntime>(
    pipeline: &PH3ImagePipeline<R>,
    batch: &ContinuousBatch<R>,
) {
    if let Some(trace) = &pipeline.execution.continuous_trace {
        trace.record(&Event::CohortPublished {
            cohort_job: batch.job,
            ordered_members: &batch.members,
            plan_hash: &batch.prepared.stage.plan().plan_hash,
            program_fingerprint: batch
                .prepared
                .stage
                .executable()
                .execution_plan()
                .payload()
                .program_fingerprint(),
        });
    }
}

pub(super) fn committed<R: DeviceRuntime, S: ImagePipelineSourceAdapter>(
    execution: &ImagePipelineExecution<R, S>,
    prepared: &CompiledSteady<R>,
    job: u64,
    times: &[f32],
    lanes: &[step::StepLane<'_>],
    steps: &[PH3ImageSamplingStep],
    start_sequence: Option<u64>,
) {
    if let Some(trace) = &execution.continuous_trace {
        trace.record(&Event::PackedStepCommitted {
            start_sequence,
            cohort_job: job,
            plan_hash: &prepared.stage.plan().plan_hash,
            program_fingerprint: prepared
                .stage
                .executable()
                .execution_plan()
                .payload()
                .program_fingerprint(),
            packed_request_count: lanes.len(),
            time_basis: times,
            members: Members {
                recipe: &prepared.recipe,
                lanes,
                steps,
            },
        });
    }
}

pub(super) fn started<R: DeviceRuntime, S: ImagePipelineSourceAdapter>(
    execution: &ImagePipelineExecution<R, S>,
    prepared: &CompiledSteady<R>,
    job: u64,
    lanes: &[step::StepLane<'_>],
) -> Option<u64> {
    let trace = execution.continuous_trace.as_ref()?;
    let members = lanes
        .iter()
        .map(|lane| ImageTraceStepStartMember {
            request: lane.identity.as_str(),
            step_index: lane.sampling.step_index(),
        })
        .collect::<Vec<_>>();
    trace.handle().record(&TimedEvent::PackedStepStarted {
        cohort_job: job,
        plan_hash: &prepared.stage.plan().plan_hash,
        program_fingerprint: prepared
            .stage
            .executable()
            .execution_plan()
            .payload()
            .program_fingerprint(),
        members: &members,
    })
}

struct Members<'a> {
    recipe: &'a ferrum_models::p_h3_image::PreparedPH3ImageDenoiser,
    lanes: &'a [step::StepLane<'a>],
    steps: &'a [PH3ImageSamplingStep],
}
impl Serialize for Members<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        use serde::ser::Error;
        let runs = self.recipe.request_row_runs();
        if self.lanes.len() != self.steps.len() || self.lanes.len() != runs.len() {
            return Err(S::Error::custom(
                "committed trace membership differs from original recipe",
            ));
        }
        let mut sequence = serializer.serialize_seq(Some(self.lanes.len()))?;
        let mut start = 0u64;
        for ((lane, step), row_runs) in self.lanes.iter().zip(self.steps).zip(runs) {
            let end = row_runs
                .iter()
                .try_fold(start, |end, run| end.checked_add(run.length()))
                .ok_or_else(|| S::Error::custom("packed trace row extent overflows"))?;
            sequence.serialize_element(&Member {
                request: lane.identity,
                packed_row_start: start,
                packed_row_end: end,
                step_index: step.index(),
                sigma: step.sigma(),
                next_sigma: step.next_sigma(),
                scheduler_timestep: step.timestep(),
                row_runs,
            })?;
            start = end;
        }
        sequence.end()
    }
}
#[derive(Serialize)]
struct Member<'a> {
    request: &'a RequestIdentity,
    packed_row_start: u64,
    packed_row_end: u64,
    step_index: usize,
    sigma: f32,
    next_sigma: f32,
    scheduler_timestep: f32,
    row_runs: &'a [PH3ImageCurveBlockRowRun],
}

impl<R: DeviceRuntime> PH3ImagePipeline<R> {
    pub(in crate::image_generation::p_h3) fn trace_phase_started(
        &self,
        correlation: Option<ImageRequestCorrelation>,
        request: Option<&RequestIdentity>,
        phase: ImageTracePhase,
    ) -> Option<u64> {
        record_phase_start(
            self.execution.continuous_trace.as_ref(),
            correlation,
            request,
            phase,
        )
    }
    pub(in crate::image_generation::p_h3) fn trace_phase_finished(
        &self,
        start: Option<u64>,
        outcome: ImageTracePhaseOutcome,
    ) {
        if let (Some(trace), Some(start_sequence)) = (&self.execution.continuous_trace, start) {
            trace.handle().record(&TimedEvent::PhaseFinished {
                start_sequence,
                outcome,
            });
        }
    }
    pub(in crate::image_generation::p_h3) fn trace_phase_finished_at(
        &self,
        start: Option<u64>,
        outcome: ImageTracePhaseOutcome,
        at: Instant,
    ) {
        record_phase_finish(self.execution.continuous_trace.as_ref(), start, outcome, at);
    }
    pub(super) fn trace_request_link(
        &self,
        correlation: ImageRequestCorrelation,
        request: &RequestIdentity,
    ) {
        if let Some(trace) = &self.execution.continuous_trace {
            trace.handle().record(&TimedEvent::RequestLinked {
                request_id: correlation.request_id,
                request: request.as_str(),
            });
        }
    }
    pub(super) fn trace_request_exit(
        &self,
        request: &ContinuousRequest<R>,
        reason: ImageDenoisingExitReason,
    ) {
        if let Some(trace) = &self.execution.continuous_trace {
            trace.record(&Event::RequestExitedAtFence {
                request: &request.geometry.identity,
                completed_steps: request.step_index(),
                reason,
            });
        }
    }
}

pub(super) fn record_phase_finish(
    trace: Option<&ImageContinuousTraceRecorder>,
    start: Option<u64>,
    outcome: ImageTracePhaseOutcome,
    at: Instant,
) {
    if let (Some(trace), Some(start_sequence)) = (trace, start) {
        trace.handle().record_at(
            at,
            &TimedEvent::PhaseFinished {
                start_sequence,
                outcome,
            },
        );
    }
}

fn record_phase_start(
    trace: Option<&ImageContinuousTraceRecorder>,
    correlation: Option<ImageRequestCorrelation>,
    request: Option<&RequestIdentity>,
    phase: ImageTracePhase,
) -> Option<u64> {
    // Direct run has no service UUID before the actual expanded request is
    // prepared. An uncorrelated span must not invent a request identity.
    if correlation.is_none() && request.is_none() {
        return None;
    }
    trace?.handle().record(&TimedEvent::PhaseStarted {
        request_id: correlation.map(|c| c.request_id),
        request: request.map(RequestIdentity::as_str),
        phase,
    })
}

#[cfg(test)]
mod diagnostic_identity_tests {
    use super::*;
    use crate::image_generation::continuous_trace::ImageContinuousTraceLimits;

    #[test]
    fn reference_latent_trace_borrows_original_payload_and_releases_with_request() {
        let account = |capacity| {
            HostMemoryAccount::with_allocation_alignment(
                HostMemoryLimits::new(
                    capacity,
                    HostMemoryClassLimits {
                        input_preprocess_bytes: capacity,
                        output_bytes: 0,
                        transfer_staging_bytes: 0,
                        weight_backing_bytes: 0,
                    },
                )
                .unwrap(),
                64,
            )
            .unwrap()
        };
        // F32 low mantissa bits, signed zero and a subnormal are hashed as
        // the original logical payload, excluding its alignment padding.
        let bytes: Vec<_> = [0x3f80_0001u32, 0x8000_0000, 0x3eaa_aaab, 0x0000_0001]
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect();
        let capacity = account(bytes.len()).backing_bytes(bytes.len()).unwrap();
        let host = account(capacity);
        let persistent = host
            .reserve_scope(&[(HostMemoryClass::InputPreprocess, capacity)])
            .unwrap();
        let mut backing = persistent
            .account()
            .allocate_zeroed(HostMemoryClass::InputPreprocess, bytes.len())
            .unwrap();
        backing.as_mut().copy_from_slice(&bytes);
        let latent = ManagedHostTensor::new(
            ProgramValueId::new("value.reference.original").unwrap(),
            ResolvedTensorSpec::new(
                vec![1, 1, 1, 2, 2],
                ElementType::F32,
                ResolvedTensorLayout::Contiguous,
            )
            .unwrap(),
            backing.freeze(),
        )
        .unwrap();
        let before = host.usage().unwrap();
        assert_eq!(before.allocated_bytes(), capacity);
        assert_eq!(before.committed_bytes(), capacity);
        let request = RequestIdentity::new("request.reference.digest").unwrap();
        let serialized = {
            let digest = ReferenceLatentDigest::new(&latent);
            assert!(std::ptr::eq(digest.shape, latent.tensor().dimensions()));
            serde_json::to_value(Event::RequestInitialized {
                request: &request,
                job: 1,
                seed: 42,
                width: 32,
                height: 32,
                editing: true,
                reference_latent: Some(digest),
            })
            .unwrap()
        };
        assert_eq!(serialized["kind"], "request_initialized");
        assert_eq!(
            serialized["reference_latent"]["shape"],
            serde_json::json!([1, 1, 1, 2, 2])
        );
        assert_eq!(
            serialized["reference_latent"]["sha256"],
            "ffebfdebdb7f43ee48d34012f9aec74b9760a95b9dcf4024a237035f2ca73476"
        );
        assert_eq!(latent.bytes().bytes(), bytes);
        assert_eq!(
            host.usage().unwrap().allocated_bytes(),
            before.allocated_bytes()
        );
        assert_eq!(
            host.usage().unwrap().committed_bytes(),
            before.committed_bytes()
        );
        drop(latent);
        persistent.close().unwrap();
        // Serialized diagnostic metadata remains valid without retaining the
        // original request allocation or extending its scope authority.
        assert!(serialized["reference_latent"].is_object());
        assert_eq!(host.usage().unwrap().allocated_bytes(), 0);
        assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    }

    #[test]
    fn reference_latent_trace_is_omitted_for_generation() {
        let request = RequestIdentity::new("request.generation.digest").unwrap();
        let serialized = serde_json::to_value(Event::RequestInitialized {
            request: &request,
            job: 1,
            seed: 42,
            width: 32,
            height: 32,
            editing: false,
            reference_latent: None,
        })
        .unwrap();
        assert_eq!(serialized["kind"], "request_initialized");
        assert_eq!(serialized["editing"], false);
        assert!(serialized.get("reference_latent").is_none());
    }

    #[test]
    fn prompt_expansion_diagnostics_run_omits_unidentified_span_but_service_and_model_spans_are_real(
    ) {
        let host = HostMemoryAccount::new(
            HostMemoryLimits::new(
                262144,
                HostMemoryClassLimits {
                    input_preprocess_bytes: 65536,
                    output_bytes: 65536,
                    transfer_staging_bytes: 65536,
                    weight_backing_bytes: 65536,
                },
            )
            .unwrap(),
        );
        let (recorder, handle) = ImageContinuousTraceRecorder::new(
            &host,
            ImageContinuousTraceLimits {
                maximum_bytes: 8192,
            },
        )
        .unwrap();
        assert_eq!(
            record_phase_start(
                Some(&recorder),
                None,
                None,
                ImageTracePhase::PromptExpansion
            ),
            None
        );
        assert_eq!(handle.status().recorded, 0);
        let correlation = ImageRequestCorrelation::new(None);
        assert_eq!(
            record_phase_start(
                Some(&recorder),
                Some(correlation),
                None,
                ImageTracePhase::PromptExpansion
            ),
            Some(1)
        );
        let request = RequestIdentity::new("request.actual.after.prepare").unwrap();
        assert_eq!(
            record_phase_start(
                Some(&recorder),
                None,
                Some(&request),
                ImageTracePhase::Refiner
            ),
            Some(2)
        );
        let mut bytes = Vec::new();
        handle.write_jsonl(&mut bytes).unwrap();
        let records: Vec<serde_json::Value> = bytes
            .split(|&c| c == b'\n')
            .filter(|line| !line.is_empty())
            .map(|line| serde_json::from_slice(line).unwrap())
            .collect();
        assert_eq!(
            records[0]["event"]["request_id"],
            correlation.request_id.to_string()
        );
        assert!(records[0]["event"]["request"].is_null());
        assert_eq!(records[1]["event"]["request"], request.as_str());
        assert!(records[1]["event"]["request_id"].is_null());
        drop(recorder);
        drop(handle);
        assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    }
}
