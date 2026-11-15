use super::{
    request::invalid, RawVideoOutput, VideoGenerationRequest, VideoGenerationResult,
    VideoOutputPlan, VideoRequestLimits,
};
use crate::iterative_service::{
    self as core, BatchWorker, JobCancellationHandle, OutputLease, ServiceConfig, ServiceError,
    ServiceSnapshot, WorkerFailure,
};
use ferrum_output::mp4::{Mp4CancellationHandle, Mp4EncodingPermit, Mp4Error, VideoAudioSpec};
use ferrum_runtime::HostMemoryAccount;
use ferrum_types::RequestCorrelation;
use std::{
    fmt,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};
use tokio::sync::oneshot;
use uuid::Uuid;

type Worker = dyn BatchWorker<Request = VideoGenerationRequest, Output = RawVideoOutput>;
type Core = core::ServiceCore<VideoGenerationRequest, RawVideoOutput>;
type CoreJob = core::Job<VideoGenerationRequest, RawVideoOutput>;

#[derive(Debug, Clone, Copy)]
pub struct VideoServiceConfig {
    service: ServiceConfig,
    ingress: VideoRequestLimits,
}
impl VideoServiceConfig {
    /// Validate before constructing/opening a concrete model adapter.
    pub fn new(service: ServiceConfig, ingress: VideoRequestLimits) -> Result<Self, ServiceError> {
        ingress.validate()?;
        if service.maximum_pending == 0
            || service.maximum_pending_bytes == 0
            || service.max_num_seqs == 0
        {
            return Err(invalid(
                "video service queue and active limits must be positive",
            ));
        }
        service
            .maximum_pending
            .checked_mul(std::mem::size_of::<
                core::Queued<VideoGenerationRequest, RawVideoOutput>,
            >())
            .filter(|n| *n <= isize::MAX as usize)
            .ok_or_else(|| invalid("video queue extent overflow"))?;
        Instant::now()
            .checked_add(Duration::from_millis(service.request_batch_max_wait_ms))
            .ok_or_else(|| invalid("video collection deadline overflow"))?;
        Ok(Self { service, ingress })
    }
    pub fn service(&self) -> ServiceConfig {
        self.service
    }
    pub fn ingress(&self) -> VideoRequestLimits {
        self.ingress
    }
}

pub struct VideoService {
    core: Core,
    config: VideoServiceConfig,
    host: HostMemoryAccount,
}
impl VideoService {
    /// Only an actual engine adapter supplies the existing worker protocol.
    pub(crate) fn start_worker(
        worker: Box<Worker>,
        config: VideoServiceConfig,
        host: HostMemoryAccount,
    ) -> Result<Self, ServiceError> {
        Ok(Self {
            core: Core::start_with_host(worker, config.service, Some(host.clone()))?,
            config,
            host,
        })
    }
    pub fn host_account(&self) -> &HostMemoryAccount {
        &self.host
    }
    pub fn snapshot(&self) -> ServiceSnapshot {
        self.core.snapshot()
    }
    pub fn try_submit(
        &self,
        request: VideoGenerationRequest,
        output: VideoOutputPlan,
    ) -> Result<VideoJob, ServiceError> {
        self.try_submit_with_client_attempt(request, output, None)
    }
    pub fn try_submit_with_client_attempt(
        &self,
        request: VideoGenerationRequest,
        output: VideoOutputPlan,
        attempt: Option<Uuid>,
    ) -> Result<VideoJob, ServiceError> {
        let correlation = RequestCorrelation::new(attempt);
        request
            .validate(self.config.ingress, &self.host, output.spec())
            .map_err(|e| self.core.submission_error(correlation, e))?;
        let expected = output.spec();
        let seed = request.parameters.seed;
        let (response, receiver) = oneshot::channel();
        let job = self.core.try_submit_request_with_output_and_correlation(
            request,
            Some(core::output_gate::QueuedOutput {
                gate: Arc::new(output),
                lease: None,
                response,
            }),
            correlation,
        )?;
        let control = Arc::new(Cancellation {
            core: job.cancellation_handle(),
            state: Mutex::new(CancellationState::default()),
        });
        Ok(VideoJob {
            inner: Some(job),
            output: Some(receiver),
            control,
            expected,
            host: self.host.clone(),
            correlation,
            seed,
        })
    }
    /// Joins the sole model worker and retains its original failure owner.
    /// Accepted VideoJobs separately own codec completion and must be drained
    /// by their caller before the product declares all media work shut down.
    pub async fn shutdown(&self) -> Result<VideoServiceShutdown, ServiceError> {
        self.core
            .shutdown()
            .await
            .map(|inner| VideoServiceShutdown { inner })
    }
    /// A finite run seals and drains the same core, then waits for the same
    /// encoded result. Even rejected requests retire the supplied real worker.
    pub(crate) async fn run_with_worker(
        worker: Box<Worker>,
        config: VideoServiceConfig,
        host: HostMemoryAccount,
        request: VideoGenerationRequest,
        output: VideoOutputPlan,
    ) -> VideoServiceRun {
        let core = Core::unstarted(worker.as_ref(), config.service, Some(host.clone()));
        let service = Self { core, config, host };
        let job = service.try_submit(request, output);
        let shutdown = VideoServiceShutdown {
            inner: service.core.run_to_completion(worker),
        };
        let result = match job {
            Ok(job) => job.wait().await,
            Err(error) => Err(VideoJobError::Generation(error)),
        };
        VideoServiceRun { result, shutdown }
    }
}

#[derive(Default)]
struct CancellationState {
    requested: bool,
    finished: bool,
    codec: Option<Mp4CancellationHandle>,
}
struct Cancellation {
    core: JobCancellationHandle,
    state: Mutex<CancellationState>,
}
impl Cancellation {
    fn cancel(&self) {
        let codec = {
            let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
            if state.finished {
                return;
            }
            state.requested = true;
            state.codec.clone()
        };
        self.core.cancel();
        if let Some(codec) = codec {
            codec.cancel();
        }
    }
    fn requested(&self) -> bool {
        self.state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .requested
    }
    fn bind_codec(&self, codec: Mp4CancellationHandle) {
        let cancelled = {
            let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
            state.codec = Some(codec.clone());
            state.requested
        };
        if cancelled {
            codec.cancel();
        }
    }
    fn finish(&self) -> bool {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        state.finished = true;
        state.codec = None;
        state.requested
    }
}
#[derive(Clone)]
pub struct VideoCancellationHandle(Arc<Cancellation>);
impl VideoCancellationHandle {
    pub fn cancel(&self) {
        self.0.cancel();
    }
}

#[derive(Debug)]
pub enum VideoJobError {
    Generation(ServiceError),
    Encoding(Mp4Error),
    InvalidOutput(ServiceError),
    Cancelled,
}
impl VideoJobError {
    pub fn is_cancelled(&self) -> bool {
        matches!(self, Self::Cancelled)
            || matches!(self, Self::Generation(e) if e.kind() == core::ServiceErrorKind::Cancelled)
            || matches!(self, Self::Encoding(e) if e.kind == ferrum_output::mp4::Mp4ErrorKind::Cancelled)
    }
}
impl fmt::Display for VideoJobError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Generation(e) | Self::InvalidOutput(e) => fmt::Display::fmt(e, f),
            Self::Encoding(e) => fmt::Display::fmt(e, f),
            Self::Cancelled => f.write_str("video cancelled before artifact publication"),
        }
    }
}
impl std::error::Error for VideoJobError {}

pub struct VideoJob {
    inner: Option<CoreJob>,
    output: Option<oneshot::Receiver<OutputLease>>,
    control: Arc<Cancellation>,
    expected: VideoAudioSpec,
    host: HostMemoryAccount,
    correlation: RequestCorrelation,
    seed: u64,
}
impl VideoJob {
    pub fn correlation(&self) -> RequestCorrelation {
        self.correlation
    }
    pub fn cancellation_handle(&self) -> VideoCancellationHandle {
        VideoCancellationHandle(self.control.clone())
    }
    pub fn cancel(&self) {
        self.control.cancel();
    }
    pub async fn wait(mut self) -> Result<VideoGenerationResult, VideoJobError> {
        let result = self.complete().await;
        let cancelled = self.control.finish();
        // A successful codec result is already reaped. Cancellation that wins
        // the publication lock can discard it safely; later cancels cannot.
        match result {
            Ok(_) if cancelled => Err(VideoJobError::Cancelled),
            other => other,
        }
    }
    async fn complete(&mut self) -> Result<VideoGenerationResult, VideoJobError> {
        let raw = self
            .inner
            .take()
            .expect("one result waiter")
            .wait()
            .await
            .map_err(VideoJobError::Generation)?;
        raw.validate(&self.host, self.expected)
            .map_err(VideoJobError::InvalidOutput)?;
        let lease = self
            .output
            .take()
            .expect("one output grant receiver")
            .await
            .map_err(|_| {
                VideoJobError::InvalidOutput(invalid("video output grant was not delivered"))
            })?;
        let permit = lease.into_inner::<Mp4EncodingPermit>().map_err(|_| {
            VideoJobError::InvalidOutput(invalid("video output grant belongs to another codec"))
        })?;
        if self.control.requested() {
            return Err(VideoJobError::Cancelled);
        }
        let (rgb, pcm) = raw.into_parts();
        let codec = permit.start(rgb, pcm).map_err(VideoJobError::Encoding)?;
        self.control.bind_codec(codec.cancellation_handle());
        // A cancellation racing the above start/bind is applied to the real
        // codec, then awaited through kill+reap, never reported early.
        let artifact = codec.wait().await.map_err(VideoJobError::Encoding)?;
        Ok(VideoGenerationResult {
            correlation: self.correlation,
            seed: self.seed,
            artifact,
        })
    }
}
impl Drop for VideoJob {
    fn drop(&mut self) {
        self.control.cancel();
    }
}

pub struct VideoServiceRun {
    pub result: Result<VideoGenerationResult, VideoJobError>,
    pub shutdown: VideoServiceShutdown,
}
pub struct VideoGeneratorOwner {
    _worker: Box<Worker>,
}
#[must_use = "retain the original failed model owner or explicitly take it for recovery"]
pub struct VideoServiceShutdown {
    inner: core::Retirement<VideoGenerationRequest, RawVideoOutput>,
}
impl VideoServiceShutdown {
    pub fn failure(&self) -> Option<&WorkerFailure> {
        self.inner.failure.as_ref()
    }
    pub fn has_failure(&self) -> bool {
        self.inner.failure.is_some()
    }
    pub fn into_parts(mut self) -> (Option<VideoGeneratorOwner>, Option<WorkerFailure>) {
        (
            self.inner
                .worker
                .take()
                .map(|worker| VideoGeneratorOwner { _worker: worker }),
            self.inner.failure.take(),
        )
    }
}

#[cfg(test)]
mod tests;
