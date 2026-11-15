use super::validation::Media;
use serde::Serialize;
use serde_json::Value;
use std::path::PathBuf;

#[derive(Debug, Serialize)]
pub struct Observation {
    pub index: usize,
    pub request_index: usize,
    pub dispatched: bool,
    pub dispatch_offset_seconds: f64,
    pub submission_seconds: Option<f64>,
    pub terminal_observed_seconds: Option<f64>,
    pub artifact_received_seconds: Option<f64>,
    pub request_returned_seconds: f64,
    pub polls: usize,
    pub job_id: Option<String>,
    pub last_job: Option<Value>,
    pub cancellation: Option<Value>,
    pub remote_terminal_unknown: bool,
    pub artifact: Option<PathBuf>,
    pub artifact_bytes: Option<u64>,
    pub artifact_sha256: Option<String>,
    pub media: Option<Media>,
    pub error: Option<String>,
    pub evidence_error: Option<String>,
}

impl Observation {
    pub fn new(index: usize, request_index: usize, dispatch_offset_seconds: f64) -> Self {
        Self {
            index,
            request_index,
            dispatched: true,
            dispatch_offset_seconds,
            submission_seconds: None,
            terminal_observed_seconds: None,
            artifact_received_seconds: None,
            request_returned_seconds: 0.0,
            polls: 0,
            job_id: None,
            last_job: None,
            cancellation: None,
            remote_terminal_unknown: false,
            artifact: None,
            artifact_bytes: None,
            artifact_sha256: None,
            media: None,
            error: None,
            evidence_error: None,
        }
    }
    pub fn succeeded(&self) -> bool {
        self.error.is_none()
            && self.evidence_error.is_none()
            && self.artifact_received_seconds.is_some()
            && self.media.is_some()
    }
}

#[derive(Serialize)]
pub struct Summary {
    pub planned: usize,
    pub offered: usize,
    pub not_dispatched: usize,
    pub successes: usize,
    pub errors: usize,
    pub completion_fraction: f64,
    pub p50_seconds: Option<f64>,
    pub p99_seconds: Option<f64>,
    pub successful_videos_per_minute: f64,
    pub successful_generated_frames_per_second: f64,
    pub successful_video_seconds_per_wall_second: f64,
    pub client_observation_window_seconds: f64,
    pub observed_slo_met: bool,
    pub latency_slo_p99_seconds: f64,
    pub required_completion_fraction: f64,
    pub peak_vram_bytes: Option<u64>,
    pub peak_server_rss_bytes: Option<u64>,
}

impl Summary {
    pub fn calculate(requests: &[Observation], seconds: f64, slo: f64) -> Self {
        let mut latencies: Vec<_> = requests
            .iter()
            .filter(|r| r.succeeded())
            .filter_map(|r| r.artifact_received_seconds)
            .collect();
        latencies.sort_by(f64::total_cmp);
        let successes = latencies.len();
        let p50 = (!latencies.is_empty()).then(|| ferrum_bench_core::percentile(&latencies, 0.50));
        let p99 = (!latencies.is_empty()).then(|| ferrum_bench_core::percentile(&latencies, 0.99));
        let offered = requests.iter().filter(|r| r.dispatched).count();
        let frames: u64 = requests
            .iter()
            .filter(|r| r.succeeded())
            .filter_map(|r| r.media.as_ref())
            .map(|m| m.frames)
            .sum();
        let video_seconds: f64 = requests
            .iter()
            .filter(|r| r.succeeded())
            .filter_map(|r| r.media.as_ref())
            .map(|m| m.frames as f64 / m.fps)
            .sum();
        Self {
            planned: requests.len(),
            offered,
            not_dispatched: requests.len() - offered,
            successes,
            errors: offered - successes,
            completion_fraction: successes as f64 / requests.len().max(1) as f64,
            p50_seconds: p50,
            p99_seconds: p99,
            successful_videos_per_minute: successes as f64 * 60.0 / seconds,
            successful_generated_frames_per_second: frames as f64 / seconds,
            successful_video_seconds_per_wall_second: video_seconds / seconds,
            client_observation_window_seconds: seconds,
            observed_slo_met: !requests.is_empty()
                && successes == requests.len()
                && p99.is_some_and(|p| p <= slo),
            latency_slo_p99_seconds: slo,
            required_completion_fraction: 1.0,
            peak_vram_bytes: None,
            peak_server_rss_bytes: None,
        }
    }
}

#[derive(Serialize)]
pub struct Cell {
    pub phase: String,
    pub configured_concurrency: usize,
    pub actual_client_width: usize,
    pub repeat: usize,
    pub summary: Summary,
    pub requests: Vec<Observation>,
    pub resources: Option<super::resources::ResourceReport>,
    pub client_errors: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn failed_requests_remain_in_slo_denominator() {
        let mut good = Observation::new(0, 0, 0.0);
        good.artifact_received_seconds = Some(10.0);
        good.media = Some(Media {
            width: 832,
            height: 480,
            frames: 124,
            fps: 24.0,
            audio: true,
        });
        let mut bad = Observation::new(1, 0, 0.0);
        bad.error = Some("failed job".into());
        let report = Summary::calculate(&[good, bad], 20.0, 120.0);
        assert_eq!(report.offered, 2);
        assert_eq!(report.errors, 1);
        assert_eq!(report.p99_seconds, Some(10.0));
        assert_eq!(report.completion_fraction, 0.5);
        assert!(!report.observed_slo_met);
    }
}
