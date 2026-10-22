//! Offline evidence bindings. No inference of identities from filenames.
use crate::paired_cluster as core;
use serde_json::Value;
use std::path::PathBuf;

#[derive(Clone, Debug)]
pub struct Artifact {
    pub path: PathBuf,
    pub sha256: core::Digest,
    pub bytes: u64,
}

/// A digest already present in an original frozen JSON receipt. The pointer
/// must name its hex string, not a newly invented normalization algorithm.
#[derive(Clone, Debug)]
pub struct DigestEvidence {
    pub artifact: Artifact,
    pub pointer: String,
    pub expected: core::Digest,
}

#[derive(Clone, Debug)]
pub struct BuildEvidence {
    pub artifact: Artifact,
    pub source_pointer: String,
    pub source_identity: String,
    pub binary_pointer: String,
    pub binary_sha256: core::Digest,
}

#[derive(Clone, Debug)]
pub struct ExpectedRequest {
    pub key: core::RequestKey,
    pub scenario: String,
    /// Exact original source metadata. JSON numeric representation is immaterial.
    pub source: Value,
    /// Names an existing digest within source, e.g. /parent_sha256 or
    /// /recipe_sha256. key.source means that declared original source digest.
    pub source_digest_pointer: String,
}

#[derive(Clone, Debug)]
pub struct InputPolicy {
    pub model: String,
    pub protocol: String,
    pub deadline_ms: u64,
    pub temperature: f64,
    pub request_policy: Value,
    pub provenance: Value,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ServerIdentity {
    /// Values from the actual monitor_start / initial_host receipt.
    Linux {
        boot_id: String,
        pid: u32,
        start_ticks: u64,
    },
    MacOs {
        pid: u32,
        start_abstime: u64,
        uuid: [u8; 16],
    },
}

#[derive(Clone, Debug)]
pub struct RunManifest {
    pub repeat_id: u64,
    pub config: Artifact,
    pub records: Artifact,
    pub summary: Artifact,
    pub client_exit: Artifact,
    pub monitor: Artifact,
    pub server: ServerIdentity,
    pub server_build: BuildEvidence,
    pub client_build: BuildEvidence,
    /// Original summary's client epoch, not an epoch synthesized by joining
    /// stages. This API accepts exactly one config/JSONL/summary per run.
    pub epoch_unix_ms: f64,
    pub policy: InputPolicy,
}

#[derive(Clone, Debug)]
pub struct PairManifest {
    pub baseline: RunManifest,
    pub candidate: RunManifest,
}

#[derive(Clone, Debug)]
pub struct Manifest {
    pub cell: DigestEvidence,
    pub inputs: DigestEvidence,
    pub workload: core::Workload,
    pub requests: Vec<ExpectedRequest>,
    pub pairs: Vec<PairManifest>,
}

#[derive(Clone, Copy, Debug)]
pub struct ReaderLimits {
    pub max_runs: usize,
    pub max_requests_per_run: usize,
    pub max_events_per_run: usize,
    pub max_artifact_bytes: usize,
    pub max_line_bytes: usize,
    pub max_total_read_bytes: u64,
    pub max_issues: usize,
    pub max_raw_failures: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReaderIssueKind {
    Budget,
    Io,
    Size,
    Hash,
    Json,
    DuplicateJsonField,
    Identity,
    Source,
    Input,
    Epoch,
    DuplicateIndex,
    UnexpectedIndex,
    Summary,
    Timing,
    Events,
    Usage,
    Exit,
}

#[derive(Clone, Debug)]
pub struct ReaderIssue {
    pub kind: ReaderIssueKind,
    pub index: Option<usize>,
    /// Identity/structure issues block pairing. Incomplete metric evidence is
    /// retained in a Run, where the analyzer preserves Fail versus Unresolved.
    pub blocks_pairing: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RawFailureKind {
    RecordedFailure,
    Timeout,
    InvalidProtocol,
    DeadlineExceeded,
    ClientExit,
}

#[derive(Clone, Debug)]
pub struct RawFailure {
    pub index: Option<usize>,
    pub kind: RawFailureKind,
    /// False if the row could not be unambiguously matched to the frozen input.
    /// Such evidence is an audit observation, never a candidate SLO assertion.
    pub matched_input: bool,
}

#[derive(Debug)]
pub struct RunAudit {
    pub run: Option<core::Run>,
    /// Matched rows plus explicit Missing placeholders, retained even when a
    /// structural issue prevents forming a paired Run. Ambiguous duplicates
    /// become Missing; they are never resolved by choosing the first/last row.
    pub observations: Vec<core::Request>,
    pub issues: Vec<ReaderIssue>,
    pub issues_omitted: usize,
    pub raw_failures: Vec<RawFailure>,
    pub raw_failures_omitted: usize,
    pub observed_failures: usize,
    pub matched_failures: usize,
    pub observed_timeouts: usize,
    pub source_summary: Option<serde_json::Value>,
}

#[derive(Debug)]
pub struct Audit {
    /// Baseline, candidate in manifest order. Structural errors do not erase
    /// these audits or the explicit failure observations inside them.
    pub runs: Vec<RunAudit>,
    pub identity_issues: Vec<ReaderIssueKind>,
    pub report: Option<Result<core::Report, core::Issue>>,
    pub candidate_observed_fail: bool,
    pub method_status: &'static str,
}
