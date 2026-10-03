//! Test-build-only capture of original cold geometry. Never a calibration source.
//! Spawned workers do not inherit this task scope. Synchronous selection/writes
//! cannot be forcibly interrupted at the original deadline; no budget is reset.
use super::profile_export::StagedFile;
use ferrum_scheduler::implementations::continuous::cost_model::structured_v2::{
    StructuredInputGeometryWorkV1, StructuredSettingsV2,
};
use ferrum_types::{FerrumError, Result};
use serde::{ser::SerializeSeq, Deserialize, Serialize, Serializer};
use std::{
    cell::RefCell, future::Future, io::BufWriter, num::NonZeroU64, path::PathBuf, time::Instant,
};

const BUFFER_BYTES: usize = 64 * 1024;
const MAX_CAPTURE_BYTES: u64 = 256 * 1024 * 1024;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CaptureOptions {
    pub output: PathBuf,
    pub maximum_bytes: NonZeroU64,
    /// Hash of the bounded test input; the original guard binds its identities.
    pub input_sha256: [u8; 32],
}

/// Only this module attests completion, after original cleanup and publication.
#[derive(Debug, Serialize)]
pub struct CaptureReport {
    path: PathBuf,
    bytes: u64,
    sha256: String,
    complete: bool,
    failure: Option<&'static str>,
    expected_matrices: usize,
    observed_matrices: usize,
    elapsed_ns: u128,
    diagnostic_buffer_bytes: usize,
    diagnostic_retained_bytes: usize,
}
impl CaptureReport {
    pub fn is_complete(&self) -> bool {
        self.complete
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Phase {
    Armed,
    InventoryRetired,
    SeriesRetired,
    Shutdown,
}
struct State {
    writer: Option<BufWriter<StagedFile>>,
    started: Instant,
    phase: Phase,
    failure: Option<&'static str>,
    expected: Option<usize>,
    matrices: usize,
    results: usize,
    selection_ended: bool,
    retained_bytes: usize,
}
tokio::task_local! { static CAPTURE: RefCell<State>; }
fn invalid(message: impl std::fmt::Display) -> FerrumError {
    FerrumError::invalid_request(format!("test geometry capture: {message}"))
}

/// Explicit typed scope around the unchanged product startup future. A returned
/// error alone cannot attest successful capture or cleanup; inspect the report.
pub async fn capture<T>(
    options: CaptureOptions,
    run: impl Future<Output = T>,
) -> Result<(T, CaptureReport)> {
    if armed() || options.maximum_bytes.get() > MAX_CAPTURE_BYTES {
        return Err(invalid(
            "nested capture or diagnostic byte limit unsupported",
        ));
    }
    let mut file =
        StagedFile::create(&options.output, options.maximum_bytes.get()).map_err(invalid)?;
    file.preserve_unpublished();
    let writer = BufWriter::with_capacity(BUFFER_BYTES, file);
    let retained_bytes = std::mem::size_of::<RefCell<State>>()
        .checked_add(writer.capacity())
        .and_then(|n| n.checked_add(writer.get_ref().retained_path_bytes()?))
        .ok_or_else(|| invalid("diagnostic retained byte overflow"))?;
    let mut state = State {
        writer: Some(writer),
        started: Instant::now(),
        phase: Phase::Armed,
        failure: None,
        expected: None,
        matrices: 0,
        results: 0,
        selection_ended: false,
        retained_bytes,
    };
    #[derive(Serialize)]
    struct Header {
        kind: &'static str,
        input_sha256: [u8; 32],
        maximum_bytes: u64,
        buffer_bytes: usize,
    }
    state.emit(&Header {
        kind: "ferrum.test.cold_geometry.v1",
        input_sha256: options.input_sha256,
        maximum_bytes: options.maximum_bytes.get(),
        buffer_bytes: BUFFER_BYTES,
    });
    // The file owns its bounded paths; do not retain another PathBuf through startup.
    drop(options);
    CAPTURE
        .scope(RefCell::new(state), async {
            let result = run.await;
            let report = CAPTURE.with(|state| state.borrow_mut().finish());
            Ok((result, report?))
        })
        .await
}
impl State {
    fn fail(&mut self, reason: &'static str) {
        self.failure.get_or_insert(reason);
    }
    fn emit(&mut self, value: &impl Serialize) {
        if self.failure.is_some() {
            return;
        }
        let writer = self
            .writer
            .as_mut()
            .expect("writer consumed only at scope exit");
        if serde_json::to_writer(&mut *writer, value).is_err()
            || std::io::Write::write_all(writer, b"\n").is_err()
        {
            self.fail("diagnostic_write_or_byte_limit");
        }
    }
    fn finish(&mut self) -> Result<CaptureReport> {
        if self.phase != Phase::Shutdown
            || !self.selection_ended
            || self.expected != Some(self.matrices)
            || self.matrices == 0
            || self.results != self.matrices
        {
            self.fail("incomplete_inventory_or_cleanup");
        }
        #[derive(Serialize)]
        struct Footer {
            kind: &'static str,
            matrices: usize,
        }
        self.emit(&Footer {
            kind: "completed_after_shutdown",
            matrices: self.matrices,
        });
        let writer = self.writer.take().expect("one capture finish");
        let buffer_bytes = writer.capacity();
        let file = match writer.into_inner() {
            Ok(file) => file,
            Err(error) => {
                self.fail("diagnostic_flush_or_byte_limit");
                // Preserve actual partial bytes without retrying buffered I/O.
                error.into_inner().into_parts().0
            }
        };
        let mut receipt = file.unpublished_receipt();
        let mut complete = false;
        if self.failure.is_none() {
            match file.publish() {
                Ok(published) => {
                    receipt = published;
                    complete = true;
                }
                Err(_) => self.fail("diagnostic_publish_failed"),
            }
        }
        Ok(CaptureReport {
            path: receipt.path,
            bytes: receipt.bytes,
            sha256: receipt.sha256,
            complete,
            failure: self.failure,
            expected_matrices: self.expected.unwrap_or(0),
            observed_matrices: self.matrices,
            elapsed_ns: self.started.elapsed().as_nanos(),
            diagnostic_buffer_bytes: buffer_bytes,
            diagnostic_retained_bytes: self.retained_bytes,
        })
    }
}
fn with_state(f: impl FnOnce(&mut State)) {
    let _ = CAPTURE.try_with(|state| f(&mut state.borrow_mut()));
}
pub(crate) fn armed() -> bool {
    CAPTURE.try_with(|_| ()).is_ok()
}

pub(crate) fn startup_inputs(
    config: &ferrum_types::EngineConfig,
    templates: &[crate::AutomaticCostProbeTemplate],
) {
    #[derive(Serialize)]
    struct Startup<'a> {
        kind: &'static str,
        config: &'a ferrum_types::EngineConfig,
        templates: Templates<'a>,
    }
    with_state(|state| {
        state.emit(&Startup {
            kind: "actual_startup_inputs",
            config,
            templates: Templates(templates),
        })
    });
}
struct Templates<'a>(&'a [crate::AutomaticCostProbeTemplate]);
impl Serialize for Templates<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        #[derive(Serialize)]
        struct Template<'a> {
            output: crate::AutomaticCostProbeOutput,
            request_bytes: &'a [u8],
        }
        let mut seq = serializer.serialize_seq(Some(self.0.len()))?;
        for t in self.0 {
            seq.serialize_element(&Template {
                output: t.output(),
                request_bytes: t.serialized_request(),
            })?;
        }
        seq.end()
    }
}
// Array positions are the same original case indices recorded by matrix().
// Case and its optional acquisition contain bounded scalar identities only;
// borrow the existing table and prompt lengths rather than cloning either.
pub(crate) fn selection_inputs<C: Serialize>(
    cases: &[C],
    prompts: &[usize],
    chunk: usize,
    prefill_row_ceiling: Option<std::num::NonZeroU32>,
) {
    #[derive(Serialize)]
    struct Inputs<'a, C> {
        kind: &'static str,
        cases: &'a [C],
        prompts: &'a [usize],
        chunk: usize,
        prefill_row_ceiling: Option<std::num::NonZeroU32>,
    }
    with_state(|state| {
        state.emit(&Inputs {
            kind: "original_cases",
            cases,
            prompts,
            chunk,
            prefill_row_ceiling,
        });
    });
}

pub(crate) fn begin_selection(expected: usize) {
    with_state(|state| {
        if state.expected.replace(expected).is_some() {
            state.fail("more_than_one_original_selection");
        }
    });
}
pub(crate) fn population(index: usize, key: &impl Serialize) {
    #[derive(Serialize)]
    struct Population<'a, K> {
        kind: &'static str,
        population_index: usize,
        key: &'a K,
    }
    with_state(|state| {
        state.emit(&Population {
            kind: "population",
            population_index: index,
            key,
        })
    });
}
pub(crate) fn end_selection() {
    with_state(|state| state.selection_ended = true);
}

struct AxisBits<'a>(&'a [f64]);
impl Serialize for AxisBits<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        let mut seq = serializer.serialize_seq(Some(self.0.len()))?;
        for value in self.0 {
            seq.serialize_element(&value.to_bits())?;
        }
        seq.end()
    }
}
struct Rows<'a>(&'a [&'a [f64]]);
impl Serialize for Rows<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        let mut seq = serializer.serialize_seq(Some(self.0.len()))?;
        for row in self.0 {
            seq.serialize_element(&AxisBits(row))?;
        }
        seq.end()
    }
}
pub(crate) fn matrix(
    cases: &[usize],
    rows: &[&[f64]],
    anchors: &[usize],
    settings: &StructuredSettingsV2,
    work: &StructuredInputGeometryWorkV1,
    maximum_scratch_bytes: usize,
) {
    #[derive(Serialize)]
    struct Matrix<'a> {
        kind: &'static str,
        ordinal: usize,
        cases: &'a [usize],
        axis_bits: Rows<'a>,
        mandatory_anchors: &'a [usize],
        settings: &'a StructuredSettingsV2,
        visits_before: u64,
        maximum_visits: u64,
        exhausted_before: bool,
        maximum_scratch_bytes: usize,
    }
    with_state(|state| {
        let ordinal = state.matrices;
        state.matrices += 1;
        state.emit(&Matrix {
            kind: "original_matrix",
            ordinal,
            cases,
            axis_bits: Rows(rows),
            mandatory_anchors: anchors,
            settings,
            visits_before: work.visits(),
            maximum_visits: work.maximum_visits(),
            exhausted_before: work.exhausted(),
            maximum_scratch_bytes,
        });
    });
}
pub(crate) fn matrix_result(
    audit: &impl Serialize,
    gap: &impl Serialize,
    selected_cases: &[usize],
    work: &StructuredInputGeometryWorkV1,
) {
    #[derive(Serialize)]
    struct Record<'a, A, G> {
        kind: &'static str,
        ordinal: usize,
        audit: &'a A,
        gap: &'a G,
        final_selected_cases: &'a [usize],
        visits_after: u64,
        exhausted_after: bool,
    }
    with_state(|state| {
        let ordinal = state.results;
        state.results += 1;
        state.emit(&Record {
            kind: "original_result",
            ordinal,
            audit,
            gap,
            final_selected_cases: selected_cases,
            visits_after: work.visits(),
            exhausted_after: work.exhausted(),
        });
    });
}
pub(crate) fn inventory_retired(
    complete: bool,
    deadline: tokio::time::Instant,
    actual_requests: usize,
    actual_actions: usize,
    selected_requests: usize,
    selected_actions: usize,
) {
    #[derive(Serialize)]
    struct Retired {
        kind: &'static str,
        complete: bool,
        deadline_expired: bool,
        remaining_ns: u128,
        actual_requests: usize,
        actual_actions: usize,
        selected_requests: usize,
        selected_actions: usize,
    }
    with_state(|state| {
        let now = tokio::time::Instant::now();
        state.emit(&Retired {
            kind: "inventory_retired",
            complete,
            deadline_expired: now >= deadline,
            remaining_ns: deadline.saturating_duration_since(now).as_nanos(),
            actual_requests,
            actual_actions,
            selected_requests,
            selected_actions,
        });
        if complete && state.phase == Phase::Armed {
            state.phase = Phase::InventoryRetired;
        } else {
            state.fail("inventory_not_complete_or_retired");
        }
    });
}
pub(crate) fn series_retired(success: bool) {
    with_state(|state| {
        if success && state.phase == Phase::InventoryRetired {
            state.phase = Phase::SeriesRetired;
        } else {
            state.fail("series_cleanup_failed");
        }
    });
}
pub(crate) fn shutdown_finished(success: bool) {
    with_state(|state| {
        if success && state.phase == Phase::SeriesRetired {
            state.phase = Phase::Shutdown;
        } else {
            state.fail("tracked_drain_or_shutdown_incomplete");
        }
    });
}
#[cfg(test)]
mod tests;
