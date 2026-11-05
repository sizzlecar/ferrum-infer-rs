//! Opt-in CPU diagnostics for the ordinary managed archive conversion path.
//! Source touching changes residency and is never performed by `component`.
use serde::Serialize;
use std::{num::NonZeroUsize, time::Instant};

#[derive(Clone, Copy, Debug, Default, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DenseArchiveSourceTouch {
    #[default]
    AsMapped,
    /// Read one byte per stride and the final byte. This is a residency probe,
    /// not a full read, a cold-cache guarantee, or disk bandwidth measurement.
    Stride(NonZeroUsize),
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub struct Bf16ToF16BlockCounts {
    pub simd_blocks_attempted: u64,
    pub simd_blocks_converted: u64,
    pub scalar_fallback_blocks: u64,
    pub scalar_tail_elements: u64,
    pub portable_scalar_elements: u64,
}

impl Bf16ToF16BlockCounts {
    pub fn backend() -> &'static str {
        if cfg!(all(target_arch = "x86_64", target_endian = "little")) {
            "sse2_8_bf16_lanes"
        } else if cfg!(all(target_arch = "aarch64", target_endian = "little")) {
            "neon_8_bf16_lanes"
        } else {
            "portable_scalar"
        }
    }
}

/// Process counters can include other threads. They distinguish CPU time and
/// fault/block-I/O observations; none is a direct device or disk duration.
#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct DenseArchivePhaseTiming {
    pub calls: u64,
    pub process_counter_calls: u64,
    pub wall_ns: u64,
    pub process_user_cpu_ns: Option<u64>,
    pub process_system_cpu_ns: Option<u64>,
    pub minor_faults: Option<u64>,
    pub major_faults: Option<u64>,
    pub input_block_operations: Option<u64>,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct DenseArchiveMaterializationReport {
    pub completed: bool,
    pub borrowed: bool,
    pub composite: bool,
    pub affine: bool,
    pub source_parts: usize,
    pub source_bytes: u64,
    pub destination_bytes: u64,
    pub source_touch_loads: u64,
    pub total_wall_ns: u64,
    pub metadata_validation: DenseArchivePhaseTiming,
    pub source_touch: DenseArchivePhaseTiming,
    pub reserve_allocate_zeroed: DenseArchivePhaseTiming,
    /// Includes source loads, destination stores/first-touch and direct writes
    /// to final packed offsets. May still include page faults after a touch.
    pub conversion_and_direct_packing: DenseArchivePhaseTiming,
    pub payload_assembly: DenseArchivePhaseTiming,
    pub bf16_to_f16: Bf16ToF16BlockCounts,
}

impl DenseArchiveMaterializationReport {
    pub fn unattributed_wall_ns(&self) -> u64 {
        let attributed = [
            self.metadata_validation.wall_ns,
            self.source_touch.wall_ns,
            self.reserve_allocate_zeroed.wall_ns,
            self.conversion_and_direct_packing.wall_ns,
            self.payload_assembly.wall_ns,
        ]
        .into_iter()
        .fold(0_u64, u64::saturating_add);
        self.total_wall_ns.saturating_sub(attributed)
    }
}

#[derive(Clone, Copy)]
struct ProcessUsage {
    user_ns: u64,
    system_ns: u64,
    minor_faults: u64,
    major_faults: u64,
    input_blocks: u64,
}

fn process_usage() -> Option<ProcessUsage> {
    #[cfg(unix)]
    {
        let mut value = std::mem::MaybeUninit::<libc::rusage>::uninit();
        // SAFETY: getrusage initializes this complete structure on success.
        if unsafe { libc::getrusage(libc::RUSAGE_SELF, value.as_mut_ptr()) } != 0 {
            return None;
        }
        // SAFETY: success was checked above.
        let value = unsafe { value.assume_init() };
        let ns = |time: libc::timeval| {
            u64::try_from(time.tv_sec)
                .ok()?
                .checked_mul(1_000_000_000)?
                .checked_add(u64::try_from(time.tv_usec).ok()?.checked_mul(1000)?)
        };
        Some(ProcessUsage {
            user_ns: ns(value.ru_utime)?,
            system_ns: ns(value.ru_stime)?,
            minor_faults: u64::try_from(value.ru_minflt).ok()?,
            major_faults: u64::try_from(value.ru_majflt).ok()?,
            input_blocks: u64::try_from(value.ru_inblock).ok()?,
        })
    }
    #[cfg(not(unix))]
    None
}

pub(super) struct Started {
    wall: Instant,
    usage: Option<ProcessUsage>,
}

pub(super) fn start<const ENABLED: bool>() -> Option<Started> {
    ENABLED.then(|| Started {
        wall: Instant::now(),
        usage: process_usage(),
    })
}

pub(super) fn finish(started: Option<Started>, timing: &mut DenseArchivePhaseTiming) {
    let Some(started) = started else { return };
    let usage = process_usage();
    timing.calls += 1;
    timing.wall_ns = timing
        .wall_ns
        .saturating_add(u64::try_from(started.wall.elapsed().as_nanos()).unwrap_or(u64::MAX));
    let add = |target: &mut Option<u64>, before: u64, after: u64| {
        *target = Some(
            target
                .unwrap_or(0)
                .saturating_add(after.saturating_sub(before)),
        );
    };
    if let (Some(before), Some(after)) = (started.usage, usage) {
        timing.process_counter_calls += 1;
        add(
            &mut timing.process_user_cpu_ns,
            before.user_ns,
            after.user_ns,
        );
        add(
            &mut timing.process_system_cpu_ns,
            before.system_ns,
            after.system_ns,
        );
        add(
            &mut timing.minor_faults,
            before.minor_faults,
            after.minor_faults,
        );
        add(
            &mut timing.major_faults,
            before.major_faults,
            after.major_faults,
        );
        add(
            &mut timing.input_block_operations,
            before.input_blocks,
            after.input_blocks,
        );
    }
}

pub(super) fn touch<const ENABLED: bool>(
    bytes: &[u8],
    policy: DenseArchiveSourceTouch,
    report: &mut DenseArchiveMaterializationReport,
) {
    if !ENABLED {
        return;
    }
    let DenseArchiveSourceTouch::Stride(stride) = policy else {
        return;
    };
    let started = start::<true>();
    let mut checksum = 0_u8;
    for offset in (0..bytes.len()).step_by(stride.get()) {
        checksum = checksum.wrapping_add(std::hint::black_box(bytes[offset]));
        report.source_touch_loads += 1;
    }
    if let Some(last) = bytes.last() {
        checksum = checksum.wrapping_add(std::hint::black_box(*last));
        report.source_touch_loads += 1;
    }
    std::hint::black_box(checksum);
    finish(started, &mut report.source_touch);
}
