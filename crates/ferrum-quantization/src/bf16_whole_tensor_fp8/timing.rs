//! Per-source-group diagnostics; neither wall nor thread CPU is GPU timing.
//! Disabled DEBUG logging reads no clocks. A missing CPU clock stays unavailable.

use super::WeightComponentSpec;
use std::time::{Duration, Instant};

const TARGET: &str = "ferrum_quantization::bf16_whole_tensor_fp8";

#[derive(Clone, Copy)]
pub(super) enum Phase {
    SourceResolve,
    Amax,
    HostAllocateZeroed,
    Encode,
    LoraCompose,
}

struct Start {
    wall: Instant,
    cpu: Option<Duration>,
}

impl Start {
    fn now() -> Self {
        Self {
            wall: Instant::now(),
            cpu: thread_cpu(),
        }
    }

    fn elapsed(self) -> Measurement {
        // Keep CPU reads inside the wall envelope, as in backend enqueue timing.
        let cpu = cpu_delta(self.cpu, thread_cpu());
        Measurement {
            wall: self.wall.elapsed(),
            cpu,
        }
    }
}

#[derive(Clone, Copy)]
struct Measurement {
    wall: Duration,
    cpu: Option<Duration>,
}

impl Measurement {
    fn add(self, other: Self) -> Self {
        Self {
            wall: self.wall.saturating_add(other.wall),
            cpu: self.cpu.zip(other.cpu).and_then(|(a, b)| a.checked_add(b)),
        }
    }
}

pub(super) struct GroupTiming {
    started: Option<Start>,
    phases: [Option<Measurement>; 5],
    pub(super) mode: &'static str,
}

impl GroupTiming {
    pub(super) fn new() -> Self {
        Self::with_enabled(tracing::enabled!(target: TARGET, tracing::Level::DEBUG))
    }

    fn with_enabled(enabled: bool) -> Self {
        Self {
            started: enabled.then(Start::now),
            phases: [None; 5],
            mode: "converted",
        }
    }

    pub(super) fn measure<T>(&mut self, phase: Phase, operation: impl FnOnce() -> T) -> T {
        if self.started.is_none() {
            return operation();
        }
        let start = Start::now();
        let result = operation();
        let elapsed = start.elapsed();
        let slot = &mut self.phases[phase as usize];
        *slot = Some(slot.map_or(elapsed, |previous| previous.add(elapsed)));
        result
    }

    pub(super) fn report(
        self,
        sources: &[&WeightComponentSpec],
        outputs: &[&WeightComponentSpec],
        success: bool,
    ) {
        let Some(started) = self.started else {
            return;
        };
        let total = started.elapsed();
        let wall = |phase: Phase| self.phases[phase as usize].map(|value| micros(value.wall));
        let cpu =
            |phase: Phase| self.phases[phase as usize].and_then(|value| value.cpu.map(micros));
        tracing::debug!(
            target: TARGET,
            mode = self.mode,
            success,
            source_count = sources.len(),
            output_count = outputs.len(),
            first_source = ?sources.first().map(|spec| &spec.id),
            first_output = ?outputs.first().map(|spec| &spec.id),
            total_wall_us = micros(total.wall),
            total_thread_cpu_us = ?total.cpu.map(micros),
            source_resolve_wall_us = ?wall(Phase::SourceResolve),
            source_resolve_thread_cpu_us = ?cpu(Phase::SourceResolve),
            amax_wall_us = ?wall(Phase::Amax),
            amax_thread_cpu_us = ?cpu(Phase::Amax),
            host_allocate_zeroed_wall_us = ?wall(Phase::HostAllocateZeroed),
            host_allocate_zeroed_thread_cpu_us = ?cpu(Phase::HostAllocateZeroed),
            encode_wall_us = ?wall(Phase::Encode),
            encode_thread_cpu_us = ?cpu(Phase::Encode),
            lora_compose_wall_us = ?wall(Phase::LoraCompose),
            lora_compose_caller_thread_cpu_us = ?cpu(Phase::LoraCompose),
            "BF16 whole-tensor FP8 source-group materialization timing"
        );
    }
}

fn micros(duration: Duration) -> u64 {
    u64::try_from(duration.as_micros()).unwrap_or(u64::MAX)
}

fn cpu_delta(start: Option<Duration>, end: Option<Duration>) -> Option<Duration> {
    start
        .zip(end)
        .and_then(|(start, end)| end.checked_sub(start))
}

#[cfg(any(target_os = "macos", target_os = "linux", target_os = "android"))]
fn thread_cpu() -> Option<Duration> {
    let mut time = std::mem::MaybeUninit::<libc::timespec>::uninit();
    // SAFETY: the OS writes one timespec on success. This reads only the calling
    // thread's CPU clock; failure never becomes a measured zero duration.
    if unsafe { libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, time.as_mut_ptr()) } != 0 {
        return None;
    }
    // SAFETY: clock_gettime returned success and initialized the timespec.
    let time = unsafe { time.assume_init() };
    let seconds = u64::try_from(time.tv_sec).ok()?;
    let nanos = u32::try_from(time.tv_nsec).ok()?;
    (nanos < 1_000_000_000).then(|| Duration::new(seconds, nanos))
}

#[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "android")))]
fn thread_cpu() -> Option<Duration> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_or_backwards_cpu_samples_do_not_become_zero_measurements() {
        let one = Duration::from_micros(1);
        let two = Duration::from_micros(2);
        assert_eq!(cpu_delta(Some(one), Some(two)), Some(one));
        for (start, end) in [(None, Some(two)), (Some(one), None), (Some(two), Some(one))] {
            assert_eq!(cpu_delta(start, end), None);
        }
        let measured = Measurement {
            wall: two,
            cpu: Some(one),
        };
        let unavailable = Measurement {
            wall: one,
            cpu: None,
        };
        assert_eq!(measured.add(unavailable).wall, one + two);
        assert_eq!(measured.add(unavailable).cpu, None);
    }

    #[test]
    fn disabled_observation_and_recorded_errors_preserve_operation_results() {
        let mut disabled = GroupTiming::with_enabled(false);
        let mut calls = 0;
        let result: Result<(), &str> = disabled.measure(Phase::SourceResolve, || {
            calls += 1;
            Err("source unavailable")
        });
        assert_eq!(result, Err("source unavailable"));
        assert_eq!(calls, 1);
        assert!(disabled.started.is_none());
        assert!(disabled.phases.iter().all(Option::is_none));

        let mut enabled = GroupTiming::with_enabled(true);
        let result: Result<(), &str> =
            enabled.measure(Phase::HostAllocateZeroed, || Err("capacity"));
        assert_eq!(result, Err("capacity"));
        assert!(enabled.phases[Phase::HostAllocateZeroed as usize].is_some());
        assert!(enabled.phases[Phase::Encode as usize].is_none());
    }
}
