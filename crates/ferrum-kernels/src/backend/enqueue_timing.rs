//! Host boundaries for command enqueue; CPU time is diagnostic, never authority.

use std::time::Duration;
#[cfg(feature = "cuda")]
use std::time::Instant;

use ferrum_interfaces::vnext::{DeviceEnqueueStage, DeviceSubmissionTimingSink};

#[cfg(feature = "cuda")]
pub(crate) fn measure<S: DeviceSubmissionTimingSink, T>(
    sink: &S,
    stage: DeviceEnqueueStage,
    thread_cpu: bool,
    operation: impl FnOnce() -> T,
) -> T {
    measure_with_clock(sink, stage, thread_cpu, &mut HostClock, operation)
}

trait Clock {
    type Start;
    fn start_wall(&mut self) -> Self::Start;
    fn elapsed_wall(&mut self, started: Self::Start) -> Duration;
    fn thread_cpu(&mut self) -> Option<Duration>;
}

#[cfg(feature = "cuda")]
struct HostClock;

#[cfg(feature = "cuda")]
impl Clock for HostClock {
    type Start = Instant;

    fn start_wall(&mut self) -> Instant {
        Instant::now()
    }

    fn elapsed_wall(&mut self, started: Instant) -> Duration {
        started.elapsed()
    }

    fn thread_cpu(&mut self) -> Option<Duration> {
        #[cfg(target_os = "linux")]
        {
            let mut time = std::mem::MaybeUninit::<libc::timespec>::uninit();
            // This reads only the calling submission thread's CPU clock.
            if unsafe { libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, time.as_mut_ptr()) } != 0
            {
                return None;
            }
            let time = unsafe { time.assume_init() };
            let seconds = u64::try_from(time.tv_sec).ok()?;
            let nanos = u32::try_from(time.tv_nsec).ok()?;
            (nanos < 1_000_000_000).then(|| Duration::new(seconds, nanos))
        }
        #[cfg(not(target_os = "linux"))]
        {
            None
        }
    }
}

#[inline]
fn measure_with_clock<S: DeviceSubmissionTimingSink, C: Clock, T>(
    sink: &S,
    stage: DeviceEnqueueStage,
    thread_cpu: bool,
    clock: &mut C,
    operation: impl FnOnce() -> T,
) -> T {
    if !S::ENABLED {
        return operation();
    }
    // Both clocks bracket exactly this invocation. Wall time also includes the
    // adjacent CPU-clock reads; neither value is GPU time or removable CPU work.
    let started = clock.start_wall();
    let cpu_started = thread_cpu.then(|| clock.thread_cpu()).flatten();
    let result = operation();
    let cpu_ended = thread_cpu.then(|| clock.thread_cpu()).flatten();
    let wall = clock.elapsed_wall(started);
    let cpu = cpu_started
        .zip(cpu_ended)
        .and_then(|(start, end)| end.checked_sub(start));
    sink.record_device_enqueue(stage, wall, cpu);
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrum_interfaces::vnext::{DeviceSubmissionStage, DisabledDeviceSubmissionTimingSink};
    use std::collections::VecDeque;
    use std::sync::Mutex;

    #[derive(Default)]
    struct Sink(Mutex<Vec<(DeviceEnqueueStage, Duration, Option<Duration>)>>);

    impl DeviceSubmissionTimingSink for Sink {
        const ENABLED: bool = true;
        fn record_device_submission(&self, _: DeviceSubmissionStage, _: Duration) {
            panic!("child timing must not record an overlapping parent");
        }
        fn record_device_enqueue(
            &self,
            stage: DeviceEnqueueStage,
            wall: Duration,
            cpu: Option<Duration>,
        ) {
            self.0.lock().unwrap().push((stage, wall, cpu));
        }
    }

    struct TestClock {
        cpu: VecDeque<Option<Duration>>,
        wall_reads: usize,
        forbidden: bool,
    }

    impl Clock for TestClock {
        type Start = ();
        fn start_wall(&mut self) {
            assert!(!self.forbidden);
            self.wall_reads += 1;
        }
        fn elapsed_wall(&mut self, _: ()) -> Duration {
            assert!(!self.forbidden);
            self.wall_reads += 1;
            Duration::from_millis(7)
        }
        fn thread_cpu(&mut self) -> Option<Duration> {
            assert!(!self.forbidden);
            self.cpu.pop_front().expect("unexpected CPU clock read")
        }
    }

    #[test]
    fn disabled_enqueue_timing_reads_no_clocks_or_sink_and_calls_once() {
        let mut clock = TestClock {
            cpu: VecDeque::new(),
            wall_reads: 0,
            forbidden: true,
        };
        let mut calls = 0;
        let result = measure_with_clock(
            &DisabledDeviceSubmissionTimingSink,
            DeviceEnqueueStage::GraphLaunch,
            true,
            &mut clock,
            || {
                calls += 1;
                Err::<(), _>("driver error")
            },
        );
        assert_eq!(result, Err("driver error"));
        assert_eq!(calls, 1);
        assert_eq!(clock.wall_reads, 0);
    }

    #[test]
    fn graph_enqueue_timing_preserves_return_and_error_with_paired_cpu() {
        let sink = Sink::default();
        let mut clock = TestClock {
            cpu: [
                Some(Duration::from_millis(2)),
                Some(Duration::from_millis(5)),
            ]
            .into(),
            wall_reads: 0,
            forbidden: false,
        };
        let mut calls = 0;
        let result = measure_with_clock(
            &sink,
            DeviceEnqueueStage::GraphLaunch,
            true,
            &mut clock,
            || {
                calls += 1;
                Err::<usize, _>(17)
            },
        );
        assert_eq!(result, Err(17));
        assert_eq!(calls, 1);
        assert_eq!(clock.wall_reads, 2);
        assert_eq!(
            *sink.0.lock().unwrap(),
            [(
                DeviceEnqueueStage::GraphLaunch,
                Duration::from_millis(7),
                Some(Duration::from_millis(3))
            )]
        );
        let value = measure_with_clock(
            &sink,
            DeviceEnqueueStage::EagerCommand,
            false,
            &mut clock,
            || 23,
        );
        assert_eq!(value, 23);
        assert_eq!(
            sink.0.lock().unwrap()[1],
            (
                DeviceEnqueueStage::EagerCommand,
                Duration::from_millis(7),
                None
            )
        );
    }

    #[test]
    fn unavailable_or_regressing_thread_clock_keeps_wall_observation() {
        for cpu in [
            [None, Some(Duration::from_millis(3))],
            [Some(Duration::from_millis(3)), None],
            [
                Some(Duration::from_millis(3)),
                Some(Duration::from_millis(2)),
            ],
        ] {
            let sink = Sink::default();
            let mut clock = TestClock {
                cpu: cpu.into(),
                wall_reads: 0,
                forbidden: false,
            };
            measure_with_clock(
                &sink,
                DeviceEnqueueStage::GraphLaunch,
                true,
                &mut clock,
                || (),
            );
            assert_eq!(
                *sink.0.lock().unwrap(),
                [(
                    DeviceEnqueueStage::GraphLaunch,
                    Duration::from_millis(7),
                    None
                )]
            );
        }
    }
}
