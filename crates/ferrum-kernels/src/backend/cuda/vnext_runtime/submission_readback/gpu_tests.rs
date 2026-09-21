use super::*;
use std::ffi::c_void;
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread::JoinHandle;
use std::time::Duration;

mod upload_gate_tests;

type TestSnapshot = PinnedReadbackSnapshot<Arc<PinnedHostStorage>>;

struct CallbackGate {
    entered: Sender<()>,
    release: Receiver<()>,
    exited: Arc<AtomicBool>,
}

// CUDA forbids CUDA calls from this callback. Only Rust channel operations and
// an atomic store occur here, including all destructors of CallbackGate.
unsafe extern "C" fn wait_for_release(argument: *mut c_void) {
    let gate = unsafe { Box::from_raw(argument.cast::<CallbackGate>()) };
    let _ = gate.entered.send(());
    // An independent bound also releases the stream if the test thread itself
    // stops making progress. Successful pending assertions require exited=false.
    let _ = gate.release.recv_timeout(Duration::from_secs(20));
    gate.exited.store(true, Ordering::Release);
}

struct DrainGate {
    release: Option<Sender<()>>,
    stream: Arc<CudaStream>,
    reader: Option<JoinHandle<()>>,
    // Keep the exact DMA source and pinned destination alive through cleanup.
    _snapshot: Arc<TestSnapshot>,
}

impl DrainGate {
    fn release_and_drain(&mut self) -> Result<(), DriverError> {
        if let Some(release) = self.release.take() {
            let _ = release.send(());
        }
        self.stream.synchronize()
    }
}

impl Drop for DrainGate {
    fn drop(&mut self) {
        // Any assertion/error must release the callback before a CUDA drain or
        // pinned allocation destructor could wait for work behind that callback.
        let _ = self.release_and_drain();
        if let Some(reader) = self.reader.take() {
            let _ = reader.join();
        }
    }
}

fn write_words(region: CudaBufferRegion, first: u8, second: u8) -> CudaDeviceCommand {
    CudaDeviceCommand::operation(
        "test.readback.u32-write",
        vec![region],
        move |stream, regions| {
            for (offset, value) in [(0, first), (4, second)] {
                unsafe {
                    cudarc::driver::result::memset_d8_async(
                        regions[0].device_ptr + offset,
                        value,
                        4,
                        stream.cu_stream(),
                    )
                }
                .map_err(|error| {
                    CudaDeviceRuntimeError::driver("test readback U32 write", error)
                })?;
            }
            Ok(())
        },
    )
    .unwrap()
}

fn record_fence(stream: &CudaDeviceStream, commands: Vec<CudaDeviceCommand>) -> CudaDeviceFence {
    let event = stream.stream.record_event(None).unwrap();
    stream.state.submission_recorded().unwrap();
    CudaDeviceFence {
        event,
        timing: CudaFenceTiming::NotRequested,
        command_timing: CudaFenceCommandTiming::NotRequested,
        attribution: None,
        stream_state: stream.state.clone(),
        terminal_accounted: AtomicBool::new(false),
        _stream: stream.stream.clone(),
        _blas: stream.blas.clone(),
        _commands: commands,
    }
}

#[test]
#[ignore = "requires an actual CUDA device"]
fn staged_u32_snapshot_reads_parent_while_same_lane_child_is_pending_on_cuda() {
    let runtime = CudaDeviceRuntime::new(
        crate::backend::cuda::vnext_ops::cuda_vnext_runtime_config(
            0,
            DeviceId::new("device.test.submission-readback").unwrap(),
            AttentionExecutionPolicy::Portable,
        )
        .unwrap(),
    )
    .unwrap();
    let stream = runtime.create_stream().unwrap();
    let base = stream.stream.alloc_zeros::<u8>(16).unwrap();
    let pointer = base.device_ptr(&stream.stream).0;
    let region = CudaBufferRegion {
        _allocation: Arc::new(CudaAllocation {
            _base: base,
            aligned_ptr: pointer,
            requested_bytes: 16,
        }),
        _core_retention: None,
        reusable_address_scope: None,
        runtime_instance: runtime.runtime_instance,
        device_ptr: pointer + 4,
        length_bytes: 8,
        element_type: ElementType::U32,
    };
    // Core's typed request/budget/terminal tests cover authority. This test uses
    // the exact private DMA/read helper called by production prepare(), with a
    // real retained allocation and nonzero source/destination offsets.
    let host_storage = Arc::new(allocate_host_storage(&region, 16).unwrap());
    let snapshot = Arc::new(
        PinnedReadbackSnapshot::new(region.clone(), Arc::clone(&host_storage), 16).unwrap(),
    );
    let parent_write = write_words(region.clone(), 0x12, 0x34);
    stream.state.begin_submission().unwrap();
    parent_write.enqueue(&stream.stream, &stream.blas).unwrap();
    snapshot.enqueue(&stream.stream, 4..12).unwrap();
    let parent = record_fence(&stream, vec![parent_write]);

    let (entered_tx, entered_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let exited = Arc::new(AtomicBool::new(false));
    let mut cleanup = DrainGate {
        release: Some(release_tx),
        stream: stream.stream.clone(),
        reader: None,
        _snapshot: Arc::clone(&snapshot),
    };
    let gate = Box::into_raw(Box::new(CallbackGate {
        entered: entered_tx,
        release: release_rx,
        exited: Arc::clone(&exited),
    }));
    stream.state.begin_submission().unwrap();
    let launch = unsafe {
        cudarc::driver::result::stream::launch_host_function(
            stream.stream.cu_stream(),
            wait_for_release,
            gate.cast(),
        )
    };
    if let Err(error) = launch {
        // A failed enqueue never transfers the callback box to CUDA.
        drop(unsafe { Box::from_raw(gate) });
        panic!("enqueue same-lane child gate: {error}");
    }
    let child_write = write_words(region.clone(), 0x56, 0x78);
    child_write.enqueue(&stream.stream, &stream.blas).unwrap();
    let child = record_fence(&stream, vec![child_write]);

    entered_rx.recv_timeout(Duration::from_secs(5)).unwrap();
    assert!(
        matches!(runtime.query_fence(&parent), FenceQuery::Terminal(terminal) if terminal.terminal().is_succeeded())
    );
    assert!(matches!(runtime.query_fence(&child), FenceQuery::Pending));
    assert!(!exited.load(Ordering::Acquire));

    let (read_tx, read_rx) = mpsc::channel();
    let reading_snapshot = Arc::clone(&snapshot);
    cleanup.reader = Some(std::thread::spawn(move || {
        let _ = read_tx.send(reading_snapshot.read());
    }));
    let observed = read_rx.recv_timeout(Duration::from_secs(3));
    // Record both facts before releasing the gate. A stream-synchronizing read
    // times out instead of deadlocking cleanup or passing after child completion.
    let child_was_pending = matches!(runtime.query_fence(&child), FenceQuery::Pending);
    let gate_was_closed = !exited.load(Ordering::Acquire);
    cleanup.release_and_drain().unwrap();
    assert!(child_was_pending && gate_was_closed);
    let observed = observed
        .expect("parent snapshot must not await the child stream")
        .unwrap();
    let expected = [
        0, 0, 0, 0, 0x12, 0x12, 0x12, 0x12, 0x34, 0x34, 0x34, 0x34, 0, 0, 0, 0,
    ];
    assert_eq!(observed, expected);
    assert_eq!(snapshot.read().unwrap(), expected);
    assert!(
        matches!(runtime.query_fence(&child), FenceQuery::Terminal(terminal) if terminal.terminal().is_succeeded())
    );
    let device = stream.stream.clone_dtoh(&region._allocation._base).unwrap();
    assert_eq!(
        device,
        [0, 0, 0, 0, 0x56, 0x56, 0x56, 0x56, 0x78, 0x78, 0x78, 0x78, 0, 0, 0, 0]
    );
    // Reuse the exact pinned allocation after every prior snapshot/DMA owner
    // is released. Core separately proves the lease-holding typed handle cannot
    // make a slot idle early; this exercises the same reset/copy/read helper.
    drop(cleanup);
    drop(snapshot);
    let reused = PinnedReadbackSnapshot::new(region.clone(), host_storage, 16).unwrap();
    stream.state.begin_submission().unwrap();
    reused.enqueue(&stream.stream, 8..16).unwrap();
    let next = record_fence(&stream, Vec::new());
    assert!(runtime.wait_fence(&next).unwrap().terminal().is_succeeded());
    assert_eq!(
        reused.read().unwrap(),
        [0, 0, 0, 0, 0, 0, 0, 0, 0x56, 0x56, 0x56, 0x56, 0x78, 0x78, 0x78, 0x78]
    );
}
