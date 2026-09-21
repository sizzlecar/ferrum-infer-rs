//! Diagnostic only: isolate the two upload API shapes used by production.
//! A pageable copy is permitted to return early or block. Only the preallocated
//! pinned case requires enqueue to return while the same-lane gate is closed.

use super::*;

#[derive(Clone, Copy, Debug)]
struct CopyShape {
    name: &'static str,
    row_bytes: usize,
    rows: usize,
    destination_pitch: usize,
}

impl CopyShape {
    fn payload_bytes(self) -> usize {
        self.row_bytes.checked_mul(self.rows).unwrap()
    }

    fn destination_span(self) -> usize {
        self.destination_pitch
            .checked_mul(self.rows - 1)
            .unwrap()
            .checked_add(self.row_bytes)
            .unwrap()
    }
}

enum UploadSource {
    Pageable(Box<[u8]>),
    Pinned(PinnedHostSlice<u8>),
}

impl UploadSource {
    fn enqueue(
        &self,
        stream: &CudaStream,
        destination: u64,
        shape: CopyShape,
    ) -> Result<(), DriverError> {
        match self {
            Self::Pageable(bytes) => enqueue_bytes(stream, bytes, destination, shape),
            Self::Pinned(host) => {
                // Use the pinned allocation's transfer API. This diagnostic's
                // lifetime proof does not depend on the crate's event-tracking
                // policy: cleanup retains both allocations until it releases
                // the gate, joins the writer, and explicitly drains the stream.
                let (bytes, _guard) = unsafe { host.stream_synced_slice(stream) };
                enqueue_bytes(stream, bytes, destination, shape)
            }
        }
    }
}

fn enqueue_bytes(
    stream: &CudaStream,
    bytes: &[u8],
    destination: u64,
    shape: CopyShape,
) -> Result<(), DriverError> {
    assert_eq!(bytes.len(), shape.payload_bytes());
    if shape.rows == 1 {
        return unsafe {
            cudarc::driver::result::memcpy_htod_async(destination, bytes, stream.cu_stream())
        };
    }
    let copy = cudarc::driver::sys::CUDA_MEMCPY2D {
        srcXInBytes: 0,
        srcY: 0,
        srcMemoryType: cudarc::driver::sys::CUmemorytype::CU_MEMORYTYPE_HOST,
        srcHost: bytes.as_ptr().cast(),
        srcDevice: 0,
        srcArray: std::ptr::null_mut(),
        srcPitch: shape.row_bytes,
        dstXInBytes: 0,
        dstY: 0,
        dstMemoryType: cudarc::driver::sys::CUmemorytype::CU_MEMORYTYPE_DEVICE,
        dstHost: std::ptr::null_mut(),
        dstDevice: destination,
        dstArray: std::ptr::null_mut(),
        dstPitch: shape.destination_pitch,
        WidthInBytes: shape.row_bytes,
        Height: shape.rows,
    };
    unsafe { cudarc::driver::sys::cuMemcpy2DAsync_v2(&copy, stream.cu_stream()) }.result()
}

struct UploadGateCleanup {
    release: Option<Sender<()>>,
    writer: Option<JoinHandle<()>>,
    stream: Arc<CudaStream>,
    // Neither source nor destination can disappear on an assertion/error path.
    _source: Arc<UploadSource>,
    _destination: Arc<CudaSlice<u8>>,
}

impl UploadGateCleanup {
    fn release_join_and_drain(&mut self) -> Result<(), DriverError> {
        if let Some(release) = self.release.take() {
            let _ = release.send(());
        }
        // Join first: a previously blocked API may enqueue after gate release.
        // Draining before join alone would not cover that late enqueue.
        if let Some(writer) = self.writer.take() {
            let _ = writer.join();
        }
        self.stream.synchronize()
    }
}

impl Drop for UploadGateCleanup {
    fn drop(&mut self) {
        let _ = self.release_join_and_drain();
    }
}

fn observe_copy(context: &Arc<CudaContext>, shape: CopyShape, pinned: bool) {
    let stream = context.new_stream().unwrap();
    let payload = (0..shape.payload_bytes())
        .map(|index| (index.wrapping_mul(29).wrapping_add(7) % 251) as u8)
        .collect::<Vec<_>>();
    let source = if pinned {
        let mut host = unsafe { context.alloc_pinned::<u8>(payload.len()) }.unwrap();
        host.as_mut_slice().unwrap().copy_from_slice(&payload);
        UploadSource::Pinned(host)
    } else {
        UploadSource::Pageable(payload.clone().into_boxed_slice())
    };
    let source = Arc::new(source);
    let weak_source = Arc::downgrade(&source);
    const PREFIX: usize = 19;
    const SUFFIX: usize = 23;
    const GUARD: u8 = 0xA7;
    let total = PREFIX + shape.destination_span() + SUFFIX;
    let destination = Arc::new(stream.alloc_zeros::<u8>(total).unwrap());
    let pointer = destination.device_ptr(&stream).0;
    unsafe { cudarc::driver::result::memset_d8_async(pointer, GUARD, total, stream.cu_stream()) }
        .unwrap();
    stream.synchronize().unwrap();
    let mut expected = vec![GUARD; total];
    for row in 0..shape.rows {
        let start = PREFIX + row * shape.destination_pitch;
        expected[start..start + shape.row_bytes]
            .copy_from_slice(&payload[row * shape.row_bytes..(row + 1) * shape.row_bytes]);
    }

    let (entered_tx, entered_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let exited = Arc::new(AtomicBool::new(false));
    let mut cleanup = UploadGateCleanup {
        release: Some(release_tx),
        writer: None,
        stream: Arc::clone(&stream),
        _source: Arc::clone(&source),
        _destination: Arc::clone(&destination),
    };
    let gate = Box::into_raw(Box::new(CallbackGate {
        entered: entered_tx,
        release: release_rx,
        exited: Arc::clone(&exited),
    }));
    if let Err(error) = unsafe {
        cudarc::driver::result::stream::launch_host_function(
            stream.cu_stream(),
            wait_for_release,
            gate.cast(),
        )
    } {
        drop(unsafe { Box::from_raw(gate) });
        panic!("upload gate enqueue failed: {error}");
    }
    let marker = stream.record_event(None).unwrap();
    entered_rx.recv_timeout(Duration::from_secs(5)).unwrap();
    let marker_was_pending = matches!(
        unsafe { cudarc::driver::result::event::query(marker.cu_event()) },
        Err(DriverError(
            cudarc::driver::sys::CUresult::CUDA_ERROR_NOT_READY
        ))
    );

    let (result_tx, result_rx) = mpsc::channel();
    let writer_stream = Arc::clone(&stream);
    cleanup.writer = Some(std::thread::spawn(move || {
        let started = Instant::now();
        let result = writer_stream
            .context()
            .bind_to_thread()
            .and_then(|()| source.enqueue(&writer_stream, pointer + PREFIX as u64, shape));
        let _ = result_tx.send((result, started.elapsed()));
    }));
    // The caller has no source owner after moving it into the writer. Cleanup
    // must still retain it after enqueue returns and before DMA can proceed.
    let early = result_rx.recv_timeout(Duration::from_secs(2));
    let returned_before_release = early.is_ok();
    let gate_closed = !exited.load(Ordering::Acquire);
    // Do not issue another CUDA call here: a blocked pageable copy could hold
    // a driver lock and delay query as well. The entered callback plus its
    // unchanged exited flag proves that the queued marker is still pending.
    let source_retained = weak_source.upgrade().is_some();
    cleanup.release_join_and_drain().unwrap();
    let (result, elapsed) = match early {
        Ok(result) => result,
        Err(mpsc::RecvTimeoutError::Timeout) => result_rx
            .recv_timeout(Duration::from_secs(2))
            .expect("copy enqueue must finish after gate release"),
        Err(error) => panic!("copy worker disconnected: {error}"),
    };
    println!(
        "upload_gate shape={} memory={} payload_bytes={} rows={} row_bytes={} dst_pitch={} returned_before_release={} enqueue_us={} gate_closed={} marker_pending={}",
        shape.name,
        if pinned { "pinned" } else { "pageable" },
        shape.payload_bytes(),
        shape.rows,
        shape.row_bytes,
        shape.destination_pitch,
        returned_before_release,
        elapsed.as_micros(),
        gate_closed,
        marker_was_pending,
    );
    result.unwrap();
    assert!(
        gate_closed && marker_was_pending,
        "pending proof needs a closed gate"
    );
    assert!(
        source_retained,
        "upload source must survive an early enqueue return"
    );
    if pinned {
        assert!(
            returned_before_release,
            "pinned upload waited for the same-lane gate"
        );
    }
    assert_eq!(stream.clone_dtoh(destination.as_ref()).unwrap(), expected);
    drop(cleanup);
    assert!(weak_source.upgrade().is_none());
}

#[test]
#[ignore = "requires an actual CUDA device; bounded upload synchronization diagnostic"]
fn pageable_and_pinned_h2d_enqueue_observe_same_lane_gate_on_cuda() {
    let context = CudaContext::new(0).unwrap();
    // Match the production explicit-fence policy, without a model or runtime.
    unsafe { context.disable_event_tracking() };
    let shapes = [
        CopyShape {
            name: "contiguous-small",
            row_bytes: 64,
            rows: 1,
            destination_pitch: 64,
        },
        CopyShape {
            name: "contiguous-large",
            row_bytes: 1024 * 1024,
            rows: 1,
            destination_pitch: 1024 * 1024,
        },
        CopyShape {
            name: "strided-small",
            row_bytes: 16,
            rows: 32,
            destination_pitch: 48,
        },
        CopyShape {
            name: "strided-large",
            row_bytes: 128,
            rows: 132,
            destination_pitch: 256,
        },
    ];
    for shape in shapes {
        for pinned in [false, true] {
            observe_copy(&context, shape, pinned);
        }
    }
}
