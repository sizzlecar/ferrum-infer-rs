//! One-stream owner for fixed pinned sources, graph handles and destination.
//! Raw copies do not record PinnedHostSlice's internal event: explicit fences
//! below, including every error/Drop path, establish its mutation lifetime.

use super::*;
use cudarc::driver::{
    sys, CudaContext, CudaEvent, CudaSlice, CudaStream, DevicePtr, PinnedHostSlice,
};
use std::cell::Cell;
use std::sync::Arc;

fn driver<T>(result: std::result::Result<T, cudarc::driver::DriverError>) -> Result<T> {
    result.map_err(|error| error.to_string())
}

struct Storage {
    arena: Arc<CudaSlice<u8>>,
    pinned: Vec<PinnedHostSlice<u8>>,
}

// EndCapture can return an owned graph before the subsequent status query
// fails. This uninstantiated graph must be destroyed on every early return.
struct UnclaimedGraph(sys::CUgraph);

impl Drop for UnclaimedGraph {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe { sys::cuGraphDestroy(self.0) };
        }
    }
}

// Each invocation owns pageable source Boxes until its fence. The same guard
// also fences partial pinned submissions before the enclosing owner can mutate
// the pins. An indeterminate fence quarantines host sources rather than freeing
// memory still potentially referenced by CUDA.
struct PendingWave<'a> {
    transfers: Option<Vec<CudaProgramBindingTransfer>>,
    stream: Arc<CudaStream>,
    submitted: bool,
    indeterminate: &'a Cell<bool>,
}

impl Drop for PendingWave<'_> {
    fn drop(&mut self) {
        if self.submitted && self.stream.synchronize().is_err() {
            self.indeterminate.set(true);
            if let Some(transfers) = self.transfers.take() {
                std::mem::forget(transfers);
            }
        }
    }
}

pub(super) struct Fixture {
    geometry: Geometry,
    shapes: Vec<CopyShape>,
    storage: Option<Storage>,
    source_pointers: Vec<*const u8>,
    destination: u64,
    stream: Arc<CudaStream>,
    start: CudaEvent,
    stop: CudaEvent,
    graph: sys::CUgraph,
    executable: sys::CUgraphExec,
    capture_active: bool,
    indeterminate: Cell<bool>,
    graph_nodes: usize,
    common_setup_ns: u128,
    pin_setup_ns: u128,
    graph_setup_ns: u128,
}

impl Fixture {
    pub(super) fn new(geometry: Geometry) -> Result<Self> {
        let setup = Instant::now();
        let context = driver(CudaContext::new(0))?;
        // Match the runtime's explicit completion ownership. Only this fixture
        // thread/stream accesses the allocations, with no concurrent replay.
        unsafe { context.disable_event_tracking() };
        let stream = driver(context.new_stream())?;
        let arena = Arc::new(driver(
            stream.clone_htod(&vec![CANARY; geometry.arena_bytes]),
        )?);
        driver(stream.synchronize())?;
        let destination = arena.device_ptr(&stream).0;
        let start = driver(context.new_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT)))?;
        let stop = driver(context.new_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT)))?;
        let templates = geometry.prepare(0)?;
        let shapes = templates.iter().map(CopyShape::from).collect::<Vec<_>>();
        let common_setup_ns = setup.elapsed().as_nanos();
        let setup = Instant::now();
        let mut pinned = Vec::new();
        let mut source_pointers = Vec::new();
        for transfer in &templates {
            // Initialize every allocated byte before capture sees the pointer.
            let mut host = driver(unsafe { context.alloc_pinned::<u8>(transfer.payload.len()) })?;
            driver(host.as_mut_slice())?.copy_from_slice(&transfer.payload);
            source_pointers.push(driver(host.as_ptr())?);
            pinned.push(host);
        }
        let pin_setup_ns = setup.elapsed().as_nanos();
        let mut fixture = Self {
            geometry,
            shapes,
            storage: Some(Storage { arena, pinned }),
            source_pointers,
            destination,
            stream,
            start,
            stop,
            graph: std::ptr::null_mut(),
            executable: std::ptr::null_mut(),
            capture_active: false,
            indeterminate: Cell::new(false),
            graph_nodes: 0,
            common_setup_ns,
            pin_setup_ns,
            graph_setup_ns: 0,
        };
        let setup = Instant::now();
        fixture.capture(false)?;
        fixture.graph_setup_ns = setup.elapsed().as_nanos();
        Ok(fixture)
    }

    fn storage(&self) -> &Storage {
        self.storage.as_ref().expect("live fixture storage")
    }

    fn validate(&self, transfers: &[CudaProgramBindingTransfer]) -> Result<()> {
        if self.indeterminate.get() {
            return Err("prior submission has indeterminate completion".into());
        }
        if transfers.len() != self.shapes.len() {
            return Err("changed sparse transfer count".into());
        }
        for (transfer, shape) in transfers.iter().zip(&self.shapes) {
            if CopyShape::from(transfer) != *shape
                || transfer.payload.len() != shape.width * shape.rows
                || shape.offset + shape.pitch * (shape.rows as u64 - 1) + shape.width as u64
                    > self.geometry.arena_bytes as u64
            {
                return Err("changed or invalid sparse transfer geometry".into());
            }
        }
        Ok(())
    }

    fn stage(&mut self, transfers: &[CudaProgramBindingTransfer]) -> Result<()> {
        self.validate(transfers)?;
        // All preceding uses completed before this method. as_mut_slice also
        // performs cudarc's own pinned-event synchronization, included in net.
        let storage = self.storage.as_mut().ok_or("retired fixture storage")?;
        for (host, transfer) in storage.pinned.iter_mut().zip(transfers) {
            driver(host.as_mut_slice())?.copy_from_slice(&transfer.payload);
        }
        Ok(())
    }

    fn copy(&self, index: usize, source: *const u8) -> Result<()> {
        let shape = &self.shapes[index];
        if shape.rows == 1 {
            driver(unsafe {
                sys::cuMemcpyHtoDAsync_v2(
                    self.destination + shape.offset,
                    source.cast(),
                    shape.width,
                    self.stream.cu_stream(),
                )
                .result()
            })
        } else {
            // Same sparse 2D descriptor as CudaDeviceRuntime's binding prelude.
            // No maximum-row padding is copied into the destination's gaps.
            let descriptor = sys::CUDA_MEMCPY2D {
                srcXInBytes: 0,
                srcY: 0,
                srcMemoryType: sys::CUmemorytype::CU_MEMORYTYPE_HOST,
                srcHost: source.cast(),
                srcDevice: 0,
                srcArray: std::ptr::null_mut(),
                srcPitch: shape.width,
                dstXInBytes: 0,
                dstY: 0,
                dstMemoryType: sys::CUmemorytype::CU_MEMORYTYPE_DEVICE,
                dstHost: std::ptr::null_mut(),
                dstDevice: self.destination + shape.offset,
                dstArray: std::ptr::null_mut(),
                dstPitch: shape.pitch as usize,
                WidthInBytes: shape.width,
                Height: shape.rows,
            };
            driver(unsafe {
                sys::cuMemcpy2DAsync_v2(&descriptor, self.stream.cu_stream()).result()
            })
        }
    }

    fn capture(&mut self, abort_after_first: bool) -> Result<()> {
        driver(self.stream.synchronize())?;
        driver(unsafe {
            sys::cuStreamBeginCapture_v2(
                self.stream.cu_stream(),
                sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED,
            )
            .result()
        })?;
        self.capture_active = true;
        // No host slice access, synchronization, allocation or timing in capture.
        // On panic the enclosing Fixture Drop ends capture before storage drops.
        let encoded = (|| {
            for (index, pointer) in self.source_pointers.iter().enumerate() {
                self.copy(index, *pointer)?;
                if abort_after_first {
                    return Err("injected capture abort".to_owned());
                }
            }
            Ok(())
        })();
        let mut graph = UnclaimedGraph(std::ptr::null_mut());
        let ended = driver(unsafe {
            sys::cuStreamEndCapture(self.stream.cu_stream(), &mut graph.0).result()
        });
        let mut state = sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_INVALIDATED;
        driver(unsafe { sys::cuStreamIsCapturing(self.stream.cu_stream(), &mut state).result() })?;
        self.capture_active = state != sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE;
        if self.capture_active {
            return Err("capture did not terminate".into());
        }
        if ended.is_err() || encoded.is_err() {
            ended?;
            return encoded;
        }
        let graph = std::mem::replace(&mut graph.0, std::ptr::null_mut());
        self.graph = graph;
        driver(unsafe {
            sys::cuGraphGetNodes(graph, std::ptr::null_mut(), &mut self.graph_nodes).result()
        })?;
        if self.graph_nodes == 0 {
            return Err("empty sparse upload graph".into());
        }
        driver(unsafe {
            sys::cuGraphInstantiateWithFlags(&mut self.executable, graph, 0).result()
        })?;
        driver(unsafe { sys::cuGraphUpload(self.executable, self.stream.cu_stream()).result() })?;
        driver(self.stream.synchronize())
    }

    pub(super) fn run(
        &mut self,
        route: Route,
        wave: u64,
        fail_after: Option<usize>,
    ) -> Result<Measurement> {
        let net = Instant::now();
        let transfers = self.geometry.prepare(wave)?;
        if matches!(route, Route::BoxDirect) {
            self.validate(&transfers)?;
        } else {
            self.stage(&transfers)?;
        }
        let mut pending = PendingWave {
            transfers: Some(transfers),
            stream: Arc::clone(&self.stream),
            submitted: false,
            indeterminate: &self.indeterminate,
        };
        // Events are prebuilt and reused after each wave's terminal fence. Start
        // follows fresh payload construction/staging on every route.
        driver(self.start.record(&self.stream))?;
        pending.submitted = true;
        let host_api = Instant::now();
        match route {
            Route::PinnedGraph => driver(unsafe {
                sys::cuGraphLaunch(self.executable, self.stream.cu_stream()).result()
            })?,
            Route::BoxDirect | Route::PinnedDirect => {
                for (index, transfer) in pending
                    .transfers
                    .as_ref()
                    .ok_or("missing live payload")?
                    .iter()
                    .enumerate()
                {
                    let source = if matches!(route, Route::BoxDirect) {
                        transfer.payload.as_ptr()
                    } else {
                        self.source_pointers[index]
                    };
                    self.copy(index, source)?;
                    if fail_after == Some(index + 1) {
                        return Err("injected partial submission".into());
                    }
                }
            }
        }
        let host_api_ns = host_api.elapsed().as_nanos();
        driver(self.stop.record(&self.stream))?;
        driver(self.stop.synchronize())?;
        pending.submitted = false;
        drop(pending); // Include per-wave pageable payload retirement in net.
        let net_ns = net.elapsed().as_nanos();
        let stream_interval_ns = f64::from(driver(self.start.elapsed_ms(&self.stop))?) * 1e6;
        Ok(Measurement {
            net_ns,
            host_api_ns,
            stream_interval_ns,
        })
    }

    pub(super) fn readback(&self) -> Result<Vec<u8>> {
        driver(self.stream.clone_dtoh(self.storage().arena.as_ref()))
    }

    pub(super) fn reject_changed_shape(&mut self, wave: u64) -> Result<()> {
        let mut transfers = self.geometry.prepare(wave)?;
        transfers[0].destination_offset_bytes += 1;
        self.stage(&transfers)
    }

    pub(super) fn abort_capture_after_copy(&mut self) -> Result<()> {
        // Keep the original executable; an aborted capture is destroyed without
        // replacing it, and no captured copies have executed.
        match self.capture(true) {
            Err(error) if error == "injected capture abort" => Ok(()),
            Err(error) => Err(error),
            Ok(()) => Err("capture unexpectedly accepted injected abort".into()),
        }
    }

    pub(super) fn destination_observer(&self) -> (Arc<CudaStream>, Arc<CudaSlice<u8>>) {
        (Arc::clone(&self.stream), Arc::clone(&self.storage().arena))
    }

    pub(super) fn launch_for_drop(&mut self, wave: u64) -> Result<()> {
        let transfers = self.geometry.prepare(wave)?;
        self.stage(&transfers)?;
        driver(unsafe { sys::cuGraphLaunch(self.executable, self.stream.cu_stream()).result() })
    }

    pub(super) fn print_setup(&self) {
        println!(
            "sparse_upload_prelude_setup {}",
            json!({
                "participants": self.geometry.participants, "arena_bytes": self.geometry.arena_bytes,
                "logical_writes": self.geometry.windows.len(), "graph_nodes": self.graph_nodes,
                "copies": self.shapes.len(),
                "copies_1d": self.shapes.iter().filter(|shape| shape.rows == 1).count(),
                "copies_2d": self.shapes.iter().filter(|shape| shape.rows > 1).count(),
                "live_bytes": self.shapes.iter().map(|shape| shape.width * shape.rows).sum::<usize>(),
                "common_setup_ns": self.common_setup_ns, "pinned_setup_ns": self.pin_setup_ns,
                "capture_instantiate_upload_ns": self.graph_setup_ns,
                "shapes": self.shapes.iter().map(|shape| json!({
                    "offset": shape.offset, "pitch": shape.pitch, "width": shape.width, "rows": shape.rows,
                })).collect::<Vec<_>>(),
            })
        );
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let mut safe = self.stream.context().bind_to_thread().is_ok();
        if safe && self.capture_active {
            let mut abandoned = std::ptr::null_mut();
            unsafe {
                sys::cuStreamEndCapture(self.stream.cu_stream(), &mut abandoned);
                if !abandoned.is_null() {
                    sys::cuGraphDestroy(abandoned);
                }
            }
            let mut state = sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_INVALIDATED;
            safe = unsafe { sys::cuStreamIsCapturing(self.stream.cu_stream(), &mut state) }
                == sys::CUresult::CUDA_SUCCESS
                && state == sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE;
        }
        safe = safe && self.stream.synchronize().is_ok();
        if safe {
            unsafe {
                if !self.executable.is_null() {
                    sys::cuGraphExecDestroy(self.executable);
                }
                if !self.graph.is_null() {
                    sys::cuGraphDestroy(self.graph);
                }
            }
            drop(self.storage.take());
        } else if let Some(storage) = self.storage.take() {
            // Unknown completion: intentionally quarantine pins/device memory.
            // This path is not represented as a successful benchmark sample.
            std::mem::forget(storage);
        }
    }
}
