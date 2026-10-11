//! Single-stream explicit lifetime ownership for the transfer experiment.
use super::*;
use cudarc::driver::{
    sys, CudaContext, CudaEvent, CudaFunction, CudaSlice, CudaStream, DevicePtr, LaunchConfig,
    PinnedHostSlice, PushKernelArg,
};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};
use std::cell::Cell;
use std::sync::Arc;

fn driver<T>(value: std::result::Result<T, cudarc::driver::DriverError>) -> Result<T> {
    value.map_err(|error| error.to_string())
}

struct Storage {
    arena: Arc<CudaSlice<u8>>,
    packet: CudaSlice<u8>,
    pinned: PinnedHostSlice<u8>,
    function: CudaFunction,
}

struct PendingWave<'a> {
    transfers: Option<Vec<CudaProgramBindingTransfer>>,
    packet: Option<Packet>,
    stream: Arc<CudaStream>,
    submitted: bool,
    indeterminate: &'a Cell<bool>,
}

impl Drop for PendingWave<'_> {
    fn drop(&mut self) {
        if self.submitted && self.stream.synchronize().is_err() {
            self.indeterminate.set(true);
            // Both pageable source kinds may still be read by CUDA. The
            // enclosing owner also quarantines pins, devices and module.
            if let Some(transfers) = self.transfers.take() {
                std::mem::forget(transfers);
            }
            if let Some(packet) = self.packet.take() {
                std::mem::forget(packet);
            }
        }
    }
}

pub(super) struct Fixture {
    geometry: Geometry,
    storage: Option<Storage>,
    stream: Arc<CudaStream>,
    start: CudaEvent,
    stop: CudaEvent,
    indeterminate: Cell<bool>,
    setup_ns: u128,
    compile_load_ns: u128,
}

impl Fixture {
    pub(super) fn new(geometry: Geometry) -> Result<Self> {
        let setup = Instant::now();
        let context = driver(CudaContext::new(0))?;
        // This owner supplies every terminal fence and failure quarantine.
        unsafe { context.disable_event_tracking() };
        let stream = driver(context.new_stream())?;
        let arena = Arc::new(driver(stream.alloc_zeros::<u8>(geometry.arena_bytes))?);
        driver(unsafe {
            sys::cuMemsetD8Async(
                arena.device_ptr(&stream).0,
                CANARY,
                geometry.arena_bytes,
                stream.cu_stream(),
            )
            .result()
        })?;
        let packet = driver(stream.alloc_zeros::<u8>(geometry.packet_capacity))?;
        let mut pinned = driver(unsafe { context.alloc_pinned::<u8>(geometry.packet_capacity) })?;
        driver(pinned.as_mut_slice())?.fill(0);
        driver(stream.synchronize())?;
        let start = driver(context.new_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT)))?;
        let stop = driver(context.new_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT)))?;
        let compile = Instant::now();
        let ptx = compile_ptx_with_opts(
            include_str!("scatter.cu"),
            CompileOptions {
                arch: Some("compute_120"),
                ..Default::default()
            },
        )
        .map_err(|error| error.to_string())?;
        let function =
            driver(driver(context.load_module(ptx))?.load_function("ferrum_test_scatter"))?;
        let compile_load_ns = compile.elapsed().as_nanos();
        Ok(Self {
            geometry,
            storage: Some(Storage {
                arena,
                packet,
                pinned,
                function,
            }),
            stream,
            start,
            stop,
            indeterminate: Cell::new(false),
            setup_ns: setup.elapsed().as_nanos(),
            compile_load_ns,
        })
    }

    fn storage(&self) -> &Storage {
        self.storage.as_ref().expect("live fixture")
    }

    fn check_live(&self) -> Result<()> {
        if self.indeterminate.get() {
            return Err("indeterminate completion prohibits reuse".into());
        }
        if self.storage.is_none() {
            return Err("retired fixture".into());
        }
        Ok(())
    }

    fn stage(
        &mut self,
        transfers: &[CudaProgramBindingTransfer],
        packet: Option<&Packet>,
    ) -> Result<()> {
        self.check_live()?;
        let host = driver(
            self.storage
                .as_mut()
                .ok_or("retired fixture")?
                .pinned
                .as_mut_slice(),
        )?;
        if let Some(packet) = packet {
            host.get_mut(..packet.bytes.len())
                .ok_or("packet exceeds pinned capacity")?
                .copy_from_slice(&packet.bytes);
        } else {
            let mut offset = 0;
            for transfer in transfers {
                let end = offset + transfer.payload.len();
                host.get_mut(offset..end)
                    .ok_or("direct payload exceeds pinned capacity")?
                    .copy_from_slice(&transfer.payload);
                offset = end;
            }
        }
        Ok(())
    }

    fn direct_copy(&self, transfer: &CudaProgramBindingTransfer, source: *const u8) -> Result<()> {
        let destination = self.storage().arena.device_ptr(&self.stream).0;
        if transfer.row_count == 1 {
            return driver(unsafe {
                sys::cuMemcpyHtoDAsync_v2(
                    destination + transfer.destination_offset_bytes,
                    source.cast(),
                    transfer.row_bytes,
                    self.stream.cu_stream(),
                )
                .result()
            });
        }
        let copy = sys::CUDA_MEMCPY2D {
            srcXInBytes: 0,
            srcY: 0,
            srcMemoryType: sys::CUmemorytype::CU_MEMORYTYPE_HOST,
            srcHost: source.cast(),
            srcDevice: 0,
            srcArray: std::ptr::null_mut(),
            srcPitch: transfer.row_bytes,
            dstXInBytes: 0,
            dstY: 0,
            dstMemoryType: sys::CUmemorytype::CU_MEMORYTYPE_DEVICE,
            dstHost: std::ptr::null_mut(),
            dstDevice: destination + transfer.destination_offset_bytes,
            dstArray: std::ptr::null_mut(),
            dstPitch: transfer.destination_stride_bytes as usize,
            WidthInBytes: transfer.row_bytes,
            Height: transfer.row_count,
        };
        driver(unsafe { sys::cuMemcpy2DAsync_v2(&copy, self.stream.cu_stream()).result() })
    }

    fn packet_copy(&self, source: *const u8, bytes: usize) -> Result<()> {
        if bytes > self.geometry.packet_capacity {
            return Err("packet exceeds device capacity".into());
        }
        driver(unsafe {
            sys::cuMemcpyHtoDAsync_v2(
                self.storage().packet.device_ptr(&self.stream).0,
                source.cast(),
                bytes,
                self.stream.cu_stream(),
            )
            .result()
        })
    }

    fn scatter(&self, count: u32) -> Result<()> {
        let packet = self.storage().packet.device_ptr(&self.stream).0;
        let arena = self.storage().arena.device_ptr(&self.stream).0;
        let mut launch = self.stream.launch_builder(&self.storage().function);
        launch.arg(&packet).arg(&arena).arg(&count);
        driver(unsafe {
            launch.launch(LaunchConfig {
                grid_dim: (count, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })
        })?;
        Ok(())
    }

    pub(super) fn run(
        &mut self,
        route: Route,
        wave: u64,
        fail_after_h2d: bool,
    ) -> Result<Measurement> {
        self.check_live()?;
        let net = Instant::now();
        let transfers = self.geometry.prepare(wave)?;
        let packet = if route.scatter() {
            Some(Packet::prepare(&transfers, self.geometry.arena_bytes)?)
        } else {
            Packet::validate(&transfers, self.geometry.arena_bytes)?;
            None
        };
        let live_bytes = transfers
            .iter()
            .map(|transfer| transfer.payload.len())
            .sum();
        let upload_bytes = packet
            .as_ref()
            .map_or(live_bytes, |packet| packet.bytes.len());
        let transfer_count = transfers.len();
        if route.pinned() {
            self.stage(&transfers, packet.as_ref())?;
        }
        let pinned_source = if route.pinned() {
            driver(self.storage().pinned.as_ptr())?
        } else {
            std::ptr::null()
        };
        let mut pending = PendingWave {
            transfers: Some(transfers),
            packet,
            stream: Arc::clone(&self.stream),
            submitted: false,
            indeterminate: &self.indeterminate,
        };
        driver(self.start.record(&self.stream))?;
        let api = Instant::now();
        pending.submitted = true; // Before any copy can access a host source.
        if route.scatter() {
            let packet = pending.packet.as_ref().ok_or("missing packet")?;
            let source = if route.pinned() {
                pinned_source
            } else {
                packet.bytes.as_ptr()
            };
            self.packet_copy(source, packet.bytes.len())?;
            if fail_after_h2d {
                return Err("injected failure after H2D before scatter".into());
            }
            self.scatter(packet.descriptor_count)?;
        } else {
            let mut offset = 0;
            for transfer in pending.transfers.as_ref().ok_or("missing transfers")? {
                let source = if route.pinned() {
                    unsafe { pinned_source.add(offset) }
                } else {
                    transfer.payload.as_ptr()
                };
                self.direct_copy(transfer, source)?;
                offset += transfer.payload.len();
            }
        }
        let host_api_ns = api.elapsed().as_nanos();
        driver(self.stop.record(&self.stream))?;
        driver(self.stop.synchronize())?;
        pending.submitted = false;
        drop(pending);
        let net_ns = net.elapsed().as_nanos();
        let stream_interval_ns = f64::from(driver(self.start.elapsed_ms(&self.stop))?) * 1e6;
        Ok(Measurement {
            net_ns,
            host_api_ns,
            stream_interval_ns,
            transfers: transfer_count,
            live_bytes,
            upload_bytes,
        })
    }

    pub(super) fn readback(&self) -> Result<Vec<u8>> {
        driver(self.stream.clone_dtoh(self.storage().arena.as_ref()))
    }

    pub(super) fn reset_destination(&self) -> Result<()> {
        self.check_live()?;
        driver(unsafe {
            sys::cuMemsetD8Async(
                self.storage().arena.device_ptr(&self.stream).0,
                CANARY,
                self.geometry.arena_bytes,
                self.stream.cu_stream(),
            )
            .result()
        })?;
        driver(self.stream.synchronize())
    }

    pub(super) fn check_unknown_completion_rejects_before_staging(
        &mut self,
        wave: u64,
    ) -> Result<()> {
        let before = driver(self.storage().pinned.as_slice())?.to_vec();
        // State-machine injection while idle, not a real CUDA driver failure.
        self.indeterminate.set(true);
        let rejected = self.run(Route::PinnedScatter, wave, false).is_err();
        let unchanged = driver(self.storage().pinned.as_slice())? == before.as_slice();
        driver(self.stream.synchronize())?; // Proven terminal before state reset.
        self.indeterminate.set(false);
        if rejected && unchanged {
            Ok(())
        } else {
            Err("unknown-completion state allowed mutation".into())
        }
    }

    pub(super) fn destination_observer(&self) -> (Arc<CudaStream>, Arc<CudaSlice<u8>>) {
        (Arc::clone(&self.stream), Arc::clone(&self.storage().arena))
    }

    pub(super) fn launch_pinned_scatter_for_drop(&mut self, wave: u64) -> Result<()> {
        self.check_live()?;
        let transfers = self.geometry.prepare(wave)?;
        let packet = Packet::prepare(&transfers, self.geometry.arena_bytes)?;
        self.stage(&transfers, Some(&packet))?;
        // Sources referenced by this DMA are owned pins, retained by Storage.
        self.packet_copy(driver(self.storage().pinned.as_ptr())?, packet.bytes.len())?;
        self.scatter(packet.descriptor_count)
    }

    pub(super) fn print_setup(&self) {
        println!(
            "compact_upload_scatter_setup {}",
            json!({
                "participants": self.geometry.participants, "gdn_nodes": self.geometry.gdn_nodes,
                "causal_nodes": self.geometry.causal_nodes, "base_pages": self.geometry.base_pages,
                "physical_causal_pitch": CAUSAL_SLOT_BYTES, "arena_bytes": self.geometry.arena_bytes,
                "packet_capacity": self.geometry.packet_capacity, "total_cold_setup_ns": self.setup_ns,
                "nvrtc_compile_and_module_load_ns": self.compile_load_ns,
                "same_pinned_storage_for_both_arms": true,
            })
        );
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let safe =
            self.stream.context().bind_to_thread().is_ok() && self.stream.synchronize().is_ok();
        if safe {
            drop(self.storage.take());
        } else if let Some(storage) = self.storage.take() {
            // Includes function/module, packet, arena and all pinned sources.
            std::mem::forget(storage);
        }
    }
}
