//! Fresh production scratch lifecycle, not a Core admission benchmark.

use super::super::binding_scatter::{BindingScatter, PreparedScatter};
use super::{CudaProgramBindingTransfer, Geometry, Packet, CANARY, CAUSAL_SLOT_BYTES};
use cudarc::driver::{sys, CudaContext, CudaEvent, CudaSlice, CudaStream, DevicePtr};
use serde::Serialize;
use std::sync::Arc;
use std::time::Instant;

type Result<T> = std::result::Result<T, String>;

fn checked<T, E: std::fmt::Display>(result: std::result::Result<T, E>) -> Result<T> {
    result.map_err(|error| error.to_string())
}

#[derive(Clone, Copy, Debug, Serialize)]
enum Transport {
    SparseDirect,
    FreshProductionCompact,
}

struct WaveStorage {
    transfers: Vec<CudaProgramBindingTransfer>,
    compact: Option<PreparedScatter>,
    arena: Arc<CudaSlice<u8>>,
}

// A partial submission keeps every pageable source, destination, scratch and
// module alive until an actual drain; unknown completion leaks this whole set.
struct PendingWave {
    storage: Option<WaveStorage>,
    stream: Arc<CudaStream>,
    submitted: bool,
}

impl Drop for PendingWave {
    fn drop(&mut self) {
        if self.submitted && self.stream.synchronize().is_err() {
            if let Some(storage) = self.storage.take() {
                std::mem::forget(storage);
            }
        }
    }
}

#[derive(Default, Serialize)]
struct Timing {
    net_ns: u128,
    fresh_payload_and_coalesce_ns: u128,
    common_fixture_validation_ns: u128,
    production_prepare_packet_reserve_alloc_ready_ns: u128,
    pending_owner_construction_ns: u128,
    start_event_record_ns: u128,
    upload_and_scatter_api_ns: u128,
    terminal_event_record_and_wait_ns: u128,
    source_owner_drop_and_free_enqueue_ns: u128,
    stream_interval_ns: f64,
    sparse_transfers: usize,
    live_bytes: usize,
    uploaded_bytes: usize,
}

impl Timing {
    fn add(&mut self, other: Self) {
        self.net_ns += other.net_ns;
        self.fresh_payload_and_coalesce_ns += other.fresh_payload_and_coalesce_ns;
        self.common_fixture_validation_ns += other.common_fixture_validation_ns;
        self.production_prepare_packet_reserve_alloc_ready_ns +=
            other.production_prepare_packet_reserve_alloc_ready_ns;
        self.pending_owner_construction_ns += other.pending_owner_construction_ns;
        self.start_event_record_ns += other.start_event_record_ns;
        self.upload_and_scatter_api_ns += other.upload_and_scatter_api_ns;
        self.terminal_event_record_and_wait_ns += other.terminal_event_record_and_wait_ns;
        self.source_owner_drop_and_free_enqueue_ns += other.source_owner_drop_and_free_enqueue_ns;
        self.stream_interval_ns += other.stream_interval_ns;
        self.sparse_transfers += other.sparse_transfers;
        self.live_bytes += other.live_bytes;
        self.uploaded_bytes += other.uploaded_bytes;
    }
}

struct FreshFixture {
    geometry: Geometry,
    arena: Option<Arc<CudaSlice<u8>>>,
    scatter: BindingScatter,
    stream: Arc<CudaStream>,
    allocation_stream: Arc<CudaStream>,
    start: CudaEvent,
    stop: CudaEvent,
    setup_ns: u128,
    product_module_load_ns: u128,
}

impl FreshFixture {
    fn new(geometry: Geometry) -> Result<Self> {
        let setup = Instant::now();
        let context = checked(CudaContext::new(0))?;
        // Matches the product runtime; this fixture supplies explicit fences.
        unsafe { context.disable_event_tracking() };
        let stream = checked(context.new_stream())?;
        let allocation_stream = checked(context.new_stream())?;
        let start = checked(context.new_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT)))?;
        let stop = checked(context.new_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT)))?;
        let load = Instant::now();
        let scatter = checked(BindingScatter::load(&context))?;
        let product_module_load_ns = load.elapsed().as_nanos();
        let arena = Arc::new(checked(stream.alloc_zeros::<u8>(geometry.arena_bytes))?);
        if let Err(error) = stream.synchronize() {
            std::mem::forget(arena);
            return Err(error.to_string());
        }
        Ok(Self {
            geometry,
            arena: Some(arena),
            scatter,
            stream,
            allocation_stream,
            start,
            stop,
            setup_ns: setup.elapsed().as_nanos(),
            product_module_load_ns,
        })
    }

    fn arena(&self) -> Result<&Arc<CudaSlice<u8>>> {
        self.arena.as_ref().ok_or_else(|| "retired arena".into())
    }

    fn reset(&self) -> Result<()> {
        checked(unsafe {
            cudarc::driver::result::memset_d8_async(
                self.arena()?.device_ptr(&self.stream).0,
                CANARY,
                self.geometry.arena_bytes,
                self.stream.cu_stream(),
            )
        })?;
        checked(self.stream.synchronize())
    }

    fn readback(&self) -> Result<Vec<u8>> {
        checked(self.stream.clone_dtoh(self.arena()?.as_ref()))
    }

    fn run(&self, route: Transport, wave: u64) -> Result<Timing> {
        let net = Instant::now();
        let stage = Instant::now();
        let transfers = self.geometry.prepare(wave)?;
        let mut timing = Timing {
            fresh_payload_and_coalesce_ns: stage.elapsed().as_nanos(),
            sparse_transfers: transfers.len(),
            live_bytes: transfers
                .iter()
                .map(|transfer| transfer.payload.len())
                .sum(),
            ..Timing::default()
        };
        // The same pre-existing fixture validator is charged to both arms.
        // It does not represent the excluded Core authority/lease validation.
        let stage = Instant::now();
        Packet::validate(&transfers, self.geometry.arena_bytes)?;
        timing.common_fixture_validation_ns = stage.elapsed().as_nanos();
        let stage = Instant::now();
        let compact = match route {
            Transport::SparseDirect => None,
            Transport::FreshProductionCompact => Some(
                checked(self.scatter.prepare(
                    &self.allocation_stream,
                    &transfers,
                    self.geometry.arena_bytes as u64,
                ))?
                .ok_or("production scratch preparation fell back; no compact sample")?,
            ),
        };
        timing.production_prepare_packet_reserve_alloc_ready_ns = stage.elapsed().as_nanos();
        timing.uploaded_bytes = compact
            .as_ref()
            .map_or(timing.live_bytes, |prepared| prepared.bytes.len());
        let stage = Instant::now();
        let mut pending = PendingWave {
            storage: Some(WaveStorage {
                transfers,
                compact,
                arena: Arc::clone(self.arena()?),
            }),
            stream: Arc::clone(&self.stream),
            submitted: false,
        };
        timing.pending_owner_construction_ns = stage.elapsed().as_nanos();
        let stage = Instant::now();
        checked(self.start.record(&self.stream))?;
        timing.start_event_record_ns = stage.elapsed().as_nanos();
        pending.submitted = true;
        let stage = Instant::now();
        let storage = pending.storage.as_ref().ok_or("missing wave owner")?;
        let destination = storage.arena.device_ptr(&self.stream).0;
        if let Some(prepared) = &storage.compact {
            checked(prepared.owner.upload(&self.stream, &prepared.bytes))?;
            checked(prepared.owner.scatter(&self.stream, destination))?;
        } else {
            for transfer in &storage.transfers {
                direct_copy(&self.stream, destination, transfer)?;
            }
        }
        timing.upload_and_scatter_api_ns = stage.elapsed().as_nanos();
        let stage = Instant::now();
        checked(self.stop.record(&self.stream))?;
        checked(self.stop.synchronize())?;
        pending.submitted = false;
        timing.terminal_event_record_and_wait_ns = stage.elapsed().as_nanos();
        let stage = Instant::now();
        // There is no extra scatter-owner alias: this queues its actual free
        // on the dedicated allocation stream and returns its reservation.
        drop(pending);
        timing.source_owner_drop_and_free_enqueue_ns = stage.elapsed().as_nanos();
        timing.net_ns = net.elapsed().as_nanos();
        timing.stream_interval_ns = f64::from(checked(self.start.elapsed_ms(&self.stop))?) * 1e6;
        Ok(timing)
    }

    fn drain_frees(&self) -> Result<u128> {
        let start = Instant::now();
        checked(self.allocation_stream.synchronize())?;
        Ok(start.elapsed().as_nanos())
    }
}

impl Drop for FreshFixture {
    fn drop(&mut self) {
        if self.stream.synchronize().is_err() {
            if let Some(arena) = self.arena.take() {
                std::mem::forget(arena);
            }
        }
        // Cleanup only; timed physical-free drain is reported per arm.
        let _ = self.allocation_stream.synchronize();
    }
}

// Same ordinary 1D/2D CUDA calls and byte/pitch fields as the product Sparse
// prelude. Destination ownership is fixture-local, not a fabricated Core view.
fn direct_copy(
    stream: &CudaStream,
    arena: u64,
    transfer: &CudaProgramBindingTransfer,
) -> Result<()> {
    let destination = arena + transfer.destination_offset_bytes;
    if transfer.row_count == 1 {
        return checked(unsafe {
            cudarc::driver::result::memcpy_htod_async(
                destination,
                transfer.payload.as_ref(),
                stream.cu_stream(),
            )
        });
    }
    let copy = sys::CUDA_MEMCPY2D {
        srcXInBytes: 0,
        srcY: 0,
        srcMemoryType: sys::CUmemorytype::CU_MEMORYTYPE_HOST,
        srcHost: transfer.payload.as_ptr().cast(),
        srcDevice: 0,
        srcArray: std::ptr::null_mut(),
        srcPitch: transfer.row_bytes,
        dstXInBytes: 0,
        dstY: 0,
        dstMemoryType: sys::CUmemorytype::CU_MEMORYTYPE_DEVICE,
        dstHost: std::ptr::null_mut(),
        dstDevice: destination,
        dstArray: std::ptr::null_mut(),
        dstPitch: transfer.destination_stride_bytes as usize,
        WidthInBytes: transfer.row_bytes,
        Height: transfer.row_count,
    };
    checked(unsafe { sys::cuMemcpy2DAsync_v2(&copy, stream.cu_stream()).result() })
}

#[test]
#[ignore = "fresh production compact allocation/packing/fence/drop versus sparse direct; exclusive CUDA"]
fn compact_binding_fresh_lifecycle_paired_net() {
    const WARM: usize = 2;
    const QUADS: usize = 4;
    const WAVES: usize = 8;
    println!(
        "compact_binding_fresh_lifecycle_plan {}",
        serde_json::json!({
            "scope":"synthetic fixture and real production BindingScatter lifecycle, not public Core or serving performance",
            "geometry_scope":"two retained diagnostic geometries; base_pages64 is not a measured current serving batch or page distribution",
            "cases":[{"participants":8,"gdn_nodes":48,"causal_nodes":16,"base_pages":6},{"participants":32,"gdn_nodes":48,"causal_nodes":16,"base_pages":64}],
            "warm_waves_per_route":WARM,"quads":QUADS,"waves_per_arm":WAVES,
            "order":"ABBA/BAAB alternating; each arm repeats the same fresh wave IDs within a quad",
            "net":"fresh payload/coalesce + common fixture validator + real compact prepare if selected + pending-owner construction + H2D/scatter + terminal event fence + all transient source/owner Drop and free enqueue",
            "prepare_scope":"production bounds, expanded row sorting, serialization, reservation, fresh device allocation, allocation-stream synchronize; combined because no production timing hooks are added",
            "allocation_semantics":"previous free may be included in next production prepare readiness wait; additional free drain only after each arm and separately reported",
            "stream_interval":"events after prepare to last copy/scatter, including host enqueue gaps; not pure DMA/SM busy",
            "oracle":"one canary initialization per arm, then full arena readback every wave; shrinking prefixes preserve old bytes; oracle/readback outside net and may give asynchronous free extra progress, so continuous serving allocator contention is not reproduced",
            "excluded":"Core fresh authority/leases, model compute, counter plumbing, whole product command/fence/reaper overhead; no timing claim for these",
            "product_symbol":"ferrum_program_binding_scatter_v1 from embedded GATHER_COLUMNS; one load per geometry, no private NVRTC kernel",
            "performance_assertions":false,
        })
    );
    for (participants, pages) in [(8, 6), (32, 64)] {
        let geometry = Geometry::new(participants, 48, 16, pages);
        let fixture = FreshFixture::new(geometry.clone()).unwrap();
        println!(
            "compact_binding_fresh_lifecycle_setup {}",
            serde_json::json!({
                "participants":participants,"base_pages":pages,"arena_bytes":geometry.arena_bytes,
                "physical_causal_pitch":CAUSAL_SLOT_BYTES,"service_reference_max_tokens":2048,
                "setup_ns":fixture.setup_ns,"product_module_load_ns":fixture.product_module_load_ns,
                "async_allocator":fixture.stream.context().has_async_alloc(),
                "preallocated_device_packet":false,"pinned_host_staging":false,
            })
        );
        for route in [Transport::SparseDirect, Transport::FreshProductionCompact] {
            fixture.reset().unwrap();
            let mut expected = vec![CANARY; geometry.arena_bytes];
            for wave in 1..=WARM as u64 {
                fixture.run(route, wave).unwrap();
                geometry.apply_expected(&mut expected, wave);
                assert_eq!(fixture.readback().unwrap(), expected);
            }
            let drain_ns = fixture.drain_frees().unwrap();
            println!(
                "compact_binding_fresh_lifecycle_warm_drain {}",
                serde_json::json!({
                    "participants":participants,"base_pages":pages,"route":route,"drain_ns":drain_ns,
                })
            );
        }
        for quad in 0..QUADS {
            let a = Transport::SparseDirect;
            let b = Transport::FreshProductionCompact;
            let order = if quad % 2 == 0 {
                [a, b, b, a]
            } else {
                [b, a, a, b]
            };
            let first_wave = 100 + (quad * WAVES) as u64;
            for (position, route) in order.into_iter().enumerate() {
                fixture.reset().unwrap();
                let mut expected = vec![CANARY; geometry.arena_bytes];
                let mut total = Timing::default();
                let mut shorter_prefix_transitions = 0;
                for index in 0..WAVES {
                    let wave = first_wave + index as u64;
                    if index > 0
                        && geometry
                            .windows(wave - 1)
                            .iter()
                            .zip(geometry.windows(wave))
                            .any(|((_, before, _), (_, after, _))| after < *before)
                    {
                        shorter_prefix_transitions += 1;
                    }
                    total.add(fixture.run(route, wave).unwrap());
                    geometry.apply_expected(&mut expected, wave);
                    assert_eq!(fixture.readback().unwrap(), expected);
                }
                assert!(shorter_prefix_transitions > 0);
                let terminal_free_drain_ns = fixture.drain_frees().unwrap();
                println!(
                    "compact_binding_fresh_lifecycle_measurement {}",
                    serde_json::json!({
                        "participants":participants,"gdn_nodes":48,"causal_nodes":16,"base_pages":pages,
                        "quad":quad,"position":position,"route":route,"first_wave":first_wave,"waves":WAVES,
                        "shorter_prefix_transitions":shorter_prefix_transitions,"full_arena_oracles":WAVES,
                        "timing":total,"terminal_free_drain_ns":terminal_free_drain_ns,
                    })
                );
            }
        }
    }
}
