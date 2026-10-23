//! Backend-only terminal transaction: no Core lane locks or serving claims.
//! The fixture follows the numerical-status test's real stream/fence path;
//! it does not construct a private Core DeviceCommandBatch.

use super::*;
use cudarc::driver::{CudaFunction, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;
use ferrum_runtime::{BufferUsage, ResourceId};
use serde_json::json;
use std::sync::Weak;

const CAPACITY: usize = 32;
const BYTES: usize = 64;
const CANARY: u8 = 0xa5;
const PRODUCER: &str = r#"
.version 6.0
.target sm_30
.address_size 64
.visible .entry write_tokens(.param .u64 addresses, .param .u32 count,
                            .param .u32 generation, .param .u64 delay) {
    .reg .pred %done;
    .reg .b32 %row, %n, %gen, %value;
    .reg .b64 %table, %offset, %ptr, %start, %now, %elapsed, %delay;
    mov.u32 %row, %tid.x;
    ld.param.u32 %n, [count];
    setp.ge.u32 %done, %row, %n;
    @%done bra END;
    ld.param.u64 %delay, [delay];
    mov.u64 %start, %clock64;
WAIT:
    mov.u64 %now, %clock64;
    sub.u64 %elapsed, %now, %start;
    setp.lt.u64 %done, %elapsed, %delay;
    @%done bra WAIT;
    ld.param.u64 %table, [addresses];
    mul.wide.u32 %offset, %row, 8;
    add.u64 %table, %table, %offset;
    ld.global.u64 %ptr, [%table];
    ld.param.u32 %gen, [generation];
    mul.lo.u32 %value, %gen, 1009;
    mad.lo.u32 %value, %row, 31, %value;
    add.u32 %value, %value, 17;
    st.global.u32 [%ptr], %value;
END:
    ret;
}
"#;

type Result<T> = std::result::Result<T, CudaDeviceRuntimeError>;

#[derive(Clone, Copy, Debug)]
enum Route {
    Scalar,
    AsyncPacket,
    PooledPacket,
    ProductPacket,
}

struct Packet {
    host: Mutex<PinnedHostSlice<u32>>,
}

impl Packet {
    fn enqueue(
        &self,
        stream: &CudaStream,
        sources: &[CudaBufferRegion],
        fail_after: Option<usize>,
    ) -> Result<()> {
        let mut host = self
            .host
            .lock()
            .map_err(|_| CudaDeviceRuntimeError::contract("terminal packet host mutex poisoned"))?;
        // One DMA-use guard covers every copy, including a partial enqueue.
        let (values, _dma_guard) = unsafe { host.stream_synced_mut_slice(stream) };
        for (ordinal, (value, source)) in values.iter_mut().zip(sources).enumerate() {
            unsafe {
                cudarc::driver::result::memcpy_dtoh_async(
                    std::slice::from_mut(value),
                    source.device_ptr,
                    stream.cu_stream(),
                )
            }
            .map_err(|e| CudaDeviceRuntimeError::driver("terminal packet copy", e))?;
            if fail_after == Some(ordinal + 1) {
                return Err(CudaDeviceRuntimeError::contract(
                    "test injected failure after a queued packet prefix",
                ));
            }
        }
        Ok(())
    }
}

/// Preserve every source and pinned destination if any queued work is unknown.
/// Quarantine is the production runtime's existing retained-command mechanism.
struct Pending<'a> {
    runtime: &'a CudaDeviceRuntime,
    stream: &'a mut CudaDeviceStream,
    commands: Vec<CudaDeviceCommand>,
    fence: Option<CudaDeviceFence>,
    begun: bool,
    terminal: bool,
}

impl Drop for Pending<'_> {
    fn drop(&mut self) {
        if self.begun && !self.terminal {
            self.stream.state.fail();
            if let Some(fence) = &mut self.fence {
                self.commands.append(&mut fence._commands);
            }
            self.runtime
                .quarantine(self.stream, std::mem::take(&mut self.commands));
        }
    }
}

struct Fixture {
    stream: CudaDeviceStream,
    sources: Vec<CudaDeviceBuffer>,
    expected: Vec<Vec<u8>>,
    table: CudaDeviceBuffer,
    producer: CudaFunction,
    last_packet: Weak<Packet>,
    last_product_packet: Option<Weak<dyn std::any::Any + Send + Sync>>,
    pooled_host: Option<PinnedHostSlice<u32>>,
    pool_returns: u64,
}

fn buffer(
    runtime: &CudaDeviceRuntime,
    stream: &Arc<CudaStream>,
    bytes: usize,
    element: ElementType,
    name: &str,
) -> Result<CudaDeviceBuffer> {
    let base = stream
        .clone_htod(&vec![CANARY; bytes])
        .map_err(|e| CudaDeviceRuntimeError::driver("test allocation", e))?;
    let pointer = base.device_ptr(stream).0;
    Ok(CudaDeviceBuffer {
        descriptor: BufferDescriptor {
            resource_id: ResourceId::new(name).unwrap(),
            size_bytes: bytes as u64,
            alignment_bytes: 16,
            usage: BufferUsage::Transfer,
            element_type: element,
        },
        runtime_instance: runtime.runtime_instance,
        allocation: Arc::new(CudaAllocation {
            _base: base,
            aligned_ptr: pointer,
            requested_bytes: bytes as u64,
            _tensor_owner: None,
        }),
    })
}

impl Fixture {
    fn new(runtime: &CudaDeviceRuntime) -> Result<Self> {
        let stream = runtime.create_stream()?;
        let sources = (0..CAPACITY)
            .map(|index| {
                buffer(
                    runtime,
                    &stream.stream,
                    BYTES,
                    ElementType::U32,
                    &format!("test.terminal.source.{index}"),
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let table = buffer(
            runtime,
            &stream.stream,
            CAPACITY * 8,
            ElementType::U8,
            "test.terminal.addresses",
        )?;
        let module = stream
            .stream
            .context()
            .load_module(Ptx::from_src(PRODUCER))
            .map_err(|e| CudaDeviceRuntimeError::driver("test producer module", e))?;
        let producer = module
            .load_function("write_tokens")
            .map_err(|e| CudaDeviceRuntimeError::driver("test producer function", e))?;
        stream
            .stream
            .synchronize()
            .map_err(|e| CudaDeviceRuntimeError::driver("test setup", e))?;
        Ok(Self {
            stream,
            sources,
            expected: vec![vec![CANARY; BYTES]; CAPACITY],
            table,
            producer,
            last_packet: Weak::new(),
            last_product_packet: None,
            pooled_host: None,
            pool_returns: 0,
        })
    }

    fn prepare_pool(&mut self) -> Result<u128> {
        assert!(self.pooled_host.is_none());
        let started = Instant::now();
        let mut host = unsafe { self.stream.stream.context().alloc_pinned::<u32>(CAPACITY) }
            .map_err(|e| CudaDeviceRuntimeError::driver("terminal packet cold pool pin", e))?;
        host.as_mut_slice()
            .map_err(|e| CudaDeviceRuntimeError::driver("terminal pool initialize", e))?
            .fill(0xdeadbeef);
        self.pooled_host = Some(host);
        Ok(started.elapsed().as_nanos())
    }

    fn run(
        &mut self,
        runtime: &CudaDeviceRuntime,
        route: Route,
        count: usize,
        generation: u32,
        delay: u64,
        fail_after: Option<usize>,
    ) -> Result<([u32; CAPACITY], serde_json::Value)> {
        let started = Instant::now();
        let mapping = (0..count)
            .map(|row| {
                let source = (row * 7 + generation as usize) % CAPACITY;
                let offset = 4 * (1 + (source + generation as usize) % 10);
                (source, offset)
            })
            .collect::<Vec<_>>();
        let regions = mapping
            .iter()
            .map(|&(source, offset)| self.sources[source].region(offset as u64..offset as u64 + 4))
            .collect::<Result<Vec<_>>>()?;
        let addresses = regions
            .iter()
            .flat_map(|r| r.device_ptr.to_le_bytes())
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let mut producer_regions = regions.clone();
        producer_regions.push(self.table.region(0..(count * 8) as u64)?);
        let function = self.producer.clone();
        let producer = CudaDeviceCommand::operation_with_host_storage_and_blas(
            "test.terminal.producer",
            producer_regions,
            vec![addresses],
            move |stream, _, regions, host| {
                let table = regions.last().unwrap().device_ptr;
                unsafe {
                    cudarc::driver::result::memcpy_htod_async(
                        table,
                        host[0].as_ref(),
                        stream.cu_stream(),
                    )
                }
                .map_err(|e| CudaDeviceRuntimeError::driver("test address upload", e))?;
                let count = count as u32;
                let mut launch = stream.launch_builder(&function);
                launch.arg(&table).arg(&count).arg(&generation).arg(&delay);
                unsafe {
                    launch.launch(LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (32, 1, 1),
                        shared_mem_bytes: 0,
                    })
                }
                .map_err(|e| CudaDeviceRuntimeError::driver("test token producer", e))?;
                Ok(())
            },
        )?;
        let mut commands = vec![producer];
        let packet = if matches!(route, Route::AsyncPacket | Route::PooledPacket) {
            let mut host = if matches!(route, Route::PooledPacket) {
                self.pooled_host.take().ok_or_else(|| {
                    CudaDeviceRuntimeError::contract("terminal pinned pool is unavailable")
                })?
            } else {
                unsafe { self.stream.stream.context().alloc_pinned::<u32>(count) }
                    .map_err(|e| CudaDeviceRuntimeError::driver("terminal packet pin", e))?
            };
            host.as_mut_slice()
                .map_err(|e| CudaDeviceRuntimeError::driver("packet poison", e))?
                .fill(0xdeadbeef);
            let packet = Arc::new(Packet {
                host: Mutex::new(host),
            });
            self.last_packet = Arc::downgrade(&packet);
            let retained = Arc::clone(&packet);
            commands.push(CudaDeviceCommand::operation(
                "test.terminal.async_packet",
                regions.clone(),
                move |stream, regions| retained.enqueue(stream, regions, fail_after),
            )?);
            Some(packet)
        } else {
            None
        };
        let product_payload = if matches!(route, Route::ProductPacket) {
            let (tail, payload) =
                terminal_readback::prepare_regions(runtime, &mut self.stream, regions.clone())?
                    .ok_or_else(|| {
                        CudaDeviceRuntimeError::contract("product terminal pool declined")
                    })?;
            if let Some(after) = fail_after {
                terminal_readback::inject_failure(&payload, after);
            }
            self.last_product_packet = Some(Arc::downgrade(&payload));
            commands.push(tail);
            Some(payload)
        } else {
            None
        };
        let prepared_ns = started.elapsed().as_nanos();
        let mut pending = Pending {
            runtime,
            stream: &mut self.stream,
            commands,
            fence: None,
            begun: false,
            terminal: false,
        };
        pending.stream.state.begin_submission()?;
        pending.begun = true;
        let event_start = pending
            .stream
            .stream
            .record_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(|e| CudaDeviceRuntimeError::driver("test start event", e))?;
        for command in &pending.commands {
            command.enqueue(&pending.stream.stream, &pending.stream.blas)?;
        }
        let event = pending
            .stream
            .stream
            .record_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(|e| CudaDeviceRuntimeError::driver("terminal original fence", e))?;
        pending.stream.state.submission_recorded()?;
        pending.fence = Some(CudaDeviceFence {
            event,
            timing: CudaFenceTiming::NotRequested,
            command_timing: CudaFenceCommandTiming::NotRequested,
            attribution: None,
            execution_facts: None,
            stream_state: Arc::clone(&pending.stream.state),
            terminal_accounted: AtomicBool::new(false),
            _stream: Arc::clone(&pending.stream.stream),
            _blas: Arc::clone(&pending.stream.blas),
            _commands: std::mem::take(&mut pending.commands),
        });
        let enqueued_ns = started.elapsed().as_nanos();
        let fence = pending.fence.as_ref().unwrap();
        let receipt = runtime.wait_fence(fence).map_err(|_| {
            CudaDeviceRuntimeError::contract("terminal original fence was indeterminate")
        })?;
        if !receipt.terminal().is_succeeded() {
            return Err(CudaDeviceRuntimeError::contract(
                "terminal original fence failed",
            ));
        }
        pending.terminal = true;
        let gpu_ns = cuda_event_elapsed_ns(&event_start, &fence.event);
        let mut output = [0_u32; CAPACITY];
        if let Some(packet) = &packet {
            let host = packet.host.lock().map_err(|_| {
                CudaDeviceRuntimeError::contract("terminal packet extract mutex poisoned")
            })?;
            let values = host
                .as_slice()
                .map_err(|e| CudaDeviceRuntimeError::driver("terminal packet extract", e))?;
            output[..count].copy_from_slice(&values[..count]);
            assert!(
                values[count..].iter().all(|&word| word == 0xdeadbeef),
                "inactive pinned packet tail changed"
            );
        } else if matches!(route, Route::Scalar) {
            for (row, &(source, offset)) in mapping.iter().enumerate() {
                let bytes = runtime.readback(
                    pending.stream,
                    &self.sources[source],
                    CopyRegion::new(offset as u64, 0, 4).unwrap(),
                    HostTransferLayout::new(ElementType::U32, 1).unwrap(),
                )?;
                output[row] = u32::from_le_bytes(bytes.try_into().unwrap());
            }
        }
        drop(receipt);
        drop(pending);
        if let Some(payload) = product_payload {
            let values = runtime
                .finish_terminal_u32_readbacks(payload)
                .expect("CUDA accepted product terminal packet")?;
            assert_eq!(values.len(), count);
            output[..count].copy_from_slice(&values);
        }
        let extracted_ns = started.elapsed().as_nanos();
        if let Some(packet) = packet {
            if matches!(route, Route::PooledPacket) {
                // No cached graph or old command may retain this wave's Packet.
                // Only the pinned storage returns; all source mappings are fresh.
                let packet = Arc::try_unwrap(packet).map_err(|_| {
                    CudaDeviceRuntimeError::contract("terminal packet still has an owner")
                })?;
                let host = packet.host.into_inner().map_err(|_| {
                    CudaDeviceRuntimeError::contract("terminal pool return mutex poisoned")
                })?;
                assert!(self.pooled_host.is_none());
                self.pooled_host = Some(host);
                self.pool_returns += 1;
            } else {
                drop(packet);
            }
        }
        drop(event_start);
        drop(regions);
        drop(mapping);
        // Updating the CPU canary oracle is outside measured terminal work.
        let total_ns = started.elapsed().as_nanos();
        for row in 0..count {
            let source = (row * 7 + generation as usize) % CAPACITY;
            let offset = 4 * (1 + (source + generation as usize) % 10);
            let expected = generation * 1009 + row as u32 * 31 + 17;
            assert_eq!(output[row], expected, "fresh token/order mismatch");
            self.expected[source][offset..offset + 4].copy_from_slice(&expected.to_le_bytes());
        }
        Ok((
            output,
            json!({ "route": format!("{route:?}"), "participants": count,
            "generation": generation, "prepare_ns": prepared_ns,
            "producer_delay_cycles": delay,
            "enqueue_ns": enqueued_ns - prepared_ns,
            "wait_extract_ns": extracted_ns - enqueued_ns,
            "destruction_ns": total_ns - extracted_ns, "total_ns": total_ns,
            "producer_and_packet_gpu_ns": gpu_ns,
            "token_dtoh_calls": count, "token_dtoh_bytes": count * 4 }),
        ))
    }

    fn check_sources(&mut self, runtime: &CudaDeviceRuntime) -> Result<()> {
        for (source, expected) in self.sources.iter().zip(&self.expected) {
            let actual = runtime.readback(
                &mut self.stream,
                source,
                CopyRegion::new(0, 0, BYTES as u64).unwrap(),
                HostTransferLayout::new(ElementType::U32, (BYTES / 4) as u64).unwrap(),
            )?;
            assert_eq!(&actual, expected, "source guards or inactive words changed");
        }
        Ok(())
    }
}

fn measured_group(
    fixture: &mut Fixture,
    runtime: &CudaDeviceRuntime,
    route: Route,
    count: usize,
    first_generation: u32,
) -> serde_json::Value {
    const WAVES: u32 = 16;
    let fields = [
        "prepare_ns",
        "enqueue_ns",
        "wait_extract_ns",
        "destruction_ns",
        "total_ns",
    ];
    let mut totals = [0_u64; 5];
    let mut gpu_total_ns = 0_u64;
    let mut gpu_samples = 0_u32;
    for wave in 0..WAVES {
        let (_, measured) = fixture
            .run(runtime, route, count, first_generation + wave, 0, None)
            .unwrap();
        for (total, field) in totals.iter_mut().zip(fields) {
            *total += measured[field].as_u64().unwrap();
        }
        if let Some(ns) = measured["producer_and_packet_gpu_ns"].as_u64() {
            gpu_total_ns += ns;
            gpu_samples += 1;
        }
    }
    json!({ "route": format!("{route:?}"), "participants": count,
        "fresh_waves": WAVES, "first_generation": first_generation,
        "totals_ns": fields.into_iter().zip(totals).collect::<std::collections::BTreeMap<_, _>>(),
        "means_ns": fields.into_iter().zip(totals.map(|n| n as f64 / WAVES as f64))
            .collect::<std::collections::BTreeMap<_, _>>(),
        "producer_and_packet_gpu_total_ns": gpu_total_ns, "gpu_samples": gpu_samples,
        "token_dtoh_calls": count * WAVES as usize, "token_dtoh_bytes": count * 4 * WAVES as usize })
}

#[test]
#[ignore = "requires an exclusive CUDA device; backend terminal transaction diagnostic"]
fn scattered_terminal_packet_preserves_fresh_values_and_complete_transaction_pairs() {
    let cold_started = Instant::now();
    let runtime = CudaDeviceRuntime::new(
        crate::backend::cuda::vnext_ops::cuda_vnext_runtime_config(
            0,
            DeviceId::new("device.test.terminal-packet").unwrap(),
            AttentionExecutionPolicy::Portable,
        )
        .unwrap(),
    )
    .unwrap();
    let mut scalar = Fixture::new(&runtime).unwrap();
    let mut packet = Fixture::new(&runtime).unwrap();
    let mut pooled = Fixture::new(&runtime).unwrap();
    let pool_pin_ns = pooled.prepare_pool().unwrap();
    println!(
        "terminal_packet {}",
        json!({ "kind": "scope", "cold_setup_ns": cold_started.elapsed().as_nanos(),
        "scope": "backend fresh mapping/pin/producer enqueue/original fence/readback/extract/drop; excludes Core lane locks, admission, argmax, model execution and serving",
        "producer": "same one-CTA fresh U32 writer plus pointer-table upload in all three arms",
        "graph_capture": false, "packet_reused_across_waves": false,
        "pooled_storage_capacity_u32": CAPACITY, "pool_cold_pin_ns": pool_pin_ns,
        "pooled_storage_rule": "only pinned allocation reused; fresh Packet and mappings each wave; return after successful original fence, extraction and last command drop",
        "persistent_source_capacity": CAPACITY, "source_allocations_per_arm": CAPACITY,
        "timing_events": "one start and original completion fence per transaction; B/C additionally retain pinned DMA-use guard event",
        "gpu_span_limit": "A span ends before scalar readbacks; B/C include async copies. Only complete host total is comparable net transaction time" })
    );
    for (generation, count) in [(1, 32), (2, 8), (3, 32)] {
        let a = scalar
            .run(&runtime, Route::Scalar, count, generation, 100_000, None)
            .unwrap();
        let b = packet
            .run(
                &runtime,
                Route::AsyncPacket,
                count,
                generation,
                100_000,
                None,
            )
            .unwrap();
        assert_eq!(a.0, b.0);
        let c = pooled
            .run(
                &runtime,
                Route::PooledPacket,
                count,
                generation,
                100_000,
                None,
            )
            .unwrap();
        assert_eq!(a.0, c.0);
        scalar.check_sources(&runtime).unwrap();
        packet.check_sources(&runtime).unwrap();
        pooled.check_sources(&runtime).unwrap();
        assert!(pooled.last_packet.upgrade().is_none());
        assert!(pooled.pooled_host.is_some());
        assert!(
            packet.last_packet.upgrade().is_none(),
            "fresh pin survived terminal destruction"
        );
        println!(
            "terminal_packet {}",
            json!({ "kind": "correctness_transaction",
            "first_sample": generation == 1, "a": a.1, "b": b.1, "c": c.1 })
        );
    }
    for count in [8, 32] {
        for pair in 0..8 {
            let generation = 1000 + count as u32 * 100 + pair * 16;
            // All six permutations, then a reverse pair: each pair of arms
            // appears in each relative order four times across eight rounds.
            let (order, indices) = [
                ("ABC", [0, 1, 2]),
                ("CBA", [2, 1, 0]),
                ("BCA", [1, 2, 0]),
                ("ACB", [0, 2, 1]),
                ("CAB", [2, 0, 1]),
                ("BAC", [1, 0, 2]),
                ("ABC", [0, 1, 2]),
                ("CBA", [2, 1, 0]),
            ][pair as usize];
            let mut rows = [
                serde_json::Value::Null,
                serde_json::Value::Null,
                serde_json::Value::Null,
            ];
            for index in indices {
                rows[index] = match index {
                    0 => measured_group(&mut scalar, &runtime, Route::Scalar, count, generation),
                    1 => {
                        measured_group(&mut packet, &runtime, Route::AsyncPacket, count, generation)
                    }
                    2 => measured_group(
                        &mut pooled,
                        &runtime,
                        Route::PooledPacket,
                        count,
                        generation,
                    ),
                    _ => unreachable!(),
                };
            }
            println!(
                "terminal_packet {}",
                json!({ "kind": "pair", "pair": pair,
                "order": order, "a": rows[0], "b": rows[1], "c": rows[2] })
            );
        }
        scalar.check_sources(&runtime).unwrap();
        packet.check_sources(&runtime).unwrap();
        pooled.check_sources(&runtime).unwrap();
        assert!(pooled.pooled_host.is_some());
    }
    // A safe injected partial enqueue error exercises actual quarantine retention.
    let mut failed = Fixture::new(&runtime).unwrap();
    assert!(failed
        .run(&runtime, Route::AsyncPacket, 32, 999, 100_000, Some(3))
        .is_err());
    assert!(failed.last_packet.upgrade().is_some());
    assert!(!runtime.quarantined().is_empty());
    failed.stream.stream.synchronize().unwrap();
    runtime.release_quarantine(failed.stream.id);
    failed.stream.state.synchronized();
    assert!(failed.last_packet.upgrade().is_none());
    let mut failed_pool = Fixture::new(&runtime).unwrap();
    failed_pool.prepare_pool().unwrap();
    assert!(failed_pool
        .run(&runtime, Route::PooledPacket, 32, 1001, 100_000, Some(3))
        .is_err());
    assert!(failed_pool.pooled_host.is_none());
    assert!(failed_pool.last_packet.upgrade().is_some());
    failed_pool.stream.stream.synchronize().unwrap();
    runtime.release_quarantine(failed_pool.stream.id);
    failed_pool.stream.state.synchronized();
    assert!(failed_pool.last_packet.upgrade().is_none());
    assert!(
        failed_pool.pooled_host.is_none(),
        "quarantined storage re-entered pool"
    );
    assert_eq!(failed_pool.pool_returns, 0);
    assert!(failed_pool
        .run(&runtime, Route::PooledPacket, 8, 1002, 0, None)
        .is_err());
    // Pool ownership destruction is a distinct cold/close lifecycle cost.
    let pool_close_started = Instant::now();
    drop(pooled.pooled_host.take());
    println!(
        "terminal_packet {}",
        json!({ "kind": "pool_close",
        "pool_unpin_ns": pool_close_started.elapsed().as_nanos(),
        "successful_exclusive_returns": pooled.pool_returns,
        "cold_pin_ns": pool_pin_ns })
    );
    println!(
        "terminal_packet {}",
        json!({ "kind": "correctness",
        "fresh_values_and_order": true, "capacity_32_8_32": true,
        "all_source_bytes_and_guards": true, "partial_enqueue_retained_until_drain": true,
        "pooled_partial_failure_never_returns_storage": true,
        "backend_only_no_core_cancellation_claim": true })
    );
}

#[test]
#[ignore = "requires a CUDA device; actual terminal prepare/tail/finish lifecycle"]
fn product_terminal_packet_reuses_only_after_original_fence() {
    let runtime = CudaDeviceRuntime::new(
        crate::backend::cuda::vnext_ops::cuda_vnext_runtime_config(
            0,
            DeviceId::new("device.test.product-terminal-packet").unwrap(),
            AttentionExecutionPolicy::Portable,
        )
        .unwrap(),
    )
    .unwrap();
    let mut scalar = Fixture::new(&runtime).unwrap();
    let mut product = Fixture::new(&runtime).unwrap();
    assert!(terminal_readback::available_storage(&product.stream).is_none());
    assert!(
        terminal_readback::prepare_regions(&runtime, &mut product.stream, Vec::new())
            .unwrap()
            .is_none()
    );
    let one_source = product.sources[0].region(4..8).unwrap();
    assert!(terminal_readback::prepare_regions(
        &runtime,
        &mut product.stream,
        vec![one_source; CAPACITY + 1]
    )
    .unwrap()
    .is_none());
    let wrong_element = product.table.region(4..8).unwrap();
    assert!(
        terminal_readback::prepare_regions(&runtime, &mut product.stream, vec![wrong_element])
            .is_err()
    );
    assert!(terminal_readback::available_storage(&product.stream).is_none());
    for (generation, count) in [(11, 32), (12, 8), (13, 32)] {
        let reference = scalar
            .run(&runtime, Route::Scalar, count, generation, 100_000, None)
            .unwrap();
        let actual = product
            .run(
                &runtime,
                Route::ProductPacket,
                count,
                generation,
                100_000,
                None,
            )
            .unwrap();
        assert_eq!(actual.0, reference.0);
        scalar.check_sources(&runtime).unwrap();
        product.check_sources(&runtime).unwrap();
        assert!(product
            .last_product_packet
            .as_ref()
            .unwrap()
            .upgrade()
            .is_none());
        let storage = terminal_readback::available_storage(&product.stream).unwrap();
        assert_eq!(storage.len(), CAPACITY);
        assert_eq!(&storage[..count], &actual.0[..count]);
        assert!(storage[count..].iter().all(|&v| v == 0xdeadbeef));
        println!(
            "terminal_packet_product {}",
            json!({
                "participants": count, "generation": generation,
                "actual_product_prepare_tail_finish": true,
                "all_values_order_and_source_guards": true,
                "inactive_capacity_poison_preserved": true,
                "no_core_batch_or_submit_integration_claim": true,
                "a": reference.1, "b": actual.1,
            })
        );
    }
    // A prepared packet checks out the only storage. Decline another prepare
    // without allocating a second pin; early finish cannot consume live work.
    let regions = vec![product.sources[0].region(4..8).unwrap()];
    let (tail, payload) =
        terminal_readback::prepare_regions(&runtime, &mut product.stream, regions.clone())
            .unwrap()
            .unwrap();
    assert!(
        terminal_readback::prepare_regions(&runtime, &mut product.stream, regions.clone())
            .unwrap()
            .is_none()
    );
    assert!(runtime
        .finish_terminal_u32_readbacks(Arc::clone(&payload))
        .unwrap()
        .is_err());
    drop(tail);
    drop(payload);
    assert!(
        terminal_readback::prepare_regions(&runtime, &mut product.stream, regions)
            .unwrap()
            .is_none(),
        "discarded packet must not silently allocate or recycle"
    );

    let mut failed = Fixture::new(&runtime).unwrap();
    assert!(failed
        .run(&runtime, Route::ProductPacket, 32, 21, 100_000, Some(3))
        .is_err());
    assert!(failed
        .last_product_packet
        .as_ref()
        .unwrap()
        .upgrade()
        .is_some());
    assert!(terminal_readback::available_storage(&failed.stream).is_none());
    failed.stream.stream.synchronize().unwrap();
    runtime.release_quarantine(failed.stream.id);
    failed.stream.state.synchronized();
    assert!(failed
        .last_product_packet
        .as_ref()
        .unwrap()
        .upgrade()
        .is_none());
    assert!(terminal_readback::available_storage(&failed.stream).is_none());
    println!(
        "terminal_packet_product {}",
        json!({
            "kind": "lifecycle", "pool_busy_declines_before_enqueue": true,
            "premature_finish_rejected": true, "discarded_pin_not_reallocated": true,
            "partial_failure_retained_until_quarantine_drain": true,
            "failed_pin_not_recycled": true,
        })
    );
}
