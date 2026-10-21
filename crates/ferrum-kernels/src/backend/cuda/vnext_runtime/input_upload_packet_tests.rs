//! Complete backend upload transaction diagnostic, not Core Request admission.
//! A uses the original encode_upload; B has fresh packet allocation every wave.
//! Targets are separate real CUDA allocations, never offsets in a fake arena.

use super::*;
use cudarc::driver::{CudaFunction, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::compile_ptx;
use ferrum_runtime::{BufferUsage, ResourceId};
use serde_json::json;
use std::sync::Weak;

type Result<T> = std::result::Result<T, CudaDeviceRuntimeError>;
const BYTES: usize = 128;
const CANARY: u8 = 0xa5;
const HEADER: usize = 24;

#[derive(Clone, Copy, Debug)]
enum Route {
    OriginalUploads,
    FreshAddressedPacket,
    ProductPacket,
}

struct Write {
    target: usize,
    offset: u64,
    layout: HostTransferLayout,
    bytes: Vec<u8>,
}

struct PacketOwner {
    device: CudaSlice<u8>,
    function: CudaFunction,
    count: u32,
}

// Same partial-submit retention boundary as terminal_packet_tests. Even an
// error after H2D, before scatter/fence, transfers all owners to quarantine.
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

fn buffer(
    runtime: &CudaDeviceRuntime,
    stream: &Arc<CudaStream>,
    name: String,
    element_type: ElementType,
) -> Result<CudaDeviceBuffer> {
    let base = stream
        .clone_htod(&[CANARY; BYTES])
        .map_err(|e| CudaDeviceRuntimeError::driver("input fixture allocation", e))?;
    let ptr = base.device_ptr(stream).0;
    Ok(CudaDeviceBuffer {
        descriptor: BufferDescriptor {
            resource_id: ResourceId::new(name).unwrap(),
            size_bytes: BYTES as u64,
            alignment_bytes: 16,
            usage: BufferUsage::Activations,
            element_type,
        },
        runtime_instance: runtime.runtime_instance,
        allocation: Arc::new(CudaAllocation {
            _base: base,
            aligned_ptr: ptr,
            requested_bytes: BYTES as u64,
            _tensor_owner: None,
        }),
    })
}

fn addressed_packet(
    runtime: &CudaDeviceRuntime,
    targets: &[CudaDeviceBuffer],
    writes: &[Write],
    function: &CudaFunction,
    fail_after_h2d: bool,
) -> Result<(CudaDeviceCommand, Weak<PacketOwner>, usize)> {
    let invalid = CudaDeviceRuntimeError::contract;
    if writes.is_empty() {
        return Err(invalid("empty input packet"));
    }
    let count = u32::try_from(writes.len()).map_err(|_| invalid("input packet count overflow"))?;
    let mut regions = Vec::with_capacity(writes.len());
    let mut ranges = Vec::with_capacity(writes.len());
    let mut total = writes
        .len()
        .checked_mul(HEADER)
        .ok_or_else(|| invalid("input packet header overflow"))?;
    for write in writes {
        let target = targets
            .get(write.target)
            .ok_or_else(|| invalid("unknown input target"))?;
        runtime.validate_buffer(target)?;
        write
            .layout
            .validate_bytes(write.bytes.len())
            .map_err(|error| CudaDeviceRuntimeError::contract(error.to_string()))?;
        if write.layout.element_type() != target.descriptor.element_type {
            return Err(invalid("input packet element type mismatch"));
        }
        let length =
            u64::try_from(write.bytes.len()).map_err(|_| invalid("input byte length overflow"))?;
        let end = checked_end(
            write.offset,
            length,
            target.descriptor.size_bytes,
            "input packet",
        )?;
        let region = target.region(write.offset..end)?;
        let physical_end = region
            .device_ptr
            .checked_add(length)
            .ok_or_else(|| invalid("input pointer overflow"))?;
        ranges.push((region.device_ptr, physical_end));
        regions.push(region);
        total = total
            .checked_add(write.bytes.len())
            .ok_or_else(|| invalid("input packet length overflow"))?;
    }
    ranges.sort_unstable();
    if ranges.windows(2).any(|pair| pair[0].1 > pair[1].0) {
        return Err(invalid(
            "overlapping input writes require ordered original uploads",
        ));
    }
    let mut bytes = Vec::with_capacity(total);
    let mut source = writes.len() * HEADER;
    for (write, region) in writes.iter().zip(&regions) {
        for field in [source as u64, region.device_ptr, write.bytes.len() as u64] {
            bytes.extend_from_slice(&field.to_le_bytes());
        }
        source += write.bytes.len();
    }
    for write in writes {
        bytes.extend_from_slice(&write.bytes);
    }
    // Fresh allocation and cross-stream readiness are intentionally inside net.
    let device = unsafe { runtime.allocation_stream.alloc::<u8>(total) }
        .map_err(|e| CudaDeviceRuntimeError::driver("input packet allocation", e))?;
    let owner = Arc::new(PacketOwner {
        device,
        function: function.clone(),
        count,
    });
    if let Err(error) = runtime.allocation_stream.synchronize() {
        std::mem::forget(owner);
        return Err(CudaDeviceRuntimeError::driver(
            "input packet allocation readiness",
            error,
        ));
    }
    let weak = Arc::downgrade(&owner);
    let command = CudaDeviceCommand::operation_with_host_storage_and_blas(
        "test.input_upload_packet",
        regions,
        vec![bytes.into_boxed_slice()],
        move |stream, _, _, host| {
            let packet = owner.device.device_ptr(stream).0;
            unsafe {
                cudarc::driver::result::memcpy_htod_async(
                    packet,
                    host[0].as_ref(),
                    stream.cu_stream(),
                )
            }
            .map_err(|e| CudaDeviceRuntimeError::driver("input packet H2D", e))?;
            if fail_after_h2d {
                return Err(invalid("injected failure after packet H2D before scatter"));
            }
            let mut launch = stream.launch_builder(&owner.function);
            launch.arg(&packet).arg(&owner.count);
            unsafe {
                launch.launch(LaunchConfig {
                    grid_dim: (owner.count, 1, 1),
                    block_dim: (32, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .map_err(|e| CudaDeviceRuntimeError::driver("input packet scatter", e))?;
            Ok(())
        },
    )?;
    Ok((command, weak, total))
}

struct Fixture {
    stream: CudaDeviceStream,
    targets: Vec<CudaDeviceBuffer>,
    observers: Vec<CudaDeviceBuffer>,
    expected: Vec<Vec<u8>>,
    function: CudaFunction,
    width: usize,
    last_packet: Weak<PacketOwner>,
    setup_ns: u128,
}

impl Fixture {
    fn new(runtime: &CudaDeviceRuntime, function: &CudaFunction, width: usize) -> Result<Self> {
        let started = Instant::now();
        let stream = runtime.create_stream()?;
        let mut targets = Vec::new();
        let mut observers = Vec::new();
        for index in 0..2 * width {
            let element = if index < width {
                ElementType::U32
            } else {
                ElementType::F32
            };
            targets.push(buffer(
                runtime,
                &stream.stream,
                format!("test.input.target.{index}"),
                element,
            )?);
            observers.push(buffer(
                runtime,
                &stream.stream,
                format!("test.input.observer.{index}"),
                element,
            )?);
        }
        stream
            .stream
            .synchronize()
            .map_err(|e| CudaDeviceRuntimeError::driver("input fixture ready", e))?;
        let unique: BTreeSet<_> = targets.iter().map(|b| b.allocation.aligned_ptr).collect();
        assert_eq!(
            unique.len(),
            2 * width,
            "targets must be separate live allocations"
        );
        Ok(Self {
            stream,
            targets,
            observers,
            expected: vec![vec![CANARY; BYTES]; 2 * width],
            function: function.clone(),
            width,
            last_packet: Weak::new(),
            setup_ns: started.elapsed().as_nanos(),
        })
    }

    fn writes(&self, generation: u32) -> Vec<Write> {
        let count = 2 * self.width;
        (0..count)
            .map(|order| {
                // Five is coprime with both 16 and 64; all targets occur once.
                let target = (order * 5 + generation as usize) % count;
                let offset = (4 * (1 + (target + generation as usize) % 12)) as u64;
                let (element, elements, bytes) = if target < self.width {
                    (
                        ElementType::U32,
                        2,
                        [
                            generation.wrapping_mul(1009) + target as u32,
                            generation.wrapping_mul(1013) + target as u32 + 7,
                        ]
                        .into_iter()
                        .flat_map(u32::to_le_bytes)
                        .collect(),
                    )
                } else {
                    let value = 1.0_f32 + (generation % 31) as f32 / 32.0 + target as f32 / 128.0;
                    (ElementType::F32, 1, value.to_le_bytes().to_vec())
                };
                Write {
                    target,
                    offset,
                    layout: HostTransferLayout::new(element, elements).unwrap(),
                    bytes,
                }
            })
            .collect()
    }

    fn run(
        &mut self,
        runtime: &CudaDeviceRuntime,
        route: Route,
        generation: u32,
        successor: bool,
        detach_targets: bool,
        fail_after_h2d: bool,
    ) -> Result<serde_json::Value> {
        assert!(!detach_targets || successor);
        let net = Instant::now();
        let writes = self.writes(generation);
        let payload_bytes: usize = writes.iter().map(|w| w.bytes.len()).sum();
        let mut commands = Vec::with_capacity(writes.len() + self.targets.len());
        let upload_bytes = match route {
            Route::OriginalUploads => {
                assert!(!fail_after_h2d);
                for write in &writes {
                    commands.push(runtime.encode_upload(
                        &write.bytes,
                        write.layout,
                        &self.targets[write.target],
                        write.offset,
                    )?);
                }
                self.last_packet = Weak::new();
                payload_bytes
            }
            Route::FreshAddressedPacket => {
                let (command, weak, bytes) = addressed_packet(
                    runtime,
                    &self.targets,
                    &writes,
                    &self.function,
                    fail_after_h2d,
                )?;
                self.last_packet = weak;
                commands.push(command);
                bytes
            }
            Route::ProductPacket => {
                assert!(!fail_after_h2d);
                let entries = writes
                    .iter()
                    .map(|write| {
                        runtime
                            .encode_upload(
                                &write.bytes,
                                write.layout,
                                &self.targets[write.target],
                                write.offset,
                            )
                            .map(|command| (DeviceCommandPhase::DynamicBinding, None, command))
                    })
                    .collect::<Result<Vec<_>>>()?;
                let entries = input_upload_packet::coalesce(runtime, entries)?;
                assert_eq!(entries.len(), 1);
                assert_eq!(entries[0].2.compute_dispatch_count, 1);
                assert_eq!(entries[0].2.transfer_command_count, 1);
                assert!(entries[0].2.replay_key.is_none());
                assert!(runtime.input_upload_scatter.live_bytes_for_test() > 0);
                commands.extend(entries.into_iter().map(|(_, _, command)| command));
                payload_bytes + writes.len() * HEADER
            }
        };
        // Correctness-only successor observes every byte, including old holes.
        // Measurement runs omit it; no hidden D2D copies enter upload timings.
        if successor {
            for (target, observer) in self.targets.iter().zip(&self.observers) {
                commands.push(runtime.encode_copy(
                    target,
                    observer,
                    CopyRegion::new(0, 0, BYTES as u64).unwrap(),
                )?);
            }
        }
        let weak_targets: Vec<_> = if detach_targets {
            self.targets
                .iter()
                .map(|b| Arc::downgrade(&b.allocation))
                .collect()
        } else {
            Vec::new()
        };
        if detach_targets {
            drop(std::mem::take(&mut self.targets));
        }
        drop(writes);
        let prepared_ns = net.elapsed().as_nanos();
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
        let start = pending
            .stream
            .stream
            .record_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(|e| CudaDeviceRuntimeError::driver("input upload start event", e))?;
        let enqueue = Instant::now();
        for command in &pending.commands {
            command.enqueue(&pending.stream.stream, &pending.stream.blas)?;
        }
        let enqueue_ns = enqueue.elapsed().as_nanos();
        let event = pending
            .stream
            .stream
            .record_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(|e| CudaDeviceRuntimeError::driver("input upload original fence", e))?;
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
        let fence = pending.fence.as_ref().unwrap();
        let receipt = runtime
            .wait_fence(fence)
            .map_err(|_| CudaDeviceRuntimeError::contract("input upload fence indeterminate"))?;
        if !receipt.terminal().is_succeeded() {
            return Err(CudaDeviceRuntimeError::contract(
                "input upload fence failed",
            ));
        }
        pending.terminal = true;
        let stream_interval_ns = cuda_event_elapsed_ns(&start, &fence.event);
        let terminal_ns = net.elapsed().as_nanos();
        drop(receipt);
        drop(pending);
        drop(start);
        if !matches!(route, Route::OriginalUploads) {
            // Includes completion of the fresh scratch's queued deallocation.
            runtime
                .allocation_stream
                .synchronize()
                .map_err(|e| CudaDeviceRuntimeError::driver("input packet retirement", e))?;
        }
        let total_ns = net.elapsed().as_nanos();
        assert!(
            self.last_packet.upgrade().is_none(),
            "packet retained after terminal retirement"
        );
        assert!(
            weak_targets.iter().all(|w| w.upgrade().is_none()),
            "caller-free target owner outlived all commands"
        );
        if matches!(route, Route::ProductPacket) {
            assert_eq!(runtime.input_upload_scatter.live_bytes_for_test(), 0);
        }
        // Full byte oracle and readback are outside the measured transaction.
        for write in self.writes(generation) {
            let offset = write.offset as usize;
            self.expected[write.target][offset..offset + write.bytes.len()]
                .copy_from_slice(&write.bytes);
        }
        self.check(runtime, successor)?;
        Ok(
            json!({"route":format!("{route:?}"),"participants":self.width,"generation":generation,
            "uploads":self.width*2,"payload_bytes":payload_bytes,"uploaded_bytes":upload_bytes,
            "h2d_calls":if matches!(route,Route::OriginalUploads){self.width*2}else{1},
            "scatter_dispatches":usize::from(!matches!(route,Route::OriginalUploads)),
            "prepare_ns":prepared_ns,"enqueue_api_ns":enqueue_ns,"through_terminal_ns":terminal_ns,
            "retirement_ns":total_ns-terminal_ns,"total_ns":total_ns,"stream_interval_ns":stream_interval_ns,
            "successor_copies":if successor{self.width*2}else{0},"caller_targets_dropped_before_enqueue":detach_targets}),
        )
    }

    fn check(&mut self, runtime: &CudaDeviceRuntime, observer: bool) -> Result<()> {
        let buffers = if observer {
            &self.observers
        } else {
            &self.targets
        };
        assert_eq!(buffers.len(), self.expected.len());
        for (buffer, expected) in buffers.iter().zip(&self.expected) {
            let bytes = runtime.readback(
                &mut self.stream,
                buffer,
                CopyRegion::new(0, 0, BYTES as u64).unwrap(),
                HostTransferLayout::new(buffer.descriptor.element_type, (BYTES / 4) as u64)
                    .unwrap(),
            )?;
            assert_eq!(
                &bytes, expected,
                "fresh values, prior bytes, or guard holes changed"
            );
        }
        Ok(())
    }
}

#[test]
#[ignore = "exclusive CUDA: original independent uploads vs fresh addressed packet complete transaction"]
fn input_upload_packet_preserves_independent_targets_and_complete_cost() {
    let setup = Instant::now();
    let runtime = CudaDeviceRuntime::new(
        super::super::vnext_ops::cuda_vnext_runtime_config(
            0,
            DeviceId::new("device.test.input-upload-packet").unwrap(),
            AttentionExecutionPolicy::Portable,
        )
        .unwrap(),
    )
    .unwrap();
    let ptx = compile_ptx(include_str!("input_upload_packet_tests/scatter.cu")).unwrap();
    let module = runtime.context.load_module(ptx).unwrap();
    let function = module
        .load_function("ferrum_test_input_upload_scatter")
        .unwrap();
    println!(
        "input_upload_packet_plan {}",
        json!({"runtime_and_module_setup_ns":setup.elapsed().as_nanos(),
        "scope":"backend physical allocation transaction; no Core Request admission or serving claim",
        "baseline":"original runtime.encode_upload for every 8B U32[2] and 4B F32 value",
        "candidate":"fresh packet allocation + allocation-stream sync + packing + one pageable H2D + addressed scatter",
        "net":"fresh payload construction, bounds/owner validation, command/source storage, packet allocation/readiness, enqueue/events, original runtime fence, all transient retirement including packet free synchronization",
        "stream_interval":"after preparation to terminal event, includes host enqueue gaps; not pure GPU busy",
        "excluded":"persistent target/observer/stream/module setup and full-byte readback oracle",
        "reuse_arm":false,"performance_threshold":null,"order":"four alternating ABBA/BAAB quads; four fresh waves per position"})
    );
    for width in [8, 32] {
        // Independent physical sets for A/B, with fresh source order every wave.
        let mut arms = [
            Fixture::new(&runtime, &function, width).unwrap(),
            Fixture::new(&runtime, &function, width).unwrap(),
        ];
        println!(
            "input_upload_packet_setup {}",
            json!({"participants":width,"arm_setup_ns":[arms[0].setup_ns,arms[1].setup_ns],"target_allocations_per_arm":width*2})
        );
        for arm in 0..2 {
            let route = if arm == 0 {
                Route::OriginalUploads
            } else {
                Route::FreshAddressedPacket
            };
            for generation in 1..=4 {
                arms[arm]
                    .run(&runtime, route, generation, true, false, false)
                    .unwrap();
            }
        }
        for quad in 0..4 {
            let order = if quad % 2 == 0 {
                [0, 1, 1, 0]
            } else {
                [1, 0, 0, 1]
            };
            for (position, arm) in order.into_iter().enumerate() {
                let route = if arm == 0 {
                    Route::OriginalUploads
                } else {
                    Route::FreshAddressedPacket
                };
                let mut rows = Vec::new();
                for wave in 0..4 {
                    rows.push(
                        arms[arm]
                            .run(&runtime, route, 100 + quad * 4 + wave, false, false, false)
                            .unwrap(),
                    );
                }
                println!(
                    "input_upload_packet_measurement {}",
                    json!({"participants":width,"quad":quad,"position":position,"arm":arm,"waves":rows})
                );
            }
        }
        for route in [Route::OriginalUploads, Route::FreshAddressedPacket] {
            let mut lifetime = Fixture::new(&runtime, &function, width).unwrap();
            lifetime
                .run(&runtime, route, 777, true, true, false)
                .unwrap();
        }
        let mut failure = Fixture::new(&runtime, &function, width).unwrap();
        assert!(failure
            .run(
                &runtime,
                Route::FreshAddressedPacket,
                888,
                false,
                false,
                true
            )
            .is_err());
        assert!(
            failure.last_packet.upgrade().is_some(),
            "partial H2D lost packet before drain"
        );
        assert_eq!(runtime.quarantined().len(), 1);
        runtime.synchronize(&mut failure.stream).unwrap();
        runtime.allocation_stream.synchronize().unwrap();
        assert!(runtime.quarantined().is_empty());
        assert!(failure.last_packet.upgrade().is_none());
        failure.check(&runtime, false).unwrap();
        println!(
            "input_upload_packet_correctness {}",
            json!({"participants":width,
            "all_destination_bytes_and_holes":true,"fresh_order_and_values":true,
            "same_stream_full_buffer_successor":true,"caller_owner_drop_before_enqueue":true,
            "real_h2d_then_injected_failure_quarantined_until_drain":true,"real_driver_fault":false})
        );
    }
}

#[test]
#[ignore = "CUDA product input packet: actual coalescer, independent owners, fresh values and fences"]
fn input_upload_product_preserves_bytes_lifetimes_and_fallbacks() {
    let runtime = CudaDeviceRuntime::new(
        super::super::vnext_ops::cuda_vnext_runtime_config(
            0,
            DeviceId::new("device.test.input-upload-product").unwrap(),
            AttentionExecutionPolicy::Portable,
        )
        .unwrap(),
    )
    .unwrap();
    // Fixture's diagnostic function is unused by ProductPacket. The product
    // path loads and invokes the embedded addressed-scatter implementation.
    let ptx = compile_ptx(include_str!("input_upload_packet_tests/scatter.cu")).unwrap();
    let module = runtime.context.load_module(ptx).unwrap();
    let function = module
        .load_function("ferrum_test_input_upload_scatter")
        .unwrap();
    for width in [4, 8, 32] {
        let mut fixture = Fixture::new(&runtime, &function, width).unwrap();
        for generation in 1..=4 {
            fixture
                .run(
                    &runtime,
                    Route::ProductPacket,
                    generation,
                    true,
                    false,
                    false,
                )
                .unwrap();
        }
        fixture
            .run(&runtime, Route::ProductPacket, 19, true, true, false)
            .unwrap();
    }

    let fixture = Fixture::new(&runtime, &function, 8).unwrap();
    let writes = fixture.writes(23);
    let entries = || {
        writes
            .iter()
            .map(|write| {
                (
                    DeviceCommandPhase::DynamicBinding,
                    None,
                    runtime
                        .encode_upload(
                            &write.bytes,
                            write.layout,
                            &fixture.targets[write.target],
                            write.offset,
                        )
                        .unwrap(),
                )
            })
            .collect::<Vec<_>>()
    };
    let snapshot = |commands: &[(DeviceCommandPhase, Option<u32>, CudaDeviceCommand)]| {
        commands
            .iter()
            .map(|(phase, node, command)| {
                let executable = command.executable.as_ref().unwrap();
                (
                    *phase,
                    *node,
                    executable.regions[0].device_ptr,
                    executable.host_storage[0].to_vec(),
                )
            })
            .collect::<Vec<_>>()
    };
    // Capacity refusal must retain every command, source byte and order.
    let mut held = Vec::new();
    while let Some(reservation) = runtime.input_upload_scatter.hold_test_budget() {
        held.push(reservation);
    }
    let original = entries();
    let expected = snapshot(&original);
    let fallback = input_upload_packet::coalesce(&runtime, original).unwrap();
    assert_eq!(snapshot(&fallback), expected);
    drop((fallback, held));
    assert_eq!(runtime.input_upload_scatter.live_bytes_for_test(), 0);

    // A repeated real destination forces the complete run back to ordered
    // copies. A packet may never race two overlapping logical uploads.
    let mut original = entries();
    let first = &writes[0];
    original.push((
        DeviceCommandPhase::DynamicBinding,
        None,
        runtime
            .encode_upload(
                &first.bytes,
                first.layout,
                &fixture.targets[first.target],
                first.offset,
            )
            .unwrap(),
    ));
    let expected = snapshot(&original);
    let fallback = input_upload_packet::coalesce(&runtime, original).unwrap();
    assert_eq!(snapshot(&fallback), expected);
    assert_eq!(runtime.input_upload_scatter.live_bytes_for_test(), 0);
    drop(fallback);

    // A node-attributed command is a boundary even if it is also a host copy.
    let mut original = entries();
    original[8].1 = Some(17);
    let grouped = input_upload_packet::coalesce(&runtime, original).unwrap();
    assert_eq!(grouped.len(), 9); // eight merged; boundary and seven direct
    assert_eq!(grouped[0].2.compute_dispatch_count, 1);
    assert_eq!(grouped[1].1, Some(17));
    assert_eq!(grouped[1].2.compute_dispatch_count, 0);
    drop(grouped);
    runtime.allocation_stream.synchronize().unwrap();
    assert_eq!(runtime.input_upload_scatter.live_bytes_for_test(), 0);

    // The typed marker cannot authorize a foreign buffer, wrong element type,
    // or an out-of-range destination; original encode_upload refuses them.
    let target = &fixture.targets[0];
    assert!(runtime
        .encode_upload(
            &[0; 4],
            HostTransferLayout::new(ElementType::F32, 1).unwrap(),
            target,
            0
        )
        .is_err());
    assert!(runtime
        .encode_upload(
            &[0; 8],
            HostTransferLayout::new(ElementType::U32, 2).unwrap(),
            target,
            BYTES as u64 - 4
        )
        .is_err());
    let mut foreign = entries();
    foreign[0].2.runtime_instance = runtime.runtime_instance + 1;
    assert!(input_upload_packet::coalesce(&runtime, foreign).is_err());
    println!("product input packet: byte-exact fresh independent allocations, same-stream successor, caller-owner release, shared capacity/overlap fallback, attributed boundary and typed rejection verified");
}
