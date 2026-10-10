use super::*;

const CANARY: u8 = 0xa5;

struct Fixture {
    runtime: CudaDeviceRuntime,
    stream: CudaDeviceStream,
    region: CudaBufferRegion,
}

impl Fixture {
    fn new() -> Self {
        let runtime = CudaDeviceRuntime::new(
            crate::backend::cuda::vnext_ops::cuda_vnext_runtime_config(
                0,
                DeviceId::new("device.test.pinned-bindings").unwrap(),
                AttentionExecutionPolicy::Portable,
            )
            .unwrap(),
        )
        .unwrap()
        .with_program_binding_upload_transport(ProgramBindingUploadTransport::PinnedDirect);
        let stream = runtime.create_stream().unwrap();
        let base = stream.stream.clone_htod(&[CANARY; 96]).unwrap();
        let pointer = base.device_ptr(&stream.stream).0;
        stream.stream.synchronize().unwrap();
        let region = CudaBufferRegion {
            _allocation: Arc::new(CudaAllocation {
                _base: base,
                aligned_ptr: pointer,
                requested_bytes: 96,
            }),
            _core_retention: None,
            reusable_address_scope: None,
            runtime_instance: runtime.runtime_instance,
            device_ptr: pointer,
            length_bytes: 96,
            element_type: ElementType::U8,
        };
        Self {
            runtime,
            stream,
            region,
        }
    }

    fn executable(&self, value: u8) -> Arc<CudaCommandExecutable> {
        let mut contiguous = self.region.clone();
        contiguous.device_ptr += 8;
        contiguous.length_bytes = 5;
        let mut strided = self.region.clone();
        strided.device_ptr += 40;
        strided.length_bytes = 20;
        Arc::new(CudaCommandExecutable {
            regions: vec![contiguous, strided],
            host_storage: vec![
                vec![value; 5].into_boxed_slice(),
                (0..12)
                    .map(|i| value.wrapping_add(i))
                    .collect::<Vec<_>>()
                    .into_boxed_slice(),
            ],
            work_declaration: CudaWorkDeclaration::Static,
            enqueue: Mutex::new(CudaEnqueueAction::ProgramBindingPrelude(
                ProgramBindingPrelude {
                    shapes: vec![(5, 5, 1), (8, 4, 3)],
                    counters: self.runtime.program_binding_upload_counters.clone(),
                    plan: DeviceProgramBindingUploadSnapshot {
                        live_payload_bytes: 17,
                        planned_upload_bytes: 17,
                        logical_arena_bytes: 96,
                        physical_arena_bytes: 96,
                        ..Default::default()
                    },
                },
            )),
        })
    }

    fn command(&self, executable: Arc<CudaCommandExecutable>) -> CudaDeviceCommand {
        CudaDeviceCommand {
            runtime_instance: self.runtime.runtime_instance,
            operation: "test.pinned_binding_prelude",
            batching_form: DeviceBatchingForm::ParticipantLoop,
            participant_start: 0,
            participant_count: 1,
            token_count: 1,
            compute_dispatch_count: 0,
            transfer_command_count: 2,
            executable: Some(executable),
            fence_dependencies: Vec::new(),
            replay_key: None,
            reusable_address_scope: None,
            replay_gap_reason: None,
            program_binding_patch: None,
            reusable_execution: None,
            completion_checks: Vec::new(),
        }
    }

    fn prepare(
        &self,
        stream: &CudaDeviceStream,
        command: &CudaDeviceCommand,
    ) -> PinnedUploadSubmission {
        PinnedUploadSubmission::prepare(
            &self.runtime.context,
            &stream.pinned_upload_pool,
            stream.state.is_quiescent(),
            std::slice::from_ref(command),
        )
        .unwrap()
    }

    fn enqueue(
        &self,
        stream: &CudaDeviceStream,
        command: CudaDeviceCommand,
        mut pins: PinnedUploadSubmission,
    ) -> CudaDeviceFence {
        stream.state.begin_submission().unwrap();
        command
            .enqueue_with_pinned_sources(&stream.stream, &stream.blas, pins.sources_for(0))
            .unwrap();
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
            _commands: vec![command],
            pinned_uploads: Mutex::new(pins),
        }
    }

    fn assert_bytes(&self, value: u8, include_strided: bool) {
        let actual = self
            .stream
            .stream
            .clone_dtoh(&self.region._allocation._base)
            .unwrap();
        let mut expected = vec![CANARY; 96];
        expected[8..13].fill(value);
        if include_strided {
            for row in 0..3 {
                for col in 0..4 {
                    expected[40 + row * 8 + col] = value.wrapping_add((row * 4 + col) as u8);
                }
            }
        }
        assert_eq!(actual, expected);
    }
}

#[test]
#[ignore = "requires an actual CUDA device"]
fn pinned_binding_upload_fresh_payloads_and_submission_private_fences() {
    let fixture = Fixture::new();
    // Fresh source contents, both copy geometries, and every untouched byte.
    for value in [3, 71, 193] {
        let command = fixture.command(fixture.executable(value));
        let pins = fixture.prepare(&fixture.stream, &command);
        assert!(pins.lease.is_some());
        let fence = fixture.enqueue(&fixture.stream, command, pins);
        fixture.runtime.wait_fence(&fence).unwrap();
        fixture.assert_bytes(value, true);
        assert!(!fixture.stream.pinned_upload_pool.0.lock().unwrap().loaned);
    }
    let stats = fixture.runtime.program_binding_upload_counters.snapshot();
    assert_eq!(stats.pinned_upload_batches, 3);
    assert_eq!(stats.pinned_upload_reuse_batches, 2);
    assert_eq!(stats.pinned_upload_allocation_bytes, 32);
    assert_eq!(stats.pinned_upload_bytes, 51);
    assert_eq!(stats.successful_1d_copies, 3);
    assert_eq!(stats.successful_2d_copies, 3);

    // The same Arc executable on another stream gets a distinct submission
    // lease. Its terminal fence cannot return the first stream's live pins.
    let executable = fixture.executable(29);
    let first = fixture.command(executable.clone());
    let first_pins = fixture.prepare(&fixture.stream, &first);
    let first_base = first_pins.lease.as_ref().unwrap().base;
    let first_fence = fixture.enqueue(&fixture.stream, first, first_pins);
    // Avoid a destination write race; completion alone does not publish pins.
    first_fence.event.synchronize().unwrap();
    let other = fixture.runtime.create_stream().unwrap();
    let second = fixture.command(executable.clone());
    let second_pins = fixture.prepare(&other, &second);
    assert_ne!(first_base, second_pins.lease.as_ref().unwrap().base);
    let other_fence = fixture.enqueue(&other, second, second_pins);
    fixture.runtime.wait_fence(&other_fence).unwrap();
    assert!(fixture.stream.pinned_upload_pool.0.lock().unwrap().loaned);

    // Same-stream in-flight fallback also cannot release the previous owner.
    let fallback = fixture.command(executable);
    let fallback_pins = fixture.prepare(&fixture.stream, &fallback);
    assert!(fallback_pins.lease.is_none());
    let fallback_fence = fixture.enqueue(&fixture.stream, fallback, fallback_pins);
    fixture.runtime.wait_fence(&fallback_fence).unwrap();
    assert!(fixture.stream.pinned_upload_pool.0.lock().unwrap().loaned);
    fixture.runtime.wait_fence(&first_fence).unwrap();
    assert!(!fixture.stream.pinned_upload_pool.0.lock().unwrap().loaned);
    fixture.assert_bytes(29, true);
    assert_eq!(
        fixture
            .runtime
            .program_binding_upload_counters
            .snapshot()
            .pinned_upload_fallback_batches,
        1
    );
}

#[test]
#[ignore = "requires an actual CUDA device"]
fn pinned_binding_upload_partial_failure_quarantine_and_budget() {
    let mut fixture = Fixture::new();
    assert!(fixture
        .stream
        .pinned_upload_pool
        .borrow(&fixture.runtime.context, MAX_PINNED_UPLOAD_BYTES + 1)
        .unwrap()
        .is_none());
    assert!(!fixture.stream.pinned_upload_pool.0.lock().unwrap().loaned);
    let command = fixture.command(fixture.executable(43));
    let mut pins = fixture.prepare(&fixture.stream, &command);
    fixture.stream.state.begin_submission().unwrap();
    let mut sources = pins.sources_for(0).unwrap();
    // Fail after one actual accepted copy, before the next copy. This is an
    // internal contract fault injection, not a simulated CUDA success/error.
    sources.offsets = &sources.offsets[..1];
    assert!(command
        .enqueue_with_pinned_sources(&fixture.stream.stream, &fixture.stream.blas, Some(sources))
        .is_err());
    fixture.stream.state.fail();
    fixture
        .runtime
        .quarantine(&fixture.stream, vec![command], pins);
    assert!(fixture.stream.pinned_upload_pool.0.lock().unwrap().loaned);
    assert!(fixture
        .stream
        .pinned_upload_pool
        .borrow(&fixture.runtime.context, 17)
        .unwrap()
        .is_none());
    fixture.runtime.synchronize(&mut fixture.stream).unwrap();
    fixture.assert_bytes(43, false);
    let pool = fixture.stream.pinned_upload_pool.0.lock().unwrap();
    assert!(!pool.loaned);
    assert!(pool.idle.is_none());
    drop(pool);
    assert!(!fixture.stream.state.is_quiescent());
    let stats = fixture.runtime.program_binding_upload_counters.snapshot();
    assert_eq!(stats.failed_preludes, 1);
    assert_eq!(stats.successful_upload_bytes, 5);
    assert_eq!(stats.successful_1d_copies, 1);
    assert_eq!(stats.successful_2d_copies, 0);

    // Dropping a fence is not a completion proof, even if the device has in
    // fact finished. Deliberately quarantine this one 32-byte allocation.
    let mut abandoned = Fixture::new();
    let command = abandoned.command(abandoned.executable(9));
    let pins = abandoned.prepare(&abandoned.stream, &command);
    let fence = abandoned.enqueue(&abandoned.stream, command, pins);
    drop(fence);
    abandoned
        .runtime
        .synchronize(&mut abandoned.stream)
        .unwrap();
    abandoned.assert_bytes(9, true);
    assert!(abandoned.stream.pinned_upload_pool.0.lock().unwrap().loaned);
    assert!(abandoned
        .stream
        .pinned_upload_pool
        .borrow(&abandoned.runtime.context, 17)
        .unwrap()
        .is_none());
}
