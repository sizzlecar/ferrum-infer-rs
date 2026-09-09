use std::collections::BTreeSet;
use std::sync::Barrier;

use ferrum_interfaces::vnext::{
    BufferUsage, DeviceBatchingForm, DeviceId, DynamicStorageAllocator, DynamicStorageProfile,
    DynamicStorageView, ElementType, ResourceId,
};

use super::*;

fn runtime(capacity: u64) -> CpuDeviceRuntime {
    CpuDeviceRuntime::new(DeviceDescriptor {
        id: DeviceId::new("device.cpu.runtime-test").unwrap(),
        class: DeviceClass::Host,
        ordinal: 0,
        total_memory_bytes: capacity,
        runtime_implementation_fingerprint: "1".repeat(64),
        capabilities: BTreeSet::new(),
        dynamic_storage_profiles: BTreeSet::from([DynamicStorageProfile::new(
            DynamicStorageAllocator::LinearArena,
            DynamicStorageView::Contiguous,
        )
        .unwrap()]),
    })
    .unwrap()
}

fn buffer(runtime: &CpuDeviceRuntime, bytes: &[u8]) -> CpuDeviceBuffer {
    let buffer = CpuDeviceBuffer::allocate(
        BufferDescriptor {
            resource_id: ResourceId::new("resource.cpu.runtime-test").unwrap(),
            size_bytes: bytes.len() as u64,
            alignment_bytes: 16,
            usage: BufferUsage::Transfer,
            element_type: ElementType::U8,
        },
        runtime.runtime_instance,
        &runtime.budget,
    )
    .unwrap();
    buffer
        .region(0..bytes.len() as u64)
        .unwrap()
        .write(bytes)
        .unwrap();
    buffer
}

fn entries(
    commands: Vec<CpuDeviceCommand>,
) -> Vec<(DeviceCommandPhase, Option<u32>, CpuDeviceCommand)> {
    commands
        .into_iter()
        .map(|command| (DeviceCommandPhase::Compute, None, command))
        .collect()
}

fn read(buffer: &CpuDeviceBuffer) -> Vec<u8> {
    buffer
        .region(0..buffer.descriptor.size_bytes)
        .unwrap()
        .with_read(<[u8]>::to_vec)
        .unwrap()
}

#[test]
fn allocations_are_aligned_and_capacity_is_released_with_last_region() {
    let budget = MemoryBudget::new(257);
    for alignment in [1, 2, 16, 64, 4096] {
        let storage = Storage::new(257, alignment, &budget).unwrap();
        assert_eq!(storage.bytes().as_ptr() as usize % alignment as usize, 0);
        assert_eq!(storage.bytes(), [0; 257]);
        assert!(Storage::new(1, 1, &budget).is_err());
        assert_eq!(budget.used(), 257);
        drop(storage);
        assert_eq!(budget.used(), 0);
    }
    let runtime = runtime(4);
    let buffer = buffer(&runtime, &[1, 2, 3, 4]);
    let retained = buffer.region(1..3).unwrap();
    drop(buffer);
    assert_eq!(runtime.resident_bytes(), 4);
    assert_eq!(retained.with_read(<[u8]>::to_vec).unwrap(), [2, 3]);
    drop(retained);
    assert_eq!(runtime.resident_bytes(), 0);
    assert_eq!(runtime.peak_resident_bytes(), 4);
}

#[test]
fn allocation_overflow_and_invalid_layout_do_not_consume_capacity() {
    let budget = MemoryBudget::new(u64::MAX);
    for (size, alignment) in [(0, 1), (1, 0), (1, 3), (u64::MAX, 4096)] {
        assert!(Storage::new(size, alignment, &budget).is_err());
        assert_eq!(budget.used(), 0);
    }
    let reservation = budget.reserve(u64::MAX).unwrap();
    assert!(budget.reserve(1).is_err());
    assert_eq!(budget.used(), u64::MAX);
    drop(reservation);
    assert_eq!(budget.used(), 0);
}

#[test]
fn concurrent_reservations_cannot_exceed_capacity() {
    let budget = MemoryBudget::new(128);
    let barrier = Arc::new(Barrier::new(8));
    let workers: Vec<_> = (0..8)
        .map(|_| {
            let budget = Arc::clone(&budget);
            let barrier = Arc::clone(&barrier);
            std::thread::spawn(move || {
                barrier.wait();
                let reservation = budget.reserve(64).ok();
                let acquired = reservation.is_some();
                barrier.wait();
                drop(reservation);
                acquired
            })
        })
        .collect();
    assert_eq!(
        workers
            .into_iter()
            .map(|worker| usize::from(worker.join().unwrap()))
            .sum::<usize>(),
        2
    );
    assert_eq!(budget.used(), 0);
    assert_eq!(budget.peak(), 128);
}

#[test]
fn upload_owns_source_bytes_and_is_charged_until_submission_finishes() {
    let runtime = runtime(8);
    let destination = buffer(&runtime, &[0; 4]);
    let mut bytes = [1, 2, 3, 4];
    let command = runtime
        .encode_upload(
            &bytes,
            HostTransferLayout::new(ElementType::U8, 4).unwrap(),
            &destination,
            0,
        )
        .unwrap();
    bytes.fill(9);
    assert_eq!(runtime.resident_bytes(), 8);
    assert!(runtime
        .encode_upload(
            &bytes[..1],
            HostTransferLayout::new(ElementType::U8, 1).unwrap(),
            &destination,
            0
        )
        .is_err());
    let mut stream = runtime.create_stream().unwrap();
    let fence = runtime
        .execute_entries(&mut stream, entries(vec![command]), true)
        .unwrap();
    assert!(runtime
        .wait_fence(&fence)
        .unwrap()
        .terminal()
        .is_succeeded());
    assert!(runtime.submission_attribution(&fence).is_some());
    assert_eq!(read(&destination), [1, 2, 3, 4]);
    assert_eq!(runtime.resident_bytes(), 4);
}

#[test]
fn copies_handle_overlap_and_different_allocations_without_extra_storage() {
    let runtime = runtime(12);
    let first = buffer(&runtime, &[1, 2, 3, 4, 5, 6]);
    let second = buffer(&runtime, &[0; 6]);
    let overlap = runtime
        .encode_copy(&first, &first, CopyRegion::new(0, 2, 4).unwrap())
        .unwrap();
    let outward = runtime
        .encode_copy(&first, &second, CopyRegion::new(0, 0, 6).unwrap())
        .unwrap();
    let inward = runtime
        .encode_copy(&second, &first, CopyRegion::new(4, 0, 2).unwrap())
        .unwrap();
    let mut stream = runtime.create_stream().unwrap();
    let fence = runtime
        .execute_entries(&mut stream, entries(vec![overlap, outward, inward]), false)
        .unwrap();
    assert!(runtime
        .wait_fence(&fence)
        .unwrap()
        .terminal()
        .is_succeeded());
    assert_eq!(read(&second), [1, 2, 1, 2, 3, 4]);
    assert_eq!(read(&first), [3, 4, 1, 2, 3, 4]);
    assert_eq!(runtime.peak_resident_bytes(), 12);
}

#[test]
fn readback_respects_host_offset_and_rejects_bounds_and_dtype() {
    let runtime = runtime(32);
    let source = buffer(&runtime, &[10, 11, 12, 13]);
    let mut stream = runtime.create_stream().unwrap();
    let layout = HostTransferLayout::new(ElementType::U8, 6).unwrap();
    assert_eq!(
        runtime
            .readback(
                &mut stream,
                &source,
                CopyRegion::new(1, 2, 2).unwrap(),
                layout
            )
            .unwrap(),
        [0, 0, 11, 12, 0, 0]
    );
    assert_eq!(runtime.resident_bytes(), 4);
    for region in [
        CopyRegion::new(3, 0, 2).unwrap(),
        CopyRegion::new(0, 5, 2).unwrap(),
    ] {
        assert!(runtime
            .readback(&mut stream, &source, region, layout)
            .is_err());
    }
    assert!(runtime
        .readback(
            &mut stream,
            &source,
            CopyRegion::new(0, 0, 2).unwrap(),
            HostTransferLayout::new(ElementType::F16, 1).unwrap()
        )
        .is_err());
    assert!(source.region(0..0).is_err());
    assert!(source.region(0..5).is_err());
    assert!(source.region(0..4).unwrap().write(&[0]).is_err());
}

struct WriteAndFail {
    region: CpuBufferRegion,
    panic: bool,
}

impl CpuKernelLaunch for WriteAndFail {
    fn validate_runtime(&self, instance: u64) -> Result<(), CpuRuntimeError> {
        self.region.validate_runtime(instance)
    }
    fn execute(&self) -> Result<(), CpuRuntimeError> {
        self.region.write(&[5])?;
        if self.panic {
            panic!("test CPU computation failed after a write");
        }
        Err(CpuRuntimeError::new("test CPU arithmetic failure"))
    }
}

#[test]
fn failure_after_write_returns_failed_quiescent_fence_and_prevents_retry() {
    for panic in [false, true] {
        let runtime = runtime(16);
        let buffer = buffer(&runtime, &[9]);
        let first = runtime.encode_zero(&buffer, 0, 1).unwrap();
        let failing = CpuDeviceCommand::compute(
            "cpu.test.failure",
            vec![Box::new(WriteAndFail {
                region: buffer.region(0..1).unwrap(),
                panic,
            })],
            DeviceBatchingForm::Scalar,
            1,
            1,
        )
        .unwrap();
        let last = runtime.encode_zero(&buffer, 0, 1).unwrap();
        let mut stream = runtime.create_stream().unwrap();
        let fence = runtime
            .execute_entries(&mut stream, entries(vec![first, failing, last]), true)
            .unwrap();
        assert!(matches!(
            runtime.wait_fence(&fence).unwrap().terminal(),
            DeviceTerminal::FailedButQuiescent(_)
        ));
        assert_eq!(read(&buffer), [5]);
        assert_eq!(runtime.stream_state(&stream), StreamState::Failed);
        assert!(runtime.submission_attribution(&fence).is_none());
        assert!(runtime.synchronize(&mut stream).is_err());
        let retry = runtime.encode_zero(&buffer, 0, 1).unwrap();
        assert!(runtime
            .execute_entries(&mut stream, entries(vec![retry]), false)
            .is_err());
        assert_eq!(read(&buffer), [5]);
        assert_eq!(runtime.resident_bytes(), 1);
    }
}

#[test]
fn foreign_command_is_rejected_before_any_earlier_command_runs() {
    let first = runtime(16);
    let second = runtime(16);
    let first_buffer = buffer(&first, &[9]);
    let second_buffer = buffer(&second, &[8]);
    let first_command = first.encode_zero(&first_buffer, 0, 1).unwrap();
    let second_command = second.encode_zero(&second_buffer, 0, 1).unwrap();
    let mut stream = first.create_stream().unwrap();
    assert!(first
        .execute_entries(
            &mut stream,
            entries(vec![first_command, second_command]),
            false
        )
        .is_err());
    assert_eq!(read(&first_buffer), [9]);
    assert_eq!(read(&second_buffer), [8]);
    assert_eq!(first.stream_state(&stream), StreamState::Ready);
    assert_eq!(second.stream_state(&stream), StreamState::Failed);
    assert!(second.synchronize(&mut stream).is_err());
    assert!(first
        .encode_copy(
            &first_buffer,
            &second_buffer,
            CopyRegion::new(0, 0, 1).unwrap()
        )
        .is_err());
    let command = first.encode_zero(&first_buffer, 0, 1).unwrap();
    let fence = first
        .execute_entries(&mut stream, entries(vec![command]), false)
        .unwrap();
    assert!(matches!(
        second.query_fence(&fence),
        FenceQuery::Indeterminate(_)
    ));
    assert!(second.wait_fence(&fence).is_err());
}

#[test]
fn cpu_requires_host_descriptor_and_rejects_unimplemented_submission_modes() {
    let runtime = runtime(16);
    for class in [DeviceClass::Reference, DeviceClass::Accelerator] {
        let mut descriptor = runtime.descriptor().clone();
        descriptor.class = class;
        assert!(CpuDeviceRuntime::new(descriptor).is_err());
    }
    assert!(validate_submission_requirements(
        DeviceTimingMode::Off,
        DeviceComputePathRequirement::EagerOnly,
        false
    )
    .is_ok());
    assert!(validate_submission_requirements(
        DeviceTimingMode::Off,
        DeviceComputePathRequirement::Adaptive,
        false
    )
    .is_ok());
    for path in [
        DeviceComputePathRequirement::ReplayedOnly,
        DeviceComputePathRequirement::ReplayedWithDeclaredEagerBoundaries,
    ] {
        assert!(validate_submission_requirements(DeviceTimingMode::Off, path, false).is_err());
    }
    assert!(validate_submission_requirements(
        DeviceTimingMode::Verification,
        DeviceComputePathRequirement::Adaptive,
        false
    )
    .is_err());
    assert!(validate_submission_requirements(
        DeviceTimingMode::Off,
        DeviceComputePathRequirement::Adaptive,
        true
    )
    .is_err());
}
