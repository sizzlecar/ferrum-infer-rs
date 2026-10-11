//! Real production scatter helpers and runtime quarantine, without fabricated
//! Core bindings. These tests do not establish public dispatch authorization.

use super::super::{CudaCommandExecutable, CudaDeviceRuntime};
use super::*;
use ferrum_interfaces::vnext::{DeviceId, DeviceRuntime};
use ferrum_types::AttentionExecutionPolicy;

const CANARY: u8 = 0xd3;
const PITCH: usize = 131_096;
const ARENA_BYTES: usize = PITCH + 128;

// Raw-address tests must remain safe if an assertion unwinds after enqueue.
// A failed drain preserves the complete held owner, including module/source.
struct DrainOnDrop<T> {
    held: Option<T>,
    stream: Arc<CudaStream>,
}

impl<T> DrainOnDrop<T> {
    fn new(held: T, stream: &Arc<CudaStream>) -> Self {
        Self {
            held: Some(held),
            stream: Arc::clone(stream),
        }
    }

    fn get(&self) -> &T {
        self.held.as_ref().unwrap()
    }

    fn take(&mut self) -> T {
        self.held.take().unwrap()
    }
}

impl<T> Drop for DrainOnDrop<T> {
    fn drop(&mut self) {
        if self.held.is_some() && self.stream.synchronize().is_err() {
            if let Some(held) = self.held.take() {
                std::mem::forget(held);
            }
        }
    }
}

// Once the command moves into runtime quarantine, also keep the test safe if
// an assertion unwinds before the explicit production synchronize below.
struct QuarantineCleanup {
    runtime: Arc<CudaDeviceRuntime>,
    stream: Arc<CudaStream>,
    stream_id: u64,
}

impl Drop for QuarantineCleanup {
    fn drop(&mut self) {
        if self.stream.synchronize().is_ok() {
            self.runtime.release_quarantine(self.stream_id);
        } else {
            // Preserve the runtime's actual quarantined commands and owners.
            std::mem::forget(Arc::clone(&self.runtime));
        }
    }
}

fn transfers(width: usize, seed: u8) -> Vec<CudaProgramBindingTransfer> {
    vec![
        CudaProgramBindingTransfer {
            destination_offset_bytes: 4,
            destination_stride_bytes: 0,
            row_bytes: 3,
            row_count: 1,
            payload: vec![seed, seed.wrapping_add(1), seed.wrapping_add(2)].into_boxed_slice(),
        },
        CudaProgramBindingTransfer {
            destination_offset_bytes: 16,
            destination_stride_bytes: PITCH as u64,
            row_bytes: width,
            row_count: 2,
            payload: (0..width * 2)
                .map(|index| seed.wrapping_add(index as u8))
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        },
    ]
}

fn apply(expected: &mut [u8], transfers: &[CudaProgramBindingTransfer]) {
    for transfer in transfers {
        for row in 0..transfer.row_count {
            let start = transfer.destination_offset_bytes as usize
                + row * transfer.destination_stride_bytes as usize;
            let source = row * transfer.row_bytes;
            expected[start..start + transfer.row_bytes]
                .copy_from_slice(&transfer.payload[source..source + transfer.row_bytes]);
        }
    }
}

fn small_budget(scatter: &mut BindingScatter, packet_bytes: usize, live_bytes: usize) {
    scatter.budget = Arc::new(ScratchBudget {
        limits: ScratchLimits {
            packet_bytes,
            live_bytes,
        },
        live: AtomicUsize::new(0),
    });
}

fn packet_bytes(transfers: &[CudaProgramBindingTransfer]) -> usize {
    Packet::prepare(transfers, ARENA_BYTES as u64, usize::MAX)
        .unwrap()
        .unwrap()
        .bytes
        .len()
}

fn canary_arena(stream: &Arc<CudaStream>) -> Arc<CudaSlice<u8>> {
    let arena = Arc::new(stream.alloc_zeros::<u8>(ARENA_BYTES).unwrap());
    let mut pending = DrainOnDrop::new(arena, stream);
    unsafe {
        cudarc::driver::result::memset_d8_async(
            pending.get().device_ptr(stream).0,
            CANARY,
            ARENA_BYTES,
            stream.cu_stream(),
        )
    }
    .unwrap();
    stream.synchronize().unwrap();
    pending.take()
}

#[test]
#[ignore = "real CUDA compact binding owners, sparse bytes and bounded scratch; exclusive GPU"]
fn binding_scatter_fresh_owners_fences_and_capacity() {
    let context = CudaContext::new(0).unwrap();
    unsafe { context.disable_event_tracking() };
    let allocation = context.new_stream().unwrap();
    let first_stream = context.new_stream().unwrap();
    let second_stream = context.new_stream().unwrap();
    let first_arena = canary_arena(&first_stream);
    let second_arena = canary_arena(&second_stream);
    let first_input = transfers(48, 17);
    let second_input = transfers(48, 111);
    let bytes = packet_bytes(&first_input);
    let mut scatter = BindingScatter::load(&context).unwrap();
    small_budget(&mut scatter, bytes, 2 * bytes);

    let first = scatter
        .prepare(&allocation, &first_input, ARENA_BYTES as u64)
        .unwrap()
        .unwrap();
    let first = DrainOnDrop::new((first, Arc::clone(&first_arena)), &first_stream);
    let second = scatter
        .prepare(&allocation, &second_input, ARENA_BYTES as u64)
        .unwrap()
        .unwrap();
    let second = DrainOnDrop::new((second, Arc::clone(&second_arena)), &second_stream);
    assert_eq!(scatter.budget.live.load(Ordering::Acquire), 2 * bytes);
    assert_ne!(
        first.get().0.owner.device.device_ptr(&first_stream).0,
        second.get().0.owner.device.device_ptr(&second_stream).0
    );
    assert!(scatter
        .prepare(&allocation, &first_input, ARENA_BYTES as u64)
        .unwrap()
        .is_none());
    assert!(scatter
        .prepare(&allocation, &transfers(49, 31), ARENA_BYTES as u64)
        .unwrap()
        .is_none());
    assert_eq!(
        first_stream.clone_dtoh(first_arena.as_ref()).unwrap(),
        vec![CANARY; ARENA_BYTES]
    );
    assert_eq!(
        second_stream.clone_dtoh(second_arena.as_ref()).unwrap(),
        vec![CANARY; ARENA_BYTES]
    );
    // Both fresh packets are submitted before either fence is observed. They
    // own different scratch and cannot overwrite each other's descriptors.
    for (pending, stream) in [(&first, &first_stream), (&second, &second_stream)] {
        let (prepared, arena) = pending.get();
        prepared.owner.upload(stream, &prepared.bytes).unwrap();
        prepared
            .owner
            .scatter(stream, arena.device_ptr(stream).0)
            .unwrap();
    }
    let first_fence = first_stream.record_event(None).unwrap();
    let second_fence = second_stream.record_event(None).unwrap();
    // Reject a second execution of the same owner before any replacement H2D.
    let replacement = vec![0; first.get().0.bytes.len()];
    assert!(first
        .get()
        .0
        .owner
        .upload(&first_stream, &replacement)
        .is_err());
    first_fence.synchronize().unwrap();
    second_fence.synchronize().unwrap();
    let mut expected_first = vec![CANARY; ARENA_BYTES];
    let mut expected_second = vec![CANARY; ARENA_BYTES];
    apply(&mut expected_first, &first_input);
    apply(&mut expected_second, &second_input);
    assert_eq!(
        first_stream.clone_dtoh(first_arena.as_ref()).unwrap(),
        expected_first
    );
    assert_eq!(
        second_stream.clone_dtoh(second_arena.as_ref()).unwrap(),
        expected_second
    );
    // A completed owner's retained lifetime still holds its reservation.
    assert_eq!(scatter.budget.live.load(Ordering::Acquire), 2 * bytes);
    drop(first);
    assert_eq!(scatter.budget.live.load(Ordering::Acquire), bytes);
    let unsubmitted = scatter
        .prepare(&allocation, &first_input, ARENA_BYTES as u64)
        .unwrap()
        .unwrap();
    assert_eq!(scatter.budget.live.load(Ordering::Acquire), 2 * bytes);
    drop(unsubmitted);
    assert_eq!(scatter.budget.live.load(Ordering::Acquire), bytes);

    let smaller_input = transfers(24, 203);
    let smaller = scatter
        .prepare(&allocation, &smaller_input, ARENA_BYTES as u64)
        .unwrap()
        .unwrap();
    let smaller = DrainOnDrop::new((smaller, Arc::clone(&first_arena)), &first_stream);
    smaller
        .get()
        .0
        .owner
        .upload(&first_stream, &smaller.get().0.bytes)
        .unwrap();
    smaller
        .get()
        .0
        .owner
        .scatter(&first_stream, first_arena.device_ptr(&first_stream).0)
        .unwrap();
    first_stream
        .record_event(None)
        .unwrap()
        .synchronize()
        .unwrap();
    apply(&mut expected_first, &smaller_input);
    assert_eq!(
        first_stream.clone_dtoh(first_arena.as_ref()).unwrap(),
        expected_first
    );
    drop((smaller, second));
    allocation.synchronize().unwrap();
    assert_eq!(scatter.budget.live.load(Ordering::Acquire), 0);
    println!("compact binding owners: fresh packets, full sparse bytes/holes, unchanged old tail, single use, bounded allocation and terminal release");
}

#[test]
#[ignore = "real H2D partial failure retains compact scratch in runtime quarantine until drain; exclusive GPU"]
fn binding_scatter_partial_upload_quarantine_retains_budget() {
    let runtime = Arc::new(
        CudaDeviceRuntime::new(
            crate::backend::cuda::vnext_ops::cuda_vnext_runtime_config(
                0,
                DeviceId::new("device.test.compact-binding-quarantine").unwrap(),
                AttentionExecutionPolicy::Portable,
            )
            .unwrap(),
        )
        .unwrap(),
    );
    let mut stream = runtime.create_stream().unwrap();
    let _cleanup = QuarantineCleanup {
        runtime: Arc::clone(&runtime),
        stream: Arc::clone(&stream.stream),
        stream_id: stream.id,
    };
    let arena = canary_arena(&stream.stream);
    let input = transfers(48, 73);
    let bytes = packet_bytes(&input);
    let mut scatter = BindingScatter::load(&runtime.context).unwrap();
    small_budget(&mut scatter, bytes, bytes);
    let PreparedScatter {
        bytes: payload,
        owner,
    } = scatter
        .prepare(&runtime.allocation_stream, &input, ARENA_BYTES as u64)
        .unwrap()
        .unwrap();
    let retained_arena = Arc::clone(&arena);
    let mut command = super::super::tests::command("test.compact_binding_partial_upload");
    command.runtime_instance = runtime.runtime_instance;
    command.executable = Some(Arc::new(CudaCommandExecutable::static_work(
        Vec::new(),
        vec![payload],
        Box::new(move |stream, _, _, storage| {
            let _retained_arena = &retained_arena;
            owner.upload(stream, &storage[0])?;
            Err(CudaDeviceRuntimeError::contract(
                "injected host error after H2D before scatter",
            ))
        }),
    )));
    let shared_executable = Arc::clone(command.executable.as_ref().unwrap());
    let mut pending = DrainOnDrop::new(command, &stream.stream);
    stream.state.begin_submission().unwrap();
    let result = pending.get().enqueue(&stream.stream, &stream.blas);
    // Exactly the production partial-submit retention sequence. This exercises
    // real runtime quarantine/drain, not DeviceCommandBatch/Core admission.
    stream.state.fail();
    runtime.quarantine(&stream, vec![pending.take()]);
    assert!(result.is_err());
    drop((pending, shared_executable));
    assert_eq!(runtime.quarantined().len(), 1);
    assert_eq!(scatter.budget.live.load(Ordering::Acquire), bytes);
    assert!(scatter
        .prepare(&runtime.allocation_stream, &input, ARENA_BYTES as u64)
        .unwrap()
        .is_none());
    runtime.synchronize(&mut stream).unwrap();
    assert!(runtime.quarantined().is_empty());
    assert_eq!(scatter.budget.live.load(Ordering::Acquire), 0);
    assert_eq!(
        stream.stream.clone_dtoh(arena.as_ref()).unwrap(),
        vec![CANARY; ARENA_BYTES]
    );
    // Draining releases ownership but does not turn a failed execution lane
    // healthy. A future fresh allocation can nevertheless recover its budget.
    let next = scatter
        .prepare(&runtime.allocation_stream, &input, ARENA_BYTES as u64)
        .unwrap()
        .unwrap();
    assert_eq!(scatter.budget.live.load(Ordering::Acquire), bytes);
    drop(next);
    runtime.allocation_stream.synchronize().unwrap();
    assert_eq!(scatter.budget.live.load(Ordering::Acquire), 0);
    println!("compact binding partial H2D: host-injected error, shared executable handle lifetime, quarantine holds budget until successful drain; no real driver fault injected");
}
