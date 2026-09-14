use super::*;
use crate::vnext::DeviceCommandPhase;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

#[derive(Clone)]
struct CpuBuffer(Arc<Mutex<Vec<u8>>>);

impl CpuBuffer {
    fn new(bytes: impl Into<Vec<u8>>) -> Self {
        Self(Arc::new(Mutex::new(bytes.into())))
    }

    fn bytes(&self) -> Vec<u8> {
        self.0.lock().unwrap().clone()
    }
}

enum CpuCommand {
    Copy {
        source: CpuBuffer,
        destination: CpuBuffer,
        region: CopyRegion,
    },
    Zero(CpuBuffer),
}

impl CpuCommand {
    fn execute(self) {
        match self {
            Self::Copy {
                source,
                destination,
                region,
            } => {
                let source = source.bytes();
                let start = region.source_offset_bytes() as usize;
                let end = start + region.length_bytes() as usize;
                let destination_start = region.destination_offset_bytes() as usize;
                destination.0.lock().unwrap()[destination_start..destination_start + end - start]
                    .copy_from_slice(&source[start..end]);
            }
            Self::Zero(buffer) => buffer.0.lock().unwrap().fill(0),
        }
    }
}

fn encode(
    source: &CpuBuffer,
    destination: &CpuBuffer,
    region: CopyRegion,
) -> Result<CpuCommand, &'static str> {
    Ok(CpuCommand::Copy {
        source: source.clone(),
        destination: destination.clone(),
        region,
    })
}

struct Fixture {
    resource: ResourceId,
    sequence: [CpuBuffer; 2],
    compact: [CpuBuffer; 2],
}

impl Fixture {
    fn new() -> Self {
        Self {
            resource: ResourceId::new("state/bytes").unwrap(),
            sequence: [
                CpuBuffer::new([99, 99, 1, 2, 3, 99]),
                CpuBuffer::new([99, 4, 5, 6, 7, 8, 99]),
            ],
            compact: [CpuBuffer::new([77; 7]), CpuBuffer::new([77; 8])],
        }
    }

    // Physical pairings after the independently tested copies_to planner:
    // sequence segments split 3/5 bytes, compact segments split 5/3 bytes.
    // These tests exercise direction, owned encoding and phase ordering; the
    // resource-backed integration fixture covers logical-view construction.
    fn copies(&self, kind: StateTransferKind) -> Vec<PlannedStateTransferCopy<'_, CpuBuffer>> {
        [(0, 0, 2, 1, 3), (1, 0, 1, 4, 2), (1, 1, 3, 3, 3)]
            .into_iter()
            .map(|(sequence, compact, source, destination, length)| {
                PlannedStateTransferCopy::from_sequence_to_checkpoint(
                    kind,
                    &self.resource,
                    &self.sequence[sequence],
                    &self.compact[compact],
                    CopyRegion::new(source, destination, length).unwrap(),
                )
                .unwrap()
            })
            .collect()
    }

    fn encoded(&self, kind: StateTransferKind) -> Vec<CpuCommand> {
        encode_planned_copies(&self.copies(kind), encode).unwrap()
    }
}

fn execute(batch: DeviceCommandBatch<CpuCommand>, expected_phase: DeviceCommandPhase) {
    for entry in batch.into_entries() {
        assert_eq!(entry.phase(), expected_phase);
        let (_, node, work, command) = entry.into_parts();
        assert!(node.is_none());
        assert!(work.is_none());
        command.execute();
    }
}

#[test]
fn capture_and_restore_preserve_distinct_physical_origins_and_capacity_sentinels() {
    let mut fixture = Fixture::new();
    let mut capture = DeviceCommandBatch::with_capacity(3);
    append_copy_commands(
        StateTransferKind::Capture,
        fixture.encoded(StateTransferKind::Capture),
        &mut capture,
    );
    // Encoding and appending cannot eagerly change device state.
    assert_eq!(fixture.compact[0].bytes(), [77; 7]);
    execute(capture, DeviceCommandPhase::ResultBinding);
    assert_eq!(fixture.compact[0].bytes(), [77, 1, 2, 3, 4, 5, 77]);
    assert_eq!(fixture.compact[1].bytes(), [77, 77, 77, 6, 7, 8, 77, 77]);
    assert_eq!(fixture.sequence[0].bytes(), [99, 99, 1, 2, 3, 99]);
    assert_eq!(fixture.sequence[1].bytes(), [99, 4, 5, 6, 7, 8, 99]);

    fixture.sequence = [CpuBuffer::new([55; 6]), CpuBuffer::new([55; 7])];
    let mut restore = DeviceCommandBatch::with_capacity(3);
    append_copy_commands(
        StateTransferKind::Restore,
        fixture.encoded(StateTransferKind::Restore),
        &mut restore,
    );
    execute(restore, DeviceCommandPhase::DynamicBinding);
    assert_eq!(fixture.sequence[0].bytes(), [55, 55, 1, 2, 3, 55]);
    assert_eq!(fixture.sequence[1].bytes(), [55, 4, 5, 6, 7, 8, 55]);
    assert_eq!(fixture.compact[0].bytes(), [77, 1, 2, 3, 4, 5, 77]);
    assert_eq!(fixture.compact[1].bytes(), [77, 77, 77, 6, 7, 8, 77, 77]);
}

#[test]
fn restore_copies_follow_existing_initialization_without_creating_model_work() {
    let mut fixture = Fixture::new();
    for command in fixture.encoded(StateTransferKind::Capture) {
        command.execute();
    }
    fixture.sequence = [CpuBuffer::new([55; 6]), CpuBuffer::new([55; 7])];
    let mut restore = DeviceCommandBatch::with_capacity(5);
    for buffer in &fixture.sequence {
        restore.push_initialization(CpuCommand::Zero(buffer.clone()));
    }
    append_copy_commands(
        StateTransferKind::Restore,
        fixture.encoded(StateTransferKind::Restore),
        &mut restore,
    );
    let mut reached_copies = false;
    for entry in restore.into_entries() {
        match entry.phase() {
            DeviceCommandPhase::Initialization => assert!(!reached_copies),
            DeviceCommandPhase::DynamicBinding => reached_copies = true,
            other => panic!("unexpected native restore phase: {other:?}"),
        }
        let (_, node, work, command) = entry.into_parts();
        assert!(node.is_none());
        assert!(work.is_none());
        command.execute();
    }
    assert!(reached_copies);
    assert_eq!(fixture.sequence[0].bytes(), [0, 0, 1, 2, 3, 0]);
    assert_eq!(fixture.sequence[1].bytes(), [0, 4, 5, 6, 7, 8, 0]);
}

#[test]
fn encode_failure_discards_prior_commands_and_preserves_device_error_and_range() {
    struct TrackedCommand {
        _command: CpuCommand,
        drops: Arc<AtomicUsize>,
    }
    impl Drop for TrackedCommand {
        fn drop(&mut self) {
            self.drops.fetch_add(1, Ordering::SeqCst);
        }
    }

    let fixture = Fixture::new();
    let drops = Arc::new(AtomicUsize::new(0));
    let mut calls = 0;
    let result = encode_planned_copies(
        &fixture.copies(StateTransferKind::Restore),
        |source, destination, region| {
            calls += 1;
            if calls == 2 {
                return Err("injected copy encoder failure");
            }
            Ok(TrackedCommand {
                _command: encode(source, destination, region).unwrap(),
                drops: Arc::clone(&drops),
            })
        },
    );
    match result {
        Err(StateTransferCopyEncodeError::Runtime {
            resource_id,
            region,
            error,
        }) => {
            assert_eq!(resource_id, fixture.resource);
            assert_eq!(region.source_offset_bytes(), 4);
            assert_eq!(region.destination_offset_bytes(), 1);
            assert_eq!(region.length_bytes(), 2);
            assert_eq!(error, "injected copy encoder failure");
        }
        _ => panic!("the original device encode failure must be returned"),
    }
    assert_eq!(calls, 2);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
    assert_eq!(fixture.sequence[0].bytes(), [99, 99, 1, 2, 3, 99]);
    assert_eq!(fixture.compact[0].bytes(), [77; 7]);
}

#[test]
fn copy_retention_keeps_both_logical_owners_and_every_segment_until_released() {
    let sequence = Arc::new(());
    let checkpoint = Arc::new(());
    let first_segment = Arc::new(());
    let second_segment = Arc::new(());
    let probes = [
        Arc::downgrade(&sequence),
        Arc::downgrade(&checkpoint),
        Arc::downgrade(&first_segment),
        Arc::downgrade(&second_segment),
    ];
    let retentions = StateTransferCopyRetentions {
        _owners: vec![
            DeviceBufferRetention::pair(sequence, checkpoint),
            DeviceBufferRetention::plan(first_segment),
            DeviceBufferRetention::plan(second_segment),
        ],
    };
    assert!(probes.iter().all(|owner| owner.upgrade().is_some()));
    drop(retentions);
    assert!(probes.iter().all(|owner| owner.upgrade().is_none()));
}
