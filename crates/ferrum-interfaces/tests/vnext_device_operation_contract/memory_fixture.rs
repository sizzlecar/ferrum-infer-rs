//! Opt-in byte execution for the typed token fixtures. Existing contract-only
//! runtimes continue to use their original command and readback behavior.
use super::*;
use std::ops::Range;

#[derive(Clone)]
pub(crate) struct TestMemoryBuffer {
    bytes: Arc<Mutex<Vec<u8>>>,
    registry: Arc<Mutex<TestMemoryRegistry>>,
}

impl fmt::Debug for TestMemoryBuffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TestMemoryBuffer")
            .field("bytes", &self.bytes.lock().unwrap().len())
            .finish()
    }
}

impl TestMemoryBuffer {
    pub(crate) fn new(size: u64, registry: Arc<Mutex<TestMemoryRegistry>>) -> Self {
        Self {
            bytes: Arc::new(Mutex::new(vec![0; usize::try_from(size).unwrap()])),
            registry,
        }
    }

    fn register(&self, action: MemoryAction, compute: bool) -> TestCommand {
        let mut registry = self.registry.lock().unwrap();
        let id = registry.next;
        registry.next += 1;
        registry.pending.insert(id, action);
        TestCommand::Memory(id, compute)
    }
}

type Bytes = Arc<Mutex<Vec<u8>>>;

enum MemoryAction {
    Copy(Bytes, Range<usize>, Bytes, Range<usize>),
    Upload(Vec<u8>, Bytes, Range<usize>),
    Transform(Vec<(Bytes, usize, Bytes, usize, bool)>),
}

#[derive(Default)]
pub(crate) struct TestMemoryRegistry {
    next: u64,
    pending: BTreeMap<u64, MemoryAction>,
}

pub(super) fn encode_copy(
    source: &TestBuffer,
    destination: &TestBuffer,
    region: CopyRegion,
) -> TestCommand {
    let source = source.memory.as_ref().unwrap();
    let destination = destination.memory.as_ref().unwrap();
    assert!(Arc::ptr_eq(&source.registry, &destination.registry));
    source.register(
        MemoryAction::Copy(
            Arc::clone(&source.bytes),
            range(region.source_offset_bytes(), region.length_bytes()),
            Arc::clone(&destination.bytes),
            range(region.destination_offset_bytes(), region.length_bytes()),
        ),
        false,
    )
}

pub(super) fn encode_upload(source: &[u8], destination: &TestBuffer, offset: u64) -> TestCommand {
    let destination = destination.memory.as_ref().unwrap();
    destination.register(
        MemoryAction::Upload(
            source.to_vec(),
            Arc::clone(&destination.bytes),
            range(offset, source.len() as u64),
        ),
        false,
    )
}

pub(super) fn encode_token_transform(
    invocation: &BatchedOperationInvocation<'_, TestBuffer>,
) -> TestCommand {
    let mut transforms = Vec::new();
    let mut registry_owner = None;
    for (row, participant) in invocation.participants().iter().enumerate() {
        let mut regions = Vec::new();
        for role in [ResolvedValueRole::Input, ResolvedValueRole::Output] {
            let binding = participant
                .bindings()
                .iter()
                .find(|binding| binding.role() == role && binding.ordinal() == 0)
                .unwrap();
            let component = &binding.storage().components()[0];
            let view = participant
                .views()
                .iter()
                .find(|view| view.resource_id() == component.resource_id())
                .unwrap();
            // Product I/O has ActualSequences demand and already arrives as a
            // participant window. This fixture's fixed intermediate vector is
            // batch-wide: each participant owns one F32 element in that vector.
            let row_offset = if binding.value_id().as_str() == "value.intermediate" {
                assert_eq!(binding.tensor().element_type(), ElementType::F32);
                assert!(!view.uses_packed_batch_coordinates());
                assert!(invocation.participants().len() as u64 * 4 <= component.length_bytes());
                row as u64 * 4
            } else {
                0
            };
            let translated = view
                .translate(component.offset_bytes() + row_offset, 4)
                .unwrap();
            let mut pieces = translated.iter();
            let piece = pieces.next().unwrap();
            assert!(pieces.next().is_none());
            let (buffer, physical, _retention) = piece.buffer_and_physical_range();
            let memory = buffer.memory.as_ref().unwrap();
            registry_owner = Some(memory.clone());
            regions.push((
                Arc::clone(&memory.bytes),
                usize::try_from(physical.start).unwrap(),
                binding.tensor().element_type(),
            ));
        }
        let (destination, destination_offset, destination_type) = regions.pop().unwrap();
        let (source, source_offset, source_type) = regions.pop().unwrap();
        assert!(matches!(
            (source_type, destination_type),
            (ElementType::U32, ElementType::F32) | (ElementType::F32, ElementType::U32)
        ));
        transforms.push((
            source,
            source_offset,
            destination,
            destination_offset,
            source_type == ElementType::U32,
        ));
    }
    registry_owner
        .unwrap()
        .register(MemoryAction::Transform(transforms), true)
}

pub(super) fn execute(registry: &Arc<Mutex<TestMemoryRegistry>>, commands: &[TestCommand]) {
    for command in commands {
        let TestCommand::Memory(id, _) = command else {
            continue;
        };
        let action = registry.lock().unwrap().pending.remove(id).unwrap();
        match action {
            MemoryAction::Copy(source, source_range, destination, destination_range) => {
                let bytes = source.lock().unwrap()[source_range].to_vec();
                destination.lock().unwrap()[destination_range].copy_from_slice(&bytes);
            }
            MemoryAction::Upload(bytes, destination, destination_range) => {
                destination.lock().unwrap()[destination_range].copy_from_slice(&bytes);
            }
            MemoryAction::Transform(rows) => {
                for (source, source_offset, destination, destination_offset, to_float) in rows {
                    let bytes: [u8; 4] = source.lock().unwrap()[source_offset..source_offset + 4]
                        .try_into()
                        .unwrap();
                    let output = if to_float {
                        (u32::from_le_bytes(bytes) as f32).to_le_bytes()
                    } else {
                        (f32::from_le_bytes(bytes) as u32 + 1).to_le_bytes()
                    };
                    destination.lock().unwrap()[destination_offset..destination_offset + 4]
                        .copy_from_slice(&output);
                }
            }
        }
    }
}

pub(super) fn readback(
    source: &TestMemoryBuffer,
    region: CopyRegion,
    layout: HostTransferLayout,
) -> Vec<u8> {
    let bytes = source.bytes.lock().unwrap();
    let mut output = vec![0; usize::try_from(layout.byte_len().unwrap()).unwrap()];
    output[range(region.destination_offset_bytes(), region.length_bytes())]
        .copy_from_slice(&bytes[range(region.source_offset_bytes(), region.length_bytes())]);
    output
}

fn range(offset: u64, length: u64) -> Range<usize> {
    usize::try_from(offset).unwrap()..usize::try_from(offset.checked_add(length).unwrap()).unwrap()
}
