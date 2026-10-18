//! A test backend for actual Core resident submission, not GPU computation.
//! Physical tables retain original allocations; current-use authority is never
//! cached in the program and its linear dependencies are consumed on each call.
use super::*;
use std::any::Any;

struct PhysicalTable {
    _retentions: Vec<DeviceBufferRetention>,
    _keeper: Arc<dyn Send + Sync>,
}

struct ResidentProgram {
    nodes: usize,
    participants: usize,
    _tables: Vec<Arc<SequencePhysicalBindingTable<TestBuffer>>>,
}

pub(super) fn compile_table(
    runtime: &TestRuntime,
    table: &SequencePhysicalBindingTable<TestBuffer>,
) -> Result<Option<Arc<dyn Any + Send + Sync>>, TestRuntimeError> {
    if !runtime.trace.lock().unwrap().resident_enabled {
        return Ok(None);
    }
    Ok(Some(Arc::new(PhysicalTable {
        _retentions: table
            .rows()
            .flat_map(|row| row.physical_regions())
            .map(|region| region.borrowed_buffer_and_physical_range().2.clone())
            .collect(),
        _keeper: Arc::clone(table.lifetime_keeper()),
    })))
}

pub(super) fn compile_program(
    runtime: &TestRuntime,
    program: &(dyn Any + Send + Sync),
    wave: &PreparedPhysicalSegmentWave<'_, TestBuffer>,
) -> Result<Option<Arc<dyn Any + Send + Sync>>, TestRuntimeError> {
    if !runtime.trace.lock().unwrap().resident_enabled {
        return Ok(None);
    }
    let program = program
        .downcast_ref::<TestSegmentProgram>()
        .ok_or(TestRuntimeError("foreign resident cold program"))?;
    assert_eq!(program.0.len(), wave.node_count());
    let participants = wave.sequence_tables().len();
    assert!(participants > 0);
    for (ordinal, &(index, regions, dependencies)) in program.0.iter().enumerate() {
        let node = wave.node(ordinal).unwrap();
        assert_eq!(node.node_index(), index);
        assert_eq!(node.participant_count(), participants);
        assert_eq!(node.declaration().regions().len(), regions);
        assert_eq!(node.declaration().dependencies().len(), dependencies);
        for participant in 0..participants {
            for region in 0..regions {
                let region = node.region(participant, region).unwrap();
                let bytes: u64 = region
                    .physical_regions()
                    .map(|part| {
                        let (buffer, range, _) = part.borrowed_buffer_and_physical_range();
                        assert!(
                            range.start < range.end && range.end <= buffer.descriptor.size_bytes
                        );
                        range.end - range.start
                    })
                    .sum();
                assert_eq!(bytes, region.descriptor().size_bytes);
            }
        }
    }
    runtime.trace.lock().unwrap().resident_compile_calls += 1;
    Ok(Some(Arc::new(ResidentProgram {
        nodes: wave.node_count(),
        participants,
        _tables: wave.sequence_tables().to_vec(),
    })))
}

pub(super) fn encode_program(
    runtime: &TestRuntime,
    program: &(dyn Any + Send + Sync),
    request: &mut DeviceResidentDecodeRequest<'_>,
) -> Result<Option<DeviceEncodedResidentDecode<TestCommand>>, TestRuntimeError> {
    let program = program
        .downcast_ref::<ResidentProgram>()
        .ok_or(TestRuntimeError("foreign resident hot program"))?;
    assert!(program.nodes > 0);
    assert_eq!(
        request.work_shape().participant_token_ranges().len(),
        program.participants
    );
    let ranges = request
        .work_shape()
        .participant_token_ranges()
        .iter()
        .map(|row| row.source_token_range())
        .collect();
    let dependencies = request
        .take_dependencies()
        .expect("fresh resident dependency authority");
    assert!(request.take_dependencies().is_none());
    let mut trace = runtime.trace.lock().unwrap();
    trace.resident_encode_calls += 1;
    trace.resident_source_ranges.push(ranges);
    Ok(Some(DeviceEncodedResidentDecode::new(
        dependencies.encode(None),
        TestCommand::CoalescedProgramBinding,
    )))
}
