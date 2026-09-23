use super::super::{coalesce_program_binding_transfers, CudaProgramBindingWrite};
use super::*;

fn binding(participants: u32, tokens: u64) -> OperationCostCommand {
    OperationCostCommand::new(
        "fixture.binding",
        DeviceCommandPhase::DynamicBinding,
        DeviceBatchingForm::ParticipantLoop,
        0,
        participants,
        tokens,
        0,
        1,
    )
    .unwrap()
}

fn spans(values: &[(u64, u64)]) -> Vec<ProgramBindingCostWrite> {
    values
        .iter()
        .map(|&(offset, length)| ProgramBindingCostWrite::new(offset, length).unwrap())
        .collect()
}

#[test]
fn compiled_slot_offsets_project_the_same_contiguous_and_strided_encoder_transfers() {
    let command = binding(3, 3);
    let first = spans(&[(0, 2), (2, 2), (16, 4), (32, 4)]);
    let second = spans(&[(0, 8), (32, 8)]);
    let patches = [
        ProgramBindingCostPatch {
            node_index: 1,
            command: &command,
            writes: &first,
        },
        ProgramBindingCostPatch {
            node_index: 4,
            command: &command,
            writes: &second,
        },
    ];
    let projected = project_binding_slots(
        128,
        [(1, 8, 40), (4, 64, 48)].into_iter(),
        &patches,
        &mut || Ok(()),
    )
    .unwrap();
    let mut actual_writes = Vec::new();
    for (slot, patch) in [8, 64].into_iter().zip(&patches) {
        for write in patch.writes {
            actual_writes.push(
                CudaProgramBindingWrite::new(
                    slot + write.offset_bytes(),
                    vec![(slot + write.offset_bytes()) as u8; write.length_bytes() as usize]
                        .into_boxed_slice(),
                )
                .unwrap(),
            );
        }
    }
    actual_writes.reverse(); // actual coalescer, unlike pure helper, sorts its input
    let actual = coalesce_program_binding_transfers(actual_writes, 128).unwrap();
    assert_eq!(projected.transfer_command_count(), actual.len() as u64);
    assert_eq!(actual.len(), 2);
    assert_eq!(
        (
            actual[0].destination_offset_bytes,
            actual[0].destination_stride_bytes,
            actual[0].row_bytes,
            actual[0].row_count
        ),
        (8, 16, 4, 3)
    );
    assert_eq!(
        actual[0].payload.as_ref(),
        &[8, 8, 10, 10, 24, 24, 24, 24, 40, 40, 40, 40]
    );
    assert_eq!(
        (
            actual[1].destination_offset_bytes,
            actual[1].destination_stride_bytes,
            actual[1].row_bytes,
            actual[1].row_count
        ),
        (64, 32, 8, 2)
    );
    assert_eq!(
        projected.native_operation(),
        PROGRAM_BINDING_NATIVE_OPERATION
    );
    assert_eq!(projected.batching(), DeviceBatchingForm::ParticipantLoop);
    assert_eq!(
        (projected.participant_count(), projected.token_count()),
        (3, 3)
    );
    assert_eq!(projected.compute_dispatch_count(), 0);
}

#[test]
fn incomplete_layout_overlap_slot_escape_and_incompatible_work_are_not_predictions() {
    let command = binding(2, 2);
    let other = binding(2, 3);
    let valid = spans(&[(0, 4)]);
    let overlap = spans(&[(0, 4), (2, 4)]);
    let escape = spans(&[(7, 2)]);
    for writes in [&[][..], overlap.as_slice(), escape.as_slice()] {
        let patches = [ProgramBindingCostPatch {
            node_index: 1,
            command: &command,
            writes,
        }];
        assert!(
            project_binding_slots(32, [(1, 8, 8)].into_iter(), &patches, &mut || Ok(())).is_err()
        );
    }
    let first = ProgramBindingCostPatch {
        node_index: 1,
        command: &command,
        writes: &valid,
    };
    assert!(project_binding_slots(32, [(2, 8, 8)].into_iter(), &[first], &mut || Ok(())).is_err());
    let patches = [
        ProgramBindingCostPatch {
            node_index: 1,
            command: &command,
            writes: &valid,
        },
        ProgramBindingCostPatch {
            node_index: 4,
            command: &other,
            writes: &valid,
        },
    ];
    assert!(project_binding_slots(32, [(1, 8, 8)].into_iter(), &patches, &mut || Ok(())).is_err());
    assert!(project_binding_slots(
        32,
        [(1, 8, 8), (4, 24, 8)].into_iter(),
        &patches,
        &mut || Ok(())
    )
    .is_err());
}

#[test]
fn binding_projection_honors_budget_and_never_requires_a_cuda_context() {
    let command = binding(1, 1);
    let writes = spans(&[(0, 4), (8, 4)]);
    let patches = [ProgramBindingCostPatch {
        node_index: 0,
        command: &command,
        writes: &writes,
    }];
    let mut polls = 0;
    let result = project_binding_slots(16, [(0, 0, 16)].into_iter(), &patches, &mut || {
        polls += 1;
        if polls == 3 {
            Err(VNextError::InvalidExecutionPlan {
                reason: "expired query".to_owned(),
            })
        } else {
            Ok(())
        }
    });
    assert!(result.is_err());
    assert_eq!(polls, 3);
}
