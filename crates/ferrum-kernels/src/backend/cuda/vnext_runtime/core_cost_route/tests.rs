use super::super::{coalesce_program_binding_transfers, CudaProgramBindingWrite};
use super::*;

#[test]
fn indexed_binding_groups_actual_payload_and_future_recipe_use_the_same_physical_plan() {
    let command = binding(2, 2);
    let first = spans(&[(0, 2), (2, 2), (16, 8)]);
    let second = spans(&[(0, 1), (1, 3), (16, 8)]);
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
        96,
        [(1, 0, 32), (4, 64, 32)].into_iter(),
        &patches,
        &mut || Ok(()),
        SloStructuredCostCapture::HostSettledV1,
    )
    .unwrap();
    let mut actual_writes = Vec::new();
    for (slot, patch) in [0, 64].into_iter().zip(&patches) {
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
    actual_writes.reverse();
    let actual = coalesce_program_binding_transfers(actual_writes, 96).unwrap();
    assert_eq!(actual.len(), 2);
    assert_eq!(actual[0].payload.as_ref(), &[0, 0, 2, 2, 64, 65, 65, 65]);
    assert_eq!(
        actual[1].payload.as_ref(),
        &[16; 8].into_iter().chain([80; 8]).collect::<Vec<_>>()
    );
    assert!(actual
        .iter()
        .all(|t| t.row_count == 2 && t.destination_stride_bytes == 64));
    let observed = super::super::selected_cost::program_binding(
        actual.iter().map(|t| {
            (
                t.destination_stride_bytes,
                t.row_bytes as u64,
                t.row_count as u64,
            )
        }),
        2,
        SloStructuredCostCapture::HostSettledV1,
    )
    .unwrap();
    let expected = projected.statistical_evidence().unwrap();
    assert_eq!(projected.transfer_command_count(), 2);
    assert_eq!(observed.family_signature(), expected.family_signature());
    assert_eq!(observed.work(), expected.work());
    assert_eq!(observed.work().host_to_device_bytes, 24);
    assert_eq!(
        observed.algorithm_work().unwrap().unwrap(),
        expected.algorithm_work().unwrap().unwrap()
    );
}

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
        SloStructuredCostCapture::HostSettledV1,
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
    let observed = super::super::selected_cost::program_binding(
        actual.iter().map(|t| {
            (
                t.destination_stride_bytes,
                t.row_bytes as u64,
                t.row_count as u64,
            )
        }),
        command.token_count(),
        SloStructuredCostCapture::HostSettledV1,
    )
    .unwrap();
    let predicted = projected.statistical_evidence().unwrap();
    predicted.validate_command(3, 0, 2).unwrap();
    assert_eq!(observed.family_signature(), predicted.family_signature());
    assert_eq!(observed.work(), predicted.work());
    assert_eq!(observed.work().host_to_device_bytes, 28);
    predicted
        .algorithm_work()
        .unwrap()
        .unwrap()
        .validate_command(predicted)
        .unwrap();
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
        assert!(project_binding_slots(
            32,
            [(1, 8, 8)].into_iter(),
            &patches,
            &mut || Ok(()),
            SloStructuredCostCapture::Disabled
        )
        .is_err());
    }
    let first = ProgramBindingCostPatch {
        node_index: 1,
        command: &command,
        writes: &valid,
    };
    assert!(project_binding_slots(
        32,
        [(2, 8, 8)].into_iter(),
        &[first],
        &mut || Ok(()),
        SloStructuredCostCapture::Disabled
    )
    .is_err());
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
    assert!(project_binding_slots(
        32,
        [(1, 8, 8)].into_iter(),
        &patches,
        &mut || Ok(()),
        SloStructuredCostCapture::Disabled
    )
    .is_err());
    assert!(project_binding_slots(
        32,
        [(1, 8, 8), (4, 24, 8)].into_iter(),
        &patches,
        &mut || Ok(()),
        SloStructuredCostCapture::Disabled,
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
    let result = project_binding_slots(
        16,
        [(0, 0, 16)].into_iter(),
        &patches,
        &mut || {
            polls += 1;
            if polls == 3 {
                Err(VNextError::InvalidExecutionPlan {
                    reason: "expired query".to_owned(),
                })
            } else {
                Ok(())
            }
        },
        SloStructuredCostCapture::HostSettledV1,
    );
    assert!(result.is_err());
    assert_eq!(polls, 3);
}

#[test]
fn cuda_binding_numeric_growth_matches_actual_coalescing_without_changing_statistical_domain() {
    let command = binding(8, 8);
    let mut prior = None;
    for row_bytes in [424, 432, 448] {
        let writes = spans(&[(0, row_bytes), (1024, row_bytes), (2048, row_bytes)]);
        let projected = project_binding_slots(
            4096,
            [(1, 0, 3072)].into_iter(),
            &[ProgramBindingCostPatch {
                node_index: 1,
                command: &command,
                writes: &writes,
            }],
            &mut || Ok(()),
            SloStructuredCostCapture::HostSettledV1,
        )
        .unwrap();
        let actual = coalesce_program_binding_transfers(
            writes
                .iter()
                .map(|write| {
                    CudaProgramBindingWrite::new(
                        write.offset_bytes(),
                        vec![7; write.length_bytes() as usize].into_boxed_slice(),
                    )
                    .unwrap()
                })
                .collect(),
            4096,
        )
        .unwrap();
        assert_eq!(actual.len(), 1);
        assert_eq!(actual[0].destination_stride_bytes, 1024);
        assert_eq!(actual[0].row_count, 3);
        assert_eq!(actual[0].row_bytes as u64, row_bytes);
        let evidence = super::super::selected_cost::program_binding(
            actual.iter().map(|t| {
                (
                    t.destination_stride_bytes,
                    t.row_bytes as u64,
                    t.row_count as u64,
                )
            }),
            8,
            SloStructuredCostCapture::HostSettledV1,
        )
        .unwrap();
        let predicted = projected.statistical_evidence().unwrap();
        assert_eq!(evidence.family_signature(), predicted.family_signature());
        assert_eq!(evidence.work(), predicted.work());
        assert_eq!(evidence.work().host_to_device_bytes, 3 * row_bytes);
        if let Some(family) = prior {
            assert_eq!(family, *predicted.family_signature());
        }
        prior = Some(*predicted.family_signature());
    }
    // Statistical sharing must not weaken exact slot/range validation.
    let escape = spans(&[(0, 424), (1024, 424), (2048, 1025)]);
    let patches = [ProgramBindingCostPatch {
        node_index: 1,
        command: &command,
        writes: &escape,
    }];
    assert!(project_binding_slots(
        4096,
        [(1, 0, 3072)].into_iter(),
        &patches,
        &mut || Ok(()),
        SloStructuredCostCapture::HostSettledV1,
    )
    .is_err());
    assert!(coalesce_program_binding_transfers(
        vec![CudaProgramBindingWrite::new(2048, vec![0; 1025].into_boxed_slice()).unwrap()],
        3072,
    )
    .is_err());
}
