use super::*;

fn write(offset: u64, bytes: u64) -> ProgramBindingCostWrite {
    ProgramBindingCostWrite::new(offset, bytes).unwrap()
}

#[test]
fn contiguous_then_strided_layout_preserves_every_original_payload_once() {
    let writes = [
        write(0, 2),
        write(2, 2),
        write(16, 4),
        write(32, 4),
        write(64, 8),
        write(96, 8),
    ];
    let result = coalesce_sorted_program_binding_writes(&writes, 104, &mut || Ok(())).unwrap();
    assert_eq!(
        result,
        vec![
            ProgramBindingTransferLayout {
                destination_offset_bytes: 0,
                destination_stride_bytes: 16,
                row_bytes: 4,
                row_count: 3,
                source_writes: 0..4
            },
            ProgramBindingTransferLayout {
                destination_offset_bytes: 64,
                destination_stride_bytes: 32,
                row_bytes: 8,
                row_count: 2,
                source_writes: 4..6
            },
        ]
    );
    assert_eq!(
        result
            .iter()
            .flat_map(|row| row.source_writes.clone())
            .collect::<Vec<_>>(),
        (0..writes.len()).collect::<Vec<_>>()
    );
}

#[test]
fn unequal_gaps_split_real_transfers_and_no_padding_is_uploaded() {
    let result = coalesce_sorted_program_binding_writes(
        &[write(8, 4), write(24, 4), write(48, 4), write(52, 4)],
        56,
        &mut || Ok(()),
    )
    .unwrap();
    assert_eq!(result.len(), 2);
    assert_eq!(
        (
            result[0].destination_stride_bytes,
            result[0].row_count,
            result[0].row_bytes
        ),
        (16, 2, 4)
    );
    assert_eq!(
        (
            result[1].destination_offset_bytes,
            result[1].row_count,
            result[1].row_bytes
        ),
        (48, 1, 8)
    );
}

#[test]
fn invalid_spans_and_expired_query_cannot_produce_a_transfer_layout() {
    assert!(ProgramBindingCostWrite::new(0, 0).is_err());
    assert!(ProgramBindingCostWrite::new(u64::MAX, 1).is_err());
    for writes in [
        vec![],
        vec![write(0, 8), write(4, 4)],
        vec![write(8, 1), write(0, 1)],
        vec![write(15, 2)],
    ] {
        assert!(coalesce_sorted_program_binding_writes(&writes, 16, &mut || Ok(())).is_err());
    }
    let mut polls = 0;
    assert!(
        coalesce_sorted_program_binding_writes(&[write(0, 1), write(4, 1)], 8, &mut || {
            polls += 1;
            if polls == 2 {
                Err(invalid_operation("fixture query deadline"))
            } else {
                Ok(())
            }
        })
        .is_err()
    );
    assert_eq!(polls, 2);
}
