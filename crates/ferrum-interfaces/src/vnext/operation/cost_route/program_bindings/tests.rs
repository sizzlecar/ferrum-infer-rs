use super::*;

fn assert_exact_live_cover(
    writes: &[ProgramBindingCostWrite],
    arena_size: usize,
    transfers: &[ProgramBindingTransferLayout],
) {
    const UNWRITTEN: u8 = 0xa7;
    let mut expected = vec![UNWRITTEN; arena_size];
    let mut actual = expected.clone();
    let mut actual_live = vec![false; arena_size];
    let mut expected_live = actual_live.clone();
    let mut covered = vec![0_usize; writes.len()];
    let payloads = writes
        .iter()
        .enumerate()
        .map(|(index, write)| {
            let payload = (0..write.length_bytes())
                .map(|j| ((index as u64 * 17 + j * 31) % 251) as u8)
                .collect::<Vec<_>>();
            let start = write.offset_bytes() as usize;
            expected[start..start + payload.len()].copy_from_slice(&payload);
            expected_live[start..start + payload.len()].fill(true);
            payload
        })
        .collect::<Vec<_>>();
    let mut prior_start = None;
    for transfer in transfers {
        assert!(prior_start.is_none_or(|prior| prior < transfer.destination_offset_bytes));
        prior_start = Some(transfer.destination_offset_bytes);
        assert!(transfer.row_count > 0);
        assert!(transfer.destination_stride_bytes >= transfer.row_bytes);
        let mut packed = Vec::new();
        for range in &transfer.source_write_ranges {
            assert!(!range.is_empty() && range.end <= writes.len());
            for index in range.clone() {
                covered[index] += 1;
                packed.extend_from_slice(&payloads[index]);
            }
        }
        assert_eq!(
            packed.len(),
            transfer.row_count * transfer.row_bytes as usize
        );
        for row in 0..transfer.row_count {
            let start = transfer.destination_offset_bytes as usize
                + row * transfer.destination_stride_bytes as usize;
            let source = row * transfer.row_bytes as usize;
            assert!(
                actual_live[start..start + transfer.row_bytes as usize]
                    .iter()
                    .all(|written| !written),
                "transfers may not alias live destinations"
            );
            actual_live[start..start + transfer.row_bytes as usize].fill(true);
            actual[start..start + transfer.row_bytes as usize]
                .copy_from_slice(&packed[source..source + transfer.row_bytes as usize]);
        }
    }
    assert!(covered.iter().all(|&count| count == 1), "{covered:?}");
    assert_eq!(actual_live, expected_live);
    assert_eq!(
        actual, expected,
        "all holes and interleaved writes must retain exact bytes"
    );
}

#[test]
fn indexed_binding_groups_join_cross_slot_rows_without_copying_other_widths() {
    let writes = [
        write(0, 2),
        write(2, 2),
        write(16, 8),
        write(64, 1),
        write(65, 3),
        write(80, 8),
        write(128, 4),
        write(144, 8),
    ];
    let result = coalesce_sorted_program_binding_writes(&writes, 152, &mut || Ok(())).unwrap();
    assert_eq!(result.len(), 2);
    assert_eq!(result[0].source_write_ranges, vec![0..2, 3..5, 6..7]);
    assert_eq!(result[1].source_write_ranges, vec![2..3, 5..6, 7..8]);
    assert_eq!(
        (
            result[0].row_bytes,
            result[0].row_count,
            result[0].destination_stride_bytes
        ),
        (4, 3, 64)
    );
    assert_eq!(
        (
            result[1].row_bytes,
            result[1].row_count,
            result[1].destination_stride_bytes
        ),
        (8, 3, 64)
    );
    assert_exact_live_cover(&writes, 152, &result);
}

#[test]
fn indexed_binding_groups_preserve_original_ties_and_do_not_join_unequal_stride() {
    let writes = [write(0, 4), write(16, 8), write(64, 4), write(96, 4)];
    let result = coalesce_sorted_program_binding_writes(&writes, 100, &mut || Ok(())).unwrap();
    // Width-only regrouping also needs three transfers. Retain the original.
    assert_eq!(
        result
            .iter()
            .map(|t| (t.destination_offset_bytes, t.row_count))
            .collect::<Vec<_>>(),
        vec![(0, 1), (16, 1), (64, 2)]
    );
    assert_eq!(result[2].source_write_ranges, vec![2..4]);
    assert_exact_live_cover(&writes, 100, &result);
}

#[test]
fn indexed_binding_groups_exhaustive_small_live_spans_never_alias_or_fill_holes() {
    // Each digit selects 1..3 live bytes and either no gap or a 3-byte hole.
    // This checks payload semantics, not a preferred grouping implementation.
    for mut pattern in 0..6_usize.pow(4) {
        let mut writes = Vec::new();
        let mut end = 0;
        for _ in 0..4 {
            let digit = pattern % 6;
            pattern /= 6;
            let bytes = (digit % 3 + 1) as u64;
            let gap = if digit < 3 { 0 } else { 3 };
            writes.push(write(end + gap, bytes));
            end += gap + bytes;
        }
        let result = coalesce_sorted_program_binding_writes(&writes, end, &mut || Ok(())).unwrap();
        assert!(result.len() <= writes.len());
        assert_exact_live_cover(&writes, end as usize, &result);
    }
}

#[test]
fn indexed_binding_groups_do_not_return_a_partial_plan_after_budget_expiry() {
    let writes = [write(0, 4), write(16, 8), write(64, 4), write(80, 8)];
    let mut total_polls = 0;
    let complete = coalesce_sorted_program_binding_writes(&writes, 88, &mut || {
        total_polls += 1;
        Ok(())
    })
    .unwrap();
    assert_eq!(complete.len(), 2);
    for expires_at in 1..=total_polls {
        let mut polls = 0;
        assert!(
            coalesce_sorted_program_binding_writes(&writes, 88, &mut || {
                polls += 1;
                if polls == expires_at {
                    Err(invalid_operation("fixture deadline"))
                } else {
                    Ok(())
                }
            })
            .is_err()
        );
        assert_eq!(polls, expires_at);
    }
}

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
                source_write_ranges: vec![0..4]
            },
            ProgramBindingTransferLayout {
                destination_offset_bytes: 64,
                destination_stride_bytes: 32,
                row_bytes: 8,
                row_count: 2,
                source_write_ranges: vec![4..6]
            },
        ]
    );
    assert_eq!(
        result
            .iter()
            .flat_map(|row| row.source_write_ranges.iter().cloned().flatten())
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
