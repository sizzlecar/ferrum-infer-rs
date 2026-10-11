//! Diagnostic interpretation of the bytes consumed by the production scatter.
//! Checks exact ordered destination coverage against ordinary provider writes;
//! it does not reconstruct a second production packet or enqueue either path.

pub(super) struct ExpectedWrite<'a> {
    pub offset: u64,
    pub bytes: &'a [u8],
}

pub(super) fn matches(
    packet: &[u8],
    descriptor_count: u32,
    arena_bytes: u64,
    expected: &[ExpectedWrite<'_>],
) -> bool {
    let Some(header) = (descriptor_count as usize).checked_mul(40) else {
        return false;
    };
    if descriptor_count == 0 || header > packet.len() || expected.is_empty() {
        return false;
    }
    let mut prior_end = 0;
    for write in expected {
        let Some(end) = write.offset.checked_add(write.bytes.len() as u64) else {
            return false;
        };
        if write.bytes.is_empty() || write.offset < prior_end || end > arena_bytes {
            return false;
        }
        prior_end = end;
    }
    let mut decoded_rows = Vec::new();
    let mut source_end = header;
    for descriptor in packet[..header].chunks_exact(40) {
        let field = |index: usize| {
            let mut bytes = [0; 8];
            bytes.copy_from_slice(&descriptor[index * 8..(index + 1) * 8]);
            u64::from_le_bytes(bytes)
        };
        let (source, destination, width, rows, pitch) =
            (field(0), field(1), field(2), field(3), field(4));
        let (Ok(source), Ok(width_usize), Ok(rows_usize)) = (
            usize::try_from(source),
            usize::try_from(width),
            usize::try_from(rows),
        ) else {
            return false;
        };
        let Some(payload_bytes) = width_usize.checked_mul(rows_usize) else {
            return false;
        };
        let Some(end) = source.checked_add(payload_bytes) else {
            return false;
        };
        if width == 0
            || rows == 0
            || (rows > 1 && pitch < width)
            || source != source_end
            || end > packet.len()
        {
            return false;
        }
        source_end = end;
        if decoded_rows.try_reserve(rows_usize).is_err() {
            return false;
        }
        for row in 0..rows {
            let Some(row_offset) = row
                .checked_mul(pitch)
                .and_then(|v| destination.checked_add(v))
            else {
                return false;
            };
            if row_offset
                .checked_add(width)
                .is_none_or(|end| end > arena_bytes)
            {
                return false;
            }
            // The validated product bounds above make both usize operations safe.
            decoded_rows.push((row_offset, source + row as usize * width_usize, width_usize));
        }
    }
    // Equal-width coalescing may legally visit high-address rows before a
    // different-width transfer fills the intervening rows. Compare destination
    // coverage in address order without changing or reconstructing the packet.
    decoded_rows.sort_unstable_by_key(|row| row.0);
    let mut write_index = 0;
    let mut consumed = 0;
    for (mut destination_cursor, mut source_cursor, mut remaining) in decoded_rows {
        while remaining != 0 {
            let Some(write) = expected.get(write_index) else {
                return false;
            };
            if write.offset.checked_add(consumed as u64) != Some(destination_cursor) {
                return false;
            }
            let count = remaining.min(write.bytes.len() - consumed);
            if packet[source_cursor..source_cursor + count]
                != write.bytes[consumed..consumed + count]
            {
                return false;
            }
            source_cursor += count;
            destination_cursor += count as u64;
            remaining -= count;
            consumed += count;
            if consumed == write.bytes.len() {
                write_index += 1;
                consumed = 0;
            }
        }
    }
    source_end == packet.len() && write_index == expected.len() && consumed == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn packet(descriptors: &[[u64; 5]], bytes: &[u8]) -> Vec<u8> {
        descriptors
            .iter()
            .flatten()
            .flat_map(|v| v.to_le_bytes())
            .chain(bytes.iter().copied())
            .collect()
    }

    #[test]
    fn compiled_oracle_packet_compares_merged_rows_and_preserves_holes() {
        let expected = [
            ExpectedWrite {
                offset: 4,
                bytes: &[1, 2],
            },
            ExpectedWrite {
                offset: 6,
                bytes: &[3, 4],
            },
            ExpectedWrite {
                offset: 12,
                bytes: &[5, 6, 7, 8],
            },
        ];
        let original = packet(&[[40, 4, 4, 2, 8]], &[1, 2, 3, 4, 5, 6, 7, 8]);
        assert!(matches(&original, 1, 16, &expected));
        let mut changed = original.clone();
        changed[43] ^= 1;
        assert!(
            !matches(&changed, 1, 16, &expected),
            "actual packet byte mismatch"
        );
        let holes_overwritten = packet(&[[40, 4, 8, 1, 0]], &[1, 2, 3, 4, 5, 6, 7, 8]);
        assert!(!matches(&holes_overwritten, 1, 16, &expected));
        assert!(
            !matches(&original, 1, 16, &expected[..2]),
            "extra node write cannot disappear from the reference"
        );
        assert!(
            !matches(&original, 1, 15, &expected),
            "arena extent remains exact"
        );
        let mut wrong_source = original;
        wrong_source[..8].copy_from_slice(&0_u64.to_le_bytes());
        assert!(
            !matches(&wrong_source, 1, 16, &expected),
            "descriptor bytes are not payload"
        );
    }

    #[test]
    fn compiled_oracle_packet_rejects_aliasing_overflow_and_unused_bytes() {
        let expected = [
            ExpectedWrite {
                offset: 4,
                bytes: &[1, 2],
            },
            ExpectedWrite {
                offset: 8,
                bytes: &[3, 4],
            },
        ];
        let original = packet(&[[80, 4, 2, 1, 0], [82, 8, 2, 1, 0]], &[1, 2, 3, 4]);
        assert!(matches(&original, 2, 10, &expected));
        for second in [
            [80, 8, 2, 1, 0],
            [82, 4, 2, 1, 0],
            [82, u64::MAX, 2, 1, 0],
            [82, 8, 2, u64::MAX, 4],
        ] {
            assert!(!matches(
                &packet(&[[80, 4, 2, 1, 0], second], &[1, 2, 3, 4]),
                2,
                10,
                &expected
            ));
        }
        let mut trailing = original;
        trailing.push(0);
        assert!(!matches(&trailing, 2, 10, &expected));
    }

    #[test]
    fn compiled_oracle_packet_accepts_real_interleaved_width_coalescing() {
        use super::super::super::binding_transfers::{coalesce_sorted_writes, BindingWrite};
        let expected = [
            ExpectedWrite {
                offset: 4,
                bytes: &[1, 2],
            },
            ExpectedWrite {
                offset: 10,
                bytes: &[3, 4, 5],
            },
            ExpectedWrite {
                offset: 20,
                bytes: &[6, 7],
            },
            ExpectedWrite {
                offset: 30,
                bytes: &[8, 9, 10],
            },
            ExpectedWrite {
                offset: 36,
                bytes: &[11, 12],
            },
        ];
        let writes = expected
            .iter()
            .map(|write| BindingWrite::new(write.offset, write.bytes.len() as u64).unwrap())
            .collect::<Vec<_>>();
        let transfers = coalesce_sorted_writes(&writes, 38).unwrap();
        // This reaches the existing cross-intervening-width grouping branch.
        assert!(transfers
            .iter()
            .any(|transfer| transfer.source_write_ranges.len() > 1));
        let mut descriptors = Vec::new();
        let mut payload = Vec::new();
        for transfer in &transfers {
            let source = transfers.len() * 40 + payload.len();
            for sources in &transfer.source_write_ranges {
                for index in sources.clone() {
                    payload.extend_from_slice(expected[index].bytes);
                }
            }
            descriptors.push([
                source as u64,
                transfer.destination_offset_bytes,
                transfer.row_bytes,
                transfer.row_count as u64,
                transfer.destination_stride_bytes,
            ]);
        }
        let original = packet(&descriptors, &payload);
        assert!(matches(&original, descriptors.len() as u32, 38, &expected));
        // Sorting must not hide an overlap or duplicated destination row.
        let mut overlapping = descriptors.clone();
        overlapping[1][1] = 5;
        assert!(!matches(
            &packet(&overlapping, &payload),
            overlapping.len() as u32,
            38,
            &expected
        ));
        let mut duplicated = descriptors;
        duplicated[1][1] = 4;
        assert!(!matches(
            &packet(&duplicated, &payload),
            duplicated.len() as u32,
            38,
            &expected
        ));
    }
}
