//! Equalize live prefixes within one provider-owned set of logical rows.
//!
//! The caller must establish that each unused logical tail is writable and is
//! ignored by the consumer. Neither physical capacity nor neighboring gaps are
//! authorized. The maximum logical size is only a bound, never an upload size.

pub(crate) fn uniform_live_prefix_width(
    lengths: impl IntoIterator<Item = usize>,
    logical_row_bytes: u64,
) -> Result<usize, String> {
    let mut width = None;
    for length in lengths {
        let bytes = u64::try_from(length)
            .map_err(|_| "program binding live prefix exceeds u64".to_owned())?;
        if bytes == 0 || bytes > logical_row_bytes {
            return Err("program binding payload does not fit its logical row".to_owned());
        }
        width = Some(width.map_or(length, |previous: usize| previous.max(length)));
    }
    width.ok_or_else(|| "program binding live prefix requires a nonempty row set".to_owned())
}

pub(crate) fn pad_live_prefix(
    payload: Box<[u8]>,
    prefix_bytes: usize,
) -> Result<Box<[u8]>, String> {
    if payload.is_empty() || payload.len() > prefix_bytes {
        return Err("program binding payload does not fit its live prefix".to_owned());
    }
    if payload.len() == prefix_bytes {
        return Ok(payload);
    }
    let mut row = payload.into_vec();
    row.try_reserve_exact(prefix_bytes - row.len())
        .map_err(|error| format!("program binding live prefix allocation failed: {error}"))?;
    row.resize(prefix_bytes, 0);
    Ok(row.into_boxed_slice())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn causal_payload(table_entries: i32, addresses: &[u64]) -> Box<[u8]> {
        let mut bytes = Vec::new();
        for word in [table_entries, 16, 1, 17, 0, 0] {
            bytes.extend_from_slice(&word.to_ne_bytes());
        }
        for address in addresses {
            bytes.extend_from_slice(&address.to_ne_bytes());
        }
        bytes.into_boxed_slice()
    }

    #[test]
    fn uniform_prefix_preserves_live_rows_and_ignores_large_context_capacity() {
        let rows = [causal_payload(1, &[3]), causal_payload(2, &[4, 5])];
        // The real 262144-token ABI reserves this much per row. It must not
        // determine payload allocation or overwrite the rest of either row.
        let logical_row_bytes = 24 + 262_144 / 16 * 8;
        let width =
            uniform_live_prefix_width(rows.iter().map(|row| row.len()), logical_row_bytes).unwrap();
        assert_eq!(width, 40);
        let stride = logical_row_bytes as usize;
        let mut arena = vec![0xa5; 32 + stride * 2 + 16];
        for (index, row) in rows.into_iter().enumerate() {
            let original = row.clone();
            let image = pad_live_prefix(row, width).unwrap();
            assert_eq!(&image[..original.len()], original.as_ref());
            assert!(image[original.len()..].iter().all(|&byte| byte == 0));
            let offset = 32 + index * stride;
            arena[offset..offset + image.len()].copy_from_slice(&image);
            assert!(arena[offset + image.len()..offset + stride]
                .iter()
                .all(|&byte| byte == 0xa5));
        }
        assert!(arena[..32].iter().all(|&byte| byte == 0xa5));
        assert!(arena[32 + stride * 2..].iter().all(|&byte| byte == 0xa5));
    }

    #[test]
    fn uniform_prefix_equal_width_and_single_participant_keep_owned_payloads() {
        for rows in [
            vec![causal_payload(1, &[7])],
            vec![causal_payload(1, &[7]); 2],
        ] {
            let width =
                uniform_live_prefix_width(rows.iter().map(|row| row.len()), 131_096).unwrap();
            for row in rows {
                let pointer = row.as_ptr();
                let expected = row.clone();
                let image = pad_live_prefix(row, width).unwrap();
                assert_eq!(image.as_ptr(), pointer);
                assert_eq!(image, expected);
            }
        }
    }

    #[test]
    fn uniform_prefix_preserves_int8_scale_addresses_after_actual_kv_entries() {
        // Complete payloads contain KV addresses followed immediately by scale
        // addresses. Padding is after both, never inserted at a maximum KV count.
        let short = causal_payload(1, &[0x1111, 0xaaaa]);
        let long = causal_payload(2, &[0x2222, 0x3333, 0xbbbb, 0xcccc]);
        let width = uniform_live_prefix_width([short.len(), long.len()], 131_096).unwrap();
        let expected = short.clone();
        let short = pad_live_prefix(short, width).unwrap();
        assert_eq!(&short[..expected.len()], expected.as_ref());
        assert_eq!(&short[..4], &1_i32.to_ne_bytes());
        assert_eq!(&short[20..24], &0_i32.to_ne_bytes());
        assert_eq!(&short[32..40], &0xaaaa_u64.to_ne_bytes());
        assert!(short[40..].iter().all(|&byte| byte == 0));
        assert_eq!(pad_live_prefix(long.clone(), width).unwrap(), long);
    }

    #[test]
    fn uniform_prefix_rejects_invalid_rows_and_reports_reservation_failure() {
        assert!(uniform_live_prefix_width([], 8).is_err());
        assert!(uniform_live_prefix_width([0], 8).is_err());
        assert!(uniform_live_prefix_width([8, 9], 8).is_err());
        assert_eq!(uniform_live_prefix_width([8], u64::MAX).unwrap(), 8);
        assert!(pad_live_prefix(Box::new([]), 8).is_err());
        assert!(pad_live_prefix(vec![1; 8].into_boxed_slice(), 7).is_err());
        // Vec::try_reserve_exact rejects an address-space-overflowing capacity;
        // this reaches the actual allocation failure path without allocating it.
        let error = pad_live_prefix(vec![1].into_boxed_slice(), usize::MAX).unwrap_err();
        assert!(error.contains("allocation failed"));
    }
}
