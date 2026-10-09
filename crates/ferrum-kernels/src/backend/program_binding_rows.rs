//! Complete one provider-owned logical row, never the surrounding arena.
//!
//! The caller must establish that the unused logical tail is writable and is
//! ignored by the consumer. Physical slot capacity alone does not establish it.

pub(crate) fn complete_logical_row(
    payload: Box<[u8]>,
    logical_row_bytes: u64,
) -> Result<Box<[u8]>, String> {
    let row_bytes = usize::try_from(logical_row_bytes)
        .map_err(|_| "program binding logical row exceeds usize".to_owned())?;
    if payload.is_empty() || payload.len() > row_bytes {
        return Err("program binding payload does not fit its logical row".to_owned());
    }
    if payload.len() == row_bytes {
        return Ok(payload);
    }
    let mut row = payload.into_vec();
    row.try_reserve_exact(row_bytes - row.len())
        .map_err(|error| format!("program binding logical row allocation failed: {error}"))?;
    row.resize(row_bytes, 0);
    Ok(row.into_boxed_slice())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn causal_payload(addresses: &[u64]) -> Box<[u8]> {
        let mut bytes = Vec::new();
        for word in [addresses.len() as i32, 16, 1, 17, 0, 0] {
            bytes.extend_from_slice(&word.to_ne_bytes());
        }
        for address in addresses {
            bytes.extend_from_slice(&address.to_ne_bytes());
        }
        bytes.into_boxed_slice()
    }

    #[test]
    fn complete_rows_preserve_control_addresses_and_zero_only_logical_tail() {
        let live = causal_payload(&[0x1122334455667788, 0x9988776655443322]);
        let expected = live.clone();
        let row = complete_logical_row(live, 64).unwrap();
        assert_eq!(&row[..expected.len()], expected.as_ref());
        assert_eq!(&row[20..24], &0_i32.to_ne_bytes());
        assert!(row[expected.len()..].iter().all(|&byte| byte == 0));

        // Two live rows occupy only part of a larger physical slot. Copying
        // their complete images must leave both neighboring slots and the
        // unused physical capacity untouched.
        let mut arena = vec![0xa5; 224];
        for (offset, addresses) in [(32, vec![3]), (96, vec![4, 5])] {
            let image = complete_logical_row(causal_payload(&addresses), 64).unwrap();
            arena[offset..offset + image.len()].copy_from_slice(&image);
        }
        assert!(arena[..32].iter().all(|&byte| byte == 0xa5));
        assert!(arena[160..].iter().all(|&byte| byte == 0xa5));
        assert_eq!(&arena[32..64], causal_payload(&[3]).as_ref());
        assert_eq!(&arena[96..136], causal_payload(&[4, 5]).as_ref());
    }

    #[test]
    fn complete_rows_reject_empty_short_and_unallocatable_rows() {
        assert!(complete_logical_row(Box::new([]), 8).is_err());
        assert!(complete_logical_row(vec![1; 8].into_boxed_slice(), 7).is_err());
        assert!(complete_logical_row(vec![1].into_boxed_slice(), u64::MAX).is_err());
    }

    #[test]
    fn already_complete_row_keeps_its_owned_payload() {
        let bytes = causal_payload(&[7]);
        let pointer = bytes.as_ptr();
        let row = complete_logical_row(bytes, 32).unwrap();
        assert_eq!(row.as_ptr(), pointer);
        assert_eq!(row.as_ref(), causal_payload(&[7]).as_ref());
    }
}
