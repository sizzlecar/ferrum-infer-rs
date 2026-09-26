//! Exact committed-byte trajectory shared by the live sampler and calibration
//! source replay. These pure checks grant no token, owner or sample authority.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InvalidCommittedUtf8;

pub fn advance_committed_utf8_fragment(
    pending: &[u8],
    next: &[u8],
) -> Result<Vec<u8>, InvalidCommittedUtf8> {
    let mut combined = Vec::with_capacity(pending.len().saturating_add(next.len()));
    combined.extend_from_slice(pending);
    combined.extend_from_slice(next);
    match std::str::from_utf8(&combined) {
        Ok(text) => {
            if text.contains('\u{FFFD}') || contains_output_replacement_mojibake(text) {
                Err(InvalidCommittedUtf8)
            } else {
                Ok(Vec::new())
            }
        }
        Err(error) if error.error_len().is_none() => {
            let valid_prefix = std::str::from_utf8(&combined[..error.valid_up_to()])
                .map_err(|_| InvalidCommittedUtf8)?;
            if valid_prefix.contains('\u{FFFD}')
                || contains_output_replacement_mojibake(valid_prefix)
            {
                return Err(InvalidCommittedUtf8);
            }
            let fragment = &combined[error.valid_up_to()..];
            if fragment.is_empty() || fragment.len() > 3 {
                return Err(InvalidCommittedUtf8);
            }
            Ok(fragment.to_vec())
        }
        Err(_) => Err(InvalidCommittedUtf8),
    }
}

pub fn contains_output_replacement_mojibake(text: &str) -> bool {
    let mut chars = text.chars();
    let mut a = chars.next();
    let mut b = chars.next();
    let mut c = chars.next();
    loop {
        if matches!(
            (a, b, c),
            (Some('\u{00ef}'), Some('\u{00bf}'), Some('\u{00bd}'))
        ) {
            return true;
        }
        if c.is_none() {
            return false;
        }
        a = b;
        b = c;
        c = chars.next();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn committed_utf8_preserves_live_pending_and_rejects_invalid_or_replacement_text() {
        let pending = advance_committed_utf8_fragment(&[], &[b'a', 0xc3]).unwrap();
        assert_eq!(pending, [0xc3]);
        assert!(advance_committed_utf8_fragment(&pending, &[0xa9])
            .unwrap()
            .is_empty());
        assert!(advance_committed_utf8_fragment(&pending, b"x").is_err());
        assert!(advance_committed_utf8_fragment(&[], &[0xa9]).is_err());
        assert!(advance_committed_utf8_fragment(&[], "\u{fffd}".as_bytes()).is_err());
        assert!(advance_committed_utf8_fragment(&[], "\u{ef}\u{bf}\u{bd}".as_bytes()).is_err());
        assert_eq!(
            advance_committed_utf8_fragment(&[], &[0xf0, 0x9f, 0x92]).unwrap(),
            [0xf0, 0x9f, 0x92]
        );
        assert!(
            advance_committed_utf8_fragment(&[0xf0, 0x9f, 0x92], &[0xa9])
                .unwrap()
                .is_empty()
        );
        assert!(advance_committed_utf8_fragment(&[], &[0xed, 0xa0]).is_err());
    }
}
