use super::*;

#[test]
fn array_order_pair_and_complete_raw_digest_are_preserved() {
    // Keep the deterministic fixture typed, including whitespace in prompt text.
    let value = serde_json::json!([
        {"id":17,"conversations":[{"from":"human","value":" H "},{"from":"gpt","value":"A"}]},
        null
    ]);
    let source = format!(" \n{} \n", value);
    let mut indices = Vec::new();
    let receipt = visit_sharegpt_records(
        source.as_bytes(),
        ShareGptReadLimits::default(),
        |index, record| {
            indices.push(index);
            let ShareGptRecord::Value(value) = record else {
                panic!("array row")
            };
            if index == 0 {
                let pair = sharegpt_first_pair(&value).unwrap();
                assert_eq!(pair.prompt, " H ");
                assert_eq!(pair.assistant, "A");
                assert_eq!(sharegpt_original_id(&value).as_deref(), Some("17"));
            } else {
                assert!(sharegpt_first_pair(&value).is_none());
            }
            Ok(())
        },
    )
    .unwrap();
    assert_eq!(indices, [0, 1]);
    assert_eq!(receipt.source_format, "json_array");
    assert_eq!(receipt.source_bytes, source.len() as u64);
    assert_eq!(
        receipt.source_sha256,
        format!("{:x}", Sha256::digest(source.as_bytes()))
    );
}

#[test]
fn malformed_nonblank_jsonl_has_an_index_but_blank_lines_do_not() {
    let source = b"\n {}\n \t\ninvalid\nnull\n";
    let mut kinds = Vec::new();
    let receipt = visit_sharegpt_records(
        source.as_slice(),
        ShareGptReadLimits::default(),
        |index, record| {
            kinds.push((index, matches!(record, ShareGptRecord::MalformedJsonLine)));
            Ok(())
        },
    )
    .unwrap();
    assert_eq!(kinds, [(0, false), (1, true), (2, false)]);
    assert_eq!(receipt.records, 3);
    assert_eq!(
        receipt.source_sha256,
        format!("{:x}", Sha256::digest(source))
    );
}

#[test]
fn source_record_and_count_limits_fail_without_partial_receipts() {
    let source = b"[{},{}]";
    let exact = ShareGptReadLimits {
        maximum_source_bytes: NonZeroU64::new(source.len() as u64).unwrap(),
        ..Default::default()
    };
    assert!(visit_sharegpt_records(source.as_slice(), exact, |_, _| Ok(())).is_ok());
    let short = ShareGptReadLimits {
        maximum_source_bytes: NonZeroU64::new(source.len() as u64 - 1).unwrap(),
        ..exact
    };
    assert!(visit_sharegpt_records(source.as_slice(), short, |_, _| Ok(())).is_err());
    let one = ShareGptReadLimits {
        maximum_records: NonZeroU64::new(1).unwrap(),
        ..exact
    };
    assert!(visit_sharegpt_records(source.as_slice(), one, |_, _| Ok(())).is_err());
    let tiny = ShareGptReadLimits {
        maximum_record_bytes: NonZeroUsize::new(8).unwrap(),
        ..Default::default()
    };
    for oversized in [
        b"[{\"long\":\"abcdefghijkl\"}]".as_slice(),
        b"{\"long\":\"abcdefghijkl\"}\n",
    ] {
        assert!(visit_sharegpt_records(oversized, tiny, |_, _| Ok(())).is_err());
    }
}

#[test]
fn callback_failure_and_trailing_array_data_are_errors() {
    assert!(
        visit_sharegpt_records(b"[{}]".as_slice(), ShareGptReadLimits::default(), |_, _| {
            Err(FerrumError::config("selected hash mismatch"))
        })
        .is_err()
    );
    assert!(visit_sharegpt_records(
        b"[{}]junk".as_slice(),
        ShareGptReadLimits::default(),
        |_, _| Ok(())
    )
    .is_err());
}
