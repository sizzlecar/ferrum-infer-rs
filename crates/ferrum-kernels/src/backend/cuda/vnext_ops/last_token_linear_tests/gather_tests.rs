use super::*;
use ferrum_interfaces::vnext::{ResourceWorkShape, TokenSpanWork};

fn weight() -> LastTokenByteSpan {
    LastTokenByteSpan {
        pointer: 0x10000,
        bytes: 16 * 32 * 2,
    }
}

fn scratch() -> LastTokenByteSpan {
    LastTokenByteSpan {
        pointer: 0x20000,
        bytes: 2 * 16 * 2,
    }
}

#[test]
fn last_token_gather_workspace_tracks_sequences_not_prefill_tokens() {
    let workspace = last_token_gather_workspace(17).unwrap();
    assert_eq!(
        workspace.size_formula(),
        &ProviderWorkspaceSizeFormula::ActualSequences {
            bytes_per_sequence: 34
        }
    );
    assert_eq!(workspace.alignment_bytes(), 16);
    assert_eq!(workspace.scope(), ProviderWorkspaceScope::Invocation);
    assert_eq!(
        workspace.reuse_policy(),
        ProviderWorkspaceReusePolicy::OverwriteBeforeRead
    );
    let work = ResourceWorkShape::from_token_spans(vec![
        TokenSpanWork::from_token_ids_with_fit(&[1, 2, 3], 0..3, 9).unwrap(),
        TokenSpanWork::from_token_ids(&[4, 5], 1..2).unwrap(),
    ])
    .unwrap();
    assert_eq!(workspace.evaluate_bytes(&work).unwrap(), 80);
    assert_eq!(workspace.evaluate_fit_bytes(&work).unwrap(), 80);
    assert!(last_token_gather_workspace(0).is_err());
    assert!(last_token_gather_workspace(u64::MAX).is_err());
}

#[test]
fn last_token_gather_rejects_alias_without_rejecting_safe_fallback_layouts() {
    let mut rows = row_layouts(2);
    rows[1].input_pointer += 18;
    validate_last_token_f16_access(16, 32, weight(), &rows).unwrap();
    assert!(last_token_output_rows_are_contiguous(32, &rows));
    assert!(last_token_gather_scratch_is_disjoint(
        16,
        32,
        weight(),
        scratch(),
        &rows
    ));

    for alias in [
        rows[0].input_pointer,
        rows[0].output_pointer,
        weight().pointer,
    ] {
        assert!(!last_token_gather_scratch_is_disjoint(
            16,
            32,
            weight(),
            LastTokenByteSpan {
                pointer: alias,
                ..scratch()
            },
            &rows
        ));
        // The scratch is unused on fallback; ordinary input/output access is valid.
        validate_last_token_f16_access(16, 32, weight(), &rows).unwrap();
    }
    for invalid in [
        LastTokenByteSpan {
            bytes: 62,
            ..scratch()
        },
        LastTokenByteSpan {
            pointer: scratch().pointer + 2,
            ..scratch()
        },
        LastTokenByteSpan {
            pointer: u64::MAX - 15,
            ..scratch()
        },
    ] {
        assert!(!last_token_gather_scratch_is_disjoint(
            16,
            32,
            weight(),
            invalid,
            &rows
        ));
    }
    rows[1].output_pointer += 16;
    validate_last_token_f16_access(16, 32, weight(), &rows).unwrap();
    assert!(!last_token_output_rows_are_contiguous(32, &rows));
    rows.reverse();
    validate_last_token_f16_access(16, 32, weight(), &rows).unwrap();
    assert!(!last_token_output_rows_are_contiguous(32, &rows));
}

#[test]
fn last_token_gather_does_not_hide_invalid_scalar_access() {
    let rows = row_layouts(2);
    for invalid_weight in [
        LastTokenByteSpan {
            bytes: weight().bytes - 2,
            ..weight()
        },
        LastTokenByteSpan {
            pointer: u64::MAX - 15,
            ..weight()
        },
        LastTokenByteSpan {
            pointer: rows[0].output_pointer,
            ..weight()
        },
    ] {
        assert!(validate_last_token_f16_access(16, 32, invalid_weight, &rows).is_err());
    }
    for bad in [
        LastTokenF16Row {
            input_bytes: 30,
            ..rows[1]
        },
        LastTokenF16Row {
            output_bytes: 62,
            ..rows[1]
        },
        LastTokenF16Row {
            input_pointer: rows[0].output_pointer,
            ..rows[1]
        },
        LastTokenF16Row {
            output_pointer: rows[0].input_pointer,
            ..rows[1]
        },
        LastTokenF16Row {
            output_pointer: rows[0].output_pointer,
            ..rows[1]
        },
        LastTokenF16Row {
            input_pointer: 0,
            ..rows[1]
        },
    ] {
        assert!(validate_last_token_f16_access(16, 32, weight(), &[rows[0], bad]).is_err());
    }
    assert!(validate_last_token_f16_access(u64::MAX, 32, weight(), &rows).is_err());
    assert!(validate_last_token_f16_access(16, 0, weight(), &rows).is_err());
}

#[test]
#[ignore = "requires an actual CUDA device"]
fn gathered_f16_last_token_projection_matches_f64_and_preserves_fallback_on_cuda() {
    for (hidden, outputs) in [(17, 19), (256, 128)] {
        for rows in [2, 4, 8] {
            for layout in [
                Layout {
                    input_gap: 0,
                    output_gap: 0,
                    reverse_outputs: false,
                    first_prefill_tokens: 3,
                },
                Layout {
                    input_gap: 3,
                    output_gap: 0,
                    reverse_outputs: false,
                    first_prefill_tokens: 1,
                },
                Layout {
                    input_gap: 0,
                    output_gap: 0,
                    reverse_outputs: false,
                    first_prefill_tokens: 1,
                },
            ] {
                projection_case_with_gather(rows, hidden, outputs, layout, true);
            }
        }
    }
    for layout in [
        Layout {
            input_gap: 0,
            output_gap: 5,
            reverse_outputs: false,
            first_prefill_tokens: 3,
        },
        Layout {
            input_gap: 0,
            output_gap: 0,
            reverse_outputs: true,
            first_prefill_tokens: 3,
        },
    ] {
        projection_case_with_gather(4, 17, 19, layout, true);
    }
}

#[test]
fn last_token_gather_replay_binds_mode_copy_mapping_and_scratch_address() {
    let launches = [LastTokenDenseLinearLaunch {
        input_region: 5,
        output_region: 2,
        rows: 2,
    }];
    let fingerprint = "a".repeat(64);
    let key = |packed, scratch, sources: &[usize], bytes| {
        last_token_linear_replay_key(
            &fingerprint,
            packed,
            16,
            32,
            &launches,
            scratch,
            sources,
            bytes,
        )
    };
    let gathered = key(false, Some(5), &[1, 3], 32);
    assert_ne!(gathered, key(true, None, &[], 32));
    assert_ne!(gathered, key(false, Some(6), &[1, 3], 32));
    assert_ne!(gathered, key(false, Some(5), &[3, 1], 32));
    assert_ne!(gathered, key(false, Some(5), &[1, 3], 64));
    let bind = |scratch_pointer| {
        gathered.bind_runtime_payload(
            "test.last-token-gather",
            [
                (weight().pointer, weight().bytes, ElementType::F16),
                (scratch_pointer, scratch().bytes, ElementType::U8),
            ]
            .into_iter(),
            &[],
        )
    };
    assert_ne!(bind(scratch().pointer), bind(scratch().pointer + 128));
    assert_eq!(bind(scratch().pointer), bind(scratch().pointer));
}
