use super::*;

pub fn verify_completed_input(
    kind: AttentionKind,
    prefix: &[std::ops::Range<usize>],
    suffix: &[std::ops::Range<usize>],
) {
    let fixture = Fixture::new(kind);
    let total = suffix.last().unwrap().end;
    let tokens: Arc<[u32]> = (0..total)
        .map(|index| ((index * 7 + 3) % 32) as u32)
        .collect();
    let initial: Arc<[u32]> = Arc::from(&tokens[..prefix[0].end]);
    let source = fixture.admit_with_ceiling("growing-source", Arc::clone(&initial), total);
    let cold = fixture.admit_with_ceiling("growing-cold", initial, total);
    let restored = fixture.admit("growing-restored", Arc::clone(&tokens));
    let mut boundary_state = None;
    for span in prefix {
        // Only the actual input so far is bound to this FullPlan. The fit
        // ceiling above reserves no knowledge of future token values.
        let current: Arc<[u32]> = Arc::from(&tokens[..span.end]);
        fixture.extend(&source, Arc::clone(&current));
        fixture.extend(&cold, Arc::clone(&current));
        let expected = fixture.execute(&source, Arc::clone(&current), span.clone());
        expected.assert_same(
            &fixture.execute(&cold, current, span.clone()),
            "growing prefix",
        );
        expected.assert_state_nonzero();
        boundary_state = Some(expected);
    }
    assert_eq!(prefix.last().unwrap().len(), 1);
    let checkpoint = fixture.capture_completed_input(&source);
    let boundary = prefix.last().unwrap().end;
    assert_eq!(checkpoint.full_input(), &tokens[..boundary]);
    let expected = suffix
        .iter()
        .map(|span| {
            let current: Arc<[u32]> = Arc::from(&tokens[..span.end]);
            fixture.extend(&source, Arc::clone(&current));
            fixture.execute(&source, current, span.clone())
        })
        .collect::<Vec<_>>();
    expected
        .last()
        .unwrap()
        .assert_state_changed(boundary_state.as_ref().unwrap(), "growing suffix");
    source.try_complete().unwrap();
    drop(source);

    let mut changed = tokens.to_vec();
    changed[boundary - 1] = (changed[boundary - 1] + 1) % 32;
    let changed: Arc<[u32]> = changed.into();
    let wrong = fixture.admit("growing-changed-target", Arc::clone(&changed));
    fixture.assert_restore_rejected(&wrong, &checkpoint, changed);
    wrong.try_abort_if_quiescent().unwrap();
    drop(wrong);
    // Bind the actual shorter admission too, so this rejects an empty suffix
    // rather than a caller input that differs from the target's admission.
    let no_suffix: Arc<[u32]> = Arc::from(checkpoint.full_input());
    let empty = fixture.admit("growing-empty-suffix-target", Arc::clone(&no_suffix));
    fixture.assert_restore_rejected(&empty, &checkpoint, no_suffix);
    empty.try_abort_if_quiescent().unwrap();
    drop(empty);
    fixture.restore(&restored, &checkpoint, Arc::clone(&tokens));
    for (span, expected) in suffix.iter().zip(expected) {
        let current: Arc<[u32]> = Arc::from(&tokens[..span.end]);
        fixture.extend(&cold, Arc::clone(&current));
        expected.assert_same(
            &fixture.execute(&cold, current, span.clone()),
            "growing cold suffix",
        );
        expected.assert_same(
            &fixture.execute(&restored, Arc::clone(&tokens), span.clone()),
            "completed-input restored suffix",
        );
    }
    drop(checkpoint);
    cold.try_complete().unwrap();
    restored.try_complete().unwrap();
}

pub fn verify(
    kind: AttentionKind,
    prefix: &[std::ops::Range<usize>],
    suffix: &[std::ops::Range<usize>],
) {
    verify_with_timing(kind, prefix, suffix, DeviceTimingMode::Off);
}

pub fn verify_with_timing(
    kind: AttentionKind,
    prefix: &[std::ops::Range<usize>],
    suffix: &[std::ops::Range<usize>],
    timing: DeviceTimingMode,
) {
    let fixture = Fixture::new(kind).with_checkpoint_timing(timing);
    let total = suffix.last().unwrap().end;
    let tokens: Arc<[u32]> = (0..total)
        .map(|index| ((index * 7 + 3) % 32) as u32)
        .collect();
    let source = fixture.admit("source", Arc::clone(&tokens));
    let cold = fixture.admit("cold", Arc::clone(&tokens));
    let restored = fixture.admit("restored", Arc::clone(&tokens));
    eprintln!(
        "{kind:?}: hidden={HIDDEN}, activation={:?}, native block projections, single participant, prefix={prefix:?}, suffix={suffix:?}; real embedding+attention FullPlan, native public capture/restore",
        kind.activation_type()
    );
    let mut previous = None;
    for span in prefix {
        let expected = fixture.execute(&source, Arc::clone(&tokens), span.clone());
        let actual = fixture.execute(&cold, Arc::clone(&tokens), span.clone());
        expected.assert_same(&actual, "cold prefix");
        expected.assert_state_nonzero();
        previous = Some(expected);
    }
    let checkpoint = fixture.capture(&source);
    assert_eq!(checkpoint.completed_tokens(), prefix.last().unwrap().end);
    if kind == AttentionKind::CausalInt8 {
        assert_eq!(
            checkpoint.logical_bytes(),
            checkpoint.completed_tokens() as u64 * (2 * 2 * 128 + 2 * 2 * 4)
        );
    }
    assert_eq!(
        checkpoint.token_prefix(),
        &tokens[..checkpoint.completed_tokens()]
    );
    let expected = suffix
        .iter()
        .map(|span| fixture.execute(&source, Arc::clone(&tokens), span.clone()))
        .collect::<Vec<_>>();
    expected
        .last()
        .unwrap()
        .assert_state_changed(previous.as_ref().unwrap(), "suffix");
    source.try_complete().unwrap();
    drop(source);
    // The source has overwritten its boundary values/appended KV, then ended.
    // The expected suffix is a host snapshot; only the independently owned
    // checkpoint can initialize this fresh target after its source is gone.
    fixture.restore(&restored, &checkpoint, Arc::clone(&tokens));
    for (span, expected) in suffix.iter().zip(expected) {
        expected.assert_same(
            &fixture.execute(&cold, Arc::clone(&tokens), span.clone()),
            "cold suffix",
        );
        expected.assert_same(
            &fixture.execute(&restored, Arc::clone(&tokens), span.clone()),
            "restored suffix",
        );
    }
    drop(checkpoint);
    cold.try_complete().unwrap();
    restored.try_complete().unwrap();
    drop((cold, restored));
    // Change a real token at the same prefix frontier: nonzero constant state
    // would otherwise make a continuation comparison vacuous.
    let mut changed = tokens.to_vec();
    let index = prefix.last().unwrap().end - 1;
    changed[index] = (changed[index] + 1) % 32;
    let changed: Arc<[u32]> = changed.into();
    let control = fixture.admit("changed-prefix", Arc::clone(&changed));
    let mut changed_observation = None;
    for span in prefix {
        changed_observation = Some(fixture.execute(&control, Arc::clone(&changed), span.clone()));
    }
    changed_observation
        .unwrap()
        .assert_state_changed(previous.as_ref().unwrap(), "changed prefix");
    control.try_complete().unwrap();
}

pub fn verify_eager_numerical_failure() {
    let fixture = Fixture::with_execution_options(AttentionKind::CausalInt8, false, Some(31));
    let tokens: Arc<[u32]> = Arc::from([3, 4]);
    let source = fixture.admit("eager-baseline", Arc::clone(&tokens));
    let expected = fixture.execute(&source, Arc::clone(&tokens), 0..1);
    source.try_abort_if_quiescent().unwrap();
    drop(source);
    let bad_tokens: Arc<[u32]> = Arc::from([31, 4]);
    let bad = fixture.admit("eager-nonfinite", Arc::clone(&bad_tokens));
    fixture.execute_numerical_failure(&bad, bad_tokens, 0..1, false);
    drop(bad);
    let good = fixture.admit("eager-after-failure", Arc::clone(&tokens));
    expected.assert_same(&fixture.execute(&good, tokens, 0..1), "eager after failure");
    good.try_complete().unwrap();
}
