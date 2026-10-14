//! Exercise the binding-only encoder through real resident programs. These
//! profiles always retain MarkerV2 dependencies; hybrid small-row G32 would
//! bypass the MMQ replay hook and is covered separately by `hybrid`.
use super::*;

fn compare_waves(kind: AttentionKind, participants: u32, width: usize, prefill: bool) {
    let definition = || {
        if prefill {
            Family::prefill(kind, 2048)
        } else {
            Family::new(kind)
        }
    };
    let eager = Fixture::for_family(kind, false, participants, definition());
    let replay = Fixture::for_family(kind, true, participants, definition());
    let tokens: Vec<Arc<[u32]>> = (0..participants)
        .map(|participant| {
            (0..4 * width)
                .map(|position| ((position * 7 + participant as usize * 3 + 1) % 32) as u32)
                .collect()
        })
        .collect();
    let sessions = [&eager, &replay].map(|fixture| {
        tokens
            .iter()
            .enumerate()
            .map(|(i, t)| fixture.admit(&format!("binding-replay-{i}"), t.clone()))
            .collect::<Vec<_>>()
    });
    for wave in 0..4 {
        let range = wave * width..(wave + 1) * width;
        let expected =
            eager.execute_participants(&sessions[0], &tokens, range.clone(), Path::Eager);
        // The shared helper resolves the actual resident catalog, requires its
        // attention/FFN retained dependencies and exact seal, then submits with
        // ReplayedOnly. Different source positions change the real state/KV
        // bindings; neither eager fallback nor stale binding payload can pass.
        let path = if wave < 2 { Path::Warm } else { Path::Replay };
        let actual = replay.execute_participants(&sessions[1], &tokens, range.clone(), path);
        expected.assert_same(&actual);
        for participant in 0..participants {
            assert_eq!(
                actual.values[&(participant, "output".into())].len(),
                width * HIDDEN as usize * 4
            );
            if kind == AttentionKind::Causal {
                assert_eq!(
                    actual.values[&(participant, "state.kv".into())].len(),
                    range.end * 1024
                );
            }
        }
        actual.dump(kind, participants, range, path);
    }
    for group in sessions {
        for session in group {
            session.try_complete().unwrap();
        }
    }
}

#[test]
#[ignore = "requires the qualified MMQ prefill CUDA artifact and real GDN state"]
fn mmq_binding_only_prefill_replay_preserves_gdn_and_ffn_state() {
    // Large-row MMQ must remain a genuine resident path, including changing
    // source ranges. This does not force multi-token FP16 causal replay.
    for width in [33, 54] {
        compare_waves(AttentionKind::GatedDelta, 1, width, true);
    }
}

#[test]
#[ignore = "requires qualified MarkerV2 CUDA providers and real FP16 KV state"]
fn marker_binding_only_decode_replay_preserves_causal_kv_and_ffn_state() {
    // M8 attention uses MMVQ and M9 uses MMQ; FFN uses the actual role policy.
    // Both banks carry the same lifetime/authority obligation on every wave.
    for participants in [8, 9] {
        compare_waves(AttentionKind::Causal, participants, 1, false);
    }
}

#[test]
#[ignore = "requires exclusive CUDA, actual MMQ prefill and fresh GDN state allocations"]
fn mmq_prefill_cold_capture_and_fresh_request_replay_cost() {
    use ferrum_types::InvocationPreparationStrategy;
    use std::time::Instant;

    struct Preparation;
    impl InvocationPreparationSink for Preparation {
        fn record_preparation(&self, _: InvocationPreparationStats) {}
    }
    fn elapsed_ns(start: Instant) -> u64 {
        u64::try_from(start.elapsed().as_nanos()).unwrap()
    }
    fn execute(
        fixture: &Fixture,
        session: &Arc<SequenceSession<Runtime>>,
        tokens: &Arc<[u32]>,
        range: Range<usize>,
        path: Path,
    ) -> BatchObservation {
        fixture.execute_participant_ranges_with_preparation(
            &fixture.lane,
            &fixture.reaper,
            std::slice::from_ref(session),
            std::slice::from_ref(tokens),
            &[range],
            path,
            InvocationPreparationStrategy::Full,
            &Preparation,
            Some(16),
        )
    }
    fn state_pointers(observation: &BatchObservation) -> Vec<u64> {
        let rows = observation.binding_rows.as_ref().unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].len(), 16);
        let pointers = rows[0]
            .chunks_exact(8)
            .map(|bytes| u64::from_le_bytes(bytes.try_into().unwrap()))
            .collect::<Vec<_>>();
        assert!(pointers.iter().all(|pointer| *pointer != 0));
        pointers
    }

    // Two admitted slots keep the old request's real state alive while a new
    // request executes. Each submission still has exactly one participant.
    // Both arms use the same weights, initial state and tokens. This is a Core
    // mechanism diagnostic, not full-model or ShareGPT performance evidence.
    let width = 54;
    let kind = AttentionKind::GatedDelta;
    let arms = ["eager", "reusable"];
    let mut totals = [0_u64; 2];
    let fixtures = [false, true].map(|reusable| {
        let arm = usize::from(reusable);
        let start = Instant::now();
        let fixture = Fixture::for_family(kind, reusable, 2, Family::prefill(kind, 2048));
        let wall_ns = elapsed_ns(start);
        totals[arm] += wall_ns;
        println!(
            "{}",
            serde_json::json!({"kind":"prefill_capture_cost_stage",
            "arm":arms[arm],"stage":"fixture_cold_prepare","wall_ns":wall_ns,
            "inventory":fixture.lane.reusable_executable_preparation().unwrap()})
        );
        fixture
    });
    let tokens: Arc<[u32]> = (0..2 * width)
        .map(|position| ((position * 7 + 1) % 32) as u32)
        .collect();
    let sessions = fixtures.each_ref().map(|fixture| {
        let arm = usize::from(fixture.reusable_bucket.is_some());
        let start = Instant::now();
        let session = fixture.admit("prefill-capture-original", Arc::clone(&tokens));
        let wall_ns = elapsed_ns(start);
        totals[arm] += wall_ns;
        println!(
            "{}",
            serde_json::json!({"kind":"prefill_capture_cost_stage",
            "arm":arms[arm],"stage":"original_admission","wall_ns":wall_ns})
        );
        session
    });

    let mut old_pointers = [Vec::new(), Vec::new()];
    for wave in 0..2 {
        let observations = [0, 1].map(|arm| {
            let fixture = &fixtures[arm];
            let before = fixture.lane.reusable_executable_preparation().unwrap();
            let start = Instant::now();
            let observation = execute(
                fixture,
                &sessions[arm],
                &tokens,
                wave * width..(wave + 1) * width,
                if arm == 0 { Path::Eager } else { Path::Warm },
            );
            let wall_ns = elapsed_ns(start);
            totals[arm] += wall_ns;
            old_pointers[arm] = state_pointers(&observation);
            println!(
                "{}",
                serde_json::json!({"kind":"prefill_capture_cost_stage",
                "arm":arms[arm],"stage":if wave==0 {"first_warm"} else {"capture_and_upload"},
                "range":[wave*width,(wave+1)*width],"wall_ns":wall_ns,
                "inventory_before":before,
                "inventory_after":fixture.lane.reusable_executable_preparation().unwrap()})
            );
            observation
        });
        observations[0].assert_same(&observations[1]);
        assert_eq!(
            observations[1].values[&(0, "output".into())].len(),
            width * HIDDEN as usize * 4
        );
        observations[1].dump(kind, 1, wave * width..(wave + 1) * width, Path::Warm);
    }
    let captured = fixtures[1].lane.reusable_executable_preparation().unwrap();
    assert!(
        captured.captured_executables() > 0,
        "prefill must actually capture"
    );
    assert!(
        captured.uploaded_executables() > 0,
        "prefill must actually upload"
    );
    assert!(
        captured.resident_executables() > 0,
        "prefill must retain executable inventory"
    );
    assert_eq!(
        captured.captured_executables(),
        captured.uploaded_executables()
    );
    assert_eq!(
        captured.uploaded_executables(),
        captured.resident_executables()
    );

    let fresh_tokens: Arc<[u32]> = (0..width)
        .map(|position| ((position * 5 + 3) % 32) as u32)
        .collect();
    let fresh_sessions = fixtures.each_ref().map(|fixture| {
        let arm = usize::from(fixture.reusable_bucket.is_some());
        let start = Instant::now();
        let session = fixture.admit("prefill-capture-fresh", Arc::clone(&fresh_tokens));
        let wall_ns = elapsed_ns(start);
        totals[arm] += wall_ns;
        println!(
            "{}",
            serde_json::json!({"kind":"prefill_capture_cost_stage",
            "arm":arms[arm],"stage":"fresh_admission_old_owner_live","wall_ns":wall_ns})
        );
        session
    });
    let fresh = [0, 1].map(|arm| {
        let start = Instant::now();
        let observation = execute(
            &fixtures[arm],
            &fresh_sessions[arm],
            &fresh_tokens,
            0..width,
            if arm == 0 { Path::Eager } else { Path::Replay },
        );
        let wall_ns = elapsed_ns(start);
        totals[arm] += wall_ns;
        let pointers = state_pointers(&observation);
        assert!(
            pointers
                .iter()
                .all(|pointer| !old_pointers[arm].contains(pointer)),
            "fresh GDN conv and recurrent state must not reuse either live old state pointer"
        );
        println!(
            "{}",
            serde_json::json!({"kind":"prefill_capture_cost_stage",
            "arm":arms[arm],"stage":"fresh_request_complete","wall_ns":wall_ns,
            "path":if arm==0 {"Eager"} else {"ReplayedOnly"},
            "old_state_pointers":old_pointers[arm],"fresh_state_pointers":pointers,
            "inventory":fixtures[arm].lane.reusable_executable_preparation().unwrap()})
        );
        observation
    });
    fresh[0].assert_same(&fresh[1]);
    assert_eq!(
        fresh[1].values[&(0, "output".into())].len(),
        width * HIDDEN as usize * 4
    );
    assert_eq!(
        captured,
        fixtures[1].lane.reusable_executable_preparation().unwrap(),
        "fresh request must reuse the captured inventory without recapture or eviction"
    );
    fresh[1].dump(kind, 1, 0..width, Path::Replay);
    for arm in [0, 1] {
        let start = Instant::now();
        fresh_sessions[arm].try_complete().unwrap();
        sessions[arm].try_complete().unwrap();
        let wall_ns = elapsed_ns(start);
        totals[arm] += wall_ns;
        println!(
            "{}",
            serde_json::json!({"kind":"prefill_capture_cost_stage",
            "arm":arms[arm],"stage":"complete_both_requests","wall_ns":wall_ns})
        );
    }
    println!(
        "{}",
        serde_json::json!({"kind":"prefill_cold_capture_fresh_request_diagnostic",
        "prefill_tokens":width,"active_slots":2,"submitted_participants":1,
        "full_output_and_state_bits_equal":true,"fresh_physical_state_distinct":true,
        "fresh_replayed_only":true,"fresh_inventory_unchanged":true,
        "eager_total_wall_ns":totals[0],"reusable_total_wall_ns":totals[1],
        "cost_scope":"cold fixture preparation, both admissions, first warm, capture/upload, fresh replay, terminal completion; each execution includes full readback; independent oracle comparison and fixture teardown excluded",
        "limit":"single GDN+FFN Core shape, fixed arm order, no whole-model throughput claim"})
    );
}
