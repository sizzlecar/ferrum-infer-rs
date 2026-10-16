//! Exact versus typed physical headroom, using real Core admission and the
//! same resident program path. Diagnostic timings include actual submission;
//! they are not CPU-exclusive durations or a serving benchmark.
use super::*;
use std::num::NonZeroU32;
use std::time::Instant;

pub(super) fn extend(
    fixture: &Fixture,
    session: &Arc<SequenceSession<Runtime>>,
    tokens: Arc<[u32]>,
    grouped: bool,
) -> Arc<SequenceBackingSnapshot<Runtime>> {
    let work = ResourceWorkShape::single(token_span(Arc::clone(&tokens), 0..tokens.len())).unwrap();
    let request =
        SequenceResourceExtensionRequest::new(work, AdmissionPressureAction::WaitForRelease)
            .unwrap();
    let request = if grouped {
        request.with_paged_growth_group(NonZeroU32::new(4).unwrap())
    } else {
        request
    };
    loop {
        match session.try_ensure_backing_covers(request.clone()).unwrap() {
            SequenceResourceExtensionDecision::Current(backing)
            | SequenceResourceExtensionDecision::Extended(backing) => return backing,
            SequenceResourceExtensionDecision::BackingDeferred(deferred) => {
                require_progress(deferred.maintain().unwrap());
            }
            SequenceResourceExtensionDecision::Deferred(reason) => {
                if fixture
                    .resources
                    .try_maintain_for_capacity_pressure(&reason)
                    .unwrap()
                    .is_none()
                {
                    require_progress(
                        fixture
                            .resources
                            .maintain_for_admission_deferred(&reason)
                            .unwrap(),
                    );
                }
            }
            SequenceResourceExtensionDecision::RetryRequired(_) => {
                panic!("extension must follow terminal retirement")
            }
            SequenceResourceExtensionDecision::PermanentRejected(reason) => {
                panic!("headroom fixture extension rejected: {reason:?}")
            }
        }
    }
}

fn kv_resource(fixture: &Fixture) -> ResourceId {
    fixture
        .compilation
        .executable()
        .execution_plan()
        .payload()
        .nodes()
        .iter()
        .find(|node| node.id().as_str() == "node.attention")
        .unwrap()
        .values()
        .iter()
        .find(|value| value.role() == ResolvedValueRole::Input && value.ordinal() == 8)
        .unwrap()
        .storage()
        .components()[0]
        .resource_id()
        .clone()
}

fn owned_pages(backing: &SequenceBackingSnapshot<Runtime>, resource: &ResourceId) -> u64 {
    let bytes: u64 = backing
        .backing_slices()
        .iter()
        .filter(|slice| slice.resource_id() == resource)
        .flat_map(|slice| slice.evidence().segments())
        .map(|segment| segment.length_bytes())
        .sum();
    assert!(bytes > 0 && bytes % 65_536 == 0);
    bytes / 65_536
}

fn check_preserved_bytes(previous: &[u8], current: &[u8], position: usize) {
    assert!(current.len() >= previous.len());
    // Paged F16 K and V layouts from the real causal state: [2, 8, 128]
    // elements per token and 16 tokens in each 64KiB page.
    let mut written = vec![false; current.len()];
    for kv in 0..2 {
        for head in 0..8 {
            for dim in 0..128 {
                let element = position / 16 * (2 * 8 * 128 * 16)
                    + if kv == 0 {
                        head * 128 * 16 + dim / 8 * 16 * 8 + position % 16 * 8 + dim % 8
                    } else {
                        8 * 128 * 16 + head * 128 * 16 + dim * 16 + position % 16
                    };
                written[element * 2..element * 2 + 2].fill(true);
            }
        }
    }
    for (offset, (&old, &new)) in previous.iter().zip(current).enumerate() {
        if !written[offset] {
            assert_eq!(old, new, "KV write escaped current token at byte {offset}");
        }
    }
}

#[test]
#[ignore = "requires exclusive CUDA, real Sequence headroom and complete resident replay"]
fn resident_decode_program_causal_four_page_headroom_matches_exact_through_exhaustion() {
    let fixtures = [0, 1].map(|_| {
        Fixture::for_family_with_causal_decode_preparation(
            AttentionKind::Causal,
            true,
            2,
            Family::new(AttentionKind::Causal)
                .with_maximum_tokens(128)
                .with_causal_sixteen_token_pages(),
            ProgramBindingUploadStrategy::CompactScatter,
            SegmentBindingOracleMode::Disabled,
            SegmentBindingOwnerViewMode::Indexed,
            ferrum_types::AttentionExecutionPolicy::NativeAdaptive,
            ferrum_types::CausalDecodePreparationMode::Packed,
        )
    });
    let tokens: Vec<Arc<[u32]>> = (0..2)
        .map(|p| (0..68).map(|t| ((t * 7 + p * 3 + 1) % 32) as u32).collect())
        .collect();
    for fixture in &fixtures {
        let state = fixture
            .states
            .iter()
            .find(|s| s.id.as_str() == "state.kv")
            .unwrap();
        assert_eq!(state.tensor.dimensions, [2, 8, 128]);
        assert_eq!(state.tensor.byte_len().unwrap(), 4096);
    }
    let sessions = fixtures.each_ref().map(|fixture| {
        tokens
            .iter()
            .enumerate()
            .map(|(p, t)| {
                fixture.admit_with_ceiling(&format!("headroom-{p}"), Arc::from(&t[..1]), 128)
            })
            .collect::<Vec<_>>()
    });
    let resources = fixtures.each_ref().map(kv_resource);
    let sinks = [Preparation::default(), Preparation::default()];
    let mut resident = [Resident::default(), Resident::default()];
    let mut previous_diagnostics: [Option<ExecuteDiagnostics>; 2] = [None, None];
    let mut previous_frames: [Option<Vec<ExecutionFrameId>>; 2] = [None, None];
    let mut previous_pages = [vec![1; 2], vec![1; 2]];
    let mut physical_appends = [0_u64; 2];
    let mut hot_net_ns = [0_u128; 2];
    let mut hot_waves = [0_u64; 2];
    let mut sealed_page_addresses: [Option<Vec<Vec<u64>>>; 2] = [None, None];
    let initial_tables = fixtures.each_ref().map(|f| {
        f._composition
            ._runtime
            .sequence_physical_binding_tables_compiled()
    });
    let initial_refreshes = fixtures
        .each_ref()
        .map(|f| f._composition._runtime.resident_decode_program_refreshes());

    for position in 0..68 {
        let ranges = vec![position..position + 1; 2];
        let mut observations = [None, None];
        let mut wave_hot = [false; 2];
        let mut wave_net_ns = [0_u128; 2];
        // Balance host ordering while keeping both arms on identical inputs.
        let order = if position % 2 == 0 { [0, 1] } else { [1, 0] };
        for arm in order {
            let fixture = &fixtures[arm];
            let runtime = &fixture._composition._runtime;
            let before_tables = runtime.sequence_physical_binding_tables_compiled();
            let before_refreshes = runtime.resident_decode_program_refreshes();
            let extension_started = Instant::now();
            let snapshots: Vec<_> = sessions[arm]
                .iter()
                .zip(&tokens)
                .map(|(session, tokens)| {
                    extend(fixture, session, Arc::from(&tokens[..=position]), arm == 1)
                })
                .collect();
            let extension_ns = extension_started.elapsed().as_nanos();
            let pages: Vec<_> = snapshots
                .iter()
                .map(|snapshot| {
                    assert_eq!(
                        snapshot.committed_tokens(),
                        position as u64 + 1,
                        "headroom must not fabricate executed or committed tokens"
                    );
                    owned_pages(snapshot, &resources[arm])
                })
                .collect();
            let expected_pages = if arm == 0 || position == 0 {
                (position + 1).div_ceil(16) as u64
            } else {
                ((position + 1).div_ceil(64) * 4) as u64
            };
            assert_eq!(
                pages,
                vec![expected_pages; 2],
                "fixture must acquire actual reserved capacity"
            );
            for (&old, &new) in previous_pages[arm].iter().zip(&pages) {
                assert!(new >= old);
                physical_appends[arm] += u64::from(new > old);
            }
            let same_capacity = previous_pages[arm] == pages;
            previous_pages[arm] = pages.clone();
            drop(snapshots);

            if position < 10 {
                let value = fixture.execute_participant_ranges_with_preparation(
                    &fixture.lane,
                    &fixture.reaper,
                    &sessions[arm],
                    &tokens,
                    &ranges,
                    Path::Warm,
                    InvocationPreparationStrategy::DecodeSegmentProgram,
                    &sinks[arm],
                    None,
                );
                value.assert_same(&value);
                observations[arm] = Some(value.values);
                continue;
            }
            let mut diagnostic = ExecuteDiagnostics::default();
            let (values, hot, frames) = execute_observed(
                fixture,
                &mut resident[arm],
                &sessions[arm],
                &tokens,
                &ranges,
                Some(&mut diagnostic),
            )
            .expect("headroom exhaustion must refresh the same resident topology");
            if [16, 32, 48, 64].contains(&position) {
                assert!(
                    hot,
                    "all physical page boundaries must use a real resident hit"
                );
            }
            if let Some(old) = &previous_frames[arm] {
                assert_ne!(
                    &frames, old,
                    "each successful wave requires fresh frame authority"
                );
            }
            previous_frames[arm] = Some(frames);
            if let Some(old) = &previous_diagnostics[arm] {
                for (participant, bytes) in &diagnostic.raw_kv {
                    check_preserved_bytes(&old.raw_kv[participant], bytes, position);
                }
            }
            for ((_, name), bytes) in &values {
                let finite = if name == "output" {
                    bytes
                        .chunks_exact(4)
                        .all(|v| f32::from_le_bytes(v.try_into().unwrap()).is_finite())
                } else {
                    bytes
                        .chunks_exact(2)
                        .all(|v| f16::from_le_bytes(v.try_into().unwrap()).is_finite())
                };
                assert!(
                    finite && !bytes.is_empty(),
                    "full result must be finite and nonempty"
                );
            }
            // A bootstrap may seal the resident packet without submitting it.
            // Its ordinary binding command guarantees only the live prefix.
            let observed_pages = if hot {
                expected_pages
            } else {
                (position + 1).div_ceil(16) as u64
            };
            let addresses: Vec<Vec<u64>> = diagnostic
                .binding_rows
                .values()
                .map(|row| {
                    row[24..24 + observed_pages as usize * 8]
                        .chunks_exact(8)
                        .map(|bytes| u64::from_le_bytes(bytes.try_into().unwrap()))
                        .collect()
                })
                .collect();
            assert!(
                addresses.iter().flatten().all(|address| *address != 0),
                "submitted binding must contain its observed physical page prefix"
            );
            if hot {
                if let Some(old) = &sealed_page_addresses[arm] {
                    for (old, new) in old.iter().zip(&addresses) {
                        assert_eq!(
                            old,
                            &new[..old.len()],
                            "append must preserve exact physical page identities"
                        );
                    }
                }
                sealed_page_addresses[arm] = Some(addresses.clone());
            }
            let table_delta = runtime.sequence_physical_binding_tables_compiled() - before_tables;
            let refresh_delta = runtime.resident_decode_program_refreshes() - before_refreshes;
            if hot && same_capacity {
                assert_eq!(
                    table_delta, 0,
                    "logical growth within owned capacity must reuse its physical table"
                );
                assert_eq!(
                    refresh_delta, 0,
                    "unchanged physical capacity must not refresh the sealed packet"
                );
            }
            if hot && !same_capacity {
                assert!(
                    table_delta > 0 && refresh_delta > 0,
                    "capacity exhaustion must perform real table append and backend refresh"
                );
            }
            let net_ns = extension_ns + diagnostic.prepare_and_submit_ns + diagnostic.retirement_ns;
            wave_hot[arm] = hot;
            wave_net_ns[arm] = net_ns;
            println!(
                "{}",
                serde_json::json!({
                    "kind":"resident_headroom_wave", "arm":if arm == 0 {"exact"} else {"group4"},
                    "position":position,"tokens":position+1,"participants":2,"resident_hot":hot,
                    "owned_pages":pages,"physical_table_compilations":table_delta,"backend_refreshes":refresh_delta,
                    "page_addresses":addresses,"extension_ns":extension_ns,
                    "prepare_and_submit_ns":diagnostic.prepare_and_submit_ns,
                    "terminal_readback_ns":diagnostic.completion_readback_ns,
                    "retirement_ns":diagnostic.retirement_ns,"host_transaction_ns":net_ns,
                    "timing_scope":"extend+prepare/submit+normal retirement; includes backend blocking if any; excludes terminal wait/readback, oracle, logging; not CPU-exclusive or serving throughput"
                })
            );
            previous_diagnostics[arm] = Some(diagnostic);
            observations[arm] = Some(values);
        }
        assert_eq!(
            observations[0], observations[1],
            "complete exact/group4 output and valid KV bits"
        );
        if wave_hot.iter().all(|hot| *hot) {
            for arm in 0..2 {
                hot_net_ns[arm] += wave_net_ns[arm];
                hot_waves[arm] += 1;
            }
        }
    }
    assert!(hot_waves.iter().all(|count| *count > 0));
    let tables = fixtures.each_ref().map(|f| {
        f._composition
            ._runtime
            .sequence_physical_binding_tables_compiled()
    });
    let refreshes = fixtures
        .each_ref()
        .map(|f| f._composition._runtime.resident_decode_program_refreshes());
    assert!(physical_appends[1] < physical_appends[0]);
    assert!(refreshes[1] - initial_refreshes[1] < refreshes[0] - initial_refreshes[0]);
    println!(
        "{}",
        serde_json::json!({"kind":"resident_headroom_complete",
            "tokens":68,"ceiling":128,"participants":2,"complete_output_valid_kv_bits_equal":true,
            "preserved_nonwritten_existing_kv_bytes":true,"physical_appends":physical_appends,
            "physical_table_compilations":[tables[0]-initial_tables[0],tables[1]-initial_tables[1]],
            "backend_refreshes":[refreshes[0]-initial_refreshes[0],refreshes[1]-initial_refreshes[1]],
            "paired_hot_waves":hot_waves,"paired_hot_host_transaction_ns":hot_net_ns,
            "performance_conclusion":"none; correctness fixture timing is diagnostic, not Off serving evidence"
        })
    );
    drop(resident);
    for group in sessions {
        for session in group {
            session.try_complete().unwrap();
        }
    }
}

#[test]
#[ignore = "requires exclusive CUDA and native V1/V2 recapture with real Sequence headroom"]
fn resident_decode_program_causal_four_page_headroom_preserves_native_partition_boundaries() {
    // Reuse the full ordinary/resident comparison and its exact pre-enqueue
    // miss, rollback, changed ProgramId and resumed-hot assertions. Physical
    // reservation must never make the old native partition topology eligible.
    compare_native_partition_boundary_with_headroom(512, true);
    compare_native_partition_boundary_with_headroom(1_024, true);
}
