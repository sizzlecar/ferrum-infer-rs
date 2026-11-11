//! Real admitted token-wave uploads, submission fences and Host owners. The
//! contract backend observes transfer ranges; it does not simulate model math.
use super::*;

pub(super) fn host() -> HostMemoryAccount {
    let alignment = 64;
    let logical_bytes = 32_usize;
    let physical_bytes = logical_bytes.next_multiple_of(alignment) + alignment - 1;
    HostMemoryAccount::with_allocation_alignment(
        HostMemoryLimits::new(
            physical_bytes,
            HostMemoryClassLimits {
                input_preprocess_bytes: physical_bytes,
                transfer_staging_bytes: 0,
                output_bytes: 0,
                weight_backing_bytes: 0,
            },
        )
        .unwrap(),
        alignment,
    )
    .unwrap()
}

pub(super) fn source(host: &HostMemoryAccount, bytes: &[u8]) -> ManagedHostBytes {
    let mut buffer = host
        .allocate_zeroed(HostMemoryClass::InputPreprocess, bytes.len())
        .unwrap();
    buffer.as_mut().copy_from_slice(bytes);
    buffer.freeze()
}

#[test]
fn managed_range_checks_coordinates_layout_and_preserves_the_exact_owner() {
    let host = host();
    let bytes = source(&host, &[0, 0, 0, 0x80, 1, 0, 0x80, 0x3f]);
    let charged = host.backing_bytes(8).unwrap();
    let layout = HostTransferLayout::new(ElementType::F32, 2).unwrap();
    let upload =
        SubmissionWaveInputUpload::managed_range(id("node.main"), 3, 7, 12, layout, bytes.clone())
            .unwrap();
    assert_eq!(upload.participant_index(), 3);
    assert_eq!(upload.input_ordinal(), 7);
    assert_eq!(upload.logical_offset_bytes(), 12);
    assert!(upload.managed_bytes().unwrap().same_region(&bytes));
    let old =
        SubmissionWaveInputUpload::managed(id("node.main"), 7, layout, bytes.clone()).unwrap();
    let explicit =
        SubmissionWaveInputUpload::managed_range(id("node.main"), 0, 7, 0, layout, bytes.clone())
            .unwrap();
    assert_eq!(old, explicit);
    for offset in [1, u64::MAX - 3] {
        assert!(SubmissionWaveInputUpload::managed_range(
            id("node.main"),
            3,
            7,
            offset,
            layout,
            bytes.clone(),
        )
        .is_err());
        assert_eq!(host.usage().unwrap().allocated_bytes(), charged);
    }
    assert!(SubmissionWaveInputUpload::managed_range(
        id("node.main"),
        3,
        7,
        0,
        HostTransferLayout::new(ElementType::F32, 1).unwrap(),
        bytes.clone(),
    )
    .is_err());
    drop((bytes, old, explicit));
    assert_eq!(host.usage().unwrap().allocated_bytes(), charged);
    drop(upload);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}

#[derive(Clone, Copy)]
enum Outcome {
    Success,
    UnknownFence,
    EncodeFailure,
    SubmitFailure,
}

fn ordinary_case(managed: [bool; 4], outcome: Outcome) {
    let fixture = fixture();
    fixture
        .runtime_trace
        .lock()
        .unwrap()
        .managed_uploads_enabled = true;
    let resources = ["z", "a"].map(|name| {
        logical_resources(
            &fixture.plan_resources,
            &format!("run.managed.{name}"),
            &format!("request.managed.{name}"),
        )
    });
    let sessions = resources
        .iter()
        .map(|r| r.open_session().unwrap())
        .collect::<Vec<_>>();
    let batch = ExecutionBatchParticipants::new(sessions.clone()).unwrap();
    let lane = fixture.plan_resources.create_execution_lane().unwrap();
    let reaper = CompletionReaper::new();
    let step = step_for(&batch, &lane, vec![one_token_span(), one_token_span()]);
    let wave = prepare_wave(&fixture.plan_resources, &fixture.plan, &step);
    let active = sessions
        .iter()
        .map(|s| TrustedActiveSequenceBinding::from_session(s).unwrap())
        .collect::<Vec<_>>();
    let identity = OperationDispatch::bind_submission_wave_identity(
        &fixture.resolved,
        active.iter(),
        &wave,
        &lane,
    )
    .unwrap();
    let providers = fixture.registry.bind_plan(&fixture.resolved).unwrap();
    let host = host();
    let physical = host.backing_bytes(32).unwrap();
    let insufficient = host
        .reserve_scope(&[(HostMemoryClass::InputPreprocess, physical - 1)])
        .unwrap();
    assert!(insufficient
        .account()
        .allocate_zeroed(HostMemoryClass::InputPreprocess, 32)
        .is_err());
    insufficient.close().unwrap();
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    let scope = host
        .reserve_scope(&[(HostMemoryClass::InputPreprocess, physical)])
        .unwrap();
    let payload = (0..32).map(|i| i as u8).collect::<Vec<_>>();
    let bytes = source(&scope.account(), &payload);
    let uploads = managed
        .iter()
        .enumerate()
        .map(|(index, managed)| {
            let participant = (index / 2) as u32;
            let offset = (index % 2 * 8) as u64;
            let layout = HostTransferLayout::new(ElementType::F32, 2).unwrap();
            if *managed {
                SubmissionWaveInputUpload::managed_range(
                    id("node.main"),
                    participant,
                    0,
                    offset,
                    layout,
                    bytes.slice(index * 8..(index + 1) * 8).unwrap(),
                )
                .unwrap()
            } else {
                SubmissionWaveInputUpload::new(
                    id("node.main"),
                    participant,
                    0,
                    offset,
                    layout,
                    payload[index * 8..(index + 1) * 8].to_vec(),
                )
                .unwrap()
            }
        })
        .collect::<Vec<_>>();
    {
        let mut trace = fixture.runtime_trace.lock().unwrap();
        trace.uploaded_ranges.clear();
        match outcome {
            Outcome::UnknownFence => trace.fence_behavior = FenceBehavior::Indeterminate,
            Outcome::EncodeFailure => trace.fail_managed_upload_at = Some(1),
            Outcome::SubmitFailure => {
                trace.submit_behavior = SubmitBehavior::DefinitelyNotSubmitted
            }
            Outcome::Success => {}
        }
    }
    let dispatched = OperationDispatch::encode_and_submit_wave_with_inputs(
        providers.providers(),
        &fixture.resolved,
        &identity,
        active.iter(),
        DeviceTimingMode::Off,
        &uploads,
        wave,
        &lane,
        &reaper,
    );
    drop((uploads, bytes));
    scope.close().unwrap();
    match outcome {
        Outcome::EncodeFailure | Outcome::SubmitFailure => {
            match dispatched {
                Err(SubmissionWaveDispatchError::InputUpload(_))
                    if matches!(outcome, Outcome::EncodeFailure) => {}
                Err(SubmissionWaveDispatchError::DefinitelyNotSubmitted { retry, .. })
                    if matches!(outcome, Outcome::SubmitFailure) =>
                {
                    // This is the real wave retry owner, which intentionally
                    // keeps its Step alive until retried or explicitly dropped.
                    drop(retry);
                }
                other => panic!("expected a real pre-submit rejection, got {other:?}"),
            }
            assert_eq!(host.usage().unwrap().committed_bytes(), 0);
            assert_eq!(reaper.retained_count(), 0);
            assert_eq!(lane.in_flight_count(), 0);
            let retry = source(&host, &payload);
            drop(retry);
            step.try_abort().unwrap();
        }
        Outcome::Success | Outcome::UnknownFence => {
            let handle = dispatched.unwrap();
            assert_eq!(host.usage().unwrap().allocated_bytes(), physical);
            assert_eq!(reaper.retained_count(), 1);
            if matches!(outcome, Outcome::UnknownFence) {
                assert!(matches!(
                    handle.poll().unwrap(),
                    CompletionObservation::Indeterminate(_)
                ));
                assert_eq!(host.usage().unwrap().allocated_bytes(), physical);
                assert_eq!(lane.in_flight_count(), 1);
            }
            let records = fixture
                .runtime_trace
                .lock()
                .unwrap()
                .uploaded_ranges
                .clone();
            let mut expected = Vec::new();
            let mut index = 0;
            while index < managed.len() {
                let mut end = index + 1;
                if !managed[index] {
                    while end < managed.len() && !managed[end] {
                        end += 1;
                    }
                }
                expected.push((managed[index], payload[index * 8..end * 8].to_vec()));
                index = end;
            }
            assert_eq!(
                records
                    .iter()
                    .map(|(managed, _, bytes)| (*managed, bytes.clone()))
                    .collect::<Vec<_>>(),
                expected
            );
            for pair in records.windows(2) {
                assert_eq!(pair[0].1 + pair[0].2.len() as u64, pair[1].1);
            }
            assert!(host
                .allocate_zeroed(HostMemoryClass::InputPreprocess, 1)
                .is_err());
            if matches!(outcome, Outcome::UnknownFence) {
                // The reaper remains the only completion owner; dropping the
                // caller's observer cannot turn unknown device work into free
                // host capacity.
                drop(handle);
                assert_eq!(host.usage().unwrap().allocated_bytes(), physical);
                fixture.runtime_trace.lock().unwrap().fence_behavior = FenceBehavior::Succeeded;
                let swept = reaper.poll_bounded(1).unwrap();
                assert!(matches!(
                    swept.entries()[0].observation(),
                    CompletionSweepObservation::Observed(CompletionObservation::Terminal(_))
                ));
            } else {
                assert!(matches!(
                    handle.wait().unwrap(),
                    CompletionObservation::Terminal(_)
                ));
                drop(handle);
            }
            assert_eq!(host.usage().unwrap().committed_bytes(), 0);
            drop(source(&host, &payload));
            assert_eq!(reaper.retained_count(), 0);
            step.try_retire_normal().unwrap();
        }
    }
    drop(providers);
    drop(active);
    drop(batch);
    for session in &sessions {
        if matches!(outcome, Outcome::EncodeFailure | Outcome::SubmitFailure) {
            session.try_abort().unwrap();
        } else {
            session.try_complete().unwrap();
        }
    }
    drop((sessions, resources, reaper, lane));
    drop(fixture.registry);
    drop(fixture.impostor_registry);
    drop(fixture.runtime);
    assert!(matches!(
        PlanRuntimeResources::close(fixture.plan_resources),
        Ok(PlanRuntimeCloseOutcome::Closed(_))
    ));
}

#[test]
fn mixed_and_managed_packed_ranges_keep_each_owner_and_caller_coordinates() {
    // Both directions matter: a legacy-first run used to silently copy later
    // managed bytes, while a managed-first run used to reject valid adjacency.
    for managed in [
        [true, false, false, true],
        [false, true, false, true],
        [true; 4],
    ] {
        ordinary_case(managed, Outcome::Success);
    }
}

#[test]
fn unknown_input_fence_keeps_the_actual_host_backing_until_terminal_proof() {
    ordinary_case([true, false, false, true], Outcome::UnknownFence);
}

#[test]
fn late_managed_encode_failure_drops_prior_commands_and_refunds_the_scope() {
    ordinary_case([true, false, false, true], Outcome::EncodeFailure);
}

#[test]
fn definitely_not_submitted_managed_input_refunds_and_allows_new_allocation() {
    ordinary_case([true, false, false, true], Outcome::SubmitFailure);
}
