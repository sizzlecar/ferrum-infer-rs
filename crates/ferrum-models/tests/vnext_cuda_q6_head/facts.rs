//! Actual admitted Q6 provider observations with the independent numerical oracle.
use super::*;
use checks::{oracle, sessions, tokens};
use runtime::{Fixture, Path};

fn inspect(
    work: &BoundSubmissionWork,
    device: &BoundSubmissionDeviceFacts,
    head: Head,
    participants: u32,
    source: std::ops::Range<usize>,
    replay: bool,
) {
    assert_eq!(device.submission(), work.submission());
    let DeviceExecutionFactsCoverage::ComputeCommands {
        described_compute_commands,
        unavailable_compute_commands,
    } = device.coverage()
    else {
        panic!("actual CUDA work must have command coverage")
    };
    assert_eq!(described_compute_commands, 1);
    assert!(
        unavailable_compute_commands > 0,
        "attention/embedding are explicitly outside this producer"
    );
    let commands = device.device().compute_commands();
    let described = commands
        .iter()
        .filter_map(|command| match &command.coverage {
            ProjectionCommandCoverage::Covered(projections) => Some((command, projections)),
            ProjectionCommandCoverage::Unavailable(_) => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(described.len(), 1);
    let (command, projections) = described[0];
    if replay {
        assert_eq!(command.path, DeviceExecutionPath::Replayed);
        assert!(command.replay_command_ordinal.is_some());
    }
    let packed = participants > 1 && source.len() == 1 && !head.transformed();
    let launch_count = if packed { 1 } else { participants as usize };
    assert_eq!(projections.len(), launch_count * head.parts());
    let mut expected_resolved = 0;
    for (ordinal, projection) in projections.iter().enumerate() {
        let launch = ordinal / head.parts();
        let part = ordinal % head.parts();
        let m = if packed { participants } else { 1 };
        let (n, output_offset) = if head == Head::Q6Split {
            if part == 0 {
                (64, 0)
            } else {
                (OUTPUTS as u32 - 64, 64)
            }
        } else {
            (OUTPUTS as u32, 0)
        };
        assert_eq!(projection.ordinal, ordinal as u32);
        assert_eq!(projection.role, DeviceProjectionRole::OutputHead);
        assert_eq!(
            projection.dimensions,
            ProjectionDimensions {
                m,
                k: family::HIDDEN as u32,
                n
            }
        );
        assert_eq!(
            projection.layout,
            ProjectionLayout {
                input_stride: family::HIDDEN as u32,
                output_stride: OUTPUTS as u32,
                output_offset,
                input_type: ElementType::F32,
                output_type: ElementType::F32,
            }
        );
        assert_eq!(
            projection.component.as_str(),
            if head == Head::Q6Split {
                if part == 0 {
                    "component.head.0"
                } else {
                    "component.head.1"
                }
            } else {
                "component.head"
            }
        );
        match &projection.arithmetic {
            ProjectionArithmetic::PackedQuantized {
                implementation,
                geometry,
            } => {
                assert!(head.quantized() && !head.transformed() && m <= 32);
                assert_eq!(
                    implementation.as_str(),
                    "ferrum.cuda.upstream-q6-f32-linear"
                );
                assert_eq!(geometry.padded_k, 512);
                assert_eq!(geometry.padded_n, n);
                assert_eq!(geometry.column_tile, 128);
                assert!(geometry.row_tile >= m);
                assert_eq!(geometry.threads, 256);
                assert_eq!(geometry.pack_abi, 1);
                assert!(geometry.blocks > 0 && geometry.workspace_bytes >= geometry.packed_bytes);
                assert_eq!(geometry.fixup_bytes > 0, geometry.fixup);
            }
            ProjectionArithmetic::Direct {
                implementation,
                row_tile,
                launch,
            } => {
                assert!(m > 32 || head.transformed() || !head.quantized());
                let tile = if m > 1 { 8 } else { 1 };
                assert_eq!(*row_tile, tile);
                assert_eq!(launch.grid, [n.div_ceil(4), m.div_ceil(tile), 1]);
                assert_eq!(launch.block, [128, 1, 1]);
                assert_eq!(launch.shared_bytes, 0);
                assert_eq!(
                    implementation.as_str(),
                    if m > 1 {
                        "vnext_gguf_linear_q6k_tiled_f32"
                    } else {
                        "vnext_gguf_linear_q6k_f32"
                    }
                );
            }
        }
        assert_eq!(
            projection.input_transforms.len(),
            usize::from(head.transformed())
        );
        if head.transformed() {
            let transform = &projection.input_transforms[0];
            assert_eq!(
                transform.implementation.as_str(),
                "vnext_gguf_hadamard_f32_f32"
            );
            assert_eq!(transform.spec.block_size.get(), 64);
            assert_eq!(
                transform.spec.application,
                HadamardApplication::BeforeMatmul {
                    input_permutation: None
                }
            );
            assert_eq!(
                transform.launch,
                ProjectionLaunchGeometry {
                    grid: [family::HIDDEN as u32 / 64, m, 1],
                    block: [256, 1, 1],
                    shared_bytes: 256,
                }
            );
        }
        let participant_ids = if packed {
            (0..participants).collect::<Vec<_>>()
        } else {
            vec![launch as u32]
        };
        assert_eq!(projection.rows.len(), participant_ids.len());
        let resolved = device
            .projection_rows()
            .iter()
            .find(|row| {
                row.submission_command_index == command.submission_command_index
                    && row.projection_ordinal == ordinal as u32
            })
            .expect("core resolves each actual projection against current work");
        assert_eq!(resolved.node_index, command.node_index.unwrap());
        assert_eq!(
            resolved.replay_command_ordinal,
            command.replay_command_ordinal
        );
        assert_eq!(resolved.rows.len(), participant_ids.len());
        for ((selector, resolved), participant) in projection
            .rows
            .iter()
            .zip(&resolved.rows)
            .zip(participant_ids)
        {
            let physical = if packed { participant } else { 0 };
            assert_eq!(selector.participant_index, participant);
            assert_eq!(selector.source, ParticipantTokenSelector::LastSourceToken);
            assert_eq!(selector.physical_rows, physical..physical + 1);
            assert_eq!(resolved.participant_index, participant);
            assert_eq!(
                resolved.source_token_range,
                (source.end - 1) as u64..source.end as u64
            );
            let packed_last =
                u64::from(participant) * source.len() as u64 + source.len() as u64 - 1;
            assert_eq!(resolved.packed_token_range, packed_last..packed_last + 1);
            assert_eq!(resolved.physical_rows, selector.physical_rows);
        }
        expected_resolved += 1;
    }
    assert_eq!(device.projection_rows().len(), expected_resolved);
}

pub fn geometry_and_rows() {
    for (head, rows, range) in [
        (Head::Q6, 4, 0..1),
        (Head::Q6Split, 4, 0..1),
        (Head::Q6, 4, 0..3),
        (Head::Q6, 33, 0..1),
        (Head::Q6Hadamard, 4, 0..1),
    ] {
        let f = Fixture::new(head, rows, None, false);
        let reference = Fixture::new(head, rows, None, false);
        let t = tokens(rows as usize, 4);
        let s = sessions(&f, "facts", &t);
        let rs = sessions(&reference, "off-reference", &t);
        let on = f.execute_with_facts(
            &s,
            &t,
            range.clone(),
            Path::Eager,
            ExecutionFactsCapture::On,
        );
        let off = reference.execute_with_facts(
            &rs,
            &t,
            range.clone(),
            Path::Eager,
            ExecutionFactsCapture::Off,
        );
        assert!(off.facts.is_none());
        assert_eq!(
            on.outputs, off.outputs,
            "observation does not alter actual arithmetic"
        );
        oracle(&f, &on.outputs, &t, range.end - 1);
        let (work, device) = on.facts.as_ref().unwrap();
        inspect(work, device, head, rows, range, false);
        // Keep pure terminal facts alive across actual retirement and release.
        f.close(s);
        reference.close(rs);
        assert!(!device.projection_rows().is_empty());
    }
}

pub fn replay() {
    for head in [Head::Q6Split, Head::Q6Hadamard] {
        let f = Fixture::new(head, 4, None, false);
        let reference = Fixture::new(head, 4, None, false);
        let t = tokens(4, 4);
        let s = sessions(&f, "captured", &t);
        let rs = sessions(&reference, "eager", &t);
        for (position, path) in [Path::Eager, Path::Warm, Path::Warm, Path::Replay]
            .into_iter()
            .enumerate()
        {
            let on = f.execute_with_facts(
                &s,
                &t,
                position..position + 1,
                path,
                ExecutionFactsCapture::On,
            );
            let expected = reference.execute_with_facts(
                &rs,
                &t,
                position..position + 1,
                Path::Eager,
                ExecutionFactsCapture::Off,
            );
            assert_eq!(on.outputs, expected.outputs);
            oracle(&f, &on.outputs, &t, position);
            let (work, device) = on.facts.as_ref().unwrap();
            inspect(
                work,
                device,
                head,
                4,
                position..position + 1,
                matches!(path, Path::Replay),
            );
        }
        f.close(s);
        reference.close(rs);
    }
}

pub fn off_capture() {
    let f = Fixture::new(Head::Q6, 4, None, false);
    let t = tokens(4, 4);
    let s = sessions(&f, "off-capture", &t);
    for position in 0..3 {
        let result = f.execute_with_facts(
            &s,
            &t,
            position..position + 1,
            Path::Warm,
            ExecutionFactsCapture::Off,
        );
        assert!(result.facts.is_none());
        oracle(&f, &result.outputs, &t, position);
    }
    let catalog_before = f.lane.reusable_execution_catalog().unwrap();
    let preparation_before = f.lane.reusable_executable_preparation().unwrap();
    let result = f.execute_with_facts(&s, &t, 3..4, Path::Replay, ExecutionFactsCapture::On);
    oracle(&f, &result.outputs, &t, 3);
    let (_, device) = result.facts.as_ref().unwrap();
    assert!(device.projection_rows().is_empty());
    assert!(device.device().compute_commands().iter().any(|command| {
        matches!(
            command.coverage,
            ProjectionCommandCoverage::Unavailable(
                DeviceExecutionFactsUnavailableReason::CachedTemplateUnavailable
            )
        )
    }));
    let catalog_after = f.lane.reusable_execution_catalog().unwrap();
    assert_eq!(catalog_before.epoch(), catalog_after.epoch());
    assert_eq!(
        catalog_before.programs(),
        catalog_after.programs(),
        "facts cannot alter executable qualification"
    );
    let preparation_after = f.lane.reusable_executable_preparation().unwrap();
    assert_eq!(
        preparation_before.captured_executables(),
        preparation_after.captured_executables()
    );
    assert_eq!(
        preparation_before.resident_executables(),
        preparation_after.resident_executables()
    );
    f.close(s);
}

/// Preserve the original admitted identities while a real retirement changes
/// physical rows. A fresh seven-session fixture would miss this transition.
pub fn survivor_compaction() {
    let f = Fixture::new(Head::Q6, 8, None, false);
    let reference = Fixture::new(Head::Q6, 8, None, false);
    let mut t = tokens(8, 8);
    let mut s = sessions(&f, "survivors", &t);
    let mut rs = sessions(&reference, "eager-survivors", &t);
    let mut expected_participants = s
        .iter()
        .map(|session| {
            BatchParticipantAuthority::new(
                session.sequence_authority(),
                session.request_authority(),
            )
        })
        .collect::<Vec<_>>();
    let target = expected_participants[5];
    assert_eq!(target.sequence_authority().sparse_id(), 5);
    assert_eq!(expected_participants[3].sequence_authority().sparse_id(), 3);

    for phase in 0..2 {
        if phase == 1 {
            // Close the actual original owners on both execution histories.
            // No re-admission or manually edited slot/generation is involved.
            for group in [&mut s, &mut rs] {
                let retired = group.remove(3);
                let weak = Arc::downgrade(&retired);
                retired.try_abort_if_quiescent().unwrap();
                drop(retired);
                assert!(weak.upgrade().is_none());
            }
            t.remove(3);
            expected_participants.remove(3);
        }
        for (step, path) in [Path::Eager, Path::Warm, Path::Warm, Path::Replay]
            .into_iter()
            .enumerate()
        {
            let position = phase * 4 + step;
            let observed = f.execute_with_facts(
                &s,
                &t,
                position..position + 1,
                path,
                ExecutionFactsCapture::On,
            );
            let expected = reference.execute_with_facts(
                &rs,
                &t,
                position..position + 1,
                Path::Eager,
                ExecutionFactsCapture::Off,
            );
            assert_eq!(
                observed.outputs, expected.outputs,
                "full eager/replay logits"
            );
            oracle(&f, &observed.outputs, &t, position);
            let (work, device) = observed.facts.as_ref().unwrap();
            inspect(
                work,
                device,
                Head::Q6,
                s.len() as u32,
                position..position + 1,
                matches!(path, Path::Replay),
            );
            let projection = &device.projection_rows()[0];
            let node = work
                .nodes()
                .iter()
                .find(|node| node.node_index() == projection.node_index)
                .unwrap();
            assert_eq!(
                node.participants()
                    .iter()
                    .map(BatchParticipantTokenRange::participant)
                    .collect::<Vec<_>>(),
                expected_participants,
                "current work must retain each original admission identity"
            );
            let physical = if phase == 0 { 5 } else { 4 };
            let row = &projection.rows[physical];
            assert_eq!(row.physical_rows, physical as u32..physical as u32 + 1);
            assert_eq!(
                node.participants()[row.participant_index as usize].participant(),
                target,
                "original slot5 becomes row4, without becoming a new participant"
            );
        }
    }
    f.close(s);
    reference.close(rs);
}
