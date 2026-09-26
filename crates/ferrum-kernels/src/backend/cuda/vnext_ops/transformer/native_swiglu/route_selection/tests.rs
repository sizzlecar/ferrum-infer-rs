use super::*;
use crate::gguf_blocks::GgufBlockFormat;
use ferrum_interfaces::vnext::{DeviceCommandPhase, WeightId};

fn matrices() -> ([weights::MatrixPart; 2], [weights::MatrixPart; 1]) {
    let part = |id: &str, format, rows, columns, output_offset| weights::MatrixPart {
        component_id: WeightId::new(id).unwrap(),
        format: weights::MatrixFormat::Block(format),
        rows,
        columns,
        output_offset,
        transform: None,
        signs_region: None,
    };
    (
        [
            part("fixture.gate", GgufBlockFormat::Q4K, 768, 512, 0),
            part("fixture.up", GgufBlockFormat::Q4K, 768, 512, 768),
        ],
        [part("fixture.down", GgufBlockFormat::Q6K, 512, 768, 0)],
    )
}

#[test]
fn stream_mmq_cost_selection_uses_physical_rows_and_counts_pack_fixup_down() {
    let (gate, down) = matrices();
    // Four owners with two rows each and eight decode owners share the physical B8 path.
    for participants in [1, 4, 8] {
        let selected = select(
            &gate,
            &down,
            512,
            768,
            8,
            participants,
            true,
            true,
            Arithmetic::StreamMmq,
        )
        .unwrap();
        assert!(selected.mmq_hit);
        assert_eq!(selected.packed_rows, Some(8));
        assert_eq!(selected.launches, 1);
        let command = &selected.command;
        assert_eq!(
            command.native_operation(),
            "vnext_native_swiglu_stream_mmq_hit"
        );
        assert_eq!(command.phase(), DeviceCommandPhase::Compute);
        assert_eq!(command.participant_count(), participants);
        assert_eq!(command.token_count(), 8);
        assert_eq!(command.compute_dispatch_count(), 7);
        assert_eq!(command.transfer_command_count(), 0);
        assert_eq!(
            command.batching(),
            if participants == 1 {
                DeviceBatchingForm::Scalar
            } else {
                DeviceBatchingForm::Packed
            }
        );
    }
}

#[test]
fn stream_mmq_cost_selection_keeps_unpacked_participant_rows_strict() {
    let (gate, down) = matrices();
    for (input, output) in [(false, true), (true, false), (false, false)] {
        let selected = select(
            &gate,
            &down,
            512,
            768,
            8,
            8,
            input,
            output,
            Arithmetic::StreamMmq,
        )
        .unwrap();
        assert!(!selected.mmq_hit);
        assert_eq!(selected.packed_rows, None);
        assert_eq!(selected.launches, 8);
        assert_eq!(selected.command.compute_dispatch_count(), 32);
        assert_eq!(
            selected.command.batching(),
            DeviceBatchingForm::ParticipantLoop
        );
        assert_eq!(
            selected.command.native_operation(),
            "vnext_native_swiglu_stream_mmq_strict_fallback"
        );
    }
}

#[test]
fn stream_mmq_cost_selection_keeps_prefill_and_ineligible_weights_strict() {
    let (gate, down) = matrices();
    for tokens in [1, 4, 7, 9, 128] {
        let selected = select(
            &gate,
            &down,
            512,
            768,
            tokens,
            1,
            true,
            true,
            Arithmetic::StreamMmq,
        )
        .unwrap();
        assert!(!selected.mmq_hit);
        assert_eq!(selected.command.compute_dispatch_count(), 4);
    }
    for index in 0..2 {
        let mut wrong = gate.clone();
        wrong[index].format = weights::MatrixFormat::Block(GgufBlockFormat::Q5K);
        let selected = select(
            &wrong,
            &down,
            512,
            768,
            8,
            8,
            true,
            true,
            Arithmetic::StreamMmq,
        )
        .unwrap();
        assert!(!selected.mmq_hit);
        assert_eq!(selected.command.compute_dispatch_count(), 4);
        wrong = gate.clone();
        wrong[index].output_offset += 1;
        assert!(
            !select(
                &wrong,
                &down,
                512,
                768,
                8,
                8,
                true,
                true,
                Arithmetic::StreamMmq
            )
            .unwrap()
            .mmq_hit
        );
        wrong = gate.clone();
        wrong[index].signs_region = Some(0);
        assert!(
            !select(
                &wrong,
                &down,
                512,
                768,
                8,
                8,
                true,
                true,
                Arithmetic::StreamMmq
            )
            .unwrap()
            .mmq_hit
        );
    }
}

#[test]
fn swiglu_cost_selection_preserves_existing_strict_and_full_q8_inventory() {
    let (gate, down) = matrices();
    for (arithmetic, per_launch) in [(Arithmetic::Strict, 4), (Arithmetic::Q8, 6)] {
        for packed in [false, true] {
            let selected =
                select(&gate, &down, 512, 768, 8, 4, packed, packed, arithmetic).unwrap();
            assert!(!selected.mmq_hit);
            assert_eq!(selected.command.native_operation(), "vnext_native_swiglu");
            assert_eq!(
                selected.command.compute_dispatch_count(),
                per_launch * if packed { 1 } else { 4 }
            );
        }
    }
}

#[test]
fn swiglu_cost_selection_rejects_invalid_extents_and_preserves_large_fallback() {
    let (gate, down) = matrices();
    for (tokens, owners) in [(0, 1), (8, 0), (4, 8), (u64::MAX, 1)] {
        assert!(select(
            &gate,
            &down,
            512,
            768,
            tokens,
            owners,
            true,
            true,
            Arithmetic::StreamMmq
        )
        .is_err());
    }
    assert!(select(
        &gate,
        &down,
        u64::MAX,
        768,
        8,
        8,
        true,
        true,
        Arithmetic::StreamMmq
    )
    .is_err());
    let selected = select(
        &gate,
        &down,
        512,
        768,
        u64::from(u16::MAX) + 1,
        2,
        true,
        true,
        Arithmetic::StreamMmq,
    )
    .unwrap();
    assert!(!selected.mmq_hit);
    assert_eq!(selected.packed_rows, None);
    assert_eq!(selected.command.compute_dispatch_count(), 8);
}

#[test]
fn residual2_m2to8_whole_scope_includes_small_prefill_and_preserves_leaf_math() {
    let (gate, mut down) = matrices();
    for q4 in [true, false] {
        down[0].format = weights::MatrixFormat::Block(if q4 {
            GgufBlockFormat::Q4K
        } else {
            GgufBlockFormat::Q6K
        });
        for whole in 1..=9_u64 {
            // One owner with a multi-token chunk, one token per owner, and mixed
            // [1, whole-1] spans share the same whole-invocation math.
            for participants in [1, whole as u32, if whole > 1 { 2 } else { 1 }] {
                for (input_packed, output_packed) in
                    [(true, true), (true, false), (false, true), (false, false)]
                {
                    let new = select(
                        &gate,
                        &down,
                        512,
                        768,
                        whole,
                        participants,
                        input_packed,
                        output_packed,
                        Arithmetic::Residual2M2To8,
                    )
                    .unwrap();
                    let hit = (2..=8).contains(&whole);
                    assert_eq!(new.mmq_hit, hit);
                    assert_eq!(new.mmq_down_hit, hit && q4);
                    assert_eq!(
                        new.launches,
                        if input_packed && output_packed {
                            1
                        } else {
                            participants.into()
                        }
                    );
                    if hit {
                        assert_eq!(
                            new.command.compute_dispatch_count(),
                            new.launches * if q4 { 9 } else { 7 }
                        );
                    }
                }
            }
        }
    }
    // An unsupported gate encoding or rotated input never acquires residual
    // math merely because the whole row count is in range.
    for index in 0..gate.len() {
        let mut bad = gate.clone();
        bad[index].format = weights::MatrixFormat::Block(GgufBlockFormat::Q5K);
        assert!(
            !select(
                &bad,
                &down,
                512,
                768,
                3,
                3,
                true,
                true,
                Arithmetic::Residual2M2To8
            )
            .unwrap()
            .mmq_hit
        );
        let mut bad = gate.clone();
        bad[index].signs_region = Some(0);
        assert!(
            !select(
                &bad,
                &down,
                512,
                768,
                3,
                1,
                false,
                false,
                Arithmetic::Residual2M2To8
            )
            .unwrap()
            .mmq_hit
        );
    }
}
