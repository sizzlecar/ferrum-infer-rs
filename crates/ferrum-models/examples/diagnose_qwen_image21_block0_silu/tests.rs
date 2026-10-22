use super::*;

fn bytes(values: &[u16]) -> Vec<u8> {
    values.iter().flat_map(|x| x.to_le_bytes()).collect()
}

#[test]
fn replay_preserves_nested_bf16_rounding_and_fp32_overflow() {
    let result = replay(0x3f80, 0x3f81);
    assert_eq!(result.silu.bits, 0x3f3b);
    assert_eq!(result.activation.bits, 0x3f3c);
    let without_boundary = bf16::from_f32((1.0_f32 / (1.0 + libm::expf(-1.0))) * (129.0 / 128.0));
    assert_eq!(without_boundary.to_bits(), 0x3f3d);
    assert_eq!(replay(0xc2b4, 0x3f80).activation.bits, 0x8000); // -90 -> -0
    assert_eq!(replay(0x8000, 0xbf80).activation.bits, 0); // -0 * -1 -> +0
    assert_eq!(replay(0xff80, 0x3f80).activation.class, "nan");
}

#[test]
fn scans_all_coordinates_and_separates_upstream_effect_from_local_residual() {
    let r_gate = bytes(&[0x4280, 0, 0x3f80, 0x4280]); // 64,0,1,64
    let c_gate = bytes(&[0x4288, 0, 0x3f80, 0x4280]); // 68,0,1,64
    let up = bytes(&[0x3f80; 4]);
    let r_out = bytes(&[0x4280, 0, 0x3f3b, 0x4280]);
    let c_out = bytes(&[0x4288, 0, 0x3f3c, 0x4288]);
    let mut points = Vec::new();
    let summary = scan(
        BoundaryView {
            shape: &[2, 2],
            gate: &r_gate,
            up: &up,
            activation: &r_out,
        },
        BoundaryView {
            shape: &[2, 2],
            gate: &c_gate,
            up: &up,
            activation: &c_out,
        },
        |p| {
            points.push((
                p.row,
                p.channel,
                p.candidate_minus_cpu_replay,
                p.cpu_replay_upstream_effect,
                p.reference_replay_minus_observed,
            ));
            Ok(())
        },
    )
    .unwrap();
    assert_eq!(summary.elements, 4);
    assert_eq!(summary.original_activation_failure_count, 2);
    assert_eq!(summary.reference_replay_bit_mismatch_count, 0);
    // The smaller local mismatch at [1,0] remains counted despite passing the
    // original threshold; emitting failures does not hide all-element evidence.
    assert_eq!(summary.candidate_replay_bit_mismatch_count, 2);
    assert_eq!(summary.changed_gate_or_up_bits_count, 1);
    assert_eq!(
        points,
        [
            (0, 0, Some(0.0), Some(4.0), Some(0.0)),
            (1, 1, Some(4.0), Some(0.0), Some(0.0))
        ]
    );
}

#[test]
fn nonfinite_observations_fail_original_criterion_and_serialize_without_nan_numbers() {
    let gate = bytes(&[0]);
    let up = bytes(&[0x3f80]);
    let out = bytes(&[0x7fc0]);
    let input = BoundaryView {
        shape: &[1, 1],
        gate: &gate,
        up: &up,
        activation: &out,
    };
    let summary = scan(input, input, |point| {
        let value = serde_json::to_value(point)?;
        assert!(value["original_absolute_error"].is_null());
        assert_eq!(value["candidate_observed_activation"]["class"], "nan");
        Ok(())
    })
    .unwrap();
    assert_eq!(summary.original_activation_failure_count, 1);
    assert_eq!(summary.original_activation_nonfinite_count, 1);
}

#[test]
fn rejects_shape_byte_overflow_and_wrong_shape_with_same_numel_before_emission() {
    let b = bytes(&[0; 4]);
    let a = BoundaryView {
        shape: &[2, 2],
        gate: &b,
        up: &b,
        activation: &b,
    };
    assert!(scan(
        a,
        BoundaryView {
            shape: &[1, 4],
            ..a
        },
        |_| panic!("invalid shape emitted a point")
    )
    .is_err());
    assert!(BoundaryView {
        shape: &[u64::MAX, 2],
        ..a
    }
    .validate()
    .is_err());
    assert!(BoundaryView {
        activation: &b[..2],
        ..a
    }
    .validate()
    .is_err());
    assert!(BoundaryView {
        shape: &[0, 4],
        ..a
    }
    .validate()
    .is_err());
}
