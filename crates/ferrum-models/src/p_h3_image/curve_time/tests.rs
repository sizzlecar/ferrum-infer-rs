use super::*;
use serde_json::json;
use PH3ImageCurveLerpArithmetic::{FusedF32, SeparateF32};

#[test]
fn scalar_hand_computed_interpolation_keeps_order_endpoints_and_finite_clamping() {
    let curve = PH3ImageCurveTimeEmbedding::new(3, 2).unwrap();
    let table = [0., 10., 4., 6., 20., -4.];
    let times = [0., 0.125, 0.25, 0.375, 0.5, 0.75, 1., -1., 2.];
    let expected = [
        0., 10., 1., 9., 2., 8., 3., 7., 4., 6., 12., 1., 20., -4., 0., 10., 20., -4.,
    ];
    for arithmetic in [SeparateF32, FusedF32] {
        let mut output = [99.; 18];
        curve
            .write_f32(&table, &times, arithmetic, &mut output)
            .unwrap();
        assert_eq!(output, expected);
    }
    assert_eq!(curve.interval(0.).unwrap(), (0, 0.));
    assert_eq!(curve.interval(1.).unwrap(), (1, 1.));
}

#[test]
fn fused_and_separate_profiles_expose_real_f32_contraction_difference() {
    let curve = PH3ImageCurveTimeEmbedding::new(2, 1).unwrap();
    let table = [-1., 3.0_f32 - 4.0 * f32::EPSILON];
    let time = (1.0_f32 + f32::EPSILON) * 0.25;
    let mut separate = [9.];
    let mut fused = [9.];
    curve
        .write_f32(&table, &[time], SeparateF32, &mut separate)
        .unwrap();
    curve
        .write_f32(&table, &[time], FusedF32, &mut fused)
        .unwrap();
    assert_eq!(separate, [0.]);
    assert_eq!(fused, [-1.0_f32 / (1_u64 << 46) as f32]);
    assert_ne!(
        SeparateF32.numerical_profile_id(),
        FusedF32.numerical_profile_id()
    );
}

#[test]
fn checkpoint_header_declares_actual_curve_geometry_not_dense_time_embedding_width() {
    let header = json!({"adaln_t_table":{"dtype":"F32","shape":[4097,16],"data_offsets":[0,262208]},
        "blocks.0.adaln_proj.linear.weight":{"dtype":"F32","shape":[96768,16],"data_offsets":[262208,6455360]}});
    let bytes = serde_json::to_vec(&header).unwrap();
    let curve = PH3ImageCurveTimeEmbedding::from_safetensors_header(&bytes, bytes.len()).unwrap();
    assert_eq!(
        (
            curve.grid_points(),
            curve.basis_dim(),
            curve.table_elements()
        ),
        (4097, 16, 65552)
    );
    assert_eq!(curve.interval(1.).unwrap(), (4095, 1.));
    assert_eq!(curve.interval(1.0 / 4096.).unwrap(), (1, 0.));
    assert_eq!(curve.interval(0.5).unwrap(), (2048, 0.));
    assert_eq!(curve.interval(0.5 + 1.0 / 8192.).unwrap(), (2048, 0.5));
    assert!(PH3ImageCurveTimeEmbedding::from_safetensors_header(&bytes, bytes.len() - 1).is_err());
}

#[test]
fn nonfinite_tables_times_and_late_result_overflow_fail_before_writes() {
    let curve = PH3ImageCurveTimeEmbedding::new(2, 1).unwrap();
    for (table, times) in [
        ([1., f32::NAN], [0., 0.5]),
        ([1., 2.], [0., f32::INFINITY]),
        ([-f32::MAX, f32::MAX], [0., 0.5]),
    ] {
        let mut output = [f32::from_bits(0x7fc0_0001); 2];
        let before = output.map(f32::to_bits);
        assert!(curve
            .write_f32(&table, &times, SeparateF32, &mut output)
            .is_err());
        assert_eq!(output.map(f32::to_bits), before);
    }
    let mut output = [7.; 3];
    assert!(curve
        .write_f32(&[0., 1.], &[0., 0.5], SeparateF32, &mut output)
        .is_err());
    assert!(curve
        .write_f32(&[0.], &[0., 0.5, 1.], SeparateF32, &mut output)
        .is_err());
    assert_eq!(output, [7.; 3]);
}

#[test]
fn malformed_table_headers_and_unrepresentable_index_domains_fail_closed() {
    for (dtype, shape, offsets) in [
        ("BF16", [4097, 16], [0, 262208]),
        ("F32", [1, 16], [0, 64]),
        ("F32", [4097, 0], [0, 0]),
        ("F32", [4097, 16], [1, 262208]),
        ("F32", [4097, 16], [5, 4]),
    ] {
        let header = serde_json::to_vec(
            &json!({"adaln_t_table":{"dtype":dtype,"shape":shape,"data_offsets":offsets}}),
        )
        .unwrap();
        assert!(
            PH3ImageCurveTimeEmbedding::from_safetensors_header(&header, header.len()).is_err()
        );
    }
    let duplicate=br#"{"adaln_t_table":{"dtype":"F32","shape":[2,1],"data_offsets":[0,8]},"adaln_t_table":{"dtype":"F32","shape":[2,1],"data_offsets":[0,8]}}"#;
    assert!(
        PH3ImageCurveTimeEmbedding::from_safetensors_header(duplicate, duplicate.len()).is_err()
    );
    assert!(PH3ImageCurveTimeEmbedding::new((1 << 24) + 2, 1).is_err());
    assert!(PH3ImageCurveTimeEmbedding::new(2, usize::MAX).is_err());
    let curve = PH3ImageCurveTimeEmbedding::new(2, 2).unwrap();
    assert!(curve.output_elements(usize::MAX).is_err());
}
