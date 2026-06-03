use ferrum_quantization::LoraLinearRef;

#[test]
fn lora_linear_ref_matches_manual_f32() {
    let base = vec![
        1.0, 2.0, -1.0, // out 0
        0.5, -0.5, 1.0, // out 1
    ];
    let a = vec![
        1.0, 0.0, 1.0, // rank 0
        0.0, 1.0, -1.0, // rank 1
    ];
    let b = vec![
        2.0, -1.0, // out 0
        0.5, 1.5, // out 1
    ];
    let linear = LoraLinearRef::new(base, a, b, 3, 2, 2, 4.0).expect("linear");
    let input = vec![
        1.0, 2.0, 3.0, // row 0
        -1.0, 0.5, 2.0, // row 1
    ];

    let out = linear.forward(&input, 2).expect("forward");

    let expected = vec![
        // row 0: base=[2, 2.5], A=[4, -1], B(A)=[9, 0.5], scale=2
        20.0,
        3.5,
        // row 1: base=[-2, 2.25], A=[1, -1.5], B(A)=[3.5, -1.75]
        5.0,
        -2.25,
    ];
    for (idx, (got, want)) in out.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - want).abs() <= 1e-5,
            "idx={idx} got={got} want={want} out={out:?}"
        );
    }
}

#[test]
fn lora_linear_ref_rejects_shape_mismatch() {
    let err = LoraLinearRef::new(vec![1.0], vec![1.0, 2.0], vec![1.0], 2, 1, 1, 1.0)
        .expect_err("shape mismatch should fail");
    assert!(err.to_string().contains("base weight shape mismatch"));
}
