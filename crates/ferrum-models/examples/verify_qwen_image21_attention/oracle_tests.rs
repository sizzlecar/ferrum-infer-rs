use super::*;
fn encode(xs: &[u16]) -> Vec<u8> {
    xs.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn choices(xs: &[u16]) -> Vec<Vec<u16>> {
    xs.iter().map(|&x| vec![x]).collect()
}
fn simple_inputs() -> (Vec<Vec<u16>>, Vec<Vec<u16>>, Vec<Vec<u16>>) {
    (
        choices(&[0x3f80, 0, 0x3f80, 0]),
        choices(&[0, 0, 0x3f80, 0]),
        choices(&[0, 0, 0x3f80, 0x3f80]),
    )
}
#[test]
fn two_key_analytic_softmax_and_causal_mask_have_distinct_bf16_results() {
    let (q, k, v) = simple_inputs();
    let full = attention_choices(&q, &k, &v, 2, 2, 1.0, Probe::FullPair).unwrap();
    // e/(1+e): rational e bounds 2.71828<e<2.71829 place the
    // result within the BF16 cell around 187/256, without calling exp.
    assert_eq!(full, vec![vec![0x3f3b]; 4]);
    let causal = attention_choices(&q, &k, &v, 2, 2, 1.0, Probe::CausalPair).unwrap();
    assert_eq!(&causal[..2], &[vec![0], vec![0]]);
    assert_eq!(&causal[2..], &full[2..]);
    assert!(!causal[0].contains(&full[0][0]));
    let wrong_scale = attention_choices(&q, &k, &v, 2, 2, 0.5, Probe::FullPair).unwrap();
    assert!(wrong_scale.iter().all(|set| !set.contains(&0x3f3b)));
}
#[test]
fn different_legal_input_branches_are_unioned_and_not_intersected() {
    let (mut q, k, v) = simple_inputs();
    q[0] = vec![0x3f80, 0x3f90];
    let joined = attention_choices(&q, &k, &v, 2, 2, 1.0, Probe::FullPair).unwrap();
    assert!(joined[0].contains(&0x3f3b));
    assert!(joined[0].contains(&0x3f41));
    for selected in [0x3f80, 0x3f90] {
        let mut branch = q.clone();
        branch[0] = vec![selected];
        let separate = attention_choices(&branch, &k, &v, 2, 2, 1.0, Probe::FullPair).unwrap();
        for (subset, superset) in separate.iter().zip(&joined) {
            assert!(subset.iter().all(|x| superset.contains(x)));
        }
    }
}
#[test]
fn single_key_identity_preserves_both_zero_signs_but_still_checks_score_domain() {
    let q = choices(&[0x3f80, 0]);
    let k = q.clone();
    let v = vec![vec![0, 0x8000], vec![0x4000]];
    assert_eq!(
        attention_choices(&q, &k, &v, 2, 2, 1.0, Probe::Single).unwrap(),
        v
    );
    let bad = choices(&[0x7f7f, 0]);
    assert!(attention_choices(&bad, &bad, &v, 2, 2, 1.0, Probe::Single).is_err());
    let nan = choices(&[0x7fc0, 0]);
    assert!(attention_choices(&nan, &k, &v, 2, 2, 1.0, Probe::Single).is_err());
}
#[test]
fn unsupported_exp_classification_and_shape_fail_before_candidate() {
    let (q, mut k, v) = simple_inputs();
    k[2] = vec![0x4380];
    assert!(attention_choices(&q, &k, &v, 2, 2, 1.0, Probe::FullPair).is_err());
    assert!(attention_choices(&q, &k, &v, 2, 2, 0.0, Probe::FullPair).is_err());
    assert!(attention_choices(&q[..2], &k, &v, 2, 2, 1.0, Probe::FullPair).is_err());
    assert!(arithmetic::Interval::new(-1.0, 1.0)
        .unwrap()
        .bf16_choices()
        .is_err());
    let (mut wide_q, k, v) = simple_inputs();
    wide_q[0] = vec![0, 0x3f80];
    let error = attention_choices(&wide_q, &k, &v, 2, 2, 1.0, Probe::FullPair).unwrap_err();
    assert!(error.to_string().contains("256 BF16 alternatives"));
}
fn tiny_probe(probe: Probe) -> Prepared {
    // hidden=4,D=2; Q/K column[0]=0.5, gamma=2, epsilon=0.75.
    // Every norm result is exactly 1. V column=[1,2,3,4]; out=I.
    let mut q = vec![0_u16; 16];
    for row in 0..4 {
        q[row * 4] = 0x3f00;
    }
    let q = encode(&q);
    let gamma = encode(&[0x4000; 2]);
    let mut v = vec![0; 16];
    for (row, x) in [0x3f80, 0x4000, 0x4040, 0x4080].into_iter().enumerate() {
        v[row * 4] = x;
    }
    let v = encode(&v);
    let mut w = vec![0; 16];
    for row in 0..4 {
        w[row * 4 + row] = 0x3f80;
    }
    let w = encode(&w);
    Prepared::prepare(4, 2, 0.75, 0.5, probe, [&q, &q, &gamma, &gamma, &v, &w]).unwrap()
}
#[test]
fn full_nine_port_comparison_checks_intermediates_and_only_declared_output_coordinates() {
    let p = tiny_probe(Probe::Single);
    assert_eq!(p.input(), encode(&[0x3f80, 0, 0, 0]));
    assert_eq!(p.expected[2], vec![vec![0x3f80]; 4]);
    let mut bytes = vec![
        encode(&[0x3f00; 4]),
        encode(&[0x3f00; 4]),
        encode(&[0x3f80; 4]),
        encode(&[0x3f80; 4]),
        encode(&[0x3f80; 4]),
        encode(&[0x3f80; 4]),
        encode(&[0x3f80, 0x4000, 0x4040, 0x4080]),
        encode(&[0x3f80, 0x4000, 0x4040, 0x4080]),
        encode(&[0x3f80, 0x4000, 0x4040, 0x4080]),
    ];
    let compare = |raw: &Vec<Vec<u8>>| {
        let records = Family::OUTPUT_IDS
            .iter()
            .zip(raw)
            .map(|(id, bytes)| (*id, bytes.as_slice()))
            .collect::<Vec<_>>();
        p.compare_records(&records).unwrap()
    };
    assert_eq!(compare(&bytes)["passed"], true);
    for port in 0..9 {
        let original = bytes[port][..2].to_vec();
        bytes[port][..2].copy_from_slice(&0x7fc0_u16.to_le_bytes());
        assert_eq!(compare(&bytes)["passed"], false, "port {port}");
        bytes[port][..2].copy_from_slice(&original);
    }
    let records = Family::OUTPUT_IDS
        .iter()
        .zip(&bytes)
        .map(|(id, bytes)| (*id, bytes.as_slice()))
        .collect::<Vec<_>>();
    assert!(p.compare_records(&records[..8]).is_err());
    let mut duplicate = records.clone();
    duplicate[8].0 = duplicate[7].0;
    assert!(p.compare_records(&duplicate).is_err());
}
#[test]
fn sampled_dense_includes_each_allowed_attention_branch_and_rejects_product_underflow() {
    let input = vec![vec![0x3f80, 0x3f81], vec![0x4000]];
    let weights = encode(&[0x3f80, 0, 0, 0x3f80]);
    let samples = dense::prepare(&input, 2, 1, &weights).unwrap();
    for x in [0x3f80, 0x3f81] {
        assert!(dense::compare(&samples, &encode(&[x, 0x4000]), 2).is_empty());
    }
    assert!(!dense::compare(&samples, &encode(&[0x3f90, 0x4000]), 2).is_empty());
    // The dense stage consumes BF16 choices, never the pre-conversion FP32
    // attention interval. Product underflow remains an explicit unresolved domain.
    let tiny = vec![vec![1], vec![0]];
    assert!(dense::prepare(&tiny, 2, 1, &weights).is_err());
}

#[test]
fn opposite_basis_rows_propagate_qk_signs_and_nontrivial_attention_before_comparison() {
    let full = tiny_probe(Probe::FullPair);
    assert_eq!(full.input(), encode(&[0x3f80, 0, 0, 0, 0xbf80, 0, 0, 0]));
    assert_eq!(
        full.expected[2],
        choices(&[0x3f80, 0x3f80, 0x3f80, 0x3f80, 0xbf80, 0xbf80, 0xbf80, 0xbf80])
    );
    // Scores are exactly +/-1. Therefore each mixed result is V*tanh(1),
    // and independent rational e bounds select these BF16 cells.
    assert_eq!(
        full.expected[7],
        choices(&[0x3f43, 0x3fc3, 0x4012, 0x4043, 0xbf43, 0xbfc3, 0xc012, 0xc043])
    );
    let causal = tiny_probe(Probe::CausalPair);
    assert_eq!(
        &causal.expected[7][..4],
        &choices(&[0x3f80, 0x4000, 0x4040, 0x4080])
    );
    assert_eq!(&causal.expected[7][4..], &full.expected[7][4..]);
    assert_eq!(full.frequencies().len(), 2 * 2 * 4);
}
