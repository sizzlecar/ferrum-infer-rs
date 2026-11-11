use super::super::causal_attention::{self, CausalInputs, CausalScratch, CausalShape};
use super::*;

#[test]
fn f16_selection_copies_bits_for_zero_and_nonzero_predicates_and_bad_extent_never_writes() {
    let input = [0x8000u16, 0x7e35, 0x3555, 0x3c01];
    let visual = [0x7d55u16, 0x7d55, 0x0001, 0xfc00];
    let bytes = |v: &[u16]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>();
    let input = bytes(&input);
    let visual = bytes(&visual);
    let mut output = vec![0xcc; 8];
    super::super::elementwise::select_f16_rows(&input, &visual, &[0, 1], &mut output, 2).unwrap();
    assert_eq!(output, [&input[..4], &visual[4..]].concat());
    super::super::elementwise::select_f16_rows(&input, &visual, &[2, 255], &mut output, 2).unwrap();
    assert_eq!(output, visual);
    let original = output.clone();
    assert!(
        super::super::elementwise::select_f16_rows(&input, &visual, &[1], &mut output, 2).is_err()
    );
    assert_eq!(output, original);
}

#[test]
fn explicit_rope_rotates_active_prefix_but_keeps_physical_kv_write_position() {
    let shape = CausalShape {
        hidden: 4,
        query_heads: 1,
        kv_heads: 1,
        head_dim: 4,
        rope_dim: 2,
        maximum_context: 4,
        epsilon: 1e-5,
        theta: 10000.0,
        interleaved: false,
        gated: false,
    };
    let norm = encoded(&[1.0; 4], CpuFloat::F16);
    let raw = encoded(&[1.0, 2.0, 3.0, 4.0], CpuFloat::F16);
    let value = encoded(&[0.5, -0.5, 0.25, 1.0], CpuFloat::F16);
    let cos = encoded(&[0.0], CpuFloat::F16);
    let sin = encoded(&[1.0], CpuFloat::F16);
    let mut kv = vec![0; 4 * 4 * 2 * 2];
    let future = 3 * shape.state_bytes_per_token();
    kv[future..].fill(0x7f);
    let before = kv.clone();
    let mut query = vec![0; 8];
    let mut accumulated = vec![0; 16];
    let mut context = vec![0; 8];
    causal_attention::step_with_rope(
        shape,
        2,
        CausalInputs {
            query: &raw,
            key: &raw,
            value: &value,
            query_norm: &norm,
            key_norm: &norm,
        },
        kv.as_mut_slice(),
        CausalScratch {
            query: &mut query,
            accumulated: &mut accumulated,
            context: &mut context,
        },
        Some((&cos, &sin)),
    )
    .unwrap();
    let inverse = (7.5f32 + shape.epsilon).sqrt().recip();
    let expected = encoded(
        &[-2.0 * inverse, inverse, 3.0 * inverse, 4.0 * inverse],
        CpuFloat::F16,
    );
    let position = 2 * shape.state_bytes_per_token();
    assert_eq!(&kv[position..position + 8], &expected);
    assert_eq!(&kv[..position], &before[..position]);
    assert_eq!(&kv[future..], &before[future..]);
    assert_eq!(&query, &expected);
}
