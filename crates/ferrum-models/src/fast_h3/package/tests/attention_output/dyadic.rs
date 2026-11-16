//! Complete-graph exact subset: raw gates include negative values and values
//! above one. FMA produces exact E2M1 groups with amax6 and unit scales.
//! Independent FP64 decoding/dots never call production NVFP4 arithmetic.
use super::*;
use half::bf16;
pub(super) fn fixture() -> (tempfile::TempDir, Arc<FastH3TransformerPackage>) {
    let config = config_with_attention(128, 2, 128, 192);
    let mut f = Fixture::with_attention(&config);
    let prefix = "minimax_h3.transformer_blocks.0.attn.to_out";
    f.packed
        .get_mut(&format!("{prefix}::_nvfp4_weight"))
        .unwrap()
        .2
        .fill(0x21);
    f.packed
        .get_mut(&format!("{prefix}::_nvfp4_alpha"))
        .unwrap()
        .2 = 0.25f32.to_le_bytes().to_vec();
    let (directory, package) = f.open_with_config(
        &config,
        FastH3WeightLimits {
            maximum_source_bytes: 8 << 20,
            ..limits()
        },
    );
    (directory, Arc::new(package.unwrap()))
}
pub(super) fn inputs(rows: usize) -> [Vec<u16>; 3] {
    let mut result = [Vec::new(), Vec::new(), Vec::new()];
    let group = [
        6., -6., 1., -1., 0.5, -0.5, 2., -2., 3., -3., 4., -4., 1.5, -1.5, 0., 6.,
    ];
    for i in 0..rows * 256 {
        let merged = group[i % 16] * if i / 256 % 2 == 0 { 1. } else { -1. };
        let pooled = 0.5;
        let gate = [-2., 2., 4., -4.][i % 4];
        for (dst, value) in result
            .iter_mut()
            .zip([merged - pooled * gate, pooled, gate])
        {
            dst.push(bf16::from_f32(value).to_bits());
        }
    }
    result
}
pub(super) fn expected(package: &FastH3TransformerPackage, rows: usize) -> Vec<u16> {
    let [x, pooled, gate] = inputs(rows);
    let decode = |bits| f64::from(bf16::from_bits(bits).to_f32());
    let merged: Vec<_> = x
        .iter()
        .zip(&pooled)
        .zip(&gate)
        .map(|((&x, &p), &g)| decode(x) + decode(p) * decode(g))
        .collect();
    let magnitudes = [0.0f64, 0.5, 1., 1.5, 2., 3., 4., 6.];
    for group in merged.chunks_exact(16) {
        assert_eq!(group.iter().map(|x| x.abs()).fold(0., f64::max), 6.);
        assert!(group.iter().all(|x| magnitudes.contains(&x.abs())));
    }
    let FastH3WeightStorage::Nvfp4 { components, .. } = &package
        .weight("transformer_blocks.0.attn.to_out.weight")
        .unwrap()
        .storage
    else {
        panic!("packed")
    };
    let payload = |suffix: &str| {
        package
            .component(
                components
                    .iter()
                    .find(|c| c.external_names[0].ends_with(suffix))
                    .unwrap(),
            )
            .unwrap()
    };
    let packed = payload("::_nvfp4_weight");
    assert!(payload("::_nvfp4_weight_scale")
        .bytes()
        .iter()
        .all(|&s| s == 0x38));
    let alpha = f64::from(f32::from_le_bytes(
        payload("::_nvfp4_alpha").bytes().try_into().unwrap(),
    ));
    let mut output = Vec::with_capacity(rows * 128);
    for row in 0..rows {
        for n in 0..128 {
            let mut dot = 0f64;
            for k in 0..256 {
                let code = (packed.bytes()[n * 128 + k / 2] >> ((k % 2) * 4)) & 15;
                let weight = magnitudes[(code & 7) as usize] * if code & 8 == 0 { 1. } else { -1. };
                dot += merged[row * 256 + k] * weight;
            }
            output.push(bf16::from_f32((dot * alpha) as f32).to_bits());
        }
    }
    output
}
#[test]
fn independent_full_graph_subset_preserves_raw_gate_and_non_square_projection() {
    let (_dir, package) = fixture();
    let out = expected(&package, 3);
    assert_eq!(out.len(), 384);
    for row in 0..3 {
        assert!(out[row * 128..(row + 1) * 128]
            .iter()
            .all(|&v| bf16::from_bits(v).to_f32() == if row % 2 == 0 { -12. } else { 12. }));
    }
    let [_, _, gate] = inputs(3);
    assert!(gate.iter().any(|&g| bf16::from_bits(g).to_f32() > 1.));
    assert!(gate.iter().any(|&g| bf16::from_bits(g).to_f32() < 0.));
}
