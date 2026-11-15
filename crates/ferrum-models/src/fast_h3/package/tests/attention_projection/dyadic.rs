//! Independent exact-subset fixture. Activation groups have amax=6 and unit
//! E4M3 scale; all E2M1 inputs, FP64 products and their sums are exact dyadics.
//! Neither the production decoder nor quantizer participates in the oracle.
use super::*;
use half::bf16;

pub(super) fn fixture() -> (tempfile::TempDir, Arc<FastH3TransformerPackage>) {
    let config = config_with_attention(128, 2, 128, 192);
    let fixture = fixture_data(&config);
    let (directory, package) = fixture.open_with_config(
        &config,
        FastH3WeightLimits {
            maximum_source_bytes: 8 << 20,
            ..limits()
        },
    );
    (directory, Arc::new(package.unwrap()))
}
pub(in crate::fast_h3::package::tests) fn fixture_data(
    config: &FastH3TransformerConfig,
) -> Fixture {
    let mut fixture = Fixture::with_attention(&config);
    for (kind, packed, alpha) in [
        (FastH3AttentionProjection::Query, 0x21, 0.25f32),
        (FastH3AttentionProjection::Key, 0xb2, 0.5),
        (FastH3AttentionProjection::Value, 0x43, 1.),
        (FastH3AttentionProjection::CompressionGate, 0xa5, 2.),
    ] {
        let prefix = format!("minimax_h3.transformer_blocks.0.attn.{}", kind.module());
        fixture
            .packed
            .get_mut(&format!("{prefix}::_nvfp4_weight"))
            .unwrap()
            .2
            .fill(packed);
        fixture
            .packed
            .get_mut(&format!("{prefix}::_nvfp4_alpha"))
            .unwrap()
            .2 = alpha.to_le_bytes().to_vec();
    }
    fixture
}
pub(in crate::fast_h3::package::tests) fn input(rows: usize) -> Vec<bf16> {
    let group = [
        6., -6., 1., -1., 0.5, -0.5, 2., -2., 3., -3., 4., -4., 1.5, -1.5, 0., 6.,
    ];
    (0..rows * 128)
        .map(|i| bf16::from_f32(group[i % 16] * if (i / 128) % 2 == 0 { 1. } else { -1. }))
        .collect()
}
pub(in crate::fast_h3::package::tests) fn expected(
    package: &FastH3TransformerPackage,
    rows: usize,
) -> BTreeMap<&'static str, Vec<u16>> {
    let x = input(rows);
    let magnitudes = [0.0f64, 0.5, 1., 1.5, 2., 3., 4., 6.];
    // All groups have true maximum6, and every value is exactly E2M1. Thus
    // unit input globalSF quantization has E4M3 scale1 and no clipping/rounding.
    for g in x.chunks_exact(16) {
        assert_eq!(g.iter().map(|v| v.to_f32().abs()).fold(0f32, f32::max), 6.);
        assert!(g
            .iter()
            .all(|v| magnitudes.contains(&f64::from(v.to_f32().abs()))));
    }
    FastH3AttentionProjection::ALL
        .into_iter()
        .map(|kind| {
            let weight = package
                .weight(&format!(
                    "transformer_blocks.0.attn.{}.weight",
                    kind.module()
                ))
                .unwrap();
            let FastH3WeightStorage::Nvfp4 { components, .. } = &weight.storage else {
                panic!("packed fixture")
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
            let scales = payload("::_nvfp4_weight_scale");
            assert!(scales.bytes().iter().all(|&b| b == 0x38)); // E4M3 1; swizzle immaterial.
            let alpha = payload("::_nvfp4_alpha");
            let alpha = f64::from(f32::from_le_bytes(alpha.bytes().try_into().unwrap()));
            let mut out = Vec::with_capacity(rows * 256);
            for row in 0..rows {
                for n in 0..256 {
                    let mut sum = 0f64;
                    for k in 0..128 {
                        let code = (packed.bytes()[n * 64 + k / 2] >> ((k % 2) * 4)) & 15;
                        let w = magnitudes[usize::from(code & 7)]
                            * if code & 8 == 0 { 1. } else { -1. };
                        sum += f64::from(x[row * 128 + k].to_f32()) * w;
                    }
                    out.push(bf16::from_f32((sum * alpha) as f32).to_bits());
                }
            }
            (kind.output_id(), out)
        })
        .collect()
}

#[test]
fn independent_attention_fixture_has_distinct_non_square_exact_projection_answers() {
    let (_directory, package) = fixture();
    let out = expected(&package, 3);
    let first: Vec<_> = FastH3AttentionProjection::ALL
        .into_iter()
        .map(|k| bf16::from_bits(out[k.output_id()][0]).to_f32())
        .collect();
    // Per 16-column group: sum even inputs=18, odd=-12; eight groups.
    assert_eq!(first, [-6., 144., 24., 1056.]);
    for values in out.values() {
        assert_eq!(values.len(), 3 * 256);
        assert_eq!(
            bf16::from_bits(values[256]).to_f32(),
            -bf16::from_bits(values[0]).to_f32()
        );
    }
}
