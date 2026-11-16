//! Independent exact-domain composition oracle. GEMM is decoded from actual
//! checkpoint E2M1 bytes and evaluated in FP64; no production provider or
//! quantizer is used. Unit-scale inputs are already exactly representable.
use super::super::attention_projection::dyadic as projection;
use super::*;
use half::bf16;
pub(super) fn fixture() -> (tempfile::TempDir, Arc<FastH3TransformerPackage>) {
    let config = config_with_attention_and_rotary(128, 2, 128, 192, 16);
    let mut f = projection::fixture_data(&config);
    for axis in ["q", "k"] {
        for bytes in f
            .dense
            .get_mut(&format!("transformer_blocks.0.attn.norm_{axis}.weight"))
            .unwrap()
            .2
            .chunks_exact_mut(2)
        {
            bytes.copy_from_slice(&bf16::ONE.to_bits().to_le_bytes());
        }
    }
    let (directory, p) = f.open_with_config(
        &config,
        FastH3WeightLimits {
            maximum_source_bytes: 8 << 20,
            ..limits()
        },
    );
    (directory, Arc::new(p.unwrap()))
}
pub(super) fn inputs(layout: &FastH3VsaLayout, batches: usize) -> [Vec<u16>; 3] {
    let rows = batches * layout.packed().sequence_rows();
    [
        projection::input(rows)
            .into_iter()
            .map(bf16::to_bits)
            .collect(),
        vec![bf16::ONE.to_bits(); rows * 96],
        vec![0; rows * 96],
    ]
}
pub(super) fn expected(
    package: &FastH3TransformerPackage,
    layout: &FastH3VsaLayout,
    batches: usize,
) -> BTreeMap<&'static str, Vec<u8>> {
    let packed = layout.packed().sequence_rows();
    let rows = layout.padded_rows();
    let width = 256;
    let raw = projection::expected(package, batches * packed);
    let mut expected = BTreeMap::new();
    for (kind, output, pooled) in [
        (
            FastH3AttentionProjection::Query,
            F::QUERY_ID,
            F::POOLED_QUERY_ID,
        ),
        (FastH3AttentionProjection::Key, F::KEY_ID, F::POOLED_KEY_ID),
        (
            FastH3AttentionProjection::Value,
            F::VALUE_ID,
            F::POOLED_VALUE_ID,
        ),
    ] {
        let raw = &raw[kind.output_id()];
        let mut tiled = vec![0u16; batches * rows * width];
        for b in 0..batches {
            for (tiled_row, &p) in layout.tile_to_packed_rows().iter().enumerate() {
                if p == FAST_H3_VSA_PADDING_ROW {
                    continue;
                }
                let row = (b * packed + p as usize) * width;
                for col in 0..width {
                    let bits = if kind == FastH3AttentionProjection::Value {
                        raw[row + col]
                    } else {
                        // Every head has constant nonzero elements in this
                        // fixture. FP64 RMS with epsilon and gamma=1 rounds to
                        // exactly +/-1, far from the BF16 rounding boundaries.
                        let x = f64::from(bf16::from_bits(raw[row + col]).to_f32());
                        assert!(raw[row..row + width].iter().all(|&v| v == raw[row]));
                        let normalized = x / (x * x + package.config().qk_norm_eps()).sqrt();
                        let y = bf16::from_f32(normalized as f32);
                        assert_eq!(y.to_f32(), x.signum() as f32);
                        y.to_bits()
                    };
                    tiled[(b * rows + tiled_row) * width + col] = bits;
                }
            }
        }
        let mut mean = Vec::new();
        for b in 0..batches {
            for h in 0..2 {
                for (tile, &count) in layout.valid_rows_per_tile().iter().enumerate() {
                    for d in 0..128 {
                        let sum: f64 = (0..usize::from(count))
                            .map(|r| {
                                f64::from(
                                    bf16::from_bits(
                                        tiled[((b * rows + tile * 128 + r) * width) + h * 128 + d],
                                    )
                                    .to_f32(),
                                )
                            })
                            .sum();
                        mean.extend_from_slice(&((sum / f64::from(count)) as f32).to_le_bytes());
                    }
                }
            }
        }
        expected.insert(
            output,
            tiled.into_iter().flat_map(u16::to_le_bytes).collect(),
        );
        expected.insert(pooled, mean);
    }
    expected.insert(
        F::GATE_ID,
        raw[FastH3AttentionProjection::CompressionGate.output_id()]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect(),
    );
    expected
}
#[test]
fn independent_attention_input_oracle_covers_real_head_rotary_shape_padding_and_nonzero_means() {
    let (_dir, p) = fixture();
    assert_eq!(
        (p.config().attention_head_dim(), p.config().rotary_dim()),
        (128, 96)
    );
    let l = layout();
    let y = expected(&p, &l, 2);
    assert_eq!(y.len(), 7);
    assert!(l.valid_rows_per_tile().iter().any(|&n| n < 128));
    assert!(y[F::POOLED_QUERY_ID]
        .chunks_exact(4)
        .any(|v| f32::from_le_bytes(v.try_into().unwrap()) != 0.));
    assert!(y[F::QUERY_ID].chunks_exact(2).any(|v| v == [0, 0]));
    let x = inputs(&l, 2);
    assert_eq!(x[0].len(), 2 * l.packed().sequence_rows() * 128);
    assert_eq!(x[1].len(), 2 * l.packed().sequence_rows() * 96);
}
