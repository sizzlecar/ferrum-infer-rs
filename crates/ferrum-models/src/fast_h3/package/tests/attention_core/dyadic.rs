//! Independent constant-value subset of the complete attention core. Q=K=0
//! makes all selected valid positions equiprobable. Head and batch signs differ;
//! raw gates include -2 and 1, and projection rows select alternating heads.
//! This does not qualify general QK, two-level-P error or tie selection parity.
use super::*;
use half::bf16;

pub(super) fn fixture() -> (tempfile::TempDir, Arc<FastH3TransformerPackage>) {
    let config = config_with_attention(128, 2, 128, 128);
    let mut f = Fixture::with_attention(&config);
    let prefix = "minimax_h3.transformer_blocks.0.attn.to_out";
    let packed = &mut f
        .packed
        .get_mut(&format!("{prefix}::_nvfp4_weight"))
        .unwrap()
        .2;
    for n in 0..128 {
        for pair in 0..128 {
            packed[n * 128 + pair] = if pair / 64 == n % 2 { 0x11 } else { 0 };
        }
    }
    f.packed
        .get_mut(&format!("{prefix}::_nvfp4_weight_scale"))
        .unwrap()
        .2
        .fill(0x38);
    f.packed
        .get_mut(&format!("{prefix}::_nvfp4_alpha"))
        .unwrap()
        .2 = 0.25f32.to_le_bytes().to_vec();
    let (dir, package) = f.open_with_config(
        &config,
        FastH3WeightLimits {
            maximum_source_bytes: 8 << 20,
            ..limits()
        },
    );
    (dir, Arc::new(package.unwrap()))
}
pub(super) fn value(batch: usize, head: usize) -> f32 {
    if (batch + head) % 2 == 0 {
        6.
    } else {
        -6.
    }
}
pub(super) fn gate(row: usize) -> f32 {
    [-2., 0., 1.][row % 3]
}

pub(super) fn inputs(
    account: &HostMemoryAccount,
    c: &FastH3AttentionCoreConfig,
    layout: &FastH3VsaLayout,
) -> Vec<ManagedHostTensor> {
    let options = c.compile_options(capacities()).unwrap();
    [
        F::QUERY_ID,
        F::KEY_ID,
        F::VALUE_ID,
        F::GATE_ID,
        F::POOLED_QUERY_ID,
        F::POOLED_KEY_ID,
        F::POOLED_VALUE_ID,
    ]
    .into_iter()
    .map(|name| {
        let spec = &options.tensor_specs()[&id(name)];
        let mut bytes = account
            .allocate_zeroed(
                HostMemoryClass::InputPreprocess,
                spec.byte_len().unwrap() as usize,
            )
            .unwrap();
        if name == F::VALUE_ID {
            for (index, dst) in bytes.as_mut().chunks_exact_mut(2).enumerate() {
                let channel = index % 256;
                let row = index / 256 % layout.padded_rows();
                let batch = index / 256 / layout.padded_rows();
                if layout.tile_to_packed_rows()[row] != FAST_H3_VSA_PADDING_ROW {
                    dst.copy_from_slice(
                        &bf16::from_f32(value(batch, channel / 128))
                            .to_bits()
                            .to_le_bytes(),
                    );
                }
            }
        } else if name == F::GATE_ID {
            for (index, dst) in bytes.as_mut().chunks_exact_mut(2).enumerate() {
                let row = index / 256 % layout.packed().sequence_rows();
                dst.copy_from_slice(&bf16::from_f32(gate(row)).to_bits().to_le_bytes());
            }
        } else if name == F::POOLED_VALUE_ID {
            for (index, dst) in bytes.as_mut().chunks_exact_mut(4).enumerate() {
                let head = index / 128 / layout.tile_count() % 2;
                let batch = index / 128 / layout.tile_count() / 2;
                dst.copy_from_slice(&value(batch, head).to_le_bytes());
            }
        }
        ManagedHostTensor::new(
            id(name),
            ResolvedTensorSpec::new(
                spec.dimensions.clone(),
                spec.element_type,
                spec.layout.clone(),
            )
            .unwrap(),
            bytes.freeze(),
        )
        .unwrap()
    })
    .collect()
}

/// Six short reference rows, indexed by batch and gate phase. The source's
/// original packed weight and F32 alpha are decoded independently in FP64.
pub(super) fn expected(package: &FastH3TransformerPackage) -> Vec<Vec<u16>> {
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
    let scales = payload("::_nvfp4_weight_scale");
    assert!(scales.bytes().iter().all(|&b| b == 0x38));
    let alpha = f64::from(f32::from_le_bytes(
        payload("::_nvfp4_alpha").bytes().try_into().unwrap(),
    ));
    let codebook = [0.0f64, 0.5, 1., 1.5, 2., 3., 4., 6.];
    (0..6)
        .map(|case| {
            let batch = case / 3;
            let g = f64::from(gate(case % 3));
            (0..128)
                .map(|n| {
                    let mut sum = 0.;
                    for k in 0..256 {
                        let v = f64::from(value(batch, k / 128));
                        let merged = v + v * g;
                        // The two quantizers' nonzero groups lie exactly in this
                        // subset: amax6 or12, E4M3 scale1 or2, E2M1 magnitude6.
                        assert!([6., 12.].contains(&merged.abs()));
                        let nibble = (packed.bytes()[n * 128 + k / 2] >> ((k % 2) * 4)) & 15;
                        let weight = codebook[(nibble & 7) as usize]
                            * if nibble & 8 == 0 { 1. } else { -1. };
                        sum += merged * weight;
                    }
                    bf16::from_f32((sum * alpha) as f32).to_bits()
                })
                .collect()
        })
        .collect()
}
#[test]
fn core_oracle_separates_heads_batches_and_unclamped_gate_phases() {
    let (_dir, p) = fixture();
    let rows = expected(&p);
    for case in 0..6 {
        for n in 0..128 {
            let expected =
                96. * f64::from(value(case / 3, n % 2) / 6.) * (1. + f64::from(gate(case % 3)));
            assert_eq!(f64::from(bf16::from_bits(rows[case][n]).to_f32()), expected);
        }
    }
    assert_ne!(rows[0], rows[1]);
    assert_ne!(rows[1], rows[2]);
    assert_ne!(rows[0], rows[3]);
}
