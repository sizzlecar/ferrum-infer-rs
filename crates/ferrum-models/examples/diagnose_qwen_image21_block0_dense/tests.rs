use super::*;
use dyadic::{Dot, Exact};
fn bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|x| half::bf16::from_f32(*x).to_bits().to_le_bytes())
        .collect()
}

#[test]
fn exact_midpoint_side_is_independent_of_fp32_fold() {
    let d = Dot::prepare(
        &bytes(&[1., 1., 1.]),
        &bytes(&[1., 1. / 256., 2_f32.powi(-24)]),
    )
    .unwrap();
    assert_eq!(d.exact_nearest_bf16, 0x3f81);
    assert_eq!(d.sequential_bf16_bits, 0x3f80);
    let distance = dyadic::midpoint_distance(d.exact, 0x3f80, 0x3f81).unwrap();
    assert_eq!((distance.numerator, distance.exponent), (1, -24));
    assert!(d.compatible(0x3f80) && d.compatible(0x3f81));
    assert!(!d.compatible(0x4000));
}
#[test]
fn exact_cancellation_and_counterfactual_are_not_reduction_oracles() {
    let d = Dot::prepare(
        &bytes(&[1.; 4]),
        &bytes(&[16_777_216., 1., -16_777_216., 1.]),
    )
    .unwrap();
    assert_eq!(d.exact.approximate(), 2.0);
    assert_eq!(d.exact_nearest_bf16, 0x4000);
    assert_eq!(d.sequential_bf16_bits, 0x3f80);
    let other = Dot::prepare(
        &bytes(&[1., 2., 1., 1.]),
        &bytes(&[16_777_216., 1., -16_777_216., 1.]),
    )
    .unwrap();
    assert_eq!(other.exact.subtract(d.exact).unwrap().approximate(), 1.0);
    assert!(!dyadic::adjacent(0x3560, 0x35b0));
    assert!(dyadic::adjacent(0xc0a3, 0xc0a2));
}
#[test]
fn unresolved_domain_and_nonfinite_never_fall_back_to_float() {
    assert!(Dot::prepare(&bytes(&[1., 1.]), &[0x7f, 0x7f, 0x80, 0]).is_err()); // lattice span exceeds i128
    assert!(Dot::prepare(&bytes(&[1.]), &[1, 0]).is_err()); // below no-underflow lattice
    assert!(Dot::prepare(&bytes(&[1.]), &[0xc1, 0x7f]).is_err());
    assert!(Dot::prepare(&[], &[]).is_err());
    assert!(Dot::prepare(&bytes(&[1.]), &bytes(&[1., 2.])).is_err());
    assert!(Dot::prepare(&bytes(&[2.]), &[0x7f, 0x7f]).is_err());
}
#[test]
fn integer_bf16_ties_signed_zero_and_carry() {
    for (n, e, want) in [
        (1, -134, 0),
        (-1, -134, 0x8000),
        (3, -135, 1),
        (-3, -135, 0x8001),
        (257, -8, 0x3f80),
        (259, -8, 0x3f82),
        (511, -8, 0x4000),
        (255, 120, 0x7f7f),
    ] {
        assert_eq!(
            Exact {
                numerator: n,
                exponent: e
            }
            .round_bf16()
            .unwrap(),
            want
        );
    }
    assert!(Exact {
        numerator: 511,
        exponent: 119
    }
    .round_bf16()
    .is_err());
}

#[test]
fn artifact_replay_checks_hashes_and_exact_oi_row() -> Result<()> {
    use ferrum_models::qwen_image21::{
        QwenImage21Block0AttentionFamily as A, QwenImage21Block0Family as B,
    };
    let temp = tempfile::tempdir()?;
    let weights = bytes(&[1., 2., 3., 4., 5., 6.]);
    safetensors::tensor::serialize_to_file(
        [(
            B::WEIGHT_NAMES[0],
            safetensors::tensor::TensorView::new(safetensors::Dtype::BF16, vec![2, 3], &weights)?,
        )],
        &None,
        &temp.path().join("model.safetensors"),
    )?;
    let archive = SafetensorsArchive::open(temp.path())?;
    let x = bytes(&[1., 2., 3., -1., -2., -3.]);
    let y = bytes(&[14., 32., -14., -32.]);
    let tensor = |id: &str, file: &str, shape: &[u64], data: &[u8]| json!({"value_id":id,"file":file,"dtype":"BF16","shape":shape,"bytes":data.len(),"sha256":artifacts::sha256(data)});
    let identity = json!({"canonical_config":{},"config_file_sha256":"fixture","inputs":[tensor(B::INPUT_ID,"input.bf16",&[2,3],&x)],
        "weights":[{"tensor":B::WEIGHT_NAMES[0],"dtype":"BF16","shape":[2,3],"bytes":weights.len(),"sha256":artifacts::sha256(&weights)}]});
    for backend in ["reference", "cuda"] {
        let dir = temp.path().join(backend);
        std::fs::create_dir(&dir)?;
        std::fs::write(dir.join("input.bf16"), &x)?;
        std::fs::write(dir.join("output.bf16"), &y)?;
        std::fs::write(
            dir.join("manifest.json"),
            serde_json::to_vec(
                &json!({"schema":"ferrum.qwen_image21.block0-artifacts.v1","backend":backend,"identity":identity,
            "outputs":[tensor(B::MODULATED1_ID,"input.bf16",&[2,3],&x),tensor(A::QUERY_ID,"output.bf16",&[2,2],&y)]}),
            )?,
        )?;
    }
    let reference = source::Artifact::open(&temp.path().join("reference"), "reference")?;
    let candidate = source::Artifact::open(&temp.path().join("cuda"), "cuda")?;
    let out = artifacts::Artifacts::create(&temp.path().join("report"))?;
    let r = report::analyze(
        report::Stage::Query,
        &[1],
        &reference,
        &candidate,
        &archive,
        &out,
    )?;
    assert_eq!(r["dot_bound_incompatible_count"], 0);
    assert_eq!(r["reference_replay_mismatch_count"], 0);
    assert_eq!(
        r["records"][0]["reference_input_dot"]["exact_nearest_bf16"],
        0x4200
    );
    assert_eq!(
        std::fs::read(out.directory().join("query-weight-row-1.bf16"))?,
        bytes(&[4., 5., 6.])
    );
    std::fs::write(temp.path().join("cuda/input.bf16"), bytes(&[0.; 6]))?;
    assert!(source::Artifact::open(&temp.path().join("cuda"), "cuda").is_err());
    Ok(())
}
