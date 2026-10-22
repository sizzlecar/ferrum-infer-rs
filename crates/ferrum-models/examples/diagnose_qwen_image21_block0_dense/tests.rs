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
fn dyadic_subtraction_rejects_minimum_signed_integer_without_sign_flip() {
    let negative_max = Exact {
        numerator: -i128::MAX,
        exponent: 0,
    };
    let one = Exact {
        numerator: 1,
        exponent: 0,
    };
    assert!(negative_max.subtract(one).is_err());
    let minimum = Exact {
        numerator: i128::MIN,
        exponent: 0,
    };
    let zero = Exact {
        numerator: 0,
        exponent: 0,
    };
    assert!(minimum.subtract(zero).is_err());
    assert!(zero.subtract(minimum).is_err());
    assert_eq!(negative_max.subtract(zero).unwrap().numerator, -i128::MAX);
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

#[test]
fn mlp_up_channel_selection_is_explicit_and_preserves_existing_selections() -> Result<()> {
    let defaults = selected_stages(&mut BTreeMap::new())?;
    assert_eq!(
        defaults,
        vec![
            (report::Stage::Query, vec![953]),
            (report::Stage::Value, vec![126, 668, 1602]),
            (report::Stage::MlpOutput, vec![2615]),
        ]
    );
    let mut options = BTreeMap::from([("--mlp-up-channels".into(), "1,0".into())]);
    let mut selected = selected_stages(&mut options)?;
    assert!(options.is_empty());
    assert_eq!(selected.pop(), Some((report::Stage::MlpUp, vec![1, 0])));
    assert_eq!(selected, defaults);
    for invalid in ["", "1,1", "-1", "0,"] {
        let mut options = BTreeMap::from([("--mlp-up-channels".into(), invalid.into())]);
        assert!(selected_stages(&mut options).is_err());
    }
    Ok(())
}

struct MlpUpFixture {
    temp: tempfile::TempDir,
    archive: SafetensorsArchive,
    reference: source::Artifact,
    candidate: source::Artifact,
}
impl MlpUpFixture {
    fn new() -> Result<Self> {
        use ferrum_models::qwen_image21::{
            QwenImage21Block0Family as B, QwenImage21Block0MlpFamily as M,
        };
        let temp = tempfile::tempdir()?;
        let weights = bytes(&[1., 2., 3., 4., 5., 6.]);
        safetensors::tensor::serialize_to_file(
            [(
                B::WEIGHT_NAMES[7],
                safetensors::tensor::TensorView::new(
                    safetensors::Dtype::BF16,
                    vec![2, 3],
                    &weights,
                )?,
            )],
            &None,
            &temp.path().join("model.safetensors"),
        )?;
        let original_input = bytes(&[1., 2., 3., -1., -2., -3.]);
        let tensor = |id: &str, file: &str, shape: &[u64], data: &[u8]| {
            json!({"value_id":id,"file":file,"dtype":"BF16","shape":shape,
                "bytes":data.len(),"sha256":artifacts::sha256(data)})
        };
        let identity = json!({"canonical_config":{},"config_file_sha256":"fixture",
            "inputs":[tensor(B::INPUT_ID,"input.bf16",&[2,3],&original_input)],
            "weights":[{"tensor":B::WEIGHT_NAMES[7],"dtype":"BF16","shape":[2,3],
                "bytes":weights.len(),"sha256":artifacts::sha256(&weights)}]});
        for (backend, input, output) in [
            (
                "reference",
                original_input.clone(),
                bytes(&[14., 32., -14., -32.]),
            ),
            (
                "cuda",
                bytes(&[1., 2., 4., -1., -2., -4.]),
                bytes(&[17., 38., -17., -38.]),
            ),
        ] {
            let dir = temp.path().join(backend);
            std::fs::create_dir(&dir)?;
            std::fs::write(dir.join("input.bf16"), &original_input)?;
            std::fs::write(dir.join("modulated2.bf16"), &input)?;
            std::fs::write(dir.join("up.bf16"), &output)?;
            std::fs::write(
                dir.join("manifest.json"),
                serde_json::to_vec(&json!({"schema":"ferrum.qwen_image21.block0-artifacts.v1",
                    "backend":backend,"identity":identity,"outputs":[
                        tensor(B::MODULATED2_ID,"modulated2.bf16",&[2,3],&input),
                        tensor(M::UP_ID,"up.bf16",&[2,2],&output)]}))?,
            )?;
        }
        Ok(Self {
            archive: SafetensorsArchive::open(temp.path())?,
            reference: source::Artifact::open(&temp.path().join("reference"), "reference")?,
            candidate: source::Artifact::open(&temp.path().join("cuda"), "cuda")?,
            temp,
        })
    }

    fn analyze(&self, name: &str, channels: &[usize]) -> Result<serde_json::Value> {
        report::analyze(
            report::Stage::MlpUp,
            channels,
            &self.reference,
            &self.candidate,
            &self.archive,
            &artifacts::Artifacts::create(&self.temp.path().join(name))?,
        )
    }
}

#[test]
fn mlp_up_replays_each_saved_input_and_preserves_selected_row_identity() -> Result<()> {
    use ferrum_models::qwen_image21::{
        QwenImage21Block0Family as B, QwenImage21Block0MlpFamily as M,
    };
    let fixture = MlpUpFixture::new()?;
    assert_eq!(
        fixture.reference.manifest.identity,
        fixture.candidate.manifest.identity
    );
    let r = fixture.analyze("report", &[1])?;
    assert_eq!(r["input_port"], B::MODULATED2_ID);
    assert_eq!(r["output_port"], M::UP_ID);
    assert_eq!(r["weight"], B::WEIGHT_NAMES[7]);
    assert_eq!(r["input_bits_identical"], false);
    assert_eq!(r["dot_bound_incompatible_count"], 0);
    assert_eq!(r["reference_replay_mismatch_count"], 0);
    for (row, (reference, candidate, effect)) in [(32.0, 38.0, "6"), (-32.0, -38.0, "-6")]
        .into_iter()
        .enumerate()
    {
        let record = &r["records"][row];
        assert_eq!(record["row"], row);
        assert_eq!(record["channel"], 1);
        assert_eq!(record["reference_output"], reference);
        assert_eq!(record["candidate_output"], candidate);
        assert_eq!(record["exact_local_profile_residual"]["numerator"], "0");
        // Exact normalizes the dyadic representation, so compare its value.
        let numerator: f64 = record["exact_mathematical_input_effect"]["numerator"]
            .as_str()
            .unwrap()
            .parse()?;
        let exponent = record["exact_mathematical_input_effect"]["exponent"]
            .as_i64()
            .unwrap() as i32;
        assert_eq!(numerator * 2_f64.powi(exponent), effect.parse::<f64>()?);
    }
    assert_eq!(
        std::fs::read(fixture.temp.path().join("report/mlp_up-weight-row-1.bf16"))?,
        bytes(&[4., 5., 6.])
    );
    // A changed local input does not weaken authentication of its saved bytes.
    std::fs::write(
        fixture.temp.path().join("cuda/modulated2.bf16"),
        bytes(&[0.; 6]),
    )?;
    assert!(fixture
        .analyze("corrupt-input", &[1])
        .unwrap_err()
        .to_string()
        .contains("artifact bytes/SHA mismatch"));
    Ok(())
}

#[test]
fn mlp_up_rejects_out_of_extent_channels_and_equal_numel_wrong_shapes() -> Result<()> {
    let mut fixture = MlpUpFixture::new()?;
    for (name, channels) in [
        ("empty", &[][..]),
        ("end", &[2][..]),
        ("overflow", &[usize::MAX][..]),
    ] {
        assert!(fixture
            .analyze(name, channels)
            .unwrap_err()
            .to_string()
            .contains("selected channels outside extent/limit"));
    }
    fixture.candidate.manifest.outputs[0].shape = vec![3, 2];
    assert!(fixture
        .analyze("input-shape", &[1])
        .unwrap_err()
        .to_string()
        .contains("local input shapes differ"));
    fixture.candidate.manifest.outputs[0].shape = vec![2, 3];
    for artifact in [&mut fixture.reference, &mut fixture.candidate] {
        artifact.manifest.outputs[1].shape = vec![1, 4];
    }
    assert!(fixture
        .analyze("output-shape", &[1])
        .unwrap_err()
        .to_string()
        .contains("dense input/output row count differs"));
    for artifact in [&mut fixture.reference, &mut fixture.candidate] {
        artifact.manifest.outputs[1].shape = vec![2, 2];
        artifact.manifest.identity.weights[0].shape = vec![3, 2];
    }
    assert!(fixture
        .analyze("weight-shape", &[1])
        .unwrap_err()
        .to_string()
        .contains("weight metadata/SHA mismatch"));
    Ok(())
}
