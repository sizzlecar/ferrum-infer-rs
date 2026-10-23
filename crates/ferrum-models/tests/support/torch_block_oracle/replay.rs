use super::*;

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ReplayConfig {
    torch: native::Config,
    reference_artifact_dir: PathBuf,
    candidate_artifact_dir: PathBuf,
    model_dir: PathBuf,
    index_file: String,
    transformer_config_file: PathBuf,
    artifact_dir: PathBuf,
    policy: MathPolicy,
}
impl ReplayConfig {
    fn from_environment() -> Result<Self> {
        let path = std::env::var_os("FERRUM_TEST_TORCH_BLOCK_CONFIG")
            .context("explicit FERRUM_TEST_TORCH_BLOCK_CONFIG is required")?;
        let result: Self = serde_json::from_slice(&native::read_bounded(
            Path::new(&path),
            native::METADATA_LIMIT,
        )?)?;
        ensure!(
            [
                &result.reference_artifact_dir,
                &result.candidate_artifact_dir,
                &result.model_dir,
                &result.transformer_config_file,
                &result.artifact_dir
            ]
            .iter()
            .all(|p| p.is_absolute()),
            "full-block source/output paths must be explicit absolute paths"
        );
        Ok(result)
    }
}

pub(super) fn original_inputs(
    directory: &Path,
    identity: &source::Identity,
    geometry: Geometry,
) -> Result<[Segment; 6]> {
    ensure!(
        identity.inputs.len() == Block::INPUT_IDS.len(),
        "block requires every original input exactly once"
    );
    let directory = directory.canonicalize()?;
    let shapes = geometry.input_shapes()?;
    let mut inputs = Vec::new();
    for (index, expected_id) in Block::INPUT_IDS.iter().enumerate() {
        let matching = identity
            .inputs
            .iter()
            .filter(|i| i.value_id == *expected_id)
            .collect::<Vec<_>>();
        ensure!(
            matching.len() == 1,
            "missing/duplicate original block input"
        );
        let input = matching[0];
        ensure!(
            input.shape == shapes[index]
                && input.dtype == if index == 5 { "F32" } else { "BF16" }
                && input.bytes == tensor_bytes(&shapes[index], if index == 5 { 4 } else { 2 })?,
            "original input dtype/shape/extent mismatch"
        );
        let path = Path::new(&input.file);
        ensure!(
            path.components().count() == 1
                && matches!(path.components().next(), Some(Component::Normal(_))),
            "original input file must be a basename"
        );
        let path = directory.join(path).canonicalize()?;
        ensure!(
            path.starts_with(&directory),
            "original input escaped its artifact directory"
        );
        let segment = Segment {
            location: SegmentLocation {
                path,
                offset: 0,
                bytes: input.bytes,
            },
            sha256: input.sha256.clone(),
        };
        segment.verify()?;
        inputs.push(segment);
    }
    inputs
        .try_into()
        .map_err(|_| anyhow::anyhow!("incomplete original input set"))
}

fn case(config: &ReplayConfig, original: &source::Artifact) -> Result<Case> {
    let identity = &original.manifest.identity;
    let canonical: QwenImage21Block0Config =
        serde_json::from_value(identity.canonical_config.clone())?;
    let layout = canonical.attention().layout();
    ensure!(
        canonical.broadcast() == QwenImage21Block0Broadcast::Elementwise
            && layout == &QwenImage21AttentionLayout::uniform(layout.sequence_length(), true)?,
        "block oracle currently requires single-batch fully visible elementwise conditioning"
    );
    let config_bytes =
        native::read_bounded(&config.transformer_config_file, native::METADATA_LIMIT)?;
    ensure!(
        native::bytes_sha(&config_bytes) == identity.config_file_sha256,
        "original transformer config SHA differs"
    );
    let transformer = ferrum_models::qwen_image21::QwenImage21Config::from_hf_json(
        &config_bytes,
        ferrum_models::qwen_image21::StructureLimits::default(),
    )?;
    ensure!(
        &transformer == canonical.transformer(),
        "canonical block config differs from actual transformer configuration"
    );
    let effective = |r: ferrum_runtime::CanonicalRational| {
        ((r.numerator() as f64 / r.denominator() as f64) as f32).to_bits()
    };
    let geometry = Geometry {
        rows: layout.rows(),
        heads: transformer.heads(),
        head_dim: transformer.head_dim(),
        mlp_width: transformer.mlp_hidden_size(),
        epsilon_f32_bits: effective(transformer.epsilon()),
        scale_f32_bits: effective(canonical.attention().scale()),
    };
    geometry.validate()?;
    let inputs = original_inputs(&config.reference_artifact_dir, identity, geometry)?;
    ensure!(
        identity.weights.len() == Block::WEIGHT_NAMES.len(),
        "block requires all original weight identities"
    );
    let archive = SafetensorsArchive::open_with_index(&config.model_dir, &config.index_file)?;
    let shapes = geometry.weight_shapes()?;
    let mut weights = Vec::new();
    for (index, name) in Block::WEIGHT_NAMES.iter().enumerate() {
        let matching = identity
            .weights
            .iter()
            .filter(|w| w.tensor == *name)
            .collect::<Vec<_>>();
        ensure!(
            matching.len() == 1,
            "block weight identity missing or duplicated"
        );
        let recorded = matching[0];
        let tensor = archive.tensor(name)?;
        ensure!(
            recorded.dtype == "BF16"
                && recorded.shape == shapes[index]
                && tensor.dtype() == safetensors::Dtype::BF16
                && tensor.shape() == recorded.shape
                && tensor.bytes().len() as u64 == recorded.bytes
                && native::bytes_sha(tensor.bytes()) == recorded.sha256,
            "complete original block weight metadata/SHA mismatch: {name}"
        );
        weights.push(Segment {
            location: SegmentLocation {
                path: config.model_dir.join(tensor.source_file()).canonicalize()?,
                offset: u64::try_from(tensor.retained_host_memory().offset_bytes())?,
                bytes: recorded.bytes,
            },
            sha256: recorded.sha256.clone(),
        });
    }
    let case = Case {
        geometry,
        policy: config.policy,
        inputs,
        weights: weights
            .try_into()
            .map_err(|_| anyhow::anyhow!("incomplete block weights"))?,
    };
    case.verify_sources()?;
    Ok(case)
}

fn replay(config: ReplayConfig) -> Result<()> {
    ensure!(
        matches!(config.torch.executable, native::Executable::Prebuilt { .. }),
        "GPU replay requires separately built prebuilt oracle"
    );
    let original = source::Artifact::open(&config.reference_artifact_dir, "reference")?;
    let case = case(&config, &original)?;
    let files = artifacts::Artifacts::create(&config.artifact_dir)?;
    files.json("declaration.json", &json!({
        "schema":"ferrum.block0.torch-whole-block.v1", "configuration":config, "case":case,
        "original_identity":original.manifest.identity, "reference_manifest_sha256":original.manifest_sha256,
        "arithmetic":ARITHMETIC,
        "criterion":"every finite CUDA boundary value: abs(cuda-torch)<=0.02+0.02*abs(torch), evaluated in F64",
        "original_reference_verdict":"unchanged; reported separately, never rewritten",
        "scope":"single-batch fully visible BF16 block with original identity RoPE probe, not FP8 deployment or nonidentity RoPE qualification",
        "input_authority":"only original six manifest input files and original nine full checkpoint tensors; candidate outputs are not passed to ATen",
        "rounding":"BF16 boundaries from standard_operations pointwise/head_rms_norm/transcendental contracts; F32 two-pass LayerNorm and dense softmax may use different reduction trees than Ferrum providers",
        "weight_transport":"verified original safetensors shard segments; no copied weight files",
        "file_contract":"pre/post SHA checks assume immutable local source files throughout child execution; not an adversarial concurrent-mutation guarantee",
        "managed_host_scope":"owned oracle output payload; source mmap/metadata and external ATen process RSS are separate",
        "math":"BF16 policy explicitly selected; F32 matmul TF32 disabled and actual Context observed; NoReducedPrecision permits splitK"
    }))?;
    let bytes = usize::try_from(case.geometry.output_bytes()?)?;
    let host = HostMemoryAccount::new(HostMemoryLimits::new(
        bytes,
        HostMemoryClassLimits {
            input_preprocess_bytes: 0,
            output_bytes: bytes,
            transfer_staging_bytes: 0,
            weight_backing_bytes: 0,
        },
    )?);
    let outcome = (|| -> Result<()> {
        let mut oracle = Oracle::prepare(config.torch.clone())?;
        let output = oracle.run(&case, &host)?;
        ensure!(
            native::bytes_sha(&native::read_bounded(
                &config.reference_artifact_dir.join("manifest.json"),
                4 * 1024 * 1024
            )?) == original.manifest_sha256
                && native::bytes_sha(&native::read_bounded(
                    &config.transformer_config_file,
                    native::METADATA_LIMIT
                )?) == original.manifest.identity.config_file_sha256,
            "original manifest/config changed during replay"
        );
        // Candidate artifact access begins only after independent propagation.
        let candidate = source::Artifact::open(&config.candidate_artifact_dir, "cuda")?;
        ensure!(
            candidate.manifest.identity == original.manifest.identity,
            "candidate original input/config/weight identity differs"
        );
        let boundaries = case.geometry.boundaries()?;
        ensure!(
            candidate.manifest.outputs.len() == boundaries.len()
                && original.manifest.outputs.len() == boundaries.len(),
            "baseline does not contain exactly the complete boundary set"
        );
        let mut records = Vec::new();
        let mut offset = 0_usize;
        let mut failures = 0_usize;
        let mut nonfinite = 0_usize;
        for (index, boundary) in boundaries.iter().enumerate() {
            let end = offset
                .checked_add(usize::try_from(boundary.bytes)?)
                .context("boundary offset overflow")?;
            let torch = &output.bytes.bytes()[offset..end];
            let (candidate_meta, cuda) = candidate.output(&boundary.value_id)?;
            let (original_meta, reference) = original.output(&boundary.value_id)?;
            ensure!(
                candidate_meta.shape == boundary.shape
                    && original_meta.shape == boundary.shape
                    && candidate_meta.bytes == boundary.bytes
                    && original_meta.bytes == boundary.bytes,
                "baseline boundary geometry differs"
            );
            let cuda_vs_torch = comparison(&cuda, torch)?;
            let reference_vs_torch = comparison(&reference, torch)?;
            failures += cuda_vs_torch.failures_under_original_threshold;
            nonfinite += cuda_vs_torch.nonfinite_pairs + reference_vs_torch.nonfinite_pairs;
            files.bytes(&format!("output-{index:02}.bf16"), torch)?;
            records.push(
                json!({"boundary":boundary, "oracle_sha256":native::bytes_sha(torch),
                "candidate_sha256":candidate_meta.sha256, "reference_sha256":original_meta.sha256,
                "cuda_vs_torch":cuda_vs_torch, "reference_vs_torch":reference_vs_torch}),
            );
            offset = end;
        }
        ensure!(
            offset == output.bytes.len(),
            "complete block output was not consumed"
        );
        files.json("oracle-metadata.json", &json!({"metadata":output.metadata,
            "actual_library_sha256":output.library_sha256, "executable_sha256":oracle.native.executable_sha256}))?;
        drop(output);
        let usage = host.usage()?;
        ensure!(
            usage.committed_bytes() == 0 && usage.allocated_bytes() == 0,
            "managed full-block payload did not retire"
        );
        let passed = failures == 0 && nonfinite == 0;
        let result = json!({"diagnostic_completed":true,"passed":passed,"independent_block_parity_established":passed,
            "candidate_threshold_failures":failures,"nonfinite_pairs":nonfinite,"boundaries":records,
            "original_reference_verdict":"unchanged", "scope":"original BF16 full-block identity-RoPE probe only",
            "managed_host_committed_bytes":usage.committed_bytes(),"managed_host_allocated_bytes":usage.allocated_bytes()});
        files.json("result.json", &result)?;
        eprintln!("{result}");
        ensure!(passed, "independent full-block comparison failed: {failures} threshold failures, {nonfinite} nonfinite pairs");
        Ok(())
    })();
    if let Err(error) = &outcome {
        files.json("failure.json", &json!({"error":format!("{error:#}"),"passed":false,
            "managed_host_committed_bytes":host.usage()?.committed_bytes(),
            "managed_host_allocated_bytes":host.usage()?.allocated_bytes(),"original_reference_verdict":"unchanged"}))?;
    }
    outcome
}

#[test]
#[ignore = "explicit deployed Torch configuration and Linux C++ toolchain; build only, no GPU"]
fn build_deployed_torch_block_oracle() {
    let config = ReplayConfig::from_environment().unwrap();
    assert!(matches!(
        config.torch.executable,
        native::Executable::Build { .. }
    ));
    let oracle = Oracle::prepare(config.torch).unwrap();
    eprintln!(
        "{}",
        json!({"executable":oracle.native.executable,"executable_sha256":oracle.native.executable_sha256,
        "build_receipt":oracle.native.build_receipt,"gpu_executed":false,"program":"aten.qwen_image21.block0"})
    );
}

#[test]
#[ignore = "exclusive GPU window, original block artifacts/checkpoint, separately built deployed ATen oracle; all boundaries must satisfy original tolerance"]
fn deployed_torch_replays_complete_block0() {
    replay(ReplayConfig::from_environment().unwrap()).unwrap();
}
