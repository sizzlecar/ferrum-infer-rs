//! One original-weight DiT evaluation with externally captured, already prepared
//! ports. This deliberately bypasses native RNG/TE/input preparation; it is not
//! an image pipeline, serving qualification, or performance benchmark.
#[cfg(any(feature = "cuda", test))]
#[path = "verify_qwen_image21_projection/artifacts.rs"]
mod artifacts;
#[cfg(any(feature = "cuda", test))]
#[path = "verify_qwen_image21_projection/arguments.rs"]
mod backend;
#[cfg(any(feature = "cuda", test))]
#[path = "verify_p_h3_image_dit/data.rs"]
mod data;
#[cfg(feature = "cuda")]
#[path = "verify_p_h3_image_dit/execution.rs"]
mod execution;
#[cfg(feature = "cuda")]
#[path = "support/tensor_component.rs"]
mod shared;
#[cfg(any(feature = "cuda", test))]
#[path = "verify_p_h3_image_dit/source.rs"]
mod source;
#[cfg(any(feature = "cuda", test))]
#[path = "verify_p_h3_image_dit/trajectory.rs"]
mod trajectory;
#[cfg(any(feature = "cuda", test))]
use anyhow::Context;
use anyhow::{bail, ensure, Result};
#[cfg(any(feature = "cuda", test))]
use std::{collections::BTreeMap, ffi::OsString, path::PathBuf};

const USAGE: &str = "verify_p_h3_image_dit --config HF_TRANSFORMER_JSON --model-dir DIR --checkpoint-file ORIGINAL_SAFETENSORS --input-bundle CAPTURE.safetensors --input-metadata CAPTURE.json --artifact-dir NEW_EXTERNAL_DIR --host-memory-bytes N --attention cudnn_sdpa|sequential --backend cuda [--input-boundary prepared|packed|mod1] [--dense-linear-source archive|bf16-roundtrip] [--trajectory single|original-euler (prepared, no taps)] [--override-bundle PRODUCTION_STEP0.safetensors --override-group reference|text|initial-state (original-euler only; max-input-bytes covers both complete bundles)] [--capture-first-block true|false] [--first-block-taps boundaries|attention|attention-output|qkv|attention-core (requires capture=true)] [--max-input-bytes N] [--max-output-bytes N] [--max-metadata-bytes N] [--max-plan-nodes N] [--cudnn-library ABSOLUTE_SO --cudnn-opaque-memory-bytes N --cudnn-workspace-bytes N --sdpa-opaque-memory-bytes N] and existing CUDA budget options";
#[cfg(any(feature = "cuda", test))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Attention {
    Sequential,
    CudnnSdpa,
}
#[cfg(any(feature = "cuda", test))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
enum InputBoundary {
    Prepared,
    Packed,
    Mod1,
}
#[cfg(any(feature = "cuda", test))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
enum DenseLinearSource {
    Archive,
    Bf16Roundtrip,
}
#[cfg(any(feature = "cuda", test))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
enum Trajectory {
    #[default]
    Single,
    OriginalEuler,
}
#[cfg(any(feature = "cuda", test))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
enum OverrideGroup {
    Reference,
    Text,
    InitialState,
}
#[cfg(any(feature = "cuda", test))]
// CPU tests validate option contracts; CUDA execution consumes the parsed payload.
#[cfg_attr(all(test, not(feature = "cuda")), allow(dead_code))]
struct InputOverride {
    bundle: PathBuf,
    group: OverrideGroup,
}
#[cfg(any(feature = "cuda", test))]
impl InputOverride {
    fn parse(
        bundle: Option<OsString>,
        group: Option<OsString>,
        mode: Trajectory,
        boundary: InputBoundary,
        taps: bool,
    ) -> Result<Option<Self>> {
        let (bundle, group) = match (bundle, group) {
            (None, None) => return Ok(None),
            (Some(bundle), Some(group)) => (bundle, group),
            _ => bail!("--override-bundle and --override-group must be supplied together"),
        };
        ensure!(
            mode == Trajectory::OriginalEuler && boundary == InputBoundary::Prepared && !taps,
            "input override requires original-euler, prepared inputs and no hidden taps"
        );
        let group = match group.to_str() {
            Some("reference") => OverrideGroup::Reference,
            Some("text") => OverrideGroup::Text,
            Some("initial-state") => OverrideGroup::InitialState,
            _ => bail!("--override-group must be reference|text|initial-state"),
        };
        Ok(Some(Self {
            bundle: bundle.into(),
            group,
        }))
    }
}
#[cfg(any(feature = "cuda", test))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
enum FirstBlockTaps {
    #[default]
    Boundaries,
    Attention,
    AttentionOutput,
    Qkv,
    AttentionCore,
}
#[cfg(any(feature = "cuda", test))]
impl FirstBlockTaps {
    fn is_boundaries(&self) -> bool {
        *self == Self::Boundaries
    }
}
#[cfg(any(feature = "cuda", test))]
// CPU tests validate parsing; file paths and execution budgets are consumed on CUDA.
#[cfg_attr(all(test, not(feature = "cuda")), allow(dead_code))]
struct Arguments {
    config: PathBuf,
    model_dir: PathBuf,
    checkpoint_file: String,
    input_bundle: PathBuf,
    input_metadata: PathBuf,
    artifact_dir: PathBuf,
    host_memory_bytes: usize,
    max_input_bytes: usize,
    max_output_bytes: usize,
    max_metadata_bytes: usize,
    max_plan_nodes: usize,
    attention: Attention,
    input_boundary: InputBoundary,
    dense_linear_source: DenseLinearSource,
    capture_first_block: bool,
    first_block_taps: FirstBlockTaps,
    trajectory: Trajectory,
    input_override: Option<InputOverride>,
    cudnn_library: Option<PathBuf>,
    cudnn_opaque_bytes: u64,
    cudnn_workspace_bytes: u64,
    sdpa_opaque_bytes: u64,
    cuda: backend::CudaOptions,
}
#[cfg(any(feature = "cuda", test))]
impl Arguments {
    fn parse(args: impl IntoIterator<Item = OsString>) -> Result<Option<Self>> {
        let mut args = args.into_iter().peekable();
        if args.peek().is_some_and(|x| x == "--help" || x == "-h") {
            args.next();
            ensure!(args.next().is_none(), "--help takes no other arguments");
            return Ok(None);
        }
        let mut v = BTreeMap::new();
        while let Some(flag) = args.next() {
            let flag = flag
                .into_string()
                .map_err(|_| anyhow::anyhow!("option must be UTF-8"))?;
            ensure!(
                v.insert(flag, args.next().context("every option needs a value")?)
                    .is_none(),
                "duplicate option"
            );
        }
        let mut take = |key: &str| -> Result<PathBuf> {
            Ok(v.remove(key)
                .with_context(|| format!("missing {key}; {USAGE}"))?
                .into())
        };
        let config = take("--config")?;
        let model_dir = take("--model-dir")?;
        let checkpoint_file = take("--checkpoint-file")?
            .into_os_string()
            .into_string()
            .map_err(|_| anyhow::anyhow!("checkpoint basename must be UTF-8"))?;
        let input_bundle = take("--input-bundle")?;
        let input_metadata = take("--input-metadata")?;
        let artifact_dir = take("--artifact-dir")?;
        let attention = match v.remove("--attention").as_deref().and_then(|s| s.to_str()) {
            Some("sequential") => Attention::Sequential,
            Some("cudnn_sdpa") => Attention::CudnnSdpa,
            _ => bail!("explicit --attention cudnn_sdpa|sequential is required"),
        };
        let cudnn_library = v.remove("--cudnn-library").map(PathBuf::from);
        let input_boundary = match v.remove("--input-boundary") {
            None => InputBoundary::Prepared,
            Some(value) => match value.to_str() {
                Some("prepared") => InputBoundary::Prepared,
                Some("packed") => InputBoundary::Packed,
                Some("mod1") => InputBoundary::Mod1,
                _ => bail!("--input-boundary must be prepared|packed|mod1"),
            },
        };
        let capture_first_block = match v.remove("--capture-first-block") {
            None => false,
            Some(value) => value
                .to_str()
                .context("capture flag must be UTF-8")?
                .parse::<bool>()?,
        };
        let first_block_taps = match v.remove("--first-block-taps") {
            None => FirstBlockTaps::Boundaries,
            Some(value) => {
                ensure!(
                    capture_first_block,
                    "--first-block-taps requires --capture-first-block true"
                );
                match value.to_str() {
                    Some("boundaries") => FirstBlockTaps::Boundaries,
                    Some("attention") => FirstBlockTaps::Attention,
                    Some("attention-output") => FirstBlockTaps::AttentionOutput,
                    Some("qkv") => FirstBlockTaps::Qkv,
                    Some("attention-core") => FirstBlockTaps::AttentionCore,
                    _ => bail!(
                        "--first-block-taps must be boundaries|attention|attention-output|qkv|attention-core"
                    ),
                }
            }
        };
        let dense_linear_source = match v.remove("--dense-linear-source") {
            None => DenseLinearSource::Archive,
            Some(value) => match value.to_str() {
                Some("archive") => DenseLinearSource::Archive,
                Some("bf16-roundtrip") => DenseLinearSource::Bf16Roundtrip,
                _ => bail!("--dense-linear-source must be archive|bf16-roundtrip"),
            },
        };
        let trajectory = match v.remove("--trajectory") {
            None => Trajectory::Single,
            Some(value) => match value.to_str() {
                Some("single") => Trajectory::Single,
                Some("original-euler") => Trajectory::OriginalEuler,
                _ => bail!("--trajectory must be single|original-euler"),
            },
        };
        trajectory::validate_options(trajectory, input_boundary, capture_first_block)?;
        let input_override = InputOverride::parse(
            v.remove("--override-bundle"),
            v.remove("--override-group"),
            trajectory,
            input_boundary,
            capture_first_block,
        )?;
        let mut number = |key: &str, default: Option<u64>| -> Result<u64> {
            match v.remove(key) {
                Some(value) => Ok(value.to_str().context("number must be UTF-8")?.parse()?),
                None => default.with_context(|| format!("missing {key}")),
            }
        };
        let host_memory_bytes = usize::try_from(number("--host-memory-bytes", None)?)?;
        let max_input_bytes = usize::try_from(number("--max-input-bytes", Some(128 << 20))?)?;
        let max_output_bytes = usize::try_from(number("--max-output-bytes", Some(128 << 20))?)?;
        let max_metadata_bytes = usize::try_from(number("--max-metadata-bytes", Some(32 << 20))?)?;
        let max_plan_nodes = usize::try_from(number("--max-plan-nodes", Some(32768))?)?;
        let cudnn_opaque_bytes = number("--cudnn-opaque-memory-bytes", Some(0))?;
        let cudnn_workspace_bytes = number("--cudnn-workspace-bytes", Some(0))?;
        let sdpa_opaque_bytes = number("--sdpa-opaque-memory-bytes", Some(0))?;
        ensure!(
            [
                host_memory_bytes,
                max_input_bytes,
                max_output_bytes,
                max_metadata_bytes,
                max_plan_nodes
            ]
            .iter()
            .all(|n| *n > 0 && *n <= isize::MAX as usize),
            "capacities must be positive and addressable"
        );
        match attention {
            Attention::CudnnSdpa => ensure!(
                cudnn_library.as_ref().is_some_and(|p| p.is_absolute())
                    && cudnn_opaque_bytes > 0
                    && sdpa_opaque_bytes > 0,
                "cuDNN requires an absolute library and explicit opaque allowances"
            ),
            Attention::Sequential => ensure!(
                cudnn_library.is_none()
                    && cudnn_opaque_bytes == 0
                    && cudnn_workspace_bytes == 0
                    && sdpa_opaque_bytes == 0,
                "sequential attention cannot silently ignore cuDNN options"
            ),
        }
        let backend::Backend::Cuda(cuda) = backend::Backend::parse_options(&mut v)? else {
            bail!("requires --backend cuda")
        };
        Ok(Some(Self {
            config,
            model_dir,
            checkpoint_file,
            input_bundle,
            input_metadata,
            artifact_dir,
            host_memory_bytes,
            max_input_bytes,
            max_output_bytes,
            max_metadata_bytes,
            max_plan_nodes,
            attention,
            input_boundary,
            dense_linear_source,
            capture_first_block,
            first_block_taps,
            trajectory,
            input_override,
            cudnn_library,
            cudnn_opaque_bytes,
            cudnn_workspace_bytes,
            sdpa_opaque_bytes,
            cuda,
        }))
    }
}
#[cfg(feature = "cuda")]
fn main() -> Result<()> {
    let Some(args) = Arguments::parse(std::env::args_os().skip(1))? else {
        println!("{USAGE}\n{}", backend::USAGE);
        return Ok(());
    };
    execution::run(args)
}

#[cfg(not(feature = "cuda"))]
fn main() -> Result<()> {
    let mut args = std::env::args_os().skip(1);
    if args
        .next()
        .is_some_and(|arg| arg == "--help" || arg == "-h")
    {
        ensure!(args.next().is_none(), "--help takes no other arguments");
        println!("{USAGE}");
        return Ok(());
    }
    bail!("requires --features cuda with the locked native operator set")
}
#[cfg(test)]
#[path = "verify_p_h3_image_dit/tests.rs"]
mod tests;
