use anyhow::{ensure, Context, Result};
use ferrum_models::fast_h3::FastH3TextTagRun;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum HiddenInput {
    DiagnosticPattern,
    Bf16File { path: PathBuf, sha256: String },
}
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Rotary {
    AnalyticPositions,
    DiagnosticIdentity,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TensorFile {
    pub path: PathBuf,
    pub sha256: String,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum LatentInput {
    File { path: PathBuf, sha256: String },
    DiagnosticZero,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Workload {
    Backbone {
        hidden_input: HiddenInput,
        rotary: Rotary,
    },
    Pdd {
        text_features: TensorFile,
        video_reference_and_noise: LatentInput,
        audio_reference_and_noise: LatentInput,
        rotary: Rotary,
    },
    PddSeededKeyframe {
        text_features: TensorFile,
        normalized_keyframe: TensorFile,
        seed: u64,
        rotary: Rotary,
    },
}
impl Workload {
    pub fn rotary(&self) -> Rotary {
        match self {
            Self::Backbone { rotary, .. }
            | Self::Pdd { rotary, .. }
            | Self::PddSeededKeyframe { rotary, .. } => *rotary,
        }
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub model_dir: PathBuf,
    pub output_dir: PathBuf,
    pub device: usize,
    pub width: usize,
    pub height: usize,
    pub frames: usize,
    pub text_rows: usize,
    pub text_tags: Vec<FastH3TextTagRun>,
    pub reference_latent_height: usize,
    pub reference_latent_width: usize,
    pub workload: Workload,
    pub host_bytes: usize,
    pub device_bytes: u64,
    pub bootstrap_bytes: u64,
    pub stream_bytes: u64,
    pub native_bytes: u64,
    pub staging_bytes: usize,
    pub cudnn_library: PathBuf,
    pub cudnn_dependencies: Vec<PathBuf>,
    pub cudnn_opaque_bytes: u64,
    pub cudnn_plan_bytes: u64,
    pub workspace_bytes: u64,
    pub maximum_rows: usize,
    pub maximum_tiles: usize,
    pub maximum_layers: usize,
    pub maximum_plan_nodes: usize,
    pub maximum_steps: usize,
    pub maximum_metadata_bytes: usize,
    pub maximum_source_bytes: usize,
    pub maximum_tensors: usize,
}
impl Config {
    pub fn read(path: &Path) -> Result<Self> {
        let file = std::fs::File::open(path)?;
        ensure!(
            file.metadata()?.len() <= 1 << 20,
            "configuration exceeds 1MiB bound"
        );
        let c: Self = serde_json::from_reader(file)?;
        for n in [
            c.width,
            c.height,
            c.frames,
            c.text_rows,
            c.reference_latent_height,
            c.reference_latent_width,
            c.host_bytes,
            c.staging_bytes,
            c.maximum_rows,
            c.maximum_tiles,
            c.maximum_layers,
            c.maximum_plan_nodes,
            c.maximum_steps,
            c.maximum_metadata_bytes,
            c.maximum_source_bytes,
            c.maximum_tensors,
        ] {
            ensure!(
                n > 0 && n <= isize::MAX as usize,
                "positive addressable capacities required"
            );
        }
        for n in [
            c.device_bytes,
            c.bootstrap_bytes,
            c.stream_bytes,
            c.native_bytes,
            c.cudnn_opaque_bytes,
            c.cudnn_plan_bytes,
            c.workspace_bytes,
        ] {
            ensure!(
                n > 0 && n <= isize::MAX as u64,
                "positive addressable device budgets required"
            );
        }
        ensure!(
            c.staging_bytes <= c.host_bytes,
            "staging exceeds Host budget"
        );
        ensure!(
            c.maximum_steps <= usize::MAX / 4,
            "time-row capacity overflows"
        );
        let rows = c
            .text_tags
            .iter()
            .try_fold(0u64, |n, r| n.checked_add(r.rows).filter(|_| r.rows > 0))
            .context("invalid text tags")?;
        ensure!(
            rows == c.text_rows as u64,
            "text tags must cover presentation rows"
        );
        let hashes: Vec<&str> = match &c.workload {
            Workload::Backbone {
                hidden_input: HiddenInput::Bf16File { sha256, .. },
                ..
            } => vec![sha256],
            Workload::Backbone { .. } => vec![],
            Workload::PddSeededKeyframe {
                text_features,
                normalized_keyframe,
                ..
            } => vec![&text_features.sha256, &normalized_keyframe.sha256],
            Workload::Pdd {
                text_features,
                video_reference_and_noise,
                audio_reference_and_noise,
                ..
            } => {
                let mut hashes = vec![text_features.sha256.as_str()];
                for input in [video_reference_and_noise, audio_reference_and_noise] {
                    if let LatentInput::File { sha256, .. } = input {
                        hashes.push(sha256.as_str());
                    }
                }
                hashes
            }
        };
        for sha256 in hashes {
            ensure!(
                sha256.len() == 64
                    && sha256
                        .bytes()
                        .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase()),
                "expected lowercase SHA256"
            );
        }
        Ok(c)
    }
}
