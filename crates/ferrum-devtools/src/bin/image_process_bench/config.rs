use super::Args;
use anyhow::{bail, ensure, Context, Result};
use ferrum_bench_core::images::{
    validate_reference_png, ImageDataset, ImageRequestMode, ImageResponseLimits, ImageSample,
    ValidatedImage,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs,
    io::Read,
    path::{Path, PathBuf},
};

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct BoundFile {
    pub path: PathBuf,
    pub sha256: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Template {
    pub schema_version: u32,
    pub label: String,
    /// Operator declaration, not an assertion of equal precision or model quality.
    pub model_and_precision: String,
    pub program: BoundFile,
    pub cwd: PathBuf,
    /// Whole-argument tokens: {prompt}, {seed}, {width}, {height}, {steps},
    /// {output}, {request_dir}, {sample_id}, {reference}, {input:NAME}.
    /// A doubled outer brace escapes a literal token; values are expanded once.
    pub args: Vec<String>,
    #[serde(default)]
    pub env: BTreeMap<String, String>,
    #[serde(default)]
    pub remove_env: Vec<String>,
    #[serde(default)]
    pub inputs: BTreeMap<String, BoundFile>,
}

pub(super) struct Prepared {
    pub template: Template,
    pub template_bytes: Vec<u8>,
    pub dataset_bytes: Vec<u8>,
    pub dataset: ImageDataset,
    pub limits: ImageResponseLimits,
    pub reference: Option<(BoundFile, ValidatedImage)>,
    pub planned_requests: usize,
}

pub(super) fn hash_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

impl BoundFile {
    fn resolve(&mut self, base: &Path) -> Result<()> {
        if !self.path.is_absolute() {
            self.path = base.join(&self.path);
        }
        self.path = self.path.canonicalize().context("resolve bound file")?;
        ensure!(self.path.is_file(), "bound path must be a regular file");
        ensure!(
            self.sha256.len() == 64 && self.sha256.bytes().all(|b| b.is_ascii_hexdigit()),
            "bound file requires a SHA-256"
        );
        self.sha256.make_ascii_lowercase();
        self.verify()
    }

    pub fn verify(&self) -> Result<()> {
        let mut file = fs::File::open(&self.path)?;
        let mut digest = Sha256::new();
        let mut buffer = [0_u8; 65536];
        loop {
            let n = file.read(&mut buffer)?;
            if n == 0 {
                break;
            }
            digest.update(&buffer[..n]);
        }
        ensure!(
            format!("{:x}", digest.finalize()) == self.sha256,
            "bound bytes changed: {}",
            self.path.display()
        );
        Ok(())
    }
}

pub(super) fn bounded_read(path: &Path, maximum: usize) -> Result<Vec<u8>> {
    let metadata = fs::symlink_metadata(path)?;
    ensure!(
        metadata.is_file(),
        "expected regular file: {}",
        path.display()
    );
    ensure!(
        metadata.len() <= u64::try_from(maximum)?,
        "file exceeds byte limit: {}",
        path.display()
    );
    let mut bytes = Vec::new();
    fs::File::open(path)?
        .take(
            u64::try_from(maximum)?
                .checked_add(1)
                .context("byte limit overflow")?,
        )
        .read_to_end(&mut bytes)?;
    ensure!(bytes.len() <= maximum, "file grew beyond byte limit");
    Ok(bytes)
}

impl Prepared {
    pub fn load(args: &Args) -> Result<Self> {
        ensure!(
            (1..=1024).contains(&args.concurrency),
            "concurrency must be in 1..=1024"
        );
        ensure!(
            args.repetitions > 0,
            "at least one measured repetition is required"
        );
        ensure!(
            (1..=86400).contains(&args.timeout_seconds),
            "timeout must be in 1..=86400 seconds"
        );
        ensure!(
            args.termination_grace_seconds <= 300,
            "termination grace exceeds 300 seconds"
        );
        ensure!(
            (1..=1_000_000).contains(&args.maximum_total_requests),
            "invalid maximum request count"
        );
        let template_path = args.command_template.canonicalize()?;
        let dataset_path = args.dataset.canonicalize()?;
        let base = template_path.parent().context("template parent")?;
        let template_bytes = bounded_read(&template_path, 16 * 1024 * 1024)?;
        let dataset_bytes = bounded_read(&dataset_path, 16 * 1024 * 1024)?;
        let mut template: Template = serde_json::from_slice(&template_bytes)?;
        ensure!(template.schema_version == 1, "unsupported template version");
        ensure!(
            !template.label.is_empty() && !template.model_and_precision.is_empty(),
            "template requires label and model_and_precision declaration"
        );
        template.program.resolve(base)?;
        if !template.cwd.is_absolute() {
            template.cwd = base.join(&template.cwd);
        }
        template.cwd = template.cwd.canonicalize()?;
        ensure!(template.cwd.is_dir(), "cwd must be a directory");
        for input in template.inputs.values_mut() {
            input.resolve(base)?;
        }
        for key in template.env.keys().chain(template.remove_env.iter()) {
            ensure!(
                !key.is_empty() && !key.contains(['=', '\0']),
                "invalid environment key"
            );
        }
        ensure!(
            !template.args.is_empty(),
            "command template has no arguments"
        );
        let limits = ImageResponseLimits::default();
        let dataset: ImageDataset = serde_json::from_slice(&dataset_bytes)?;
        dataset.validate(&limits)?;
        let planned_requests = args
            .repetitions
            .checked_add(args.warmup_repetitions)
            .and_then(|n| n.checked_mul(dataset.samples.len()))
            .context("request count overflow")?;
        ensure!(
            planned_requests <= args.maximum_total_requests,
            "planned requests exceed maximum_total_requests"
        );
        let reference = match &dataset.mode {
            ImageRequestMode::Generate => None,
            ImageRequestMode::Edit { image } => {
                let path = if image.path.is_absolute() {
                    image.path.clone()
                } else {
                    dataset_path
                        .parent()
                        .context("dataset parent")?
                        .join(&image.path)
                };
                let path = path.canonicalize()?;
                let bytes = bounded_read(&path, limits.maximum_png_bytes)?;
                let validated = validate_reference_png(&bytes, &limits)?;
                if let Some(expected) = &image.encoded_sha256 {
                    ensure!(
                        expected.eq_ignore_ascii_case(&validated.png_sha256),
                        "reference PNG SHA mismatch"
                    );
                }
                Some((
                    BoundFile {
                        path,
                        sha256: validated.png_sha256.clone(),
                    },
                    validated,
                ))
            }
        };
        let prepared = Self {
            template,
            template_bytes,
            dataset_bytes,
            dataset,
            limits,
            reference,
            planned_requests,
        };
        let values = prepared
            .template
            .args
            .iter()
            .chain(prepared.template.env.values());
        ensure!(
            values.clone().any(|value| value == "{output}"),
            "template must bind {{output}}"
        );
        for token in ["{prompt}", "{seed}", "{width}", "{height}"] {
            ensure!(
                values.clone().any(|value| value == token),
                "template must bind {token} from frozen samples"
            );
        }
        if prepared.reference.is_some() {
            ensure!(
                values.clone().any(|value| value == "{reference}"),
                "edit template must bind {{reference}}"
            );
        }
        if prepared
            .dataset
            .samples
            .iter()
            .any(|sample| sample.num_inference_steps.is_some())
        {
            ensure!(
                values.clone().any(|value| value == "{steps}"),
                "dataset steps require {{steps}} binding"
            );
        }
        for sample in &prepared.dataset.samples {
            for value in values.clone() {
                prepared.expand(value, sample, &prepared.template.cwd)?;
            }
        }
        Ok(prepared)
    }

    pub fn verify(&self) -> Result<()> {
        self.template.program.verify()?;
        for input in self.template.inputs.values() {
            input.verify()?;
        }
        if let Some((file, _)) = &self.reference {
            file.verify()?;
        }
        Ok(())
    }

    pub fn expand(&self, value: &str, sample: &ImageSample, directory: &Path) -> Result<String> {
        let path = |path: PathBuf| {
            path.into_os_string()
                .into_string()
                .map_err(|_| anyhow::anyhow!("template path is not UTF-8"))
        };
        let expanded = match value {
            "{prompt}" => sample.prompt.clone(),
            "{seed}" => sample.seed.to_string(),
            "{width}" => sample.width.to_string(),
            "{height}" => sample.height.to_string(),
            "{steps}" => sample
                .num_inference_steps
                .context("{steps} requires steps on every sample")?
                .to_string(),
            "{output}" => path(directory.join("output.png"))?,
            "{request_dir}" => path(directory.to_path_buf())?,
            "{sample_id}" => sample.id.clone(),
            "{reference}" => path(
                self.reference
                    .as_ref()
                    .context("{reference} requires edit dataset")?
                    .0
                    .path
                    .clone(),
            )?,
            _ if value.starts_with("{{") && value.ends_with("}}") => {
                value[1..value.len() - 1].to_owned()
            }
            _ if value.starts_with("{input:") && value.ends_with('}') => {
                let name = &value[7..value.len() - 1];
                path(
                    self.template
                        .inputs
                        .get(name)
                        .with_context(|| format!("unknown bound input {name}"))?
                        .path
                        .clone(),
                )?
            }
            _ if value.starts_with('{') && value.ends_with('}') => {
                bail!("unknown template token {value}")
            }
            _ => value.to_owned(),
        };
        ensure!(!expanded.contains('\0'), "command value contains NUL");
        Ok(expanded)
    }
}
