use anyhow::{bail, ensure, Context, Result};
use serde::Serialize;
use std::{collections::BTreeMap, path::PathBuf};

pub const USAGE:&str="verify_fast_h3_ffn --model-dir PATH | --fixture dyadic\n\
  --profile ffn|dit|dit-vsa [dit-vsa] --block N [0] --rows N [3] --iterations N [2]\n\
  --device N [0] --device-bytes N [1073741824] --host-bytes N [268435456]\n\
  --bootstrap-bytes N [268435456] --stream-bytes N [16777216] --native-bytes N [67108864]\n\
  --staging-bytes N [16777216] --max-source-bytes N [68719476736]\n\
  --max-metadata-bytes N [1048576] --max-tensors N [4096]\n\
  --chunk-rows N [whole] --max-plan-nodes N [8192]\n\
Requires cuda-cutlass-nvfp4 and a locked SM120 artifact; fixture also needs test-support.\n\
Real checkpoint: finite/deterministic execution and ownership checks, not a full numerical oracle.\n\
Fixture: complete FFN bitwise exact-subset oracle; not general quantization or model quality qualification.";

#[derive(Clone, Debug, Serialize)]
pub struct Arguments {
    pub model_dir: Option<PathBuf>,
    pub fixture: bool,
    pub profile: String,
    pub block: usize,
    pub rows: u64,
    pub chunk_rows: Option<std::num::NonZeroU64>,
    pub max_plan_nodes: usize,
    pub iterations: u32,
    pub device: u32,
    pub device_bytes: u64,
    pub host_bytes: usize,
    pub bootstrap_bytes: u64,
    pub stream_bytes: u64,
    pub native_bytes: u64,
    pub staging_bytes: usize,
    pub max_source_bytes: usize,
    pub max_metadata_bytes: usize,
    pub max_tensors: usize,
}
impl Arguments {
    pub fn parse(args: impl IntoIterator<Item = String>) -> Result<Option<Self>> {
        let mut iter = args.into_iter();
        let mut values = BTreeMap::new();
        while let Some(key) = iter.next() {
            if key == "--help" {
                ensure!(
                    values.is_empty() && iter.next().is_none(),
                    "--help must stand alone"
                );
                return Ok(None);
            }
            ensure!(
                [
                    "--model-dir",
                    "--fixture",
                    "--profile",
                    "--block",
                    "--rows",
                    "--chunk-rows",
                    "--max-plan-nodes",
                    "--iterations",
                    "--device",
                    "--device-bytes",
                    "--host-bytes",
                    "--bootstrap-bytes",
                    "--stream-bytes",
                    "--native-bytes",
                    "--staging-bytes",
                    "--max-source-bytes",
                    "--max-metadata-bytes",
                    "--max-tensors"
                ]
                .contains(&key.as_str()),
                "unknown argument {key}"
            );
            let value = iter
                .next()
                .with_context(|| format!("missing value for {key}"))?;
            ensure!(
                values.insert(key.clone(), value).is_none(),
                "duplicate {key}"
            );
        }
        let number = |key: &str, default: u64| -> Result<u64> {
            values.get(key).map_or(Ok(default), |v| {
                v.parse().with_context(|| format!("invalid {key}"))
            })
        };
        let positive = |key: &str, default: u64| -> Result<u64> {
            let v = number(key, default)?;
            ensure!(
                v > 0 && v < isize::MAX as u64,
                "{key} must fit a positive addressable extent"
            );
            Ok(v)
        };
        let fixture = match values.get("--fixture").map(String::as_str) {
            None => false,
            Some("dyadic") => true,
            Some(v) => bail!("unknown fixture {v}"),
        };
        let model_dir = values.get("--model-dir").map(PathBuf::from);
        ensure!(
            fixture != model_dir.is_some(),
            "select exactly one --model-dir or --fixture dyadic"
        );
        let profile = values
            .get("--profile")
            .cloned()
            .unwrap_or_else(|| "dit-vsa".into());
        ensure!(
            ["ffn", "dit", "dit-vsa"].contains(&profile.as_str()),
            "unknown deployment profile"
        );
        let result = Self {
            model_dir,
            fixture,
            profile,
            block: number("--block", 0)?.try_into()?,
            rows: positive("--rows", 3)?,
            chunk_rows: values
                .get("--chunk-rows")
                .map(|_| positive("--chunk-rows", 1).map(|n| std::num::NonZeroU64::new(n).unwrap()))
                .transpose()?,
            max_plan_nodes: positive("--max-plan-nodes", 8192)?.try_into()?,
            iterations: positive("--iterations", 2)?.try_into()?,
            device: number("--device", 0)?.try_into()?,
            device_bytes: positive("--device-bytes", 1 << 30)?,
            host_bytes: positive("--host-bytes", 256 << 20)?.try_into()?,
            bootstrap_bytes: positive("--bootstrap-bytes", 256 << 20)?,
            stream_bytes: positive("--stream-bytes", 16 << 20)?,
            native_bytes: positive("--native-bytes", 64 << 20)?,
            staging_bytes: positive("--staging-bytes", 16 << 20)?.try_into()?,
            max_source_bytes: positive("--max-source-bytes", 64 << 30)?.try_into()?,
            max_metadata_bytes: positive("--max-metadata-bytes", 1 << 20)?.try_into()?,
            max_tensors: positive("--max-tensors", 4096)?.try_into()?,
        };
        ensure!(
            !fixture || result.block == 0,
            "dyadic fixture contains only block0"
        );
        ensure!(
            result.staging_bytes <= result.host_bytes,
            "staging exceeds Host budget"
        );
        Ok(Some(result))
    }
}

#[test]
fn rejects_ambiguous_sources_and_hidden_options() {
    let parse = |s: &str| Arguments::parse(s.split_whitespace().map(str::to_owned));
    assert!(parse("--fixture dyadic --model-dir foo").is_err());
    assert!(parse("--model-dir foo --rows 0").is_err());
    assert!(parse("--model-dir foo --allow-fallback yes").is_err());
    assert_eq!(parse("--model-dir foo").unwrap().unwrap().rows, 3);
}
