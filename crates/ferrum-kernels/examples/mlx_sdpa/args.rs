use super::{fixture::Shape, Result};
use serde::Serialize;
use std::{collections::BTreeMap, path::PathBuf};

pub const USAGE: &str = "bench_mlx_sdpa --bridge <absolute dylib> --query-length N --key-length N --heads N --head-dim N [--batch 1] [--warmups 1] [--iterations 3] [--maximum-host-bytes 536870912] [--expected-mlx-version 0.32.2]\nBF16, noncausal, no mask, scale=1/8. Native bridge requires an existing Metal-enabled MLX installation. No GPU work is performed by --help.";

#[derive(Debug, Serialize)]
pub struct Args {
    pub bridge: PathBuf,
    pub shape: Shape,
    pub warmups: usize,
    pub iterations: usize,
    pub maximum_host_bytes: usize,
    pub expected_mlx_version: String,
}

impl Args {
    pub fn parse(args: impl IntoIterator<Item = String>) -> Result<Option<Self>> {
        let mut args = args.into_iter();
        let mut values = BTreeMap::new();
        while let Some(key) = args.next() {
            if key == "--help" && values.is_empty() && args.next().is_none() {
                return Ok(None);
            }
            if ![
                "--bridge",
                "--query-length",
                "--key-length",
                "--heads",
                "--head-dim",
                "--batch",
                "--warmups",
                "--iterations",
                "--maximum-host-bytes",
                "--expected-mlx-version",
            ]
            .contains(&key.as_str())
            {
                return Err(format!("unknown option {key}; {USAGE}").into());
            }
            let value = args
                .next()
                .ok_or_else(|| format!("missing value for {key}"))?;
            if values.insert(key.clone(), value).is_some() {
                return Err(format!("duplicate option {key}").into());
            }
        }
        let required = |key: &str| -> Result<String> {
            values
                .get(key)
                .cloned()
                .ok_or_else(|| format!("required option {key}; {USAGE}").into())
        };
        let number = |key: &str, default: Option<usize>| -> Result<usize> {
            match values.get(key) {
                Some(v) => Ok(v.parse()?),
                None => default.ok_or_else(|| format!("required option {key}").into()),
            }
        };
        let shape = Shape {
            batch: number("--batch", Some(1))?,
            queries: number("--query-length", None)?,
            keys: number("--key-length", None)?,
            heads: number("--heads", None)?,
            dim: number("--head-dim", None)?,
        };
        shape.validate()?;
        let result = Self {
            bridge: PathBuf::from(required("--bridge")?),
            shape,
            warmups: number("--warmups", Some(1))?,
            iterations: number("--iterations", Some(3))?,
            maximum_host_bytes: number("--maximum-host-bytes", Some(512 * 1024 * 1024))?,
            expected_mlx_version: values
                .get("--expected-mlx-version")
                .cloned()
                .unwrap_or_else(|| "0.32.2".into()),
        };
        if !result.bridge.is_absolute() || result.expected_mlx_version.is_empty() {
            return Err("bridge must be absolute and expected MLX version nonempty".into());
        }
        if result.iterations == 0
            || result
                .warmups
                .checked_add(result.iterations)
                .is_none_or(|n| n > 10_000)
        {
            return Err("iterations must be positive; warmups+iterations capacity is 10000".into());
        }
        let required = shape.host_bytes()?.max(super::fixture::probe_host_bytes()?);
        if required > result.maximum_host_bytes {
            return Err(format!(
                "Rust tensor/oracle backing needs {required} bytes, limit is {}",
                result.maximum_host_bytes
            )
            .into());
        }
        Ok(Some(result))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arguments() -> Vec<String> {
        let library = std::env::current_dir().unwrap().join("bridge-library");
        [
            "--bridge".into(),
            library.to_string_lossy().into_owned(),
            "--query-length".into(),
            "33".into(),
            "--key-length".into(),
            "17".into(),
            "--heads".into(),
            "3".into(),
            "--head-dim".into(),
            "128".into(),
        ]
        .into()
    }

    #[test]
    fn typed_arguments_reject_capacity_shape_and_duplicate_errors() {
        assert_eq!(Args::parse(arguments()).unwrap().unwrap().shape.queries, 33);
        for extra in [
            vec!["--iterations", "0"],
            vec!["--maximum-host-bytes", "1"],
            vec!["--heads", "32"],
            vec!["--unknown", "1"],
        ] {
            let mut a = arguments();
            a.extend(extra.into_iter().map(str::to_owned));
            assert!(Args::parse(a).is_err());
        }
        let mut a = arguments();
        a[3] = "0".into();
        assert!(Args::parse(a.clone()).is_err());
        a[3] = usize::MAX.to_string();
        assert!(Args::parse(a).is_err());
    }
}
