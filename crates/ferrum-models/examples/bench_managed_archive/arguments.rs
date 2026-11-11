use anyhow::{ensure, Context, Result};
use ferrum_quantization::{Bf16F16ConversionPolicy, DenseArchiveSourceTouch};
use serde::Serialize;
use std::{collections::BTreeMap, num::NonZeroUsize, path::PathBuf};

pub(super) const USAGE: &str = "bench_managed_archive --semantic-root DIR --tokenizer-root DIR --weights-file FILE --numerical-profile PROFILE --host-bytes BYTES --iterations COUNT --mode baseline|diagnostic --touch-stride-bytes N [--backing fresh|reuse|bounded-stream] [--host-alignment N] [--component-id ID] [--hash-output true|false] [--conversion-workers N]
CPU only; all input paths are explicit local product sources. JSON Lines on stdout.
--backing bounded-stream requires diagnostic mode/touch stride 0. It uses the production BF16-to-F16 range materializer, a 64MiB physical slot ceiling, and one slot across all full-model passes.
--conversion-workers defaults to 1; values 2..16 require bounded-stream. Only sufficiently large continuous BF16 parts use scoped workers; full fallbacks stay unchanged. All workers join before publishing a range.
Bounded-stream records process-wide RUSAGE_SELF deltas around each materialize_range/full fallback call. Run it as an otherwise idle CPU-only process; faults are counts, not fault duration.
Checksum/hash reads are timed separately. Source loads, conversion and destination writes remain combined inside the original range call; this is not a source-only read or isolated conversion benchmark.
Baseline requires touch stride 0 and runs the same conversion arithmetic without phase observers.
Diagnostic stride 0 preserves ordinary mmap access; e.g. 4096 explicitly touches source pages before allocation/conversion.
No cache is cleared and no cold-read guarantee is made. Host alignment defaults to 1; backing defaults to fresh.
Reuse creates one charged maximum-component slot per iteration, including its first allocation and final release in installation time.
Diagnostic conversion includes direct composite packing, destination first-touch and any remaining source faults.
Run baseline and diagnostic separately to assess observer overhead. Hashing defaults to false.
--hash-output true selects a separate correctness-validation run and suppresses the performance installation total.";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum Mode {
    Baseline,
    Diagnostic,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum Backing {
    Fresh,
    Reuse,
    BoundedStream,
}

#[derive(Clone, Debug, Serialize)]
pub(super) struct Arguments {
    pub semantic_root: PathBuf,
    pub tokenizer_root: PathBuf,
    pub weights_file: PathBuf,
    pub numerical_profile: String,
    pub host_bytes: usize,
    pub host_alignment: usize,
    pub iterations: usize,
    pub mode: Mode,
    pub backing: Backing,
    pub touch_stride_bytes: usize,
    pub component_id: Option<String>,
    pub hash_output: bool,
    pub conversion_workers: NonZeroUsize,
}

impl Arguments {
    pub fn parse(values: impl IntoIterator<Item = String>) -> Result<Option<Self>> {
        let mut values = values.into_iter().peekable();
        if values.peek().is_some_and(|s| s == "--help" || s == "-h") {
            values.next();
            ensure!(values.next().is_none(), "--help accepts no other arguments");
            return Ok(None);
        }
        let mut options = BTreeMap::new();
        while let Some(key) = values.next() {
            ensure!(key.starts_with("--"), "expected option, got {key}");
            let value = values
                .next()
                .with_context(|| format!("missing value for {key}"))?;
            ensure!(
                options.insert(key.clone(), value).is_none(),
                "duplicate option {key}"
            );
        }
        let mut take = |key: &str| {
            options
                .remove(key)
                .with_context(|| format!("missing {key}; {USAGE}"))
        };
        let semantic_root = take("--semantic-root")?.into();
        let tokenizer_root = take("--tokenizer-root")?.into();
        let weights_file = take("--weights-file")?.into();
        let numerical_profile = take("--numerical-profile")?;
        let host_bytes: usize = take("--host-bytes")?.parse()?;
        let iterations: usize = take("--iterations")?.parse()?;
        let mode = match take("--mode")?.as_str() {
            "baseline" => Mode::Baseline,
            "diagnostic" => Mode::Diagnostic,
            value => anyhow::bail!("unsupported --mode {value}"),
        };
        let touch_stride_bytes: usize = take("--touch-stride-bytes")?.parse()?;
        let backing = match options.remove("--backing").as_deref().unwrap_or("fresh") {
            "fresh" => Backing::Fresh,
            "reuse" => Backing::Reuse,
            "bounded-stream" => Backing::BoundedStream,
            value => anyhow::bail!("unsupported --backing {value}"),
        };
        let host_alignment: usize = options
            .remove("--host-alignment")
            .map(|s| s.parse())
            .transpose()?
            .unwrap_or(1);
        let hash_output = options
            .remove("--hash-output")
            .map(|s| s.parse())
            .transpose()?
            .unwrap_or(false);
        let conversion_workers = options
            .remove("--conversion-workers")
            .map(|s| s.parse::<NonZeroUsize>())
            .transpose()?
            .unwrap_or(NonZeroUsize::MIN);
        Bf16F16ConversionPolicy::new(conversion_workers)?;
        let component_id = options.remove("--component-id");
        ensure!(
            options.is_empty(),
            "unknown options: {:?}",
            options.keys().collect::<Vec<_>>()
        );
        ensure!(
            host_bytes > 0 && host_bytes <= isize::MAX as usize,
            "Host budget must be positive addressable bytes"
        );
        ensure!(
            host_alignment.is_power_of_two() && host_alignment <= isize::MAX as usize,
            "Host alignment must be an addressable power of two"
        );
        ensure!(iterations > 0, "iterations must be positive");
        ensure!(
            conversion_workers.get() == 1 || backing == Backing::BoundedStream,
            "multiple conversion workers require bounded-stream"
        );
        ensure!(
            backing != Backing::BoundedStream
                || (mode == Mode::Diagnostic && touch_stride_bytes == 0),
            "bounded-stream requires diagnostic mode and touch stride 0"
        );
        ensure!(
            mode != Mode::Baseline || touch_stride_bytes == 0,
            "baseline must use touch stride 0"
        );
        ensure!(
            !numerical_profile.is_empty() && component_id.as_ref().is_none_or(|id| !id.is_empty()),
            "profile and supplied component ID must be nonempty"
        );
        Ok(Some(Self {
            semantic_root,
            tokenizer_root,
            weights_file,
            numerical_profile,
            host_bytes,
            host_alignment,
            iterations,
            mode,
            backing,
            touch_stride_bytes,
            component_id,
            hash_output,
            conversion_workers,
        }))
    }

    pub fn source_touch(&self) -> DenseArchiveSourceTouch {
        NonZeroUsize::new(self.touch_stride_bytes).map_or(
            DenseArchiveSourceTouch::AsMapped,
            DenseArchiveSourceTouch::Stride,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn args() -> Vec<String> {
        [
            "--semantic-root",
            "semantic",
            "--tokenizer-root",
            "tokens",
            "--weights-file",
            "weights.safetensors",
            "--numerical-profile",
            "qwen3_5.f16",
            "--host-bytes",
            "4096",
            "--iterations",
            "2",
            "--mode",
            "diagnostic",
            "--touch-stride-bytes",
            "4096",
        ]
        .into_iter()
        .map(str::to_owned)
        .collect()
    }
    #[test]
    fn rejects_non_addressable_budgets_and_ambiguous_measurement_modes() {
        let good = Arguments::parse(args()).unwrap().unwrap();
        assert_eq!(good.backing, Backing::Fresh);
        assert!(
            matches!(good.source_touch(), DenseArchiveSourceTouch::Stride(n) if n.get() == 4096)
        );
        for (key, value) in [
            ("--host-bytes", "0"),
            ("--host-bytes", "18446744073709551615"),
            ("--iterations", "0"),
            ("--mode", "baseline"),
            ("--mode", "automatic"),
        ] {
            let mut values = args();
            let i = values.iter().position(|s| s == key).unwrap();
            values[i + 1] = value.into();
            assert!(Arguments::parse(values).is_err(), "{key}={value}");
        }
        for pair in [
            ["--host-alignment", "3"],
            ["--unknown", "1"],
            ["--host-bytes", "4096"],
            ["--backing", "automatic"],
        ] {
            let mut values = args();
            values.extend(pair.map(str::to_owned));
            assert!(Arguments::parse(values).is_err());
        }
        let mut baseline = args();
        let i = baseline.iter().position(|s| s == "--mode").unwrap();
        baseline[i + 1] = "baseline".into();
        let i = baseline
            .iter()
            .position(|s| s == "--touch-stride-bytes")
            .unwrap();
        baseline[i + 1] = "0".into();
        assert_eq!(
            Arguments::parse(baseline).unwrap().unwrap().mode,
            Mode::Baseline
        );
        let mut reuse = args();
        reuse.extend(["--backing", "reuse", "--hash-output", "true"].map(str::to_owned));
        let reuse = Arguments::parse(reuse).unwrap().unwrap();
        assert_eq!(reuse.backing, Backing::Reuse);
        assert!(reuse.hash_output);
    }

    #[test]
    fn bounded_stream_requires_unmodified_range_access() {
        let mut values = args();
        values.extend(["--backing", "bounded-stream"].map(str::to_owned));
        assert!(Arguments::parse(values.clone()).is_err());
        let index = values
            .iter()
            .position(|s| s == "--touch-stride-bytes")
            .unwrap();
        values[index + 1] = "0".into();
        assert_eq!(
            Arguments::parse(values.clone()).unwrap().unwrap().backing,
            Backing::BoundedStream
        );
        let index = values.iter().position(|s| s == "--mode").unwrap();
        values[index + 1] = "baseline".into();
        assert!(Arguments::parse(values).is_err());
    }

    #[test]
    fn parallel_conversion_is_explicit_bounded_and_only_for_range_mode() {
        assert_eq!(
            Arguments::parse(args())
                .unwrap()
                .unwrap()
                .conversion_workers
                .get(),
            1
        );
        let mut parallel = args();
        let position = parallel
            .iter()
            .position(|s| s == "--touch-stride-bytes")
            .unwrap();
        parallel[position + 1] = "0".into();
        parallel.extend(
            ["--backing", "bounded-stream", "--conversion-workers", "4"].map(str::to_owned),
        );
        assert_eq!(
            Arguments::parse(parallel.clone())
                .unwrap()
                .unwrap()
                .conversion_workers
                .get(),
            4
        );
        let position = parallel
            .iter()
            .position(|s| s == "--conversion-workers")
            .unwrap();
        for invalid in ["0", "17"] {
            let mut values = parallel.clone();
            values[position + 1] = invalid.into();
            assert!(Arguments::parse(values).is_err());
        }
        let position = parallel.iter().position(|s| s == "--backing").unwrap();
        for backing in ["fresh", "reuse"] {
            let mut values = parallel.clone();
            values[position + 1] = backing.into();
            assert!(Arguments::parse(values).is_err());
        }
    }
}
