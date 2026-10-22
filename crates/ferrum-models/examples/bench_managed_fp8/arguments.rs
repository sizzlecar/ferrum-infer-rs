use super::*;

pub(super) const USAGE: &str = "bench_managed_fp8 --device ORDINAL --m M --n N --k K --warmup COUNT --iterations COUNT --device-bytes N --host-bytes N --bootstrap-bytes N --stream-bytes N --native-bytes N --staging-bytes N
Requires cuda-cutlass-fp8 and the configured locked native operator set.
All budgets are explicit bytes. One stream, one flight, one prepared native shape; weights initialize once.
JSON Lines on stdout: declaration, each component-wall sample, successful retirement.
Measured session.execute includes admission, input upload, encoding, submit/wait and output readback.
It excludes compile/init, input construction, output validation and logging. No kernel or quant/GEMM split timing.
Data are generated finite dyadic tensors, not checkpoint weights or an image-quality benchmark.";

#[derive(Clone, Debug, Serialize)]
pub(super) struct Arguments {
    pub device: usize,
    pub shape: Shape,
    pub warmup: u32,
    pub iterations: u32,
    pub device_bytes: u64,
    pub host_bytes: usize,
    pub bootstrap_bytes: u64,
    pub stream_bytes: u64,
    pub native_bytes: u64,
    pub staging_bytes: usize,
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
            let value = values.next().context("every option needs one value")?;
            ensure!(
                options.insert(key.clone(), value).is_none(),
                "duplicate option {key}"
            );
        }
        let mut number = |key: &str| -> Result<u64> {
            options
                .remove(key)
                .with_context(|| format!("missing {key}; {USAGE}"))?
                .parse::<u64>()
                .with_context(|| format!("invalid integer {key}"))
        };
        let result = Self {
            device: usize::try_from(number("--device")?)?,
            shape: Shape {
                m: number("--m")?,
                n: number("--n")?,
                k: number("--k")?,
            },
            warmup: u32::try_from(number("--warmup")?)?,
            iterations: u32::try_from(number("--iterations")?)?,
            device_bytes: number("--device-bytes")?,
            host_bytes: usize::try_from(number("--host-bytes")?)?,
            bootstrap_bytes: number("--bootstrap-bytes")?,
            stream_bytes: number("--stream-bytes")?,
            native_bytes: number("--native-bytes")?,
            staging_bytes: usize::try_from(number("--staging-bytes")?)?,
        };
        ensure!(
            options.is_empty(),
            "unknown options: {:?}",
            options.keys().collect::<Vec<_>>()
        );
        result.validate()?;
        Ok(Some(result))
    }
    pub fn validate(&self) -> Result<Extents> {
        let e = self.shape.extents()?;
        ensure!(
            self.device <= i32::MAX as usize,
            "device ordinal exceeds native i32"
        );
        ensure!(self.iterations > 0, "iterations must be positive");
        ensure!(
            self.host_bytes > 0
                && self.host_bytes <= isize::MAX as usize
                && self.staging_bytes > 0
                && self.staging_bytes <= isize::MAX as usize,
            "Host and staging budgets must be positive addressable bytes"
        );
        ensure!(
            [
                self.device_bytes,
                self.bootstrap_bytes,
                self.stream_bytes,
                self.native_bytes
            ]
            .iter()
            .all(|&v| v > 0),
            "device and opaque budgets must be positive"
        );
        let opaque = self
            .bootstrap_bytes
            .checked_add(self.stream_bytes)
            .and_then(|v| v.checked_add(self.native_bytes))
            .context("opaque budget overflow")?;
        ensure!(
            opaque < self.device_bytes,
            "device total must exceed declared opaque allowances"
        );
        // Input is created once before static upload. WeightBacking drops after
        // initialization; subsequent executions retain input plus one output.
        let init = e
            .weights
            .checked_add(self.staging_bytes)
            .context("Host initialization overflow")?;
        let required = e
            .input
            .checked_add(init.max(e.output))
            .context("Host peak overflow")?;
        ensure!(
            self.host_bytes >= required,
            "Host budget {} is below declared backing peak {required}",
            self.host_bytes
        );
        Ok(e)
    }
    #[cfg(feature = "cuda-cutlass-fp8")]
    pub fn host_limits(&self) -> Result<HostMemoryLimits> {
        let e = self.validate()?;
        Ok(HostMemoryLimits::new(
            self.host_bytes,
            HostMemoryClassLimits {
                input_preprocess_bytes: e.input,
                output_bytes: e.output,
                weight_backing_bytes: e.weights,
                transfer_staging_bytes: self.staging_bytes,
            },
        )?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn valid() -> Vec<String> {
        [
            "--device",
            "0",
            "--m",
            "3",
            "--n",
            "32",
            "--k",
            "16",
            "--warmup",
            "0",
            "--iterations",
            "2",
            "--device-bytes",
            "1073741824",
            "--host-bytes",
            "4096",
            "--bootstrap-bytes",
            "1048576",
            "--stream-bytes",
            "1048576",
            "--native-bytes",
            "1048576",
            "--staging-bytes",
            "512",
        ]
        .into_iter()
        .map(str::to_owned)
        .collect()
    }
    #[test]
    fn explicit_parameters_and_true_capacity_boundaries() {
        let parsed = Arguments::parse(valid()).unwrap().unwrap();
        assert_eq!(parsed.shape.extents().unwrap().input, 96);
        for (key, value) in [
            ("--m", "0"),
            ("--n", "17"),
            ("--k", "1048576"),
            ("--iterations", "0"),
            ("--host-bytes", "100"),
            ("--device-bytes", "3145728"),
            ("--native-bytes", "0"),
        ] {
            let mut v = valid();
            let i = v.iter().position(|s| s == key).unwrap();
            v[i + 1] = value.into();
            assert!(Arguments::parse(v).is_err(), "{key}");
        }
        let mut v = valid();
        v.extend(["--m".into(), "3".into()]);
        assert!(Arguments::parse(v).is_err());
        let mut v = valid();
        v.extend(["--mystery".into(), "3".into()]);
        assert!(Arguments::parse(v).is_err());
        let mut v = valid();
        v.pop();
        assert!(Arguments::parse(v).is_err());
        assert!(Shape {
            m: u64::MAX,
            n: 16,
            k: 16
        }
        .extents()
        .is_err());
    }
}
