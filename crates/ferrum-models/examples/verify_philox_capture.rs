//! CPU-only verification of an immutable captured CUDA Philox F32 stream.
//! Tensor names are concatenated in caller order, without copying payloads.
#[path = "../tests/support/philox_normal_oracle.rs"]
mod oracle;

use anyhow::{bail, ensure, Context, Result};
use ferrum_quantization::SafetensorsArchive;
use ferrum_runtime::PhiloxSeedOffset;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeSet,
    ffi::OsString,
    fs::{File, OpenOptions},
    io::{Read, Write},
    path::{Path, PathBuf},
};

const USAGE: &str = "verify_philox_capture --capture FILE.safetensors --tensor NAME [--tensor NEXT_NAME ...] --seed U64 --offset U64 --launch-blocks U32 --maximum-input-bytes N --output NEW_JSON\nTensor order is significant: one CUDA stream is concatenated across all selected F32 tensors. The complete input file, including unselected tensors, must fit the explicit byte cap. Launch blocks must come from actual launch evidence, never a hardware-name guess. Every element uses the existing independent normal4 oracle and its 2e-5*(1+abs(expected)) bound. A mismatch report is preserved before nonzero exit; this is not Torch parity or a distribution-quality test.";

struct Arguments {
    capture: PathBuf,
    tensors: Vec<String>,
    seed: u64,
    offset: u64,
    launch_blocks: u32,
    maximum_input_bytes: usize,
    output: PathBuf,
}
impl Arguments {
    fn parse(args: impl IntoIterator<Item = OsString>) -> Result<Option<Self>> {
        let mut args = args.into_iter().peekable();
        if args.peek().is_some_and(|s| s == "--help" || s == "-h") {
            args.next();
            ensure!(args.next().is_none(), "help accepts no other options");
            return Ok(None);
        }
        let mut values = std::collections::BTreeMap::new();
        let mut tensors = Vec::new();
        let mut names = BTreeSet::new();
        while let Some(flag) = args.next() {
            let flag = flag
                .into_string()
                .map_err(|_| anyhow::anyhow!("option must be UTF-8"))?;
            ensure!(
                matches!(
                    flag.as_str(),
                    "--capture"
                        | "--tensor"
                        | "--seed"
                        | "--offset"
                        | "--launch-blocks"
                        | "--maximum-input-bytes"
                        | "--output"
                ),
                "unknown option {flag}"
            );
            let value = args
                .next()
                .with_context(|| format!("missing value for {flag}"))?;
            if flag == "--tensor" {
                let name = value
                    .into_string()
                    .map_err(|_| anyhow::anyhow!("tensor name must be UTF-8"))?;
                ensure!(
                    !name.is_empty() && names.insert(name.clone()),
                    "empty or duplicate tensor name"
                );
                tensors.push(name);
            } else {
                ensure!(
                    values.insert(flag.clone(), value).is_none(),
                    "duplicate option {flag}"
                );
            }
        }
        ensure!(
            !tensors.is_empty(),
            "at least one ordered --tensor is required"
        );
        let mut take = |key: &str| {
            values
                .remove(key)
                .with_context(|| format!("required {key}"))
        };
        let capture = take("--capture")?.into();
        let output = take("--output")?.into();
        let text = |v: OsString| {
            v.into_string()
                .map_err(|_| anyhow::anyhow!("number must be UTF-8"))
        };
        let seed = text(take("--seed")?)?.parse()?;
        let offset = text(take("--offset")?)?.parse()?;
        let launch_blocks = text(take("--launch-blocks")?)?.parse()?;
        let maximum_input_bytes = text(take("--maximum-input-bytes")?)?.parse()?;
        ensure!(launch_blocks > 0, "launch blocks must be positive");
        ensure!(
            maximum_input_bytes > 0 && maximum_input_bytes <= isize::MAX as usize,
            "file cap must be positive and addressable"
        );
        Ok(Some(Self {
            capture,
            tensors,
            seed,
            offset,
            launch_blocks,
            maximum_input_bytes,
            output,
        }))
    }
}

fn file_identity(path: &Path, cap: usize) -> Result<(usize, String)> {
    let mut file = File::open(path)?;
    let meta = file.metadata()?;
    let size = usize::try_from(meta.len())?;
    ensure!(
        meta.is_file() && size <= cap,
        "complete capture file exceeds cap or is not regular"
    );
    let mut hash = Sha256::new();
    let mut buffer = [0u8; 65536];
    let mut read = 0usize;
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        read = read.checked_add(n).context("file size overflows")?;
        ensure!(read <= size, "capture file grew during bounded hashing");
        hash.update(&buffer[..n]);
    }
    ensure!(read == size, "capture file changed during bounded hashing");
    Ok((size, format!("{:x}", hash.finalize())))
}

fn elements(shape: &[u64]) -> Result<usize> {
    ensure!(
        !shape.is_empty() && !shape.contains(&0),
        "selected tensors require nonempty positive shape"
    );
    shape.iter().try_fold(1usize, |n, &d| {
        n.checked_mul(usize::try_from(d)?)
            .context("shape product overflows")
    })
}

#[derive(Default)]
struct Moments {
    count: u64,
    mean: f64,
    m2: f64,
    min: f64,
    max: f64,
}
impl Moments {
    fn push(&mut self, value: f32) {
        let value = f64::from(value);
        if self.count == 0 {
            self.min = value;
            self.max = value;
        }
        self.count += 1;
        let delta = value - self.mean;
        self.mean += delta / self.count as f64;
        self.m2 += delta * (value - self.mean);
        self.min = self.min.min(value);
        self.max = self.max.max(value);
    }
    fn report(&self) -> serde_json::Value {
        serde_json::json!({"count":self.count,"mean":self.mean,"population_std":(self.m2 / self.count as f64).max(0.0).sqrt(),"min":self.min,"max":self.max})
    }
}

#[derive(Clone, Serialize)]
struct Difference {
    tensor: String,
    tensor_index: usize,
    global_index: u64,
    thread: u64,
    round: u64,
    component: usize,
    actual: f32,
    expected: f32,
    absolute_error: f64,
    bound: f32,
}
struct Analysis {
    report: serde_json::Value,
    mismatch_count: u64,
}

fn analyze(args: &Arguments) -> Result<Analysis> {
    oracle::known_answers();
    ensure!(
        args.launch_blocks > 0 && !args.tensors.is_empty(),
        "missing launch or tensors"
    );
    let (file_bytes, file_sha256) = file_identity(&args.capture, args.maximum_input_bytes)?;
    let directory = args
        .capture
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let basename = args
        .capture
        .file_name()
        .and_then(|s| s.to_str())
        .context("capture requires a UTF-8 safetensors basename")?;
    let archive = SafetensorsArchive::open_single_file(directory, basename)?;
    let mut selected = Vec::with_capacity(args.tensors.len());
    let mut total = 0usize;
    let mut names = BTreeSet::new();
    for name in &args.tensors {
        ensure!(names.insert(name), "duplicate tensor selection");
        let tensor = archive.tensor(name)?;
        ensure!(
            tensor.dtype() == safetensors::Dtype::F32,
            "{name}: only actual F32 capture is supported"
        );
        let count = elements(tensor.shape())?;
        ensure!(
            count.checked_mul(4) == Some(tensor.bytes().len()),
            "{name}: shape/byte extent differs"
        );
        total = total
            .checked_add(count)
            .context("selected element count overflows")?;
        selected.push(tensor);
    }
    let total_u64 = u64::try_from(total)?;
    let stride = u64::from(args.launch_blocks)
        .checked_mul(256)
        .context("CUDA stride overflows")?;
    let round_span = stride
        .checked_mul(4)
        .context("normal4 round span overflows")?;
    let state = PhiloxSeedOffset::new(args.seed, args.offset);
    let mut actual_stats = Moments::default();
    let mut oracle_stats = Moments::default();
    let mut identities = Vec::with_capacity(selected.len());
    let mut mismatch_count = 0u64;
    let mut first_mismatch = None;
    let mut maximum_error: Option<Difference> = None;
    let mut error_sq = 0f64;
    let mut oracle_sq = 0f64;
    let mut global = 0u64;
    for tensor in &selected {
        let start = global;
        for (local, bytes) in tensor.bytes().chunks_exact(4).enumerate() {
            let actual = f32::from_le_bytes(bytes.try_into().unwrap());
            ensure!(
                actual.is_finite(),
                "nonfinite actual tensor={} local={local} global={global}",
                tensor.external_name()
            );
            let thread = global % stride;
            let round = global / round_span;
            let component = ((global / stride) % 4) as usize;
            let expected = oracle::normal4(state, thread, round)[component];
            ensure!(expected.is_finite(), "nonfinite oracle at global={global}");
            let bound = 2e-5_f32 * (1.0 + expected.abs());
            let error = f64::from(actual) - f64::from(expected);
            let difference = Difference {
                tensor: tensor.external_name().into(),
                tensor_index: local,
                global_index: global,
                thread,
                round,
                component,
                actual,
                expected,
                absolute_error: error.abs(),
                bound,
            };
            if (actual - expected).abs() > bound {
                mismatch_count += 1;
                if first_mismatch.is_none() {
                    first_mismatch = Some(difference.clone());
                }
            }
            if maximum_error
                .as_ref()
                .is_none_or(|d| error.abs() > d.absolute_error)
            {
                maximum_error = Some(difference);
            }
            actual_stats.push(actual);
            oracle_stats.push(expected);
            error_sq += error * error;
            oracle_sq += f64::from(expected).powi(2);
            global += 1;
        }
        identities.push(serde_json::json!({"name":tensor.external_name(),"dtype":"F32","shape":tensor.shape(),"bytes":tensor.bytes().len(),"payload_sha256":format!("{:x}",Sha256::digest(tensor.bytes())),"global_start":start,"global_end":global}));
    }
    ensure!(
        global == total_u64 && global > 0,
        "selected iteration extent differs"
    );
    let max_abs = maximum_error.as_ref().unwrap().absolute_error;
    Ok(Analysis {
        mismatch_count,
        report: serde_json::json!({
            "schema":"ferrum.philox_capture.v1","complete":true,"known_answers_checked":true,"finite":true,
            "file":{"path":args.capture,"bytes":file_bytes,"sha256":file_sha256,"maximum_input_bytes":args.maximum_input_bytes,"contract":"complete immutable capture file cap; mmap payloads are borrowed without whole-tensor copies"},
            "selected_tensors":identities,"seed":args.seed,"offset":args.offset,"launch_blocks":args.launch_blocks,
            "threads_per_block":256,"stride":stride,"elements":global,
            "indexing":"ordered concatenation; thread=i%stride, round=i/(4*stride), component=(i/stride)%4",
            "bound":"abs(actual-expected) <= 2e-5_f32*(1+abs(expected)); unchanged independent CUDA-normal oracle contract",
            "actual":actual_stats.report(),"oracle":oracle_stats.report(),
            "mismatch_count":mismatch_count,"all_within_bound":mismatch_count==0,"first_mismatch":first_mismatch,"maximum_error":maximum_error,
            "max_abs":max_abs,"rmse":(error_sq/global as f64).sqrt(),"relative_l2":if oracle_sq>0.0 {Some((error_sq/oracle_sq).sqrt())}else{None},
            "scope":"complete CPU comparison against existing independent Random123/libm oracle; not a new GPU draw, Torch RNG parity, or distribution-quality acceptance"
        }),
    })
}

fn execute(args: &Arguments) -> Result<()> {
    let analysis = analyze(args)?;
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args.output)?;
    serde_json::to_writer_pretty(&mut file, &analysis.report)?;
    file.write_all(b"\n")?;
    file.sync_all()?;
    if analysis.mismatch_count != 0 {
        bail!(
            "{} elements exceed the existing oracle bound; report preserved at {}",
            analysis.mismatch_count,
            args.output.display()
        );
    }
    Ok(())
}
fn main() -> Result<()> {
    if let Some(args) = Arguments::parse(std::env::args_os().skip(1))? {
        execute(&args)
    } else {
        println!("{USAGE}");
        Ok(())
    }
}

#[cfg(test)]
#[path = "verify_philox_capture/tests.rs"]
mod tests;
