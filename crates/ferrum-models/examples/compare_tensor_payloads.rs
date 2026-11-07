//! Bounded, untransformed comparison of a native F32-LE capture and one
//! safetensors tensor. Metrics are observations, never a quality acceptance gate.
use anyhow::{bail, ensure, Context, Result};
use half::{bf16, f16};
use safetensors::{Dtype, SafeTensors};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    ffi::OsString,
    fs::File,
    io::Read,
    path::{Path, PathBuf},
};

const USAGE: &str = "compare_tensor_payloads --left-raw F32_LE_FILE --left-shape D0,D1,... --reference-safetensors FILE --tensor NAME --maximum-input-bytes TOTAL_BYTES\nThe byte limit covers both complete input files combined. Shapes must match exactly. Zero reference norm makes relative_l2 null; either zero norm makes cosine null. No quality threshold is applied.";

struct Arguments {
    left_raw: PathBuf,
    left_shape: Vec<usize>,
    reference: PathBuf,
    tensor: String,
    maximum_input_bytes: usize,
}

impl Arguments {
    fn parse(arguments: impl IntoIterator<Item = OsString>) -> Result<Option<Self>> {
        let mut args = arguments.into_iter().peekable();
        if args.peek().is_some_and(|v| v == "--help" || v == "-h") {
            args.next();
            ensure!(args.next().is_none(), "--help accepts no other arguments");
            return Ok(None);
        }
        let mut values = BTreeMap::new();
        while let Some(flag) = args.next() {
            let flag = flag
                .into_string()
                .map_err(|_| anyhow::anyhow!("option names require UTF-8"))?;
            ensure!(
                matches!(
                    flag.as_str(),
                    "--left-raw"
                        | "--left-shape"
                        | "--reference-safetensors"
                        | "--tensor"
                        | "--maximum-input-bytes"
                ),
                "unknown option {flag}"
            );
            let value = args
                .next()
                .with_context(|| format!("missing value for {flag}"))?;
            ensure!(
                values.insert(flag.clone(), value).is_none(),
                "duplicate option {flag}"
            );
        }
        let mut take = |name: &str| {
            values
                .remove(name)
                .with_context(|| format!("required option {name}"))
        };
        let left_raw = PathBuf::from(take("--left-raw")?);
        let reference = PathBuf::from(take("--reference-safetensors")?);
        let text = |value: OsString| {
            value
                .into_string()
                .map_err(|_| anyhow::anyhow!("numeric options and tensor name require UTF-8"))
        };
        let left_shape = text(take("--left-shape")?)?
            .split(',')
            .map(str::parse)
            .collect::<std::result::Result<Vec<usize>, _>>()?;
        let tensor = text(take("--tensor")?)?;
        ensure!(!tensor.is_empty(), "tensor name must not be empty");
        let maximum_input_bytes = text(take("--maximum-input-bytes")?)?.parse::<usize>()?;
        ensure!(
            maximum_input_bytes > 0 && maximum_input_bytes <= isize::MAX as usize,
            "input byte limit must be positive and addressable"
        );
        let elements = elements(&left_shape)?;
        ensure!(
            elements
                .checked_mul(4)
                .is_some_and(|n| n <= maximum_input_bytes),
            "left shape exceeds input byte limit"
        );
        Ok(Some(Self {
            left_raw,
            left_shape,
            reference,
            tensor,
            maximum_input_bytes,
        }))
    }
}

fn elements(shape: &[usize]) -> Result<usize> {
    ensure!(
        !shape.is_empty() && !shape.contains(&0),
        "shape requires positive dimensions; no implicit scalar or singleton conversion"
    );
    shape.iter().try_fold(1usize, |n, d| {
        n.checked_mul(*d).context("shape product overflows")
    })
}

fn read_bounded(path: &Path, limit: usize, expected: Option<usize>) -> Result<Vec<u8>> {
    let mut file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let metadata = file.metadata()?;
    ensure!(
        metadata.is_file(),
        "{} is not a regular file",
        path.display()
    );
    let size = usize::try_from(metadata.len()).context("input is not addressable")?;
    ensure!(
        size <= limit,
        "{} has {size} bytes, remaining input limit is {limit}",
        path.display()
    );
    if let Some(expected) = expected {
        ensure!(
            size == expected,
            "{} has {size} bytes, exact shape requires {expected}",
            path.display()
        );
    }
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(size)
        .context("reserve bounded input")?;
    bytes.resize(size, 0);
    file.read_exact(&mut bytes)
        .with_context(|| format!("read exact input {}", path.display()))?;
    let mut extra = [0];
    ensure!(
        file.read(&mut extra)? == 0,
        "{} grew during bounded read",
        path.display()
    );
    Ok(bytes)
}

fn coordinate(mut index: usize, shape: &[usize]) -> Vec<usize> {
    let mut result = vec![0; shape.len()];
    for (out, dimension) in result.iter_mut().zip(shape).rev() {
        *out = index % dimension;
        index /= dimension;
    }
    result
}

#[derive(Debug, Serialize)]
struct WorstDifference {
    flat_index: usize,
    coordinate: Vec<usize>,
    left: f64,
    reference: f64,
}

#[derive(Debug, Serialize)]
struct Metrics {
    finite: bool,
    max_abs: f64,
    rmse: f64,
    /// sqrt(sum((left-reference)^2)) / sqrt(sum(reference^2)); null at zero norm.
    relative_l2: Option<f64>,
    /// dot(left,reference) / (norm(left)*norm(reference)); null at either zero norm.
    cosine: Option<f64>,
    worst: WorstDifference,
}

fn compare(left: &[u8], reference: &[u8], dtype: Dtype, shape: &[usize]) -> Result<Metrics> {
    let count = elements(shape)?;
    let width = match dtype {
        Dtype::F32 => 4,
        Dtype::F16 | Dtype::BF16 => 2,
        other => bail!("unsupported reference dtype {other:?}; expected F32, F16 or BF16"),
    };
    ensure!(
        left.len() % 4 == 0 && reference.len() % width == 0,
        "payload length is not divisible by its dtype width"
    );
    ensure!(
        count.checked_mul(4) == Some(left.len())
            && count.checked_mul(width) == Some(reference.len()),
        "payload lengths do not match the exact declared shape"
    );
    let (mut squared_error, mut left_squared, mut reference_squared, mut dot) =
        (0.0, 0.0, 0.0, 0.0);
    let mut max_abs = -1.0;
    let mut worst = WorstDifference {
        flat_index: 0,
        coordinate: vec![],
        left: 0.0,
        reference: 0.0,
    };
    for (index, (a, b)) in left
        .chunks_exact(4)
        .zip(reference.chunks_exact(width))
        .enumerate()
    {
        let a = f64::from(f32::from_le_bytes(a.try_into().unwrap()));
        let b = f64::from(match dtype {
            Dtype::F32 => f32::from_le_bytes(b.try_into().unwrap()),
            Dtype::F16 => f16::from_bits(u16::from_le_bytes(b.try_into().unwrap())).to_f32(),
            Dtype::BF16 => bf16::from_bits(u16::from_le_bytes(b.try_into().unwrap())).to_f32(),
            _ => unreachable!(),
        });
        ensure!(
            a.is_finite() && b.is_finite(),
            "non-finite value at index {index}, coordinate {:?}: left={a}, reference={b}",
            coordinate(index, shape)
        );
        let difference = a - b;
        squared_error += difference * difference;
        left_squared += a * a;
        reference_squared += b * b;
        dot += a * b;
        if difference.abs() > max_abs {
            max_abs = difference.abs();
            worst = WorstDifference {
                flat_index: index,
                coordinate: vec![],
                left: a,
                reference: b,
            };
        }
    }
    worst.coordinate = coordinate(worst.flat_index, shape);
    Ok(Metrics {
        finite: true,
        max_abs,
        rmse: (squared_error / count as f64).sqrt(),
        relative_l2: (reference_squared != 0.0)
            .then(|| squared_error.sqrt() / reference_squared.sqrt()),
        cosine: (left_squared != 0.0 && reference_squared != 0.0)
            .then(|| dot / (left_squared.sqrt() * reference_squared.sqrt())),
        worst,
    })
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn run(args: &Arguments) -> Result<serde_json::Value> {
    let left_path = args
        .left_raw
        .to_str()
        .context("JSON input path requires UTF-8")?;
    let reference_path = args
        .reference
        .to_str()
        .context("JSON input path requires UTF-8")?;
    let count = elements(&args.left_shape)?;
    let raw_bytes = count.checked_mul(4).context("F32 extent overflows")?;
    let left = read_bounded(&args.left_raw, args.maximum_input_bytes, Some(raw_bytes))?;
    let remaining = args
        .maximum_input_bytes
        .checked_sub(left.len())
        .context("combined input byte limit exceeded")?;
    let reference = read_bounded(&args.reference, remaining, None)?;
    let archive = SafeTensors::deserialize(&reference).context("validate reference safetensors")?;
    let tensor = archive
        .tensor(&args.tensor)
        .context("select exact reference tensor")?;
    ensure!(
        tensor.shape() == args.left_shape.as_slice(),
        "shape mismatch: left {:?}, reference {:?}; reshape/squeeze is not permitted",
        args.left_shape,
        tensor.shape()
    );
    let metrics = compare(&left, tensor.data(), tensor.dtype(), &args.left_shape)?;
    let left_sha256 = sha256(&left);
    Ok(serde_json::json!({
        "schema_version": 1,
        "maximum_input_bytes": args.maximum_input_bytes,
        "left": {"path": left_path, "encoding": "raw_f32_le", "dtype": "F32", "shape": args.left_shape, "elements": count, "file_bytes": left.len(), "payload_bytes": left.len(), "file_sha256": left_sha256, "payload_sha256": left_sha256},
        "reference": {"path": reference_path, "encoding": "safetensors", "tensor": args.tensor, "dtype": format!("{:?}", tensor.dtype()), "shape": tensor.shape(), "elements": count, "file_bytes": reference.len(), "payload_bytes": tensor.data().len(), "file_sha256": sha256(&reference), "payload_sha256": sha256(tensor.data())},
        "metrics": metrics
    }))
}

fn main() -> Result<()> {
    let Some(args) = Arguments::parse(std::env::args_os().skip(1))? else {
        println!("{USAGE}");
        return Ok(());
    };
    serde_json::to_writer_pretty(std::io::stdout().lock(), &run(&args)?)?;
    println!();
    Ok(())
}

#[cfg(test)]
#[path = "compare_tensor_payloads/tests.rs"]
mod tests;
