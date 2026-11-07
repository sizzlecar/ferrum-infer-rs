//! Bounded, untransformed comparison of a raw F32 or floating safetensors
//! capture and one reference safetensors tensor. Metrics are observations,
//! never a quality acceptance gate.
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

const USAGE: &str = "compare_tensor_payloads (--left-raw F32_LE_FILE --left-shape D0,D1,... | --left-safetensors FILE --left-tensor NAME) --reference-safetensors FILE --tensor NAME --maximum-input-bytes TOTAL_BYTES [--group-axis AXIS] [--left-axis0-range START:END] [--reference-axis0-range START:END]\nSafetensors on either side supports F32, F16 or BF16; raw left remains F32. The byte limit covers both complete input files combined, including unselected rows. Optional nonempty half-open axis-0 ranges are independent and preserve rank. Selected shapes must match exactly; reported coordinates/group indices are relative to the selected tensors. Zero reference norm makes relative_l2 null; either zero norm makes cosine null. No quality threshold is applied.";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
struct Axis0Range {
    start: usize,
    end: usize,
}
impl Axis0Range {
    fn parse(value: &str) -> Result<Self> {
        let (start, end) = value
            .split_once(':')
            .context("axis-0 range requires START:END")?;
        ensure!(
            !start.is_empty()
                && !end.is_empty()
                && start.bytes().all(|v| v.is_ascii_digit())
                && end.bytes().all(|v| v.is_ascii_digit()),
            "axis-0 range endpoints require unsigned decimal integers"
        );
        let result = Self {
            start: start.parse()?,
            end: end.parse()?,
        };
        ensure!(
            result.start < result.end,
            "axis-0 range must be nonempty and increasing"
        );
        Ok(result)
    }
}

enum LeftInput {
    Raw { path: PathBuf, shape: Vec<usize> },
    Safetensors { path: PathBuf, tensor: String },
}
impl LeftInput {
    fn path(&self) -> &Path {
        match self {
            Self::Raw { path, .. } | Self::Safetensors { path, .. } => path,
        }
    }
}
struct Arguments {
    left: LeftInput,
    group_axis: Option<usize>,
    left_axis0_range: Option<Axis0Range>,
    reference_axis0_range: Option<Axis0Range>,
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
                        | "--left-safetensors"
                        | "--left-tensor"
                        | "--group-axis"
                        | "--left-axis0-range"
                        | "--reference-axis0-range"
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
        let raw = values.remove("--left-raw");
        let shape = values.remove("--left-shape");
        let safe = values.remove("--left-safetensors");
        let left_tensor = values.remove("--left-tensor");
        let group = values.remove("--group-axis");
        let left_range = values.remove("--left-axis0-range");
        let reference_range = values.remove("--reference-axis0-range");
        let mut take = |name: &str| {
            values
                .remove(name)
                .with_context(|| format!("required option {name}"))
        };
        let reference = PathBuf::from(take("--reference-safetensors")?);
        let text = |value: OsString| {
            value
                .into_string()
                .map_err(|_| anyhow::anyhow!("numeric options and tensor name require UTF-8"))
        };
        let left = match (raw, shape, safe, left_tensor) {
            (Some(path), Some(shape), None, None) => LeftInput::Raw {
                path: path.into(),
                shape: text(shape)?.split(',').map(str::parse)
                    .collect::<std::result::Result<Vec<usize>, _>>()?,
            },
            (None, None, Some(path), Some(tensor)) => {
                let tensor = text(tensor)?;
                ensure!(!tensor.is_empty(), "left tensor name must not be empty");
                LeftInput::Safetensors { path: path.into(), tensor }
            }
            _ => bail!("select either --left-raw with --left-shape or --left-safetensors with --left-tensor"),
        };
        let group_axis = group
            .map(|value| -> Result<usize> { Ok(text(value)?.parse()?) })
            .transpose()?;
        let parse_range = |value: OsString| Axis0Range::parse(&text(value)?);
        let left_axis0_range = left_range.map(parse_range).transpose()?;
        let reference_axis0_range = reference_range.map(parse_range).transpose()?;
        let tensor = text(take("--tensor")?)?;
        ensure!(!tensor.is_empty(), "tensor name must not be empty");
        let maximum_input_bytes = text(take("--maximum-input-bytes")?)?.parse::<usize>()?;
        ensure!(
            maximum_input_bytes > 0 && maximum_input_bytes <= isize::MAX as usize,
            "input byte limit must be positive and addressable"
        );
        if let LeftInput::Raw { shape, .. } = &left {
            ensure!(
                elements(shape)?
                    .checked_mul(4)
                    .is_some_and(|n| n <= maximum_input_bytes),
                "left shape exceeds input byte limit"
            );
        }
        Ok(Some(Self {
            left,
            group_axis,
            left_axis0_range,
            reference_axis0_range,
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

fn dtype_width(dtype: Dtype) -> Result<usize> {
    match dtype {
        Dtype::F32 => Ok(4),
        Dtype::F16 | Dtype::BF16 => Ok(2),
        other => bail!("unsupported tensor dtype {other:?}; expected F32, F16 or BF16"),
    }
}

/// Axis 0 is contiguous in the captured tensor; this only borrows its byte
/// interval. Full-file bounds have already been enforced before selection.
fn select_axis0<'a>(
    payload: &'a [u8],
    dtype: Dtype,
    shape: &[usize],
    range: Option<Axis0Range>,
) -> Result<(&'a [u8], Vec<usize>)> {
    let count = elements(shape)?;
    let width = dtype_width(dtype)?;
    ensure!(
        count.checked_mul(width) == Some(payload.len()),
        "original payload extent differs from its exact shape"
    );
    let mut selected_shape = shape.to_vec();
    let Some(range) = range else {
        return Ok((payload, selected_shape));
    };
    ensure!(
        range.start < range.end && range.end <= shape[0],
        "axis-0 range {}:{} exceeds original extent {} or is empty/reversed",
        range.start,
        range.end,
        shape[0]
    );
    let row_bytes = (count / shape[0])
        .checked_mul(width)
        .context("axis-0 row extent overflows")?;
    let start = range
        .start
        .checked_mul(row_bytes)
        .context("axis-0 start offset overflows")?;
    let end = range
        .end
        .checked_mul(row_bytes)
        .context("axis-0 end offset overflows")?;
    selected_shape[0] = range
        .end
        .checked_sub(range.start)
        .context("axis-0 selected extent underflows")?;
    Ok((
        payload
            .get(start..end)
            .context("axis-0 interval exceeds payload")?,
        selected_shape,
    ))
}

fn scalar(bytes: &[u8], dtype: Dtype) -> f64 {
    f64::from(match dtype {
        Dtype::F32 => f32::from_le_bytes(bytes.try_into().unwrap()),
        Dtype::F16 => f16::from_bits(u16::from_le_bytes(bytes.try_into().unwrap())).to_f32(),
        Dtype::BF16 => bf16::from_bits(u16::from_le_bytes(bytes.try_into().unwrap())).to_f32(),
        _ => unreachable!("dtype and exact element width were checked"),
    })
}

fn compare(
    left: &[u8],
    reference: &[u8],
    left_dtype: Dtype,
    reference_dtype: Dtype,
    shape: &[usize],
) -> Result<Metrics> {
    let count = elements(shape)?;
    compare_indices(
        left,
        reference,
        left_dtype,
        reference_dtype,
        shape,
        0..count,
    )
}

fn compare_indices(
    left: &[u8],
    reference: &[u8],
    left_dtype: Dtype,
    reference_dtype: Dtype,
    shape: &[usize],
    indices: impl Iterator<Item = usize>,
) -> Result<Metrics> {
    let count = elements(shape)?;
    let left_width = dtype_width(left_dtype)?;
    let reference_width = dtype_width(reference_dtype)?;
    ensure!(
        left.len() % left_width == 0 && reference.len() % reference_width == 0,
        "payload length is not divisible by its dtype width"
    );
    ensure!(
        count.checked_mul(left_width) == Some(left.len())
            && count.checked_mul(reference_width) == Some(reference.len()),
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
    let mut compared = 0usize;
    for index in indices {
        ensure!(
            index < count,
            "comparison index is outside the declared shape"
        );
        let a = &left[index * left_width..(index + 1) * left_width];
        let b = &reference[index * reference_width..(index + 1) * reference_width];
        compared += 1;
        let a = scalar(a, left_dtype);
        let b = scalar(b, reference_dtype);
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
    ensure!(compared > 0, "comparison group is empty");
    worst.coordinate = coordinate(worst.flat_index, shape);
    Ok(Metrics {
        finite: true,
        max_abs,
        rmse: (squared_error / compared as f64).sqrt(),
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
        .left
        .path()
        .to_str()
        .context("JSON input path requires UTF-8")?;
    let reference_path = args
        .reference
        .to_str()
        .context("JSON input path requires UTF-8")?;
    let expected = match &args.left {
        LeftInput::Raw { shape, .. } => Some(
            elements(shape)?
                .checked_mul(4)
                .context("F32 extent overflows")?,
        ),
        LeftInput::Safetensors { .. } => None,
    };
    let left = read_bounded(args.left.path(), args.maximum_input_bytes, expected)?;
    let remaining = args
        .maximum_input_bytes
        .checked_sub(left.len())
        .context("combined input byte limit exceeded")?;
    let reference = read_bounded(&args.reference, remaining, None)?;
    let archive = SafeTensors::deserialize(&reference).context("validate reference safetensors")?;
    let tensor = archive
        .tensor(&args.tensor)
        .context("select exact reference tensor")?;
    let left_archive = match &args.left {
        LeftInput::Raw { .. } => None,
        LeftInput::Safetensors { .. } => {
            Some(SafeTensors::deserialize(&left).context("validate left safetensors")?)
        }
    };
    let left_tensor = match (&args.left, &left_archive) {
        (LeftInput::Safetensors { tensor, .. }, Some(archive)) => {
            let tensor = archive.tensor(tensor).context("select exact left tensor")?;
            dtype_width(tensor.dtype()).context("left safetensors dtype")?;
            Some(tensor)
        }
        _ => None,
    };
    let (payload, original_shape, encoding, left_dtype) = match (&args.left, &left_tensor) {
        (LeftInput::Raw { shape, .. }, None) => {
            (left.as_slice(), shape.as_slice(), "raw_f32_le", Dtype::F32)
        }
        (LeftInput::Safetensors { .. }, Some(tensor)) => {
            (tensor.data(), tensor.shape(), "safetensors", tensor.dtype())
        }
        _ => unreachable!("left source and borrowed tensor agree"),
    };
    let (selected_payload, shape) =
        select_axis0(payload, left_dtype, original_shape, args.left_axis0_range)
            .context("select left axis-0 range")?;
    let (reference_payload, reference_shape) = select_axis0(
        tensor.data(),
        tensor.dtype(),
        tensor.shape(),
        args.reference_axis0_range,
    )
    .context("select reference axis-0 range")?;
    ensure!(
        reference_shape == shape,
        "shape mismatch: left {:?}, reference {:?}; reshape/squeeze is not permitted",
        shape,
        reference_shape
    );
    let shape = shape.as_slice();
    let count = elements(shape)?;
    if let Some(axis) = args.group_axis {
        ensure!(
            axis < shape.len(),
            "group axis {axis} exceeds tensor rank {}",
            shape.len()
        );
    }
    let metrics = compare(
        selected_payload,
        reference_payload,
        left_dtype,
        tensor.dtype(),
        shape,
    )?;
    let mut left_metadata = serde_json::json!({
        "path":left_path, "encoding":encoding, "dtype":format!("{left_dtype:?}"), "shape":shape,
        "elements":count, "file_bytes":left.len(), "payload_bytes":payload.len(),
        "file_sha256":sha256(&left), "payload_sha256":sha256(payload),
    });
    if let LeftInput::Safetensors { tensor, .. } = &args.left {
        left_metadata["tensor"] = serde_json::json!(tensor);
    }
    let mut report = serde_json::json!({
        "schema_version":1, "maximum_input_bytes":args.maximum_input_bytes,
        "normalization":"reference_l2",
        "left":left_metadata,
        "reference":{"path":reference_path, "encoding":"safetensors", "tensor":args.tensor,
            "dtype":format!("{:?}",tensor.dtype()), "shape":reference_shape, "elements":count,
            "file_bytes":reference.len(), "payload_bytes":tensor.data().len(),
            "file_sha256":sha256(&reference), "payload_sha256":sha256(tensor.data())},
        "metrics":metrics,
    });
    if args.left_axis0_range.is_some() || args.reference_axis0_range.is_some() {
        report["coordinate_space"] = serde_json::json!("worst coordinates and group indices are relative to the selected tensors; add each side's axis0_range.start (or 0 for null) to recover its original first-axis coordinate");
        for (side, original, range, selected) in [
            (
                "left",
                original_shape,
                args.left_axis0_range,
                selected_payload,
            ),
            (
                "reference",
                tensor.shape(),
                args.reference_axis0_range,
                reference_payload,
            ),
        ] {
            report[side]["original_shape"] = serde_json::json!(original);
            report[side]["axis0_range"] = serde_json::json!(range);
            report[side]["selected_shape"] = serde_json::json!(shape);
            report[side]["selected_payload_bytes"] = serde_json::json!(selected.len());
            report[side]["selected_payload_sha256"] = serde_json::json!(sha256(selected));
        }
    }
    if let Some(axis) = args.group_axis {
        let groups = shape[axis];
        // All products are subsets of the already checked positive full shape.
        let stride: usize = shape[axis + 1..].iter().product();
        let outer = count / (groups * stride);
        let mut grouped = Vec::new();
        for group in 0..groups {
            let indices = (0..outer).flat_map(|outer| {
                let start = (outer * groups + group) * stride;
                start..start + stride
            });
            let metrics = compare_indices(
                selected_payload,
                reference_payload,
                left_dtype,
                tensor.dtype(),
                shape,
                indices,
            )?;
            grouped.push(
                serde_json::json!({"index":group, "elements":count/groups, "metrics":metrics}),
            );
        }
        report["group_axis"] = serde_json::json!(axis);
        report["groups"] = serde_json::json!(grouped);
    }
    Ok(report)
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
