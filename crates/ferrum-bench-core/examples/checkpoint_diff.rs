//! Compare actual vNext wave artifacts across numerical profiles/backends.
//! This measures arrays; it does not establish matching model/input identities,
//! accept a numerical tolerance, or authorize a release.
use clap::Parser;
use ferrum_bench_core::release_regression::numerics::nmse;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs,
    io::Write,
    path::{Path, PathBuf},
    process::ExitCode,
};

#[derive(Parser)]
#[command(about = "Measure verified checkpoint arrays; diagnostic only, no release approval")]
struct Args {
    #[arg(long)]
    reference: PathBuf,
    #[arg(long)]
    candidate: PathBuf,
    #[arg(long)]
    output: PathBuf,
}

#[derive(Deserialize, Serialize)]
struct Wave {
    schema_version: u32,
    wave_kind: String,
    participant_count: u32,
    records: Vec<Record>,
    product_outputs: Vec<Record>,
}

#[derive(Deserialize, Serialize)]
struct Record {
    #[serde(default)]
    value: Option<Identity>,
    #[serde(default)]
    output_mode: Option<String>,
    participant_index: u32,
    token_span: Value,
    output_layout: Layout,
    raw_file: String,
    raw_bytes: u64,
    raw_sha256: String,
}

#[derive(Deserialize, Serialize)]
struct Identity {
    value_id: String,
    tensor: Tensor,
}

#[derive(Deserialize, Serialize)]
struct Tensor {
    dimensions: Vec<u64>,
}

#[derive(Deserialize, Serialize)]
struct Layout {
    element_type: String,
    element_count: u64,
}

fn error(context: &str, err: impl std::fmt::Display) -> String {
    format!("{context}: {err}")
}

fn index(wave: &Wave) -> Result<BTreeMap<(u32, String), &Record>, String> {
    if !matches!(wave.schema_version, 3 | 4) || wave.participant_count == 0 {
        return Err("unsupported or empty checkpoint wave".into());
    }
    let mut records = BTreeMap::new();
    for record in wave.records.iter().chain(&wave.product_outputs) {
        if record.participant_index >= wave.participant_count {
            return Err("checkpoint participant is outside the wave".into());
        }
        let id = match (&record.value, &record.output_mode) {
            (Some(value), None) if !value.value_id.is_empty() => {
                format!("value:{}", value.value_id)
            }
            (None, Some(mode)) if mode == "full-logits" => "product:full-logits".into(),
            _ => return Err("checkpoint has no supported unambiguous float value identity".into()),
        };
        if records
            .insert((record.participant_index, id), record)
            .is_some()
        {
            return Err("duplicate checkpoint participant/value".into());
        }
    }
    if records.is_empty() {
        return Err("checkpoint contains no float arrays".into());
    }
    Ok(records)
}

fn array(parent: &Path, record: &Record) -> Result<Vec<f32>, String> {
    let name = Path::new(&record.raw_file);
    if name.file_name().and_then(|s| s.to_str()) != Some(record.raw_file.as_str())
        || record.raw_file.contains(['/', '\\'])
    {
        return Err("checkpoint raw_file must be an adjacent file name".into());
    }
    let width = match record.output_layout.element_type.as_str() {
        "f16" | "bf16" => 2_u64,
        "f32" => 4,
        _ => return Err("checkpoint must contain F16, BF16 or F32 values".into()),
    };
    if record.output_layout.element_count == 0
        || record.output_layout.element_count.checked_mul(width) != Some(record.raw_bytes)
    {
        return Err("checkpoint layout and byte extent disagree".into());
    }
    let path = parent.join(name);
    if fs::metadata(&path)
        .map_err(|e| error("raw metadata", e))?
        .len()
        != record.raw_bytes
    {
        return Err("checkpoint raw length differs from its manifest".into());
    }
    let raw = fs::read(&path).map_err(|e| error("read raw", e))?;
    if format!("{:x}", Sha256::digest(&raw)) != record.raw_sha256 {
        return Err("checkpoint raw SHA-256 differs from its manifest".into());
    }
    let values = raw
        .chunks_exact(width as usize)
        .map(|bytes| match record.output_layout.element_type.as_str() {
            "f32" => f32::from_le_bytes(bytes.try_into().unwrap()),
            "f16" => half::f16::from_le_bytes(bytes.try_into().unwrap()).to_f32(),
            "bf16" => half::bf16::from_le_bytes(bytes.try_into().unwrap()).to_f32(),
            _ => unreachable!(),
        })
        .collect::<Vec<_>>();
    if let Some(index) = values.iter().position(|x| !x.is_finite()) {
        return Err(format!("checkpoint contains a non-finite value at {index}"));
    }
    Ok(values)
}

fn top(values: &[f32]) -> Vec<Value> {
    let mut order = (0..values.len()).collect::<Vec<_>>();
    order.sort_unstable_by(|&a, &b| values[b].total_cmp(&values[a]).then(a.cmp(&b)));
    order
        .into_iter()
        .take(10)
        .map(|index| json!({"index":index,"value":values[index]}))
        .collect()
}

fn compare(reference: &Path, candidate: &Path) -> Result<Value, String> {
    let left_bytes = fs::read(reference).map_err(|e| error("reference wave", e))?;
    let right_bytes = fs::read(candidate).map_err(|e| error("candidate wave", e))?;
    let left: Wave = serde_json::from_slice(&left_bytes).map_err(|e| error("reference wave", e))?;
    let right: Wave =
        serde_json::from_slice(&right_bytes).map_err(|e| error("candidate wave", e))?;
    if left.wave_kind != right.wave_kind || left.participant_count != right.participant_count {
        return Err("checkpoint wave kind or participant count differs".into());
    }
    let lhs = index(&left)?;
    let rhs = index(&right)?;
    if !lhs.keys().eq(rhs.keys()) {
        return Err("checkpoint value inventories differ".into());
    }
    let mut rows = Vec::new();
    for (key, a) in lhs {
        let b = rhs[&key];
        if a.token_span != b.token_span
            || a.output_layout.element_count != b.output_layout.element_count
            || a.value.as_ref().map(|v| &v.tensor.dimensions)
                != b.value.as_ref().map(|v| &v.tensor.dimensions)
        {
            return Err(format!("checkpoint {key:?} spans or shapes differ"));
        }
        let x = array(reference.parent().unwrap_or(Path::new(".")), a)?;
        let y = array(candidate.parent().unwrap_or(Path::new(".")), b)?;
        let error = nmse(&x, &y);
        if !error.is_finite() {
            return Err(format!("checkpoint {key:?} NMSE is non-finite"));
        }
        let max_abs = x
            .iter()
            .zip(&y)
            .map(|(&x, &y)| (f64::from(x) - f64::from(y)).abs())
            .fold(0.0, f64::max);
        rows.push(
            json!({"participant":key.0,"value":key.1,"reference_layout":a.output_layout,
            "candidate_layout":b.output_layout,"nmse":error,"max_abs":max_abs,
            "equal_f32_bits":x.iter().zip(&y).filter(|(x,y)|x.to_bits()==y.to_bits()).count(),
            "reference_top_logits":a.output_mode.as_ref().map(|_|top(&x)),
            "candidate_top_logits":b.output_mode.as_ref().map(|_|top(&y))}),
        );
    }
    Ok(
        json!({"schema_version":1,"scope":"checkpoint_array_diagnostic","release_approved":false,
        "input_and_weight_identity_verified":false,
        "reference_wave_sha256":format!("{:x}",Sha256::digest(&left_bytes)),
        "candidate_wave_sha256":format!("{:x}",Sha256::digest(&right_bytes)),"comparisons":rows}),
    )
}

fn run(args: Args) -> Result<(), String> {
    let mut file = fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&args.output)
        .map_err(|e| error("reserve diagnostic report", e))?;
    let result = compare(&args.reference, &args.candidate);
    let report = result.as_ref().cloned().unwrap_or_else(|error| {
        json!({"schema_version":1,
        "scope":"checkpoint_array_diagnostic","release_approved":false,"error":error})
    });
    file.write_all(&serde_json::to_vec_pretty(&report).map_err(|e| e.to_string())?)
        .map_err(|e| error("write diagnostic", e))?;
    result.map(|_| ())
}

fn main() -> ExitCode {
    match run(Args::parse()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("checkpoint diff: {error}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
#[path = "checkpoint_diff/tests.rs"]
mod tests;
