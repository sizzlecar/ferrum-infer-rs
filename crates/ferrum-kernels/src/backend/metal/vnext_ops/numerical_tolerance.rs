use std::fmt::Write as _;
use std::sync::OnceLock;

use serde_json::Value;
use sha2::{Digest, Sha256};

const CATALOG: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../scripts/release/configs/runtime_vnext_numerical_tolerances.json"
));
const MAX_RELATIVE_ERROR_DENOMINATOR_FLOOR: f64 = 1.0e-12;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub(super) enum LogicalDtype {
    Fp16,
    Bf16,
    Fp32,
}

impl LogicalDtype {
    fn as_str(self) -> &'static str {
        match self {
            Self::Fp16 => "fp16",
            Self::Bf16 => "bf16",
            Self::Fp32 => "fp32",
        }
    }
}

#[derive(Debug, Clone)]
struct NumericalTolerance {
    logical_dtype: LogicalDtype,
    max_abs: f64,
    cosine_min: f64,
    relative_l2_max: f64,
    max_nan: usize,
    max_inf: usize,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub(super) struct NumericalMetrics {
    pub(super) tolerance_id: String,
    pub(super) row_fingerprint: String,
    pub(super) element_count: usize,
    pub(super) shape: Vec<usize>,
    pub(super) logical_dtype: LogicalDtype,
    pub(super) oracle_precision: String,
    pub(super) actual_f32_sha256: String,
    pub(super) expected_f32_sha256: String,
    pub(super) actual_nan_count: usize,
    pub(super) actual_inf_count: usize,
    pub(super) expected_nan_count: usize,
    pub(super) expected_inf_count: usize,
    pub(super) max_abs: f64,
    pub(super) max_relative_error: f64,
    pub(super) max_relative_error_denominator_floor: f64,
    pub(super) relative_l2: f64,
    pub(super) cosine: f64,
}

#[allow(clippy::too_many_arguments)]
pub(super) fn assert_matches(
    label: &str,
    actual: &[f32],
    actual_shape: &[usize],
    expected: &[f32],
    expected_shape: &[usize],
    logical_dtype: LogicalDtype,
    tolerance_id: &str,
    expected_row_fingerprint: &str,
) -> Result<NumericalMetrics, String> {
    let tolerance = resolve(tolerance_id, expected_row_fingerprint)?;
    if logical_dtype != tolerance.logical_dtype {
        return Err(format!(
            "{label}: logical dtype {} does not match catalog dtype {}",
            logical_dtype.as_str(),
            tolerance.logical_dtype.as_str()
        ));
    }
    if actual_shape != expected_shape {
        return Err(format!(
            "{label}: shape mismatch: actual={actual_shape:?} expected={expected_shape:?}"
        ));
    }
    let expected_elements = actual_shape
        .iter()
        .try_fold(1_usize, |elements, dimension| {
            elements.checked_mul(*dimension)
        });
    if expected_elements != Some(actual.len()) || expected.len() != actual.len() {
        return Err(format!(
            "{label}: shape/element mismatch: shape={actual_shape:?} actual={} expected={}",
            actual.len(),
            expected.len()
        ));
    }

    let metrics = measure(
        actual,
        actual_shape,
        expected,
        logical_dtype,
        tolerance_id,
        expected_row_fingerprint,
    );
    let mut violations = Vec::new();
    if metrics.actual_nan_count > tolerance.max_nan
        || metrics.expected_nan_count > tolerance.max_nan
    {
        violations.push(format!(
            "NaN count actual={} expected={} limit={}",
            metrics.actual_nan_count, metrics.expected_nan_count, tolerance.max_nan
        ));
    }
    if metrics.actual_inf_count > tolerance.max_inf
        || metrics.expected_inf_count > tolerance.max_inf
    {
        violations.push(format!(
            "Inf count actual={} expected={} limit={}",
            metrics.actual_inf_count, metrics.expected_inf_count, tolerance.max_inf
        ));
    }
    if metrics.max_abs > tolerance.max_abs {
        violations.push(format!(
            "max_abs={} limit={}",
            metrics.max_abs, tolerance.max_abs
        ));
    }
    if metrics.relative_l2 > tolerance.relative_l2_max {
        violations.push(format!(
            "relative_l2={} limit={}",
            metrics.relative_l2, tolerance.relative_l2_max
        ));
    }
    if metrics.cosine < tolerance.cosine_min {
        violations.push(format!(
            "cosine={} limit={}",
            metrics.cosine, tolerance.cosine_min
        ));
    }
    if violations.is_empty() {
        Ok(metrics)
    } else {
        Err(format!(
            "{label}: numerical tolerance {tolerance_id} failed: {}; metrics={metrics:?}",
            violations.join(", ")
        ))
    }
}

fn measure(
    actual: &[f32],
    shape: &[usize],
    expected: &[f32],
    logical_dtype: LogicalDtype,
    tolerance_id: &str,
    row_fingerprint: &str,
) -> NumericalMetrics {
    let actual_nan_count = actual.iter().filter(|value| value.is_nan()).count();
    let actual_inf_count = actual.iter().filter(|value| value.is_infinite()).count();
    let expected_nan_count = expected.iter().filter(|value| value.is_nan()).count();
    let expected_inf_count = expected.iter().filter(|value| value.is_infinite()).count();

    let mut max_abs = 0.0_f64;
    let mut max_relative_error = 0.0_f64;
    let mut squared_difference = 0.0_f64;
    let mut actual_squared = 0.0_f64;
    let mut expected_squared = 0.0_f64;
    let mut dot_product = 0.0_f64;
    for (&actual, &expected) in actual.iter().zip(expected) {
        if !actual.is_finite() || !expected.is_finite() {
            continue;
        }
        let actual = f64::from(actual);
        let expected = f64::from(expected);
        let difference = actual - expected;
        let absolute = difference.abs();
        max_abs = max_abs.max(absolute);
        max_relative_error = max_relative_error
            .max(absolute / expected.abs().max(MAX_RELATIVE_ERROR_DENOMINATOR_FLOOR));
        squared_difference += difference * difference;
        actual_squared += actual * actual;
        expected_squared += expected * expected;
        dot_product += actual * expected;
    }

    let relative_l2 = if expected_squared == 0.0 {
        if squared_difference == 0.0 {
            0.0
        } else {
            f64::INFINITY
        }
    } else {
        (squared_difference / expected_squared).sqrt()
    };
    let cosine = if actual_squared == 0.0 && expected_squared == 0.0 {
        1.0
    } else if actual_squared == 0.0 || expected_squared == 0.0 {
        0.0
    } else {
        (dot_product / (actual_squared * expected_squared).sqrt()).clamp(-1.0, 1.0)
    };

    NumericalMetrics {
        tolerance_id: tolerance_id.to_owned(),
        row_fingerprint: row_fingerprint.to_owned(),
        element_count: actual.len(),
        shape: shape.to_vec(),
        logical_dtype,
        oracle_precision: "fp32".to_owned(),
        actual_f32_sha256: f32_tensor_sha256(actual),
        expected_f32_sha256: f32_tensor_sha256(expected),
        actual_nan_count,
        actual_inf_count,
        expected_nan_count,
        expected_inf_count,
        max_abs,
        max_relative_error,
        max_relative_error_denominator_floor: MAX_RELATIVE_ERROR_DENOMINATOR_FLOOR,
        relative_l2,
        cosine,
    }
}

fn resolve(
    tolerance_id: &str,
    expected_row_fingerprint: &str,
) -> Result<NumericalTolerance, String> {
    static DOCUMENT: OnceLock<Value> = OnceLock::new();
    let document = DOCUMENT.get_or_init(|| {
        serde_json::from_str(CATALOG).expect("checked-in numerical tolerance catalog must be JSON")
    });
    let rows = document
        .get("rows")
        .and_then(Value::as_array)
        .expect("numerical tolerance catalog rows must be an array");
    let matches = rows
        .iter()
        .filter(|row| row.get("tolerance_id").and_then(Value::as_str) == Some(tolerance_id))
        .collect::<Vec<_>>();
    if matches.len() != 1 {
        return Err(format!(
            "tolerance_id must select exactly one checked-in row: {tolerance_id}"
        ));
    }
    let row = matches[0];
    let stored_row_fingerprint = row
        .get("row_fingerprint")
        .and_then(Value::as_str)
        .ok_or_else(|| format!("{tolerance_id}.row_fingerprint must be a string"))?;
    let computed_row_fingerprint = compute_row_fingerprint(row)?;
    if stored_row_fingerprint != computed_row_fingerprint {
        return Err(format!(
            "checked-in tolerance row fingerprint does not match its contents: {tolerance_id}"
        ));
    }
    if computed_row_fingerprint != expected_row_fingerprint {
        return Err(format!(
            "checked-in tolerance row fingerprint changed: {tolerance_id}"
        ));
    }

    let logical_dtype = match row.get("dtype").and_then(Value::as_str) {
        Some("fp16") => LogicalDtype::Fp16,
        Some("bf16") => LogicalDtype::Bf16,
        Some("fp32") => LogicalDtype::Fp32,
        Some(dtype) => return Err(format!("{tolerance_id}.dtype is unsupported: {dtype}")),
        None => return Err(format!("{tolerance_id}.dtype must be a string")),
    };
    if row.get("oracle_precision").and_then(Value::as_str) != Some("fp32") {
        return Err(format!(
            "{tolerance_id}.oracle_precision must be the trusted fp32 oracle"
        ));
    }
    let bounds = row
        .get("bounds")
        .and_then(Value::as_object)
        .ok_or_else(|| format!("{tolerance_id}.bounds must be an object"))?;
    let invariants = row
        .get("invariants")
        .and_then(Value::as_object)
        .ok_or_else(|| format!("{tolerance_id}.invariants must be an object"))?;
    for invariant in ["exact_shape", "exact_dtype"] {
        if invariants.get(invariant).and_then(Value::as_bool) != Some(true) {
            return Err(format!("{tolerance_id}.{invariant} must be true"));
        }
    }
    Ok(NumericalTolerance {
        logical_dtype,
        max_abs: finite_f64(bounds, "max_abs_max", tolerance_id)?,
        cosine_min: finite_f64(bounds, "cosine_min", tolerance_id)?,
        relative_l2_max: finite_f64(bounds, "relative_l2_max", tolerance_id)?,
        max_nan: finite_usize(invariants, "max_nan", tolerance_id)?,
        max_inf: finite_usize(invariants, "max_inf", tolerance_id)?,
    })
}

fn finite_f64(
    object: &serde_json::Map<String, Value>,
    field: &str,
    tolerance_id: &str,
) -> Result<f64, String> {
    let value = object
        .get(field)
        .and_then(Value::as_f64)
        .ok_or_else(|| format!("{tolerance_id}.{field} must be numeric"))?;
    if !value.is_finite() {
        return Err(format!("{tolerance_id}.{field} must be finite"));
    }
    Ok(value)
}

fn finite_usize(
    object: &serde_json::Map<String, Value>,
    field: &str,
    tolerance_id: &str,
) -> Result<usize, String> {
    let value = object
        .get(field)
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("{tolerance_id}.{field} must be a non-negative integer"))?;
    usize::try_from(value).map_err(|_| format!("{tolerance_id}.{field} exceeds usize"))
}

fn f32_tensor_sha256(values: &[f32]) -> String {
    let mut digest = Sha256::new();
    for value in values {
        digest.update(value.to_bits().to_le_bytes());
    }
    format!("{:x}", digest.finalize())
}

fn compute_row_fingerprint(row: &Value) -> Result<String, String> {
    let mut row = row
        .as_object()
        .cloned()
        .ok_or_else(|| "numerical tolerance row must be an object".to_owned())?;
    row.remove("row_fingerprint")
        .ok_or_else(|| "numerical tolerance row fingerprint is missing".to_owned())?;
    let mut canonical = String::new();
    write_canonical_json(&Value::Object(row), &mut canonical)?;
    Ok(format!("{:x}", Sha256::digest(canonical.as_bytes())))
}

// Matches Python json.dumps(sort_keys=True, separators=(",", ":"), ensure_ascii=True).
fn write_canonical_json(value: &Value, output: &mut String) -> Result<(), String> {
    match value {
        Value::Null => output.push_str("null"),
        Value::Bool(value) => output.push_str(if *value { "true" } else { "false" }),
        Value::Number(value) => output.push_str(&value.to_string()),
        Value::String(value) => write_ascii_json_string(value, output),
        Value::Array(values) => {
            output.push('[');
            for (index, value) in values.iter().enumerate() {
                if index != 0 {
                    output.push(',');
                }
                write_canonical_json(value, output)?;
            }
            output.push(']');
        }
        Value::Object(values) => {
            output.push('{');
            let mut keys = values.keys().collect::<Vec<_>>();
            keys.sort_unstable();
            for (index, key) in keys.into_iter().enumerate() {
                if index != 0 {
                    output.push(',');
                }
                write_ascii_json_string(key, output);
                output.push(':');
                write_canonical_json(&values[key], output)?;
            }
            output.push('}');
        }
    }
    Ok(())
}

fn write_ascii_json_string(value: &str, output: &mut String) {
    output.push('"');
    for character in value.chars() {
        match character {
            '"' => output.push_str("\\\""),
            '\\' => output.push_str("\\\\"),
            '\u{0008}' => output.push_str("\\b"),
            '\u{000c}' => output.push_str("\\f"),
            '\n' => output.push_str("\\n"),
            '\r' => output.push_str("\\r"),
            '\t' => output.push_str("\\t"),
            character if character <= '\u{001f}' => {
                write!(output, "\\u{:04x}", character as u32).unwrap();
            }
            character if character.is_ascii() => output.push(character),
            character if (character as u32) <= 0xffff => {
                write!(output, "\\u{:04x}", character as u32).unwrap();
            }
            character => {
                let code_point = character as u32 - 0x1_0000;
                let high = 0xd800 + (code_point >> 10);
                let low = 0xdc00 + (code_point & 0x3ff);
                write!(output, "\\u{high:04x}\\u{low:04x}").unwrap();
            }
        }
    }
    output.push('"');
}

#[test]
fn catalog_binding_recomputes_fingerprint_and_reports_all_metrics() {
    let tolerance_id =
        "runtime-vnext.metal.causal-attention.v2.operation.fp16.none.fixed-page-split";
    let fingerprint = "d30006c0535a3b3172ac88db66f75f07df6256e321509188bb0949c7a64a9fdb";
    let metrics = assert_matches(
        "identity",
        &[0.0, 1.0, -2.0],
        &[3],
        &[0.0, 1.0, -2.0],
        &[3],
        LogicalDtype::Fp16,
        tolerance_id,
        fingerprint,
    )
    .unwrap();
    assert_eq!(metrics.max_abs, 0.0);
    assert_eq!(metrics.relative_l2, 0.0);
    assert_eq!(metrics.cosine, 1.0);
    assert_eq!(metrics.actual_nan_count, 0);
    assert_eq!(metrics.actual_inf_count, 0);
    assert_eq!(metrics.actual_f32_sha256.len(), 64);
    assert_eq!(metrics.expected_f32_sha256.len(), 64);
    let evidence = serde_json::to_value(&metrics).unwrap();
    assert_eq!(evidence["tolerance_id"], tolerance_id);
    assert_eq!(evidence["row_fingerprint"], fingerprint);
    assert_eq!(evidence["logical_dtype"], "fp16");
    assert_eq!(evidence["oracle_precision"], "fp32");

    let document: Value = serde_json::from_str(CATALOG).unwrap();
    let row = document["rows"]
        .as_array()
        .unwrap()
        .iter()
        .find(|row| row["tolerance_id"] == tolerance_id)
        .unwrap();
    assert_eq!(compute_row_fingerprint(row).unwrap(), fingerprint);
    let mut tampered = row.clone();
    tampered["bounds"]["max_abs_max"] = Value::from(1.0);
    assert_ne!(compute_row_fingerprint(&tampered).unwrap(), fingerprint);
}

#[test]
fn numerical_contract_rejects_partial_or_non_finite_comparisons() {
    let tolerance_id =
        "runtime-vnext.metal.causal-attention.v2.operation.fp16.none.fixed-page-split";
    let fingerprint = "d30006c0535a3b3172ac88db66f75f07df6256e321509188bb0949c7a64a9fdb";
    assert!(resolve("missing", "0").is_err());
    assert!(resolve(tolerance_id, &"0".repeat(64)).is_err());
    assert!(assert_matches(
        "shape",
        &[1.0, 2.0],
        &[2],
        &[1.0, 2.0],
        &[1, 2],
        LogicalDtype::Fp16,
        tolerance_id,
        fingerprint,
    )
    .is_err());
    assert!(assert_matches(
        "dtype",
        &[1.0],
        &[1],
        &[1.0],
        &[1],
        LogicalDtype::Fp32,
        tolerance_id,
        fingerprint,
    )
    .is_err());
    assert!(assert_matches(
        "nan",
        &[f32::NAN],
        &[1],
        &[0.0],
        &[1],
        LogicalDtype::Fp16,
        tolerance_id,
        fingerprint,
    )
    .unwrap_err()
    .contains("NaN count"));
    assert!(assert_matches(
        "cosine",
        &[1.0, 0.0],
        &[2],
        &[-1.0, 0.0],
        &[2],
        LogicalDtype::Fp16,
        tolerance_id,
        fingerprint,
    )
    .unwrap_err()
    .contains("cosine"));
}
