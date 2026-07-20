use std::sync::OnceLock;

use serde_json::Value;

const CATALOG: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../scripts/release/configs/runtime_vnext_numerical_tolerances.json"
));

#[derive(Debug, Clone, Copy)]
pub(super) struct NumericalTolerance {
    pub(super) max_abs: f32,
    pub(super) cosine_min: f32,
    pub(super) relative_l2_max: f32,
}

pub(super) fn resolve(
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
    if row.get("row_fingerprint").and_then(Value::as_str) != Some(expected_row_fingerprint) {
        return Err(format!(
            "checked-in tolerance row fingerprint changed: {tolerance_id}"
        ));
    }
    let bounds = row
        .get("bounds")
        .and_then(Value::as_object)
        .expect("numerical tolerance row bounds must be an object");
    Ok(NumericalTolerance {
        max_abs: finite_f32(bounds, "max_abs_max", tolerance_id)?,
        cosine_min: finite_f32(bounds, "cosine_min", tolerance_id)?,
        relative_l2_max: finite_f32(bounds, "relative_l2_max", tolerance_id)?,
    })
}

fn finite_f32(
    bounds: &serde_json::Map<String, Value>,
    field: &str,
    tolerance_id: &str,
) -> Result<f32, String> {
    let value = bounds
        .get(field)
        .and_then(Value::as_f64)
        .ok_or_else(|| format!("{tolerance_id}.{field} must be numeric"))?;
    if !value.is_finite() {
        return Err(format!("{tolerance_id}.{field} must be finite"));
    }
    Ok(value as f32)
}

#[test]
fn catalog_binding_rejects_unknown_id_or_stale_fingerprint() {
    let row = resolve(
        "runtime-vnext.metal.causal-attention.v2.operation.fp16.none.fixed-page-split",
        "d30006c0535a3b3172ac88db66f75f07df6256e321509188bb0949c7a64a9fdb",
    )
    .unwrap();
    assert!(row.max_abs > 0.0);
    assert!((0.0..=1.0).contains(&row.cosine_min));
    assert!(row.relative_l2_max >= 0.0);

    assert!(resolve("missing", "0").is_err());
    assert!(resolve(
        "runtime-vnext.metal.causal-attention.v2.operation.fp16.none.fixed-page-split",
        "0000000000000000000000000000000000000000000000000000000000000000",
    )
    .is_err());
}
