//! Source7 canonical encoding is independent of serde_json's preserve_order
//! feature. It sorts every object, including those nested in arrays and typed
//! diagnostic Values. Array order and all payload values remain significant.
use super::*;

/// The diagnostic writer must serialize this value with compact JSON and one
/// trailing newline, exactly as `record_bytes_v7` does for the collector hash.
pub fn canonical_value_v7(value: &impl Serialize) -> Result<serde_json::Value, CostProfileError> {
    let mut value = serde_json::to_value(value)?;
    value.sort_all_objects();
    Ok(value)
}

pub fn record_bytes_v7(value: &impl Serialize) -> Result<Vec<u8>, CostProfileError> {
    let mut bytes = serde_json::to_vec(&canonical_value_v7(value)?)?;
    if bytes.len() >= 8 * 1024 * 1024 {
        return Err(CostProfileError::Limit(
            "source7 record exceeds original 8MiB bound",
        ));
    }
    bytes.push(b'\n');
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[derive(Serialize)]
    struct TypedRecord {
        z: u64,
        a: serde_json::Value,
    }
    #[test]
    fn source7_canonical_objects_inside_arrays_ignore_map_insertion_order() {
        let reverse = TypedRecord {
            z: 9,
            a: serde_json::from_str(r#"{"z":[{"y":2,"a":1},{"z":{"b":4,"a":3}}],"a":0}"#).unwrap(),
        };
        let forward = TypedRecord {
            z: 9,
            a: serde_json::from_str(r#"{"a":0,"z":[{"a":1,"y":2},{"z":{"a":3,"b":4}}]}"#).unwrap(),
        };
        // The same explicit protocol bytes are required with either Cargo
        // feature set; this test never checks the implementation's map type.
        let expected =
            b"{\"a\":{\"a\":0,\"z\":[{\"a\":1,\"y\":2},{\"z\":{\"a\":3,\"b\":4}}]},\"z\":9}\n";
        assert_eq!(record_bytes_v7(&reverse).unwrap(), expected);
        assert_eq!(record_bytes_v7(&forward).unwrap(), expected);
        let canonical = canonical_value_v7(&reverse).unwrap();
        let mut directory_bytes = serde_json::to_vec(&canonical).unwrap();
        directory_bytes.push(b'\n');
        assert_eq!(directory_bytes, expected);
        let reordered = TypedRecord {
            z: 9,
            a: serde_json::from_str(r#"{"a":0,"z":[{"z":{"a":3,"b":4}},{"a":1,"y":2}]}"#).unwrap(),
        };
        assert_ne!(record_bytes_v7(&reordered).unwrap(), expected);
    }
}
