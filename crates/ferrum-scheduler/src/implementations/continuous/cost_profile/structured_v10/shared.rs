//! One complete physical source4 replay with independent child qualification.
//! Schema11 binds the immutable common source and every predeclared child.
use super::*;
mod wire;
use wire::*;
mod profile;
mod replay;
pub use profile::{
    export_structured_profile_v11, load_structured_profile_v11, ImportedStructuredCatalogV11,
    StructuredProfileExportReceiptV11,
};

/// Bounded group declaration used by the live writer before the first offer.
/// This serializable header is evidence only, never a live execution receipt.
pub fn structured_shared_source_header_v4(
    children: Vec<serde_json::Value>,
    maximum_file_bytes: u64,
    maximum_children: usize,
    maximum_retained_numeric_bytes: usize,
    maximum_retained_coordinates: usize,
) -> Result<serde_json::Value, CostProfileError> {
    if children.is_empty() || children.len() > maximum_children || maximum_children > 128 {
        return Err(invalid("invalid shared child count"));
    }
    // Check one parsed declaration at a time. Large common payloads are moved
    // into one owner, never retained in N deserialized legacy Headers.
    let mut declarations = children.into_iter();
    let first: Header = serde_json::from_value(declarations.next().unwrap())?;
    let identity_valid = |h: &Header| {
        h.artifact_type == "ferrum.structured-live-source"
            && h.schema_version == 3
            && h.model_revision == MODEL_REVISION_V2
            && h.maximum_file_bytes == maximum_file_bytes
    };
    if !identity_valid(&first) {
        return Err(invalid("source4 child source identity differs"));
    }
    let (common, first) = CommonDeclarationV4::from_header(first);
    let mut children = vec![first];
    for value in declarations {
        let h: Header = serde_json::from_value(value)?;
        if !identity_valid(&h) || !common.matches(&h) {
            return Err(invalid("source4 children lack a common physical contract"));
        }
        children.push(CommonDeclarationV4::from_header(h).1);
    }
    let mut h = HeaderV4 {
        artifact_type: "ferrum.structured-shared-live-source".into(),
        schema_version: 4,
        model_revision: MODEL_REVISION_V2.into(),
        maximum_file_bytes,
        maximum_children,
        maximum_retained_numeric_bytes,
        maximum_retained_coordinates,
        capture_protocol: [0; 32],
        common,
        children,
    };
    if h.common.opening.wall_unix_ns == 0
        || h.children.iter().enumerate().any(|(index, child)| {
            child.opened_at_ns > h.common.opening.monotonic_ns
                || h.children[..index].iter().any(|old| {
                    old.scope.owner == child.scope.owner
                        || old.capture_identity == child.capture_identity
                })
        })
    {
        return Err(invalid("source4 child clock or identity differs"));
    }
    h.capture_protocol = h.signature()?;
    let value = serde_json::to_value(h)?;
    // Include exactly the live wrapper and LF. Reject before the first offer,
    // using the unchanged loader line limit and declared common file limit.
    checked_header_record_bytes(&value, maximum_file_bytes)?;
    Ok(value)
}

fn checked_header_record_bytes(
    value: &serde_json::Value,
    maximum_file_bytes: u64,
) -> Result<usize, CostProfileError> {
    #[derive(Serialize)]
    struct Line<'a> {
        source_record_ordinal: u64,
        record: &'a serde_json::Value,
    }
    // A bounded counting sink avoids allocating an oversized serialized record.
    struct Count {
        bytes: usize,
        limit: usize,
    }
    impl std::io::Write for Count {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.bytes = self
                .bytes
                .checked_add(bytes.len())
                .filter(|n| *n <= self.limit)
                .ok_or_else(|| std::io::Error::other("source4 header record byte limit"))?;
            Ok(bytes.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    let limit = super::replay::MAX_SOURCE_RECORD_BYTES;
    let mut count = Count { bytes: 1, limit }; // terminating LF
    serde_json::to_writer(
        &mut count,
        &Line {
            source_record_ordinal: 1,
            record: value,
        },
    )
    .map_err(|_| CostProfileError::Limit("structured source record bytes"))?;
    if count.bytes as u64 > maximum_file_bytes {
        return Err(CostProfileError::Limit(
            "source4 header exceeds common file budget",
        ));
    }
    Ok(count.bytes)
}

#[cfg(test)]
#[path = "shared/header_tests.rs"]
mod header_tests;
