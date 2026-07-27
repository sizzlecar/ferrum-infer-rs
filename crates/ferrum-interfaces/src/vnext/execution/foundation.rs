use super::{BTreeMap, Digest, DynamicStorageProfile, Serialize, Sha256, VNextError};

pub(super) fn invalid_plan(reason: impl Into<String>) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: reason.into(),
    }
}

pub(super) fn validate_active_sequence_ceiling(
    maximum_active_sequences: u32,
) -> Result<(), VNextError> {
    if maximum_active_sequences == 0 {
        return Err(invalid_plan(
            "maximum active sequences protocol ceiling must be non-zero",
        ));
    }
    Ok(())
}

pub(super) fn validate_scheduled_token_ceiling(
    maximum_scheduled_tokens: u64,
) -> Result<(), VNextError> {
    if maximum_scheduled_tokens == 0 {
        return Err(invalid_plan(
            "maximum scheduled tokens protocol ceiling must be non-zero",
        ));
    }
    Ok(())
}

pub(super) fn canonical_json(value: serde_json::Value) -> serde_json::Value {
    match value {
        serde_json::Value::Array(values) => {
            serde_json::Value::Array(values.into_iter().map(canonical_json).collect())
        }
        serde_json::Value::Object(values) => serde_json::Value::Object(
            values
                .into_iter()
                .map(|(key, value)| (key, canonical_json(value)))
                .collect::<BTreeMap<_, _>>()
                .into_iter()
                .collect(),
        ),
        scalar => scalar,
    }
}

pub(super) fn canonical_fingerprint<T: Serialize>(
    value: &T,
    context: &'static str,
) -> Result<String, VNextError> {
    let value = serde_json::to_value(value).map_err(|error| VNextError::Serialization {
        context,
        message: error.to_string(),
    })?;
    let bytes =
        serde_json::to_vec(&canonical_json(value)).map_err(|error| VNextError::Serialization {
            context,
            message: error.to_string(),
        })?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

pub(super) fn is_canonical_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

pub(super) fn align_up(value: u64, alignment: u64) -> Result<u64, VNextError> {
    if alignment == 0 || !alignment.is_power_of_two() {
        return Err(invalid_plan(
            "allocation alignment is not a non-zero power of two",
        ));
    }
    value
        .checked_add(alignment - 1)
        .map(|rounded| rounded & !(alignment - 1))
        .ok_or_else(|| invalid_plan("aligned allocation size overflows u64"))
}

pub(super) fn quantize_storage_bytes(
    logical_bytes: u64,
    alignment_bytes: u64,
    profile: DynamicStorageProfile,
) -> Result<u64, VNextError> {
    let aligned = align_up(logical_bytes, alignment_bytes)?;
    match profile.allocator() {
        super::DynamicStorageAllocator::LinearArena => Ok(aligned),
        super::DynamicStorageAllocator::FixedBlockArena { block_bytes } => {
            align_up(aligned, block_bytes)
        }
    }
}
