//! Expanded, frozen complete-request population. Indices are manifest slots,
//! not request IDs selected after observing successful execution.
use super::*;

pub const COHORT_PLAN_REVISION_V2: &str = "ferrum.structured-complete-cohorts.v2";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CohortRequestV2 {
    pub manifest_prompt: u32,
    pub maximum_output: u64,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CohortV2 {
    pub manifest_case: u32,
    pub repetition: u32,
    /// Vec position is the immutable request slot, including duplicate prompts.
    pub requests: Vec<CohortRequestV2>,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CohortPlanV2 {
    /// Fit, residual, qualification, each with all repetitions expanded.
    pub phases: [Vec<CohortV2>; 3],
}
impl CohortPlanV2 {
    pub fn validate(&self) -> Result<()> {
        let mut requests = 0usize;
        for phase in &self.phases {
            if phase.is_empty() || phase.len() > 4096 {
                return Err(StructuredUnknownV2::InvalidSettings);
            }
            let mut previous = None;
            for cohort in phase {
                if cohort.requests.is_empty() || cohort.requests.len() > 4096 {
                    return Err(StructuredUnknownV2::InvalidSettings);
                }
                let coordinate = (cohort.manifest_case, cohort.repetition);
                match previous {
                    None if coordinate != (0, 0) => {
                        return Err(StructuredUnknownV2::InvalidSettings)
                    }
                    Some((case, repetition))
                        if coordinate != (case, repetition + 1) && coordinate != (case + 1, 0) =>
                    {
                        return Err(StructuredUnknownV2::InvalidSettings)
                    }
                    _ => {}
                }
                previous = Some(coordinate);
                for request in &cohort.requests {
                    if request.maximum_output == 0 || request.maximum_output > u32::MAX as u64 {
                        return Err(StructuredUnknownV2::InvalidSettings);
                    }
                }
                requests = requests
                    .checked_add(cohort.requests.len())
                    .filter(|n| *n <= 65_536)
                    .ok_or(StructuredUnknownV2::Capacity)?;
            }
        }
        Ok(())
    }
    /// Bind both this typed slot/budget population and the complete original
    /// CLI protocol. Neither a digest alone nor a success footer proves that
    /// these requests ran; source replay separately checks admission/completion.
    pub fn signature(&self, manifest_payload: &serde_json::Value) -> Result<[u8; 32]> {
        self.validate()?;
        let mut encoded = CanonicalBytes::new(8 * 1024 * 1024);
        canonical_value(manifest_payload, &mut encoded, 0)?;
        let mut hash = Sha256::new();
        hash.update(COHORT_PLAN_REVISION_V2.as_bytes());
        hash.update([0]);
        hash.update((encoded.bytes.len() as u64).to_le_bytes());
        hash.update(encoded.bytes);
        hash.update(serde_json::to_vec(self).map_err(|_| StructuredUnknownV2::InvalidInput)?);
        Ok(hash.finalize().into())
    }
}

/// Each IO write is checked before allocating/copying. A failed encoding never
/// reaches the digest; keeping a bounded prefix is not a truncated valid hash.
struct CanonicalBytes {
    bytes: Vec<u8>,
    limit: usize,
    sorting_bytes: usize,
}
impl CanonicalBytes {
    fn new(limit: usize) -> Self {
        Self {
            bytes: Vec::new(),
            limit,
            sorting_bytes: 0,
        }
    }
    fn byte(&mut self, byte: u8) -> Result<()> {
        std::io::Write::write_all(self, &[byte]).map_err(|_| StructuredUnknownV2::Capacity)
    }
    fn reserve_sorting(&mut self, entries: usize) -> Result<usize> {
        let bytes = entries
            .checked_mul(std::mem::size_of::<(&String, &serde_json::Value)>())
            .ok_or(StructuredUnknownV2::Capacity)?;
        self.sorting_bytes = self
            .sorting_bytes
            .checked_add(bytes)
            .filter(|n| *n <= self.limit)
            .ok_or(StructuredUnknownV2::Capacity)?;
        Ok(bytes)
    }
}
impl std::io::Write for CanonicalBytes {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        if bytes.len() > self.limit.saturating_sub(self.bytes.len()) {
            return Err(std::io::Error::other(
                "canonical manifest byte limit exceeded",
            ));
        }
        let required = self.bytes.len() + bytes.len();
        if required > self.bytes.capacity() {
            let target = required
                .max(self.bytes.capacity().saturating_mul(2))
                .min(self.limit);
            self.bytes
                .try_reserve_exact(target - self.bytes.len())
                .map_err(|_| std::io::Error::other("canonical manifest allocation failed"))?;
        }
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}
fn canonical_value(
    value: &serde_json::Value,
    output: &mut CanonicalBytes,
    depth: usize,
) -> Result<()> {
    if depth > 64 {
        return Err(StructuredUnknownV2::Capacity);
    }
    match value {
        serde_json::Value::Array(values) => {
            output.byte(b'[')?;
            for (index, value) in values.iter().enumerate() {
                if index != 0 {
                    output.byte(b',')?;
                }
                canonical_value(value, output, depth + 1)?;
            }
            output.byte(b']')?;
        }
        serde_json::Value::Object(values) => {
            output.byte(b'{')?;
            // Bound all simultaneously live nested sorting indices before
            // reserving memory, independently from the output byte limit.
            let reserved = output.reserve_sorting(values.len())?;
            let mut fields = Vec::new();
            fields
                .try_reserve_exact(values.len())
                .map_err(|_| StructuredUnknownV2::Capacity)?;
            fields.extend(values.iter());
            fields.sort_unstable_by(|a, b| a.0.cmp(b.0));
            for (index, (key, value)) in fields.into_iter().enumerate() {
                if index != 0 {
                    output.byte(b',')?;
                }
                serde_json::to_writer(&mut *output, key)
                    .map_err(|_| StructuredUnknownV2::Capacity)?;
                output.byte(b':')?;
                canonical_value(value, output, depth + 1)?;
            }
            output.sorting_bytes -= reserved;
            output.byte(b'}')?;
        }
        _ => {
            serde_json::to_writer(&mut *output, value).map_err(|_| StructuredUnknownV2::Capacity)?
        }
    }
    Ok(())
}

#[cfg(test)]
mod bounded_encoding_tests {
    use super::*;

    #[test]
    fn structured_v2_cohort_canonical_encoding_bounds_escaped_unicode_before_write() {
        let value = serde_json::Value::String("汉\n\\\"🦀".into());
        let expected = serde_json::to_vec(&value).unwrap();
        let mut exact = CanonicalBytes::new(expected.len());
        canonical_value(&value, &mut exact, 0).unwrap();
        assert_eq!(exact.bytes, expected);
        let mut short = CanonicalBytes::new(expected.len() - 1);
        assert_eq!(
            canonical_value(&value, &mut short, 0),
            Err(StructuredUnknownV2::Capacity)
        );
        assert!(short.bytes.len() <= short.limit);
        assert!(short.bytes.capacity() <= short.limit);
    }

    #[test]
    fn structured_v2_cohort_canonical_encoding_rejects_large_strings_and_sort_indices() {
        let value = serde_json::Value::String("x".repeat(4096));
        let mut bounded = CanonicalBytes::new(64);
        assert_eq!(
            canonical_value(&value, &mut bounded, 0),
            Err(StructuredUnknownV2::Capacity)
        );
        assert!(bounded.bytes.len() <= 64);
        assert!(bounded.bytes.capacity() <= 64);
        let object: serde_json::Value = serde_json::from_str(r#"{"a":1,"b":2,"c":3}"#).unwrap();
        let mut small_index = CanonicalBytes::new(32);
        assert_eq!(
            canonical_value(&object, &mut small_index, 0),
            Err(StructuredUnknownV2::Capacity)
        );
        assert_eq!(small_index.bytes, b"{");
    }
}
