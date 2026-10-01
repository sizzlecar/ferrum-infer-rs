//! Passive, bounded provenance emitted once after a successful installation.
//! These receipts describe existing qualified children; they grant no authority.
use super::*;
use ferrum_types::SloStructuredChildReceiptV2;
use serde::Serialize;
use std::io::{self, Write};

#[derive(Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum Change<'a> {
    Added {
        after: &'a SloStructuredChildReceiptV2,
    },
    Replaced {
        before: Box<SloStructuredChildReceiptV2>,
        after: &'a SloStructuredChildReceiptV2,
        same_domain: bool,
        independent_source: bool,
        previous_was_current_at_validation: bool,
    },
    Unchanged {
        original: &'a SloStructuredChildReceiptV2,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn installation_diagnostic_writer_enforces_byte_capacity_before_growth() {
        let limit = 2 * catalog::MAX_METADATA_BYTES;
        let mut output = BoundedJson(vec![b' '; limit - 1]);
        output.write_all(b"x").unwrap();
        assert!(output.write_all(b"y").is_err());
        assert_eq!(output.0.len(), limit);
        assert_eq!(output.0.last(), Some(&b'x'));
    }
}

#[derive(Serialize)]
struct Installation<'a> {
    schema_version: u32,
    executor_fingerprint: file::ProfileFingerprint,
    previous_runtime_epoch: Option<u64>,
    installed_runtime_epoch: u64,
    validated_at_ns: u64,
    source_schema_version: u32,
    catalog_child_count: usize,
    changed_children: Vec<Change<'a>>,
    /// Complete fixed-size numerical identities, including retained children.
    /// These borrowed diagnostic DTOs are not importable model authority.
    installed_children: Vec<InstalledIdentity<'a>>,
    retained_domain_signatures: &'a [[u8; 32]],
}

#[derive(Serialize)]
struct InstalledIdentity<'a> {
    domain_signature: &'a [u8; 32],
    owner: &'a model::structured_v2::StructuredOwnerKeyV2,
    numerical_family: Option<&'a model::structured_v2::NumericalFamilyKeyV1>,
    physical_domain_signature: Option<&'a [u8; 32]>,
}

impl<'a> InstalledIdentity<'a> {
    fn new(child: &'a ImportedStructuredModelV2) -> Self {
        Self {
            domain_signature: child.domain_signature(),
            owner: child.owner(),
            numerical_family: child.numerical_family_key(),
            physical_domain_signature: child.workload_domain().map(|domain| domain.sha256()),
        }
    }
}

// Each before/after inventory uses the existing metadata cap. The writer
// checks BEFORE extending, so unusually large diagnostic paths cannot allocate
// an unbounded JSON buffer. Failure affects only this optional diagnostic.
struct BoundedJson(Vec<u8>);
impl Write for BoundedJson {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        let remaining = (2 * catalog::MAX_METADATA_BYTES).saturating_sub(self.0.len());
        if bytes.len() > remaining {
            return Err(io::Error::other("installation diagnostic byte capacity"));
        }
        self.0.extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl EngineCostSnapshot {
    /// Called only under the runtime DEBUG gate, before feedback rebind closes
    /// the previous gate. The caller emits it only after the real commit.
    pub(in crate::continuous_engine::inner::cost_observation) fn installation_diagnostic(
        &self,
        previous: Option<&Self>,
        source: &SloCostProfileReceipt,
        validated_at_ns: u64,
        retained_domain_signatures: &[[u8; 32]],
    ) -> Result<String, &'static str> {
        let Snapshot::StructuredV2(next) = &self.inner else {
            return Err("non_structured_catalog");
        };
        let before = previous.and_then(|old| match &old.inner {
            Snapshot::StructuredV2(value) => Some(value),
            _ => None,
        });
        let source_children = &source
            .structured_whole_wave_v2
            .as_ref()
            .ok_or("missing_source_receipt")?
            .children;
        if source_children.len() > 128 || next.children.len() > 128 {
            return Err("diagnostic_child_capacity");
        }
        let previous_current = previous.is_some_and(EngineCostSnapshot::current);
        let mut changed_children = Vec::with_capacity(source_children.len());
        for after in source_children {
            let new = next
                .children
                .get(&after.domain_signature)
                .ok_or("missing_installed_child")?;
            let old = before.and_then(|catalog| {
                catalog
                    .children
                    .values()
                    .find(|old| old.same_population(new))
            });
            changed_children.push(match old {
                None => Change::Added { after },
                Some(old) => {
                    let before = receipt::child(old).map_err(|_| "original_receipt_encoding")?;
                    // Same owner/domain can legitimately be re-learned with
                    // identical coefficients; parameters alone cannot classify it.
                    let same_domain = old.domain_signature() == new.domain_signature();
                    let unchanged = same_domain
                        && before.profile_sha256 == after.profile_sha256
                        && before.source_sha256 == after.source_sha256
                        && before.parameters_sha256 == after.parameters_sha256
                        && before.protocol_sha256 == after.protocol_sha256
                        && before.capture_identity_sha256 == after.capture_identity_sha256;
                    if unchanged {
                        Change::Unchanged { original: after }
                    } else {
                        Change::Replaced {
                            same_domain,
                            independent_source: before.capture_identity_sha256
                                != after.capture_identity_sha256
                                && before.source_sha256 != after.source_sha256,
                            previous_was_current_at_validation: previous_current
                                && old.is_current_local(validated_at_ns).is_ok(),
                            before: Box::new(before),
                            after,
                        }
                    }
                }
            });
        }
        let event = Installation {
            schema_version: 1,
            executor_fingerprint: file::ProfileFingerprint::from(&self.fingerprint),
            previous_runtime_epoch: previous.map(EngineCostSnapshot::model_version),
            installed_runtime_epoch: self.model_version(),
            validated_at_ns,
            source_schema_version: source.schema_version,
            catalog_child_count: next.children.len(),
            changed_children,
            installed_children: next.children.values().map(InstalledIdentity::new).collect(),
            retained_domain_signatures,
        };
        let mut encoded = BoundedJson(Vec::new());
        serde_json::to_writer(&mut encoded, &event)
            .map_err(|_| "diagnostic_encoding_or_capacity")?;
        String::from_utf8(encoded.0).map_err(|_| "diagnostic_utf8")
    }
}
