//! A declared immutable catalog is an index, never a portfolio searched for a
//! cheap or Known result. Each child retains its own source clock/support.
use super::*;
use model::structured_v2::StructuredOwnerKeyV2;
use serde::{Deserialize, Deserializer};
use std::{fs::File, io::Read, num::NonZeroUsize, path::PathBuf};
const ARTIFACT_TYPE: &str = "ferrum.structured-v2-catalog";
const MAX_CHILDREN: usize = 128;
const MAX_METADATA_BYTES: usize = 2 * 1024 * 1024;
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    artifact_type: String,
    schema_version: u32,
    model_revision: String,
    fingerprint: file::ProfileFingerprint,
    #[serde(deserialize_with = "bounded_children")]
    children: Vec<Child>,
}
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct Child {
    profile_path: PathBuf,
    profile_sha256: [u8; 32],
    owner: StructuredOwnerKeyV2,
    domain_signature: [u8; 32],
}
fn bounded_children<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<Child>, D::Error> {
    struct Visitor;
    impl<'de> serde::de::Visitor<'de> for Visitor {
        type Value = Vec<Child>;
        fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "1..={MAX_CHILDREN} unique structured children")
        }
        fn visit_seq<A: serde::de::SeqAccess<'de>>(
            self,
            mut seq: A,
        ) -> Result<Self::Value, A::Error> {
            let mut out = Vec::new();
            while let Some(child) = seq.next_element()? {
                if out.len() == MAX_CHILDREN {
                    return Err(serde::de::Error::custom(
                        "structured catalog child capacity",
                    ));
                }
                out.push(child);
            }
            Ok(out)
        }
    }
    d.deserialize_seq(Visitor)
}
impl Manifest {
    fn validate(
        &self,
        fp: &model::ExecutionFingerprint,
        limits: &file::CostProfileLoadLimits,
    ) -> Result<(), FerrumError> {
        if self.artifact_type != ARTIFACT_TYPE
            || self.schema_version != 1
            || self.model_revision != MODEL_REVISION_V2
            || self.fingerprint != file::ProfileFingerprint::from(fp)
            || self.children.is_empty()
            || self.children.len() > MAX_CHILDREN
        {
            return Err(FerrumError::config(
                "structured catalog schema, fingerprint or count differs",
            ));
        }
        for (i, c) in self.children.iter().enumerate() {
            let path = c
                .profile_path
                .to_str()
                .ok_or_else(|| FerrumError::config("catalog child path must be UTF-8"))?;
            if path.is_empty()
                || path.len() > limits.max_source_field_bytes.get()
                || path.chars().any(char::is_control)
                || c.profile_sha256 == [0; 32]
                || c.domain_signature == [0; 32]
                || c.owner.rows == 0
                || c.owner.rows > 128
                || self.children[..i]
                    .iter()
                    .any(|p| p.domain_signature == c.domain_signature || p.owner == c.owner)
            {
                return Err(FerrumError::config(
                    "structured catalog has invalid or duplicate owner/domain/path",
                ));
            }
        }
        Ok(())
    }
}
struct Budget {
    bytes: usize,
    samples: usize,
    rows: usize,
}
impl Budget {
    fn new(limits: &file::CostProfileLoadLimits, metadata: usize) -> Result<Self, FerrumError> {
        Ok(Self {
            bytes: limits
                .max_file_bytes
                .get()
                .checked_sub(metadata)
                .ok_or_else(|| FerrumError::config("catalog metadata exceeds total byte budget"))?,
            samples: limits.max_samples.get(),
            rows: limits.max_total_shape_rows.get(),
        })
    }
    fn limits(
        &self,
        base: &file::CostProfileLoadLimits,
    ) -> Result<file::CostProfileLoadLimits, FerrumError> {
        let nonzero = |n| {
            NonZeroUsize::new(n).ok_or_else(|| {
                FerrumError::config("structured catalog aggregate capacity exhausted")
            })
        };
        Ok(file::CostProfileLoadLimits {
            max_file_bytes: nonzero(self.bytes)?,
            max_samples: nonzero(self.samples)?,
            max_total_shape_rows: nonzero(self.rows)?,
            ..base.clone()
        })
    }
    fn consume(&mut self, bytes: u64, samples: u64, rows: u64) -> Result<(), FerrumError> {
        let checked = |remaining: usize, used: u64| {
            usize::try_from(used)
                .ok()
                .and_then(|used| remaining.checked_sub(used))
                .ok_or_else(|| {
                    FerrumError::config("structured catalog aggregate capacity exceeded")
                })
        };
        let next = Self {
            bytes: checked(self.bytes, bytes)?,
            samples: checked(self.samples, samples)?,
            rows: checked(self.rows, rows)?,
        };
        *self = next;
        Ok(())
    }
}
fn read_metadata(
    path: &Path,
    limits: &file::CostProfileLoadLimits,
) -> Result<Vec<u8>, FerrumError> {
    let mut file = File::open(path).map_err(profile_error)?;
    let meta = file.metadata().map_err(profile_error)?;
    if !meta.is_file() {
        return Err(FerrumError::config(
            "structured profile/catalog must be a regular file",
        ));
    }
    let bytes = usize::try_from(meta.len())
        .ok()
        .filter(|n| *n > 0 && *n <= limits.max_file_bytes.get().min(MAX_METADATA_BYTES))
        .ok_or_else(|| FerrumError::config("structured catalog/profile metadata byte limit"))?;
    let mut out = Vec::new();
    out.try_reserve_exact(bytes)
        .map_err(|_| FerrumError::config("structured metadata allocation failed"))?;
    out.resize(bytes, 0);
    file.read_exact(&mut out).map_err(profile_error)?;
    if file.read(&mut [0; 1]).map_err(profile_error)? != 0 {
        return Err(FerrumError::config(
            "structured metadata grew while reading",
        ));
    }
    Ok(out)
}
pub(super) fn load(
    path: &Path,
    fp: &model::ExecutionFingerprint,
    limits: &file::CostProfileLoadLimits,
    clock: file::ProfileLoadClock,
    declared: u64,
) -> Result<(StructuredSnapshot, SloCostProfileReceipt), FerrumError> {
    let bytes = read_metadata(path, limits)?;
    #[derive(Deserialize)]
    struct Kind {
        schema_version: u32,
        artifact_type: Option<String>,
    }
    let kind: Kind = serde_json::from_slice(&bytes).map_err(profile_error)?;
    let digest: [u8; 32] = Sha256::digest(&bytes).into();
    if kind.schema_version == 10 && kind.artifact_type.is_none() {
        drop(bytes);
        let child =
            file::load_structured_profile_v10(path, fp, limits, clock).map_err(profile_error)?;
        if child.provenance().file_sha256 != digest {
            return Err(FerrumError::config(
                "structured child changed during startup",
            ));
        }
        let receipt = receipt::single(&child, declared)?;
        return Ok((StructuredSnapshot::single(child), receipt));
    }
    if kind.schema_version != 1 || kind.artifact_type.as_deref() != Some(ARTIFACT_TYPE) {
        return Err(FerrumError::config(
            "expected schema10 child or explicit structured V2 catalog",
        ));
    }
    let manifest: Manifest = serde_json::from_slice(&bytes).map_err(profile_error)?;
    manifest.validate(fp, limits)?;
    let metadata_bytes = bytes.len();
    drop(bytes);
    let mut budget = Budget::new(limits, metadata_bytes)?;
    let mut children = BTreeMap::new();
    let directory = path.parent().unwrap_or(Path::new("."));
    for declared_child in manifest.children {
        let path = if declared_child.profile_path.is_absolute() {
            declared_child.profile_path
        } else {
            directory.join(declared_child.profile_path)
        };
        let child = file::load_structured_profile_v10(&path, fp, &budget.limits(limits)?, clock)
            .map_err(profile_error)?;
        let p = child.provenance();
        if p.file_sha256 != declared_child.profile_sha256
            || child.domain_signature() != &declared_child.domain_signature
            || child.owner() != &declared_child.owner
        {
            return Err(FerrumError::config(
                "catalog child digest/owner/domain differs from declaration",
            ));
        }
        budget.consume(
            p.file_bytes
                .checked_add(p.source_bytes)
                .ok_or_else(|| FerrumError::config("catalog child byte overflow"))?,
            p.reserved_members,
            p.total_shape_rows,
        )?;
        if children.insert(*child.domain_signature(), child).is_some() {
            return Err(FerrumError::config("duplicate replayed catalog domain"));
        }
    }
    let snapshot = StructuredSnapshot { children };
    let receipt = receipt::catalog(path, digest, metadata_bytes, &snapshot, declared)?;
    Ok((snapshot, receipt))
}
#[cfg(test)]
mod tests;
