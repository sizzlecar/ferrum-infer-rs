use super::*;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct CacheIdentity {
    /// Cold hash of actual execution/workload identity and effective typed
    /// numerical/feedback policy. This is checked again by strict profile load.
    pub binding: [u8; 32],
    pub clock: CostMonotonicDomainV1,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(in crate::continuous_engine::inner::cost_observation) enum SourceKind {
    OwnerBlocksV7,
    PreparedOwnersV8,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ArtifactReceipt {
    pub index: usize,
    pub bytes: u64,
    pub sha256: [u8; 32],
}
impl ArtifactReceipt {
    pub fn file_name(&self) -> String {
        artifact_name(self.index)
    }
}
pub(super) fn artifact_name(index: usize) -> String {
    format!("artifact-{index}.bin")
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub(super) enum JournalEncoding {
    #[default]
    IdentityV1,
    GzipV1 {
        original_bytes: u64,
        original_sha256: [u8; 32],
    },
}
impl JournalEncoding {
    pub fn original(self, stored: ArtifactReceipt) -> (u64, [u8; 32]) {
        match self {
            Self::IdentityV1 => (stored.bytes, stored.sha256),
            Self::GzipV1 {
                original_bytes,
                original_sha256,
            } => (original_bytes, original_sha256),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct CachedSource {
    pub kind: SourceKind,
    /// Stored immutable journal artifact. Encoding carries its original full
    /// length/hash; checkpoint offsets always address that original byte stream.
    pub journal: ArtifactReceipt,
    #[serde(default)]
    pub encoding: JournalEncoding,
    /// Exact original publication prefix, which can precede the final record.
    pub checkpoint_bytes: u64,
    pub checkpoint_sha256: [u8; 32],
    pub profile: ArtifactReceipt,
    /// The exact retained subset, selected before persistence from the current
    /// immutable catalog. Independent replay must reconstruct every key.
    #[serde(default)]
    pub domains: Vec<[u8; 32]>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct CacheManifest {
    pub schema_version: u32,
    pub generation: [u8; 16],
    pub identity: CacheIdentity,
    pub sources: Vec<CachedSource>,
    pub feedback: ArtifactReceipt,
    /// Checked algorithm coordinates retained independently of numerical
    /// qualification. Its hash/identity do not grant prediction authority.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub startup_seed: Option<ArtifactReceipt>,
    /// Minimum original child expiry from independently replayed sources.
    /// This is a fast rejection bound only, never a timestamp renewal.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expires_at_ns: Option<u64>,
}
impl CacheManifest {
    pub fn validate(&self, identity: &CacheIdentity, now: u64, limits: CacheLimits) -> Result<()> {
        if !matches!(self.schema_version, 1 | 2)
            || self.generation == [0; 16]
            || self.sources.is_empty()
            || self.sources.len() > limits.maximum_sources
        {
            return Err(CacheMiss::Corrupt);
        }
        if self.identity.clock != identity.clock {
            return Err(CacheMiss::CrossBoot);
        }
        if self.identity.binding != identity.binding {
            return Err(CacheMiss::Identity);
        }
        if self.expires_at_ns.is_some_and(|expiry| now >= expiry) {
            return Err(CacheMiss::Expired);
        }
        let max_artifacts = limits
            .maximum_sources
            .checked_mul(2)
            .and_then(|n| n.checked_add(2))
            .ok_or(CacheMiss::Capacity)?;
        for a in self.artifacts() {
            if a.index >= max_artifacts || a.bytes == 0 || a.bytes > limits.maximum_bytes {
                return Err(CacheMiss::Corrupt);
            }
        }
        if let Some(seed) = self.startup_seed {
            if seed.index == self.feedback.index
                || self
                    .sources
                    .iter()
                    .any(|s| s.journal.index == seed.index || s.profile.index == seed.index)
            {
                return Err(CacheMiss::Corrupt);
            }
        }
        for s in &self.sources {
            let (raw_bytes, _) = s.encoding.original(s.journal);
            if (self.schema_version == 1 && s.encoding != JournalEncoding::IdentityV1)
                || raw_bytes == 0
                || s.checkpoint_bytes == 0
                || s.checkpoint_bytes > raw_bytes
                || raw_bytes > limits.maximum_source_bytes
                || s.journal.index == s.profile.index
                || s.profile.index == self.feedback.index
                || s.journal.index == self.feedback.index
            {
                return Err(CacheMiss::Corrupt);
            }
            if s.domains.len() > 128 || s.domains.windows(2).any(|p| p[0] >= p[1]) {
                return Err(CacheMiss::Corrupt);
            }
        }
        if self
            .sources
            .iter()
            .try_fold(0usize, |n, s| n.checked_add(s.domains.len()))
            .is_none_or(|n| n > 128)
        {
            return Err(CacheMiss::Capacity);
        }
        for (index, a) in self.artifacts().enumerate() {
            if self
                .artifacts()
                .take(index)
                .any(|b| a.index == b.index && a != b)
            {
                return Err(CacheMiss::Corrupt);
            }
        }
        Ok(())
    }
    pub fn artifacts(&self) -> impl Iterator<Item = &ArtifactReceipt> {
        self.sources
            .iter()
            .flat_map(|s| [&s.journal, &s.profile])
            .chain(std::iter::once(&self.feedback))
            .chain(self.startup_seed.iter())
    }
    pub fn generation_name(&self) -> String {
        format!("g-{}", hex(&self.generation))
    }
}
