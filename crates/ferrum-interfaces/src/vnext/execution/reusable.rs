use super::{
    canonical_fingerprint, invalid_plan, BTreeMap, Deserialize, Deserializer, DynamicBackingPoolId,
    Serialize, VNextError,
};

pub const MAX_REUSABLE_EXECUTION_BUCKETS: usize = 64;
pub const MAX_REUSABLE_EXECUTION_PROGRAM_SHAPES: usize = 256;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(transparent)]
pub struct ReusableExecutionClassId(String);

impl ReusableExecutionClassId {
    pub fn new(value: impl Into<String>) -> Result<Self, VNextError> {
        let value = value.into();
        if value.is_empty()
            || value.len() > 160
            || !value.bytes().all(|byte| {
                byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-' | b':' | b'/')
            })
        {
            return Err(invalid_plan(
                "reusable execution class id is empty, too long, or non-portable",
            ));
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl<'de> Deserialize<'de> for ReusableExecutionClassId {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Self::new(String::deserialize(deserializer)?).map_err(serde::de::Error::custom)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(transparent)]
pub struct ReusableExecutionBucketId(String);

impl ReusableExecutionBucketId {
    fn derive(
        class_id: &ReusableExecutionClassId,
        capacity: &ReusableExecutionCapacity,
    ) -> Result<Self, VNextError> {
        #[derive(Serialize)]
        struct BucketIdentity<'a> {
            domain: &'static str,
            class_id: &'a ReusableExecutionClassId,
            capacity: &'a ReusableExecutionCapacity,
        }

        Ok(Self(format!(
            "reusable-bucket/sha256/{}",
            canonical_fingerprint(
                &BucketIdentity {
                    domain: "ferrum.runtime-vnext.reusable-execution-bucket.v1",
                    class_id,
                    capacity,
                },
                "fingerprint reusable execution bucket",
            )?
        )))
    }

    fn validate_for(
        &self,
        class_id: &ReusableExecutionClassId,
        capacity: &ReusableExecutionCapacity,
    ) -> Result<(), VNextError> {
        if self != &Self::derive(class_id, capacity)? {
            return Err(invalid_plan(
                "reusable execution bucket id is not derived from its class and capacity",
            ));
        }
        Ok(())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ReusableExecutionCapacity {
    maximum_sequences: u32,
    maximum_tokens: u64,
    maximum_pages: u64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ReusableExecutionCapacityWire {
    maximum_sequences: u32,
    maximum_tokens: u64,
    maximum_pages: u64,
}

impl ReusableExecutionCapacity {
    pub fn new(
        maximum_sequences: u32,
        maximum_tokens: u64,
        maximum_pages: u64,
    ) -> Result<Self, VNextError> {
        if maximum_sequences == 0 || maximum_tokens == 0 || maximum_pages == 0 {
            return Err(invalid_plan(
                "reusable execution capacity dimensions must be non-zero",
            ));
        }
        Ok(Self {
            maximum_sequences,
            maximum_tokens,
            maximum_pages,
        })
    }

    pub const fn maximum_sequences(self) -> u32 {
        self.maximum_sequences
    }

    pub const fn maximum_tokens(self) -> u64 {
        self.maximum_tokens
    }

    pub const fn maximum_pages(self) -> u64 {
        self.maximum_pages
    }

    pub const fn covers(self, sequences: u32, tokens: u64, pages: u64) -> bool {
        sequences > 0
            && tokens > 0
            && sequences <= self.maximum_sequences
            && tokens <= self.maximum_tokens
            && pages <= self.maximum_pages
    }

    fn strictly_extends(self, previous: Self) -> bool {
        self.maximum_sequences >= previous.maximum_sequences
            && self.maximum_tokens >= previous.maximum_tokens
            && self.maximum_pages >= previous.maximum_pages
            && self != previous
    }
}

impl<'de> Deserialize<'de> for ReusableExecutionCapacity {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = ReusableExecutionCapacityWire::deserialize(deserializer)?;
        Self::new(
            wire.maximum_sequences,
            wire.maximum_tokens,
            wire.maximum_pages,
        )
        .map_err(serde::de::Error::custom)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ReusableExecutionBucketSpec {
    bucket_id: ReusableExecutionBucketId,
    class_id: ReusableExecutionClassId,
    capacity: ReusableExecutionCapacity,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ReusableExecutionBucketSpecWire {
    bucket_id: String,
    class_id: ReusableExecutionClassId,
    capacity: ReusableExecutionCapacity,
}

impl ReusableExecutionBucketSpec {
    pub fn new(
        class_id: ReusableExecutionClassId,
        capacity: ReusableExecutionCapacity,
    ) -> Result<Self, VNextError> {
        let bucket_id = ReusableExecutionBucketId::derive(&class_id, &capacity)?;
        Ok(Self {
            bucket_id,
            class_id,
            capacity,
        })
    }

    fn validate(&self) -> Result<(), VNextError> {
        self.bucket_id.validate_for(&self.class_id, &self.capacity)
    }

    pub fn bucket_id(&self) -> &ReusableExecutionBucketId {
        &self.bucket_id
    }

    pub fn class_id(&self) -> &ReusableExecutionClassId {
        &self.class_id
    }

    pub const fn capacity(&self) -> ReusableExecutionCapacity {
        self.capacity
    }
}

impl<'de> Deserialize<'de> for ReusableExecutionBucketSpec {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = ReusableExecutionBucketSpecWire::deserialize(deserializer)?;
        let spec = Self {
            bucket_id: ReusableExecutionBucketId(wire.bucket_id),
            class_id: wire.class_id,
            capacity: wire.capacity,
        };
        spec.validate().map_err(serde::de::Error::custom)?;
        Ok(spec)
    }
}

/// Exact logical startup capture case for a reusable device-program catalog.
///
/// Workspace buckets are capacity envelopes and may cover smaller work. These
/// shapes are different: each row requests one exact logical startup case.
/// The resulting physical program identity additionally includes pages,
/// provider topology, lane layout, and runtime fingerprints. Consequently one
/// logical width does not claim replay coverage for every context/topology
/// variant. v0.8.0 deliberately does not imply padded replay.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(tag = "topology", rename_all = "snake_case", deny_unknown_fields)]
pub enum ReusableExecutionProgramShape {
    UniformDecode {
        request_capacity: u32,
        token_capacity: u64,
        query_tokens_per_sequence: u32,
    },
    Prefill {
        tokens_processed: u64,
        token_capacity: u64,
        total_prompt_tokens: u64,
    },
}

impl ReusableExecutionProgramShape {
    pub fn uniform_decode(
        request_capacity: u32,
        query_tokens_per_sequence: u32,
    ) -> Result<Self, VNextError> {
        let token_capacity = u64::from(request_capacity)
            .checked_mul(u64::from(query_tokens_per_sequence))
            .ok_or_else(|| invalid_plan("reusable decode program token capacity overflows u64"))?;
        let shape = Self::UniformDecode {
            request_capacity,
            token_capacity,
            query_tokens_per_sequence,
        };
        shape.validate()?;
        Ok(shape)
    }

    pub fn prefill(
        tokens_processed: u64,
        token_capacity: u64,
        total_prompt_tokens: u64,
    ) -> Result<Self, VNextError> {
        let shape = Self::Prefill {
            tokens_processed,
            token_capacity,
            total_prompt_tokens,
        };
        shape.validate()?;
        Ok(shape)
    }

    fn validate(self) -> Result<(), VNextError> {
        match self {
            Self::UniformDecode {
                request_capacity,
                token_capacity,
                query_tokens_per_sequence,
            } => {
                let expected_tokens =
                    u64::from(request_capacity).checked_mul(u64::from(query_tokens_per_sequence));
                if request_capacity == 0
                    || query_tokens_per_sequence == 0
                    || expected_tokens != Some(token_capacity)
                {
                    return Err(invalid_plan(
                        "reusable decode program shape is empty or internally inconsistent",
                    ));
                }
            }
            Self::Prefill {
                tokens_processed,
                token_capacity,
                total_prompt_tokens,
            } => {
                if token_capacity == 0
                    || total_prompt_tokens == 0
                    || tokens_processed
                        .checked_add(token_capacity)
                        .is_none_or(|end| end > total_prompt_tokens)
                {
                    return Err(invalid_plan(
                        "reusable prefill program shape is empty or exceeds its prompt frontier",
                    ));
                }
            }
        }
        Ok(())
    }

    pub const fn request_capacity(self) -> u32 {
        match self {
            Self::UniformDecode {
                request_capacity, ..
            } => request_capacity,
            Self::Prefill { .. } => 1,
        }
    }

    pub const fn token_capacity(self) -> u64 {
        match self {
            Self::UniformDecode { token_capacity, .. } | Self::Prefill { token_capacity, .. } => {
                token_capacity
            }
        }
    }
}

impl<'de> Deserialize<'de> for ReusableExecutionProgramShape {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(tag = "topology", rename_all = "snake_case", deny_unknown_fields)]
        enum Wire {
            UniformDecode {
                request_capacity: u32,
                token_capacity: u64,
                query_tokens_per_sequence: u32,
            },
            Prefill {
                tokens_processed: u64,
                token_capacity: u64,
                total_prompt_tokens: u64,
            },
        }

        let shape = match Wire::deserialize(deserializer)? {
            Wire::UniformDecode {
                request_capacity,
                token_capacity,
                query_tokens_per_sequence,
            } => Self::UniformDecode {
                request_capacity,
                token_capacity,
                query_tokens_per_sequence,
            },
            Wire::Prefill {
                tokens_processed,
                token_capacity,
                total_prompt_tokens,
            } => Self::Prefill {
                tokens_processed,
                token_capacity,
                total_prompt_tokens,
            },
        };
        shape.validate().map_err(serde::de::Error::custom)?;
        Ok(shape)
    }
}

/// One logical startup capture case bound to its covering workspace class.
///
/// A spec budgets one observed physical variant. If startup observes more than
/// one physical identity for the same case, backend capacity fails closed; it
/// is never interpreted as universal coverage of that logical shape. Multiple
/// logical cases may independently observe the same physical identity when
/// their complete device-level identity is equal; that is physical reuse, not
/// inferred coverage from another case.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReusableExecutionProgramSpec {
    class_id: ReusableExecutionClassId,
    shape: ReusableExecutionProgramShape,
}

impl ReusableExecutionProgramSpec {
    pub fn new(
        class_id: ReusableExecutionClassId,
        shape: ReusableExecutionProgramShape,
    ) -> Result<Self, VNextError> {
        shape.validate()?;
        Ok(Self { class_id, shape })
    }

    pub fn class_id(&self) -> &ReusableExecutionClassId {
        &self.class_id
    }

    pub const fn shape(&self) -> ReusableExecutionProgramShape {
        self.shape
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReusableExecutionProgramShapeSemantics {
    Exact,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReusableExecutionCatalogMissPolicy {
    EagerFallback,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReusableExecutionCatalogLifetime {
    StartupSealed,
}

/// Fully resolved, fingerprinted logical startup-capture contract.
///
/// Physical catalog receipts remain authoritative for what was actually made
/// resident. Runtime contexts producing another pages/topology identity use
/// the explicit catalog-miss policy.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ReusableExecutionProgramPolicy {
    shape_semantics: ReusableExecutionProgramShapeSemantics,
    catalog_miss_policy: ReusableExecutionCatalogMissPolicy,
    catalog_lifetime: ReusableExecutionCatalogLifetime,
    warmup_passes: u32,
    capture_passes: u32,
    replay_validation_passes: u32,
    programs: Vec<ReusableExecutionProgramSpec>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ReusableExecutionProgramPolicyWire {
    shape_semantics: ReusableExecutionProgramShapeSemantics,
    catalog_miss_policy: ReusableExecutionCatalogMissPolicy,
    catalog_lifetime: ReusableExecutionCatalogLifetime,
    warmup_passes: u32,
    capture_passes: u32,
    replay_validation_passes: u32,
    programs: Vec<ReusableExecutionProgramSpec>,
}

impl ReusableExecutionProgramPolicy {
    pub fn exact_startup_sealed(
        warmup_passes: u32,
        capture_passes: u32,
        replay_validation_passes: u32,
        mut programs: Vec<ReusableExecutionProgramSpec>,
    ) -> Result<Self, VNextError> {
        let prerequisite_prefill_programs = programs
            .iter()
            .filter_map(|program| match program.shape() {
                ReusableExecutionProgramShape::Prefill {
                    tokens_processed,
                    total_prompt_tokens,
                    ..
                } if tokens_processed > 0 => Some(ReusableExecutionProgramSpec {
                    class_id: program.class_id().clone(),
                    shape: ReusableExecutionProgramShape::Prefill {
                        tokens_processed: 0,
                        token_capacity: tokens_processed,
                        total_prompt_tokens,
                    },
                }),
                _ => None,
            })
            .collect::<Vec<_>>();
        programs.extend(prerequisite_prefill_programs);
        programs.sort_unstable();
        programs.dedup();
        let policy = Self {
            shape_semantics: ReusableExecutionProgramShapeSemantics::Exact,
            catalog_miss_policy: ReusableExecutionCatalogMissPolicy::EagerFallback,
            catalog_lifetime: ReusableExecutionCatalogLifetime::StartupSealed,
            warmup_passes,
            capture_passes,
            replay_validation_passes,
            programs,
        };
        policy.validate()?;
        Ok(policy)
    }

    fn validate(&self) -> Result<(), VNextError> {
        if self.shape_semantics != ReusableExecutionProgramShapeSemantics::Exact
            || self.catalog_miss_policy != ReusableExecutionCatalogMissPolicy::EagerFallback
            || self.catalog_lifetime != ReusableExecutionCatalogLifetime::StartupSealed
            || self.warmup_passes == 0
            || self.capture_passes == 0
            || self.replay_validation_passes == 0
            || self.programs.is_empty()
            || self.programs.len() > MAX_REUSABLE_EXECUTION_PROGRAM_SHAPES
        {
            return Err(invalid_plan(
                "reusable execution program policy has invalid semantics, passes, or shape count",
            ));
        }
        if self.programs.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(invalid_plan(
                "reusable execution program shapes are duplicate or non-canonical",
            ));
        }
        for program in &self.programs {
            program.shape().validate()?;
        }
        let mut shape_classes = BTreeMap::new();
        for program in &self.programs {
            if shape_classes
                .insert(program.shape(), program.class_id())
                .is_some()
            {
                return Err(invalid_plan(
                    "reusable execution logical shape is assigned to multiple workspace classes",
                ));
            }
        }
        if self.programs.iter().any(|program| match program.shape() {
            ReusableExecutionProgramShape::Prefill {
                tokens_processed,
                total_prompt_tokens,
                ..
            } if tokens_processed > 0 => !self.programs.contains(&ReusableExecutionProgramSpec {
                class_id: program.class_id().clone(),
                shape: ReusableExecutionProgramShape::Prefill {
                    tokens_processed: 0,
                    token_capacity: tokens_processed,
                    total_prompt_tokens,
                },
            }),
            _ => false,
        }) {
            return Err(invalid_plan(
                "reusable prefill program policy omits a prerequisite prefix shape",
            ));
        }
        Ok(())
    }

    pub const fn shape_semantics(&self) -> ReusableExecutionProgramShapeSemantics {
        self.shape_semantics
    }

    pub const fn catalog_miss_policy(&self) -> ReusableExecutionCatalogMissPolicy {
        self.catalog_miss_policy
    }

    pub const fn catalog_lifetime(&self) -> ReusableExecutionCatalogLifetime {
        self.catalog_lifetime
    }

    pub const fn warmup_passes(&self) -> u32 {
        self.warmup_passes
    }

    pub const fn capture_passes(&self) -> u32 {
        self.capture_passes
    }

    pub const fn replay_validation_passes(&self) -> u32 {
        self.replay_validation_passes
    }

    pub fn programs(&self) -> &[ReusableExecutionProgramSpec] {
        &self.programs
    }
}

impl<'de> Deserialize<'de> for ReusableExecutionProgramPolicy {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = ReusableExecutionProgramPolicyWire::deserialize(deserializer)?;
        let policy = Self {
            shape_semantics: wire.shape_semantics,
            catalog_miss_policy: wire.catalog_miss_policy,
            catalog_lifetime: wire.catalog_lifetime,
            warmup_passes: wire.warmup_passes,
            capture_passes: wire.capture_passes,
            replay_validation_passes: wire.replay_validation_passes,
            programs: wire.programs,
        };
        policy.validate().map_err(serde::de::Error::custom)?;
        Ok(policy)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ReusableExecutionPolicy {
    maximum_reusable_lanes: u32,
    buckets: Vec<ReusableExecutionBucketSpec>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    program_policy: Option<ReusableExecutionProgramPolicy>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ReusableExecutionPolicyWire {
    maximum_reusable_lanes: u32,
    buckets: Vec<ReusableExecutionBucketSpec>,
    #[serde(default)]
    program_policy: Option<ReusableExecutionProgramPolicy>,
}

impl ReusableExecutionPolicy {
    pub fn new(
        maximum_reusable_lanes: u32,
        mut buckets: Vec<ReusableExecutionBucketSpec>,
    ) -> Result<Self, VNextError> {
        buckets.sort_by(|left, right| {
            (left.class_id(), left.capacity()).cmp(&(right.class_id(), right.capacity()))
        });
        let policy = Self {
            maximum_reusable_lanes,
            buckets,
            program_policy: None,
        };
        policy.validate()?;
        Ok(policy)
    }

    pub fn with_program_policy(
        mut self,
        program_policy: ReusableExecutionProgramPolicy,
    ) -> Result<Self, VNextError> {
        self.program_policy = Some(program_policy);
        self.validate()?;
        Ok(self)
    }

    pub(crate) fn validate(&self) -> Result<(), VNextError> {
        if self.maximum_reusable_lanes == 0
            || self.buckets.is_empty()
            || self.buckets.len() > MAX_REUSABLE_EXECUTION_BUCKETS
        {
            return Err(invalid_plan(
                "reusable execution policy has an invalid lane or bucket count",
            ));
        }
        for bucket in &self.buckets {
            bucket.validate()?;
        }
        if let Some(program_policy) = &self.program_policy {
            program_policy.validate()?;
            if program_policy.programs().iter().any(|program| {
                let shape = program.shape();
                self.smallest_covering_bucket(
                    program.class_id(),
                    shape.request_capacity(),
                    shape.token_capacity(),
                    0,
                )
                .is_none()
            }) {
                return Err(invalid_plan(
                    "reusable execution program shape has no covering workspace bucket in its class",
                ));
            }
        }
        if self.buckets.windows(2).any(|pair| {
            let left = &pair[0];
            let right = &pair[1];
            left.class_id() > right.class_id()
                || (left.class_id() == right.class_id()
                    && !right.capacity().strictly_extends(left.capacity()))
        }) {
            return Err(invalid_plan(
                "reusable execution buckets are not canonical monotonic class chains",
            ));
        }
        Ok(())
    }

    pub const fn maximum_reusable_lanes(&self) -> u32 {
        self.maximum_reusable_lanes
    }

    pub fn buckets(&self) -> &[ReusableExecutionBucketSpec] {
        &self.buckets
    }

    pub fn program_policy(&self) -> Option<&ReusableExecutionProgramPolicy> {
        self.program_policy.as_ref()
    }

    pub fn startup_capture_case_count(&self) -> usize {
        self.program_policy
            .as_ref()
            .map_or(self.buckets.len(), |policy| policy.programs().len())
    }

    pub fn bucket(
        &self,
        bucket_id: &ReusableExecutionBucketId,
    ) -> Option<&ReusableExecutionBucketSpec> {
        self.buckets
            .iter()
            .find(|bucket| bucket.bucket_id() == bucket_id)
    }

    pub fn smallest_covering_bucket(
        &self,
        class_id: &ReusableExecutionClassId,
        sequences: u32,
        tokens: u64,
        pages: u64,
    ) -> Option<&ReusableExecutionBucketSpec> {
        self.buckets.iter().find(|bucket| {
            bucket.class_id() == class_id && bucket.capacity().covers(sequences, tokens, pages)
        })
    }
}

impl<'de> Deserialize<'de> for ReusableExecutionPolicy {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = ReusableExecutionPolicyWire::deserialize(deserializer)?;
        let policy = Self {
            maximum_reusable_lanes: wire.maximum_reusable_lanes,
            buckets: wire.buckets,
            program_policy: wire.program_policy,
        };
        policy.validate().map_err(serde::de::Error::custom)?;
        Ok(policy)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ReusablePoolWorkspaceBudget {
    pool_id: DynamicBackingPoolId,
    step_bytes: u64,
    invocation_bytes: u64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ReusablePoolWorkspaceBudgetWire {
    pool_id: DynamicBackingPoolId,
    step_bytes: u64,
    invocation_bytes: u64,
}

impl ReusablePoolWorkspaceBudget {
    pub(crate) fn new(
        pool_id: DynamicBackingPoolId,
        step_bytes: u64,
        invocation_bytes: u64,
    ) -> Result<Self, VNextError> {
        let budget = Self {
            pool_id,
            step_bytes,
            invocation_bytes,
        };
        budget.validate()?;
        Ok(budget)
    }

    fn validate(&self) -> Result<(), VNextError> {
        self.step_bytes
            .checked_add(self.invocation_bytes)
            .filter(|total| *total > 0)
            .ok_or_else(|| {
                invalid_plan("reusable pool workspace budget is empty or overflows u64")
            })?;
        Ok(())
    }

    pub fn pool_id(&self) -> &DynamicBackingPoolId {
        &self.pool_id
    }

    pub const fn step_bytes(&self) -> u64 {
        self.step_bytes
    }

    pub const fn invocation_bytes(&self) -> u64 {
        self.invocation_bytes
    }

    pub fn total_bytes(&self) -> Result<u64, VNextError> {
        self.step_bytes
            .checked_add(self.invocation_bytes)
            .ok_or_else(|| invalid_plan("reusable pool workspace budget overflows u64"))
    }
}

impl<'de> Deserialize<'de> for ReusablePoolWorkspaceBudget {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = ReusablePoolWorkspaceBudgetWire::deserialize(deserializer)?;
        Self::new(wire.pool_id, wire.step_bytes, wire.invocation_bytes)
            .map_err(serde::de::Error::custom)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ResolvedReusableExecutionBucket {
    bucket: ReusableExecutionBucketSpec,
    pool_budgets: Vec<ReusablePoolWorkspaceBudget>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ResolvedReusableExecutionBucketWire {
    bucket: ReusableExecutionBucketSpec,
    pool_budgets: Vec<ReusablePoolWorkspaceBudget>,
}

impl ResolvedReusableExecutionBucket {
    pub(crate) fn new(
        bucket: ReusableExecutionBucketSpec,
        pool_budgets: Vec<ReusablePoolWorkspaceBudget>,
    ) -> Result<Self, VNextError> {
        let resolved = Self {
            bucket,
            pool_budgets,
        };
        resolved.validate()?;
        Ok(resolved)
    }

    fn validate(&self) -> Result<(), VNextError> {
        self.bucket.validate()?;
        if self
            .pool_budgets
            .windows(2)
            .any(|pair| pair[0].pool_id() >= pair[1].pool_id())
        {
            return Err(invalid_plan(
                "resolved reusable bucket pool budgets are duplicate or non-canonical",
            ));
        }
        for budget in &self.pool_budgets {
            budget.validate()?;
        }
        Ok(())
    }

    pub fn bucket(&self) -> &ReusableExecutionBucketSpec {
        &self.bucket
    }

    pub fn pool_budgets(&self) -> &[ReusablePoolWorkspaceBudget] {
        &self.pool_budgets
    }
}

impl<'de> Deserialize<'de> for ResolvedReusableExecutionBucket {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = ResolvedReusableExecutionBucketWire::deserialize(deserializer)?;
        Self::new(wire.bucket, wire.pool_budgets).map_err(serde::de::Error::custom)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ReusableExecutionMemoryPlan {
    maximum_reusable_lanes: u32,
    maximum_device_executables: u64,
    buckets: Vec<ResolvedReusableExecutionBucket>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    program_policy: Option<ReusableExecutionProgramPolicy>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ReusableExecutionMemoryPlanWire {
    maximum_reusable_lanes: u32,
    maximum_device_executables: u64,
    buckets: Vec<ResolvedReusableExecutionBucket>,
    #[serde(default)]
    program_policy: Option<ReusableExecutionProgramPolicy>,
}

impl ReusableExecutionMemoryPlan {
    pub(crate) fn new(
        maximum_reusable_lanes: u32,
        maximum_device_executables: u64,
        buckets: Vec<ResolvedReusableExecutionBucket>,
    ) -> Result<Self, VNextError> {
        Self::new_with_program_policy(
            maximum_reusable_lanes,
            maximum_device_executables,
            buckets,
            None,
        )
    }

    pub(crate) fn new_with_program_policy(
        maximum_reusable_lanes: u32,
        maximum_device_executables: u64,
        buckets: Vec<ResolvedReusableExecutionBucket>,
        program_policy: Option<ReusableExecutionProgramPolicy>,
    ) -> Result<Self, VNextError> {
        let plan = Self {
            maximum_reusable_lanes,
            maximum_device_executables,
            buckets,
            program_policy,
        };
        plan.validate_local()?;
        Ok(plan)
    }

    pub(crate) fn validate_local(&self) -> Result<(), VNextError> {
        if self.maximum_reusable_lanes == 0
            || self.maximum_device_executables == 0
            || self.buckets.is_empty()
            || self.buckets.len() > MAX_REUSABLE_EXECUTION_BUCKETS
        {
            return Err(invalid_plan(
                "reusable execution memory plan has an invalid lane, executable, or bucket count",
            ));
        }
        for bucket in &self.buckets {
            bucket.validate()?;
        }
        let mut canonical = ReusableExecutionPolicy::new(
            self.maximum_reusable_lanes,
            self.buckets
                .iter()
                .map(|bucket| bucket.bucket().clone())
                .collect(),
        )?;
        if let Some(program_policy) = self.program_policy.clone() {
            canonical = canonical.with_program_policy(program_policy)?;
        }
        if canonical.buckets().iter().ne(self
            .buckets
            .iter()
            .map(ResolvedReusableExecutionBucket::bucket))
        {
            return Err(invalid_plan(
                "reusable execution memory buckets are non-canonical",
            ));
        }
        Ok(())
    }

    pub const fn maximum_reusable_lanes(&self) -> u32 {
        self.maximum_reusable_lanes
    }

    pub const fn maximum_device_executables(&self) -> u64 {
        self.maximum_device_executables
    }

    pub fn buckets(&self) -> &[ResolvedReusableExecutionBucket] {
        &self.buckets
    }

    pub fn program_policy(&self) -> Option<&ReusableExecutionProgramPolicy> {
        self.program_policy.as_ref()
    }

    pub fn bucket(
        &self,
        bucket_id: &ReusableExecutionBucketId,
    ) -> Option<&ResolvedReusableExecutionBucket> {
        self.buckets
            .iter()
            .find(|bucket| bucket.bucket().bucket_id() == bucket_id)
    }

    pub fn smallest_covering_bucket(
        &self,
        class_id: &ReusableExecutionClassId,
        sequences: u32,
        tokens: u64,
        pages: u64,
    ) -> Option<&ResolvedReusableExecutionBucket> {
        self.buckets.iter().find(|bucket| {
            bucket.bucket().class_id() == class_id
                && bucket.bucket().capacity().covers(sequences, tokens, pages)
        })
    }

    pub(crate) fn policy(&self) -> Result<ReusableExecutionPolicy, VNextError> {
        let mut policy = ReusableExecutionPolicy::new(
            self.maximum_reusable_lanes,
            self.buckets
                .iter()
                .map(|bucket| bucket.bucket().clone())
                .collect(),
        )?;
        if let Some(program_policy) = self.program_policy.clone() {
            policy = policy.with_program_policy(program_policy)?;
        }
        Ok(policy)
    }

    pub(crate) fn pool_workspace_ceilings(
        &self,
    ) -> Result<BTreeMap<DynamicBackingPoolId, u64>, VNextError> {
        let lanes = u64::from(self.maximum_reusable_lanes);
        let mut totals = BTreeMap::new();
        for bucket in &self.buckets {
            for budget in bucket.pool_budgets() {
                let bytes = budget
                    .total_bytes()?
                    .checked_mul(lanes)
                    .ok_or_else(|| invalid_plan("reusable lane workspace budget overflows u64"))?;
                let total = totals.entry(budget.pool_id().clone()).or_insert(0_u64);
                *total = total
                    .checked_add(bytes)
                    .ok_or_else(|| invalid_plan("reusable pool workspace ceiling overflows u64"))?;
            }
        }
        Ok(totals)
    }
}

impl<'de> Deserialize<'de> for ReusableExecutionMemoryPlan {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = ReusableExecutionMemoryPlanWire::deserialize(deserializer)?;
        Self::new_with_program_policy(
            wire.maximum_reusable_lanes,
            wire.maximum_device_executables,
            wire.buckets,
            wire.program_policy,
        )
        .map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ReusableExecutionBucketSpec, ReusableExecutionCapacity, ReusableExecutionClassId,
        ReusableExecutionPolicy, ReusableExecutionProgramPolicy, ReusableExecutionProgramShape,
        ReusableExecutionProgramSpec,
    };

    fn bucket(class: &str, sequences: u32, tokens: u64) -> ReusableExecutionBucketSpec {
        ReusableExecutionBucketSpec::new(
            ReusableExecutionClassId::new(class).unwrap(),
            ReusableExecutionCapacity::new(sequences, tokens, 1).unwrap(),
        )
        .unwrap()
    }

    fn program(class: &str, shape: ReusableExecutionProgramShape) -> ReusableExecutionProgramSpec {
        ReusableExecutionProgramSpec::new(ReusableExecutionClassId::new(class).unwrap(), shape)
            .unwrap()
    }

    #[test]
    fn reusable_policy_canonicalizes_and_selects_within_one_class() {
        let policy = ReusableExecutionPolicy::new(
            1,
            vec![
                bucket("uniform-query", 4, 4),
                bucket("packed-prefill", 1, 64),
                bucket("uniform-query", 1, 1),
                bucket("uniform-query", 2, 2),
            ],
        )
        .unwrap();

        let class = ReusableExecutionClassId::new("uniform-query").unwrap();
        assert_eq!(
            policy
                .smallest_covering_bucket(&class, 3, 3, 0)
                .unwrap()
                .capacity(),
            ReusableExecutionCapacity::new(4, 4, 1).unwrap()
        );
        assert!(policy.smallest_covering_bucket(&class, 5, 5, 0).is_none());
    }

    #[test]
    fn reusable_policy_wire_rejects_derived_id_tampering() {
        let policy =
            ReusableExecutionPolicy::new(1, vec![bucket("packed-prefill", 1, 64)]).unwrap();
        let mut value = serde_json::to_value(&policy).unwrap();
        value["buckets"][0]["bucket_id"] = serde_json::Value::String("forged".to_owned());
        assert!(serde_json::from_value::<ReusableExecutionPolicy>(value).is_err());
    }

    #[test]
    fn reusable_policy_rejects_incomparable_capacities_within_class() {
        let first = bucket("mixed", 1, 64);
        let second = bucket("mixed", 4, 4);
        assert!(ReusableExecutionPolicy::new(1, vec![first, second]).is_err());
    }

    #[test]
    fn exact_program_policy_is_canonical_and_adds_prefill_prerequisite() {
        let decode_4 = program(
            "uniform-query",
            ReusableExecutionProgramShape::uniform_decode(4, 1).unwrap(),
        );
        let decode_1 = program(
            "uniform-query",
            ReusableExecutionProgramShape::uniform_decode(1, 1).unwrap(),
        );
        let final_prefill = program(
            "packed-prefill",
            ReusableExecutionProgramShape::prefill(4, 4, 8).unwrap(),
        );
        let prerequisite = program(
            "packed-prefill",
            ReusableExecutionProgramShape::prefill(0, 4, 8).unwrap(),
        );

        let program_policy = ReusableExecutionProgramPolicy::exact_startup_sealed(
            1,
            1,
            1,
            vec![
                decode_4.clone(),
                final_prefill.clone(),
                decode_1.clone(),
                decode_4.clone(),
            ],
        )
        .unwrap();

        assert_eq!(
            program_policy.programs(),
            &[prerequisite, final_prefill, decode_1, decode_4]
        );
    }

    #[test]
    fn program_policy_wire_rejects_missing_prefill_prerequisite() {
        let program_policy = ReusableExecutionProgramPolicy::exact_startup_sealed(
            1,
            1,
            1,
            vec![program(
                "packed-prefill",
                ReusableExecutionProgramShape::prefill(4, 4, 8).unwrap(),
            )],
        )
        .unwrap();
        let mut value = serde_json::to_value(program_policy).unwrap();
        value["programs"].as_array_mut().unwrap().remove(0);

        assert!(serde_json::from_value::<ReusableExecutionProgramPolicy>(value).is_err());
    }

    #[test]
    fn legacy_reusable_policy_wire_omits_program_policy() {
        let policy =
            ReusableExecutionPolicy::new(1, vec![bucket("packed-prefill", 1, 64)]).unwrap();
        let value = serde_json::to_value(&policy).unwrap();

        assert!(value.get("program_policy").is_none());
        assert_eq!(
            serde_json::from_value::<ReusableExecutionPolicy>(value).unwrap(),
            policy
        );
    }

    #[test]
    fn exact_program_policy_rejects_inconsistent_decode_shape_wire() {
        let shape = ReusableExecutionProgramShape::uniform_decode(4, 1).unwrap();
        let mut value = serde_json::to_value(shape).unwrap();
        value["token_capacity"] = serde_json::json!(3);

        assert!(serde_json::from_value::<ReusableExecutionProgramShape>(value).is_err());
    }
}
