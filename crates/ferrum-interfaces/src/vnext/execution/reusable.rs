use super::{
    canonical_fingerprint, invalid_plan, BTreeMap, Deserialize, Deserializer, DynamicBackingPoolId,
    Serialize, VNextError,
};

pub const MAX_REUSABLE_EXECUTION_BUCKETS: usize = 64;

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

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ReusableExecutionPolicy {
    maximum_reusable_lanes: u32,
    buckets: Vec<ReusableExecutionBucketSpec>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ReusableExecutionPolicyWire {
    maximum_reusable_lanes: u32,
    buckets: Vec<ReusableExecutionBucketSpec>,
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
        };
        policy.validate()?;
        Ok(policy)
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
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ReusableExecutionMemoryPlanWire {
    maximum_reusable_lanes: u32,
    maximum_device_executables: u64,
    buckets: Vec<ResolvedReusableExecutionBucket>,
}

impl ReusableExecutionMemoryPlan {
    pub(crate) fn new(
        maximum_reusable_lanes: u32,
        maximum_device_executables: u64,
        buckets: Vec<ResolvedReusableExecutionBucket>,
    ) -> Result<Self, VNextError> {
        let plan = Self {
            maximum_reusable_lanes,
            maximum_device_executables,
            buckets,
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
        let canonical = ReusableExecutionPolicy::new(
            self.maximum_reusable_lanes,
            self.buckets
                .iter()
                .map(|bucket| bucket.bucket().clone())
                .collect(),
        )?;
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

    pub(crate) fn policy(&self) -> Result<ReusableExecutionPolicy, VNextError> {
        ReusableExecutionPolicy::new(
            self.maximum_reusable_lanes,
            self.buckets
                .iter()
                .map(|bucket| bucket.bucket().clone())
                .collect(),
        )
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
        Self::new(
            wire.maximum_reusable_lanes,
            wire.maximum_device_executables,
            wire.buckets,
        )
        .map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ReusableExecutionBucketSpec, ReusableExecutionCapacity, ReusableExecutionClassId,
        ReusableExecutionPolicy,
    };

    fn bucket(class: &str, sequences: u32, tokens: u64) -> ReusableExecutionBucketSpec {
        ReusableExecutionBucketSpec::new(
            ReusableExecutionClassId::new(class).unwrap(),
            ReusableExecutionCapacity::new(sequences, tokens, 1).unwrap(),
        )
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
}
