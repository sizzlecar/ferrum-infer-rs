//! Separately observed row categories for an explicitly empirical model.
//! These rows retain physical order. Only a statistical consumer may permute
//! complete work/numeric/category tuples within an unchanged role segment.
use super::*;

pub const HOST_ROW_MULTISET_FEATURE_SCHEMA_V2: u32 = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HostRowRoleV2 {
    Prefill,
    Decode,
}

impl HostRowRoleV2 {
    pub fn matches_work(self, work: ActualRowWork) -> bool {
        matches!(
            (self, work),
            (Self::Prefill, ActualRowWork::Prefill { .. })
                | (Self::Decode, ActualRowWork::Decode { .. })
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HostRowStaticCostFeaturesV2 {
    pub role: HostRowRoleV2,
    /// Installed host policy and this row's categorical work, including its
    /// first/final-token branches and mask upload. Never a request identity.
    pub categorical_signature: [u8; 32],
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HostRowMultisetCostFeaturesV2 {
    pub schema_version: u32,
    /// Actual whole-wave product, readback route and row-order protocol. The
    /// provider signature and physical role layout remain separate evidence.
    pub wave_policy_signature: [u8; 32],
    #[serde(deserialize_with = "bounded_static_rows")]
    pub rows: Vec<HostRowStaticCostFeaturesV2>,
}

impl HostRowMultisetCostFeaturesV2 {
    pub fn validate(&self, expected_rows: usize) -> Result<(), CostFeatureError> {
        if self.schema_version != HOST_ROW_MULTISET_FEATURE_SCHEMA_V2 {
            return Err(CostFeatureError::UnsupportedSchema);
        }
        if self.rows.is_empty()
            || self.rows.len() > MAX_COST_ROWS
            || self.rows.len() != expected_rows
        {
            return Err(CostFeatureError::InvalidState);
        }
        Ok(())
    }
}

fn bounded_static_rows<'de, D>(
    deserializer: D,
) -> Result<Vec<HostRowStaticCostFeaturesV2>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    struct Rows;
    impl<'de> serde::de::Visitor<'de> for Rows {
        type Value = Vec<HostRowStaticCostFeaturesV2>;
        fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(
                formatter,
                "at most {MAX_COST_ROWS} physical static cost rows"
            )
        }
        fn visit_seq<A: serde::de::SeqAccess<'de>>(
            self,
            mut seq: A,
        ) -> Result<Self::Value, A::Error> {
            let mut rows = Vec::new();
            while let Some(row) = seq.next_element()? {
                if rows.len() == MAX_COST_ROWS {
                    return Err(serde::de::Error::custom("static cost row limit exceeded"));
                }
                rows.push(row);
            }
            Ok(rows)
        }
    }
    deserializer.deserialize_seq(Rows)
}
