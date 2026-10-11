//! Closed, explicitly versioned local-row arithmetic; never offered concurrency.
use super::*;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct G32MmqPrefillPolicy {
    pub format: ProjectionBlockFormat,
    pub g32: StagedNumericalArithmetic,
    pub mmq: UpstreamProjectionPolicy,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum G32MmqPrefillRoute {
    G32,
    MmqMarkerV2,
    Strict,
}
impl G32MmqPrefillPolicy {
    pub fn new(format: ProjectionBlockFormat) -> Result<Self, String> {
        let g32 = crate::vnext::Q8ActSwiGluProfile::Q4KQ5KIq4Xs
            .arithmetic()
            .projections[0]
            .leaves
            .iter()
            .find(|leaf| leaf.format == format)
            .ok_or("G32/MMQ hybrid only supports its original three formats")?
            .arithmetic
            .clone();
        Ok(Self {
            format,
            g32,
            mmq: UpstreamProjectionPolicy {
                format,
                fallback: StrictProjectionFallback::RetainBaseArithmetic {},
                routes: vec![UpstreamProjectionRouteDeclaration {
                    arithmetic: if format == ProjectionBlockFormat::Iq4Xs {
                        UpstreamProjectionArithmetic::MmqD4MarkerV2
                    } else {
                        UpstreamProjectionArithmetic::MmqDs4MarkerV2
                    },
                    layout: UpstreamProjectionLayout::Columns,
                    local_rows: BTreeSet::new(),
                    prefill_rows: Some(UpstreamPrefillRows {
                        first: 33,
                        last: 2048,
                    }),
                }],
            },
        })
    }
    pub fn validate(&self) -> Result<(), String> {
        if self.format.is_extra() {
            return Err("G32/MMQ hybrid only supports its original three formats".into());
        }
        self.g32.validate()?;
        self.mmq.validate()?;
        if self != &Self::new(self.format)? {
            return Err("G32/MMQ v1 requires exact AtN five stages and the disjoint 33..2048 MMQ MarkerV2 domain".into());
        }
        Ok(())
    }
    pub const fn select(local_rows: u32) -> G32MmqPrefillRoute {
        match local_rows {
            1..=32 => G32MmqPrefillRoute::G32,
            33..=2048 => G32MmqPrefillRoute::MmqMarkerV2,
            _ => G32MmqPrefillRoute::Strict,
        }
    }
    /// Conservative sufficient proof: any grouping of a contiguous real wave
    /// with at most32 rows has only G32 local launches. Larger waves are not
    /// fast-pathed even if a backend might split them into smaller groups.
    pub fn g32_partition(
        total: u64,
        ranges: impl IntoIterator<Item = std::ops::Range<u64>>,
    ) -> Result<bool, String> {
        if total == 0 {
            return Err("empty hybrid wave".into());
        }
        let mut covered = 0_u64;
        for range in ranges {
            let width = range
                .end
                .checked_sub(range.start)
                .filter(|n| *n > 0)
                .ok_or("empty or reversed participant range")?;
            if range.start != covered {
                return Err("participant ranges do not exactly cover the wave".into());
            }
            covered = covered
                .checked_add(width)
                .ok_or("participant coverage overflows")?;
            if covered > total {
                return Err("participant range exceeds wave".into());
            }
        }
        if covered != total {
            return Err("participant ranges omit wave rows".into());
        }
        Ok(total <= 32)
    }
    pub fn staged(self) -> StagedNumericalArithmetic {
        StagedNumericalArithmetic {
            schema_version: NUMERICAL_ARITHMETIC_SCHEMA_VERSION_G32_MMQ,
            stages: vec![NumericalArithmeticStage::G32MmqPrefillProjection { policy: self }],
        }
    }
}
impl StagedNumericalArithmetic {
    /// The MMQ branch only; callers must still select using actual local rows.
    pub fn upstream_policy(&self) -> Option<&UpstreamProjectionPolicy> {
        match self.stages.as_slice() {
            [NumericalArithmeticStage::UpstreamProjection { policy }] => Some(policy),
            [NumericalArithmeticStage::G32MmqPrefillProjection { policy }] => Some(&policy.mmq),
            _ => None,
        }
    }
    pub fn g32_mmq_policy(&self) -> Option<&G32MmqPrefillPolicy> {
        match self.stages.as_slice() {
            [NumericalArithmeticStage::G32MmqPrefillProjection { policy }] => Some(policy),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vnext::*;

    #[test]
    fn hybrid_g32_mmq_contract_preserves_exact_arithmetic_and_legacy_schema_boundaries() {
        for profile in [
            UpstreamMarkerV2Profile::SwiGluG32MmqPrefill,
            UpstreamMarkerV2Profile::GatedDeltaG32MmqPrefill,
            UpstreamMarkerV2Profile::CausalG32MmqPrefill,
        ] {
            let contract = profile.arithmetic();
            contract.validate().unwrap();
            for projection in &contract.projections {
                for leaf in &projection.leaves {
                    let policy = leaf.arithmetic.g32_mmq_policy().unwrap();
                    let old = Q8ActSwiGluProfile::Q4KQ5KIq4Xs.arithmetic();
                    assert_eq!(
                        policy.g32,
                        old.projections[0]
                            .leaves
                            .iter()
                            .find(|l| l.format == leaf.format)
                            .unwrap()
                            .arithmetic
                    );
                    for m in [1, 3, 4, 7, 8, 9, 16, 32] {
                        assert_eq!(G32MmqPrefillPolicy::select(m), G32MmqPrefillRoute::G32);
                        assert!(!policy.mmq.routes.iter().any(|r| r.contains_rows(m)));
                    }
                    for m in [33, 155, 2048] {
                        assert_eq!(
                            G32MmqPrefillPolicy::select(m),
                            G32MmqPrefillRoute::MmqMarkerV2
                        );
                        assert!(policy.mmq.routes[0].contains_rows(m));
                    }
                    for m in [0, 2049, u32::MAX] {
                        assert_eq!(G32MmqPrefillPolicy::select(m), G32MmqPrefillRoute::Strict);
                    }
                    let mut invalid = policy.clone();
                    invalid.mmq.routes[0].prefill_rows.as_mut().unwrap().first = 32;
                    assert!(invalid.validate().is_err());
                    let mut invalid = policy.clone();
                    if let NumericalArithmeticStage::IntegerDot {
                        values_per_partial, ..
                    } = &mut invalid.g32.stages[1]
                    {
                        *values_per_partial = 4;
                    }
                    assert!(invalid.validate().is_err());
                    for schema in [
                        NUMERICAL_ARITHMETIC_SCHEMA_VERSION,
                        NUMERICAL_ARITHMETIC_SCHEMA_VERSION_UPSTREAM,
                    ] {
                        let mut invalid = leaf.arithmetic.clone();
                        invalid.schema_version = schema;
                        assert!(invalid.validate().is_err());
                    }
                }
            }
            for schema in [1, 2, 3] {
                let mut invalid = contract.clone();
                invalid.schema_version = schema;
                assert!(invalid.validate().is_err());
            }
            let json = serde_json::to_vec(&contract).unwrap();
            assert_eq!(
                serde_json::from_slice::<CompositeNumericalArithmetic>(&json).unwrap(),
                contract
            );
        }
        let old = UpstreamMarkerV2Profile::SwiGluPrefill.arithmetic();
        let wire = serde_json::to_string(&old).unwrap();
        assert!(!wire.contains("g32_mmq"));
        assert_eq!(
            serde_json::from_str::<CompositeNumericalArithmetic>(&wire).unwrap(),
            old
        );
    }
}

#[cfg(test)]
mod partition_tests {
    use super::*;
    #[test]
    fn hybrid_g32_binding_partition_requires_complete_positive_small_wave() {
        for widths in [
            vec![1],
            vec![3, 7, 9],
            vec![8, 8, 8, 8],
            vec![32],
            vec![1, 32],
            vec![33],
            vec![2048],
        ] {
            let mut cursor = 0;
            let parts: Vec<_> = widths
                .iter()
                .map(|&n| {
                    let r = cursor..cursor + n;
                    cursor += n;
                    r
                })
                .collect();
            assert_eq!(
                G32MmqPrefillPolicy::g32_partition(cursor, parts).unwrap(),
                cursor <= 32
            );
        }
        for (total, parts) in [
            (0, vec![]),
            (32, vec![]),
            (32, vec![0..31]),
            (32, vec![0..16, 15..32]),
            (32, vec![0..16, 17..32]),
            (32, vec![0..0, 0..32]),
            (32, vec![1..33]),
            (32, vec![0..33]),
            (
                u64::MAX,
                vec![
                    0..u64::MAX,
                    std::ops::Range {
                        start: u64::MAX,
                        end: 0,
                    },
                ],
            ),
        ] {
            assert!(G32MmqPrefillPolicy::g32_partition(total, parts).is_err());
        }
    }
}
