//! Same nonzero six-format physical weights as X, with a separate large-row
//! profile. This fixture never selects the Y G32 cutover arithmetic.
use super::*;

pub const PROFILE: &str = "fixture.attention-ffn.require-extra-large-prefill-marker-v2";

impl Family {
    pub fn extra_prefill(kind: AttentionKind, maximum_tokens: u64) -> Self {
        assert!(maximum_tokens >= 2048);
        let mut value = Self::extra(kind);
        value.selected = match kind {
            AttentionKind::GatedDelta => UpstreamMarkerV2Profile::GatedDeltaExtraLargePrefill,
            AttentionKind::Causal => UpstreamMarkerV2Profile::CausalExtraLargePrefill,
            _ => unreachable!(),
        };
        value.maximum_tokens = maximum_tokens;
        value
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extra_large_prefill_fixture_preserves_x_weights_small_routes_and_real_state_capacity() {
        for kind in [AttentionKind::GatedDelta, AttentionKind::Causal] {
            let mut old = Family::extra(kind);
            old.maximum_tokens = 2052;
            let new = Family::extra_prefill(kind, 2052);
            let schema = new.weight_schema(&kind).unwrap();
            assert_eq!(schema, old.weight_schema(&kind).unwrap());
            assert_eq!(new.states(), old.states());
            let old_weights = Weights::new(&old.weight_schema(&kind).unwrap());
            let new_weights = Weights::new(&schema);
            for component in &schema.components {
                assert_eq!(
                    old_weights.component(component).unwrap().bytes(),
                    new_weights.component(component).unwrap().bytes()
                );
            }
            for (before, after) in [
                (old.attention_arithmetic(), new.attention_arithmetic()),
                (old.swiglu_arithmetic(), new.swiglu_arithmetic()),
            ] {
                after.validate().unwrap();
                assert_eq!(
                    after.schema_version,
                    COMPOSITE_NUMERICAL_ARITHMETIC_SCHEMA_VERSION_UPSTREAM_EXTRA_PREFILL
                );
                assert_ne!(before, after);
                for (a, b) in before.projections.iter().zip(&after.projections) {
                    assert_eq!(a.role, b.role);
                    assert_eq!(a.leaves.len(), b.leaves.len());
                    for (a, b) in a.leaves.iter().zip(&b.leaves) {
                        assert_eq!(a.format, b.format);
                        let policies = [&a.arithmetic, &b.arithmetic].map(|v| {
                            let [NumericalArithmeticStage::UpstreamProjection { policy }] =
                                v.stages.as_slice()
                            else {
                                panic!("X and new prefill must retain upstream arithmetic")
                            };
                            policy
                        });
                        let select = |index: usize, rows| {
                            policies[index].routes.iter().find(|route| {
                                route.layout == UpstreamProjectionLayout::Columns
                                    && route.contains_rows(rows)
                            })
                        };
                        for rows in 1..=32 {
                            assert_eq!(
                                select(0, rows).map(|r| r.arithmetic),
                                select(1, rows).map(|r| r.arithmetic),
                                "small-row arithmetic/fallback changed for {rows}"
                            );
                        }
                        for rows in [33, 54, 2048] {
                            let selected = select(1, rows).unwrap();
                            if a.format.is_extra() {
                                assert!(select(0, rows).is_none());
                                assert_eq!(
                                    selected.arithmetic,
                                    UpstreamProjectionArithmetic::MmqD4ExtraMarkerV2
                                );
                            } else {
                                assert_eq!(
                                    select(0, rows).map(|r| r.arithmetic),
                                    Some(selected.arithmetic)
                                );
                            }
                        }
                        assert!(select(1, 2049).is_none());
                    }
                }
            }
            let registration = TypedFamilyRegistration::new(new);
            let definition = registration
                .define(&serde_json::to_value(kind).unwrap())
                .unwrap();
            let prepared = registration.prepare(&definition, &id(PROFILE)).unwrap();
            definition
                .numerical_profiles()
                .resolve(&id(PROFILE))
                .unwrap()
                .validate_program(prepared.program())
                .unwrap();
            for state in prepared.program().states() {
                if let StateCapacityDemand::TokenScaled { maximum_tokens, .. } =
                    state.capacity_demand
                {
                    assert_eq!(maximum_tokens, 2052);
                }
            }
        }
    }
}
