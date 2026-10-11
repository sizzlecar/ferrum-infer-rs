//! Preserve Z's six nonzero formats and actual state; change only the profile.
use super::*;

pub const PROFILE: &str = "fixture.attention-ffn.require-extra-all-rows-marker-v2";

impl Family {
    pub fn extra_all_rows(kind: AttentionKind, maximum_tokens: u64) -> Self {
        let mut value = Self::extra_prefill(kind, maximum_tokens);
        value.selected = match kind {
            AttentionKind::GatedDelta => UpstreamMarkerV2Profile::GatedDeltaExtraAllRows,
            AttentionKind::Causal => UpstreamMarkerV2Profile::CausalExtraAllRows,
            _ => unreachable!(),
        };
        value
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extra_all_rows_fixture_preserves_z_weights_state_and_unchanged_routes() {
        for kind in [AttentionKind::GatedDelta, AttentionKind::Causal] {
            let old = Family::extra_prefill(kind, 2052);
            let new = Family::extra_all_rows(kind, 2052);
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
                for (a, b) in before.projections.iter().zip(&after.projections) {
                    assert_eq!(a.role, b.role);
                    assert_eq!(a.leaves.len(), b.leaves.len());
                    for (a, b) in a.leaves.iter().zip(&b.leaves) {
                        assert_eq!(a.format, b.format);
                        if !a.format.is_extra() {
                            assert_eq!(a, b);
                            continue;
                        }
                        let policies = [&a.arithmetic, &b.arithmetic]
                            .map(|v| v.upstream_policy().expect("actual extra projection policy"));
                        let selected = |index: usize, rows| {
                            policies[index].routes.iter().find(|r| {
                                r.layout == UpstreamProjectionLayout::Columns
                                    && r.contains_rows(rows)
                            })
                        };
                        for rows in [1, 4, 8, 16, 32, 33, 2048] {
                            assert_eq!(
                                selected(0, rows).map(|r| r.arithmetic),
                                selected(1, rows).map(|r| r.arithmetic)
                            );
                        }
                        for rows in 1..=32 {
                            let route = selected(1, rows).expect("all small rows supported");
                            if selected(0, rows).is_none() {
                                assert_eq!(
                                    route.arithmetic,
                                    UpstreamProjectionArithmetic::MmqD4ExtraMarkerV2
                                );
                            }
                        }
                        for rows in [0, 2049] {
                            assert!(selected(1, rows).is_none());
                        }
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
