//! Typed identities read from installed processor objects, in execution order.
//! No request-parameter cache or processor display name is an authority.
use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum BuiltinLogitsProcessorCostV1 {
    Temperature {
        bits: u32,
    },
    TopK {
        k: usize,
    },
    TopP {
        bits: u32,
    },
    MinP {
        bits: u32,
    },
    RepetitionPenalty {
        bits: u32,
    },
    PresenceFrequencyPenalty {
        presence_bits: u32,
        frequency_bits: u32,
    },
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BuiltinSamplerCostV1 {
    GreedyV1,
    MultinomialV1,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SamplingCostIdentityV1 {
    pub processors: Vec<BuiltinLogitsProcessorCostV1>,
    pub sampler: BuiltinSamplerCostV1,
}
impl SamplingConfig {
    /// Ordered installed algorithms and their actual stored numeric parameters.
    /// This declaration is inspected before capture, never in token sampling.
    pub fn cost_identity(&self) -> Option<SamplingCostIdentityV1> {
        Some(SamplingCostIdentityV1 {
            processors: self
                .processor_chain
                .processors
                .iter()
                .map(|p| p.cost_identity())
                .collect::<Option<Vec<_>>>()?,
            sampler: self.sampler.cost_identity()?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    struct UnknownTopK;
    impl LogitsProcessor for UnknownTopK {
        fn process(&self, _ctx: &mut SamplingContext) -> Result<()> {
            Ok(())
        }
        fn name(&self) -> &str {
            "top_k"
        }
    }
    #[test]
    fn same_named_parameter_drift_and_unknown_processors_cannot_share_identity() {
        let base = SamplingParams {
            temperature: 0.7,
            top_k: Some(4),
            top_p: 0.8,
            repetition_penalty: 1.1,
            presence_penalty: 0.2,
            frequency_penalty: 0.3,
            min_p: Some(0.01),
            ..SamplingParams::greedy()
        };
        let expected = SamplingConfig::from_params(&base).cost_identity().unwrap();
        let changes = [
            SamplingParams {
                temperature: 0.6,
                ..base.clone()
            },
            SamplingParams {
                top_k: Some(2),
                ..base.clone()
            },
            SamplingParams {
                top_p: 0.9,
                ..base.clone()
            },
            SamplingParams {
                repetition_penalty: 1.2,
                ..base.clone()
            },
            SamplingParams {
                presence_penalty: 0.4,
                ..base.clone()
            },
            SamplingParams {
                frequency_penalty: 0.4,
                ..base.clone()
            },
            SamplingParams {
                min_p: Some(0.02),
                ..base.clone()
            },
        ];
        for changed in changes {
            let installed = SamplingConfig::from_params(&changed);
            assert_eq!(
                installed.processor_chain.processor_names(),
                SamplingConfig::from_params(&base)
                    .processor_chain
                    .processor_names()
            );
            assert_ne!(installed.cost_identity().unwrap(), expected);
        }
        let unknown = SamplingConfig {
            processor_chain: LogitsProcessorChain::new().add_processor(Box::new(UnknownTopK)),
            sampler: Box::new(GreedySampler),
        };
        assert!(unknown.cost_identity().is_none());
    }
    #[test]
    fn identity_describes_parameters_actually_used_by_installed_processor() {
        let plan = SamplingConfig::from_params(&SamplingParams {
            top_k: Some(4),
            ..SamplingParams::greedy()
        });
        let later_request = SamplingParams {
            top_k: Some(2),
            ..SamplingParams::greedy()
        };
        let frequencies = HashMap::new();
        let mut logits = [5., 4., 3., 2., 1.];
        let mut ctx = SamplingContext::new(0, &later_request, &mut logits, &[], &frequencies, 5);
        plan.processor_chain.process(&mut ctx).unwrap();
        assert_eq!(ctx.logits.iter().filter(|v| v.is_finite()).count(), 4);
        assert_eq!(
            plan.cost_identity().unwrap().processors,
            [BuiltinLogitsProcessorCostV1::TopK { k: 4 }]
        );
        assert_ne!(
            plan.cost_identity(),
            SamplingConfig::from_params(&later_request).cost_identity()
        );
    }
}
