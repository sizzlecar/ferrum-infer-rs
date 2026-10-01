//! Checked identities for an input-only opportunity inventory. A unique
//! projected identity does not promise that a request reaches this input.
//! Actual source8 membership, cohort fences and qualification remain unchanged.
use ferrum_scheduler::implementations::continuous::cost_model::structured_v2::{
    NumericalFamilyKeyV1, StructuredInputV2, StructuredOwnerKeyV2, StructuredPopulationPolicyV1,
    StructuredUnknownV2,
};

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub(in crate::continuous_engine::inner::calibration) enum CheckedPopulationKey {
    NumericalFamily(NumericalFamilyKeyV1),
    ExactOwner(StructuredOwnerKeyV2),
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub(super) enum CasePopulation {
    Unique(CheckedPopulationKey),
    Alternatives(Vec<CheckedPopulationKey>),
    /// A failed/unavailable branch cannot be removed from the population.
    /// Retain successful identities for diagnostics without assigning a floor.
    Unknown {
        known_alternatives: Vec<CheckedPopulationKey>,
    },
}

impl CasePopulation {
    fn possible_keys(&self) -> &[CheckedPopulationKey] {
        match self {
            Self::Unique(key) => std::slice::from_ref(key),
            Self::Alternatives(keys) => keys,
            Self::Unknown { known_alternatives } => known_alternatives,
        }
    }
}

/// The caller supplies every checked alternative of the same projected input.
/// `complete` must be false if any route/host branch is Unknown. Neither a
/// PrefixKind label nor a desired product may be used to filter alternatives:
/// narrowing requires the real producer's validated prefix/host constraints.
pub(super) fn classify_alternatives(
    inputs: &[StructuredInputV2],
    policy: StructuredPopulationPolicyV1,
    complete: bool,
) -> Result<CasePopulation, StructuredUnknownV2> {
    let mut keys = Vec::new();
    keys.try_reserve(inputs.len())
        .map_err(|_| StructuredUnknownV2::Capacity)?;
    for input in inputs {
        // StructuredInputV2 is created through the checked recipe projection;
        // do not let an unbound input become a fallback exact population.
        if input.physical_domain_signature().is_none() {
            return Err(StructuredUnknownV2::WrongDomain);
        }
        let key = match policy {
            StructuredPopulationPolicyV1::ExactOwnerV1 => {
                CheckedPopulationKey::ExactOwner(input.owner().clone())
            }
            StructuredPopulationPolicyV1::HomogeneousOrdinaryDecodeV1 => {
                match input.numerical_family_key() {
                    Ok(key) => CheckedPopulationKey::NumericalFamily(key),
                    Err(StructuredUnknownV2::UnsupportedScope) => {
                        CheckedPopulationKey::ExactOwner(input.owner().clone())
                    }
                    Err(reason) => return Err(reason),
                }
            }
        };
        if !keys.contains(&key) {
            keys.push(key);
        }
    }
    if !complete || keys.is_empty() {
        return Ok(CasePopulation::Unknown {
            known_alternatives: keys,
        });
    }
    if keys.len() == 1 {
        Ok(CasePopulation::Unique(keys.pop().unwrap()))
    } else {
        Ok(CasePopulation::Alternatives(keys))
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub(super) struct CaseOpportunity {
    pub population: CasePopulation,
    /// Zero or one original fresh eligible input, conditional on this original
    /// cohort completing. Identity uniqueness alone never establishes one.
    /// Ordinary future decode is zero when EOS/Stop can prevent reaching it.
    /// A prepared suffix may be one only with validated original prefix/release
    /// constraints; potential Greedy/Full alternatives cannot each receive one.
    pub minimum_fresh_members: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub(super) struct PopulationMemberGroup {
    pub key: CheckedPopulationKey,
    /// Only these original cohort indices may enter budget::fresh_span marked.
    /// It must still skip a cut cohort under the original source8 phase fence.
    pub guaranteed_case_indices: Vec<usize>,
    /// Includes guaranteed indices as well as ambiguous/unreachable identities.
    /// This is inventory evidence, never a count of original members.
    pub possible_case_indices: Vec<usize>,
}

pub(super) fn member_groups(
    cases: &[CaseOpportunity],
) -> Result<Vec<PopulationMemberGroup>, StructuredUnknownV2> {
    let mut groups: Vec<PopulationMemberGroup> = Vec::new();
    for (index, case) in cases.iter().enumerate() {
        if case.minimum_fresh_members > 1 {
            return Err(StructuredUnknownV2::InvalidInput);
        }
        let guaranteed = case.minimum_fresh_members == 1
            && matches!(&case.population, CasePopulation::Unique(_));
        for key in case.population.possible_keys() {
            let position = match groups.iter().position(|group| group.key == *key) {
                Some(position) => position,
                None => {
                    groups
                        .try_reserve(1)
                        .map_err(|_| StructuredUnknownV2::Capacity)?;
                    groups.push(PopulationMemberGroup {
                        key: key.clone(),
                        guaranteed_case_indices: Vec::new(),
                        possible_case_indices: Vec::new(),
                    });
                    groups.len() - 1
                }
            };
            let group = &mut groups[position];
            if group.possible_case_indices.last() != Some(&index) {
                group
                    .possible_case_indices
                    .try_reserve(1)
                    .map_err(|_| StructuredUnknownV2::Capacity)?;
                group.possible_case_indices.push(index);
            }
            if guaranteed && group.guaranteed_case_indices.last() != Some(&index) {
                group
                    .guaranteed_case_indices
                    .try_reserve(1)
                    .map_err(|_| StructuredUnknownV2::Capacity)?;
                group.guaranteed_case_indices.push(index);
            }
        }
    }
    Ok(groups)
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;
    use ferrum_interfaces::{execution_cost::*, vnext::DeviceCommandPhase};
    use std::num::{NonZeroU32, NonZeroU64};

    const POLICY: StructuredPopulationPolicyV1 =
        StructuredPopulationPolicyV1::HomogeneousOrdinaryDecodeV1;

    pub(in super::super) fn domain() -> CostWorkloadDomainV1 {
        CostWorkloadDomainV1::new_vnext(
            &ExecutorCostIdentity {
                schema_version: EXECUTOR_COST_IDENTITY_SCHEMA,
                model_weights: [1; 32],
                numerical_policy: [2; 32],
                device_runtime: [3; 32],
                execution_config: [4; 32],
            },
            CostWorkloadLimitsV1 {
                maximum_rows: NonZeroU32::new(8).unwrap(),
                maximum_context_tokens: NonZeroU32::new(1024).unwrap(),
                maximum_scheduled_tokens_per_wave: NonZeroU64::new(8).unwrap(),
                output_vocabulary_elements: NonZeroU64::new(4096).unwrap(),
                repetition_slot_capacity: 512,
                fixed_state_bytes_per_row: 32,
            },
        )
        .unwrap()
    }

    // The production canonical builder and physical-domain binding are the
    // same checked construction chain used by scheduler numerical-family tests.
    // No serialized key or manually edited StructuredInput creates authority.
    pub(in super::super) fn input(
        rows: u32,
        generated: u64,
        product: CostProductOutput,
        heterogeneous: bool,
        bound: bool,
    ) -> StructuredInputV2 {
        input_with_context(rows, generated, 64, product, heterogeneous, bound)
    }

    pub(in super::super) fn input_with_context(
        rows: u32,
        generated: u64,
        kv_tokens: u32,
        product: CostProductOutput,
        heterogeneous: bool,
        bound: bool,
    ) -> StructuredInputV2 {
        let mut selected = SelectedCommandCostBuilderV1::new_with_algorithm_work(u64::from(rows));
        selected
            .kernel(
                SelectedAlgorithmClassV1::new("fixture.probe-population", 1, [1; 32], [2; 32])
                    .unwrap(),
                KernelNumericWorkV1 {
                    logical_units: u64::from(rows) * 8,
                    padded_units: u64::from(rows) * 8,
                    inner_units_per_logical_unit: 2,
                    grid: [rows, 1, 1],
                    scratch_bytes: u64::from(rows) * 32,
                    staged_weight_bytes: 0,
                },
            )
            .unwrap();
        let selected = selected.finish().unwrap();
        let mut builder = CanonicalWaveCostBuilder::new_with_structured_statistics(0, product);
        builder
            .physical_command(CostPhysicalCommand {
                native_op_id: "fixture.probe-population",
                command_index: 0,
                node_index: Some(0),
                command_phase: DeviceCommandPhase::Compute,
                provider: Some(CostProviderIdentity {
                    provider_id: "numerical-fixture",
                    implementation_fingerprint: "v1",
                    operation_fingerprint: "v1",
                }),
                path: CostCommandPath::Eager,
                participant_start: 0,
                participant_count: rows,
                token_count: u64::from(rows),
                batching_form: "packed",
                compute_dispatch_count: 1,
                transfer_command_count: 0,
                reusable_graph_node_count: None,
                statistical_evidence: Some(&selected),
            })
            .unwrap();
        builder
            .core_readback_route(CoreReadbackRoute::SubmissionStaged)
            .unwrap();
        for row in 0..rows {
            builder
                .row(CanonicalCostRow {
                    work: ActualRowWork::Decode { kv_tokens },
                    host_policy_signature: [3; 32],
                    mask_upload_required: false,
                    output: CostRowOutput::Decode {
                        requires_full_logits: product == CostProductOutput::FullLogits,
                        repetition_tokens: 0,
                        repetition_penalty_bits: 1f32.to_bits(),
                    },
                    host_features: Some(HostCostFeaturesV1 {
                        policy: HostCostPolicyV2 {
                            empirical_content_domain: Some(HostContentDomainV1::PlainTextGreedyV1),
                            categorical_signature: if heterogeneous && row + 1 == rows {
                                [5; 32]
                            } else {
                                [4; 32]
                            },
                            decoder_text_bytes_per_token: 4,
                            decoder_scratch_bytes_per_token: 8,
                            raw_token_bytes_bound: 4,
                        },
                        state: HostCostStateV1 {
                            generated_tokens_before: generated,
                            maximum_output_tokens: generated + 18,
                            sampling_history_tokens: generated,
                            sampling_history_scope: CostSamplingHistoryScope::FullGeneration,
                            pending_decoded_utf8: false,
                            completion_state_signature: satisfied_completion_cost_signature(),
                        },
                    }),
                })
                .unwrap();
        }
        let wave = builder
            .finish_with_captured_structure(
                ActualWaveKind::Decode,
                ActualWavePath::PlanRuntime,
                ActualWaveGraphState::Disabled,
                ActualWaveRowOrder::Ordered,
                u64::from(rows) * 32,
            )
            .unwrap();
        let selected = wave.statistical.as_ref().unwrap();
        let recipe = selected.structured_capture().unwrap().unwrap();
        if bound {
            StructuredInputV2::from_actual_with_domain(&wave.exact, selected, recipe, &domain())
                .unwrap()
        } else {
            StructuredInputV2::from_actual(&wave.exact, selected, recipe).unwrap()
        }
    }

    #[test]
    fn checked_population_unique_widths_do_not_imply_a_reachable_member() {
        let a = input(1, 3, CostProductOutput::GreedyToken, false, true);
        let b = input(2, 3, CostProductOutput::GreedyToken, false, true);
        assert_ne!(a.owner(), b.owner());
        let unique = classify_alternatives(&[a, b], POLICY, true).unwrap();
        assert!(matches!(
            unique,
            CasePopulation::Unique(CheckedPopulationKey::NumericalFamily(_))
        ));
        let groups = member_groups(&[
            CaseOpportunity {
                population: unique.clone(),
                minimum_fresh_members: 0,
            },
            CaseOpportunity {
                population: unique,
                minimum_fresh_members: 1,
            },
        ])
        .unwrap();
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].possible_case_indices, [0, 1]);
        assert_eq!(groups[0].guaranteed_case_indices, [1]);
    }

    #[test]
    fn checked_population_product_alternatives_and_unknown_never_double_members() {
        let greedy = input(2, 3, CostProductOutput::GreedyToken, false, true);
        let full = input(2, 3, CostProductOutput::FullLogits, false, true);
        let alternatives = classify_alternatives(&[greedy.clone(), full], POLICY, true).unwrap();
        assert!(matches!(&alternatives, CasePopulation::Alternatives(keys) if keys.len() == 2));
        let partial = classify_alternatives(std::slice::from_ref(&greedy), POLICY, false).unwrap();
        assert!(
            matches!(&partial, CasePopulation::Unknown { known_alternatives } if known_alternatives.len() == 1)
        );
        let absent = classify_alternatives(&[], POLICY, true).unwrap();
        assert!(
            matches!(&absent, CasePopulation::Unknown { known_alternatives } if known_alternatives.is_empty())
        );
        let groups = member_groups(&[
            CaseOpportunity {
                population: alternatives,
                minimum_fresh_members: 1,
            },
            CaseOpportunity {
                population: partial,
                minimum_fresh_members: 1,
            },
            CaseOpportunity {
                population: absent,
                minimum_fresh_members: 1,
            },
        ])
        .unwrap();
        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0].possible_case_indices, [0, 1]);
        assert_eq!(groups[1].possible_case_indices, [0]);
        assert!(groups
            .iter()
            .all(|group| group.guaranteed_case_indices.is_empty()));
    }

    #[test]
    fn checked_population_only_checked_unsupported_inputs_fall_back_to_exact() {
        let first_decode = input(1, 0, CostProductOutput::GreedyToken, false, true);
        let heterogeneous = input(2, 3, CostProductOutput::GreedyToken, true, true);
        for original in [first_decode, heterogeneous] {
            assert_eq!(
                original.numerical_family_key(),
                Err(StructuredUnknownV2::UnsupportedScope)
            );
            let owner = original.owner().clone();
            assert_eq!(
                classify_alternatives(&[original], POLICY, true).unwrap(),
                CasePopulation::Unique(CheckedPopulationKey::ExactOwner(owner))
            );
        }
        let a = input(1, 3, CostProductOutput::GreedyToken, false, true);
        let b = input(2, 3, CostProductOutput::GreedyToken, false, true);
        assert!(
            matches!(classify_alternatives(&[a, b], StructuredPopulationPolicyV1::ExactOwnerV1, true).unwrap(), CasePopulation::Alternatives(keys) if keys.len() == 2 && keys.iter().all(|key| matches!(key, CheckedPopulationKey::ExactOwner(_))))
        );
        let unbound = input(2, 3, CostProductOutput::GreedyToken, false, false);
        for policy in [POLICY, StructuredPopulationPolicyV1::ExactOwnerV1] {
            for complete in [false, true] {
                assert_eq!(
                    classify_alternatives(std::slice::from_ref(&unbound), policy, complete),
                    Err(StructuredUnknownV2::WrongDomain)
                );
            }
        }
    }

    #[test]
    fn checked_population_member_floor_cannot_multiply_one_original_cohort() {
        let input = input(2, 3, CostProductOutput::GreedyToken, false, true);
        let population = classify_alternatives(&[input], POLICY, true).unwrap();
        assert_eq!(
            member_groups(&[CaseOpportunity {
                population,
                minimum_fresh_members: 2
            }]),
            Err(StructuredUnknownV2::InvalidInput)
        );
    }
}
