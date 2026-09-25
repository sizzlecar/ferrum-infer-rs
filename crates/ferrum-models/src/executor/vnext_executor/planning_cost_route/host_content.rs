//! Bind pending-host uncertainty at the real VNext product producer. Its core
//! route depends on numeric work, the global product, masks and readback; per-row
//! host branches never reach a provider once those product inputs are fixed.
use super::*;

impl<R: DeviceRuntime> VNextModelExecutor<R> {
    pub(in crate::executor::vnext_executor) fn project_future_cost_route_with_host_content(
        &self,
        view: &ExecutionCostRouteView,
        state: &ExecutionCostRouteState,
        query: &FutureWaveCostQuery<'_>,
        host: &FutureHostPendingQueryV2<'_>,
        budget: &mut dyn ResourcePlanningBudget,
    ) -> ExecutionCostRouteAvailability<ExecutionCostRouteForecastV2> {
        let result = (|| {
            if !view.structured_capture_enabled()
                || host.eligible_rows.len() > query.rows.len()
                || query.rows.len() > MAX_COST_ROWS
            {
                return Err(U::OutputBranch);
            }
            // Exactly one replay advances private state. No real Step, upload,
            // allocation or submission is performed by this producer.
            let projection = self.project_future_cost_route_inner(view, state, query, budget)?;
            let selected = projection
                .statistical_evidence
                .as_ref()
                .ok_or(U::OutputBranch)?;
            let recipe = selected
                .structured_capture()
                .ok_or(U::OutputBranch)?
                .map_err(|_| U::OutputBranch)?;
            recipe.algorithm_work().map_err(|_| U::ProviderRoute)?;
            let mode = match recipe.device().product() {
                StructuredCostProductV1::FullLogits => VNextProductOutputMode::FullLogits,
                StructuredCostProductV1::GreedyToken => VNextProductOutputMode::GreedyToken,
            };
            let mut positions = Vec::new();
            positions
                .try_reserve_exact(host.eligible_rows.len())
                .map_err(|_| U::Capacity)?;
            for eligible in host.eligible_rows {
                if !budget.has_budget() {
                    return Err(U::BudgetExhausted);
                }
                let position = eligible.physical_position;
                if positions
                    .last()
                    .is_some_and(|previous| *previous >= position)
                    || mode != VNextProductOutputMode::FullLogits
                {
                    return Err(U::OutputBranch);
                }
                let row = query.rows.get(position as usize).ok_or(U::InvalidInput)?;
                let FutureCostOutput::Decode { policy: actual } = row.output else {
                    return Err(U::OutputBranch);
                };
                require_same_pending_product_inputs(
                    actual,
                    eligible.clean_policy,
                    mode,
                    self.io.output_elements,
                )?;
                positions.push(position);
            }
            if !budget.has_budget() {
                return Err(U::BudgetExhausted);
            }
            // The constructor also checks actual plain-text policies, physical
            // positions, generated history, the anchor's pending/full relation
            // and global product stability: a fixed full peer admits any subset;
            // otherwise FullLogits requires a nonempty subset. A Greedy branch
            // admits only the conditional empty subset. This is the same OR
            // rule used by product_output_mode_for_roles above this core route.
            let pending =
                HostPendingSetV2::new(&projection.shape, recipe, &positions, host.constraint)
                    .map_err(|_| U::OutputBranch)?;
            if !budget.has_budget() {
                return Err(U::BudgetExhausted);
            }
            Ok(ExecutionCostRouteForecastV2 {
                projection,
                host_content: HostContentForecastV2::Unresolved(pending),
            })
        })();
        match result {
            Ok(projection) => ExecutionCostRouteAvailability::Known(projection),
            Err(reason) => ExecutionCostRouteAvailability::Unknown(reason),
        }
    }
}

fn require_same_pending_product_inputs(
    actual: &LogitsReturnPolicy,
    clean: &LogitsReturnPolicy,
    mode: VNextProductOutputMode,
    vocabulary_size: usize,
) -> std::result::Result<(), U> {
    if mode != VNextProductOutputMode::FullLogits
        || !matches!(
            clean,
            LogitsReturnPolicy::GreedyArgmax {
                repetition_penalty: None,
                ..
            }
        )
    {
        return Err(U::OutputBranch);
    }
    let expected = ProductTokenMaskContent::AllValid {
        vocabulary_size: u64::try_from(vocabulary_size).map_err(|_| U::Capacity)?,
    };
    for policy in [actual, clean, &LogitsReturnPolicy::FullLogits] {
        let role = VNextParticipantOutputRole::Decode(policy.clone());
        if masks::requested(&role, mode, vocabulary_size)? != expected {
            return Err(U::OutputBranch);
        }
        let repetition = product_repetition_input(Some(policy), mode);
        if !repetition.token_ids.is_empty() || repetition.penalty != 1.0 {
            return Err(U::OutputBranch);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrum_interfaces::model_executor::{GreedyRepetitionPenalty, TokenSelectionMask};

    #[test]
    fn pending_product_inputs_reject_changed_policy_and_greedy_product() {
        let clean = LogitsReturnPolicy::GreedyArgmax {
            token_mask: Some(TokenSelectionMask::new(vec![1, 0, 1])),
            repetition_penalty: None,
        };
        for actual in [&clean, &LogitsReturnPolicy::FullLogits] {
            require_same_pending_product_inputs(
                actual,
                &clean,
                VNextProductOutputMode::FullLogits,
                3,
            )
            .unwrap();
        }
        assert!(require_same_pending_product_inputs(
            &clean,
            &clean,
            VNextProductOutputMode::GreedyToken,
            3,
        )
        .is_err());
        let repetition = LogitsReturnPolicy::GreedyArgmax {
            token_mask: None,
            repetition_penalty: Some(GreedyRepetitionPenalty::new(1.2, vec![1])),
        };
        for invalid in [&repetition, &LogitsReturnPolicy::FullLogits] {
            assert!(require_same_pending_product_inputs(
                &LogitsReturnPolicy::FullLogits,
                invalid,
                VNextProductOutputMode::FullLogits,
                3,
            )
            .is_err());
        }
    }

    #[test]
    fn pending_product_subset_modes_follow_the_actual_whole_product_selector() {
        let clean = LogitsReturnPolicy::GreedyArgmax {
            token_mask: Some(TokenSelectionMask::new(vec![1, 0, 1])),
            repetition_penalty: None,
        };
        for subset in 0..8 {
            let rows: Vec<_> = (0..3)
                .map(|position| {
                    VNextParticipantOutputRole::Decode(if subset & (1 << position) == 0 {
                        clean.clone()
                    } else {
                        LogitsReturnPolicy::FullLogits
                    })
                })
                .collect();
            let mode = product_output_mode_for_roles(VNextExecutionWaveKind::Decode, &rows);
            assert_eq!(
                mode,
                if subset == 0 {
                    VNextProductOutputMode::GreedyToken
                } else {
                    VNextProductOutputMode::FullLogits
                }
            );
            for fixed in [
                VNextParticipantOutputRole::FinalPrefill,
                VNextParticipantOutputRole::Decode(LogitsReturnPolicy::FullLogits),
            ] {
                let kind = if matches!(fixed, VNextParticipantOutputRole::FinalPrefill) {
                    VNextExecutionWaveKind::Mixed
                } else {
                    VNextExecutionWaveKind::Decode
                };
                assert_eq!(
                    product_output_mode_for_roles(kind, rows.iter().chain([&fixed])),
                    VNextProductOutputMode::FullLogits
                );
            }
        }
    }
}
