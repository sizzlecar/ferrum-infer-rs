//! Adapt only a proved zero-submission cohort; never replay a partial result.

use super::*;

impl EngineInner {
    pub(super) async fn handle_token_deferral(
        &self,
        inputs: &[&TokenBatchInput],
        reason: NoSubmissionReason,
        capacity_pressure_root_width: &mut Option<usize>,
    ) -> Result<Vec<Vec<usize>>> {
        let request_ids = inputs
            .iter()
            .map(|input| input.request_id.clone())
            .collect::<Vec<_>>();
        match reason {
            NoSubmissionReason::Rejected(error) => Err(error),
            NoSubmissionReason::UnsupportedShape => {
                let (prefills, decodes): (Vec<_>, Vec<_>) = (0..inputs.len())
                    .partition(|&index| matches!(inputs[index].work, TokenWork::Prefill(_)));
                if !prefills.is_empty() && !decodes.is_empty() {
                    Ok(vec![prefills, decodes])
                } else if inputs.len() > 1 {
                    let midpoint = inputs.len() / 2;
                    Ok(vec![
                        (0..midpoint).collect(),
                        (midpoint..inputs.len()).collect(),
                    ])
                } else {
                    Err(FerrumError::unsupported(
                        "executor rejected a single token work shape",
                    ))
                }
            }
            NoSubmissionReason::Deferred(ExecutorExecutionDeferral::RequestState(deferral)) => {
                let affected = deferral.request_ids().to_vec();
                if affected
                    .iter()
                    .any(|request| !request_ids.contains(request))
                {
                    return Err(FerrumError::internal(
                        "token readiness deferral names another cohort",
                    ));
                }
                self.defer_for_request_state_readiness(deferral).await?;
                Ok(vec![request_ids
                    .iter()
                    .enumerate()
                    .filter_map(|(index, request)| (!affected.contains(request)).then_some(index))
                    .collect()])
            }
            NoSubmissionReason::Deferred(ExecutorExecutionDeferral::Capacity(deferral)) => {
                if let Some(retry) = deferral.validated_maintenance_retry_scope(&request_ids)? {
                    let affected = retry.affected_request_ids();
                    let receipt = self
                        .scheduler
                        .defer_retry_after_execution_maintenance(retry)?;
                    if receipt.deferred_count() != affected.len() {
                        return Err(FerrumError::scheduler(
                            "token maintenance lost deferred participants",
                        ));
                    }
                    return Ok(vec![request_ids
                        .iter()
                        .enumerate()
                        .filter_map(|(index, request)| {
                            (!affected.contains(request)).then_some(index)
                        })
                        .collect()]);
                }
                if inputs.len() > 1 {
                    let (prefills, decodes): (Vec<_>, Vec<_>) = (0..inputs.len())
                        .partition(|&index| matches!(inputs[index].work, TokenWork::Prefill(_)));
                    if !prefills.is_empty() && !decodes.is_empty() {
                        return Ok(vec![prefills, decodes]);
                    }
                    if inputs
                        .iter()
                        .all(|input| matches!(input.work, TokenWork::Decode { .. }))
                    {
                        let first_capacity_split = capacity_pressure_root_width.is_none();
                        let root_width = *capacity_pressure_root_width.get_or_insert(inputs.len());
                        match deferral.stage() {
                            ExecutorExecutionCapacityStage::SequenceExtension
                                if first_capacity_split =>
                            {
                                self.scheduler
                                    .record_decode_capacity_pressure(root_width, None);
                            }
                            ExecutorExecutionCapacityStage::SequenceExtension => {}
                            ExecutorExecutionCapacityStage::StepAdmission
                            | ExecutorExecutionCapacityStage::SubmissionWave => {
                                self.scheduler
                                    .record_decode_execution_capacity_pressure(root_width);
                            }
                        }
                    }
                    let midpoint = inputs.len() / 2;
                    return Ok(vec![
                        (0..midpoint).collect(),
                        (midpoint..inputs.len()).collect(),
                    ]);
                }
                let input = inputs[0];
                let observed = deferral.observed();
                let wait = AdmissionDeferral::new(
                    DeferredAction::WaitForRelease,
                    AdmissionWakeEpochs::new(
                        observed.coordinator_id,
                        observed.release_epoch,
                        observed.capacity_epoch,
                        0,
                    ),
                    deferral.wait_condition().clone(),
                );
                let release = self.execution_capacity_release_snapshot()?;
                let action = match input.work {
                    TokenWork::Decode { .. } => self
                        .scheduler
                        .defer_decode_for_execution_capacity(&request_ids, wait, &release)?,
                    TokenWork::Prefill(chunk) => {
                        if chunk.tokens_to_process() == 1
                            && !release
                                .has_external_releaser(&input.request_id, deferral.wait_condition())
                        {
                            return Err(FerrumError::resource_exhausted(
                                "one-token prefill does not fit and has no external capacity releaser",
                            ));
                        }
                        self.scheduler.defer_prefill_for_execution_capacity(
                            &input.request_id,
                            wait,
                            &release,
                        )?
                    }
                };
                match action {
                    ExecutionCapacityAction::Deferred { count: 1 } => Ok(Vec::new()),
                    ExecutionCapacityAction::Deferred { .. } => Err(FerrumError::scheduler(
                        "token capacity deferral lost its exact participant",
                    )),
                    ExecutionCapacityAction::YieldPlanned { transaction } => {
                        // Recompute is completed by the existing close registry
                        // after the original incarnation reaches actual Ended.
                        self.execute_capacity_yield(
                            &transaction,
                            capacity_pressure_root_width.unwrap_or(inputs.len()),
                            None,
                        )
                        .await?;
                        Ok(Vec::new())
                    }
                    ExecutionCapacityAction::InvariantViolation { violation } => {
                        Err(FerrumError::internal(format!(
                            "token capacity episode {} violated {:?}",
                            violation.episode_id().get(),
                            violation.class()
                        )))
                    }
                }
            }
        }
    }
}
