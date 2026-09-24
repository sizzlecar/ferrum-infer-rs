//! Explicit observed entrypoints, preserving original authority and fallback.
use super::*;
use crate::continuous_engine::EngineInner;
use ferrum_interfaces::model_executor::*;
use ferrum_types::Result;

impl EngineInner {
    pub(in crate::continuous_engine) async fn cost_prefill(
        &self,
        input: &PlanRuntimePrefillInput,
        call: &mut Option<ObservedCostCall>,
    ) -> Result<PlanRuntimePrefillOutcome> {
        if let Some(call) = call.as_deref_mut() {
            let observed = {
                match call.context() {
                    Ok(mut context) => {
                        self.model_executor
                            .plan_runtime_prefill_with_capacity_observed(input, &mut context)
                            .await
                    }
                    Err(_) => ObservedDispatch::Unavailable,
                }
            };
            match observed {
                ObservedDispatch::Executed(result) => return result,
                ObservedDispatch::Unavailable => call.reject(CostCallRejection::Unavailable),
            }
        }
        self.model_executor
            .plan_runtime_prefill_with_capacity(input)
            .await
    }
    pub(in crate::continuous_engine) async fn cost_batch_prefill(
        &self,
        inputs: &[PlanRuntimePrefillInput],
        call: &mut Option<ObservedCostCall>,
    ) -> Result<PlanRuntimeBatchPrefillOutcome> {
        if let Some(call) = call.as_deref_mut() {
            let observed = {
                match call.context() {
                    Ok(mut context) => {
                        self.model_executor
                            .plan_runtime_batch_prefill_with_capacity_observed(inputs, &mut context)
                            .await
                    }
                    Err(_) => ObservedDispatch::Unavailable,
                }
            };
            match observed {
                ObservedDispatch::Executed(result) => return result,
                ObservedDispatch::Unavailable => call.reject(CostCallRejection::Unavailable),
            }
        }
        self.model_executor
            .plan_runtime_batch_prefill_with_capacity(inputs)
            .await
    }
    pub(in crate::continuous_engine) async fn cost_batch_decode(
        &self,
        inputs: &[PlanRuntimeDecodeInput],
        call: &mut Option<ObservedCostCall>,
    ) -> Result<PlanRuntimeBatchDecodeOutcome> {
        if let Some(call) = call.as_deref_mut() {
            let observed = {
                match call.context() {
                    Ok(mut context) => {
                        self.model_executor
                            .plan_runtime_batch_decode_with_capacity_observed(inputs, &mut context)
                            .await
                    }
                    Err(_) => ObservedDispatch::Unavailable,
                }
            };
            match observed {
                ObservedDispatch::Executed(result) => return result,
                ObservedDispatch::Unavailable => call.reject(CostCallRejection::Unavailable),
            }
        }
        self.model_executor
            .plan_runtime_batch_decode_with_capacity(inputs)
            .await
    }
    pub(in crate::continuous_engine) async fn cost_mixed(
        &self,
        prefills: &[PlanRuntimePrefillInput],
        decodes: &[PlanRuntimeDecodeInput],
        call: &mut Option<ObservedCostCall>,
    ) -> Result<PlanRuntimeMixedBatchOutcome> {
        if let Some(call) = call.as_deref_mut() {
            let observed = {
                match call.context() {
                    Ok(mut context) => {
                        self.model_executor
                            .plan_runtime_mixed_batch_with_capacity_observed(
                                prefills,
                                decodes,
                                &mut context,
                            )
                            .await
                    }
                    Err(_) => ObservedDispatch::Unavailable,
                }
            };
            match observed {
                ObservedDispatch::Executed(result) => return result,
                ObservedDispatch::Unavailable => call.reject(CostCallRejection::Unavailable),
            }
        }
        self.model_executor
            .plan_runtime_mixed_batch_with_capacity(prefills, decodes)
            .await
    }
}
