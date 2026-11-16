//! Project image receipts into shared execution-cost observations.
use super::*;
use crate::image_generation::ImageResidencyPolicy;
#[cfg(test)]
use crate::image_generation::ImageStage;
#[cfg(test)]
pub(crate) use crate::iterative_service::batch_efficiency::{BatchEfficiency, ExecutionContext};
pub use crate::iterative_service::{
    BatchExecutionFeedback as ImageBatchExecutionFeedback,
    BatchStageFeedback as ImageBatchStageFeedback,
};
impl<R: DeviceRuntime> QwenImage21Pipeline<R> {
    pub(crate) fn service_batch_execution_feedback(
        &self,
        outputs: &[ImageGenerationOutput],
    ) -> Option<ImageBatchExecutionFeedback> {
        if self.execution.config.residency != ImageResidencyPolicy::Adaptive {
            return None;
        }
        let receipt = &outputs.first()?.receipt;
        let mut feedback = ImageBatchExecutionFeedback::default();
        for stage in &receipt.stages {
            let execution = &stage.execution;
            if execution.plan.static_bytes == 0 || execution.completed_executions == 0 {
                continue;
            }
            let initialization_us = if execution.weights_reused {
                0
            } else {
                execution.initialization_wall_duration_us
            };
            if let Some(previous) = feedback
                .stages
                .iter_mut()
                .find(|s| s.stage == execution.plan.stage.into())
            {
                previous.initialization_us =
                    previous.initialization_us.saturating_add(initialization_us);
            } else {
                feedback.stages.push(ImageBatchStageFeedback {
                    stage: execution.plan.stage.into(),
                    initialization_us,
                    weights_retained: self
                        .execution
                        .resident_stages
                        .has_weights(execution.plan.stage),
                });
            }
        }
        Some(feedback)
    }
}

#[cfg(test)]
mod scheduler_tests;
