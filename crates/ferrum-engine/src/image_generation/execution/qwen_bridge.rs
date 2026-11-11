//! Qwen metadata/solver façade delegates actual stage ownership to the shared
//! execution owner. No second actor, allocator, or resident cache is created.
use super::*;
impl<R: DeviceRuntime> QwenImage21Pipeline<R> {
    pub fn residency_policy(&self) -> ImageResidencyPolicy {
        self.execution.residency_policy()
    }
    #[cfg(test)]
    pub(in crate::image_generation) fn validate_resident_capacity(
        &self,
        stages: &[&PreparedStage<R>],
    ) -> anyhow::Result<()> {
        self.execution.validate_capacity_with_prefix(stages, None)
    }
    pub(in crate::image_generation) fn validate_capacity_with_prefix(
        &self,
        stages: &[&PreparedStage<R>],
        prefix: Option<(&PreparedStage<R>, &PreparedStage<R>)>,
    ) -> anyhow::Result<()> {
        self.execution.validate_capacity_with_prefix(stages, prefix)
    }
    #[cfg(test)]
    pub(in crate::image_generation) fn capacity_with_prefix(
        &self,
        stages: &[&PreparedStage<R>],
        prefix: Option<(&PreparedStage<R>, &PreparedStage<R>)>,
    ) -> anyhow::Result<ImageDeviceCapacityReport> {
        self.execution.capacity_with_prefix(stages, prefix)
    }
    pub(in crate::image_generation) fn resident_preference_with_prefix(
        &self,
        stages: &[&PreparedStage<R>],
        prefix: Option<(&PreparedStage<R>, &PreparedStage<R>)>,
    ) -> anyhow::Result<ImageDeviceCapacityReport> {
        self.execution
            .resident_preference_with_prefix(stages, prefix)
    }
    pub(in crate::image_generation) fn start_prefix_prefill(
        &mut self,
        prepared: PreparedStage<R>,
        job: u64,
        receipts: &mut Vec<ImageStageReceipt>,
    ) -> Result<(ActiveImageStage<R>, SharedComponentWeights<R>, bool), ImageGenerationFailure<R>>
    {
        self.execution.start_prefix_prefill(prepared, job, receipts)
    }
    pub(in crate::image_generation) fn start_stage(
        &mut self,
        prepared: PreparedStage<R>,
        job: u64,
        receipts: &mut Vec<ImageStageReceipt>,
    ) -> Result<ActiveImageStage<R>, ImageGenerationFailure<R>> {
        self.execution.start_stage(prepared, job, receipts)
    }
    pub(in crate::image_generation) fn finish_stage(
        &mut self,
        active: ActiveImageStage<R>,
        receipts: &mut Vec<ImageStageReceipt>,
    ) -> Result<(), ImageGenerationFailure<R>> {
        self.execution.finish_stage(active, receipts)
    }
    pub fn shutdown(&mut self) -> Result<Vec<ImageStageReceipt>, ImageGenerationError<R>> {
        self.execution.shutdown()
    }
    pub(in crate::image_generation) fn adaptive_capacity_ceiling(&self) -> anyhow::Result<u64> {
        self.execution.adaptive_capacity_ceiling()
    }
    pub(in crate::image_generation) fn prepare_family<P: ModelFamilyProvider>(
        &self,
        stage: ImageStage,
        role: ImageStagePlanRole,
        batch_size: usize,
        provider: P,
        config: &P::Config,
        profile: &NumericalProfileId,
    ) -> anyhow::Result<PreparedModelFamily> {
        self.execution
            .prepare_family(stage, role, batch_size, provider, config, profile)
    }
}
