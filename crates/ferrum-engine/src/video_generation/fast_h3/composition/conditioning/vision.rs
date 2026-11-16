use super::*;
use ferrum_models::qwen3_vl_vision::Qwen3VLVisionConfig;

impl PreparedConditioner {
    pub(super) fn vision_forward(
        &mut self,
        config: &Qwen3VLVisionConfig,
        patches: ManagedHostTensor,
        host: &HostMemoryAccount,
        actor: &mut TensorSerialActor<Device>,
        initialization: ComponentInitializationConfig,
    ) -> Result<[TensorExecutionOutput; 4]> {
        if !self
            .last_vision
            .as_ref()
            .is_some_and(|old| old.checkpoint.config() == config)
        {
            let profile = NumericalProfileId::new(Qwen3VLVisionFamily::CUDA_PROFILE_ID)
                .map_err(|e| prep("vision profile", e))?;
            let checkpoint = Arc::new(
                PreparedQwen3VLVisionCheckpoint::prepare(
                    config.clone(),
                    self.assets.encoder.clone(),
                    &profile,
                    self.limits.vision,
                )
                .map_err(|e| prep("vision source", e))?,
            );
            let compiled =
                self.compile(checkpoint.family().clone(), checkpoint.compile_options())?;
            self.last_vision = Some(VisionDomain {
                checkpoint,
                compiled,
            });
        }
        let vision = self.last_vision.as_ref().expect("prepared vision domain");
        let mut inputs = vision
            .checkpoint
            .managed_position_inputs(host)
            .map_err(|e| prep("vision positions", e))?
            .to_vec();
        inputs.push(patches);
        let run = forward_run(&initialization)?;
        let request_id = initialization.request_id.clone();
        encode_vision(
            PreparedVisionConditioning {
                checkpoint: vision.checkpoint.clone(),
                inputs,
            },
            vision.compiled.clone(),
            self.cuda.runtime.clone(),
            host,
            actor,
            initialization,
            run,
            request_id,
        )
    }
}
