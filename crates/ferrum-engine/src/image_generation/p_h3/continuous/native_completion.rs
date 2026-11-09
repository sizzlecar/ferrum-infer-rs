//! A decoder can be retried after owned denoising progress only when its
//! pre-submission admission failed cleanly. Submitted work is never classified
//! by diagnostic text and every cleanup failure keeps its original owner.
use super::*;
use crate::image_generation::ImageStage;
use ferrum_execution::component::ComponentInitializationError;

pub(in crate::image_generation) fn clean_decoder_admission<R: DeviceRuntime>(
    failure: ImageGenerationFailure<R>,
) -> std::result::Result<(), ImageGenerationFailure<R>> {
    match failure {
        ImageGenerationFailure::Pipeline {
            source,
            stages,
            weights,
            host_scopes,
            host_backing,
        } if stages.is_empty() => match clean_decoder_admission(*source) {
            Ok(()) => {
                drop((weights, host_scopes, host_backing));
                Ok(())
            }
            Err(source) => Err(ImageGenerationFailure::Pipeline {
                source: Box::new(source),
                stages,
                weights,
                host_scopes,
                host_backing,
            }),
        },
        ImageGenerationFailure::AllocationRejected {
            stage: ImageStage::Decoder,
            ..
        }
        | ImageGenerationFailure::CapacityRejected {
            stage: ImageStage::Decoder,
            ..
        } => Ok(()),
        ImageGenerationFailure::InitializationCapacityRejected { plan, .. }
            if plan.stage == ImageStage::Decoder =>
        {
            Ok(())
        }
        failure => {
            if !matches!(&failure, ImageGenerationFailure::Initialization { plan, .. }
                if plan.stage == ImageStage::Decoder)
            {
                return Err(failure);
            }
            let failure = match failure.recover_allocation() {
                Ok((ImageStage::Decoder, _)) => return Ok(()),
                Ok(_) => unreachable!("validated decoder initialization plan"),
                Err(failure) => failure,
            };
            match failure {
                ImageGenerationFailure::Initialization {
                    source:
                        ComponentInitializationError::Contract(VNextError::DeviceCapacityUnavailable(_)),
                    ..
                } => Ok(()),
                ImageGenerationFailure::Initialization {
                    plan,
                    source:
                        ComponentInitializationError::Pool {
                            source: VNextError::DeviceCapacityUnavailable(_),
                            transaction,
                            ..
                        },
                } => match transaction.release() {
                    Ok(released) => {
                        drop(released);
                        Ok(())
                    }
                    Err(source) => {
                        Err(ImageGenerationFailure::InitializationRelease { plan, source })
                    }
                },
                failure => Err(failure),
            }
        }
    }
}
