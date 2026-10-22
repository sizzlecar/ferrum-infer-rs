//! Decoder-only microbatches. Views share the admitted denormalized input;
//! each execution produces its own ordinary, accounted Output allocation.
use super::*;

impl<R: DeviceRuntime> QwenImage21Pipeline<R> {
    pub(super) fn decode_batch(
        &self,
        stage: PreparedStage<R>,
        decoder: &QwenImage21VaeDecoderConfig,
        input: ManagedHostTensor,
        request_batch: usize,
        job: u64,
        stages: &mut Vec<ImageStageReceipt>,
    ) -> Result<Vec<TensorExecutionOutput>, ImageGenerationFailure<R>> {
        let preparation = (|| -> anyhow::Result<_> {
            let expected = decoder.input_tensor()?;
            ensure!(
                expected.dimensions()[0]
                    == u64::try_from(self.config.decoder_batch_policy.batch_size(request_batch))?,
                "decoder plan differs from the selected batch policy"
            );
            Ok((
                decoder_inputs(&self.host, input, &expected, request_batch)?,
                decoder.output_tensor()?,
            ))
        })();
        let (inputs, expected_output) =
            preparation.map_err(|source| ImageGenerationFailure::Processing {
                source,
                active: None,
            })?;
        let mut vae = stage.start(
            self,
            job,
            ComponentWeightSource::Stored(self.assets.sources.vae.as_ref()),
        )?;
        let mut images = Vec::with_capacity(request_batch);
        for input in inputs {
            let (active, outputs) = vae.execute(job, vec![input])?;
            vae = active;
            let validation = (|| -> anyhow::Result<()> {
                let rgba = only_output(outputs, QwenImage21VaeDecoderFamily::OUTPUT_ID)?;
                ensure!(
                    rgba.tensor() == &expected_output,
                    "actual RGBA differs from the decoder's compiled batch shape"
                );
                QwenImage21RgbaOutput::new(&self.host, rgba.tensor(), rgba.bytes())?;
                for index in 0..usize::try_from(expected_output.dimensions()[0])? {
                    images.push(rgba.batch_item(index)?);
                }
                Ok(())
            })();
            if let Err(source) = validation {
                return Err(vae.processing(source));
            }
        }
        // A close error retains the original session/root; no later stage or
        // successful image response is permitted until that close succeeds.
        stages.push(vae.close()?);
        Ok(images)
    }
}

/// Validate complete NCHW geometry and parent ownership before initialization.
/// The Vec contains only bounded typed views, never copied tensor payloads.
pub(super) fn decoder_inputs(
    host: &HostMemoryAccount,
    input: ManagedHostTensor,
    expected: &ResolvedTensorSpec,
    request_batch: usize,
) -> anyhow::Result<Vec<ManagedHostTensor>> {
    ensure!(
        (1..=STRUCTURED_ATTENTION_BF16_MAX_BATCH).contains(&request_batch),
        "decoder request batch exceeds the supported layout bound"
    );
    ensure!(
        host.owns(input.bytes())
            && input.value_id().as_str() == QwenImage21VaeDecoderFamily::INPUT_ID
            && input.tensor().element_type() == ElementType::Bf16
            && expected.element_type() == ElementType::Bf16
            && input.tensor().layout() == &ResolvedTensorLayout::Contiguous
            && expected.layout() == &ResolvedTensorLayout::Contiguous,
        "decoder input requires the original Host parent and contiguous BF16 NCHW"
    );
    let actual = input.tensor().dimensions();
    let planned = expected.dimensions();
    ensure!(
        actual.len() == 4
            && planned.len() == 4
            && actual[0] == u64::try_from(request_batch)?
            && actual[1..] == planned[1..]
            && (planned[0] == 1 || planned[0] == actual[0]),
        "decoder input and compiled microbatch dimensions differ"
    );
    let stride = usize::try_from(expected.minimum_storage_bytes()?)?;
    let executions = request_batch / usize::try_from(planned[0])?;
    ensure!(
        stride.checked_mul(executions) == Some(input.bytes().len()),
        "decoder batch byte extent differs from its exact input backing"
    );
    (0..executions)
        .map(|index| {
            let start = index
                .checked_mul(stride)
                .context("decoder slice offset overflows")?;
            let end = start
                .checked_add(stride)
                .context("decoder slice end overflows")?;
            Ok(ManagedHostTensor::new(
                input.value_id().clone(),
                expected.clone(),
                input.bytes().slice(start..end)?,
            )?)
        })
        .collect()
}
