use super::*;
use ferrum_models::fast_h3::audio_vae::{
    FastH3AudioLatentLayout, FastH3AudioLatentUnits, FastH3AudioWaveform,
    PreparedFastH3AudioDecoder,
};

/// Complete original decoder graph and exact original/derived weight source.
/// Two independent mono batch items represent each request's stereo channels.
pub struct FastH3AudioDecodePlan<R: DeviceRuntime> {
    source: Arc<PreparedFastH3AudioDecoder>,
    compiled: Arc<CompiledTensorComponent<R>>,
}
impl<R: DeviceRuntime> FastH3AudioDecodePlan<R> {
    pub fn compile(
        source: Arc<PreparedFastH3AudioDecoder>,
        registry: &OperationRuntimeRegistry<R>,
        catalog: &CapabilityCatalog,
        materializers: &WeightMaterializerRegistry,
        policy: &TensorComponentPolicy,
    ) -> anyhow::Result<Self> {
        let compiled = CompiledTensorComponent::compile(
            source.family().clone(),
            registry,
            catalog,
            policy,
            materializers,
            source.compile_options(),
        )?;
        Ok(Self {
            source,
            compiled: Arc::new(compiled),
        })
    }
    pub fn source(&self) -> &PreparedFastH3AudioDecoder {
        &self.source
    }
    pub fn compiled(&self) -> &Arc<CompiledTensorComponent<R>> {
        &self.compiled
    }
    /// P is the caller's existing provider/materializer/phase owner. It is
    /// returned only with the original checked component close receipt.
    pub fn initialize<P: Send + 'static>(
        self: Arc<Self>,
        runtime: Arc<R>,
        host: HostMemoryAccount,
        initialization: ComponentInitializationConfig,
        phase: P,
    ) -> Result<
        FastH3AudioDecodeSession<R, P>,
        RetainedComponentInitializationError<R, (Arc<Self>, P)>,
    > {
        let session = self.compiled.initialize_retained_with_pool_policy(
            runtime,
            host.clone(),
            ComponentWeightSource::Stored(self.source.as_ref()),
            initialization,
            (self.clone(), phase),
            ferrum_execution::component::ColdPoolInitializationPolicy::OnExecutionAdmission,
        )?;
        Ok(FastH3AudioDecodeSession {
            plan: self,
            session,
            host,
        })
    }
}

pub struct FastH3AudioDecodeSession<R: DeviceRuntime, P: Send + 'static> {
    plan: Arc<FastH3AudioDecodePlan<R>>,
    session: RetainedTensorComponentSession<R, (Arc<FastH3AudioDecodePlan<R>>, P)>,
    host: HostMemoryAccount,
}
impl<R: DeviceRuntime, P: Send + 'static> FastH3AudioDecodeSession<R, P> {
    /// Consume only final normalized target rows, [requests*2*L,C]. The full
    /// L*800 sample tail is preserved independently of the video's frame count.
    pub fn decode(
        &mut self,
        actor: &mut TensorSerialActor<R>,
        run: RunId,
        request: RequestIdentity,
        target_latents: &ManagedHostTensor,
        cancelled: impl Fn() -> bool,
    ) -> Result<FastH3AudioWaveform, FastH3DecodeError<R>> {
        if cancelled() {
            return Err(FastH3DecodeError::Cancelled);
        }
        if !self.host.owns(target_latents.bytes()) {
            return Err(anyhow::anyhow!("audio target has another Host parent").into());
        }
        let input = self.plan.source.managed_input(
            target_latents,
            FastH3AudioLatentLayout::PackedStereoTokens,
            FastH3AudioLatentUnits::Normalized,
            &self.host,
        )?;
        let output = self
            .session
            .session_mut()
            .execute_on_actor(
                actor,
                derived_run(&run, "audio", 0, 0)?,
                request,
                vec![input],
            )
            .map_err(FastH3DecodeError::Execution)?;
        if cancelled() {
            return Err(FastH3DecodeError::Cancelled);
        }
        if output.len() != 1 {
            return Err(anyhow::anyhow!("audio decoder must return its single waveform").into());
        }
        Ok(self.plan.source.interleave_stereo(&output[0], &self.host)?)
    }
    pub fn initialization_receipt(
        &self,
    ) -> &ferrum_execution::component::ComponentInitializationReceipt {
        self.session.session().initialization_receipt()
    }
    pub fn close(
        self,
    ) -> Result<
        ((Arc<FastH3AudioDecodePlan<R>>, P), PlanRuntimeCloseReceipt),
        RetainedComponentCloseError<R, (Arc<FastH3AudioDecodePlan<R>>, P)>,
    > {
        self.session.close()
    }
}
