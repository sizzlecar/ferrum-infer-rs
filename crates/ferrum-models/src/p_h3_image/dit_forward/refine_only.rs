//! Fresh-request text refinement on the caller's existing attached session and
//! serial actor. This component has no sample, timestep or solver mutation.
use super::*;
use anyhow::{ensure, Context, Result};
use ferrum_execution::{
    component::{ComponentInitializationReceipt, TensorComponentSession},
    TensorExecutionError, TensorSerialActor,
};
use half::bf16;
use std::fmt;

/// Borrowing the real per-request scope keeps its authorization live until the
/// complete cache set is published. A parent account is not a substitute.
pub struct PH3ImageRefinementDestination<'a> {
    pub(super) request: &'a RequestIdentity,
    pub(super) scope: &'a HostMemoryScope,
}
impl<'a> PH3ImageRefinementDestination<'a> {
    pub fn new(request: &'a RequestIdentity, scope: &'a HostMemoryScope) -> Self {
        Self { request, scope }
    }
}

#[must_use = "execution failures may retain the original retirement owner"]
pub enum PH3ImageTextRefinementError<R: DeviceRuntime> {
    Preparation(anyhow::Error),
    Execution(TensorExecutionError<R>),
}
impl<R: DeviceRuntime> fmt::Debug for PH3ImageTextRefinementError<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Preparation(e) => f.debug_tuple("Preparation").field(e).finish(),
            Self::Execution(e) => f.debug_tuple("Execution").field(e).finish(),
        }
    }
}

pub(crate) struct PH3ImageTextRefinement<'a, R: DeviceRuntime> {
    config: &'a PH3ImageDiTConfig,
    session: &'a mut TensorComponentSession<R>,
    weights: &'a SharedComponentWeights<R>,
    limits: PH3ImageDiTLimits,
    work_host: HostMemoryAccount,
}
impl<'a, R: DeviceRuntime> PH3ImageTextRefinement<'a, R> {
    /// Only an actual attachment receipt authenticates this supplied installed
    /// snapshot. A cold initialized session must not borrow unrelated evidence.
    pub fn bind(
        config: &'a PH3ImageDiTConfig,
        limits: PH3ImageDiTLimits,
        session: &'a mut TensorComponentSession<R>,
        weights: &'a SharedComponentWeights<R>,
        work_host: HostMemoryAccount,
    ) -> Result<Self> {
        ensure!(
            session.host_account().same_parent(&work_host),
            "refinement work must use the existing Host parent"
        );
        ensure!(
            config.phase == PH3ImageDiTPhase::RefineOnly,
            "fresh text consumer requires explicit RefineOnly phase"
        );
        let expected = TypedFamilyRegistration::new(PH3ImageDiTFamily::new(limits)?)
            .prepare_with_profile(
                &serde_json::to_value(config)?,
                &NumericalProfileId::new(PH3ImageDiTFamily::cuda_profile_id(config))
                    .map_err(invalid)?,
            )?;
        let plan = session.executable().execution_plan().payload();
        ensure!(
            plan.family_id() == expected.family_id()
                && plan.prepared_family_fingerprint() == expected.fingerprint()?
                && plan.program_fingerprint() == expected.program().fingerprint()?,
            "refinement session program/profile/config differs from submitted fresh request family"
        );
        let ComponentInitializationReceipt::SharedWeights { evidence } =
            session.initialization_receipt()
        else {
            anyhow::bail!("RefineOnly consumer requires an attached shared-weight session")
        };
        ensure!(
            evidence == weights.evidence()
                && evidence.weight_plan_fingerprint()
                    == plan.execution_weights().stored()?.fingerprint()?,
            "refinement session belongs to a different installed physical weight snapshot"
        );
        let ports = plan
            .tensor_ports()
            .context("RefineOnly session lacks typed ports")?;
        ensure!(
            ports.inputs().len() == 1
                && ports.inputs()[0].value_id().as_str() == PH3ImageDiTFamily::RAW_TEXT_ID
                && ports.inputs()[0].tensor().dimensions()
                    == [config.text_rows()?, config.text_dim]
                && ports.inputs()[0].tensor().element_type() == ElementType::Bf16
                && ports.outputs().len() == 1
                && ports.outputs()[0].value_id().as_str() == PH3ImageDiTFamily::REFINED_TEXT_ID
                && ports.outputs()[0].tensor().dimensions() == [config.text_rows()?, config.hidden]
                && ports.outputs()[0].tensor().element_type() == ElementType::Bf16,
            "RefineOnly ports differ from exact fresh text geometry"
        );
        Ok(Self {
            config,
            session,
            weights,
            limits,
            work_host,
        })
    }
    /// The manifest, actual submission and returned output stay inside this
    /// call; callers cannot pair another run's equal-shaped output with it.
    /// All request destination allocations are admitted before GPU submission.
    pub fn execute_on_actor(
        &mut self,
        actor: &mut TensorSerialActor<R>,
        run: RunId,
        request: RequestIdentity,
        connector: &ManagedHostTensor,
        destinations: &[PH3ImageRefinementDestination<'_>],
    ) -> std::result::Result<Vec<PH3ImageRefinedTextCache>, PH3ImageTextRefinementError<R>> {
        // A resident session may have been initialized under an earlier,
        // already closed scope. Only this invocation's explicit grant can
        // authorize new preprocessing allocations.
        let host = self.work_host.clone();
        let c = self.config;
        let prepare = || -> Result<_> {
            inputs::check(
                &host,
                connector,
                &[c.text_rows()?, c.text_dim],
                ElementType::F32,
            )?;
            ensure!(
                destinations.len() == c.requests.len(),
                "refinement destinations do not cover submitted fresh requests"
            );
            ensure!(
                destinations
                    .iter()
                    .enumerate()
                    .all(|(i, d)| destinations[..i]
                        .iter()
                        .all(|old| !std::ptr::eq(old.scope, d.scope))),
                "fresh requests require separate persistent scope guards"
            );
            let requests = destinations
                .iter()
                .map(|d| d.request.clone())
                .collect::<Vec<_>>();
            let keys = PH3ImageDiTConditioningKey::from_connector(
                c,
                connector,
                &requests,
                self.weights.evidence(),
                self.limits.max_metadata_bytes,
            )?;
            let pending = cache::PreparedRefinedTextCaches::new(
                c,
                &host,
                keys,
                destinations,
                self.limits.max_metadata_bytes,
            )?;
            let count = c
                .text_rows()?
                .checked_mul(c.text_dim)
                .context("fresh text extent overflow")?;
            let bytes = usize::try_from(
                count
                    .checked_mul(2)
                    .context("fresh text BF16 bytes overflow")?,
            )?;
            let mut allocation = host.allocate_zeroed(HostMemoryClass::InputPreprocess, bytes)?;
            for (source, target) in connector
                .bytes()
                .bytes()
                .chunks_exact(4)
                .zip(allocation.as_mut().chunks_exact_mut(2))
            {
                target.copy_from_slice(
                    &bf16::from_f32(f32::from_le_bytes(source.try_into().unwrap()))
                        .to_bits()
                        .to_le_bytes(),
                );
            }
            let input = inputs::managed(
                PH3ImageDiTFamily::RAW_TEXT_ID,
                config::tensor(vec![c.text_rows()?, c.text_dim], ElementType::Bf16),
                allocation.freeze(),
            )?;
            Ok((pending, input))
        };
        let (pending, input) = prepare().map_err(PH3ImageTextRefinementError::Preparation)?;
        let outputs = self
            .session
            .execute_on_actor(actor, run, request, vec![input])
            .map_err(PH3ImageTextRefinementError::Execution)?;
        let publish = || -> Result<_> {
            ensure!(
                outputs.len() == 1,
                "refinement returned an undeclared output"
            );
            let declared = &self
                .session
                .executable()
                .execution_plan()
                .payload()
                .tensor_ports()
                .context("RefineOnly typed outputs disappeared")?
                .outputs()[0];
            pending.publish(c, &host, declared, &outputs[0])
        };
        publish().map_err(PH3ImageTextRefinementError::Preparation)
    }
}
#[cfg(test)]
mod tests;
