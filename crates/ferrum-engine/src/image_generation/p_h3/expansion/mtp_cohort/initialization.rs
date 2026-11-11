//! Metadata preflight and one original union installation. Every constructor
//! uses the admitted phase account; a failed transition retains its original
//! cleanup observer together with all previously acquired native owners.
use super::super::mtp_assets::{MtpExpansionAssets, MtpWaveHostRequirements};
use super::*;
use crate::image_generation::host_budget::ExecutionHostBudget;
use ferrum_execution::{executor_lifecycle::ExecutorShutdownPhase, ExecutorBuildError};
use ferrum_models::vnext::qwen35::mtp::{self as model, Qwen35MtpRole as Role};
#[cfg(test)]
use ferrum_runtime::HostMemoryClass;
use ferrum_runtime::{HostBackingPreparation, HostMemoryScope, ProgramValueId};

pub(crate) struct MtpHostRequirements {
    budget: ExecutionHostBudget,
    startup_device_bytes: u64,
    backing: HostBackingPreparation,
}

impl MtpHostRequirements {
    pub(crate) fn budget(&self) -> ExecutionHostBudget {
        self.budget
    }
    pub(crate) fn startup_device_bytes(&self) -> u64 {
        self.startup_device_bytes
    }
    pub(crate) fn materializer_additional_bytes(&self) -> usize {
        self.backing.additional_bytes()
    }
    #[cfg(test)]
    pub(crate) fn claims(&self) -> [(HostMemoryClass, usize); 4] {
        self.budget.claims()
    }
}

/// An explicit MTP preparation step, not a change to server request capacity.
/// The main verifier can consume two rows per participant in one real wave.
#[cfg(test)]
pub(crate) fn prepare_engine(prepared: &mut PreparedExpansionCohort) -> Result<()> {
    let verify_tokens = verify_tokens(prepared.requests.len())?;
    prepared.engine.batching.max_num_batched_tokens = prepared
        .engine
        .batching
        .max_num_batched_tokens
        .max(verify_tokens);
    validate_geometry(&prepared.requests, &prepared.engine)
}

/// Physical preflight over the actual three role plans. Request tables and
/// histories survive both consumers; main and draft execute serially, while
/// their independent state/pool residency can coexist. This is not admission.
pub(crate) fn host_requirements<R: DeviceRuntime>(
    native: &NativeTextExpansion,
    assets: &MtpExpansionAssets,
    composition: &RuntimeComposition<R>,
    requests: &[PreparedTextExpansion],
    engine: &EngineConfig,
    host: &HostMemoryAccount,
) -> Result<MtpHostRequirements> {
    validate_geometry(requests, engine)?;
    let union_source = assets.source(Role::InstallUnion, host)?;
    let main_source = assets.source(Role::MainVerify, host)?;
    let draft_source = assets.source(Role::OneDraft, host)?;
    let union = union_source.compile(composition, engine)?;
    let main = main_source.compile(composition, engine)?;
    let draft = draft_source.compile(composition, engine)?;
    // The machine batches one kind at a time. Do not invent a mixed wave by
    // summing max(chunk, 2) for requests with different chunk sizes.
    let prefill_rows = requests
        .iter()
        .map(|r| r.prefill_chunk_tokens)
        .collect::<Vec<_>>();
    let prefill = main.wave_host_requirements(&prefill_rows)?;
    let verify = main.wave_host_requirements(&vec![2; requests.len()])?;
    let draft_wave = draft.wave_host_requirements(&vec![1; requests.len()])?;
    let wave = MtpWaveHostRequirements {
        input_bytes: prefill
            .input_bytes
            .max(verify.input_bytes)
            .max(draft_wave.input_bytes),
        output_bytes: prefill
            .output_bytes
            .max(verify.output_bytes)
            .max(draft_wave.output_bytes),
    };
    let hidden_id = ProgramValueId::new(model::MAIN_HIDDEN_TAIL).map_err(model_error)?;
    let hidden = main
        .compilation()
        .value_tensors()
        .get(&hidden_id)
        .ok_or_else(|| FerrumError::model("MTP main accepted-hidden shape is missing"))?;
    let hidden_bytes = usize::try_from(hidden.minimum_storage_bytes().map_err(model_error)?)
        .map_err(model_error)?;
    let accepted = host.backing_bytes(hidden_bytes).map_err(model_error)?;
    let mut retained_input = 0;
    let mut retained_output = 0;
    for request in requests {
        retained_input = add(
            retained_input,
            assets.main_input_backing_bytes(native, request, host)?,
        )?;
        retained_output = add(
            retained_output,
            execute::history_backing_bytes(host, request.generation.max_new_tokens)
                .map_err(model_error)?,
        )?;
        // The accepted owner retains both original hidden rows, not just the
        // selected row. A successor wave can be read back before it replaces it.
        retained_output = add(retained_output, accepted)?;
    }
    let budget = union.phase_host_requirements(retained_input, retained_output, &[wave])?;
    let context = u64::try_from(
        engine
            .runtime
            .max_model_len
            .ok_or_else(|| FerrumError::config("MTP context ceiling is missing"))?,
    )
    .map_err(model_error)?;
    let count = u32::try_from(requests.len()).map_err(model_error)?;
    let prefill_tokens = prefill_rows.iter().try_fold(0usize, |n, &v| add(n, v))?;
    let main_memory = main.plan().payload().memory();
    let draft_memory = draft.plan().payload().memory();
    let union_memory = union.plan().payload().memory();
    let main_peak = main_memory
        .startup_peak_bytes(
            context,
            count,
            u64::try_from(prefill_tokens.max(verify_tokens(requests.len())?))
                .map_err(model_error)?,
        )
        .map_err(model_error)?;
    let draft_peak = draft_memory
        .startup_peak_bytes(context, count, u64::from(count))
        .map_err(model_error)?;
    let shared_static = union_memory.static_bytes();
    if main_memory.static_bytes() > shared_static || draft_memory.static_bytes() > shared_static {
        return Err(FerrumError::model(
            "MTP consumer static plan exceeds its union installation",
        ));
    }
    let consumers = shared_static
        .checked_add(
            main_peak
                .checked_sub(main_memory.static_bytes())
                .ok_or_else(|| FerrumError::model("MTP main startup extent is invalid"))?,
        )
        .and_then(|n| n.checked_add(draft_peak.checked_sub(draft_memory.static_bytes())?))
        .ok_or_else(|| FerrumError::config("MTP simultaneous device extent overflows"))?;
    // The installation-only plan is closed before attaching either consumer.
    // Its minimum pools still exist during construction, despite no execution.
    let installer = union_memory
        .dynamic_pools()
        .iter()
        .try_fold(shared_static, |n, pool| {
            n.checked_add(pool.provisioning().minimum_resident_bytes())
                .ok_or_else(|| FerrumError::config("MTP installer device extent overflows"))
        })?;
    let startup_device_bytes = consumers.max(installer);
    Ok(MtpHostRequirements {
        budget,
        startup_device_bytes,
        backing: union.into_host_backing(),
    })
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Stage {
    Inputs,
    Union,
    CloseInstaller,
    Main,
    Draft,
    Ready,
    Transferred,
    Closed,
}

pub(crate) enum InitializationPoll<R: DeviceRuntime> {
    Pending { made_progress: bool },
    Ready(MtpExpansionCohort<R>),
}

/// The product wrapper retains this state on every error until try_close proves
/// retirement. No successful partial owner is entrusted to an error string or
/// dropped merely because a later constructor failed.
#[must_use = "retain MTP initialization until Ready or checked close"]
pub(crate) struct MtpInitialization<R: DeviceRuntime> {
    native: Arc<NativeTextExpansion>,
    assets: Arc<MtpExpansionAssets>,
    prepared: Option<PreparedExpansionCohort>,
    phase: Option<Arc<HostMemoryScope>>,
    host: HostMemoryAccount,
    inputs: Vec<Arc<CausalRequestInputs>>,
    owners: cleanup::PartialProgramOwners<R>,
    failure: Option<ExecutorBuildError>,
    stage: Stage,
    closing: bool,
}

impl<R: DeviceRuntime> MtpInitialization<R> {
    /// Taking the grant has no native side effects. All subsequent source and
    /// compilation objects bind phase.account(), never a nominal parent grant.
    pub(crate) fn new(
        native: Arc<NativeTextExpansion>,
        assets: Arc<MtpExpansionAssets>,
        prepared: PreparedExpansionCohort,
        phase: HostMemoryScope,
    ) -> Self {
        let phase = Arc::new(phase);
        Self {
            native,
            assets,
            prepared: Some(prepared),
            host: phase.account(),
            phase: Some(phase),
            inputs: Vec::new(),
            owners: cleanup::PartialProgramOwners::new(),
            failure: None,
            stage: Stage::Inputs,
            closing: false,
        }
    }

    #[cfg(test)]
    pub(crate) fn failure(&self) -> Option<&ExecutorBuildError> {
        self.failure.as_ref()
    }

    pub(crate) fn poll(
        &mut self,
        composition: &RuntimeComposition<R>,
    ) -> Result<InitializationPoll<R>> {
        if let Some(failure) = &self.failure {
            return Err(failure.cause().clone());
        }
        if self.closing || matches!(self.stage, Stage::Transferred | Stage::Closed) {
            return Err(FerrumError::internal(
                "MTP initialization cannot resume after transfer/close",
            ));
        }
        match self.advance(composition) {
            Ok(progress) => Ok(progress),
            Err(failure) => {
                let error = failure.cause().clone();
                self.failure = Some(failure);
                Err(error)
            }
        }
    }

    fn advance(
        &mut self,
        composition: &RuntimeComposition<R>,
    ) -> ExecutorBuildResult<InitializationPoll<R>> {
        let prepared = self
            .prepared
            .as_ref()
            .ok_or_else(|| FerrumError::internal("MTP initialization lost preparation"))?;
        let phase = self
            .phase
            .as_ref()
            .ok_or_else(|| FerrumError::internal("MTP initialization lost phase custody"))?;
        match self.stage {
            Stage::Inputs => {
                validate_geometry(&prepared.requests, &prepared.engine)?;
                for request in &prepared.requests {
                    self.inputs
                        .push(self.assets.main_inputs(&self.native, request, &self.host)?);
                }
                self.stage = Stage::Union;
            }
            Stage::Union => {
                let source = self.assets.source(Role::InstallUnion, &self.host)?;
                let compiled = source.compile(composition, &prepared.engine)?;
                let (installer, weights) = compiled.initialize_union_retained(Arc::clone(phase))?;
                self.owners.installed(installer, weights);
                self.stage = Stage::CloseInstaller;
            }
            Stage::CloseInstaller => {
                if !self.owners.close_installer()? {
                    // No wave or admitted request can exist on this role. A
                    // retained reference has no progress notification here;
                    // latch a failure while keeping its original close owner.
                    return Err(FerrumError::backend(
                        "MTP installation-only plan remains referenced after checked close",
                    )
                    .into());
                }
                self.stage = Stage::Main;
            }
            Stage::Main | Stage::Draft => {
                let role = if self.stage == Stage::Main {
                    Role::MainVerify
                } else {
                    Role::OneDraft
                };
                let source = self.assets.source(role, &self.host)?;
                let compiled = source.compile(composition, &prepared.engine)?;
                let program =
                    compiled.attach_retained(self.owners.weights()?, Arc::clone(phase))?;
                if self.stage == Stage::Main {
                    self.owners.attached_main(program);
                    self.stage = Stage::Draft;
                } else {
                    self.owners.attached_draft(program);
                    self.stage = Stage::Ready;
                }
            }
            Stage::Ready => {
                let (main, draft, weights) = self.owners.take_ready()?;
                let machine = MtpExpansionCohort::from_programs(
                    Arc::clone(&self.native),
                    main,
                    draft,
                    weights,
                    self.prepared.take().expect("checked preparation"),
                    std::mem::take(&mut self.inputs),
                );
                self.phase.take(); // Original native custody retains the same Arc grant.
                self.stage = Stage::Transferred;
                return Ok(InitializationPoll::Ready(machine));
            }
            Stage::Transferred | Stage::Closed => unreachable!("poll rejects terminal state"),
        }
        Ok(InitializationPoll::Pending {
            made_progress: true,
        })
    }

    /// Retires this initializer's owners only. After Ready the caller owns the
    /// running machine and must separately close it; transfer is not retirement.
    pub(crate) fn try_close(&mut self) -> Result<bool> {
        self.closing = true;
        if self.stage == Stage::Transferred {
            return Ok(true);
        }
        if let Some(observer) = self.failure.as_ref().and_then(ExecutorBuildError::cleanup) {
            match observer.snapshot().phase() {
                ExecutorShutdownPhase::Draining => return Ok(false),
                ExecutorShutdownPhase::Retained => {
                    observer.retry();
                    return Ok(false);
                }
                ExecutorShutdownPhase::Retired => {}
            }
        }
        if !self.owners.poll_close()? {
            return Ok(false);
        }
        self.inputs.clear();
        self.prepared.take();
        self.phase.take();
        self.stage = Stage::Closed;
        Ok(true)
    }
}

fn verify_tokens(count: usize) -> Result<usize> {
    count
        .checked_mul(2)
        .filter(|&n| n > 0)
        .ok_or_else(|| FerrumError::config("MTP verify batch extent is empty or overflows"))
}

fn validate_geometry(requests: &[PreparedTextExpansion], engine: &EngineConfig) -> Result<()> {
    if engine.batching.max_num_batched_tokens < verify_tokens(requests.len())?
        || engine.scheduler.max_running_requests < requests.len()
        || requests.iter().any(|r| r.prefill_chunk_tokens == 0)
    {
        return Err(FerrumError::config(
            "MTP preparation does not admit its declared two-row verifier",
        ));
    }
    let prefill = requests
        .iter()
        .try_fold(0, |n, r| add(n, r.prefill_chunk_tokens))?;
    if prefill > engine.batching.max_num_batched_tokens {
        return Err(FerrumError::config(
            "MTP prefill exceeds its declared token capacity",
        ));
    }
    Ok(())
}

fn add(a: usize, b: usize) -> Result<usize> {
    a.checked_add(b)
        .ok_or_else(|| FerrumError::config("MTP Host extent overflows"))
}
