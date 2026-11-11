//! Metadata-only candidate compilation followed by initialization of the exact
//! retained result. The borrow ties compilation to the actual registries used
//! for binding; callers cannot substitute a reconstructed capability list.

use super::*;

mod causal_program;
mod phase_custody;
pub use causal_program::{
    CausalAcceptedOutput, CausalCommittedWave, CausalComputedWave, CausalProgramSession,
    CausalTailTwoOutput, CausalVerifiedDraftOutput, CausalWaveFailure, CompiledCausalProgram,
};
pub(super) mod static_installation;

/// This executor either published the original completed installation or
/// attached to it without reading, materializing or uploading static weights.
/// Evidence identifies the original physical owner, not a second initialization.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ProgramSharedWeightsReceipt {
    Published {
        evidence: SharedComponentWeightsEvidence,
        source: ResolvedModelSource,
    },
    Attached {
        evidence: SharedComponentWeightsEvidence,
        source: ResolvedModelSource,
    },
}

impl ProgramSharedWeightsReceipt {
    pub fn evidence(&self) -> &SharedComponentWeightsEvidence {
        match self {
            Self::Published { evidence, .. } | Self::Attached { evidence, .. } => evidence,
        }
    }
}

/// Sealed token-program installation and the frozen weight artifact identity
/// consumed by its successful cold initialization. Core's snapshot deliberately
/// does not authenticate files; this product handle also binds those sources.
/// There is no public constructor or Deserialize path.
#[must_use = "shared program weights retain the original physical installation"]
pub struct SharedProgramWeights<R: DeviceRuntime> {
    pub(super) weights: SharedComponentWeights<R>,
    pub(super) source: ResolvedModelSource,
}

impl<R: DeviceRuntime> Clone for SharedProgramWeights<R> {
    fn clone(&self) -> Self {
        Self {
            weights: self.weights.clone(),
            source: self.source.clone(),
        }
    }
}

impl<R: DeviceRuntime> SharedProgramWeights<R> {
    pub fn evidence(&self) -> &SharedComponentWeightsEvidence {
        self.weights.evidence()
    }
    pub fn source(&self) -> &ResolvedModelSource {
        &self.source
    }
    pub(super) fn validate_for(
        &self,
        plan: &ExecutionPlan,
        runtime: &Arc<R>,
        source: &ResolvedModelSource,
    ) -> Result<()> {
        if source != &self.source {
            return Err(FerrumError::model(
                "shared program weights differ from the frozen weight artifact source",
            ));
        }
        self.weights
            .validate_for(plan, runtime)
            .map_err(|e| FerrumError::model(format!("shared static attachment: {e}")))
    }
    /// Delegate retirement to the original Core owner. A referenced result
    /// retains the sealed source binding; failed retirement preserves Core's
    /// original failure owner and retry API without authorizing another attach.
    pub fn close(
        self,
    ) -> std::result::Result<
        SharedProgramWeightsCloseOutcome<R>,
        SharedComponentWeightsCloseFailure<R>,
    > {
        match self.weights.close()? {
            SharedComponentWeightsCloseOutcome::Closed(receipt) => {
                Ok(SharedProgramWeightsCloseOutcome::Closed(receipt))
            }
            SharedComponentWeightsCloseOutcome::Referenced { weights } => {
                Ok(SharedProgramWeightsCloseOutcome::Referenced {
                    weights: Self {
                        weights,
                        source: self.source,
                    },
                })
            }
        }
    }
}

#[must_use = "referenced program weights retain the original physical claim"]
pub enum SharedProgramWeightsCloseOutcome<R: DeviceRuntime> {
    Closed(SharedComponentWeightsCloseReceipt),
    Referenced { weights: SharedProgramWeights<R> },
}

pub(super) enum StaticInstallation<'a, R: DeviceRuntime> {
    Private,
    PublishShared,
    AttachShared(&'a SharedProgramWeights<R>),
}

/// Preserve only fresh static-planning capacity failures across the model
/// boundary. Contract errors and all initialization failures remain terminal.
#[derive(Debug, thiserror::Error)]
pub enum ProgramCompilationError {
    #[error("compiled plan requires {required_bytes} bytes, exceeding usable capacity {usable_capacity_bytes}")]
    Capacity {
        required_bytes: u64,
        usable_capacity_bytes: u64,
    },
    #[error(transparent)]
    Invalid(#[from] FerrumError),
}

impl From<ProgramCompilationError> for FerrumError {
    fn from(error: ProgramCompilationError) -> Self {
        match error {
            ProgramCompilationError::Invalid(error) => error,
            capacity => FerrumError::config(capacity.to_string()),
        }
    }
}

impl From<ProgramCompilationError> for crate::ExecutorBuildError {
    fn from(error: ProgramCompilationError) -> Self {
        FerrumError::from(error).into()
    }
}

impl ProgramCompilationError {
    fn from_plan(error: VNextError) -> Self {
        match error {
            VNextError::PlanCapacityExceeded {
                required_bytes,
                usable_capacity_bytes,
            } => Self::Capacity {
                required_bytes,
                usable_capacity_bytes,
            },
            error => Self::Invalid(FerrumError::model(format!("vNext plan compile: {error}"))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn optional_layout_capacity_preserves_only_typed_fresh_plan_pressure() {
        assert!(matches!(
            ProgramCompilationError::from_plan(VNextError::PlanCapacityExceeded {
                required_bytes: 8192,
                usable_capacity_bytes: 8191,
            }),
            ProgramCompilationError::Capacity {
                required_bytes: 8192,
                usable_capacity_bytes: 8191,
            }
        ));
        assert!(matches!(
            ProgramCompilationError::from_plan(VNextError::InvalidExecutionPlan {
                reason: "static memory exceeds usable capacity".into(),
            }),
            ProgramCompilationError::Invalid(_)
        ));
        let error = FerrumError::from(ProgramCompilationError::from(FerrumError::device(
            "allocator out of memory",
        )));
        assert!(matches!(error, FerrumError::Device { .. }));
    }
}

pub struct RuntimeComposition<R: DeviceRuntime> {
    pub(super) runtime: Arc<R>,
    pub(super) registry: Arc<OperationRuntimeRegistry<R>>,
    weight_materializers: Arc<WeightMaterializerRegistry>,
    pub(super) catalog: CapabilityCatalog,
}

/// No Deserialize or public field construction: only successful static
/// compilation produces this initialization input. It borrows the exact
/// prepared source and runtime composition, including on the Auto path.
pub struct CompiledProgram<'a, R: DeviceRuntime> {
    pub(super) composition: &'a RuntimeComposition<R>,
    pub(super) prepared: PreparedExecutionInput<'a>,
    pub(super) info: ModelInfo,
    pub(super) config: ProgramExecutorConfig,
    pub(super) compilation: ProgramPlanCompilation,
    pub(super) language_io: VNextLanguageIoIds,
    pub(super) checkpoint_selection: Option<VNextCheckpointSelection>,
    pub(super) repetition_capacity: u64,
    pub(super) executor_startup: StartupPhaseTimer,
}

impl<R: DeviceRuntime> RuntimeComposition<R> {
    pub fn new(
        runtime: Arc<R>,
        registry: OperationRuntimeRegistry<R>,
        weight_materializers: WeightMaterializerRegistry,
        catalog: CapabilityCatalog,
    ) -> Self {
        Self::from_shared(
            runtime,
            Arc::new(registry),
            Arc::new(weight_materializers),
            catalog,
        )
    }

    /// Share the actual providers and materializers with an existing runtime
    /// composition. This does not share an execution lane or dynamic I/O Host
    /// authority; compilation still checks the original runtime/catalog pair.
    pub fn from_shared(
        runtime: Arc<R>,
        registry: Arc<OperationRuntimeRegistry<R>>,
        weight_materializers: Arc<WeightMaterializerRegistry>,
        catalog: CapabilityCatalog,
    ) -> Self {
        Self {
            runtime,
            registry,
            weight_materializers,
            catalog,
        }
    }

    pub fn runtime(&self) -> &R {
        self.runtime.as_ref()
    }
    /// Clone the original runtime owner for a neighboring tensor component.
    /// This neither creates a runtime nor changes its capacity authority.
    pub fn runtime_handle(&self) -> Arc<R> {
        Arc::clone(&self.runtime)
    }

    /// Compile an actual tensor component against this composition's same
    /// providers, catalog and materializers, without reconstructing a registry.
    pub fn compile_tensor_component(
        &self,
        family: ferrum_runtime::PreparedModelFamily,
        policy: &crate::component::TensorComponentPolicy,
        options: &ferrum_runtime::ProgramPlanCompileOptions,
    ) -> std::result::Result<crate::component::CompiledTensorComponent<R>, VNextError> {
        crate::component::CompiledTensorComponent::compile(
            family,
            &self.registry,
            &self.catalog,
            policy,
            &self.weight_materializers,
            options,
        )
    }
    pub fn catalog(&self) -> &CapabilityCatalog {
        &self.catalog
    }
    pub fn weight_materializers(&self) -> &WeightMaterializerRegistry {
        &self.weight_materializers
    }

    pub fn compile_model<'a>(
        &'a self,
        prepared: PreparedExecutionInput<'a>,
        info: ModelInfo,
        engine_config: &EngineConfig,
        config: ProgramExecutorConfig,
        materializer_selection: WeightMaterializerSelection,
    ) -> std::result::Result<CompiledProgram<'a, R>, ProgramCompilationError> {
        self.compile_model_inner(
            prepared,
            info,
            engine_config,
            config,
            materializer_selection,
            None,
        )
    }

    pub fn compile_model_with_causal_inputs<'a>(
        &'a self,
        prepared: PreparedExecutionInput<'a>,
        info: ModelInfo,
        engine_config: &EngineConfig,
        config: ProgramExecutorConfig,
        materializer_selection: WeightMaterializerSelection,
        inputs: &CausalInputSchema,
    ) -> std::result::Result<CompiledProgram<'a, R>, ProgramCompilationError> {
        self.compile_model_inner(
            prepared,
            info,
            engine_config,
            config,
            materializer_selection,
            Some(inputs),
        )
    }

    fn compile_model_inner<'a>(
        &'a self,
        prepared: PreparedExecutionInput<'a>,
        info: ModelInfo,
        engine_config: &EngineConfig,
        config: ProgramExecutorConfig,
        materializer_selection: WeightMaterializerSelection,
        causal_inputs: Option<&CausalInputSchema>,
    ) -> std::result::Result<CompiledProgram<'a, R>, ProgramCompilationError> {
        prepared.validate_info(&info)?;
        if self.runtime.descriptor() != self.catalog.device() {
            return Err(FerrumError::device(
                "vNext static compilation catalog differs from the actual device runtime",
            )
            .into());
        }
        let executor_startup = StartupPhaseTimer::start("executor_composition_total");
        let checkpoint_selection = VNextCheckpointSelection::from_config(
            engine_config.runtime.vnext_checkpoint_capture.as_ref(),
        )?;
        let family = prepared.family();
        let language_io = match causal_inputs {
            Some(inputs) => ProgramExecutor::<R>::resolve_language_io_ids_with_inputs(
                family.program(),
                Some(inputs),
            )?,
            None => ProgramExecutor::<R>::resolve_language_io_ids(family.program())?,
        };
        if causal_inputs.is_some()
            && (checkpoint_selection.is_some()
                || (config.device_reusable_execution_enabled
                    && !config.supports_causal_input_replay()))
        {
            return Err(FerrumError::unsupported(
                "causal conditioning requires no checkpoint capture and either eager or on-demand decode-only execution",
            )
            .into());
        }
        let input_capacity = u64::try_from(config.maximum_model_tokens)
            .map_err(|_| FerrumError::config("vNext model length exceeds u64"))?;
        let vocabulary_size = u64::try_from(info.vocab_size)
            .map_err(|_| FerrumError::config("vNext vocabulary exceeds u64"))?;
        let repetition_capacity = input_capacity.min(vocabulary_size);
        let tensor = |dimensions, element_type| ProgramTensorSpec {
            dimensions,
            element_type,
            layout: ResolvedTensorLayout::Contiguous,
        };
        let mut input_tensors = BTreeMap::from([
            (
                language_io.token_input.clone(),
                tensor(vec![input_capacity], ElementType::U32),
            ),
            (
                language_io.token_mask_input.clone(),
                tensor(vec![vocabulary_size], ElementType::U8),
            ),
            (
                language_io.repetition_token_ids_input.clone(),
                tensor(vec![repetition_capacity], ElementType::U32),
            ),
            (
                language_io.repetition_offsets_input.clone(),
                tensor(vec![2], ElementType::U32),
            ),
            (
                language_io.repetition_penalty_input.clone(),
                tensor(vec![1], ElementType::F32),
            ),
        ]);
        if let Some(schema) = causal_inputs {
            for spec in schema.specs() {
                let mut dimensions = vec![input_capacity];
                dimensions.extend(&spec.row_dimensions);
                input_tensors.insert(spec.value_id.clone(), tensor(dimensions, spec.element_type));
            }
        }
        let mut options = ProgramPlanCompileOptions::new(input_tensors)
            .map_err(|error| FerrumError::model(format!("vNext compile input: {error}")))?
            .with_weight_arena_packing(config.weight_arena_packing);
        config.plan_observation.apply(family, &mut options)?;
        if let Some(selection) = &checkpoint_selection {
            selection.retain_in(&mut options);
        }
        options.require_weight_materializer_selection(materializer_selection);
        let compile_phase = StartupPhaseTimer::start("plan_compile");
        let compilation = ProgramPlanCompiler::compile_with_weight_materializers(
            family,
            &self.catalog,
            &config.runtime_policy,
            &self.registry.planning(),
            &self.weight_materializers,
            &options,
        )
        .map_err(ProgramCompilationError::from_plan)?;
        config
            .plan_observation
            .validate_compilation(family, &compilation)?;
        compile_phase.finish();
        Ok(CompiledProgram {
            composition: self,
            prepared,
            info,
            config,
            compilation,
            language_io,
            checkpoint_selection,
            repetition_capacity,
            executor_startup,
        })
    }
}

impl<'a, R: DeviceRuntime> CompiledProgram<'a, R> {
    pub fn prepared(&self) -> &PreparedExecutionInput<'a> {
        &self.prepared
    }
    pub fn compilation(&self) -> &ProgramPlanCompilation {
        &self.compilation
    }

    /// Additional materializer WeightBacking for this retained plan and source.
    /// Keep the returned lease through initialization: its warm backing may
    /// already be charged to the caller's Host account. Source-adapter backing
    /// and TransferStaging are separate requirements, not included here.
    pub fn prepare_static_host_backing(&self) -> Result<HostBackingPreparation> {
        self.compilation
            .executable()
            .execution_plan()
            .prepare_host_backing(self.prepared.family(), Some(self.prepared.weights()))
            .map_err(|error| FerrumError::model(format!("vNext static Host backing: {error}")))
    }

    /// The exact retained bounded-upload policy, not an allocation grant.
    pub fn static_initialization_policy(&self) -> StaticInitializationPolicy {
        self.config.static_initialization
    }

    /// Evaluate the retained compiled plan without allocating device memory.
    pub fn startup_peak_bytes(&self, workload: ferrum_types::StartupWorkload) -> Result<u64> {
        let (context, frontier, sequences, tokens) = match workload {
            ferrum_types::StartupWorkload::Prefill {
                context_tokens,
                chunk_tokens,
            } => (context_tokens, context_tokens, 1, chunk_tokens),
            ferrum_types::StartupWorkload::Decode {
                context_tokens,
                active_sequences,
            } => (context_tokens, 1, active_sequences, active_sequences),
        };
        let context = u64::try_from(context)
            .map_err(|_| FerrumError::config("startup context exceeds u64"))?;
        let frontier = u64::try_from(frontier)
            .map_err(|_| FerrumError::config("startup sequence frontier exceeds u64"))?;
        let sequences = u32::try_from(sequences)
            .map_err(|_| FerrumError::config("startup sequence count exceeds u32"))?;
        let tokens = u64::try_from(tokens)
            .map_err(|_| FerrumError::config("startup token count exceeds u64"))?;
        self.compilation
            .executable()
            .execution_plan()
            .payload()
            .memory()
            .startup_workload_peak_bytes(context, frontier, sequences, tokens)
            .map_err(|error| FerrumError::config(format!("compiled startup memory: {error}")))
    }

    /// Attach final evaluation evidence after confirming the report describes
    /// this retained compilation's immutable policy.
    pub fn set_startup_memory_plan(&mut self, plan: ferrum_types::StartupMemoryPlan) -> Result<()> {
        let memory = self
            .compilation
            .executable()
            .execution_plan()
            .payload()
            .memory();
        if plan.selected.context_tokens != self.config.maximum_model_tokens
            || plan.selected.max_sequences as u64 != u64::from(memory.maximum_active_sequences())
            || plan.selected.max_batch_tokens as u64
                != self
                    .config
                    .runtime_policy
                    .maximum_scheduled_tokens()
                    .map_err(|error| FerrumError::config(error.to_string()))?
            || plan.request.usable_capacity_bytes != memory.usable_capacity_bytes()
            || plan.context_peak_bytes > memory.usable_capacity_bytes()
            || plan.decode_peak_bytes > memory.usable_capacity_bytes()
        {
            return Err(FerrumError::config(
                "startup memory report differs from the retained compiled plan",
            ));
        }
        self.config.startup_memory_plan = Some(plan);
        Ok(())
    }

    pub fn initialize<F>(self, resolve_plan: F) -> crate::ExecutorBuildResult<ProgramExecutor<R>>
    where
        F: FnOnce(
            &ResolvedRuntimePolicy,
            &CapabilityCatalog,
            &ProgramPlanCompilation,
        ) -> Result<ResolvedModelPlan>,
    {
        ProgramExecutor::from_compiled_model(
            self,
            StaticInstallation::Private,
            None,
            None,
            None,
            resolve_plan,
        )
        .map(|(executor, _)| executor)
    }

    /// Install static weights using the caller's actual Host authority.
    ///
    /// Managed source/materializer backing and upload staging use this account
    /// on the same runtime and resource transaction as ordinary initialization.
    /// The caller must bind Host-aware sources to the same account and retain
    /// any prepared backing leases through this call. This entrypoint does not
    /// change subsequent causal input/output allocation or execution lanes.
    pub fn initialize_with_static_host<F>(
        self,
        host: &HostMemoryAccount,
        resolve_plan: F,
    ) -> crate::ExecutorBuildResult<ProgramExecutor<R>>
    where
        F: FnOnce(
            &ResolvedRuntimePolicy,
            &CapabilityCatalog,
            &ProgramPlanCompilation,
        ) -> Result<ResolvedModelPlan>,
    {
        ProgramExecutor::from_compiled_model(
            self,
            StaticInstallation::Private,
            Some(host),
            None,
            None,
            resolve_plan,
        )
        .map(|(executor, _)| executor)
    }

    /// Explicit causal I/O profile. Both static installation and every later
    /// wave consume this original Host authority; legacy construction is unchanged.
    pub fn initialize_with_managed_io<F>(
        self,
        host: &HostMemoryAccount,
        profile: ManagedCausalIoProfile,
        resolve_plan: F,
    ) -> crate::ExecutorBuildResult<ProgramExecutor<R>>
    where
        F: FnOnce(
            &ResolvedRuntimePolicy,
            &CapabilityCatalog,
            &ProgramPlanCompilation,
        ) -> Result<ResolvedModelPlan>,
    {
        ProgramExecutor::from_compiled_model(
            self,
            StaticInstallation::Private,
            Some(host),
            Some(ManagedCausalIo {
                host: host.clone(),
                profile,
            }),
            None,
            resolve_plan,
        )
        .map(|(executor, _)| executor)
    }

    /// Install and execute with this original Host authority while retaining an
    /// owned phase resource through whole-executor retirement. The resource may
    /// be an already granted HostMemoryScope or a tuple of phase-owned leases.
    /// This creates no additional Host account, grant or retirement reporter.
    ///
    /// Successful construction transfers custody to the original ExecutorOwner;
    /// failed construction transfers it to the original cleanup observer, or
    /// releases it only when no retained native cleanup remains. An unhandled
    /// panic after native acquisition starts continues unwinding and quarantines
    /// the resource alongside runtime's original abandonment fallback. Such a
    /// raw-panic quarantine currently has no recovery/retirement receipt and
    /// cannot be reported as reclaimed. Pre-acquisition panic drops normally.
    pub fn initialize_with_managed_io_retained<F, P>(
        self,
        host: &HostMemoryAccount,
        profile: ManagedCausalIoProfile,
        owned_phase: P,
        resolve_plan: F,
    ) -> crate::ExecutorBuildResult<crate::ExecutorOwner>
    where
        P: Send + 'static,
        F: FnOnce(
            &ResolvedRuntimePolicy,
            &CapabilityCatalog,
            &ProgramPlanCompilation,
        ) -> Result<ResolvedModelPlan>,
    {
        self.initialize_retained_installation(
            StaticInstallation::Private,
            host,
            profile,
            owned_phase,
            resolve_plan,
        )
        .map(|(owner, _)| owner)
    }

    /// Publish one completed static installation while the original phase
    /// remains retained by this executor's ordinary shutdown authority.
    pub fn initialize_shared_weights_with_managed_io_retained<F, P>(
        self,
        host: &HostMemoryAccount,
        profile: ManagedCausalIoProfile,
        owned_phase: P,
        resolve_plan: F,
    ) -> crate::ExecutorBuildResult<(crate::ExecutorOwner, SharedProgramWeights<R>)>
    where
        P: Send + 'static,
        F: FnOnce(
            &ResolvedRuntimePolicy,
            &CapabilityCatalog,
            &ProgramPlanCompilation,
        ) -> Result<ResolvedModelPlan>,
    {
        self.initialize_retained_installation(
            StaticInstallation::PublishShared,
            host,
            profile,
            owned_phase,
            resolve_plan,
        )
        .map(|(owner, weights)| {
            (
                owner,
                weights.expect("shared publication returns its original owner"),
            )
        })
    }

    /// Attach managed causal execution to an existing installation. Static
    /// source bytes are not read again; phase custody follows the same close
    /// and failed-construction path as an ordinary managed executor.
    pub fn initialize_with_shared_weights_and_managed_io_retained<F, P>(
        self,
        weights: &SharedProgramWeights<R>,
        host: &HostMemoryAccount,
        profile: ManagedCausalIoProfile,
        owned_phase: P,
        resolve_plan: F,
    ) -> crate::ExecutorBuildResult<crate::ExecutorOwner>
    where
        P: Send + 'static,
        F: FnOnce(
            &ResolvedRuntimePolicy,
            &CapabilityCatalog,
            &ProgramPlanCompilation,
        ) -> Result<ResolvedModelPlan>,
    {
        self.initialize_retained_installation(
            StaticInstallation::AttachShared(weights),
            host,
            profile,
            owned_phase,
            resolve_plan,
        )
        .map(|(owner, _)| owner)
    }

    fn initialize_retained_installation<F, P>(
        self,
        installation: StaticInstallation<'_, R>,
        host: &HostMemoryAccount,
        profile: ManagedCausalIoProfile,
        owned_phase: P,
        resolve_plan: F,
    ) -> crate::ExecutorBuildResult<(crate::ExecutorOwner, Option<SharedProgramWeights<R>>)>
    where
        P: Send + 'static,
        F: FnOnce(
            &ResolvedRuntimePolicy,
            &CapabilityCatalog,
            &ProgramPlanCompilation,
        ) -> Result<ResolvedModelPlan>,
    {
        let custody = phase_custody::PhaseCustody::new(owned_phase);
        let result = ProgramExecutor::from_compiled_model(
            self,
            installation,
            Some(host),
            Some(ManagedCausalIo {
                host: host.clone(),
                profile,
            }),
            Some(custody.native_started()),
            resolve_plan,
        );
        match result {
            Ok((executor, weights)) => {
                let owner = crate::ExecutorOwner::new(Box::new(executor));
                custody.handoff(&owner.shutdown_observer());
                Ok((owner, weights))
            }
            Err(error) => {
                custody.finish_error(&error);
                Err(error)
            }
        }
    }

    /// Initialize once and publish that exact readonly physical installation.
    /// The returned executor and handle share the original capacity claim;
    /// sequence state and dynamic pools are never part of the shared owner.
    pub fn initialize_shared_weights<F>(
        self,
        resolve_plan: F,
    ) -> crate::ExecutorBuildResult<(ProgramExecutor<R>, SharedProgramWeights<R>)>
    where
        F: FnOnce(
            &ResolvedRuntimePolicy,
            &CapabilityCatalog,
            &ProgramPlanCompilation,
        ) -> Result<ResolvedModelPlan>,
    {
        ProgramExecutor::from_compiled_model(
            self,
            StaticInstallation::PublishShared,
            None,
            None,
            None,
            resolve_plan,
        )
        .map(|(executor, weights)| {
            (
                executor,
                weights.expect("shared publication returns its original owner"),
            )
        })
    }

    /// Attach an independently compiled token executor to a compatible original
    /// installation. Complete weight identity, physical placement, runtime and
    /// capacity ceilings must agree; equal weight names alone are insufficient.
    /// The prepared source is not read on this path. All ordinary executor
    /// preflight, admission, completion and shutdown rules remain in effect.
    pub fn initialize_with_shared_weights<F>(
        self,
        weights: &SharedProgramWeights<R>,
        resolve_plan: F,
    ) -> crate::ExecutorBuildResult<ProgramExecutor<R>>
    where
        F: FnOnce(
            &ResolvedRuntimePolicy,
            &CapabilityCatalog,
            &ProgramPlanCompilation,
        ) -> Result<ResolvedModelPlan>,
    {
        ProgramExecutor::from_compiled_model(
            self,
            StaticInstallation::AttachShared(weights),
            None,
            None,
            None,
            resolve_plan,
        )
        .map(|(executor, _)| executor)
    }

    /// Cold per-wave physical I/O bound. This metadata query grants no credit.
    pub fn managed_io_requirements(
        &self,
        host: &HostMemoryAccount,
        profile: ManagedCausalIoProfile,
        work: &[ManagedCausalIoWork],
    ) -> Result<ManagedCausalIoRequirements> {
        let io = ProgramExecutor::<R>::resolve_io(
            self.compilation.executable(),
            &self.language_io,
            self.info.vocab_size,
            usize::try_from(self.repetition_capacity)
                .map_err(|_| FerrumError::config("managed repetition capacity exceeds usize"))?,
        )?;
        profile.validate_logits_type(io.output_element_type)?;
        managed_io::requirements_with_inputs(
            host,
            self.info.vocab_size,
            profile,
            work,
            self.language_io.causal_inputs.as_ref(),
        )
    }
}

/// Both public initialization paths retain the original committed transaction
/// and feed their result into the driver's existing activation/cleanup chain.
pub(super) fn initialize_static_with_host<R: DeviceRuntime>(
    transaction: ResourceTransaction<RuntimeResourceDriver<R>, TransactionCommitted>,
    family: &PreparedModelFamily,
    plan: &ExecutionPlan,
    source: &dyn WeightComponentSource,
    host: Option<&HostMemoryAccount>,
    policy: StaticInitializationPolicy,
) -> std::result::Result<
    InitializedResourceTransaction<RuntimeResourceDriver<R>>,
    StaticInitializationFailure<RuntimeResourceDriver<R>>,
> {
    match host {
        Some(host) => transaction.initialize_static_managed(family, plan, source, host, policy),
        None => transaction.initialize_static(family, plan, source, policy),
    }
}

#[cfg(test)]
mod static_host_tests;
