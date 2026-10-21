//! Metadata-only candidate compilation followed by initialization of the exact
//! retained result. The borrow ties compilation to the actual registries used
//! for binding; callers cannot substitute a reconstructed capability list.

use super::*;

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
    pub(super) registry: OperationRuntimeRegistry<R>,
    weight_materializers: WeightMaterializerRegistry,
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
        let language_io = ProgramExecutor::<R>::resolve_language_io_ids(family.program())?;
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
        let mut options = ProgramPlanCompileOptions::new(BTreeMap::from([
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
        ]))
        .map_err(|error| FerrumError::model(format!("vNext compile input: {error}")))?;
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
        ProgramExecutor::from_compiled_model(self, resolve_plan)
    }
}
