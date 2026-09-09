//! Metadata-only candidate compilation followed by initialization of the exact
//! retained result. The borrow ties compilation to the actual registries used
//! for binding; callers cannot substitute a reconstructed capability list.

use super::*;

pub struct VNextRuntimeComposition<R: DeviceRuntime> {
    pub(super) runtime: Arc<R>,
    pub(super) registry: OperationRuntimeRegistry<R>,
    weight_materializers: WeightMaterializerRegistry,
    pub(super) catalog: CapabilityCatalog,
}

/// No Deserialize or public field construction: only successful static
/// compilation produces this initialization input. It borrows the exact
/// prepared source and runtime composition, including on the Auto path.
pub struct VNextCompiledModel<'a, R: DeviceRuntime> {
    pub(super) composition: &'a VNextRuntimeComposition<R>,
    pub(super) prepared: &'a PreparedProductionModel,
    pub(super) info: ModelInfo,
    pub(super) config: VNextExecutorConfig,
    pub(super) compilation: ProgramPlanCompilation,
    pub(super) language_io: VNextLanguageIoIds,
    pub(super) checkpoint_selection: Option<VNextCheckpointSelection>,
    pub(super) repetition_capacity: u64,
    pub(super) executor_startup: StartupPhaseTimer,
}

impl<R: DeviceRuntime> VNextRuntimeComposition<R> {
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
        prepared: &'a PreparedProductionModel,
        info: ModelInfo,
        engine_config: &EngineConfig,
        config: VNextExecutorConfig,
        materializer_selection: WeightMaterializerSelection,
    ) -> Result<VNextCompiledModel<'a, R>> {
        if self.runtime.descriptor() != self.catalog.device() {
            return Err(FerrumError::device(
                "vNext static compilation catalog differs from the actual device runtime",
            ));
        }
        let executor_startup = StartupPhaseTimer::start("executor_composition_total");
        let checkpoint_selection = VNextCheckpointSelection::from_config(
            engine_config.runtime.vnext_checkpoint_capture.as_ref(),
        )?;
        let family = prepared.family();
        let language_io = VNextModelExecutor::<R>::resolve_language_io_ids(family.program())?;
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
        .map_err(|error| FerrumError::model(format!("vNext plan compile: {error}")))?;
        config
            .plan_observation
            .validate_compilation(family, &compilation)?;
        compile_phase.finish();
        Ok(VNextCompiledModel {
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

impl<R: DeviceRuntime> VNextCompiledModel<'_, R> {
    pub fn prepared(&self) -> &PreparedProductionModel {
        self.prepared
    }
    pub fn compilation(&self) -> &ProgramPlanCompilation {
        &self.compilation
    }

    pub fn initialize<F>(self, resolve_plan: F) -> Result<VNextModelExecutor<R>>
    where
        F: FnOnce(
            &PreparedProductionModel,
            &ResolvedRuntimePolicy,
            &CapabilityCatalog,
            &ProgramPlanCompilation,
        ) -> Result<ResolvedModelPlan>,
    {
        VNextModelExecutor::from_compiled_model(self, resolve_plan)
    }
}
