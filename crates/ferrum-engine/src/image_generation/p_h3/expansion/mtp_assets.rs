//! Opt-in MTP model preparation. One union installation owns all static
//! weights; main and draft attach without another conversion or upload.
//! Token rendering/extraction stay with NativeTextExpansion. This module owns
//! no scheduler, new budget, or global cache.
use super::*;
use crate::image_generation::host_budget::{stored_staging_bytes, ExecutionHostBudget};
use ferrum_execution::driver::SharedProgramWeights;
use ferrum_execution::{CausalProgramSession, CausalTailTwoOutput, CompiledCausalProgram};
use ferrum_models::{
    qwen35_config::Qwen35TextConfig,
    vnext::qwen35::mtp::{self as model, Qwen35MtpFamilyProvider, Qwen35MtpRole},
};
use ferrum_runtime::{
    ElementType, ExecutionPlan, HostBackingPreparation, HostMemoryClass, ModelFamilyRegistration,
    ModelFamilyRegistry, PreparedModelFamily, ProgramPlanCompilation, ProgramPlanCompileOptions,
    ProgramTensorSpec, ProgramValueId, ResolvedTensorLayout, StaticInitializationPolicy,
    TypedFamilyRegistration,
};
use std::collections::BTreeMap;

mod inputs;

struct RoleAssets {
    defined: DefinedProductionModel,
    model: ManagedProductionModel,
}

/// Metadata and the already verified source mmap, never a second device owner.
pub(crate) struct MtpExpansionAssets {
    domain: Arc<()>,
    union: RoleAssets,
    main: RoleAssets,
    draft: RoleAssets,
    text: Qwen35TextConfig,
    source_backing: PH3ImagePromptExpansionSourceBacking,
    conversion: ferrum_quantization::Bf16F16ConversionPolicy,
}

struct Registry(TypedFamilyRegistration<Qwen35MtpFamilyProvider>);
impl ModelFamilyRegistry for Registry {
    fn registrations(&self) -> Vec<&dyn ModelFamilyRegistration> {
        vec![&self.0]
    }
}

/// A one-installation lease of the actual source and original Host account.
/// Keep it alive through initialization, including the original fence recovery.
pub(crate) struct MtpExpansionSource<'a> {
    assets: &'a MtpExpansionAssets,
    role: Qwen35MtpRole,
    host: &'a HostMemoryAccount,
    source: ManagedProductionWeightSource<'a>,
}

/// The materializer lease is moved into the original PhaseCustody before any
/// native installation. The source itself uses its original managed transfer
/// owners; an unknown submission is recovered by the original Core owner.
pub(crate) struct CompiledMtpExpansion<'compiled, 'source, R: DeviceRuntime> {
    assets: &'source MtpExpansionAssets,
    role: Qwen35MtpRole,
    host: &'source HostMemoryAccount,
    source: &'compiled ManagedProductionWeightSource<'source>,
    compiled: CompiledCausalProgram<'compiled, R>,
    engine: EngineConfig,
    static_policy: StaticInitializationPolicy,
    backing: HostBackingPreparation,
    needs_staging: bool,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct MtpWaveHostRequirements {
    pub(crate) input_bytes: usize,
    pub(crate) output_bytes: usize,
}

impl MtpExpansionAssets {
    pub(crate) fn from_native(native: &NativeTextExpansion) -> Result<Self> {
        let sources = Arc::clone(native.defined.sources());
        let raw = serde_json::from_slice(sources.config_json()).map_err(model_error)?;
        let text = Qwen35TextConfig::from_hf_config_value(&raw).map_err(model_error)?;
        let define = |role: Qwen35MtpRole| -> Result<_> {
            let defined = model::define_from_sources(Arc::clone(&sources), role)?;
            let prepared =
                Arc::new(defined.prepare(&role.numerical_profile().parse().map_err(model_error)?)?);
            Ok((defined, prepared))
        };
        let (defined, prepared) = define(Qwen35MtpRole::InstallUnion)?;
        // This shares the verified original mapping. If a process-mapped
        // policy exists its same original limit must admit the complete union.
        let union_model = native.model.with_prepared(prepared)?;
        let union = RoleAssets {
            defined,
            model: union_model,
        };
        let (defined, prepared) = define(Qwen35MtpRole::MainVerify)?;
        let main = RoleAssets {
            defined,
            model: union.model.with_prepared(prepared)?,
        };
        let (defined, prepared) = define(Qwen35MtpRole::OneDraft)?;
        let draft = RoleAssets {
            defined,
            model: union.model.with_prepared(prepared)?,
        };
        for consumer in [&main, &draft] {
            if consumer.model.prepared().family().weights()
                != union.model.prepared().family().weights()
                || consumer.model.prepared().family().program().weights()
                    != union.model.prepared().family().program().weights()
            {
                return Err(FerrumError::model(
                    "MTP roles have different union weight declarations",
                ));
            }
        }
        Ok(Self {
            domain: Arc::clone(&native.domain),
            union,
            main,
            draft,
            text,
            source_backing: native.source_backing,
            conversion: native.source_conversion_policy,
        })
    }

    fn role(&self, role: Qwen35MtpRole) -> Result<&RoleAssets> {
        match role {
            Qwen35MtpRole::InstallUnion => Ok(&self.union),
            Qwen35MtpRole::MainVerify => Ok(&self.main),
            Qwen35MtpRole::OneDraft => Ok(&self.draft),
            Qwen35MtpRole::MainWithDraftWeights => Err(FerrumError::config(
                "MTP execution uses MainVerify; ordinary main remains on the original route",
            )),
        }
    }

    pub(crate) fn source<'a>(
        &'a self,
        role: Qwen35MtpRole,
        host: &'a HostMemoryAccount,
    ) -> Result<MtpExpansionSource<'a>> {
        Ok(MtpExpansionSource {
            assets: self,
            role,
            host,
            source: self.role(role)?.model.weight_source(host),
        })
    }

    fn resolve(
        &self,
        role: Qwen35MtpRole,
        engine: &EngineConfig,
        policy: &ferrum_runtime::ResolvedRuntimePolicy,
        catalog: &ferrum_runtime::CapabilityCatalog,
        compilation: &ProgramPlanCompilation,
    ) -> Result<ferrum_runtime::ResolvedModelPlan> {
        let asset = self.role(role)?;
        let numerical = NumericalProfileResolution::from_static_plan(
            engine.numerical_execution.clone(),
            ferrum_types::KvStorageFormat::F16,
            asset.defined.definition(),
            asset.model.prepared().family(),
            catalog,
            policy,
            compilation.executable().execution_plan(),
            vec![],
        )
        .map_err(model_error)?;
        let registry = Registry(TypedFamilyRegistration::new(
            Qwen35MtpFamilyProvider::new().map_err(model_error)?,
        ));
        crate::product_composition::resolve_model_plan_with_registry(
            engine,
            asset.model.prepared(),
            catalog,
            policy,
            compilation,
            &numerical,
            &registry,
        )
    }
}

impl<'source> MtpExpansionSource<'source> {
    pub(crate) fn compile<'a, R: DeviceRuntime>(
        &'a self,
        composition: &'a RuntimeComposition<R>,
        engine: &EngineConfig,
    ) -> Result<CompiledMtpExpansion<'a, 'source, R>> {
        validate_engine(engine)?;
        let mut engine = engine.clone();
        if self.role == Qwen35MtpRole::InstallUnion {
            // This graph only installs the shared weights. It never executes
            // token work and must not reserve another reusable lane/catalog.
            engine.backend.enable_reusable_execution = false;
            engine
                .backend
                .reusable_execution_capture
                .maximum_executables_per_lane = None;
        }
        engine.numerical_execution = self.role.numerical_profile().parse().map_err(model_error)?;
        let model = self.assets.role(self.role)?.model.prepared();
        let info = model.model_info(engine.model.model_id.clone(), engine.backend.device.clone());
        let config =
            ProgramExecutorConfig::from_engine_config(&engine, &info, composition.runtime())?;
        let options = input_options(
            model.family(),
            &self.assets.text,
            config.maximum_model_tokens,
            model.descriptor().vocabulary_size(),
            config.weight_arena_packing,
        )?;
        let materializer = WeightMaterializerSelection::exact(
            WeightMaterializerId::new(IDENTITY_WEIGHT_MATERIALIZER_ID).map_err(model_error)?,
        );
        let prepared = self.source.execution_input()?;
        let compiled = if self.role == Qwen35MtpRole::InstallUnion {
            composition.compile_causal_weight_installation(
                prepared,
                config.runtime_policy,
                options,
                materializer,
            )
        } else {
            let (token, decisions, hidden) = if self.role == Qwen35MtpRole::OneDraft {
                (model::TOKEN_IDS, model::GREEDY_TOKENS, model::HIDDEN_TAIL)
            } else {
                (
                    "value.input.token_ids",
                    model::MAIN_GREEDY_TOKENS,
                    model::MAIN_HIDDEN_TAIL,
                )
            };
            composition.compile_causal_program_with_declared_eager_boundaries(
                prepared,
                config.runtime_policy,
                options,
                materializer,
                CausalTailTwoOutput::new(value(token)?, value(decisions)?, value(hidden)?)?,
            )
        }
        .map_err(FerrumError::from)?;
        let plan = compiled.compilation().executable().execution_plan();
        // Consumers attach to an existing installation. They must not arm or
        // materialize another source slot, even when their recipes are equal.
        if self.role == Qwen35MtpRole::InstallUnion {
            match self.assets.source_backing {
                PH3ImagePromptExpansionSourceBacking::Fresh => {}
                PH3ImagePromptExpansionSourceBacking::ReusableSingleSlot => self
                    .source
                    .prepare_reusable_static_backing(plan)
                    .map_err(model_error)?,
                PH3ImagePromptExpansionSourceBacking::BoundedStreaming => self
                    .source
                    .prepare_bounded_streaming_static_backing_with_policy(
                        plan,
                        64 << 20,
                        self.assets.conversion,
                    )
                    .map_err(model_error)?,
            }
        }
        let backing = if self.role == Qwen35MtpRole::InstallUnion {
            plan.prepare_host_backing(model.family(), Some(&self.source))
                .map_err(model_error)?
        } else {
            HostBackingPreparation::new(0)
        };
        Ok(CompiledMtpExpansion {
            assets: self.assets,
            role: self.role,
            host: self.host,
            source: &self.source,
            compiled,
            engine,
            static_policy: config.static_initialization,
            backing,
            needs_staging: composition.runtime().requires_managed_static_staging(),
        })
    }
}

impl<R: DeviceRuntime> CompiledMtpExpansion<'_, '_, R> {
    pub(crate) fn compilation(&self) -> &ProgramPlanCompilation {
        self.compiled.compilation()
    }
    pub(crate) fn plan(&self) -> &ExecutionPlan {
        self.compilation().executable().execution_plan()
    }

    /// Keep the original materializer assumptions alive after metadata-only
    /// planning. This moves its existing lease; it performs no materialization.
    pub(crate) fn into_host_backing(self) -> HostBackingPreparation {
        self.backing
    }

    /// Static installation demand only. The cohort must add simultaneously
    /// live request tables, history, main/draft outputs and current-wave inputs.
    /// Actual union schema includes all 15 MTP source tensors, exactly once.
    pub(crate) fn static_host_requirements(&self) -> Result<ExecutionHostBudget> {
        if self.role != Qwen35MtpRole::InstallUnion {
            return Ok(ExecutionHostBudget::default());
        }
        Ok(ExecutionHostBudget {
            weights: self
                .source
                .maximum_group_backing_bytes(self.plan())
                .map_err(model_error)?
                .checked_add(self.backing.additional_bytes())
                .ok_or_else(|| FerrumError::config("MTP static backing overflows"))?,
            staging: if self.needs_staging {
                stored_staging_bytes(
                    self.host,
                    self.plan().payload().execution_weights(),
                    self.static_policy,
                )
                .map_err(model_error)?
            } else {
                0
            },
            ..ExecutionHostBudget::default()
        })
    }

    /// Combine the original union installation demand with the caller's real
    /// persistent conditioning/history and simultaneously live wave payloads.
    /// Retained outputs include any accepted hidden owner held for the draft.
    /// This is a preflight calculation, never a second admission authority.
    pub(crate) fn phase_host_requirements(
        &self,
        retained_input: usize,
        retained_output: usize,
        concurrent_waves: &[MtpWaveHostRequirements],
    ) -> Result<ExecutionHostBudget> {
        if self.role != Qwen35MtpRole::InstallUnion {
            return Err(FerrumError::config(
                "MTP phase budget requires the complete union plan",
            ));
        }
        let mut budget = self.static_host_requirements()?;
        budget.input = retained_input;
        budget.output = retained_output;
        for wave in concurrent_waves {
            budget.input = add(budget.input, wave.input_bytes)?;
            budget.output = add(budget.output, wave.output_bytes)?;
        }
        Ok(budget)
    }

    /// Exact current-wave payloads with each backing padded independently.
    /// This excludes persistent conditioning/history and other live waves.
    pub(crate) fn wave_host_requirements(&self, rows: &[usize]) -> Result<MtpWaveHostRequirements> {
        if self.role == Qwen35MtpRole::InstallUnion || rows.is_empty() || rows.contains(&0) {
            return Err(FerrumError::config(
                "MTP wave requires executable nonempty participants",
            ));
        }
        let family = self.compiled.prepared().family();
        let tensors = self.compilation().value_tensors();
        let mut input_bytes = 0usize;
        let mut output_bytes = 0usize;
        for &rows in rows {
            for id in family.program().inputs() {
                let spec = tensors
                    .get(id)
                    .ok_or_else(|| FerrumError::model("MTP input shape missing"))?;
                let dims = spec.dimensions();
                if dims.is_empty() || rows as u64 > dims[0] {
                    return Err(FerrumError::config("MTP wave exceeds input row capacity"));
                }
                let bytes = (rows as u64)
                    .checked_mul(spec.element_type().size_bytes())
                    .and_then(|start| dims[1..].iter().try_fold(start, |n, d| n.checked_mul(*d)))
                    .and_then(|n| usize::try_from(n).ok())
                    .ok_or_else(|| FerrumError::config("MTP input extent overflows"))?;
                input_bytes = add(
                    input_bytes,
                    self.host.backing_bytes(bytes).map_err(model_error)?,
                )?;
            }
            for id in family.program().outputs() {
                let spec = tensors
                    .get(id)
                    .ok_or_else(|| FerrumError::model("MTP output shape missing"))?;
                let bytes = usize::try_from(spec.minimum_storage_bytes().map_err(model_error)?)
                    .map_err(model_error)?;
                output_bytes = add(
                    output_bytes,
                    self.host.backing_bytes(bytes).map_err(model_error)?,
                )?;
            }
        }
        Ok(MtpWaveHostRequirements {
            input_bytes,
            output_bytes,
        })
    }

    /// Caller closes the installation-only session before attaching consumers,
    /// retaining the returned sealed physical/source owner throughout.
    pub(crate) fn initialize_union_retained<P: Send + 'static>(
        self,
        owned_phase: P,
    ) -> ExecutorBuildResult<(CausalProgramSession<R>, SharedProgramWeights<R>)> {
        if self.role != Qwen35MtpRole::InstallUnion {
            return Err(FerrumError::config("only the union graph may publish MTP weights").into());
        }
        if self.assets.source_backing == PH3ImagePromptExpansionSourceBacking::Fresh {
            self.source
                .prepare_process_mapped_weights()
                .map_err(model_error)?;
        }
        let Self {
            assets,
            role,
            host,
            compiled,
            engine,
            static_policy,
            backing,
            ..
        } = self;
        compiled.initialize_shared_weights_retained(
            host,
            static_policy,
            (owned_phase, backing),
            |policy, catalog, compilation| {
                assets.resolve(role, &engine, policy, catalog, compilation)
            },
        )
    }

    pub(crate) fn attach_retained<P: Send + 'static>(
        self,
        weights: &SharedProgramWeights<R>,
        owned_phase: P,
    ) -> ExecutorBuildResult<CausalProgramSession<R>> {
        if self.role == Qwen35MtpRole::InstallUnion {
            return Err(
                FerrumError::config("installation-only graph is not an MTP consumer").into(),
            );
        }
        let Self {
            assets,
            role,
            host,
            compiled,
            engine,
            static_policy,
            backing,
            ..
        } = self;
        compiled.initialize_with_shared_weights_retained(
            weights,
            host,
            static_policy,
            (owned_phase, backing),
            |policy, catalog, compilation| {
                assets.resolve(role, &engine, policy, catalog, compilation)
            },
        )
    }
}

fn input_options(
    family: &PreparedModelFamily,
    text: &Qwen35TextConfig,
    context: usize,
    vocabulary: usize,
    packing: ferrum_runtime::WeightArenaPacking,
) -> Result<ProgramPlanCompileOptions> {
    let context = u64::try_from(context).map_err(model_error)?;
    let hidden = u64::try_from(text.hidden_size).map_err(model_error)?;
    let pairs = u64::try_from(text.full_attention_rope_dim() / 2).map_err(model_error)?;
    let vocab = u64::try_from(vocabulary).map_err(model_error)?;
    let tensors = family
        .program()
        .inputs()
        .iter()
        .map(|id| {
            let (dimensions, element_type) = match id.as_str() {
                "value.input.token_ids" | model::TOKEN_IDS => (vec![context], ElementType::U32),
                "value.input.visual_rows" | model::ACCEPTED_HIDDEN => {
                    (vec![context, hidden], ElementType::F16)
                }
                "value.input.visual_replace_mask" => (vec![context], ElementType::U8),
                "value.input.rope_cos"
                | "value.input.rope_sin"
                | model::ROPE_COS
                | model::ROPE_SIN => (vec![context, pairs], ElementType::F16),
                "value.input.greedy_token_mask" => (vec![vocab], ElementType::U8),
                "value.input.greedy_repetition_token_ids" => {
                    (vec![context.min(vocab)], ElementType::U32)
                }
                "value.input.greedy_repetition_offsets" => (vec![2], ElementType::U32),
                "value.input.greedy_repetition_penalty" => (vec![1], ElementType::F32),
                _ => return Err(FerrumError::model("unexpected MTP role input")),
            };
            Ok((
                id.clone(),
                ProgramTensorSpec {
                    dimensions,
                    element_type,
                    layout: ResolvedTensorLayout::Contiguous,
                },
            ))
        })
        .collect::<Result<BTreeMap<_, _>>>()?;
    let mut options = ProgramPlanCompileOptions::new(tensors)
        .map_err(model_error)?
        .with_weight_arena_packing(packing);
    model::configure_compile_options(family, &mut options).map_err(model_error)?;
    Ok(options)
}

fn validate_engine(engine: &EngineConfig) -> Result<()> {
    let p = &engine.sampling.default_params;
    p.validate()?;
    let greedy = SamplingParams::greedy();
    if p.temperature != 0.
        || p.top_p != 1.
        || p.top_k.is_some()
        || p.repetition_penalty != 1.
        || p.presence_penalty != 0.
        || p.frequency_penalty != 0.
        || p.min_p.is_some()
        || p.tfs.is_some()
        || p.typical_p.is_some()
        || p.mirostat.is_some()
        || !p.stop_sequences.is_empty()
        || p.response_format != greedy.response_format
        || p.structured_output_start != greedy.structured_output_start
        || p.response_completion_boundary != greedy.response_completion_boundary
        || p.model_output_protocol != greedy.model_output_protocol
        || engine.runtime.vnext_checkpoint_capture.is_some()
    {
        return Err(FerrumError::unsupported(
            "one-draft MTP requires greedy execution without logits processors or checkpoint capture",
        ));
    }
    if engine.backend.enable_reusable_execution
        && (engine.backend.reusable_execution_capture.preparation
            != ferrum_types::ReusableExecutionPreparationMode::OnDemand
            || engine.backend.reusable_execution_capture.scope
                != ferrum_types::ReusableExecutionCaptureScope::DecodeOnly)
    {
        return Err(FerrumError::unsupported(
            "one-draft MTP replay requires on-demand decode-only preparation",
        ));
    }
    Ok(())
}
fn value(id: &str) -> Result<ProgramValueId> {
    ProgramValueId::new(id).map_err(model_error)
}
fn add(a: usize, b: usize) -> Result<usize> {
    a.checked_add(b)
        .ok_or_else(|| FerrumError::config("MTP Host requirement overflows"))
}
