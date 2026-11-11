//! Token programs with a bounded greedy terminal, using the same static
//! installation and Core sequence ownership as the ordinary language driver.
//! This layer does not interpret model names or invent sampling-policy inputs.
use super::*;
mod replay;
mod wave;
pub use wave::{
    CausalAcceptedOutput, CausalCommittedWave, CausalComputedWave, CausalVerifiedDraftOutput,
    CausalWaveFailure,
};

/// The actual model program declares these values. The compiled terminal must
/// be the typed F16 tail-two operation, not an arbitrary pair of output buffers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CausalTailTwoOutput {
    token_input: ProgramValueId,
    decisions: ProgramValueId,
    hidden: ProgramValueId,
}

impl CausalTailTwoOutput {
    pub fn new(
        token_input: ProgramValueId,
        decisions: ProgramValueId,
        hidden: ProgramValueId,
    ) -> Result<Self> {
        if token_input == decisions || token_input == hidden || decisions == hidden {
            return Err(FerrumError::model("causal terminal ports must be distinct"));
        }
        Ok(Self {
            token_input,
            decisions,
            hidden,
        })
    }

    fn validate_program(&self, program: &ModelProgram) -> Result<()> {
        if !program.inputs().contains(&self.token_input)
            || program.outputs().len() != 2
            || !program.outputs().contains(&self.decisions)
            || !program.outputs().contains(&self.hidden)
        {
            return Err(FerrumError::model(
                "causal terminal differs from declared program I/O",
            ));
        }
        let embeddings = program
            .blocks()
            .iter()
            .flat_map(|block| &block.nodes)
            .filter(|node| is_language_token_embedding_operation(node.operation_id.as_str()))
            .collect::<Vec<_>>();
        if embeddings.len() != 1 || embeddings[0].inputs.first() != Some(&self.token_input) {
            return Err(FerrumError::model(
                "causal token input must feed its sole token embedding",
            ));
        }
        let terminals = program
            .blocks()
            .iter()
            .flat_map(|b| &b.nodes)
            .filter(|node| {
                node.operation_id.as_str() == ferrum_runtime::TAIL_TWO_GREEDY_F16_OPERATION_ID
                    && node.outputs == [self.decisions.clone(), self.hidden.clone()]
            })
            .count();
        if terminals != 1 {
            return Err(FerrumError::model(
                "causal program requires one declared tail-two terminal",
            ));
        }
        Ok(())
    }
}

/// Only a real token compilation produces this value. An installation-only
/// graph can cover a union of main/draft weights without being executable as
/// a language request. Its consumers attach to the original physical owner.
pub struct CompiledCausalProgram<'a, R: DeviceRuntime> {
    composition: &'a RuntimeComposition<R>,
    prepared: PreparedExecutionInput<'a>,
    policy: ResolvedRuntimePolicy,
    compilation: ProgramPlanCompilation,
    terminal: Option<CausalTailTwoOutput>,
}

impl<R: DeviceRuntime> RuntimeComposition<R> {
    pub fn compile_causal_program<'a>(
        &'a self,
        prepared: PreparedExecutionInput<'a>,
        policy: ResolvedRuntimePolicy,
        options: ProgramPlanCompileOptions,
        materializer: WeightMaterializerSelection,
        terminal: CausalTailTwoOutput,
    ) -> std::result::Result<CompiledCausalProgram<'a, R>, ProgramCompilationError> {
        terminal.validate_program(prepared.family().program())?;
        self.compile_causal_inner(
            prepared,
            policy,
            options,
            materializer,
            Some(terminal),
            false,
        )
    }

    /// Replay only eligible compute segments. Providers declaring an eager
    /// boundary retain ordinary execution and fresh invocation ownership.
    /// The resolved policy promises runtime repeatability; each captured
    /// segment must independently provide bitwise eager equivalence.
    pub fn compile_causal_program_with_declared_eager_boundaries<'a>(
        &'a self,
        prepared: PreparedExecutionInput<'a>,
        policy: ResolvedRuntimePolicy,
        options: ProgramPlanCompileOptions,
        materializer: WeightMaterializerSelection,
        terminal: CausalTailTwoOutput,
    ) -> std::result::Result<CompiledCausalProgram<'a, R>, ProgramCompilationError> {
        terminal.validate_program(prepared.family().program())?;
        self.compile_causal_inner(
            prepared,
            policy,
            options,
            materializer,
            Some(terminal),
            true,
        )
    }

    pub fn compile_causal_weight_installation<'a>(
        &'a self,
        prepared: PreparedExecutionInput<'a>,
        policy: ResolvedRuntimePolicy,
        options: ProgramPlanCompileOptions,
        materializer: WeightMaterializerSelection,
    ) -> std::result::Result<CompiledCausalProgram<'a, R>, ProgramCompilationError> {
        self.compile_causal_inner(prepared, policy, options, materializer, None, false)
    }

    fn compile_causal_inner<'a>(
        &'a self,
        prepared: PreparedExecutionInput<'a>,
        policy: ResolvedRuntimePolicy,
        mut options: ProgramPlanCompileOptions,
        materializer: WeightMaterializerSelection,
        terminal: Option<CausalTailTwoOutput>,
        declared_eager_boundaries: bool,
    ) -> std::result::Result<CompiledCausalProgram<'a, R>, ProgramCompilationError> {
        if self.runtime.descriptor() != self.catalog.device() {
            return Err(FerrumError::device("causal catalog differs from its runtime").into());
        }
        if !matches!(policy.work_limits(), RuntimeWorkLimits::Tokens { .. }) {
            return Err(FerrumError::unsupported(
                "bounded causal programs require token execution",
            )
            .into());
        }
        let policy = if declared_eager_boundaries {
            replay::resolve_policy_with_eager_boundaries(policy, terminal.is_some())?
        } else {
            replay::resolve_policy(policy, terminal.is_some())?
        };
        if !self.runtime.supports_managed_tensor_io() {
            return Err(FerrumError::unsupported(
                "causal programs require managed uploads and readbacks",
            )
            .into());
        }
        if let Some(terminal) = &terminal {
            options.retain_completion_value(terminal.decisions.clone());
            options.retain_completion_value(terminal.hidden.clone());
        }
        options.require_weight_materializer_selection(materializer);
        let compilation = ProgramPlanCompiler::compile_with_weight_materializers(
            prepared.family(),
            &self.catalog,
            &policy,
            &self.registry.planning(),
            &self.weight_materializers,
            &options,
        )
        .map_err(ProgramCompilationError::from_plan)?;
        if let Some(terminal) = &terminal {
            ResolvedCausalTerminal::resolve(compilation.executable(), terminal, prepared.facts())?;
        }
        Ok(CompiledCausalProgram {
            composition: self,
            prepared,
            policy,
            compilation,
            terminal,
        })
    }
}

struct ResolvedCausalTerminal {
    spec: CausalTailTwoOutput,
    node: NodeId,
    decisions: ResolvedValueBinding,
    hidden: ResolvedValueBinding,
}

impl ResolvedCausalTerminal {
    fn resolve(
        executable: &impl ExecutablePlanView,
        spec: &CausalTailTwoOutput,
        facts: &crate::CausalExecutionFacts,
    ) -> Result<Self> {
        let nodes = executable.execution_plan().payload().nodes();
        let terminal_nodes = nodes
            .iter()
            .filter(|node| {
                node.operation_id().as_str() == ferrum_runtime::TAIL_TWO_GREEDY_F16_OPERATION_ID
                    && node
                        .values()
                        .iter()
                        .any(|v| v.value_id() == &spec.decisions)
            })
            .collect::<Vec<_>>();
        let [node] = terminal_nodes.as_slice() else {
            return Err(FerrumError::model(
                "compiled causal terminal is missing or ambiguous",
            ));
        };
        let output = |id: &ProgramValueId,
                      ordinal,
                      dtype,
                      dimensions: &[u64]|
         -> Result<ResolvedValueBinding> {
            let value = node
                .values()
                .iter()
                .find(|v| v.role() == ResolvedValueRole::Output && v.ordinal() == ordinal)
                .ok_or_else(|| FerrumError::model("compiled causal terminal port missing"))?;
            if value.value_id() != id
                || value.tensor().element_type() != dtype
                || value.tensor().dimensions() != dimensions
                || !matches!(value.tensor().layout(), ResolvedTensorLayout::Contiguous)
                || value.storage().components().len() != 1
            {
                return Err(FerrumError::model(
                    "compiled causal terminal port has the wrong physical layout",
                ));
            }
            Ok(value.clone())
        };
        let decisions = output(&spec.decisions, 0, ElementType::U32, &[2])?;
        let hidden = output(
            &spec.hidden,
            1,
            ElementType::F16,
            &[2, facts.hidden_size() as u64],
        )?;
        let projection = node
            .values()
            .iter()
            .find(|v| v.role() == ResolvedValueRole::Input && v.ordinal() == 1)
            .ok_or_else(|| FerrumError::model("causal head projection is missing"))?;
        if projection.tensor().dimensions()
            != [facts.vocabulary_size() as u64, facts.hidden_size() as u64]
        {
            return Err(FerrumError::model(
                "causal head projection differs from prepared execution facts",
            ));
        }
        let token_ports = nodes
            .iter()
            .flat_map(|node| node.values())
            .filter(|v| v.value_id() == &spec.token_input && v.role() == ResolvedValueRole::Input)
            .collect::<Vec<_>>();
        if token_ports.is_empty()
            || token_ports.iter().any(|v| {
                v.access() != TensorAccess::Read
                    || v.tensor().element_type() != ElementType::U32
                    || v.tensor().dimensions().len() != 1
                    || !matches!(v.tensor().layout(), ResolvedTensorLayout::Contiguous)
            })
        {
            return Err(FerrumError::model(
                "compiled causal token input must be contiguous readonly U32 rows",
            ));
        }
        Ok(Self {
            spec: spec.clone(),
            node: node.id().clone(),
            decisions,
            hidden,
        })
    }
}

/// Original initialized resources, providers and source-authenticated plan.
/// All requests and frames remain Core owners. The retained caller phase is
/// attached to the exact plan root until its checked close succeeds.
pub struct CausalProgramSession<R: DeviceRuntime> {
    plan: ResolvedModelPlan,
    providers: BoundOperationProviderSet<R>,
    resources: Arc<PlanRuntimeResources<R>>,
    host: HostMemoryAccount,
    terminal: Option<ResolvedCausalTerminal>,
    shared_receipt: Option<ProgramSharedWeightsReceipt>,
    _static_attribution: Option<StaticProviderAttributionWitness>,
    vocabulary_size: usize,
    replay: Option<replay::CausalReplay<R>>,
}

impl<'a, R: DeviceRuntime> CompiledCausalProgram<'a, R> {
    pub fn compilation(&self) -> &ProgramPlanCompilation {
        &self.compilation
    }
    pub fn prepared(&self) -> &PreparedExecutionInput<'a> {
        &self.prepared
    }

    pub fn initialize_shared_weights_retained<F, P>(
        self,
        host: &HostMemoryAccount,
        static_policy: StaticInitializationPolicy,
        owned_phase: P,
        resolve_plan: F,
    ) -> crate::ExecutorBuildResult<(CausalProgramSession<R>, SharedProgramWeights<R>)>
    where
        P: Send + 'static,
        F: FnOnce(
            &ResolvedRuntimePolicy,
            &CapabilityCatalog,
            &ProgramPlanCompilation,
        ) -> Result<ResolvedModelPlan>,
    {
        self.initialize_retained(
            StaticInstallation::PublishShared,
            host,
            static_policy,
            owned_phase,
            resolve_plan,
        )
        .map(|(session, weights)| {
            (
                session,
                weights.expect("publication returns original weights"),
            )
        })
    }

    pub fn initialize_with_shared_weights_retained<F, P>(
        self,
        weights: &SharedProgramWeights<R>,
        host: &HostMemoryAccount,
        static_policy: StaticInitializationPolicy,
        owned_phase: P,
        resolve_plan: F,
    ) -> crate::ExecutorBuildResult<CausalProgramSession<R>>
    where
        P: Send + 'static,
        F: FnOnce(
            &ResolvedRuntimePolicy,
            &CapabilityCatalog,
            &ProgramPlanCompilation,
        ) -> Result<ResolvedModelPlan>,
    {
        self.initialize_retained(
            StaticInstallation::AttachShared(weights),
            host,
            static_policy,
            owned_phase,
            resolve_plan,
        )
        .map(|(session, _)| session)
    }

    fn initialize_retained<F, P>(
        self,
        installation: StaticInstallation<'_, R>,
        host: &HostMemoryAccount,
        static_policy: StaticInitializationPolicy,
        owned_phase: P,
        resolve_plan: F,
    ) -> crate::ExecutorBuildResult<(CausalProgramSession<R>, Option<SharedProgramWeights<R>>)>
    where
        P: Send + 'static,
        F: FnOnce(
            &ResolvedRuntimePolicy,
            &CapabilityCatalog,
            &ProgramPlanCompilation,
        ) -> Result<ResolvedModelPlan>,
    {
        let custody = phase_custody::PhaseCustody::new(owned_phase);
        let result = (|| -> crate::ExecutorBuildResult<_> {
            let plan = resolve_plan(&self.policy, &self.composition.catalog, &self.compilation)?;
            if plan.execution_plan() != self.compilation.executable().execution_plan() {
                return Err(FerrumError::model(
                    "causal resolution changed the compiled execution plan",
                )
                .into());
            }
            self.prepared.validate_resolved_plan(&plan)?;
            static_installation::validate_static_installation(
                self.composition,
                self.prepared,
                &plan,
                &installation,
            )?;
            let terminal = self
                .terminal
                .as_ref()
                .map(|terminal| {
                    ResolvedCausalTerminal::resolve(&plan, terminal, self.prepared.facts())
                })
                .transpose()?;
            let providers = self
                .composition
                .registry
                .bind_plan(&plan)
                .map_err(|e| FerrumError::model(format!("causal provider binding: {e}")))?;
            // Resolve and validate the bounded catalog before any native
            // installation. Its physical lane is acquired lazily by real work.
            let replay = replay::CausalReplay::from_plan(&plan)?;
            let run_id = RunId::new(format!("run.causal.{}", uuid::Uuid::new_v4()))
                .map_err(|e| FerrumError::internal(e.to_string()))?;
            let initialized = static_installation::initialize_program_resources(
                self.composition,
                self.prepared,
                &plan,
                installation,
                Some(host),
                static_policy,
                &run_id,
                Some(custody.native_started()),
            )?;
            let session = CausalProgramSession {
                plan,
                providers,
                resources: initialized.plan_resources,
                host: host.clone(),
                terminal,
                shared_receipt: initialized.shared_weights_receipt,
                _static_attribution: initialized.static_provider_attribution,
                vocabulary_size: self.prepared.facts().vocabulary_size(),
                replay,
            };
            Ok((session, initialized.published_weights))
        })();
        match result {
            Ok((session, weights)) => {
                custody.handoff_to_plan(&session.resources);
                Ok((session, weights))
            }
            Err(error) => {
                custody.finish_error(&error);
                Err(error)
            }
        }
    }
}

impl<R: DeviceRuntime> CausalProgramSession<R> {
    pub fn plan(&self) -> &ResolvedModelPlan {
        &self.plan
    }
    pub fn host_account(&self) -> &HostMemoryAccount {
        &self.host
    }
    pub fn shared_weights_receipt(&self) -> Option<&ProgramSharedWeightsReceipt> {
        self.shared_receipt.as_ref()
    }
    pub fn resources(&self) -> &Arc<PlanRuntimeResources<R>> {
        &self.resources
    }
    pub fn installation_only(&self) -> bool {
        self.terminal.is_none()
    }

    /// Bind the original admitted main-model visual/position inputs once.
    /// Draft hidden inputs remain per-wave inputs, never a fabricated immutable
    /// future table. Core checks the complete actual conditioning layout.
    pub fn bind_conditioning_inputs(
        &self,
        session: &SequenceSession<R>,
        inputs: &CausalRequestInputs,
    ) -> Result<()> {
        let plan = self
            .resources
            .trusted_runtime_binding()
            .map_err(|e| FerrumError::model(e.to_string()))?;
        if session.resources().plan_evidence() != plan.evidence() {
            return Err(FerrumError::model(
                "conditioning session belongs to another initialized plan",
            ));
        }
        let inputs = inputs.sequence_conditioning(&self.host)?;
        session
            .bind_conditioning_inputs(Arc::new(inputs))
            .map_err(|e| FerrumError::model(e.to_string()))
    }

    pub fn close(
        self,
    ) -> std::result::Result<PlanRuntimeCloseOutcome<R>, PlanRuntimeCloseFailure<R>> {
        tracing::debug!(
            target: "ferrum.causal.replay",
            plan = %self.plan.execution_plan().plan_hash(),
            replay = %self.reusable_execution_snapshot(),
            "retiring causal program"
        );
        let Self {
            resources, replay, ..
        } = self;
        drop(replay);
        PlanRuntimeResources::close(resources)
    }
}
