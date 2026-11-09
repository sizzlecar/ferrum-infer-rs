//! A finite source-bound cohort, before native resources are acquired.
use super::*;

pub(crate) struct PreparedExpansionCohort {
    pub(in crate::image_generation::p_h3::expansion) requests: Vec<PreparedTextExpansion>,
    pub(in crate::image_generation::p_h3::expansion) engine: EngineConfig,
}

impl PreparedExpansionCohort {
    pub(crate) fn len(&self) -> usize {
        self.requests.len()
    }

    /// Bind each original visual request after its producer has retired.
    /// Initialization still validates every request against its source.
    pub(crate) fn requests_mut(&mut self) -> &mut [PreparedTextExpansion] {
        &mut self.requests
    }

    /// Move surviving preparations into a newly validated cohort after
    /// cancellation or a recoverable visual-input rejection.
    pub(crate) fn into_requests(self) -> Vec<PreparedTextExpansion> {
        self.requests
    }

    /// Preserve the original text or visual single-request contract exactly.
    pub(in crate::image_generation::p_h3::expansion) fn single(
        prepared: PreparedTextExpansion,
    ) -> Self {
        Self {
            engine: prepared.engine.clone(),
            requests: vec![prepared],
        }
    }
}

impl PreparedTextExpansion {
    /// Keep the original input owner across an initialization failure whose
    /// native retirement is unresolved. This does not duplicate its payload.
    pub(crate) fn retained_causal_inputs(
        &self,
    ) -> Option<Arc<ferrum_execution::CausalRequestInputs>> {
        self.causal_inputs.as_ref().map(Arc::clone)
    }

    /// The actual persistent visual input claim, with this account's padding.
    /// This query neither reserves memory nor binds request inputs.
    pub(crate) fn retained_visual_input_bytes(&self, host: &HostMemoryAccount) -> Result<usize> {
        self.visual
            .as_ref()
            .map_or(Ok(0), |visual| visual.retained_input_bytes(host))
    }
}

impl NativeTextExpansion {
    /// Forward the pipeline's explicit profile selection to its separately
    /// owned ordinary causal executor. This changes neither model policy nor
    /// runtime allowances. MTP has a different executor and is not instrumented
    /// by this diagnostic.
    pub(crate) fn configure_request_timing(
        &self,
        request: &mut PreparedTextExpansion,
        mode: ferrum_runtime::DeviceTimingMode,
    ) -> Result<()> {
        self.validate_preparation(request)?;
        if !matches!(
            mode,
            ferrum_runtime::DeviceTimingMode::Off | ferrum_runtime::DeviceTimingMode::Kernel
        ) || (mode != ferrum_runtime::DeviceTimingMode::Off
            && self.mtp != PH3ImagePromptExpansionMtp::Off)
        {
            return Err(FerrumError::unsupported(
                "prompt expansion timing supports Off or explicit Kernel with MTP Off",
            ));
        }
        request.device_timing = mode;
        Ok(())
    }

    /// Compare borrowed preparation metadata before taking any request tokens
    /// into a finite cohort. This neither reserves capacity nor admits work.
    pub(crate) fn text_cohort_compatible(
        &self,
        first: &PreparedTextExpansion,
        other: &PreparedTextExpansion,
    ) -> Result<bool> {
        if [first, other]
            .iter()
            .any(|request| request.visual.is_some() || request.causal_inputs.is_some())
        {
            return Ok(false);
        }
        self.cohort_compatible(first, other)
    }

    /// Compare metadata for homogeneous text or visual requests. Unbound
    /// visual plans may be grouped before their independently owned producers
    /// run; bound plans must each retain their matching immutable input owner.
    pub(crate) fn cohort_compatible(
        &self,
        first: &PreparedTextExpansion,
        other: &PreparedTextExpansion,
    ) -> Result<bool> {
        for request in [first, other] {
            if self.validate_cohort_request(request).is_err()
                || request.engine.runtime.startup_memory_plan.is_some()
            {
                return Ok(false);
            }
        }
        self.same_cohort_contract(first, other)
    }

    /// Freeze compatible text requests into one bounded execution plan. This
    /// grants neither Host capacity nor native sequence admission. Visual
    /// requests retain their existing independent single-request path.
    pub(crate) fn prepare_text_cohort(
        &self,
        requests: Vec<PreparedTextExpansion>,
    ) -> Result<PreparedExpansionCohort> {
        for request in &requests {
            self.validate_preparation(request)?;
            if request.visual.is_some() || request.causal_inputs.is_some() {
                return Err(FerrumError::unsupported(
                    "multi-request prompt expansion requires text-only input schemas",
                ));
            }
        }
        self.prepare_cohort(requests)
    }

    /// Freeze one original source's compatible text-only or visual-only group.
    /// No input bytes, Host capacity or native resources are acquired here.
    pub(crate) fn prepare_cohort(
        &self,
        requests: Vec<PreparedTextExpansion>,
    ) -> Result<PreparedExpansionCohort> {
        let first = requests
            .first()
            .ok_or_else(|| FerrumError::request_validation("empty prompt-expansion cohort"))?;
        u32::try_from(requests.len()).map_err(|_| {
            FerrumError::request_validation("expansion cohort exceeds native sequence capacity")
        })?;
        let mut engine = first.engine.clone();
        let mut context = 0usize;
        let mut chunk = 0usize;
        let mut output = 0usize;
        for request in &requests {
            self.validate_cohort_request(request)?;
            if !self.same_cohort_contract(first, request)? {
                return Err(FerrumError::request_validation(
                    "prompt-expansion cohort source, profile, input schema or execution policies differ",
                ));
            }
            let ceiling = request
                .tokens
                .len()
                .checked_add(request.generation.max_new_tokens as usize)
                .ok_or_else(|| FerrumError::config("cohort context overflows"))?;
            context = context.max(ceiling);
            chunk = chunk
                .checked_add(request.prefill_chunk_tokens)
                .ok_or_else(|| FerrumError::config("cohort token-wave extent overflows"))?;
            output = output.max(request.generation.max_new_tokens as usize);
        }
        if requests.len() > 1 && engine.runtime.startup_memory_plan.is_some() {
            return Err(FerrumError::request_validation(
                "a request-specific startup memory plan cannot authorize another cohort",
            ));
        }
        engine.runtime.max_model_len = Some(context);
        engine.scheduler.max_running_requests = requests.len();
        engine.scheduler.max_waiting_requests = requests.len();
        engine.batching.max_num_batched_tokens = chunk;
        if self.mtp == PH3ImagePromptExpansionMtp::OneDraft {
            engine.batching.max_num_batched_tokens = chunk.max(
                requests
                    .len()
                    .checked_mul(2)
                    .ok_or_else(|| FerrumError::config("MTP verify batch extent overflows"))?,
            );
        }
        engine.sampling.default_params.max_tokens = output;
        Ok(PreparedExpansionCohort { requests, engine })
    }

    fn validate_cohort_request(&self, request: &PreparedTextExpansion) -> Result<()> {
        self.validate_preparation(request)?;
        if !valid_geometry(request) {
            return Err(FerrumError::request_validation(
                "cohort request differs from its prepared context or chunk limit",
            ));
        }
        let model = self.selected_model(request)?.prepared();
        if !matches!(
            &request.engine.numerical_execution,
            ferrum_types::NumericalExecutionPolicy::Require(id)
                if id == &model.family().numerical_profile().id
        ) {
            return Err(FerrumError::request_validation(
                "cohort policy differs from its original prepared numerical profile",
            ));
        }
        if let Some(visual) = &request.visual {
            if visual.tokens().maximum_tokens() != request.engine.runtime.max_model_len.unwrap()
                || !request.tokens.iter().map(|id| id.get()).eq(visual
                    .tokens()
                    .tokens()
                    .iter()
                    .copied())
            {
                return Err(FerrumError::request_validation(
                    "visual cohort tokens differ from their original input plan",
                ));
            }
            if request.causal_inputs.is_some() {
                self.validate_bound_inputs(request)?;
            }
        } else if request.causal_inputs.is_some() {
            return Err(FerrumError::request_validation(
                "text cohort request carries an undeclared causal input owner",
            ));
        }
        Ok(())
    }

    fn same_cohort_contract(
        &self,
        first: &PreparedTextExpansion,
        other: &PreparedTextExpansion,
    ) -> Result<bool> {
        let schema_matches = match (&first.visual, &other.visual) {
            (None, None) => true,
            (Some(first), Some(other)) => first.schema() == other.schema(),
            _ => false,
        };
        // Source domain validation above and this exact frozen model identity
        // bind the family/profile to the original source, not just its name.
        Ok(schema_matches
            && first.device_timing == other.device_timing
            && first.causal_inputs.is_some() == other.causal_inputs.is_some()
            && std::ptr::eq(
                self.selected_model(first)?.prepared(),
                self.selected_model(other)?.prepared(),
            )
            && request_independent_policy(&first.engine)?
                == request_independent_policy(&other.engine)?)
    }
}

fn valid_geometry(request: &PreparedTextExpansion) -> bool {
    !request.tokens.is_empty()
        && request.prefill_chunk_tokens > 0
        && request.generation.max_new_tokens > 0
        && request
            .tokens
            .len()
            .checked_add(request.generation.max_new_tokens as usize)
            .is_some_and(|ceiling| request.engine.runtime.max_model_len == Some(ceiling))
}

fn request_independent_policy(engine: &EngineConfig) -> Result<serde_json::Value> {
    // EngineConfig has no PartialEq. Compare its complete declared product
    // configuration, excluding only geometry already validated per request.
    // Unknown future policy fields therefore remain part of compatibility.
    let mut policy = engine.clone();
    policy.runtime.max_model_len = None;
    policy.batching.max_num_batched_tokens = 0;
    policy.sampling.default_params.max_tokens = 0;
    serde_json::to_value(policy).map_err(model_error)
}
