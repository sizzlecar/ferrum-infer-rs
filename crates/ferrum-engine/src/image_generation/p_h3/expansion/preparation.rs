//! Physical backing requirements over the actual compiled causal plan. These
//! bounds exclude tokenizer/parser heap and existing executor token metadata.
use super::*;
use crate::image_generation::host_budget::{stored_staging_bytes, ExecutionHostBudget};
use ferrum_execution::{ManagedCausalIoWork, ManagedCausalReadback};
use ferrum_runtime::{ExecutablePlanView, HostBackingPreparation, HostMemoryClass};

pub(crate) struct ExpansionHostRequirements {
    pub(crate) budget: ExecutionHostBudget,
    // Preserve materializer assumptions through actual initialization.
    _materializer: ExpansionBacking,
    pub(crate) startup_device_bytes: u64,
}
enum ExpansionBacking {
    Ordinary(HostBackingPreparation),
    Mtp(super::mtp_cohort::initialization::MtpHostRequirements),
}
impl ExpansionHostRequirements {
    /// Original backing assumptions retained across sequential stages. The
    /// remaining WeightBacking budget is that stage's conversion peak.
    pub(crate) fn materializer_additional_bytes(&self) -> usize {
        match &self._materializer {
            ExpansionBacking::Ordinary(backing) => backing.additional_bytes(),
            ExpansionBacking::Mtp(backing) => backing.materializer_additional_bytes(),
        }
    }

    pub(crate) fn claims(&self) -> [(HostMemoryClass, usize); 4] {
        self.budget.claims()
    }
}

impl NativeTextExpansion {
    pub(crate) fn host_requirements<R: DeviceRuntime>(
        &self,
        composition: &RuntimeComposition<R>,
        prepared: &PreparedTextExpansion,
        host: &HostMemoryAccount,
    ) -> Result<ExpansionHostRequirements> {
        self.execution_host_requirements(
            composition,
            std::slice::from_ref(prepared),
            &prepared.engine,
            host,
        )
    }

    pub(crate) fn cohort_host_requirements<R: DeviceRuntime>(
        &self,
        composition: &RuntimeComposition<R>,
        prepared: &super::cohort::PreparedExpansionCohort,
        host: &HostMemoryAccount,
    ) -> Result<ExpansionHostRequirements> {
        self.execution_host_requirements(composition, &prepared.requests, &prepared.engine, host)
    }

    fn execution_host_requirements<R: DeviceRuntime>(
        &self,
        composition: &RuntimeComposition<R>,
        requests: &[PreparedTextExpansion],
        engine: &EngineConfig,
        host: &HostMemoryAccount,
    ) -> Result<ExpansionHostRequirements> {
        if self.mtp == PH3ImagePromptExpansionMtp::OneDraft {
            let assets = super::mtp_assets::MtpExpansionAssets::from_native(self)?;
            let mut engine = engine.clone();
            engine.batching.max_num_batched_tokens = engine.batching.max_num_batched_tokens.max(
                requests
                    .len()
                    .checked_mul(2)
                    .ok_or_else(|| FerrumError::config("MTP batch extent overflows"))?,
            );
            let requirements = super::mtp_cohort::initialization::host_requirements(
                self,
                &assets,
                composition,
                requests,
                &engine,
                host,
            )?;
            return Ok(ExpansionHostRequirements {
                budget: requirements.budget(),
                startup_device_bytes: requirements.startup_device_bytes(),
                _materializer: ExpansionBacking::Mtp(requirements),
            });
        }
        let prepared = requests
            .first()
            .ok_or_else(|| FerrumError::config("empty expansion execution"))?;
        for request in requests {
            self.validate_preparation(request)?;
        }
        let source = self.selected_model(prepared)?.weight_source(host);
        let compiled = self.compile_with_engine(composition, &source, prepared, engine)?;
        let plan = compiled.compilation().executable().execution_plan();
        self.validate_static_source_policy(&source, plan)?;
        let materializer = compiled.prepare_static_host_backing()?;
        let conversion = source
            .maximum_group_backing_bytes(plan)
            .map_err(model_error)?;
        let query = |decode: bool| {
            let work = requests
                .iter()
                .map(|request| ManagedCausalIoWork {
                    immediate_tokens: if decode {
                        1
                    } else {
                        request.prefill_chunk_tokens
                    },
                    mask_upload: true,
                    output: if decode {
                        ManagedCausalReadback::GreedyToken
                    } else {
                        if self.is_bf16() {
                            ManagedCausalReadback::Bf16Logits
                        } else {
                            ManagedCausalReadback::F16Logits
                        }
                    },
                })
                .collect::<Vec<_>>();
            compiled.managed_io_requirements(host, self.managed_io_profile(), &work)
        };
        let prefill = query(false)?;
        let decode = query(true)?;
        // One wave retains I/O for every participant. Each prefill chunk is at
        // least one token, and the model vocabulary/readback type is shared, so
        // these full-cohort bounds also cover mixed prefill/decode waves.
        let input = prefill.input_bytes.max(decode.input_bytes);
        let wave_output = prefill.output_bytes.max(decode.output_bytes);
        let mut history = 0usize;
        let mut retained_input = 0usize;
        for request in requests {
            history = history
                .checked_add(
                    execute::history_backing_bytes(host, request.generation.max_new_tokens)
                        .map_err(model_error)?,
                )
                .ok_or_else(|| FerrumError::config("cohort token histories overflow"))?;
            retained_input = retained_input
                .checked_add(
                    request
                        .visual
                        .as_ref()
                        .map(|v| v.retained_input_bytes(host))
                        .transpose()?
                        .unwrap_or(0),
                )
                .ok_or_else(|| FerrumError::config("visual prompt Input backing overflows"))?;
        }
        let budget = ExecutionHostBudget {
            input: input
                .checked_add(retained_input)
                .ok_or_else(|| FerrumError::config("visual prompt Input backing overflows"))?,
            output: history
                .checked_add(wave_output)
                .ok_or_else(|| FerrumError::config("prompt Output backing overflows"))?,
            staging: if composition.runtime().requires_managed_static_staging() {
                stored_staging_bytes(
                    host,
                    plan.payload().execution_weights(),
                    compiled.static_initialization_policy(),
                )
                .map_err(model_error)?
            } else {
                0
            },
            weights: conversion
                .checked_add(materializer.additional_bytes())
                .ok_or_else(|| FerrumError::config("prompt WeightBacking overflows"))?,
        };
        let context_tokens = engine.runtime.max_model_len.expect("prepared context");
        let startup_device_bytes = if requests.len() == 1 {
            compiled
                .startup_peak_bytes(ferrum_types::StartupWorkload::Prefill {
                    context_tokens,
                    chunk_tokens: prepared.prefill_chunk_tokens,
                })?
                .max(
                    compiled.startup_peak_bytes(ferrum_types::StartupWorkload::Decode {
                        context_tokens,
                        active_sequences: 1,
                    })?,
                )
        } else {
            // Unlike the shallow Decode startup probe, this evaluates the
            // real compiled physical plan with every admitted sequence at
            // the full shared context ceiling. Provider workspace and
            // allocator padding remain part of the same plan's formula.
            plan.payload()
                .memory()
                .startup_peak_bytes(
                    u64::try_from(context_tokens).map_err(model_error)?,
                    u32::try_from(requests.len()).map_err(model_error)?,
                    u64::try_from(engine.batching.max_num_batched_tokens).map_err(model_error)?,
                )
                .map_err(model_error)?
        };
        Ok(ExpansionHostRequirements {
            budget,
            _materializer: ExpansionBacking::Ordinary(materializer),
            startup_device_bytes,
        })
    }
}
