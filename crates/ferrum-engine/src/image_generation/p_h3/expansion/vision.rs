//! One real HF-presented visual producer on the original runtime composition.
//! This module is not a service phase: callers still own admission, cancellation
//! and the handoff to the causal executor. Success requires an actual close ack.
use super::visual_assets::ExpansionVisualAssets;
use crate::image_generation::host_budget::{stored_staging_bytes, sum_ports, ExecutionHostBudget};
use anyhow::{ensure, Context, Result};
use ferrum_execution::{
    component::{
        CompiledTensorComponent, ComponentInitializationConfig, ComponentWeightSource,
        RetainedComponentCloseError, RetainedComponentInitializationError,
        RetainedTensorComponentSession, TensorComponentPolicy,
    },
    RuntimeComposition, TensorExecutionError, TensorExecutionOutput, TensorExecutorConfig,
    TensorSerialActor,
};
use ferrum_models::{
    qwen3_vl_vision::{
        hf::{
            Qwen3VLVisionHfFamily, Qwen3VLVisionHfImagePlan, Qwen3VLVisionHfLimits,
            Qwen3VLVisionHfProcessor,
        },
        Qwen3VLVisionFamily,
    },
    qwen_image21::image_input::ManagedRgba8Image,
    vnext::ManagedProductionModel,
};
use ferrum_quantization::TensorArchive;
use ferrum_runtime::*;
use std::{fmt, sync::Arc};

pub(crate) struct ExpansionVisionOptions {
    pub limits: Qwen3VLVisionHfLimits,
    pub profile: NumericalProfileId,
    pub device_capacity_bytes: u64,
    pub static_policy: StaticInitializationPolicy,
    pub executor: TensorExecutorConfig,
}

pub(crate) struct PreparedExpansionVision<R: DeviceRuntime> {
    plan: Qwen3VLVisionHfImagePlan,
    assets: Arc<ExpansionVisualAssets>,
    component: CompiledTensorComponent<R>,
    runtime: Arc<R>,
    static_policy: StaticInitializationPolicy,
    executor: TensorExecutorConfig,
}

/// Per-class simultaneous peaks, excluding the caller's borrowed decoded RGBA
/// and the later causal F16 bridge. The original materializer assumption must
/// remain alive through static initialization, just like the causal query.
pub(crate) struct ExpansionVisionHostRequirements {
    pub budget: ExecutionHostBudget,
    pub startup_device_bytes: u64,
    _materializer: HostBackingPreparation,
}

impl ExpansionVisionHostRequirements {
    /// Original retained backing, separate from this producer's temporary
    /// conversion peak when several visual stages execute in sequence.
    pub(crate) fn materializer_additional_bytes(&self) -> usize {
        self._materializer.additional_bytes()
    }
}

/// The only successful output is main F32. The same original phase grant is
/// returned for the causal handoff, and the receipt proves device retirement.
pub(crate) struct CompletedExpansionVision {
    output: TensorExecutionOutput,
    phase: HostMemoryScope,
    close: PlanRuntimeCloseReceipt,
    request: RequestIdentity,
}
impl CompletedExpansionVision {
    pub(crate) fn output(&self) -> &TensorExecutionOutput {
        &self.output
    }
    pub(crate) fn request(&self) -> &RequestIdentity {
        &self.request
    }
    pub(crate) fn into_parts(
        self,
    ) -> (
        TensorExecutionOutput,
        HostMemoryScope,
        PlanRuntimeCloseReceipt,
    ) {
        (self.output, self.phase, self.close)
    }
}

/// Every post-admission failure retains the original phase, plus the actual
/// initializer journal/session/error/close owner. No formatted error replaces
/// recovery custody, and no Drop path is represented as synchronized success.
#[must_use = "retain the original visual phase and recovery owners until explicit retirement"]
pub(crate) enum ExpansionVisionFailure<R: DeviceRuntime> {
    Preflight {
        source: anyhow::Error,
        phase: HostMemoryScope,
    },
    Initialization {
        source: RetainedComponentInitializationError<R, ExpansionVisionOwnedPhase>,
    },
    Execution {
        source: TensorExecutionError<R>,
        session: RetainedTensorComponentSession<R, ExpansionVisionOwnedPhase>,
    },
    Output {
        source: anyhow::Error,
        session: RetainedTensorComponentSession<R, ExpansionVisionOwnedPhase>,
        outputs: Vec<TensorExecutionOutput>,
    },
    Close {
        source: RetainedComponentCloseError<R, ExpansionVisionOwnedPhase>,
        output: TensorExecutionOutput,
    },
}
impl<R: DeviceRuntime> fmt::Display for ExpansionVisionFailure<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Preflight { source, .. } => write!(f, "upsampler vision preflight: {source:#}"),
            Self::Initialization { source, .. } => {
                write!(f, "upsampler vision initialization: {source}")
            }
            Self::Execution { source, .. } => write!(f, "upsampler vision execution: {source}"),
            Self::Output { source, .. } => write!(f, "upsampler vision output: {source:#}"),
            Self::Close { source, .. } => {
                write!(f, "upsampler vision close: {source}")
            }
        }
    }
}
impl<R: DeviceRuntime> fmt::Debug for ExpansionVisionFailure<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}
impl<R: DeviceRuntime> std::error::Error for ExpansionVisionFailure<R> {}

pub(crate) type ExpansionVisionOwnedPhase = (HostMemoryScope, HostBackingPreparation);

impl<R: DeviceRuntime> PreparedExpansionVision<R> {
    /// All inputs are immutable prepared metadata. The source is the exact
    /// already-verified mmap: there is deliberately no filesystem path input.
    pub(crate) fn compile(
        composition: &RuntimeComposition<R>,
        model: &ManagedProductionModel,
        processor: &Qwen3VLVisionHfProcessor,
        source_width: u32,
        source_height: u32,
        options: ExpansionVisionOptions,
    ) -> Result<Self> {
        let assets = Arc::new(ExpansionVisualAssets::from_processor_with_profile(
            model,
            processor.clone(),
            options.limits,
            options.profile.clone(),
        )?);
        Self::compile_assets(composition, assets, source_width, source_height, options)
    }
    /// Reuse one immutable, fully validated source across request dimensions.
    pub(crate) fn compile_assets(
        composition: &RuntimeComposition<R>,
        assets: Arc<ExpansionVisualAssets>,
        source_width: u32,
        source_height: u32,
        options: ExpansionVisionOptions,
    ) -> Result<Self> {
        ensure!(
            options.device_capacity_bytes > 0,
            "positive visual device capacity required"
        );
        let limits = assets.limits();
        ensure!(
            &options.profile == assets.profile()
                && options.limits.max_metadata_bytes == limits.max_metadata_bytes
                && options.limits.max_plan_nodes == limits.max_plan_nodes
                && options.limits.max_stored_bytes == limits.max_stored_bytes,
            "visual compilation differs from frozen asset profile/limits"
        );
        let plan = assets.plan(source_width, source_height)?;
        let family = TypedFamilyRegistration::new(Qwen3VLVisionHfFamily::new(options.limits)?)
            .prepare_with_profile(&serde_json::to_value(plan.config())?, &options.profile)?;
        let descriptor = composition.runtime().descriptor();
        let policy = TensorComponentPolicy::new(
            options
                .device_capacity_bytes
                .min(descriptor.total_memory_bytes),
            descriptor
                .dynamic_storage_profiles
                .iter()
                .copied()
                .collect(),
        )?;
        let component = composition.compile_tensor_component(
            family,
            &policy,
            &plan.config().compile_options()?,
        )?;
        let ports = component
            .executable()
            .execution_plan()
            .payload()
            .tensor_ports()
            .context("HF visual component lacks actual tensor ports")?;
        ensure!(
            ports.inputs().len() == 4 && ports.outputs().len() == 1,
            "HF vision must have four actual inputs and one main output"
        );
        let output = &ports.outputs()[0];
        ensure!(
            output.value_id().as_str() == Qwen3VLVisionHfFamily::OUTPUT_ID
                && output.tensor().element_type() == ElementType::F32
                && output.tensor().dimensions()
                    == [
                        plan.config().merged_rows()?,
                        plan.config().architecture().output_size()
                    ],
            "HF visual compiled main output differs from geometry"
        );
        Ok(Self {
            plan,
            assets,
            component,
            runtime: composition.runtime_handle(),
            static_policy: options.static_policy,
            executor: options.executor,
        })
    }
    pub(crate) fn plan(&self) -> &Qwen3VLVisionHfImagePlan {
        &self.plan
    }
    pub(crate) fn component(&self) -> &CompiledTensorComponent<R> {
        &self.component
    }
    /// Pointer identity is deliberately stricter than equal model names or
    /// paths: this binds the original frozen source, processor and profile.
    pub(crate) fn same_reuse_domain(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.assets, &other.assets)
            && Arc::ptr_eq(&self.runtime, &other.runtime)
            && self.component.family().family_id() == other.component.family().family_id()
            && self.component.family().canonical_config()
                == other.component.family().canonical_config()
            && self.component.family().numerical_profile()
                == other.component.family().numerical_profile()
    }
    pub(crate) fn host_requirements(
        &self,
        host: &HostMemoryAccount,
    ) -> Result<ExpansionVisionHostRequirements> {
        let plan = self.component.executable().execution_plan();
        let ports = plan
            .payload()
            .tensor_ports()
            .context("HF visual ports disappeared")?;
        let source = self.assets.weights().weight_source(host);
        let materializer = plan.prepare_host_backing(self.component.family(), Some(&source))?;
        let conversion = self.assets.weights().maximum_group_backing_bytes(
            self.component.family(),
            plan,
            host,
        )?;
        let budget = ExecutionHostBudget {
            input: self
                .plan
                .input_backing_bytes(host)?
                .max(sum_ports(host, ports.inputs().iter())?),
            output: sum_ports(host, ports.outputs().iter())?,
            staging: if self.runtime.requires_managed_static_staging() {
                stored_staging_bytes(host, plan.payload().execution_weights(), self.static_policy)?
            } else {
                0
            },
            weights: conversion
                .checked_add(materializer.additional_bytes())
                .context("HF visual WeightBacking capacity overflow")?,
        };
        let memory = plan.payload().memory();
        let startup_device_bytes = memory
            .static_bytes()
            .checked_add(memory.minimum_runnable_request_bytes())
            .context("HF visual startup capacity overflow")?;
        Ok(ExpansionVisionHostRequirements {
            budget,
            startup_device_bytes,
            _materializer: materializer,
        })
    }
    /// The caller supplies its existing actor and transfers its actual phase
    /// grant. Success returns that grant only after the original session closes.
    pub(crate) fn execute_on_actor(
        &self,
        phase: HostMemoryScope,
        original: &ManagedRgba8Image,
        actor: &mut TensorSerialActor<R>,
        run_id: RunId,
        request_id: RequestIdentity,
        transaction_id: TransactionId,
    ) -> std::result::Result<CompletedExpansionVision, ExpansionVisionFailure<R>> {
        let host = phase.account();
        let inputs = (|| -> Result<Vec<ManagedHostTensor>> {
            let patches = self.plan.managed_patches(&host, original)?;
            let table = TensorArchive::tensor(
                self.assets.weights().source().archive(),
                Qwen3VLVisionFamily::POSITION_EXTERNAL_NAME,
            )?;
            let positions = self.plan.config().managed_position_inputs(&host, &table)?;
            let mut inputs = vec![patches];
            inputs.extend(positions);
            Ok(inputs)
        })();
        let inputs = match inputs {
            Ok(inputs) => inputs,
            Err(source) => return Err(ExpansionVisionFailure::Preflight { source, phase }),
        };
        let source = self.assets.weights().weight_source(&host);
        let backing = match self
            .component
            .executable()
            .execution_plan()
            .prepare_host_backing(self.component.family(), Some(&source))
        {
            Ok(backing) => backing,
            Err(error) => {
                return Err(ExpansionVisionFailure::Preflight {
                    source: error.into(),
                    phase,
                })
            }
        };
        let initialized = self.component.initialize_retained(
            self.runtime.clone(),
            host.clone(),
            ComponentWeightSource::Stored(&source),
            ComponentInitializationConfig {
                run_id: run_id.clone(),
                request_id: request_id.clone(),
                transaction_id,
                static_policy: self.static_policy,
                executor: self.executor.clone(),
            },
            (phase, backing),
        );
        let mut session = match initialized {
            Ok(session) => session,
            Err(source) => return Err(ExpansionVisionFailure::Initialization { source }),
        };
        let mut outputs =
            match session
                .session_mut()
                .execute_on_actor(actor, run_id, request_id.clone(), inputs)
            {
                Ok(outputs) => outputs,
                Err(source) => return Err(ExpansionVisionFailure::Execution { source, session }),
            };
        let checked = (|| -> Result<()> {
            ensure!(
                outputs.len() == 1,
                "HF visual execution returned multiple outputs"
            );
            let output = &outputs[0];
            let expected = &self
                .component
                .executable()
                .execution_plan()
                .payload()
                .tensor_ports()
                .context("HF visual output contract disappeared")?
                .outputs()[0];
            ensure!(output.value_id()==expected.value_id() && output.tensor()==expected.tensor() && host.owns(output.bytes()) && output.bytes().class()==HostMemoryClass::Output,"actual HF main output differs from the submitted program or original Host authority");
            Ok(())
        })();
        if let Err(source) = checked {
            return Err(ExpansionVisionFailure::Output {
                source,
                session,
                outputs,
            });
        }
        let output = outputs.pop().unwrap();
        match session.close() {
            Ok(((phase, _backing), close)) => Ok(CompletedExpansionVision {
                output,
                phase,
                close,
                request: request_id,
            }),
            Err(source) => Err(ExpansionVisionFailure::Close { source, output }),
        }
    }
}

#[cfg(test)]
mod tests;
