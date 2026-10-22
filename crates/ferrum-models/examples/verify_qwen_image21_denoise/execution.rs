//! One compilation, static initialization and executor for all Euler intervals.
use super::{
    artifacts,
    inputs::{self, HostBudget, Inputs},
    shared::{self, id, StoredComponentRuntime, TensorPolicy},
};
use anyhow::{ensure, Context, Result};
use ferrum_diffusion::{Bf16EulerProfile, FlowMatchEulerSchedule, ManagedBf16Euler};
use ferrum_execution::{PreparedTensorExecutor, TensorExecutorConfig};
use ferrum_models::qwen_image21::{
    QwenImage21ModelTimestep, QwenImage21Timestep256, QwenImage21TransformerConfig,
    QwenImage21TransformerFamily as Family,
};
use ferrum_runtime::*;
use serde_json::{json, Value};

pub struct ResultArtifact {
    pub sample: ManagedHostTensor,
    pub provenance: Value,
    pub retirement: Value,
    pub steps: Vec<Value>,
}

pub fn execute<R: DeviceRuntime>(
    config: &QwenImage21TransformerConfig,
    family: &PreparedModelFamily,
    source: &dyn WeightComponentSource,
    composition: StoredComponentRuntime<'_, R>,
    host: &HostMemoryAccount,
    budget: &HostBudget,
    inputs: Inputs,
    schedule: FlowMatchEulerSchedule,
    mut inspect: impl FnMut(&Value, &ManagedHostTensor) -> Result<()>,
) -> Result<ResultArtifact> {
    ensure!(
        [&inputs.sample, &inputs.text, &inputs.frequencies]
            .iter()
            .all(|t| host.owns(t.bytes())),
        "all inputs must belong to the shared Host account"
    );
    let mut options = config.compile_options()?;
    options.require_weight_materializer(id(IDENTITY_WEIGHT_MATERIALIZER_ID)?);
    options.retain_completion_value(id(Family::VELOCITY_ID)?);
    let policy = TensorPolicy {
        capacity_bytes: composition.capacity_bytes,
        storage_profiles: composition
            .runtime
            .descriptor()
            .dynamic_storage_profiles
            .iter()
            .copied()
            .collect(),
    };
    let compiled = ProgramPlanCompiler::compile_with_weight_materializers(
        family,
        composition.catalog,
        &policy,
        &composition.registry.planning(),
        composition.materializers,
        &options,
    )?;
    let executable = compiled.executable();
    let plan = executable.execution_plan();
    let ports = plan
        .payload()
        .tensor_ports()
        .context("DiT plan has no tensor ports")?;
    ensure!(
        ports.outputs().len() == 1
            && ports.outputs()[0].value_id().as_str() == Family::VELOCITY_ID
            && ports.outputs()[0].tensor() == inputs.sample.tensor()
            && usize::try_from(ports.outputs()[0].tensor().minimum_storage_bytes()?)?
                == budget.velocity,
        "DiT must return exactly the Euler velocity shape"
    );
    let weight_bytes =
        family
            .stored_weight_schema()?
            .components
            .iter()
            .try_fold(0_u64, |sum, c| {
                sum.checked_add(c.physical_bytes()?)
                    .ok_or_else(|| anyhow::anyhow!("weight bytes overflow"))
            })?;
    // This is the only call which can read/upload static weights. Nothing in
    // the step loop holds a source or materializer entry point.
    let resources = shared::initialize_resources(
        composition.runtime.clone(),
        family,
        plan,
        source,
        weight_bytes,
        host,
        budget.staging,
    )?;
    let execution: Result<_> = (|| {
        let mut executor = PreparedTensorExecutor::new(
            executable.clone(),
            composition.registry.bind_plan(executable)?,
            resources.clone(),
            host.clone(),
            TensorExecutorConfig::default(),
        )
        .map_err(|e| anyhow::anyhow!("denoise executor: {e}"))?;
        let mut solver = ManagedBf16Euler::new(
            schedule,
            host.clone(),
            inputs.sample.tensor().clone(),
            Bf16EulerProfile::CudaScalarBf16StagedRneV1,
        )?;
        let adapter = QwenImage21Timestep256::new()?;
        let Inputs {
            mut sample,
            text,
            frequencies,
        } = inputs;
        let mut reports = Vec::new();
        while let Some(step) = solver.current_step() {
            let before = artifacts::sha256(sample.bytes().bytes());
            let time = inputs::time_input(config, host, &adapter, step.timestep().as_f32())?;
            let time_hash = artifacts::sha256(time.bytes().bytes());
            let started = std::time::Instant::now();
            let outputs = executor
                .execute(
                    id(format!("run.denoise.step.{}", step.index()))?,
                    id(format!("request.denoise.step.{}", step.index()))?,
                    vec![sample.clone(), text.clone(), time, frequencies.clone()],
                )
                .map_err(|e| {
                    anyhow::anyhow!(
                        "DiT step {} failed before Euler advancement: {e}",
                        step.index()
                    )
                })?;
            let [velocity] = outputs.as_slice() else {
                anyhow::bail!("unexpected denoiser output arity")
            };
            ensure!(
                velocity.value_id().as_str() == Family::VELOCITY_ID,
                "unexpected denoiser output ID"
            );
            inputs::finite(velocity.bytes().bytes()).with_context(|| {
                format!(
                    "nonfinite velocity at step {}; Euler not advanced",
                    step.index()
                )
            })?;
            let velocity_hash = artifacts::sha256(velocity.bytes().bytes());
            let (_, next) = solver
                .advance(&sample, velocity.tensor(), velocity.bytes())
                .with_context(|| format!("Euler step {} failed without advancing", step.index()))?;
            sample = next; // release old latent only after the complete update.
            drop(outputs);
            let usage = host.usage()?;
            let live = budget.inputs[0] + budget.inputs[1] + budget.inputs[3];
            ensure!(
                usage.committed_bytes() == live && usage.allocated_bytes() == live,
                "step completion retains a velocity/input backing beyond its declared lifetime"
            );
            let report = json!({"step":step.index(),"sigma":step.sigma(),"next_sigma":step.next_sigma(),
                "scheduler_timestep":step.timestep().as_f32(),"model_timestep_bf16":QwenImage21ModelTimestep::from_scheduler_timestep(step.timestep().as_f32())?.bf16_bits(),
                "time_features_sha256":time_hash,"input_latent_sha256":before,"velocity_sha256":velocity_hash,
                "latent_sha256":artifacts::sha256(sample.bytes().bytes()),"nonfinite_count":0,
                "completed_steps":solver.step_index(),"host_committed_after_step":usage.committed_bytes(),
                "host_allocated_after_step":usage.allocated_bytes(),"cold_step_seconds":started.elapsed().as_secs_f64(),
                "retirement":"synchronous executor completion; velocity and old latent owners released; static plan remains live"});
            inspect(&report, &sample)?;
            reports.push(report);
        }
        ensure!(solver.is_complete(), "incomplete schedule");
        let provenance = json!({"family":family.family_id(),"profile":family.numerical_profile().id,"canonical_config":family.canonical_config(),
            "family_fingerprint":family.fingerprint()?,"plan_hash":plan.plan_hash(),"node_count":plan.payload().nodes().len(),
            "runtime_fingerprint":composition.runtime.descriptor().runtime_implementation_fingerprint,"backend":composition.provenance,
            "initialized_weight_bytes":weight_bytes,"static_initializations":1,"executors":1,"euler_profile":solver.profile().id(),
            "host_budget":budget,"memory_scope":"managed inputs, velocity, two latent backings and static staging; not total RSS or qualified opaque CUDA memory"});
        Ok((sample, provenance, reports))
    })();
    let closed = match PlanRuntimeResources::close(resources) {
        Ok(PlanRuntimeCloseOutcome::Closed(receipt)) => Ok(receipt.released_static_resources()),
        Ok(PlanRuntimeCloseOutcome::Referenced { strong_count, .. }) => Err(anyhow::anyhow!(
            "denoise plan retains {strong_count} owners"
        )),
        Err(e) => Err(anyhow::anyhow!(
            "denoise retirement failed: {:?}",
            e.failure()
        )),
    };
    match (execution, closed) {
        (Ok((sample, provenance, steps)), Ok(released)) => {
            let usage = host.usage()?;
            ensure!(
                usage.committed_bytes() == budget.inputs[0]
                    && usage.allocated_bytes() == budget.inputs[0],
                "only final latent may remain after plan retirement"
            );
            Ok(ResultArtifact {
                sample,
                provenance,
                steps,
                retirement: json!({"released_static_resources":released,
                "host_committed_after_close":usage.committed_bytes(),"host_allocated_after_close":usage.allocated_bytes(),"remaining_owner":"final normalized packed latent"}),
            })
        }
        (Err(e), Ok(_)) | (Ok(_), Err(e)) => Err(e),
        (Err(e), Err(close)) => Err(e.context(format!("retirement also failed: {close:#}"))),
    }
}
