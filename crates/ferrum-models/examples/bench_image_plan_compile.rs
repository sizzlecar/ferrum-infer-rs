//! CPU compilation of real Qwen Image graphs, without weights or device work.
//! Usage: bench_image_plan_compile TRANSFORMER_CONFIG LAYOUT_JSON [REPETITIONS]
//! The supplied layout is the typed QwenImage21T2iLayout wire representation.
//! Reference providers measure shared planner cost, not CUDA planning or inference.
use anyhow::{ensure, Context, Result};
use ferrum_execution::component::TensorComponentPolicy;
use ferrum_kernels::backend::reference::ReferenceVNextComposition;
use ferrum_models::qwen_image21::{
    QwenImage21Config, QwenImage21T2iLayout, QwenImage21TransformerConfig,
    QwenImage21TransformerFamily, QwenImage21TransformerPhase, StructureLimits,
};
use ferrum_runtime::{
    DeviceId, DeviceRuntime, ExecutablePlanView, ModelFamilyRegistration, NumericalProfileId,
    ProgramPlanCompiler, TypedFamilyRegistration,
};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{io::Write, time::Instant};

fn main() -> Result<()> {
    let args = std::env::args_os().skip(1).collect::<Vec<_>>();
    ensure!(
        (2..=3).contains(&args.len()),
        "usage: bench_image_plan_compile TRANSFORMER_CONFIG LAYOUT_JSON [REPETITIONS]"
    );
    let repetitions = args
        .get(2)
        .map(|arg| -> Result<usize> {
            Ok(arg.to_str().context("repetitions must be UTF-8")?.parse()?)
        })
        .transpose()?
        .unwrap_or(3);
    ensure!(
        (1..=1000).contains(&repetitions),
        "repetitions must be 1..=1000"
    );
    let config_bytes = std::fs::read(&args[0])?;
    let layout_bytes = std::fs::read(&args[1])?;
    let architecture = QwenImage21Config::from_hf_json(&config_bytes, StructureLimits::default())?;
    let layout: QwenImage21T2iLayout = serde_json::from_slice(&layout_bytes)?;
    let config = QwenImage21TransformerConfig::new(architecture, layout)?;
    // A metadata-only ceiling; create/compile/bind never allocates this amount.
    let capacity = 1_u64 << 40;
    let profile = NumericalProfileId::new(QwenImage21TransformerFamily::REFERENCE_PROFILE_ID)
        .map_err(anyhow::Error::msg)?;
    let registration = TypedFamilyRegistration::new(QwenImage21TransformerFamily::new()?);
    let mut output = std::io::BufWriter::new(std::io::stdout().lock());
    serde_json::to_writer(
        &mut output,
        &json!({
            "scope": "reference-provider CPU compilation and binding only; no weights, device submission, CUDA timing or inference claim",
            "config_sha256": format!("{:x}", Sha256::digest(&config_bytes)),
            "layout_sha256": format!("{:x}", Sha256::digest(&layout_bytes)),
            "config": config,
            "profile": profile,
            "capacity_bytes": capacity,
            "repetitions": repetitions,
        }),
    )?;
    writeln!(output)?;
    for phase in [
        QwenImage21TransformerPhase::Full,
        QwenImage21TransformerPhase::PrefixPrefill,
        QwenImage21TransformerPhase::PrefixDecode,
    ] {
        let config = config.clone().with_execution_phase(phase)?;
        let raw_config = serde_json::to_value(&config)?;
        let mut expected_hash = None;
        for repetition in 0..repetitions {
            let composition = ReferenceVNextComposition::create_with_capacity(
                DeviceId::new("device.reference.image-plan-benchmark")?,
                capacity,
            )?;
            let policy = TensorComponentPolicy::new(
                capacity,
                composition
                    .runtime()
                    .descriptor()
                    .dynamic_storage_profiles
                    .iter()
                    .copied()
                    .collect(),
            )?;
            let started = Instant::now();
            let family = registration.prepare_with_profile(&raw_config, &profile)?;
            let family_us = started.elapsed().as_micros();
            let mut options = config.compile_options()?;
            for value in family.program().outputs() {
                options.retain_completion_value(value.clone());
            }
            let started = Instant::now();
            let compilation = ProgramPlanCompiler::compile_with_weight_materializers(
                &family,
                composition.catalog(),
                &policy,
                &composition.registry().planning(),
                composition.weight_materializers(),
                &options,
            )?;
            let compile_us = started.elapsed().as_micros();
            let started = Instant::now();
            let bound = composition.registry().bind_plan(compilation.executable())?;
            let bind_us = started.elapsed().as_micros();
            let plan = compilation.executable().execution_plan();
            let plan_hash = plan.plan_hash().to_string();
            ensure!(
                expected_hash.as_ref().is_none_or(|hash| hash == &plan_hash),
                "identical compilation changed its plan identity"
            );
            expected_hash = Some(plan_hash.clone());
            serde_json::to_writer(
                &mut output,
                &json!({
                    "phase": phase, "repetition": repetition,
                    "family_us": family_us, "compile_us": compile_us, "bind_us": bind_us,
                    "nodes": plan.payload().nodes().len(),
                    "weights": family.program().weights().len(),
                    "plan_hash": plan_hash,
                }),
            )?;
            writeln!(output)?;
            output.flush()?;
            std::hint::black_box(bound);
            ensure!(
                composition.runtime().snapshot().live_allocations == 0,
                "compile allocated device memory"
            );
        }
    }
    Ok(())
}
