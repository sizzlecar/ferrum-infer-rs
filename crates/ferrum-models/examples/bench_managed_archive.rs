//! CPU-only diagnostics of the real prepared family's managed archive source.
//! No device runtime, GPU allocation, upload, or model inference is performed.
use anyhow::{ensure, Context, Result};
use ferrum_models::vnext::{
    open_registered_product_sources, resolve_registered_model_from_sources,
    ProductionWeightArtifact,
};
use ferrum_quantization::Bf16ToF16BlockCounts;
use ferrum_runtime::{
    HostMemoryAccount, HostMemoryClassLimits, HostMemoryLimits, ModelSourceKind,
    OriginalModelSource, OriginalModelSources,
};
use serde_json::json;
use std::{io::Write, sync::Arc, time::Instant};

#[path = "bench_managed_archive/arguments.rs"]
mod arguments;
#[path = "bench_managed_archive/measurement.rs"]
mod measurement;
#[path = "bench_managed_archive/streaming.rs"]
mod streaming;

fn emit(output: &mut impl Write, value: &impl serde::Serialize) -> Result<()> {
    serde_json::to_writer(&mut *output, value)?;
    writeln!(output)?;
    output.flush()?;
    Ok(())
}

fn main() -> Result<()> {
    let Some(args) = arguments::Arguments::parse(std::env::args().skip(1))? else {
        println!("{}", arguments::USAGE);
        return Ok(());
    };
    let mut output = std::io::BufWriter::new(std::io::stdout().lock());
    emit(
        &mut output,
        &json!({
            "kind":"invocation", "schema":"ferrum.managed_archive_cpu_diagnostic.v2", "arguments":args,
            "performance_timing_eligible":!args.hash_output,
            "cpu_architecture":std::env::consts::ARCH,
            "conversion_backend":Bf16ToF16BlockCounts::backend(),
            "limitations":[
                "CPU source preparation only; no GPU, execution/upload timing, or output-quality claim",
                "Measures archive component materialization; outer runtime prepare_uploads/identity-materializer validation is not included",
                "Production source fingerprinting reads the checkpoint before measurement; no cold-cache claim",
                "Touch-stride mode reads one byte per stride and the final byte; it changes residency but cannot guarantee fault-free subsequent conversion",
                "Allocation time measures reserve+allocate_zeroed; lazy destination page first-touch may remain in conversion",
                "Composite conversion writes directly to final ordered offsets; there is no separately timed payload-sized packing copy",
                "Diagnostic counters and timers add overhead; baseline mode runs the same materializer without them",
                "Process CPU/fault/block-operation counters may include other threads; block operations are not bytes or disk elapsed time",
                "Fresh/reuse iterations own a new installation; bounded-stream reuses one charged slot across complete passes and reports its setup/final release separately",
                "For reuse, reserve_allocate_zeroed measures slot checkout; its initial allocation is separately recorded in installation initialization",
                "Output SHA256 selects a correctness-validation run: validation wall includes hashing and no performance installation total is emitted; component timing is diagnostic only in that run",
                "Every stored family component is selected unless an exact component ID is supplied; this is not proof of native active placements"
            ]
        }),
    )?;
    let original = |kind, path: &std::path::Path| OriginalModelSource {
        kind,
        location: path.display().to_string(),
        requested_revision: None,
    };
    let started = Instant::now();
    let sources = Arc::new(open_registered_product_sources(
        &args.semantic_root,
        &args.tokenizer_root,
        ProductionWeightArtifact::safetensors_file(&args.weights_file),
        OriginalModelSources {
            semantic: original(ModelSourceKind::LocalDirectory, &args.semantic_root),
            tokenizer: original(ModelSourceKind::LocalDirectory, &args.tokenizer_root),
            weights: original(ModelSourceKind::LocalFile, &args.weights_file),
        },
    )?);
    let source_open_ns = started.elapsed().as_nanos();
    let started = Instant::now();
    let prepared = Arc::new(
        resolve_registered_model_from_sources(&sources)?
            .into_required()?
            .define_from_sources(sources)?
            .prepare(&args.numerical_profile.parse().map_err(anyhow::Error::msg)?)?,
    );
    let family_prepare_ns = started.elapsed().as_nanos();
    let started = Instant::now();
    let managed = prepared.managed_safetensors()?;
    let mapping_and_verification_ns = started.elapsed().as_nanos();
    let archive = managed.shared_archive();
    let components = prepared
        .family()
        .stored_weight_schema()?
        .components
        .iter()
        .filter(|c| {
            args.component_id
                .as_ref()
                .is_none_or(|id| c.id.as_str() == id)
        })
        .collect::<Vec<_>>();
    ensure!(
        !components.is_empty(),
        "no prepared component matches the requested selection"
    );
    let host = HostMemoryAccount::with_allocation_alignment(
        HostMemoryLimits::new(
            args.host_bytes,
            HostMemoryClassLimits {
                input_preprocess_bytes: 0,
                output_bytes: 0,
                transfer_staging_bytes: 0,
                weight_backing_bytes: args.host_bytes,
            },
        )?,
        args.host_alignment,
    )?;
    let sealed_source = managed.weight_source(&host);
    if args.backing == arguments::Backing::BoundedStream {
        let backing_bytes = components
            .iter()
            .map(|c| {
                usize::try_from(sealed_source.component_backing_bytes(c)?)
                    .map_err(anyhow::Error::from)
            })
            .collect::<Result<Vec<_>>>()?;
        emit(
            &mut output,
            &json!({
                "kind":"prepared", "source_open_and_fingerprint_ns":source_open_ns,
                "family_prepare_ns":family_prepare_ns,
                "mapping_and_fingerprint_verification_ns":mapping_and_verification_ns,
                "resolved_sources":prepared.sources().resolved_sources(),
                "family_fingerprint":prepared.family().fingerprint()?,
                "numerical_profile":prepared.family().numerical_profile().id,
                "selected_components":components.len()
            }),
        )?;
        return streaming::run(
            archive.as_ref(),
            &host,
            &components,
            &backing_bytes,
            &args,
            &mut output,
        );
    }
    let (selected_maximum_component_bytes, selected_backing_peak) = components.iter().try_fold(
        (0_usize, 0_usize),
        |(maximum, peak), component| -> Result<(usize, usize)> {
            let bytes = usize::try_from(sealed_source.component_backing_bytes(component)?)?;
            Ok((
                maximum.max(bytes),
                peak.max(if bytes == 0 {
                    0
                } else {
                    host.backing_bytes(bytes)?
                }),
            ))
        },
    )?;
    ensure!(
        selected_backing_peak <= args.host_bytes,
        "Host budget {} is below actual selected component backing peak {selected_backing_peak}",
        args.host_bytes
    );
    emit(
        &mut output,
        &json!({
            "kind":"prepared", "source_open_and_fingerprint_ns":source_open_ns,
            "family_prepare_ns":family_prepare_ns, "mapping_and_fingerprint_verification_ns":mapping_and_verification_ns,
            "resolved_sources":prepared.sources().resolved_sources(),
            "family_fingerprint":prepared.family().fingerprint()?,
            "numerical_profile":prepared.family().numerical_profile().id,
            "selected_components":components.len(),
            "selected_execution_component_bytes":components.iter().try_fold(0u64, |sum,c| {
                sum.checked_add(c.physical_bytes()?).context("selected component bytes overflow")
            })?,
            "selected_host_backing_peak_bytes":selected_backing_peak,
            "selected_maximum_converted_component_bytes":selected_maximum_component_bytes,
            "host_alignment":args.host_alignment,
        }),
    )?;
    for iteration in 0..args.iterations {
        // Only small metadata is retained. Neither output serialization nor
        // stream flushing sits inside the installation timing boundary.
        let mut samples = Vec::with_capacity(components.len());
        let mut totals = measurement::Totals::default();
        let installation_started = Instant::now();
        let source = measurement::Installation::new(
            archive.as_ref(),
            &host,
            args.backing,
            selected_maximum_component_bytes,
        )?;
        let installation_initialization_wall_ns = installation_started.elapsed().as_nanos();
        for (ordinal, component) in components.iter().enumerate() {
            let measured = measurement::measure(&source, &host, component, &args);
            match measured {
                Ok(sample) => {
                    totals.add(&sample);
                    samples.push(sample);
                }
                Err(failure) => {
                    drop(source);
                    let failed_installation_wall_ns = installation_started.elapsed().as_nanos();
                    emit(
                        &mut output,
                        &json!({
                            "kind":"component_failed", "iteration":iteration, "ordinal":ordinal,
                            "component":component, "failure":failure,
                            "failed_installation_wall_ns":failed_installation_wall_ns,
                            "performance_timing_eligible":false,
                            "host_committed_after_installation_release_bytes":host.usage()?.committed_bytes(),
                        }),
                    )?;
                    anyhow::bail!("managed source diagnostic failed: {}", failure.error);
                }
            }
        }
        let idle_host_allocated_bytes = host.usage()?.allocated_bytes();
        let release_started = Instant::now();
        drop(source);
        let installation_final_release_wall_ns = release_started.elapsed().as_nanos();
        let total_wall_ns = installation_started.elapsed().as_nanos();
        ensure!(
            host.usage()?.committed_bytes() == 0,
            "backing remained after installation release"
        );
        for (ordinal, (component, sample)) in components.iter().zip(samples).enumerate() {
            emit(
                &mut output,
                &json!({
                    "kind":"component", "iteration":iteration, "ordinal":ordinal,
                    "component":component, "sample":sample,
                }),
            )?;
        }
        emit(
            &mut output,
            &json!({
                "kind":"iteration_completed", "iteration":iteration, "totals":totals,
                "installation_wall_ns":(!args.hash_output).then_some(total_wall_ns),
                "hash_validation_wall_ns":args.hash_output.then_some(total_wall_ns),
                "installation_initialization_wall_ns":installation_initialization_wall_ns,
                "installation_final_release_wall_ns":installation_final_release_wall_ns,
                "idle_host_allocated_before_installation_release_bytes":idle_host_allocated_bytes,
                "host_committed_after_installation_release_bytes":host.usage()?.committed_bytes(),
            }),
        )?;
    }
    emit(
        &mut output,
        &json!({"kind":"completed", "host_committed_bytes":host.usage()?.committed_bytes()}),
    )
}
