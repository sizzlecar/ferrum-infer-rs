//! CPU-only coverage of the original bounded range materializer.
use super::{arguments::Arguments, emit};
use anyhow::{ensure, Context, Result};
use ferrum_quantization::{
    Bf16F16ConversionPolicy, DenseArchivePhaseTiming, ManagedDenseArchiveSource,
    PreparedBf16F16Component, TensorArchive,
};
use ferrum_runtime::{HostMemoryAccount, WeightComponentSource, WeightComponentSpec};
use serde::Serialize;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{io::Write, num::NonZeroUsize, time::Instant};

const PHYSICAL_LIMIT: usize = 64 * 1024 * 1024;

/// Same padding rule as managed_preparation::streaming::logical_capacity.
/// This diagnostic does not create or claim a native execution plan.
fn slot_capacity(host: &HostMemoryAccount, maximum: usize, limit: usize) -> Result<NonZeroUsize> {
    let alignment = host.allocation_alignment();
    let aligned = limit
        .checked_sub(alignment - 1)
        .context("slot padding exceeds ceiling")?
        / alignment
        * alignment;
    let logical = maximum.min(aligned);
    let logical =
        NonZeroUsize::new(logical - logical % 2).context("slot cannot hold an F16 element")?;
    ensure!(
        host.backing_bytes(logical.get())? <= limit,
        "slot exceeds physical ceiling"
    );
    Ok(logical)
}

#[derive(Serialize)]
struct RangeSample {
    offset_bytes: usize,
    length_bytes: usize,
    materialize: DenseArchivePhaseTiming,
    validation: DenseArchivePhaseTiming,
    release: DenseArchivePhaseTiming,
}
#[derive(Serialize)]
struct ComponentSample {
    component_id: String,
    route: &'static str,
    payload_bytes: usize,
    ranges: Vec<RangeSample>,
    checksum_u64: u64,
    output_sha256: Option<String>,
}

/// Consumes every output byte, but does not retain it or interpret it as an
/// execution weight of a different dtype. Optional SHA256 covers ordered ranges.
fn consume(bytes: &[u8], checksum: &mut u64, hash: Option<&mut Sha256>) {
    let mut chunks = bytes.chunks_exact(8);
    for chunk in &mut chunks {
        *checksum = checksum.wrapping_add(u64::from_le_bytes(chunk.try_into().unwrap()));
    }
    for byte in chunks.remainder() {
        *checksum = checksum.wrapping_add(u64::from(*byte));
    }
    if let Some(hash) = hash {
        hash.update(bytes);
    }
    std::hint::black_box(*checksum);
}

pub(super) fn run(
    archive: &dyn TensorArchive,
    host: &HostMemoryAccount,
    components: &[&WeightComponentSpec],
    backing_bytes: &[usize],
    args: &Arguments,
    output: &mut impl Write,
) -> Result<()> {
    run_with_limit(
        archive,
        host,
        components,
        backing_bytes,
        args,
        output,
        PHYSICAL_LIMIT,
    )
}

fn run_with_limit(
    archive: &dyn TensorArchive,
    host: &HostMemoryAccount,
    components: &[&WeightComponentSpec],
    backing_bytes: &[usize],
    args: &Arguments,
    output: &mut impl Write,
    physical_limit: usize,
) -> Result<()> {
    let source = ManagedDenseArchiveSource::new(archive, host);
    let conversion_policy = Bf16F16ConversionPolicy::new(args.conversion_workers)?;
    ensure!(
        components.len() == backing_bytes.len(),
        "backing metadata selection differs"
    );
    let setup = Started::new();
    let recipes = components
        .iter()
        .map(|c| source.prepare_bf16_f16_stream(c))
        .collect::<Result<Vec<Option<PreparedBf16F16Component<'_>>>, _>>()?;
    let maximum = recipes
        .iter()
        .flatten()
        .map(|p| p.full_bytes())
        .max()
        .context("selection has no BF16-to-F16 streaming component")?;
    let logical = slot_capacity(host, maximum, physical_limit)?;
    let slot_bytes = host.backing_bytes(logical.get())?;
    let fallback_peak =
        backing_bytes
            .iter()
            .zip(&recipes)
            .try_fold(0usize, |peak, (&bytes, p)| -> Result<_> {
                if p.is_some() {
                    return Ok(peak);
                }
                Ok(peak.max(if bytes == 0 {
                    0
                } else {
                    host.backing_bytes(bytes)?
                }))
            })?;
    let required_peak = slot_bytes
        .checked_add(fallback_peak)
        .context("Host peak overflow")?;
    ensure!(
        required_peak <= args.host_bytes,
        "Host budget cannot hold slot plus full fallback"
    );
    let slot = source.bounded_streaming_slot(logical)?;
    let setup = setup.finish();
    emit(
        output,
        &json!({
            "kind":"streaming_prepared", "selected_components":components.len(),
            "physical_slot_ceiling_bytes":physical_limit,
            "logical_slot_bytes":logical.get(), "physical_slot_bytes":slot_bytes,
            "maximum_conversion_workers":conversion_policy.maximum_workers().get(),
            "minimum_bytes_per_worker":Bf16F16ConversionPolicy::MINIMUM_BYTES_PER_WORKER,
            "full_fallback_peak_bytes":fallback_peak, "required_host_peak_bytes":required_peak,
            "setup":setup, "slot_lifetime":"one charged slot across all passes",
            "counter_scope":"RUSAGE_SELF; CPU-only independent process",
            "validation":"full output-byte checksum after each range; optional ordered SHA256",
            "limitations":[
                "No GPU/upload/identity-materializer/fence latency is measured",
                "All selected stored schema components, not a claim about native placement order",
                "Materialize includes source reads, conversion and destination writes/faults",
                "Setup includes metadata preparation and first slot allocation",
                "Pass totals include validation/release; range materialize excludes validation",
                "No cold-cache guarantee; source fingerprinting already read the archive",
                "Full fallback retains the slot and uses the original full component method"
            ]
        }),
    )?;
    for iteration in 0..args.iterations {
        let pass = Started::new();
        let measured = (|| -> Result<Vec<ComponentSample>> {
            let mut samples = Vec::with_capacity(components.len());
            for (component, recipe) in components.iter().zip(&recipes) {
                let full_bytes = usize::try_from(component.physical_bytes()?)?;
                let mut offset = 0usize;
                let mut ranges = Vec::new();
                let mut checksum = 0u64;
                let mut hash = args.hash_output.then(Sha256::new);
                while offset < full_bytes {
                    let length = if recipe.is_some() {
                        logical.get().min(full_bytes - offset)
                    } else {
                        full_bytes
                    };
                    let started = Started::new();
                    // The two original paths remain typed; no raw-copy stand-in.
                    let (materialize, validation, release) = if let Some(recipe) = recipe {
                        let payload = recipe.materialize_range_with_policy(
                            &slot,
                            offset,
                            length,
                            conversion_policy,
                        )?;
                        let materialize = started.finish();
                        ensure!(payload.len() == length, "range length differs from request");
                        let checked = Started::new();
                        consume(payload.bytes(), &mut checksum, hash.as_mut());
                        let validation = checked.finish();
                        let released = Started::new();
                        drop(payload);
                        (materialize, validation, released.finish())
                    } else {
                        let payload = source.component(component)?;
                        let materialize = started.finish();
                        ensure!(
                            payload.bytes().len() == length,
                            "full fallback extent differs"
                        );
                        let checked = Started::new();
                        consume(payload.bytes(), &mut checksum, hash.as_mut());
                        let validation = checked.finish();
                        let released = Started::new();
                        drop(payload);
                        (materialize, validation, released.finish())
                    };
                    ensure!(
                        host.usage()?.committed_bytes() == slot_bytes,
                        "range/fallback retained Host backing"
                    );
                    ranges.push(RangeSample {
                        offset_bytes: offset,
                        length_bytes: length,
                        materialize,
                        validation,
                        release,
                    });
                    offset = offset
                        .checked_add(length)
                        .context("component coverage overflow")?;
                }
                ensure!(offset == full_bytes, "component coverage incomplete");
                samples.push(ComponentSample {
                    component_id: component.id.to_string(),
                    route: if recipe.is_some() {
                        "bf16_f16_materialize_range"
                    } else {
                        "original_full_fallback"
                    },
                    payload_bytes: full_bytes,
                    ranges,
                    checksum_u64: checksum,
                    output_sha256: hash.map(|h| format!("{:x}", h.finalize())),
                });
            }
            Ok(samples)
        })();
        let pass_timing = pass.finish();
        let samples = match measured {
            Ok(samples) => samples,
            Err(error) => {
                drop(slot);
                emit(
                    output,
                    &json!({
                        "kind":"streaming_failed", "iteration":iteration, "error":error.to_string(),
                        "pass":pass_timing, "host_committed_after_slot_release_bytes":host.usage()?.committed_bytes()
                    }),
                )?;
                return Err(error);
            }
        };
        let payload_bytes = samples
            .iter()
            .try_fold(0u64, |n, s| n.checked_add(s.payload_bytes as u64))
            .context("pass payload total overflow")?;
        // JSON output is outside the pass clock and retains only range metadata.
        for (ordinal, sample) in samples.iter().enumerate() {
            emit(
                output,
                &json!({"kind":"streaming_component", "iteration":iteration, "ordinal":ordinal, "sample":sample}),
            )?;
        }
        emit(
            output,
            &json!({
                "kind":"streaming_pass_completed", "iteration":iteration, "components":samples.len(),
                "payload_bytes":payload_bytes, "pass_including_validation":pass_timing,
                "host_committed_bytes":host.usage()?.committed_bytes(),
                "hash_validation_enabled":args.hash_output
            }),
        )?;
    }
    let release = Started::new();
    drop(slot);
    let release = release.finish();
    ensure!(
        host.usage()?.committed_bytes() == 0,
        "slot release did not return Host budget"
    );
    emit(
        output,
        &json!({"kind":"completed", "slot_release":release, "host_committed_bytes":0}),
    )
}

struct Started {
    wall: Instant,
    usage: Option<Usage>,
}
#[derive(Clone, Copy)]
struct Usage {
    user: u64,
    system: u64,
    minor: u64,
    major: u64,
    blocks: u64,
}
impl Started {
    fn new() -> Self {
        Self {
            wall: Instant::now(),
            usage: usage(),
        }
    }
    fn finish(self) -> DenseArchivePhaseTiming {
        let after = usage();
        let mut timing = DenseArchivePhaseTiming {
            calls: 1,
            wall_ns: u64::try_from(self.wall.elapsed().as_nanos()).unwrap_or(u64::MAX),
            ..Default::default()
        };
        if let (Some(before), Some(after)) = (self.usage, after) {
            timing.process_counter_calls = 1;
            timing.process_user_cpu_ns = after.user.checked_sub(before.user);
            timing.process_system_cpu_ns = after.system.checked_sub(before.system);
            timing.minor_faults = after.minor.checked_sub(before.minor);
            timing.major_faults = after.major.checked_sub(before.major);
            timing.input_block_operations = after.blocks.checked_sub(before.blocks);
        }
        timing
    }
}
fn usage() -> Option<Usage> {
    #[cfg(unix)]
    {
        let mut value = std::mem::MaybeUninit::<libc::rusage>::uninit();
        // SAFETY: getrusage writes a complete rusage on the checked success path.
        if unsafe { libc::getrusage(libc::RUSAGE_SELF, value.as_mut_ptr()) } != 0 {
            return None;
        }
        // SAFETY: the successful call initialized the structure.
        let value = unsafe { value.assume_init() };
        let ns = |t: libc::timeval| {
            u64::try_from(t.tv_sec)
                .ok()?
                .checked_mul(1_000_000_000)?
                .checked_add(u64::try_from(t.tv_usec).ok()?.checked_mul(1000)?)
        };
        Some(Usage {
            user: ns(value.ru_utime)?,
            system: ns(value.ru_stime)?,
            minor: u64::try_from(value.ru_minflt).ok()?,
            major: u64::try_from(value.ru_majflt).ok()?,
            blocks: u64::try_from(value.ru_inblock).ok()?,
        })
    }
    #[cfg(not(unix))]
    {
        None
    }
}

#[cfg(test)]
#[path = "streaming/tests.rs"]
mod tests;
