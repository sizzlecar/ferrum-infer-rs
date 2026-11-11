use super::arguments::{Arguments, Backing, Mode};
use ferrum_quantization::{
    DenseArchiveMaterializationReport, ManagedDenseArchiveInstallation, ManagedDenseArchiveSource,
    TensorArchive,
};
use ferrum_runtime::{
    HostMemoryAccount, VNextError, WeightComponentPayload, WeightComponentSource,
    WeightComponentSpec,
};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::time::Instant;

pub(super) enum Installation<'archive> {
    Fresh(ManagedDenseArchiveSource<'archive>),
    Reuse(ManagedDenseArchiveInstallation<'archive>),
}

impl<'archive> Installation<'archive> {
    pub fn new(
        archive: &'archive dyn TensorArchive,
        host: &'archive HostMemoryAccount,
        backing: Backing,
        maximum_component_bytes: usize,
    ) -> anyhow::Result<Self> {
        let source = ManagedDenseArchiveSource::new(archive, host);
        match backing {
            Backing::Fresh => Ok(Self::Fresh(source)),
            Backing::Reuse => {
                let maximum =
                    std::num::NonZeroUsize::new(maximum_component_bytes).ok_or_else(|| {
                        anyhow::anyhow!("reuse needs at least one converted component")
                    })?;
                Ok(Self::Reuse(source.reusable_installation(maximum)?))
            }
            Backing::BoundedStream => anyhow::bail!("bounded-stream uses its range runner"),
        }
    }

    pub fn idle_backing_bytes(&self) -> usize {
        match self {
            Self::Fresh(_) => 0,
            Self::Reuse(source) => source.backing_bytes(),
        }
    }

    fn materialize<'source>(
        &'source self,
        component: &WeightComponentSpec,
        args: &Arguments,
        report: &mut DenseArchiveMaterializationReport,
    ) -> Result<WeightComponentPayload<'source>, VNextError> {
        match (self, args.mode) {
            (Self::Fresh(source), Mode::Baseline) => source.component(component),
            (Self::Reuse(source), Mode::Baseline) => source.component(component),
            (Self::Fresh(source), Mode::Diagnostic) => {
                source.component_with_diagnostics(component, args.source_touch(), report)
            }
            (Self::Reuse(source), Mode::Diagnostic) => {
                source.component_with_diagnostics(component, args.source_touch(), report)
            }
        }
    }
}

#[derive(Serialize)]
pub(super) struct Sample {
    pub materialization_wall_ns: u128,
    pub payload_release_wall_ns: u128,
    pub payload_bytes: usize,
    pub live_host_backing_bytes: usize,
    pub host_committed_after_release_bytes: usize,
    pub output_sha256: Option<String>,
    pub output_hash_wall_ns: Option<u128>,
    pub diagnostic: Option<DenseArchiveMaterializationReport>,
    pub diagnostic_unattributed_wall_ns: Option<u64>,
    pub scalar_fallback_block_fraction: Option<f64>,
}

#[derive(Serialize)]
pub(super) struct Failure {
    pub error: String,
    pub materialization_wall_ns: u128,
    pub host_committed_after_release_bytes: Option<usize>,
    pub diagnostic: Option<DenseArchiveMaterializationReport>,
}

pub(super) fn measure(
    source: &Installation<'_>,
    host: &HostMemoryAccount,
    component: &WeightComponentSpec,
    args: &Arguments,
) -> Result<Sample, Failure> {
    let mut report = DenseArchiveMaterializationReport::default();
    let started = Instant::now();
    let result = source.materialize(component, args, &mut report);
    let materialization_wall_ns = started.elapsed().as_nanos();
    let diagnostic = (args.mode == Mode::Diagnostic).then_some(report);
    let payload = match result {
        Ok(payload) => payload,
        Err(error) => {
            return Err(Failure {
                error: error.to_string(),
                materialization_wall_ns,
                diagnostic,
                host_committed_after_release_bytes: host.usage().ok().map(|u| u.committed_bytes()),
            })
        }
    };
    let payload_bytes = payload.bytes().len();
    let live_host_backing = host.usage().map(|usage| usage.allocated_bytes());
    let hash_started = Instant::now();
    let output_sha256 = args
        .hash_output
        .then(|| format!("{:x}", Sha256::digest(payload.bytes())));
    let output_hash_wall_ns = args.hash_output.then(|| hash_started.elapsed().as_nanos());
    std::hint::black_box(payload.bytes());
    let release_started = Instant::now();
    drop(payload);
    let payload_release_wall_ns = release_started.elapsed().as_nanos();
    let after = host.usage();
    let failure = |error: String| Failure {
        error,
        materialization_wall_ns,
        diagnostic: diagnostic.clone(),
        host_committed_after_release_bytes: after.as_ref().ok().map(|u| u.committed_bytes()),
    };
    let live_host_backing_bytes = live_host_backing.map_err(|e| failure(e.to_string()))?;
    let after = after.as_ref().map_err(|e| failure(e.to_string()))?;
    if after.committed_bytes() != source.idle_backing_bytes() {
        return Err(failure(
            "source backing after payload release differs from declared idle allocation".to_owned(),
        ));
    }
    if component
        .physical_bytes()
        .map_err(|e| failure(e.to_string()))?
        != payload_bytes as u64
    {
        return Err(failure(
            "payload extent differs from the prepared component".to_owned(),
        ));
    }
    let diagnostic_unattributed_wall_ns = diagnostic.as_ref().map(|r| r.unattributed_wall_ns());
    let scalar_fallback_block_fraction = diagnostic.as_ref().and_then(|r| {
        (r.bf16_to_f16.simd_blocks_attempted != 0).then(|| {
            r.bf16_to_f16.scalar_fallback_blocks as f64 / r.bf16_to_f16.simd_blocks_attempted as f64
        })
    });
    Ok(Sample {
        materialization_wall_ns,
        payload_release_wall_ns,
        payload_bytes,
        live_host_backing_bytes,
        host_committed_after_release_bytes: after.committed_bytes(),
        output_sha256,
        output_hash_wall_ns,
        diagnostic,
        diagnostic_unattributed_wall_ns,
        scalar_fallback_block_fraction,
    })
}

#[derive(Default, Serialize)]
pub(super) struct Totals {
    pub components: usize,
    pub payload_bytes: u128,
    pub materialization_wall_ns: u128,
    pub payload_release_wall_ns: u128,
    pub peak_live_host_backing_bytes: usize,
    pub output_hash_wall_ns: u128,
    pub diagnostic: Option<DiagnosticTotals>,
}

#[derive(Default, Serialize)]
pub(super) struct DiagnosticTotals {
    pub source_bytes: u128,
    pub source_touch_loads: u128,
    pub borrowed_components: usize,
    pub composite_components: usize,
    pub affine_components: usize,
    pub metadata_validation_wall_ns: u128,
    pub source_touch_wall_ns: u128,
    pub reserve_allocate_zeroed_wall_ns: u128,
    pub conversion_and_direct_packing_wall_ns: u128,
    pub payload_assembly_wall_ns: u128,
    pub unattributed_wall_ns: u128,
    pub simd_blocks_attempted: u128,
    pub simd_blocks_converted: u128,
    pub scalar_fallback_blocks: u128,
    pub scalar_tail_elements: u128,
    pub portable_scalar_elements: u128,
    pub scalar_fallback_block_fraction: Option<f64>,
}

impl Totals {
    pub fn add(&mut self, sample: &Sample) {
        self.components += 1;
        self.payload_bytes += sample.payload_bytes as u128;
        self.materialization_wall_ns += sample.materialization_wall_ns;
        self.payload_release_wall_ns += sample.payload_release_wall_ns;
        self.peak_live_host_backing_bytes = self
            .peak_live_host_backing_bytes
            .max(sample.live_host_backing_bytes);
        self.output_hash_wall_ns += sample.output_hash_wall_ns.unwrap_or(0);
        if let Some(r) = &sample.diagnostic {
            let d = self
                .diagnostic
                .get_or_insert_with(DiagnosticTotals::default);
            d.source_bytes += r.source_bytes as u128;
            d.source_touch_loads += r.source_touch_loads as u128;
            d.borrowed_components += usize::from(r.borrowed);
            d.composite_components += usize::from(r.composite);
            d.affine_components += usize::from(r.affine);
            d.metadata_validation_wall_ns += r.metadata_validation.wall_ns as u128;
            d.source_touch_wall_ns += r.source_touch.wall_ns as u128;
            d.reserve_allocate_zeroed_wall_ns += r.reserve_allocate_zeroed.wall_ns as u128;
            d.conversion_and_direct_packing_wall_ns +=
                r.conversion_and_direct_packing.wall_ns as u128;
            d.payload_assembly_wall_ns += r.payload_assembly.wall_ns as u128;
            d.unattributed_wall_ns += r.unattributed_wall_ns() as u128;
            d.simd_blocks_attempted += r.bf16_to_f16.simd_blocks_attempted as u128;
            d.simd_blocks_converted += r.bf16_to_f16.simd_blocks_converted as u128;
            d.scalar_fallback_blocks += r.bf16_to_f16.scalar_fallback_blocks as u128;
            d.scalar_tail_elements += r.bf16_to_f16.scalar_tail_elements as u128;
            d.portable_scalar_elements += r.bf16_to_f16.portable_scalar_elements as u128;
            d.scalar_fallback_block_fraction = (d.simd_blocks_attempted != 0)
                .then(|| d.scalar_fallback_blocks as f64 / d.simd_blocks_attempted as f64);
        }
    }
}

#[cfg(test)]
#[path = "measurement/tests.rs"]
mod tests;
