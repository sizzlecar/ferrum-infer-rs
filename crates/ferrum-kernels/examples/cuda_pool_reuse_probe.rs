//! Allocation-only replay of the image B15 -> B1 transition; no kernels/inference.
//! Defaults to a dry run. Execution requires --run --budget-bytes <bytes>.
//! All shapes use one common scale bounded by that budget and observed free VRAM
//! minus 64 MiB. A reduced replay that does not fail is explicitly inconclusive.

use std::error::Error;

const REFERENCE_FREE: u64 = 47_349_825_536;
const HEADROOM: u64 = 64 * 1024 * 1024;
const TEXT_WEIGHT: u64 = 13_892_143_104;
const INITIAL: [(&str, u64); 9] = [
    ("denoiser.weight", 14_230_249_472),
    ("denoiser.dynamic.0", 2_023_096_320),
    ("denoiser.dynamic.1", 10_637_123_584),
    ("denoiser.dynamic.2", 2_107_392),
    ("denoiser.dynamic.3", 1_011_548_672),
    ("decoder.weight", 506_148_864),
    ("decoder.dynamic.0", 1_971_847_168),
    ("decoder.dynamic.1", 2_724_872_192),
    ("decoder.dynamic.2", 1_660_944_384),
];

#[derive(Default)]
struct Args {
    run: bool,
    device: usize,
    budget: Option<u64>,
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = Args::default();
    let mut input = std::env::args().skip(1);
    while let Some(flag) = input.next() {
        match flag.as_str() {
            "--run" => args.run = true,
            "--device" => args.device = input.next().ok_or("missing device ordinal")?.parse()?,
            "--budget-bytes" => {
                args.budget = Some(input.next().ok_or("missing byte budget")?.parse()?)
            }
            "--help" | "-h" => {
                println!("cuda_pool_reuse_probe [--device N] --run --budget-bytes N\nWithout --run: print the replay plan without creating a CUDA context.");
                return Ok(());
            }
            _ => return Err(format!("unknown argument: {flag}").into()),
        }
    }
    if !args.run {
        println!(
            "{}",
            serde_json::json!({
                "event": "dry_run", "cuda_context_created": false,
                "reference_free_bytes": REFERENCE_FREE, "headroom_bytes": HEADROOM,
                "maximum_budget_bytes": args.budget, "initial_shapes": INITIAL,
                "free_before_text": "denoiser.dynamic.*", "text_weight_bytes": TEXT_WEIGHT,
                "sequence": ["raw", "on OOM: unchanged retry", "on OOM: retain own pool, retry without sync", "on OOM: synchronize, retry", "on OOM: trim own pool to zero, retry"],
                "interpretation": "No failure at a smaller common scale is inconclusive, not proof of a cause or fix."
            })
        );
        return Ok(());
    }
    let budget = args
        .budget
        .filter(|value| *value > 0)
        .ok_or("--run requires a positive --budget-bytes")?;
    #[cfg(feature = "cuda")]
    return cuda::run(args.device, budget);
    #[cfg(not(feature = "cuda"))]
    {
        let _ = budget;
        Err("rebuild this example with --features cuda".into())
    }
}

#[cfg(feature = "cuda")]
mod cuda {
    use super::*;
    use cudarc::driver::{result, sys, CudaContext, CudaSlice, CudaStream, DevicePtr, DriverError};
    use serde_json::{json, Value};
    use std::{
        io::Write,
        sync::Arc,
        time::{Instant, SystemTime, UNIX_EPOCH},
    };

    struct Reporter {
        start: Instant,
        context: Arc<CudaContext>,
        stream: Arc<CudaStream>,
        pool: sys::CUmemoryPool,
    }

    struct Owner {
        name: &'static str,
        bytes: usize,
        pointer: u64,
        _allocation: CudaSlice<u8>,
    }

    fn emit(value: Value) -> Result<(), Box<dyn Error>> {
        let mut stdout = std::io::stdout().lock();
        writeln!(stdout, "{value}")?;
        stdout.flush()?;
        Ok(())
    }

    unsafe fn pool_u64(
        pool: sys::CUmemoryPool,
        attr: sys::CUmemPool_attribute,
    ) -> Result<u64, DriverError> {
        let mut value = 0u64;
        result::mem_pool::get_attribute(pool, attr, (&mut value as *mut u64).cast())?;
        Ok(value)
    }

    unsafe fn pool_i32(
        pool: sys::CUmemoryPool,
        attr: sys::CUmemPool_attribute,
    ) -> Result<i32, DriverError> {
        let mut value = 0i32;
        result::mem_pool::get_attribute(pool, attr, (&mut value as *mut i32).cast())?;
        Ok(value)
    }

    fn observed<T: serde::Serialize>(value: Result<T, DriverError>) -> Value {
        match value {
            Ok(value) => json!({"value": value}),
            Err(error) => json!({"error": format!("{error:?}")}),
        }
    }

    impl Reporter {
        // Only queries: no stream/context synchronization and no pool mutation.
        fn snapshot(&self, phase: &str, owners: &[Owner]) -> Result<(), Box<dyn Error>> {
            use sys::CUmemPool_attribute as A;
            let query_start = Instant::now();
            let stream_status = unsafe { sys::cuStreamQuery(self.stream.cu_stream()) };
            let query_us = query_start.elapsed().as_micros();
            let pool = self.pool;
            emit(json!({
                "event": "snapshot", "phase": phase,
                "elapsed_us": self.start.elapsed().as_micros(),
                "unix_us": SystemTime::now().duration_since(UNIX_EPOCH)?.as_micros(),
                "pool_handle": format!("{pool:p}"),
                "stream_handle": format!("{:p}", self.stream.cu_stream()),
                "stream_query": format!("{stream_status:?}"), "stream_query_us": query_us,
                "device_free_total_bytes": observed(self.context.mem_get_info()),
                "reserved_bytes": observed(unsafe { pool_u64(pool, A::CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT) }),
                "used_bytes": observed(unsafe { pool_u64(pool, A::CU_MEMPOOL_ATTR_USED_MEM_CURRENT) }),
                "release_threshold_bytes": observed(unsafe { pool_u64(pool, A::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD) }),
                "reuse_follow_event_dependencies": observed(unsafe { pool_i32(pool, A::CU_MEMPOOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES) }),
                "reuse_allow_opportunistic": observed(unsafe { pool_i32(pool, A::CU_MEMPOOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC) }),
                "reuse_allow_internal_dependencies": observed(unsafe { pool_i32(pool, A::CU_MEMPOOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES) }),
                "live_owners": owners.iter().map(|o| json!({"name":o.name,"bytes":o.bytes,"pointer":format!("0x{:x}",o.pointer)})).collect::<Vec<_>>()
            }))
        }

        fn allocate(
            &self,
            name: &'static str,
            bytes: usize,
            phase: &str,
            owners: &mut Vec<Owner>,
        ) -> Result<bool, Box<dyn Error>> {
            self.snapshot(&format!("{phase}.before_alloc"), owners)?;
            let start = Instant::now();
            // Allocation only; uninitialized bytes are never read or executed.
            let allocation = unsafe { self.stream.alloc::<u8>(bytes) };
            let alloc_us = start.elapsed().as_micros();
            match allocation {
                Ok(allocation) => {
                    let pointer = {
                        let (pointer, _guard) = allocation.device_ptr(&self.stream);
                        pointer
                    };
                    owners.push(Owner {
                        name,
                        bytes,
                        pointer,
                        _allocation: allocation,
                    });
                    emit(
                        json!({"event":"allocation_result","phase":phase,"bytes":bytes,"alloc_us":alloc_us,"result":"success","pointer":format!("0x{pointer:x}")}),
                    )?;
                    self.snapshot(&format!("{phase}.after_alloc"), owners)?;
                    Ok(true)
                }
                Err(error) => {
                    emit(
                        json!({"event":"allocation_result","phase":phase,"bytes":bytes,"alloc_us":alloc_us,"result":"failure","cuda_error":format!("{error:?}")}),
                    )?;
                    self.snapshot(&format!("{phase}.after_alloc"), owners)?;
                    if error.0 == sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY {
                        Ok(false)
                    } else {
                        Err(error.into())
                    }
                }
            }
        }

        fn synchronize(&self, phase: &str, owners: &[Owner]) -> Result<(), Box<dyn Error>> {
            let start = Instant::now();
            let result = self.stream.synchronize();
            emit(
                json!({"event":"explicit_stream_sync","phase":phase,"duration_us":start.elapsed().as_micros(),"result":observed(result)}),
            )?;
            self.snapshot(phase, owners)?;
            result?;
            Ok(())
        }
    }

    // Changes only this diagnostic process's pool, only after raw OOM. Restored
    // before owner cleanup on normal return, Rust error propagation, or unwinding.
    struct RetainPool {
        pool: sys::CUmemoryPool,
        original: u64,
    }

    impl RetainPool {
        fn new(pool: sys::CUmemoryPool) -> Result<Self, DriverError> {
            let attr = sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD;
            let original = unsafe { pool_u64(pool, attr)? };
            let mut keep = u64::MAX;
            unsafe {
                result::mem_pool::set_attribute(pool, attr, (&mut keep as *mut u64).cast())?;
            }
            Ok(Self { pool, original })
        }
    }

    impl Drop for RetainPool {
        fn drop(&mut self) {
            let result = unsafe {
                result::mem_pool::set_attribute(
                    self.pool,
                    sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
                    (&mut self.original as *mut u64).cast(),
                )
            };
            let _ = emit(
                json!({"event":"restore_release_threshold","original_bytes":self.original,"result":observed(result)}),
            );
        }
    }

    pub fn run(device: usize, budget: u64) -> Result<(), Box<dyn Error>> {
        let context = CudaContext::new(device)?;
        unsafe {
            context.disable_event_tracking();
        }
        if !context.has_async_alloc() {
            return Err("device lacks production async allocator; no replay performed".into());
        }
        let stream = context.new_stream()?;
        let pool = unsafe { result::device::get_mem_pool(context.cu_device())? };
        let (free, total) = context.mem_get_info()?;
        let numerator = budget
            .min((free as u64).saturating_sub(HEADROOM))
            .min(REFERENCE_FREE);
        let scale = |bytes: u64| -> Result<usize, Box<dyn Error>> {
            let bytes = ((u128::from(bytes) * u128::from(numerator) / u128::from(REFERENCE_FREE))
                / 256)
                * 256;
            if bytes == 0 {
                return Err(
                    "budget/free memory too small to preserve every nonzero aligned shape".into(),
                );
            }
            Ok(usize::try_from(bytes)?)
        };
        let shapes = INITIAL
            .iter()
            .map(|(name, bytes)| Ok((*name, scale(*bytes)?)))
            .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
        let text_bytes = scale(TEXT_WEIGHT)?;
        emit(
            json!({"event":"replay_plan","device":device,"async_allocator":context.has_async_alloc(),"event_tracking":context.is_event_tracking(),"maximum_budget_bytes":budget,"observed_free_bytes":free,"observed_total_bytes":total,"headroom_bytes":HEADROOM,"scale_numerator":numerator,"scale_denominator":REFERENCE_FREE,"initial_shapes":shapes,"text_weight_bytes":text_bytes,"inference_or_kernels":false,"scope":"own diagnostic process only; current free is an observation, not an OS reservation"}),
        )?;
        let reporter = Reporter {
            start: Instant::now(),
            context,
            stream,
            pool,
        };
        let mut owners = Vec::new();
        let outcome = (|| -> Result<(), Box<dyn Error>> {
            reporter.snapshot("start", &owners)?;
            for (name, bytes) in shapes {
                if !reporter.allocate(name, bytes, name, &mut owners)? {
                    return Err(format!(
                        "initial setup allocation failed at {name}; transition not reached"
                    )
                    .into());
                }
                reporter.synchronize(&format!("{name}.initialized"), &owners)?;
            }
            reporter.snapshot("before_free_denoiser_dynamic", &owners)?;
            owners.retain(|owner| !owner.name.starts_with("denoiser.dynamic."));
            reporter.snapshot("after_free_denoiser_dynamic_no_sync", &owners)?;
            if reporter.allocate("text.weight", text_bytes, "raw", &mut owners)? {
                return emit(
                    json!({"event":"outcome","result":"inconclusive_raw_allocation_succeeded","scale_numerator":numerator,"scale_denominator":REFERENCE_FREE,"causal_claim":false}),
                );
            }
            reporter.snapshot("raw_oom_before_threshold_change", &owners)?;
            if reporter.allocate("text.weight", text_bytes, "unchanged_retry", &mut owners)? {
                return emit(
                    json!({"event":"outcome","result":"allocation_succeeded_on_unchanged_retry","causal_claim":false}),
                );
            }
            let _restore = RetainPool::new(pool)?;
            reporter.snapshot("threshold_max_before_sync", &owners)?;
            if reporter.allocate(
                "text.weight",
                text_bytes,
                "threshold_only_retry",
                &mut owners,
            )? {
                return emit(
                    json!({"event":"outcome","result":"allocation_succeeded_after_threshold_only","causal_claim":false}),
                );
            }
            let before = unsafe {
                pool_u64(
                    pool,
                    sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT,
                )?
            };
            reporter.synchronize("retained_pool_after_sync", &owners)?;
            let after = unsafe {
                pool_u64(
                    pool,
                    sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT,
                )?
            };
            emit(
                json!({"event":"sync_reservation_comparison","reserved_before":before,"reserved_after":after,"reserved_unchanged":before==after,"live_owners_unchanged":true}),
            )?;
            if reporter.allocate("text.weight", text_bytes, "after_sync_only", &mut owners)? {
                return emit(
                    json!({"event":"outcome","result":"allocation_succeeded_after_sync","reserved_unchanged":before==after,"causal_claim":false}),
                );
            }
            reporter.snapshot("before_explicit_trim", &owners)?;
            let start = Instant::now();
            let trim = unsafe { result::mem_pool::trim_to(pool, 0) };
            emit(
                json!({"event":"explicit_trim_to_zero","duration_us":start.elapsed().as_micros(),"result":observed(trim)}),
            )?;
            reporter.snapshot("after_explicit_trim", &owners)?;
            trim?;
            let succeeded = reporter.allocate(
                "text.weight",
                text_bytes,
                "after_explicit_trim",
                &mut owners,
            )?;
            emit(
                json!({"event":"outcome","result":if succeeded {"allocation_succeeded_after_explicit_trim"} else {"oom_persists_after_sync_and_trim"},"causal_claim":false}),
            )
        })();
        // Cleanup is explicitly separate from all observed allocation attempts.
        reporter.snapshot("before_cleanup", &owners)?;
        owners.clear();
        let cleanup = reporter.synchronize("after_cleanup", &owners);
        outcome?;
        cleanup
    }
}
