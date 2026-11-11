//! Allocation-only stage-switch diagnostic. No model, inference or global pool changes.
//! Defaults to dry-run; execution requires --run and an explicit logical byte budget.
use serde::Serialize;
use serde_json::json;
use std::error::Error;

type Result<T> = std::result::Result<T, Box<dyn Error>>;
const LARGE_BYTES: u64 = 17_907_597_312;
const SMALL_BYTES: u64 = 2_415_919_104;
const PRIVATE_THRESHOLD: u64 = 29_293_019_136;
const HEADROOM: u64 = 256 * 1024 * 1024;
const ANCHOR_BYTES: u64 = 2_431_317_344;
const PRIME_BYTES: u64 = 18_656_264_192;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum Allocator {
    PrivatePool,
    DefaultAsync,
    DirectDriverSync,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum Scenario {
    CleanSwitch,
    AnchoredSwitch,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum IdlePolicy {
    Retain,
    TrimIdleBeforeLarge,
}

#[derive(Debug)]
struct Args {
    run: bool,
    allocator: Allocator,
    device: usize,
    budget: Option<u64>,
    large: u64,
    small: u64,
    threshold: u64,
    rounds: usize,
    scenario: Scenario,
    idle_policy: IdlePolicy,
    anchor: u64,
    prime: u64,
}

impl Args {
    fn parse(input: impl IntoIterator<Item = String>) -> Result<Self> {
        let mut args = Self {
            run: false,
            allocator: Allocator::PrivatePool,
            device: 0,
            budget: None,
            large: LARGE_BYTES,
            small: SMALL_BYTES,
            threshold: PRIVATE_THRESHOLD,
            rounds: 1,
            scenario: Scenario::CleanSwitch,
            idle_policy: IdlePolicy::Retain,
            anchor: ANCHOR_BYTES,
            prime: PRIME_BYTES,
        };
        let mut input = input.into_iter();
        while let Some(flag) = input.next() {
            if flag == "--run" {
                args.run = true;
                continue;
            }
            let value = input
                .next()
                .ok_or_else(|| format!("missing value for {flag}"))?;
            match flag.as_str() {
                "--allocator" => {
                    args.allocator = match value.as_str() {
                        "private_pool" => Allocator::PrivatePool,
                        "default_async" => Allocator::DefaultAsync,
                        "direct_driver_sync" => Allocator::DirectDriverSync,
                        _ => return Err(format!("unknown allocator: {value}").into()),
                    }
                }
                "--scenario" => {
                    args.scenario = match value.as_str() {
                        "clean_switch" => Scenario::CleanSwitch,
                        "anchored_switch" => Scenario::AnchoredSwitch,
                        _ => return Err(format!("unknown scenario: {value}").into()),
                    }
                }
                "--idle-policy" => {
                    args.idle_policy = match value.as_str() {
                        "retain" => IdlePolicy::Retain,
                        "trim_idle_before_large" => IdlePolicy::TrimIdleBeforeLarge,
                        _ => return Err(format!("unknown idle policy: {value}").into()),
                    }
                }
                "--anchor-bytes" => args.anchor = value.parse()?,
                "--prime-bytes" => args.prime = value.parse()?,
                "--device" => args.device = value.parse()?,
                "--budget-bytes" => args.budget = Some(value.parse()?),
                "--large-bytes" => args.large = value.parse()?,
                "--small-bytes" => args.small = value.parse()?,
                "--private-threshold-bytes" => args.threshold = value.parse()?,
                "--rounds" => args.rounds = value.parse()?,
                _ => return Err(format!("unknown argument: {flag}").into()),
            }
        }
        if args.large == 0 || args.small == 0 || args.rounds == 0 {
            return Err("allocation sizes and rounds must be positive".into());
        }
        usize::try_from(args.large)?;
        usize::try_from(args.small)?;
        args.rounds
            .checked_mul(if args.scenario == Scenario::AnchoredSwitch {
                4
            } else {
                3
            })
            .ok_or("allocation count overflows")?;
        if args.allocator == Allocator::PrivatePool && args.threshold == 0 {
            return Err("private pool requires a positive threshold".into());
        }
        if args.run && args.budget.is_none() {
            return Err("--run requires --budget-bytes".into());
        }
        if args.idle_policy == IdlePolicy::TrimIdleBeforeLarge
            && (args.allocator != Allocator::PrivatePool
                || args.scenario != Scenario::AnchoredSwitch)
        {
            return Err("trim policy requires anchored_switch with a private pool".into());
        }
        if args.scenario == Scenario::AnchoredSwitch {
            if args.anchor == 0 || args.prime == 0 {
                return Err("anchor and prime must be positive".into());
            }
            usize::try_from(args.anchor)?;
            usize::try_from(args.prime)?;
        }
        let peak = args.peak_live_bytes()?;
        if let Some(budget) = args.budget {
            if budget == 0 || budget < peak {
                return Err("budget does not cover the largest live allocation".into());
            }
            if args.allocator == Allocator::PrivatePool && args.threshold > budget {
                return Err("private pool threshold exceeds the declared budget".into());
            }
        }
        Ok(args)
    }

    fn peak_live_bytes(&self) -> Result<u64> {
        let stage = self.large.max(self.small);
        match self.scenario {
            Scenario::CleanSwitch => Ok(stage),
            Scenario::AnchoredSwitch => Ok(self.prime.max(
                self.anchor
                    .checked_add(stage)
                    .ok_or("anchor plus stage bytes overflow")?,
            )),
        }
    }
}

fn main() -> Result<()> {
    let input: Vec<_> = std::env::args().skip(1).collect();
    if input.iter().any(|s| s == "--help" || s == "-h") {
        println!("cuda_allocation_phase_probe [--run --budget-bytes N] [--allocator private_pool|default_async|direct_driver_sync] [--device N] [--large-bytes N] [--small-bytes N] [--private-threshold-bytes N] [--rounds N] [--scenario clean_switch|anchored_switch] [--idle-policy retain|trim_idle_before_large] [--anchor-bytes N] [--prime-bytes N]\nAnchored: prime -> free -> retain anchor -> small -> free -> optional private trim -> large -> free -> retire anchor. Prime models a cached footprint, not an original model tensor size.\nClean round: large -> free -> small -> free -> large -> free. All modes zero-touch and synchronize at identical boundaries. Default is dry-run. Direct driver allocation is not PyTorch's caching allocator.");
        return Ok(());
    }
    let args = Args::parse(input)?;
    println!(
        "{}",
        json!({"event":"plan", "run":args.run, "allocator":args.allocator,
        "device":args.device, "logical_budget_bytes":args.budget,
        "sequence_bytes":if args.scenario == Scenario::CleanSwitch {vec![args.large,args.small,args.large]} else {vec![args.prime,args.anchor,args.small,args.large]}, "rounds":args.rounds,
        "scenario":args.scenario,"idle_policy":args.idle_policy,"peak_live_bytes":args.peak_live_bytes()?,
        "anchor_bytes":if args.scenario==Scenario::AnchoredSwitch {Some(args.anchor)} else {None},
        "prime_semantics":"cached-footprint diagnostic, not an exact production allocation history; actual driver reservation is observed, not forced",
        "private_threshold_bytes":if args.allocator == Allocator::PrivatePool {Some(args.threshold)} else {None},
        "headroom_bytes":HEADROOM,"touch":"zero_every_byte", "sync":"after_touch_and_after_free",
        "scaling":false,"retry":false,"inference":false,
        "limits":"Logical budget and driver observations are not an OS reservation or physical peak bound. Diagnostic only; direct_driver_sync does not emulate a caching allocator."})
    );
    if !args.run {
        return Ok(());
    }
    #[cfg(feature = "cuda")]
    {
        let result = cuda::run(&args);
        if let Err(error) = &result {
            eprintln!(
                "{}",
                json!({"event":"terminal","status":"failed","error":error.to_string(),"cleanup_confirmed":false,"retry":false})
            );
        }
        return result;
    }
    #[cfg(not(feature = "cuda"))]
    Err("execution requires --features cuda".into())
}

#[cfg(feature = "cuda")]
mod cuda {
    use super::*;
    use cudarc::driver::{result, sys, CudaContext, CudaStream, DriverError};
    use std::{io::Write, sync::Arc, time::Instant};

    fn emit(value: serde_json::Value) -> Result<()> {
        let mut out = std::io::stdout().lock();
        writeln!(out, "{value}")?;
        out.flush()?;
        Ok(())
    }

    unsafe fn attribute(
        pool: sys::CUmemoryPool,
        attr: sys::CUmemPool_attribute,
    ) -> std::result::Result<u64, DriverError> {
        let mut value = 0u64;
        unsafe {
            result::mem_pool::get_attribute(pool, attr, (&mut value as *mut u64).cast())?;
        }
        Ok(value)
    }

    struct Session {
        context: Arc<CudaContext>,
        stream: Arc<CudaStream>,
        allocator: Allocator,
        observed_pool: Option<sys::CUmemoryPool>,
        owned_pool: Option<sys::CUmemoryPool>,
        // Remains present through async free until the same stream proves retirement.
        live: Option<(sys::CUdeviceptr, usize)>,
        anchor: Option<(sys::CUdeviceptr, usize)>,
        retired: bool,
    }

    impl Drop for Session {
        fn drop(&mut self) {
            if !self.retired {
                // An error/unwind may mean allocation, touch or free was submitted.
                // Do not retry, destroy the pool or release its native dependencies.
                // This diagnostic terminates; these original owners stay until then.
                std::mem::forget(Arc::clone(&self.stream));
                std::mem::forget(Arc::clone(&self.context));
                eprintln!(
                    "allocation diagnostic retained uncertain native owners; cleanup not confirmed"
                );
            }
        }
    }

    impl Session {
        fn new(args: &Args) -> Result<Self> {
            let start = Instant::now();
            let context = CudaContext::new(args.device)?;
            unsafe {
                context.disable_event_tracking();
            }
            let stream = context.new_stream()?;
            let mut session = Self {
                context,
                stream,
                allocator: args.allocator,
                observed_pool: None,
                owned_pool: None,
                live: None,
                anchor: None,
                retired: false,
            };
            if args.allocator != Allocator::DirectDriverSync && !session.context.has_async_alloc() {
                return Err("selected allocator requires async allocation support".into());
            }
            match args.allocator {
                Allocator::PrivatePool => {
                    let mut properties: sys::CUmemPoolProps = unsafe { std::mem::zeroed() };
                    properties.allocType = sys::CUmemAllocationType::CU_MEM_ALLOCATION_TYPE_PINNED;
                    properties.handleTypes =
                        sys::CUmemAllocationHandleType::CU_MEM_HANDLE_TYPE_NONE;
                    properties.location.type_ = sys::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE;
                    properties.location.id = session.context.cu_device();
                    let pool = unsafe { result::mem_pool::create(&properties)? };
                    session.owned_pool = Some(pool);
                    session.observed_pool = Some(pool);
                    let mut threshold = args.threshold;
                    unsafe {
                        result::mem_pool::set_attribute(
                            pool,
                            sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
                            (&mut threshold as *mut u64).cast(),
                        )?;
                    }
                }
                Allocator::DefaultAsync => {
                    session.observed_pool =
                        Some(unsafe { result::device::get_mem_pool(session.context.cu_device())? })
                }
                Allocator::DirectDriverSync => {}
            }
            emit(
                json!({"event":"setup", "duration_us":start.elapsed().as_micros(),"async_supported":session.context.has_async_alloc(),"event_tracking":session.context.is_event_tracking()}),
            )?;
            Ok(session)
        }

        fn snapshot(&self, phase: &str) -> Result<()> {
            let (free, total) = self.context.mem_get_info()?;
            let pool = if let Some(pool) = self.observed_pool {
                use sys::CUmemPool_attribute as A;
                let reserved = unsafe { attribute(pool, A::CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT)? };
                let used = unsafe { attribute(pool, A::CU_MEMPOOL_ATTR_USED_MEM_CURRENT)? };
                Some(json!({"identity":format!("{pool:p}"),
                    "reserved_bytes":reserved,"used_bytes":used,"unused_bytes":reserved.saturating_sub(used),
                    "threshold_bytes":unsafe{attribute(pool,A::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD)?}}))
            } else {
                None
            };
            emit(
                json!({"event":"snapshot","phase":phase,"free_bytes":free,"total_bytes":total,"pool":pool,"live_bytes":self.live.map_or(0,|(_,n)|n)+self.anchor.map_or(0,|(_,n)|n),"anchor_bytes":self.anchor.map_or(0,|(_,n)|n)}),
            )
        }

        fn timed<T>(
            &self,
            phase: &str,
            operation: &str,
            action: impl FnOnce() -> std::result::Result<T, DriverError>,
        ) -> Result<T> {
            let start = Instant::now();
            let result = action();
            emit(
                json!({"event":"operation","phase":phase,"operation":operation,"host_us":start.elapsed().as_micros(),"success":result.is_ok(),"error":result.as_ref().err().map(|e|format!("{e:?}"))}),
            )?;
            Ok(result?)
        }

        fn allocate_ready(&mut self, bytes: u64, label: &str) -> Result<()> {
            assert!(self.live.is_none());
            let bytes = usize::try_from(bytes)?;
            self.snapshot(&format!("{label}.before"))?;
            let start = Instant::now();
            let ptr = self.timed(label, "allocate", || unsafe {
                match self.allocator {
                    Allocator::PrivatePool => result::mem_pool::alloc_async(
                        self.owned_pool.unwrap(),
                        bytes,
                        self.stream.cu_stream(),
                    ),
                    Allocator::DefaultAsync => result::malloc_async(self.stream.cu_stream(), bytes),
                    Allocator::DirectDriverSync => result::malloc_sync(bytes),
                }
            })?;
            self.live = Some((ptr, bytes));
            self.timed(label, "zero_touch_enqueue", || unsafe {
                result::memset_d8_async(ptr, 0, bytes, self.stream.cu_stream())
            })?;
            self.timed(label, "touch_synchronize", || self.stream.synchronize())?;
            emit(
                json!({"event":"allocation_ready","phase":label,"bytes":bytes,"pointer":format!("0x{ptr:x}"),"host_us":start.elapsed().as_micros()}),
            )?;
            self.snapshot(&format!("{label}.ready"))
        }

        fn retire_live(&mut self, label: &str) -> Result<()> {
            let (ptr, bytes) = self.live.ok_or("no live allocation to retire")?;
            let retire = Instant::now();
            self.timed(label, "free", || unsafe {
                match self.allocator {
                    Allocator::DirectDriverSync => result::free_sync(ptr),
                    _ => result::free_async(ptr, self.stream.cu_stream()),
                }
            })?;
            self.timed(label, "free_synchronize", || self.stream.synchronize())?;
            self.live = None;
            emit(
                json!({"event":"allocation_retired","phase":label,"bytes":bytes,"host_us":retire.elapsed().as_micros()}),
            )?;
            self.snapshot(&format!("{label}.after_free"))
        }

        fn phase(&mut self, bytes: u64, label: &str) -> Result<()> {
            self.allocate_ready(bytes, label)?;
            self.retire_live(label)
        }

        fn install_anchor(&mut self, bytes: u64, label: &str) -> Result<()> {
            if self.anchor.is_some() {
                return Err("anchor already installed".into());
            }
            self.allocate_ready(bytes, label)?;
            self.anchor = self.live.take();
            self.snapshot(&format!("{label}.retained"))
        }

        fn retire_anchor(&mut self, label: &str) -> Result<()> {
            if self.live.is_some() {
                return Err("stage must retire before anchor".into());
            }
            self.live = self.anchor.take();
            self.retire_live(label)
        }

        fn trim_idle(&mut self, label: &str) -> Result<()> {
            if self.allocator != Allocator::PrivatePool
                || self.live.is_some()
                || self.anchor.is_none()
            {
                return Err(
                    "trim requires the original private pool and only the live anchor".into(),
                );
            }
            let pool = self.owned_pool.ok_or("private pool owner missing")?;
            self.timed(label, "trim_idle", || unsafe {
                result::mem_pool::trim_to(pool, 0)
            })?;
            self.snapshot(&format!("{label}.after_trim"))
        }

        fn finish(mut self) -> Result<()> {
            if self.live.is_some() || self.anchor.is_some() {
                return Err("live allocation prevents retirement".into());
            }
            self.timed("cleanup", "synchronize", || self.stream.synchronize())?;
            if let Some(pool) = self.owned_pool {
                self.timed("cleanup", "destroy_private_pool", || unsafe {
                    result::mem_pool::destroy(pool)
                })?;
                self.owned_pool = None;
                self.observed_pool = None;
            }
            self.retired = true;
            emit(
                json!({"event":"terminal","status":"completed","all_allocations_retired":true,"private_pool_destroyed":self.allocator==Allocator::PrivatePool}),
            )
        }
    }

    pub fn run(args: &Args) -> Result<()> {
        let mut session = Session::new(args)?;
        let (free, _) = session.context.mem_get_info()?;
        let required = args
            .peak_live_bytes()?
            .checked_add(HEADROOM)
            .ok_or("headroom overflows")?;
        if required > free as u64 {
            return Err(
                "observed free memory does not cover exact unscaled allocation plus headroom"
                    .into(),
            );
        }
        for round in 0..args.rounds {
            match args.scenario {
                Scenario::CleanSwitch => {
                    for (index, bytes) in
                        [args.large, args.small, args.large].into_iter().enumerate()
                    {
                        session.phase(bytes, &format!("round{round}.stage{index}"))?;
                    }
                }
                Scenario::AnchoredSwitch => {
                    session.phase(args.prime, &format!("round{round}.prime"))?;
                    session.install_anchor(args.anchor, &format!("round{round}.anchor"))?;
                    session.phase(args.small, &format!("round{round}.small"))?;
                    let label = format!("round{round}.large");
                    session.snapshot(&format!("{label}.before_policy"))?;
                    let transition = Instant::now();
                    if args.idle_policy == IdlePolicy::TrimIdleBeforeLarge {
                        session.trim_idle(&label)?;
                    }
                    session.allocate_ready(args.large, &label)?;
                    emit(
                        json!({"event":"transition_ready","phase":label,"idle_policy":args.idle_policy,"host_us":transition.elapsed().as_micros(),"includes":"trim if selected, allocation, touch, synchronize and diagnostic emission"}),
                    )?;
                    session.retire_live(&label)?;
                    session.retire_anchor(&format!("round{round}.anchor_cleanup"))?;
                }
            }
        }
        session.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn parse(s: &str) -> Result<Args> {
        Args::parse(s.split_whitespace().map(str::to_owned))
    }
    #[test]
    fn explicit_modes_preserve_exact_shapes_and_require_a_sufficient_budget() {
        for mode in ["private_pool", "default_async", "direct_driver_sync"] {
            let a = parse(&format!(
                "--run --allocator {mode} --budget-bytes 31138512896"
            ))
            .unwrap();
            assert_eq!(
                [a.large, a.small, a.large],
                [LARGE_BYTES, SMALL_BYTES, LARGE_BYTES]
            );
        }
        assert!(parse("--run").is_err());
        assert!(parse("--run --budget-bytes 1").is_err());
        assert!(parse("--allocator unknown").is_err());
        assert!(parse("--rounds 0").is_err());
    }
    #[test]
    fn dry_run_and_small_diagnostic_shapes_do_not_create_a_context() {
        assert!(!parse("").unwrap().run);
        let a=parse("--run --budget-bytes 8192 --large-bytes 4096 --small-bytes 1024 --private-threshold-bytes 8192 --rounds 2").unwrap();
        assert_eq!((a.large, a.small, a.rounds), (4096, 1024, 2));
        assert!(parse("--budget-bytes 4096 --large-bytes 4096 --small-bytes 1024").is_err());
    }
}

#[cfg(test)]
mod anchored_tests {
    use super::*;
    fn parse(s: &str) -> Result<Args> {
        Args::parse(s.split_whitespace().map(str::to_owned))
    }
    #[test]
    fn anchor_coexistence_is_budgeted_and_trim_never_targets_current_pool() {
        let base = "--run --scenario anchored_switch --budget-bytes 31138512896";
        for policy in ["retain", "trim_idle_before_large"] {
            let a = parse(&format!("{base} --idle-policy {policy}")).unwrap();
            assert_eq!(a.peak_live_bytes().unwrap(), ANCHOR_BYTES + LARGE_BYTES);
        }
        assert!(parse("--scenario anchored_switch --budget-bytes 20000000000 --private-threshold-bytes 19000000000").is_err());
        assert!(parse("--scenario anchored_switch --allocator default_async --idle-policy trim_idle_before_large").is_err());
        assert!(parse("--idle-policy trim_idle_before_large").is_err());
        assert!(parse("--scenario anchored_switch --anchor-bytes 18446744073709551615").is_err());
    }
}
