//! One managed FP8 Dense component: warm component wall time, not kernel time.
//! All tensor payloads use one Host account and the ordinary device ledger.
//! See --help. Synthetic finite dyadic data probes shapes, not model quality.
use std::any::Any;
use std::collections::BTreeMap;
#[cfg(any(feature = "cuda-cutlass-fp8", test))]
use std::collections::BTreeSet;
#[cfg(feature = "cuda-cutlass-fp8")]
use std::sync::OnceLock;

use anyhow::{ensure, Context, Result};
#[cfg(any(feature = "cuda-cutlass-fp8", test))]
use ferrum_runtime::*;
#[cfg(any(feature = "cuda-cutlass-fp8", test))]
use half::bf16;
use serde::{Deserialize, Serialize};
use serde_json::json;
#[cfg(feature = "cuda-cutlass-fp8")]
use serde_json::Value;

#[path = "bench_managed_fp8/arguments.rs"]
mod arguments;
#[cfg(feature = "cuda-cutlass-fp8")]
#[path = "bench_managed_fp8/execution.rs"]
mod execution;
#[cfg(feature = "cuda-cutlass-fp8")]
#[path = "bench_managed_fp8/family.rs"]
mod family;
#[cfg(any(feature = "cuda-cutlass-fp8", test))]
#[path = "bench_managed_fp8/pattern.rs"]
mod pattern;

#[cfg(any(feature = "cuda-cutlass-fp8", test))]
const X: &str = "value.fp8.bench.input";
#[cfg(feature = "cuda-cutlass-fp8")]
const W: &str = "value.fp8.bench.weight";
#[cfg(feature = "cuda-cutlass-fp8")]
const Y: &str = "value.fp8.bench.output";
#[cfg(feature = "cuda-cutlass-fp8")]
const PROFILE: &str = "cuda.cutlass.w8a8.bf16.rowwise_e4m3fn.v1";

#[cfg(any(feature = "cuda-cutlass-fp8", test))]
fn id<T: TryFrom<String>>(name: impl Into<String>) -> T
where
    T::Error: std::fmt::Debug,
{
    T::try_from(name.into()).expect("internal benchmark identifier")
}
#[cfg(any(feature = "cuda-cutlass-fp8", test))]
fn invalid(reason: impl Into<String>) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: reason.into(),
    }
}
#[cfg(feature = "cuda-cutlass-fp8")]
fn tensor(dimensions: Vec<u64>) -> ProgramTensorSpec {
    ProgramTensorSpec {
        dimensions,
        element_type: ElementType::Bf16,
        layout: ResolvedTensorLayout::Contiguous,
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Shape {
    m: u64,
    n: u64,
    k: u64,
}

impl Shape {
    fn extents(self) -> Result<Extents> {
        ensure!(
            [self.m, self.n, self.k]
                .iter()
                .all(|&v| v > 0 && v <= i32::MAX as u64),
            "M, N and K must be positive native i32 dimensions"
        );
        ensure!(
            self.n % 16 == 0 && self.k % 16 == 0,
            "N and K must be multiples of 16"
        );
        // Products are signed integers of magnitude <=28. This proves exact
        // FP32 accumulation under every reduction order, not merely a final-sum bound.
        ensure!(
            self.k * 28 <= 1 << 24,
            "synthetic exact-dot domain requires 28*K <= 2^24"
        );
        let bytes = |a: u64, b: u64, size: u64| -> Result<usize> {
            let v = a
                .checked_mul(b)
                .and_then(|v| v.checked_mul(size))
                .filter(|&v| v <= isize::MAX as u64)
                .context("tensor extent exceeds address space")?;
            Ok(usize::try_from(v)?)
        };
        Ok(Extents {
            input: bytes(self.m, self.k, 2)?,
            output: bytes(self.m, self.n, 2)?,
            weights: bytes(self.n, self.k, 1)?
                .checked_add(4)
                .context("weight extent overflow")?,
        })
    }
}

#[derive(Clone, Copy, Debug, Serialize)]
struct Extents {
    input: usize,
    output: usize,
    weights: usize,
}

/// Original typed native/transaction/completion owners survive a failed run.
/// This CLI does not try to invent a successful retirement or recovery.
struct Failure {
    stage: &'static str,
    message: String,
    owner: Box<dyn Any>,
}
impl Failure {
    fn retain<E: std::fmt::Display + 'static, O: 'static>(
        stage: &'static str,
        error: E,
        owners: O,
    ) -> Self {
        Self {
            stage,
            message: error.to_string(),
            owner: Box::new((error, owners)),
        }
    }
}

fn main() {
    let result = (|| -> std::result::Result<(), Failure> {
        let args = arguments::Arguments::parse(std::env::args().skip(1))
            .map_err(|e| Failure::retain("arguments", e, ()))?;
        let Some(args) = args else {
            println!("{}", arguments::USAGE);
            return Ok(());
        };
        #[cfg(feature = "cuda-cutlass-fp8")]
        {
            execution::run(args)
        }
        #[cfg(not(feature = "cuda-cutlass-fp8"))]
        {
            let _ = args;
            Err(Failure::retain(
                "feature",
                anyhow::anyhow!("requires --features cuda-cutlass-fp8"),
                (),
            ))
        }
    })();
    if let Err(failure) = result {
        eprintln!(
            "{}",
            json!({
                "schema":"ferrum.managed_fp8_benchmark.failure.v1",
                "stage":failure.stage, "error":failure.message,
                "retirement":"not claimed; original error and remaining owners retained until process exit",
            })
        );
        // No recovery authority is converted to a string and discarded. Native
        // opaque claims already have their independent process-lifetime policy.
        std::mem::forget(failure.owner);
        std::process::exit(1);
    }
}
