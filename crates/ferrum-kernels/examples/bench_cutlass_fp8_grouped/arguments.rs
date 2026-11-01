use super::Result;
use serde::Serialize;
use std::{collections::BTreeMap, path::PathBuf};

pub const USAGE: &str = "bench_cutlass_fp8_grouped --run --device ORDINAL --m M --n N --k K --branches 2|3 --warmup COUNT --iterations COUNT --device-bytes BYTES --native-reserve-bytes BYTES --host-bytes BYTES --output NEW_JSONL [--profile-phases]
Requires cuda-cutlass-fp8 with the grouped native exports. Run only on an available GPU.
One iteration is A/B/B/A (A: separate quantization, B: shared quantization).
All four samples are retained; warmup executes both paths. Inputs/weights remain fixed.
Full output equality and finite values are checked before and after measurement.
Optional phase events measure shared quantization and its following GEMMs separately.
Device budget includes owned payload plus a declared context/native reserve; it is not measured peak VRAM.
Host budget bounds tensor payloads only. No HTTP, model quality, or latency-SLO claim.";

#[derive(Debug, Clone, Serialize)]
pub struct Arguments {
    pub device: usize,
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub branches: usize,
    pub warmup: u32,
    pub iterations: u32,
    pub device_bytes: u64,
    pub native_reserve_bytes: u64,
    pub host_bytes: u64,
    pub output: PathBuf,
    pub profile_phases: bool,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct Extents {
    pub input: usize,
    pub weight: usize,
    pub output: usize,
    pub host_payload_peak_bound: u64,
}

impl Arguments {
    pub fn parse(values: impl IntoIterator<Item = String>) -> Result<Option<Self>> {
        let mut values = values.into_iter().peekable();
        if values.peek().is_some_and(|s| s == "--help" || s == "-h") {
            values.next();
            if values.next().is_some() {
                return Err("--help accepts no other arguments".into());
            }
            return Ok(None);
        }
        let mut options = BTreeMap::new();
        let mut run = false;
        let mut profile_phases = false;
        while let Some(key) = values.next() {
            match key.as_str() {
                "--run" if !run => run = true,
                "--profile-phases" if !profile_phases => profile_phases = true,
                "--run" | "--profile-phases" => return Err("duplicate flag".into()),
                _ => {
                    let value = values.next().ok_or("option needs a value")?;
                    if options.insert(key, value).is_some() {
                        return Err("duplicate option".into());
                    }
                }
            }
        }
        if !run {
            return Err("explicit --run is required".into());
        }
        let output = options.remove("--output").ok_or("missing --output")?.into();
        let mut number = |key: &str| -> Result<u64> {
            Ok(options
                .remove(key)
                .ok_or_else(|| format!("missing {key}"))?
                .parse()?)
        };
        let args = Self {
            device: number("--device")?.try_into()?,
            m: number("--m")?.try_into()?,
            n: number("--n")?.try_into()?,
            k: number("--k")?.try_into()?,
            branches: number("--branches")?.try_into()?,
            warmup: number("--warmup")?.try_into()?,
            iterations: number("--iterations")?.try_into()?,
            device_bytes: number("--device-bytes")?,
            native_reserve_bytes: number("--native-reserve-bytes")?,
            host_bytes: number("--host-bytes")?,
            output,
            profile_phases,
        };
        if !options.is_empty() {
            return Err(format!("unknown options: {:?}", options.keys()).into());
        }
        args.extents()?;
        Ok(Some(args))
    }

    pub fn extents(&self) -> Result<Extents> {
        if self.device > i32::MAX as usize
            || !matches!(self.branches, 2 | 3)
            || self.iterations == 0
            || [self.m, self.n, self.k]
                .iter()
                .any(|&x| x == 0 || x > i32::MAX as u32)
            || self.n % 16 != 0
            || self.k % 16 != 0
            || self.native_reserve_bytes == 0
            || self.device_bytes <= self.native_reserve_bytes
        {
            return Err("invalid geometry, repetitions, or device budget".into());
        }
        let bytes = |a: u32, b: u32, element: u64| -> Result<usize> {
            let bytes = u64::from(a)
                .checked_mul(u64::from(b))
                .and_then(|v| v.checked_mul(element))
                .filter(|&v| v <= isize::MAX as u64)
                .ok_or("tensor extent exceeds address space")?;
            Ok(bytes.try_into()?)
        };
        let input = bytes(self.m, self.k, 2)?;
        let weight = bytes(self.n, self.k, 1)?;
        let output = bytes(self.m, self.n, 2)?;
        // Baseline outputs stay on Host while one candidate branch is read.
        // Setup uploads one input or one weight at a time and retires it first.
        let comparison = (output as u64)
            .checked_mul(self.branches as u64 + 1)
            .ok_or("Host comparison extent overflows")?;
        let host_payload_peak_bound = comparison.max(input as u64).max(weight as u64).max(4);
        if self.host_bytes < host_payload_peak_bound {
            return Err(format!(
                "Host budget {} below tensor payload bound {host_payload_peak_bound}",
                self.host_bytes
            )
            .into());
        }
        Ok(Extents {
            input,
            weight,
            output,
            host_payload_peak_bound,
        })
    }

    #[cfg(feature = "cuda-cutlass-fp8")]
    pub fn device_payload_bytes(&self, scratch: u64) -> Result<u64> {
        let e = self.extents()?;
        (e.weight as u64)
            .checked_add(e.output as u64)
            .and_then(|v| v.checked_add(4))
            .and_then(|v| v.checked_mul(self.branches as u64))
            .and_then(|v| v.checked_add(e.input as u64))
            .and_then(|v| v.checked_add(scratch))
            .ok_or_else(|| "device payload extent overflows".into())
    }
}
