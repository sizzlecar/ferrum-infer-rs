//! Device-aware startup policy. Costs come from the compiled execution plan,
//! including its provider workspaces and physical storage layouts.

use crate::{RuntimeConfigSnapshot, RuntimeConfigSource};
use serde::{Deserialize, Serialize};

/// Sampled before model upload. Available bytes are additional allocatable
/// memory, already accounting for other processes and platform ceilings.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeviceMemorySnapshot {
    pub capacity_bytes: u64,
    pub available_bytes: u64,
    pub source: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StartupMemoryRequest {
    pub device: DeviceMemorySnapshot,
    pub usable_capacity_bytes: u64,
    pub context_is_explicit: bool,
    pub sequences_is_explicit: bool,
    pub batch_is_explicit: bool,
}

impl StartupMemoryRequest {
    pub fn from_snapshot(
        device: DeviceMemorySnapshot,
        utilization: f32,
        requested: &RuntimeConfigSnapshot,
    ) -> Result<Self, String> {
        if device.capacity_bytes == 0 || device.available_bytes > device.capacity_bytes {
            return Err("invalid device memory sample: expected available <= capacity > 0".into());
        }
        if !utilization.is_finite() || utilization <= 0.0 || utilization > 1.0 {
            return Err("memory utilization must be in (0, 1]".into());
        }
        let explicit = |key: &str| {
            requested.entries.iter().any(|entry| {
                entry.key == key
                    && matches!(
                        entry.source,
                        RuntimeConfigSource::Cli
                            | RuntimeConfigSource::Env
                            | RuntimeConfigSource::ConfigFile
                            | RuntimeConfigSource::ScriptCase
                    )
            })
        };
        let budget = requested
            .entries
            .iter()
            .find(|entry| entry.key == "FERRUM_RUNTIME_MEMORY_BUDGET_BYTES")
            .map(|entry| {
                entry
                    .effective_value
                    .parse::<u64>()
                    .map_err(|_| "runtime memory budget must be a positive byte count".to_owned())
            })
            .transpose()?;
        let usable_capacity_bytes = budget.unwrap_or_else(|| {
            ((device.available_bytes as f64) * f64::from(utilization)).floor() as u64
        });
        if usable_capacity_bytes == 0 || usable_capacity_bytes > device.available_bytes {
            return Err(format!(
                "runtime memory budget {usable_capacity_bytes} bytes cannot fit the sampled {} available bytes (device ceiling {} bytes; {})",
                device.available_bytes, device.capacity_bytes, device.source
            ));
        }
        Ok(Self {
            device,
            usable_capacity_bytes,
            context_is_explicit: explicit("FERRUM_MAX_MODEL_LEN") || explicit("FERRUM_KV_CAPACITY"),
            sequences_is_explicit: explicit("FERRUM_PAGED_MAX_SEQS"),
            batch_is_explicit: explicit("FERRUM_MAX_BATCHED_TOKENS"),
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct StartupResourceLimits {
    pub context_tokens: usize,
    pub max_sequences: usize,
    pub max_batch_tokens: usize,
}

/// Independent capacity probes, not a promise that every admitted sequence
/// can simultaneously occupy the maximum context. Runtime admission remains
/// responsible for the actual mixture of live requests.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StartupWorkload {
    Prefill {
        context_tokens: usize,
        chunk_tokens: usize,
    },
    Decode {
        /// Request allocation ceiling; sequence state is probed at a one-token
        /// committed frontier, not a full context for every sequence.
        context_tokens: usize,
        active_sequences: usize,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StartupMemoryPlan {
    pub request: StartupMemoryRequest,
    pub requested: StartupResourceLimits,
    pub selected: StartupResourceLimits,
    pub context_peak_bytes: u64,
    pub decode_peak_bytes: u64,
    pub reasons: Vec<String>,
}

impl StartupMemoryPlan {
    pub fn apply_to_engine_config(&self, engine: &mut crate::EngineConfig) -> Result<(), String> {
        engine.runtime.max_model_len = Some(self.selected.context_tokens);
        engine.scheduler.max_running_requests = self.selected.max_sequences;
        engine.batching.max_num_batched_tokens = self.selected.max_batch_tokens;
        engine.memory.usable_capacity_bytes = Some(
            usize::try_from(self.request.usable_capacity_bytes)
                .map_err(|_| "runtime memory budget exceeds the process address space")?,
        );
        engine.runtime.startup_memory_plan = Some(self.clone());
        Ok(())
    }
}

/// Find supported limits using a monotone upper bound supplied by the actual
/// compiled plan. No tensors are allocated by this search. Provider errors are
/// errors, never interpreted as evidence that a smaller request will work.
pub fn fit_startup_resources(
    request: &StartupMemoryRequest,
    target: StartupResourceLimits,
    mut peak_bytes: impl FnMut(StartupWorkload) -> Result<u64, String>,
) -> Result<StartupMemoryPlan, String> {
    if target.context_tokens == 0 || target.max_sequences == 0 || target.max_batch_tokens == 0 {
        return Err("startup context, sequence and batch limits must be positive".into());
    }
    let budget = request.usable_capacity_bytes;
    let prefill = |context_tokens: usize, chunk: usize| StartupWorkload::Prefill {
        context_tokens,
        chunk_tokens: chunk.min(context_tokens).max(1),
    };
    let batch_floor = if request.sequences_is_explicit {
        target.max_sequences
    } else {
        1
    };
    let batch_target = if request.batch_is_explicit {
        target.max_batch_tokens
    } else {
        target.max_batch_tokens.max(batch_floor)
    };
    if batch_target < batch_floor {
        return Err("explicit batch token limit is smaller than explicit sequence limit".into());
    }
    let batch_cost =
        |batch: usize, cost: &mut dyn FnMut(StartupWorkload) -> Result<u64, String>| {
            let context = if request.context_is_explicit {
                target.context_tokens
            } else {
                batch.min(target.context_tokens)
            };
            cost(prefill(context, batch))
        };
    let batch = if request.batch_is_explicit {
        require_fit(
            "explicit batch/context",
            batch_cost(batch_target, &mut peak_bytes)?,
            budget,
        )?;
        batch_target
    } else {
        largest_fit(batch_floor, batch_target, budget, |value| {
            batch_cost(value, &mut peak_bytes)
        })?
    };
    let context = if request.context_is_explicit {
        require_fit(
            "explicit context",
            peak_bytes(prefill(target.context_tokens, batch))?,
            budget,
        )?;
        target.context_tokens
    } else {
        largest_fit(1, target.context_tokens, budget, |tokens| {
            let prefill_peak = peak_bytes(prefill(tokens, batch))?;
            let decode_peak = peak_bytes(StartupWorkload::Decode {
                context_tokens: tokens,
                active_sequences: if request.sequences_is_explicit {
                    target.max_sequences
                } else {
                    1
                },
            })?;
            Ok(prefill_peak.max(decode_peak))
        })?
    };
    let decode = |active_sequences| StartupWorkload::Decode {
        context_tokens: context,
        active_sequences,
    };
    let sequences = if request.sequences_is_explicit {
        require_fit(
            "explicit concurrency",
            peak_bytes(decode(target.max_sequences))?,
            budget,
        )?;
        target.max_sequences
    } else {
        largest_fit(1, target.max_sequences.min(batch), budget, |count| {
            peak_bytes(decode(count))
        })?
    };
    let selected = StartupResourceLimits {
        context_tokens: context,
        max_sequences: sequences,
        max_batch_tokens: batch,
    };
    let mut reasons = Vec::new();
    for (name, before, after) in [
        ("context tokens", target.context_tokens, context),
        ("concurrent sequences", target.max_sequences, sequences),
        ("batch tokens", target.max_batch_tokens, batch),
    ] {
        if before != after {
            reasons.push(format!("{name}: {before} -> {after}, compiled resource plan fitted to {budget} usable bytes"));
        }
    }
    Ok(StartupMemoryPlan {
        request: request.clone(),
        requested: target,
        selected,
        context_peak_bytes: peak_bytes(prefill(context, batch))?,
        decode_peak_bytes: peak_bytes(decode(sequences))?,
        reasons,
    })
}

fn require_fit(label: &str, required: u64, budget: u64) -> Result<(), String> {
    if required > budget {
        return Err(format!(
            "{label} requires {required} bytes, exceeding the {budget}-byte runtime budget"
        ));
    }
    Ok(())
}

fn largest_fit(
    minimum: usize,
    maximum: usize,
    budget: u64,
    mut cost: impl FnMut(usize) -> Result<u64, String>,
) -> Result<usize, String> {
    require_fit("minimum runnable configuration", cost(minimum)?, budget)?;
    let mut low = minimum;
    let mut high = maximum;
    while low < high {
        let middle = low + (high - low) / 2 + (high - low) % 2;
        if cost(middle)? <= budget {
            low = middle;
        } else {
            high = middle - 1;
        }
    }
    Ok(low)
}

#[cfg(test)]
mod tests;
