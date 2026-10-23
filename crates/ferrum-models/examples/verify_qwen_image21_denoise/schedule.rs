//! Explicit checkpoint sigmas only. No schedule generation or name dispatch.
use anyhow::{ensure, Result};
use ferrum_diffusion::FlowMatchEulerSchedule;
use serde::Deserialize;
use serde_json::{json, Value};

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Scheduler {
    #[serde(rename = "_class_name")]
    class: String,
    #[serde(rename = "_diffusers_version")]
    version: String,
    base_image_seq_len: u64,
    base_shift: f64,
    invert_sigmas: bool,
    max_image_seq_len: u64,
    max_shift: f64,
    num_train_timesteps: u32,
    shift: f64,
    shift_terminal: Option<f64>,
    stochastic_sampling: bool,
    time_shift_type: String,
    use_beta_sigmas: bool,
    use_dynamic_shifting: bool,
    use_exponential_sigmas: bool,
    use_karras_sigmas: bool,
}

pub struct Schedule {
    pub solver: FlowMatchEulerSchedule,
    pub declaration: Value,
}
impl Schedule {
    pub fn parse(model_index: &[u8], scheduler: &[u8]) -> Result<Self> {
        let index: Value = serde_json::from_slice(model_index)?;
        ensure!(
            index.get("scheduler")
                == Some(&json!(["diffusers", "FlowMatchEulerDiscreteScheduler"])),
            "model index must declare the supported FlowMatchEuler scheduler"
        );
        let cfg: Scheduler = serde_json::from_slice(scheduler)?;
        ensure!(
            cfg.class == "FlowMatchEulerDiscreteScheduler" && cfg.num_train_timesteps == 1000,
            "scheduler type or time units differ from the checked timestep adapter"
        );
        ensure!(
            !cfg.invert_sigmas
                && !cfg.stochastic_sampling
                && !cfg.use_beta_sigmas
                && !cfg.use_dynamic_shifting
                && !cfg.use_exponential_sigmas
                && !cfg.use_karras_sigmas
                && cfg.shift == 1.0
                && cfg.shift_terminal.is_none()
                && cfg.time_shift_type == "exponential",
            "only explicit, unshifted, deterministic, non-inverted sigma schedules are supported"
        );
        ensure!(
            cfg.base_image_seq_len > 0
                && cfg.max_image_seq_len >= cfg.base_image_seq_len
                && cfg.base_shift.is_finite()
                && cfg.max_shift.is_finite(),
            "invalid inactive scheduler geometry"
        );
        let values = index
            .get("sample_sigmas")
            .and_then(Value::as_array)
            .ok_or_else(|| anyhow::anyhow!("model index must provide explicit sample_sigmas"))?;
        ensure!(
            !values.is_empty() && values.len() <= 256,
            "explicit sigma list needs 1..256 evaluations"
        );
        // JSON numbers enter the upstream explicit schedule as FP32, before
        // sigma*1000 and terminal zero. No interpolation from a step count.
        let sigmas = values
            .iter()
            .map(|v| {
                v.as_f64()
                    .map(|v| v as f32)
                    .ok_or_else(|| anyhow::anyhow!("sigma must be a JSON number"))
            })
            .collect::<Result<Vec<_>>>()?;
        let solver = FlowMatchEulerSchedule::new(&sigmas, 0.0, 1000.0)?;
        let declaration = json!({"scheduler":"FlowMatchEulerDiscreteScheduler", "metadata_diffusers_version":cfg.version,
            "sigmas":sigmas,"terminal_sigma":0.0,"evaluation_count":solver.steps().len(),
            "sigma_bits":sigmas.iter().map(|s|s.to_bits()).collect::<Vec<_>>(),
            "timesteps":solver.steps().iter().map(|s|s.timestep().as_f32()).collect::<Vec<_>>(),
            "schedule_source":"explicit model_index sample_sigmas; shift=1; no stochastic/dynamic/inverted schedule",
            "reference":"diffusers v0.40.0 scheduling_flow_match_euler_discrete.py set_timesteps; pinned Omni c0551d prepare_timesteps",
            "qualification":"checked source contract; not deployment Torch trajectory parity"});
        Ok(Self {
            solver,
            declaration,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    pub(super) fn config() -> Value {
        json!({"_class_name":"FlowMatchEulerDiscreteScheduler","_diffusers_version":"fixture",
        "base_image_seq_len":256,"base_shift":0.5,"invert_sigmas":false,"max_image_seq_len":8192,"max_shift":0.9,
        "num_train_timesteps":1000,"shift":1.0,"shift_terminal":null,"stochastic_sampling":false,"time_shift_type":"exponential",
        "use_beta_sigmas":false,"use_dynamic_shifting":false,"use_exponential_sigmas":false,"use_karras_sigmas":false})
    }
    #[test]
    fn explicit_sigmas_keep_fp32_times_and_terminal_endpoint() {
        let index = json!({"scheduler":["diffusers","FlowMatchEulerDiscreteScheduler"],"sample_sigmas":[1.0,0.5]});
        let s = Schedule::parse(
            &serde_json::to_vec(&index).unwrap(),
            &serde_json::to_vec(&config()).unwrap(),
        )
        .unwrap();
        assert_eq!(s.solver.steps().len(), 2);
        assert_eq!(s.solver.steps()[1].timestep().as_f32(), 500.0);
        assert_eq!(s.solver.steps()[1].next_sigma(), 0.0);
        for (key, value) in [
            ("shift", json!(2)),
            ("use_dynamic_shifting", json!(true)),
            ("stochastic_sampling", json!(true)),
            ("invert_sigmas", json!(true)),
            ("shift_terminal", json!(0.01)),
            ("num_train_timesteps", json!(999)),
            ("new_schedule_rule", json!(true)),
        ] {
            let mut bad = config();
            bad[key] = value;
            assert!(
                Schedule::parse(
                    &serde_json::to_vec(&index).unwrap(),
                    &serde_json::to_vec(&bad).unwrap()
                )
                .is_err(),
                "{key}"
            );
        }
        for sigmas in [
            json!([1.0, 1.0]),
            json!([0.0]),
            json!([1.0, 1.1]),
            json!([]),
        ] {
            let mut bad = index.clone();
            bad["sample_sigmas"] = sigmas;
            assert!(Schedule::parse(
                &serde_json::to_vec(&bad).unwrap(),
                &serde_json::to_vec(&config()).unwrap()
            )
            .is_err());
        }
    }
}
