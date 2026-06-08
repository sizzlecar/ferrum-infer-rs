use ferrum_types::{Device, FerrumError, Result, RuntimeConfigEntry, RuntimeConfigSource};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeSet, HashMap};

pub const GPU_DEVICES_RAW_KEY: &str = "FERRUM_GPU_DEVICES_RAW";
pub const REQUESTED_GPU_DEVICES_KEY: &str = "FERRUM_REQUESTED_GPU_DEVICES";
pub const SELECTED_GPU_DEVICES_KEY: &str = "FERRUM_SELECTED_GPU_DEVICES";
pub const SELECTED_DISTRIBUTED_STRATEGY_KEY: &str = "FERRUM_SELECTED_DISTRIBUTED_STRATEGY";
pub const CUDA_DEVICE_COUNT_KEY: &str = "FERRUM_CUDA_DEVICE_COUNT";
pub const SELECTED_LAYER_SPLIT_PLAN_KEY: &str = "FERRUM_SELECTED_LAYER_SPLIT_PLAN";
pub const SELECTED_LAYER_SPLIT_STAGES_KEY: &str = "FERRUM_SELECTED_LAYER_SPLIT_STAGES";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CudaLayerSplitStage {
    pub stage: usize,
    pub device: usize,
    pub layer_start: usize,
    pub layer_end: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuDeviceSelection {
    pub raw_cli_value: String,
    pub requested_gpu_devices: Vec<usize>,
    pub selected_gpu_devices: Vec<usize>,
    pub cuda_device_count: usize,
    pub selected_distributed_strategy: String,
    pub selected_layer_split_plan: Option<String>,
    pub selected_layer_split_stages: Option<Vec<CudaLayerSplitStage>>,
}

impl GpuDeviceSelection {
    pub fn primary_device(&self) -> Device {
        Device::CUDA(self.selected_gpu_devices[0])
    }

    pub fn requested_csv(&self) -> String {
        join_gpu_devices(&self.requested_gpu_devices)
    }

    pub fn selected_csv(&self) -> String {
        join_gpu_devices(&self.selected_gpu_devices)
    }

    pub fn apply_model_layer_count(&mut self, num_layers: usize) -> Result<bool> {
        if self.selected_gpu_devices.len() <= 1 {
            return Ok(false);
        }
        let stages = even_layer_split_stages_for_layers(&self.selected_gpu_devices, num_layers)?;
        self.selected_layer_split_plan = Some(format_layer_split_plan(&stages));
        self.selected_layer_split_stages = Some(stages);
        Ok(true)
    }

    pub fn runtime_config_entries(&self) -> Vec<RuntimeConfigEntry> {
        vec![
            RuntimeConfigEntry::new("FERRUM_BACKEND", "cuda", RuntimeConfigSource::Cli),
            RuntimeConfigEntry::new(
                GPU_DEVICES_RAW_KEY,
                self.raw_cli_value.clone(),
                RuntimeConfigSource::Cli,
            ),
            RuntimeConfigEntry::new(
                REQUESTED_GPU_DEVICES_KEY,
                self.requested_csv(),
                RuntimeConfigSource::Cli,
            ),
            RuntimeConfigEntry::new(
                SELECTED_GPU_DEVICES_KEY,
                self.selected_csv(),
                RuntimeConfigSource::Cli,
            ),
            RuntimeConfigEntry::new(
                SELECTED_DISTRIBUTED_STRATEGY_KEY,
                self.selected_distributed_strategy.clone(),
                RuntimeConfigSource::Cli,
            ),
            RuntimeConfigEntry::new(
                CUDA_DEVICE_COUNT_KEY,
                self.cuda_device_count.to_string(),
                RuntimeConfigSource::Cli,
            ),
        ]
        .into_iter()
        .chain(self.selected_layer_split_plan.as_ref().map(|plan| {
            RuntimeConfigEntry::new(
                SELECTED_LAYER_SPLIT_PLAN_KEY,
                plan.clone(),
                RuntimeConfigSource::Cli,
            )
        }))
        .chain(self.selected_layer_split_stages.as_ref().map(|stages| {
            RuntimeConfigEntry::new(
                SELECTED_LAYER_SPLIT_STAGES_KEY,
                serde_json::to_string(stages).expect("serialize layer split stages"),
                RuntimeConfigSource::Cli,
            )
        }))
        .collect()
    }

    pub fn insert_backend_options(&self, options: &mut HashMap<String, Value>) {
        options.insert(
            "gpu_devices_raw".to_string(),
            Value::String(self.raw_cli_value.clone()),
        );
        options.insert(
            "requested_gpu_devices".to_string(),
            serde_json::json!(self.requested_gpu_devices),
        );
        options.insert(
            "selected_gpu_devices".to_string(),
            serde_json::json!(self.selected_gpu_devices),
        );
        options.insert(
            "selected_distributed_strategy".to_string(),
            Value::String(self.selected_distributed_strategy.clone()),
        );
        options.insert(
            "cuda_device_count".to_string(),
            serde_json::json!(self.cuda_device_count),
        );
        if let Some(plan) = &self.selected_layer_split_plan {
            options.insert(
                "selected_layer_split_plan".to_string(),
                Value::String(plan.clone()),
            );
        }
        if let Some(stages) = &self.selected_layer_split_stages {
            options.insert(
                "selected_layer_split_stages".to_string(),
                serde_json::to_value(stages).expect("serialize layer split stages"),
            );
        }
    }
}

pub fn resolve_cuda_gpu_devices(
    raw: Option<&str>,
    device: &Device,
) -> Result<Option<GpuDeviceSelection>> {
    let Some(raw) = raw else {
        return Ok(None);
    };
    if !matches!(device, Device::CUDA(_)) {
        return Err(FerrumError::config(format!(
            "--gpu-devices requires the cuda backend; selected backend device is {device}"
        )));
    }

    let requested = parse_gpu_devices(raw)?;
    let available = available_cuda_device_count()?;
    validate_gpu_devices_exist(&requested, available)?;
    let selected_distributed_strategy = if requested.len() > 1 {
        "layer_split"
    } else {
        "single_gpu"
    };
    let selected_layer_split_plan =
        (requested.len() > 1).then(|| even_layer_split_plan_placeholder(&requested));

    Ok(Some(GpuDeviceSelection {
        raw_cli_value: raw.to_string(),
        requested_gpu_devices: requested.clone(),
        selected_gpu_devices: requested,
        cuda_device_count: available,
        selected_distributed_strategy: selected_distributed_strategy.to_string(),
        selected_layer_split_plan,
        selected_layer_split_stages: None,
    }))
}

pub fn parse_gpu_devices(raw: &str) -> Result<Vec<usize>> {
    let raw = raw.trim();
    if raw.is_empty() {
        return Err(FerrumError::config("--gpu-devices cannot be empty"));
    }

    let mut seen = BTreeSet::new();
    let mut devices = Vec::new();
    for (idx, part) in raw.split(',').enumerate() {
        let value = part.trim();
        if value.is_empty() {
            return Err(FerrumError::config(format!(
                "--gpu-devices has an empty GPU id at position {}",
                idx + 1
            )));
        }
        if !value.chars().all(|ch| ch.is_ascii_digit()) {
            return Err(FerrumError::config(format!(
                "--gpu-devices value {value:?} is invalid; GPU ids must be non-negative integers"
            )));
        }
        let parsed = value.parse::<usize>().map_err(|_| {
            FerrumError::config(format!(
                "--gpu-devices value {value:?} is invalid; GPU ids must fit in usize"
            ))
        })?;
        if !seen.insert(parsed) {
            return Err(FerrumError::config(format!(
                "--gpu-devices contains duplicate GPU id {parsed}"
            )));
        }
        devices.push(parsed);
    }

    Ok(devices)
}

pub fn validate_gpu_devices_exist(requested: &[usize], available_count: usize) -> Result<()> {
    if available_count == 0 {
        return Err(FerrumError::device(
            "--gpu-devices was provided but no CUDA devices are available",
        ));
    }
    for id in requested {
        if *id >= available_count {
            return Err(FerrumError::device(format!(
                "--gpu-devices requested CUDA device {id}, but only {available_count} CUDA device(s) are available"
            )));
        }
    }
    Ok(())
}

pub fn join_gpu_devices(devices: &[usize]) -> String {
    devices
        .iter()
        .map(|device| device.to_string())
        .collect::<Vec<_>>()
        .join(",")
}

fn even_layer_split_plan_placeholder(devices: &[usize]) -> String {
    devices
        .iter()
        .enumerate()
        .map(|(idx, device)| format!("stage{idx}:cuda:{device}:layers=auto"))
        .collect::<Vec<_>>()
        .join(";")
}

pub fn even_layer_split_plan_for_layers(devices: &[usize], num_layers: usize) -> Result<String> {
    even_layer_split_stages_for_layers(devices, num_layers)
        .map(|stages| format_layer_split_plan(&stages))
}

pub fn even_layer_split_stages_for_layers(
    devices: &[usize],
    num_layers: usize,
) -> Result<Vec<CudaLayerSplitStage>> {
    if devices.is_empty() {
        return Err(FerrumError::config(
            "layer split requires at least one CUDA device",
        ));
    }
    if num_layers == 0 {
        return Err(FerrumError::config(
            "layer split requires a model with at least one transformer layer",
        ));
    }
    if num_layers < devices.len() {
        return Err(FerrumError::config(format!(
            "layer split requires at least as many transformer layers ({num_layers}) as CUDA devices ({})",
            devices.len()
        )));
    }

    let base = num_layers / devices.len();
    let remainder = num_layers % devices.len();
    let mut start = 0usize;
    let mut stages = Vec::with_capacity(devices.len());
    for (idx, device) in devices.iter().enumerate() {
        let count = base + usize::from(idx < remainder);
        let end = start + count - 1;
        stages.push(CudaLayerSplitStage {
            stage: idx,
            device: *device,
            layer_start: start,
            layer_end: end,
        });
        start = end + 1;
    }
    Ok(stages)
}

pub fn format_layer_split_plan(stages: &[CudaLayerSplitStage]) -> String {
    stages
        .iter()
        .map(|stage| {
            format!(
                "stage{}:cuda:{}:layers={}-{}",
                stage.stage, stage.device, stage.layer_start, stage.layer_end
            )
        })
        .collect::<Vec<_>>()
        .join(";")
}

#[cfg(feature = "cuda")]
fn available_cuda_device_count() -> Result<usize> {
    candle_core::cuda_backend::cudarc::driver::CudaContext::device_count()
        .map(|count| count as usize)
        .map_err(|err| FerrumError::device(format!("failed to query CUDA device count: {err}")))
}

#[cfg(not(feature = "cuda"))]
fn available_cuda_device_count() -> Result<usize> {
    Err(FerrumError::unsupported(
        "--gpu-devices requires ferrum built with CUDA support",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_comma_separated_gpu_devices() {
        assert_eq!(parse_gpu_devices("0,1, 2").unwrap(), vec![0, 1, 2]);
    }

    #[test]
    fn rejects_duplicate_gpu_devices() {
        let err = parse_gpu_devices("0,1,0").unwrap_err().to_string();
        assert!(err.contains("duplicate GPU id 0"));
    }

    #[test]
    fn rejects_negative_gpu_devices() {
        let err = parse_gpu_devices("0,-1").unwrap_err().to_string();
        assert!(err.contains("non-negative integers"));
    }

    #[test]
    fn rejects_missing_gpu_device_ids() {
        let err = parse_gpu_devices("0,,1").unwrap_err().to_string();
        assert!(err.contains("empty GPU id"));
    }

    #[test]
    fn validates_requested_gpu_devices_against_available_count() {
        validate_gpu_devices_exist(&[0, 1], 2).unwrap();
        let err = validate_gpu_devices_exist(&[2], 2).unwrap_err().to_string();
        assert!(err.contains("only 2 CUDA device"));
    }

    #[test]
    fn layer_split_plan_assigns_contiguous_model_layers() {
        let plan = even_layer_split_plan_for_layers(&[0, 1], 80).unwrap();
        assert_eq!(plan, "stage0:cuda:0:layers=0-39;stage1:cuda:1:layers=40-79");
    }

    #[test]
    fn layer_split_plan_distributes_remainder_to_earlier_devices() {
        let plan = even_layer_split_plan_for_layers(&[0, 1, 2], 10).unwrap();
        assert_eq!(
            plan,
            "stage0:cuda:0:layers=0-3;stage1:cuda:1:layers=4-6;stage2:cuda:2:layers=7-9"
        );
    }

    #[test]
    fn layer_split_plan_rejects_more_devices_than_layers() {
        let err = even_layer_split_plan_for_layers(&[0, 1], 1)
            .unwrap_err()
            .to_string();
        assert!(err.contains("at least as many transformer layers"));
    }

    #[test]
    fn runtime_entries_record_raw_requested_selected_and_strategy() {
        let selection = GpuDeviceSelection {
            raw_cli_value: "1".to_string(),
            requested_gpu_devices: vec![1],
            selected_gpu_devices: vec![1],
            cuda_device_count: 2,
            selected_distributed_strategy: "single_gpu".to_string(),
            selected_layer_split_plan: None,
            selected_layer_split_stages: None,
        };
        let snapshot =
            ferrum_types::RuntimeConfigSnapshot::from_entries(selection.runtime_config_entries());
        let entry = |key: &str| {
            snapshot
                .entries
                .iter()
                .find(|entry| entry.key == key)
                .unwrap_or_else(|| panic!("missing {key}"))
        };

        assert_eq!(entry("FERRUM_BACKEND").effective_value, "cuda");
        assert_eq!(entry(GPU_DEVICES_RAW_KEY).effective_value, "1");
        assert_eq!(entry(REQUESTED_GPU_DEVICES_KEY).effective_value, "1");
        assert_eq!(entry(SELECTED_GPU_DEVICES_KEY).effective_value, "1");
        assert_eq!(
            entry(SELECTED_DISTRIBUTED_STRATEGY_KEY).effective_value,
            "single_gpu"
        );
        assert_eq!(entry(CUDA_DEVICE_COUNT_KEY).effective_value, "2");
    }

    #[test]
    fn runtime_entries_record_layer_split_plan_for_multi_gpu() {
        let mut selection = GpuDeviceSelection {
            raw_cli_value: "0,1".to_string(),
            requested_gpu_devices: vec![0, 1],
            selected_gpu_devices: vec![0, 1],
            cuda_device_count: 2,
            selected_distributed_strategy: "layer_split".to_string(),
            selected_layer_split_plan: Some(even_layer_split_plan_placeholder(&[0, 1])),
            selected_layer_split_stages: None,
        };
        assert!(selection.apply_model_layer_count(80).unwrap());
        let snapshot =
            ferrum_types::RuntimeConfigSnapshot::from_entries(selection.runtime_config_entries());
        let entry = |key: &str| {
            snapshot
                .entries
                .iter()
                .find(|entry| entry.key == key)
                .unwrap_or_else(|| panic!("missing {key}"))
        };

        assert_eq!(entry(REQUESTED_GPU_DEVICES_KEY).effective_value, "0,1");
        assert_eq!(entry(SELECTED_GPU_DEVICES_KEY).effective_value, "0,1");
        assert_eq!(
            entry(SELECTED_DISTRIBUTED_STRATEGY_KEY).effective_value,
            "layer_split"
        );
        assert_eq!(
            entry(SELECTED_LAYER_SPLIT_PLAN_KEY).effective_value,
            "stage0:cuda:0:layers=0-39;stage1:cuda:1:layers=40-79"
        );
        let stages: serde_json::Value =
            serde_json::from_str(&entry(SELECTED_LAYER_SPLIT_STAGES_KEY).effective_value).unwrap();
        assert_eq!(
            stages,
            serde_json::json!([
                {"stage": 0, "device": 0, "layer_start": 0, "layer_end": 39},
                {"stage": 1, "device": 1, "layer_start": 40, "layer_end": 79}
            ])
        );
    }
}
