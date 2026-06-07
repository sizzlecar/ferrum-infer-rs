use crate::parallel::{LayerDistribution, LayerRange, ParallelConfig, ParallelismType};
use ferrum_models::models::llama_family::LlamaFamilyLayerStageConfig;
use ferrum_types::Device;
use ferrum_types::{FerrumError, Result};
use serde::Deserialize;

#[derive(Debug, Clone)]
pub(crate) struct CudaLayerSplitExecutionPlan {
    pub(crate) parallel_config: ParallelConfig,
    pub(crate) layer_distribution: LayerDistribution,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ParsedLayerSplitPlan {
    pub(crate) stages: Vec<LayerSplitStage>,
}

impl ParsedLayerSplitPlan {
    pub(crate) fn selected_devices(&self) -> Vec<usize> {
        self.stages.iter().map(|stage| stage.device).collect()
    }

    pub(crate) fn total_layers(&self) -> usize {
        self.stages
            .last()
            .map(|stage| stage.layer_end + 1)
            .unwrap_or(0)
    }

    pub(crate) fn to_execution_plan(&self) -> CudaLayerSplitExecutionPlan {
        let devices = self
            .stages
            .iter()
            .map(|stage| Device::CUDA(stage.device))
            .collect::<Vec<_>>();
        let stage_layers = self
            .stages
            .iter()
            .map(|stage| LayerRange::new(stage.layer_start, stage.layer_end + 1))
            .collect::<Vec<_>>();
        CudaLayerSplitExecutionPlan {
            parallel_config: ParallelConfig {
                parallelism_type: ParallelismType::Pipeline,
                devices,
                tensor_parallel_size: 1,
                pipeline_parallel_size: self.stages.len(),
                communication_backend: "cuda-peer".to_string(),
                ..ParallelConfig::default()
            },
            layer_distribution: LayerDistribution {
                stage_layers,
                stage_memory: vec![0; self.stages.len()],
            },
        }
    }

    pub(crate) fn to_llama_stage_configs(&self) -> Vec<LlamaFamilyLayerStageConfig> {
        self.stages
            .iter()
            .enumerate()
            .map(|(position, stage)| {
                LlamaFamilyLayerStageConfig::pipeline_stage(
                    stage.layer_start..stage.layer_end + 1,
                    position == 0,
                    position + 1 == self.stages.len(),
                )
            })
            .collect()
    }

    pub(crate) fn llama_stage_config_for_device(
        &self,
        device: usize,
    ) -> Result<LlamaFamilyLayerStageConfig> {
        let stage_configs = self.to_llama_stage_configs();
        let (position, _stage) = self
            .stages
            .iter()
            .enumerate()
            .find(|(_, stage)| stage.device == device)
            .ok_or_else(|| {
                FerrumError::config(format!(
                    "selected_layer_split_plan does not include CUDA device {device}"
                ))
            })?;
        Ok(stage_configs[position].clone())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct LayerSplitStage {
    pub(crate) stage_index: usize,
    pub(crate) device: usize,
    pub(crate) layer_start: usize,
    pub(crate) layer_end: usize,
}

#[derive(Debug, Clone, Deserialize)]
struct StageDocument {
    stage: usize,
    device: usize,
    layer_start: usize,
    layer_end: usize,
}

pub(crate) fn parse_layer_split_stage_documents(
    value: &serde_json::Value,
) -> Result<ParsedLayerSplitPlan> {
    let docs: Vec<StageDocument> = serde_json::from_value(value.clone()).map_err(|err| {
        FerrumError::config(format!("invalid selected_layer_split_stages: {err}"))
    })?;
    let stages = docs
        .into_iter()
        .map(|doc| LayerSplitStage {
            stage_index: doc.stage,
            device: doc.device,
            layer_start: doc.layer_start,
            layer_end: doc.layer_end,
        })
        .collect::<Vec<_>>();
    validate_stage_positions(&stages)?;
    validate_contiguous_layers(&stages)?;
    Ok(ParsedLayerSplitPlan { stages })
}

fn validate_stage_positions(stages: &[LayerSplitStage]) -> Result<()> {
    for (position, stage) in stages.iter().enumerate() {
        if stage.stage_index != position {
            return Err(FerrumError::config(format!(
                "layer split stage index {} must match position {}",
                stage.stage_index, position
            )));
        }
    }
    Ok(())
}

pub(crate) fn parse_layer_split_plan(raw: &str) -> Result<ParsedLayerSplitPlan> {
    let raw = raw.trim();
    if raw.is_empty() {
        return Err(FerrumError::config(
            "selected_layer_split_plan cannot be empty",
        ));
    }

    let mut stages = Vec::new();
    for (position, part) in raw.split(';').enumerate() {
        let stage = parse_stage(part.trim(), position)?;
        stages.push(stage);
    }
    validate_stage_positions(&stages)?;
    validate_contiguous_layers(&stages)?;
    Ok(ParsedLayerSplitPlan { stages })
}

pub(crate) fn validate_layer_split_plan_for_devices(
    plan: &ParsedLayerSplitPlan,
    selected_devices: &[usize],
) -> Result<()> {
    let plan_devices = plan.selected_devices();
    if plan_devices != selected_devices {
        return Err(FerrumError::config(format!(
            "selected_layer_split_plan devices {plan_devices:?} do not match selected_gpu_devices {selected_devices:?}"
        )));
    }
    Ok(())
}

fn parse_stage(raw: &str, position: usize) -> Result<LayerSplitStage> {
    if raw.is_empty() {
        return Err(FerrumError::config(format!(
            "selected_layer_split_plan has an empty stage at position {}",
            position + 1
        )));
    }

    let mut parts = raw.split(':');
    let stage_part = parts.next().unwrap_or_default();
    let backend_part = parts.next().unwrap_or_default();
    let device_part = parts.next().unwrap_or_default();
    let layers_part = parts.next().unwrap_or_default();
    if parts.next().is_some() || backend_part != "cuda" {
        return Err(FerrumError::config(format!(
            "invalid layer split stage {raw:?}; expected stageN:cuda:DEVICE:layers=START-END"
        )));
    }

    let stage_index = stage_part
        .strip_prefix("stage")
        .and_then(|value| value.parse::<usize>().ok())
        .ok_or_else(|| {
            FerrumError::config(format!(
                "invalid layer split stage {raw:?}; expected stageN prefix"
            ))
        })?;
    if stage_index != position {
        return Err(FerrumError::config(format!(
            "layer split stage index {stage_index} must match position {position}"
        )));
    }

    let device = device_part.parse::<usize>().map_err(|_| {
        FerrumError::config(format!(
            "invalid CUDA device {device_part:?} in layer split stage {raw:?}"
        ))
    })?;
    let range = layers_part.strip_prefix("layers=").ok_or_else(|| {
        FerrumError::config(format!(
            "invalid layer split stage {raw:?}; expected layers=START-END"
        ))
    })?;
    let (start, end) = range.split_once('-').ok_or_else(|| {
        FerrumError::config(format!("invalid layer range {range:?}; expected START-END"))
    })?;
    let layer_start = start
        .parse::<usize>()
        .map_err(|_| FerrumError::config(format!("invalid layer range start {start:?}")))?;
    let layer_end = end
        .parse::<usize>()
        .map_err(|_| FerrumError::config(format!("invalid layer range end {end:?}")))?;
    if layer_end < layer_start {
        return Err(FerrumError::config(format!(
            "layer range end {layer_end} is before start {layer_start}"
        )));
    }

    Ok(LayerSplitStage {
        stage_index,
        device,
        layer_start,
        layer_end,
    })
}

fn validate_contiguous_layers(stages: &[LayerSplitStage]) -> Result<()> {
    if stages.is_empty() {
        return Err(FerrumError::config(
            "selected_layer_split_plan must contain at least one stage",
        ));
    }
    let mut expected_start = 0usize;
    for stage in stages {
        if stage.layer_start != expected_start {
            return Err(FerrumError::config(format!(
                "layer split stage{} starts at {}, expected {}",
                stage.stage_index, stage.layer_start, expected_start
            )));
        }
        expected_start = stage.layer_end + 1;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_contiguous_two_gpu_plan() {
        let plan =
            parse_layer_split_plan("stage0:cuda:0:layers=0-39;stage1:cuda:1:layers=40-79").unwrap();

        assert_eq!(plan.selected_devices(), vec![0, 1]);
        assert_eq!(plan.total_layers(), 80);
        assert_eq!(
            plan.stages[0],
            LayerSplitStage {
                stage_index: 0,
                device: 0,
                layer_start: 0,
                layer_end: 39,
            }
        );
    }

    #[test]
    fn rejects_auto_layer_ranges() {
        let err = parse_layer_split_plan("stage0:cuda:0:layers=auto;stage1:cuda:1:layers=auto")
            .unwrap_err()
            .to_string();
        assert!(err.contains("expected START-END"));
    }

    #[test]
    fn rejects_layer_gaps() {
        let err = parse_layer_split_plan("stage0:cuda:0:layers=0-39;stage1:cuda:1:layers=41-79")
            .unwrap_err()
            .to_string();
        assert!(err.contains("expected 40"));
    }

    #[test]
    fn rejects_device_mismatch() {
        let plan =
            parse_layer_split_plan("stage0:cuda:0:layers=0-39;stage1:cuda:1:layers=40-79").unwrap();
        let err = validate_layer_split_plan_for_devices(&plan, &[0, 2])
            .unwrap_err()
            .to_string();
        assert!(err.contains("do not match selected_gpu_devices"));
    }

    #[test]
    fn parses_structured_stage_documents() {
        let value = serde_json::json!([
            {"stage": 0, "device": 0, "layer_start": 0, "layer_end": 39},
            {"stage": 1, "device": 1, "layer_start": 40, "layer_end": 79}
        ]);
        let plan = parse_layer_split_stage_documents(&value).unwrap();
        assert_eq!(plan.selected_devices(), vec![0, 1]);
        assert_eq!(plan.total_layers(), 80);
    }

    #[test]
    fn converts_to_pipeline_execution_plan() {
        let plan =
            parse_layer_split_plan("stage0:cuda:0:layers=0-39;stage1:cuda:1:layers=40-79").unwrap();
        let execution = plan.to_execution_plan();

        assert_eq!(
            execution.parallel_config.parallelism_type,
            ParallelismType::Pipeline
        );
        assert_eq!(
            execution.parallel_config.devices,
            vec![Device::CUDA(0), Device::CUDA(1)]
        );
        assert_eq!(execution.parallel_config.tensor_parallel_size, 1);
        assert_eq!(execution.parallel_config.pipeline_parallel_size, 2);
        assert_eq!(execution.parallel_config.communication_backend, "cuda-peer");
        assert_eq!(execution.layer_distribution.stage_layers.len(), 2);
        assert_eq!(execution.layer_distribution.stage_layers[0].start, 0);
        assert_eq!(execution.layer_distribution.stage_layers[0].end, 40);
        assert_eq!(execution.layer_distribution.stage_layers[1].start, 40);
        assert_eq!(execution.layer_distribution.stage_layers[1].end, 80);
    }

    #[test]
    fn converts_to_llama_stage_configs() {
        let plan =
            parse_layer_split_plan("stage0:cuda:0:layers=0-39;stage1:cuda:1:layers=40-79").unwrap();
        let stages = plan.to_llama_stage_configs();

        assert_eq!(stages.len(), 2);
        assert_eq!(stages[0].source_layers, 0..40);
        assert!(stages[0].load_embedding);
        assert!(!stages[0].load_lm_head);
        assert_eq!(stages[1].source_layers, 40..80);
        assert!(!stages[1].load_embedding);
        assert!(stages[1].load_lm_head);
    }

    #[test]
    fn selects_llama_stage_config_for_cuda_device() {
        let plan =
            parse_layer_split_plan("stage0:cuda:0:layers=0-39;stage1:cuda:1:layers=40-79").unwrap();
        let stage = plan.llama_stage_config_for_device(1).unwrap();

        assert_eq!(stage.source_layers, 40..80);
        assert!(!stage.load_embedding);
        assert!(stage.load_lm_head);
    }
}
