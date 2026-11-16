//! Neutral table serialization and output validation, independent of the device.
use anyhow::{ensure, Result};
use ferrum_execution::TensorExecutionOutput;
use ferrum_models::fast_h3::*;
use ferrum_runtime::*;
use half::bf16;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap};

fn id<T: TryFrom<String, Error = VNextError>>(s: impl Into<String>) -> T {
    T::try_from(s.into()).expect("fixed artifact ID")
}

pub(super) fn inspect(
    outputs: &[TensorExecutionOutput],
    config: &FastH3TimeConditioningConfig,
) -> Result<Vec<Value>> {
    ensure!(
        outputs.len() == 2,
        "expected embedding and modulation outputs"
    );
    let mut reports = Vec::new();
    for (output, expected_id, dtype, width) in [
        (
            &outputs[0],
            FastH3TimeConditioningFamily::EMBEDDING_ID,
            ElementType::F32,
            config.embedding_dim(),
        ),
        (
            &outputs[1],
            FastH3TimeConditioningFamily::OUTPUT_ID,
            ElementType::Bf16,
            config.hidden_size() * 18,
        ),
    ] {
        ensure!(
            output.value_id() == &id::<ProgramValueId>(expected_id),
            "output ID differs"
        );
        ensure!(
            output.tensor().element_type() == dtype
                && output.tensor().dimensions().len() == 2
                && output.tensor().dimensions()[0] == config.time_rows()
                && output.tensor().dimensions()[1] == width,
            "output shape or precision differs"
        );
        let bytes = output.bytes().bytes();
        let mut maximum = 0.0f32;
        let stride = if dtype == ElementType::F32 { 4 } else { 2 };
        for b in bytes.chunks_exact(stride) {
            let v = if stride == 4 {
                f32::from_le_bytes(b.try_into()?)
            } else {
                bf16::from_bits(u16::from_le_bytes(b.try_into()?)).to_f32()
            };
            ensure!(v.is_finite(), "nonfinite conditioning output");
            maximum = maximum.max(v.abs());
        }
        reports.push(json!({"id":expected_id,"shape":output.tensor().dimensions(),"dtype":dtype,
            "bytes":bytes.len(),"sha256":format!("{:x}",Sha256::digest(bytes)),"maximum_abs":maximum,"finite":true}));
    }
    Ok(reports)
}
pub(super) fn routing(schedule: &FastH3ConditioningSchedule) -> Vec<Value> {
    use FastH3SegmentKind::*;
    (0..schedule.evaluation_count())
        .map(|step| {
            let map: BTreeMap<_, _> = [
                ("text", Text),
                ("reference_image", ReferenceImage),
                ("reference_video", ReferenceVideo),
                ("reference_audio", ReferenceAudio),
                ("target_video", TargetVideo),
                ("target_audio", TargetAudio),
            ]
            .into_iter()
            .map(|(name, kind)| (name, schedule.segment_time_index(step, kind)))
            .collect();
            json!({"evaluation":step,"row_indices":map})
        })
        .collect()
}
pub(super) fn save(
    path: &std::path::Path,
    outputs: &[TensorExecutionOutput],
    config: &FastH3TimeConditioningConfig,
    schedule: &FastH3ConditioningSchedule,
    source: &Value,
    plan: &Value,
) -> Result<()> {
    use safetensors::{
        tensor::{serialize_to_file, TensorView},
        Dtype,
    };
    let tensors = outputs
        .iter()
        .map(|o| {
            let dtype = match o.tensor().element_type() {
                ElementType::F32 => Dtype::F32,
                ElementType::Bf16 => Dtype::BF16,
                _ => anyhow::bail!("unexpected output dtype"),
            };
            let shape = o
                .tensor()
                .dimensions()
                .iter()
                .map(|&d| usize::try_from(d))
                .collect::<std::result::Result<Vec<_>, _>>()?;
            Ok((
                o.value_id().as_str(),
                TensorView::new(dtype, shape, o.bytes().bytes())?,
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    let metadata = HashMap::from([
        ("schema".into(), "ferrum.fast_h3.time_table.v1".into()),
        ("configuration".into(), serde_json::to_string(config)?),
        ("source".into(), serde_json::to_string(source)?),
        ("plan".into(), serde_json::to_string(plan)?),
        (
            "timesteps".into(),
            serde_json::to_string(schedule.timesteps())?,
        ),
        ("routing".into(), serde_json::to_string(&routing(schedule))?),
        (
            "feature_profile".into(),
            "cpu.libm_0_2_15.f32.cos_sin.max_period_10000.v1".into(),
        ),
    ]);
    serialize_to_file(tensors, &Some(metadata), path)?;
    Ok(())
}
