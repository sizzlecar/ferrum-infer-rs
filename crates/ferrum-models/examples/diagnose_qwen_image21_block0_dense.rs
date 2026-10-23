//! CPU-only, exact-dyadic diagnosis of selected real block-0 dense rows.
//! Saved upstream tensors are explicit node-local inputs. They do not turn
//! this into an independent full-block oracle or alter its failed criterion.
#[path = "verify_qwen_image21_projection/artifacts.rs"]
mod artifacts;
#[path = "diagnose_qwen_image21_block0_dense/dyadic.rs"]
mod dyadic;
#[path = "diagnose_qwen_image21_block0_dense/report.rs"]
mod report;
#[path = "diagnose_qwen_image21_block0_dense/source.rs"]
mod source;

use anyhow::{ensure, Context, Result};
use ferrum_quantization::SafetensorsArchive;
use serde_json::json;
use std::{collections::BTreeMap, ffi::OsString, path::PathBuf};
const USAGE:&str="diagnose_qwen_image21_block0_dense --reference-artifact-dir REF --candidate-artifact-dir CUDA --model-dir TRANSFORMER_DIRECTORY --artifact-dir NEW_EXTERNAL_DIRECTORY [--index-file diffusion_pytorch_model.safetensors.index.json] [--query-channels 953] [--value-channels 126,668,1602] [--mlp-output-channels 2615] [--mlp-up-channels 342,4878,10182]\nCPU-only. Reads existing checkpoint tensors; exports only selected OI rows and exact local diagnostics. Default five rows target the previously observed R36 first divergence and largest failing MLP output; they are diagnostic selections, not acceptance gates. Optional MLP-up channels add a replay using each backend's own saved MODULATED2 input, for every saved row; no GPU execution or change to the original criterion.";

fn main() -> Result<()> {
    let mut args = std::env::args_os().skip(1).peekable();
    if args.peek().is_some_and(|x| x == "--help" || x == "-h") {
        args.next();
        ensure!(args.next().is_none(), "--help takes no other arguments");
        println!("{USAGE}");
        return Ok(());
    }
    let mut v = BTreeMap::new();
    while let Some(flag) = args.next() {
        let flag = flag
            .into_string()
            .map_err(|_| anyhow::anyhow!("option names require UTF-8"))?;
        let value = args.next().context("every option requires one value")?;
        ensure!(
            v.insert(flag.clone(), value).is_none(),
            "duplicate option {flag}"
        );
    }
    let reference = path(&mut v, "--reference-artifact-dir")?;
    let candidate = path(&mut v, "--candidate-artifact-dir")?;
    let model = path(&mut v, "--model-dir")?;
    let output = path(&mut v, "--artifact-dir")?;
    let index = v
        .remove("--index-file")
        .unwrap_or_else(|| "diffusion_pytorch_model.safetensors.index.json".into())
        .into_string()
        .map_err(|_| anyhow::anyhow!("index requires UTF-8"))?;
    let selected = selected_stages(&mut v)?;
    ensure!(v.is_empty(), "unknown options: {:?}", v.keys());
    let reference = source::Artifact::open(&reference, "reference")?;
    let candidate = source::Artifact::open(&candidate, "cuda")?;
    ensure!(
        reference.manifest.identity == candidate.manifest.identity,
        "canonical config, input and weight identities must match exactly"
    );
    let archive = SafetensorsArchive::open_with_index(model, &index)?;
    let output = artifacts::Artifacts::create(&output)?;
    let selections = selected
        .iter()
        .map(|(stage, channels)| (stage.name().to_owned(), json!(channels)))
        .collect::<serde_json::Map<_, _>>();
    output.json("declaration.json",&json!({"schema":"ferrum.qwen_image21.block0-dense-diagnosis.v1",
        "reference_manifest_sha256":reference.manifest_sha256,"candidate_manifest_sha256":candidate.manifest_sha256,
        "identity":reference.manifest.identity,"selections":selections,
        "exact_domain":"finite BF16 operands; exact products on binary lattice>=2^-126; checked i128 common lattice and absolute sum; gamma(2K,2^-24) absolute-sum expansion<=BF16MAX; insufficient capacity is unresolved/error",
        "bound":"outward exact-dot F64 enclosure +/- [gamma(2K)*S_upper + 2^-8*(maxabs(exact enclosure)+gamma(2K)*S_upper)+2^-134]",
        "replay":"sequential FP32 product/add starting +0, one final integer BF16 RNE; separate from the independent exact dot",
        "original_block_status":"unchanged; this diagnostic cannot turn the previous 679-failure run into a pass",
        "qualification":"selected-node arithmetic compatibility only; not whole-domain cuBLAS, full model, Torch parity, or performance"}))?;
    let result: Result<_> = (|| {
        let mut records = Vec::new();
        for (stage, channels) in &selected {
            let r = report::analyze(*stage, channels, &reference, &candidate, &archive, &output)?;
            output.json(&format!("{}.json", stage.name()), &r)?;
            records.push(r);
        }
        let incompatible = records
            .iter()
            .map(|r| r["dot_bound_incompatible_count"].as_u64().unwrap_or(0))
            .sum::<u64>();
        let mismatches = records
            .iter()
            .map(|r| r["reference_replay_mismatch_count"].as_u64().unwrap_or(0))
            .sum::<u64>();
        output.json("result.json",&json!({"diagnostic_completed":true,"selected_dot_bound_incompatible_count":incompatible,
            "reference_replay_mismatch_count":mismatches,"records":records,"original_block_verdict":"unchanged; numerical failure remains",
            "interpretation":"compatible bounds are necessary local evidence, not proof of a particular cuBLAS reduction order or global correctness"}))?;
        ensure!(
            incompatible == 0 && mismatches == 0,
            "selected local arithmetic incompatibility or Reference replay mismatch; see report"
        );
        Ok(())
    })();
    if let Err(error) = &result {
        output.json("failure.json",&json!({"completed_successfully":false,"error":format!("{error:#}"),"unresolved_is_not_pass":true}))?;
    }
    result
}
fn selected_stages(
    values: &mut BTreeMap<String, OsString>,
) -> Result<Vec<(report::Stage, Vec<usize>)>> {
    let mut selected = vec![
        (
            report::Stage::Query,
            channels(values.remove("--query-channels"), "953")?,
        ),
        (
            report::Stage::Value,
            channels(values.remove("--value-channels"), "126,668,1602")?,
        ),
        (
            report::Stage::MlpOutput,
            channels(values.remove("--mlp-output-channels"), "2615")?,
        ),
    ];
    if let Some(value) = values.remove("--mlp-up-channels") {
        selected.push((report::Stage::MlpUp, channels(Some(value), "")?));
    }
    Ok(selected)
}
fn path(v: &mut BTreeMap<String, OsString>, key: &str) -> Result<PathBuf> {
    Ok(v.remove(key)
        .with_context(|| format!("missing {key}; {USAGE}"))?
        .into())
}
fn channels(value: Option<OsString>, default: &str) -> Result<Vec<usize>> {
    let value = value
        .unwrap_or_else(|| default.into())
        .into_string()
        .map_err(|_| anyhow::anyhow!("channel list requires UTF-8"))?;
    let items = value
        .split(',')
        .map(|v| {
            v.parse()
                .map_err(|_| anyhow::anyhow!("invalid channel list"))
        })
        .collect::<Result<Vec<usize>>>()?;
    ensure!(
        !items.is_empty()
            && items.len() <= 64
            && items
                .iter()
                .collect::<std::collections::BTreeSet<_>>()
                .len()
                == items.len(),
        "channel lists need 1..64 distinct nonnegative indices"
    );
    Ok(items)
}
#[cfg(test)]
#[path = "diagnose_qwen_image21_block0_dense/tests.rs"]
mod tests;
