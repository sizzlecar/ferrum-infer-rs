//! Release preparation consumes the qualified source closure and checks its
//! Ferrum-owned files against this candidate before invoking any compiler.
use super::*;
use ferrum_native_ops_builder::NativeOperatorSourceDefinition;

pub(super) const OPERATORS: &[(CudaNativeBuildUnit, &str)] = &[
    (CudaNativeBuildUnit::CutlassNvfp4Matmul, "cutlass-nvfp4"),
    (CudaNativeBuildUnit::SageVsaFp4CpuLists, "sage-vsa-fp4"),
];

pub(super) const PROVIDERS: &[(&str, &str)] = &[
    (
        "operation.nvfp4_matmul.bf16",
        "provider.cuda.nvfp4_matmul.bf16.cutlass.sm120.k128",
    ),
    (
        "operation.nvfp4_quantize.bf16",
        "provider.cuda.nvfp4_quantize.bf16.rcp-approx-ftz.sm120",
    ),
    (
        "operation.sparse_fp4_attention.checked_host.bf16",
        "provider.cuda.sparse_fp4_attention.checked_host.bf16.sage.sm120",
    ),
    (
        "operation.sparse_fp4_attention.region_topk.bf16",
        "provider.cuda.sparse_fp4_attention.region_topk.bf16.sage.sm120",
    ),
    (
        "operation.sparse_fp4_qkv_quantize.bf16",
        "provider.cuda.sparse_fp4_qkv_quantize.bf16.sage.sm120",
    ),
];

pub(super) fn prepare(args: &Build) -> Result<Vec<(&'static str, PathBuf, PathBuf)>> {
    let root = match (args.architecture, &args.fast_h3_source) {
        (Architecture::Sm89, None) => return Ok(Vec::new()),
        (Architecture::Sm89, Some(_)) => return Err("FastH3 native sources require SM120".into()),
        (Architecture::Sm120, None) => return Err("SM120 release requires --fast-h3-source".into()),
        (Architecture::Sm120, Some(root)) => root,
    };
    let mut prepared = Vec::new();
    for &(unit, name) in OPERATORS {
        let source = root.join(name);
        let plan_path = source.join("source.plan.json");
        let plan: NativeOperatorSourceBuildPlan = read_json(&plan_path)?;
        if plan.operator != unit.artifact_operator() {
            return Err(format!("{name} source plan has wrong operator").into());
        }
        let owned = args
            .workspace
            .join("crates/ferrum-kernels/native")
            .join(name);
        let definition: NativeOperatorSourceDefinition =
            read_json(&owned.join("source.seed.json"))?;
        for path in definition
            .translation_units
            .iter()
            .chain(&definition.headers)
        {
            let relative = path
                .strip_prefix("kernels/")
                .ok_or("invalid owned source path")?;
            // Sage's adapted params header occupies the upstream include location.
            let relative = if name == "sage-vsa-fp4" && relative == "sage/params.h" {
                "params.h"
            } else {
                relative
            };
            let actual = digest_file(&owned.join(checked_relative(relative)?))?;
            let expected = plan
                .translation_units
                .iter()
                .chain(&plan.headers)
                .find(|member| member.path == *path)
                .ok_or_else(|| format!("qualified {name} closure omits {path}"))?;
            if actual != expected.sha256 {
                return Err(format!(
                    "qualified {name} closure differs from candidate source {path}"
                )
                .into());
            }
        }
        prepared.push((name, source, plan_path));
    }
    Ok(prepared)
}
