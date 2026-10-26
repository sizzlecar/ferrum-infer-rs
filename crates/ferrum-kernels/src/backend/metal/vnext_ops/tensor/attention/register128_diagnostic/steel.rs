//! Ignored-test-only official source bridge. No MLX runtime or allocator is used.
use super::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::Read,
    path::{Component, Path},
};

pub(super) const LOCATOR: &str = "FERRUM_METAL_SDPA_REFERENCE_MANIFEST";
pub(super) const KERNEL: &str = "steel_attention_bfloat16_bq32_bk16_bd128_wm4_wn1_maskbfloat16";
const ARGS: [&str; 8] = [
    "bfloat16_t",
    "32",
    "16",
    "128",
    "4",
    "1",
    "bfloat16_t",
    "float",
];
pub(super) const COMPILER: &str = "MSL3.1(explicit native bfloat);fast_math=false;FC bool200=Q%32==0,201=K%16==0,300/301/302=false;AttnParams reprC152/align8/strides56;BSHD;threads32x4x1;dynamicTG0";

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(super) struct Params {
    pub b: i32,
    pub h: i32,
    pub d: i32,
    pub ql: i32,
    pub kl: i32,
    pub gqa_factor: i32,
    pub scale: f32,
    pub nq: i32,
    pub nk: i32,
    pub nq_aligned: i32,
    pub nk_aligned: i32,
    pub ql_rem: i32,
    pub kl_rem: i32,
    pub ql_off: i32,
    pub q_strides: [i64; 3],
    pub k_strides: [i64; 3],
    pub v_strides: [i64; 3],
    pub o_strides: [i64; 3],
}
impl Params {
    pub(super) fn new(c: family::Config) -> Result<Self, VNextError> {
        let i32n = |n| i32::try_from(n).map_err(|_| invalid("Steel parameter exceeds i32"));
        let row = c
            .heads
            .checked_mul(128)
            .ok_or_else(|| invalid("Steel row extent overflow"))?;
        let batch = c
            .sequence
            .checked_mul(row)
            .ok_or_else(|| invalid("Steel batch extent overflow"))?;
        let stride = [
            i64::try_from(batch).map_err(|_| invalid("Steel stride overflow"))?,
            128,
            i64::try_from(row).map_err(|_| invalid("Steel stride overflow"))?,
        ];
        Ok(Self {
            b: i32n(c.batch)?,
            h: i32n(c.heads)?,
            d: 128,
            ql: i32n(c.sequence)?,
            kl: i32n(c.sequence)?,
            gqa_factor: 1,
            scale: 0.125,
            nq: i32n(c.sequence.div_ceil(32))?,
            nk: i32n(c.sequence.div_ceil(16))?,
            nq_aligned: i32n(c.sequence / 32)?,
            nk_aligned: i32n(c.sequence / 16)?,
            ql_rem: i32n(c.sequence % 32)?,
            kl_rem: i32n(c.sequence % 16)?,
            ql_off: 0,
            q_strides: stride,
            k_strides: stride,
            v_strides: stride,
            o_strides: stride,
        })
    }
}

#[derive(Debug, Deserialize, Serialize)]
struct SourceFile {
    path: String,
    kind: String,
    bytes: u64,
    sha256: String,
    quoted_includes: Vec<String>,
    #[serde(flatten)]
    annotations: BTreeMap<String, Value>,
}
#[derive(Debug, Deserialize, Serialize)]
struct Instantiation {
    kernel_name: String,
    template: String,
    arguments: Vec<String>,
    source: String,
}
#[derive(Debug, Deserialize, Serialize)]
struct ParameterAbi {
    size_bytes: usize,
    alignment_bytes: usize,
    strides_offset_bytes: usize,
    fields: Vec<[String; 2]>,
    #[serde(flatten)]
    annotations: BTreeMap<String, Value>,
}
#[derive(Debug, Deserialize, Serialize)]
struct Constant {
    index: u64,
    name: String,
    #[serde(rename = "type")]
    data_type: String,
    value: Value,
}
#[derive(Debug, Deserialize, Serialize)]
struct Abi {
    params: ParameterAbi,
    function_constants: Vec<Constant>,
    threads_per_threadgroup: [u64; 3],
    dynamic_threadgroup_bytes: u64,
    #[serde(flatten)]
    annotations: BTreeMap<String, Value>,
}
#[derive(Debug, Deserialize, Serialize)]
struct Expectations {
    fast_math_enabled: bool,
    no_injected_ferrum_headers: bool,
    #[serde(flatten)]
    annotations: BTreeMap<String, Value>,
}
#[derive(Debug, Deserialize, Serialize)]
struct Manifest {
    schema_version: u32,
    files: Vec<SourceFile>,
    entrypoints: Vec<String>,
    instantiation: Instantiation,
    abi: Abi,
    compiler_expectations: Expectations,
    #[serde(flatten)]
    annotations: BTreeMap<String, Value>,
}

pub(super) struct Source {
    pub source: String,
    pub fingerprint: String,
    pub report: Value,
}
fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn safe_path(value: &str) -> bool {
    !value.is_empty()
        && value
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"_./-".contains(&b))
        && Path::new(value)
            .components()
            .all(|c| matches!(c, Component::Normal(_)))
        && value
            .split('/')
            .all(|s| !s.is_empty() && s != "." && s != "..")
}
fn limited_read(path: &Path, maximum: u64) -> Result<Vec<u8>, String> {
    let mut data = Vec::new();
    fs::File::open(path)
        .map_err(|e| e.to_string())?
        .take(maximum + 1)
        .read_to_end(&mut data)
        .map_err(|e| e.to_string())?;
    if data.len() as u64 > maximum {
        return Err(format!("source exceeds bounded read: {}", path.display()));
    }
    Ok(data)
}
fn verify_payload(file: &SourceFile, bytes: &[u8]) -> Result<(), String> {
    if bytes.len() as u64 != file.bytes || hash(bytes) != file.sha256 {
        return Err(format!("source size/hash mismatch: {}", file.path));
    }
    Ok(())
}

fn quoted_include(line: &str) -> Result<Option<&str>, String> {
    let trimmed = line.trim();
    let Some(after_hash) = trimmed.strip_prefix('#') else {
        return Ok(None);
    };
    let directive = after_hash.trim_start();
    let Some(rest) = directive.strip_prefix("include") else {
        return Ok(None);
    };
    if rest.starts_with(|c: char| c.is_ascii_alphanumeric() || c == '_') {
        return Ok(None);
    }
    let rest = rest.trim();
    if rest.starts_with('<') && rest.contains('>') {
        return Ok(None);
    }
    if let Some(rest) = rest.strip_prefix('"') {
        if let Some((target, trailing)) = rest.split_once('"') {
            if trailing.trim().is_empty() || trailing.trim_start().starts_with("//") {
                return Ok(Some(target));
            }
        }
    }
    Err(format!("unsupported nonliteral include: {line}"))
}
fn expand(
    name: &str,
    headers: &BTreeMap<String, String>,
    stack: &mut Vec<String>,
    output: &mut String,
) -> Result<(), String> {
    if stack.iter().any(|s| s == name) {
        return Err(format!("include graph cycle at {name}"));
    }
    let source = headers
        .get(name)
        .ok_or_else(|| format!("undeclared header {name}"))?;
    let guard = format!(
        "FERRUM_SDPA_HEADER_{}",
        hash(name.as_bytes()).to_uppercase()
    );
    output.push_str(&format!(
        "\n#ifndef {guard}\n#define {guard}\n#line 1 \"{name}\"\n"
    ));
    stack.push(name.to_owned());
    for (i, line) in source.lines().enumerate() {
        if line.trim() == "#pragma once" {
            output.push('\n');
        } else if let Some(target) = quoted_include(line)? {
            // Every occurrence remains inside its original conditional block.
            // Header guards, not global traversal deduplication, implement once.
            expand(target, headers, stack, output)?;
            output.push_str(&format!("\n#line {} \"{name}\"\n", i + 2));
        } else {
            output.push_str(line);
            output.push('\n');
        }
        if output.len() > 8 << 20 {
            return Err("expanded source exceeds 8MiB diagnostic limit".into());
        }
    }
    stack.pop();
    output.push_str(&format!("\n#endif // {guard}\n"));
    Ok(())
}
impl Source {
    /// Called only by the explicitly ignored reference diagnostic. A missing
    /// artifact is an error; it never skips the test or affects production.
    pub(super) fn from_locator() -> Result<Self, String> {
        let path = std::env::var_os(LOCATOR)
            .ok_or_else(|| format!("missing required ignored-test artifact locator {LOCATOR}"))?;
        Self::load(Path::new(&path))
    }
    fn load(path: &Path) -> Result<Self, String> {
        let path = path.canonicalize().map_err(|e| e.to_string())?;
        let raw = limited_read(&path, 1 << 20)?;
        let manifest: Manifest = serde_json::from_slice(&raw).map_err(|e| e.to_string())?;
        if manifest.schema_version != 1
            || manifest.instantiation.kernel_name != KERNEL
            || manifest.instantiation.template != "attention"
            || manifest
                .instantiation
                .arguments
                .iter()
                .map(String::as_str)
                .ne(ARGS)
            || manifest.compiler_expectations.fast_math_enabled
            || !manifest.compiler_expectations.no_injected_ferrum_headers
        {
            return Err("unsupported reference manifest contract".into());
        }
        let expected_source = format!(
            "instantiate_kernel(\"{KERNEL}\", attention, {});",
            ARGS.join(", ")
        );
        if manifest.instantiation.source != expected_source {
            return Err("instantiation declaration differs from typed reference".into());
        }
        let p = &manifest.abi.params;
        let fields: Vec<[String; 2]> = [
            "B",
            "H",
            "D",
            "qL",
            "kL",
            "gqa_factor",
            "scale",
            "NQ",
            "NK",
            "NQ_aligned",
            "NK_aligned",
            "qL_rem",
            "kL_rem",
            "qL_off",
            "Q_strides",
            "K_strides",
            "V_strides",
            "O_strides",
        ]
        .iter()
        .enumerate()
        .map(|(i, name)| {
            [
                name.to_string(),
                if i == 6 {
                    "f32"
                } else if i >= 14 {
                    "i64[3]"
                } else {
                    "i32"
                }
                .to_owned(),
            ]
        })
        .collect();
        if p.size_bytes != std::mem::size_of::<Params>()
            || p.alignment_bytes != std::mem::align_of::<Params>()
            || p.strides_offset_bytes != std::mem::offset_of!(Params, q_strides)
            || p.fields != fields
            || manifest.abi.threads_per_threadgroup != [32, 4, 1]
            || manifest.abi.dynamic_threadgroup_bytes != 0
        {
            return Err("unsupported reference ABI".into());
        }
        let expected_constants = [
            (200, "align_Q", serde_json::json!("Q%32==0")),
            (201, "align_K", serde_json::json!("K%16==0")),
            (300, "has_mask", serde_json::json!(false)),
            (301, "do_causal", serde_json::json!(false)),
            (302, "has_sinks", serde_json::json!(false)),
        ];
        if manifest.abi.function_constants.len() != expected_constants.len()
            || !manifest
                .abi
                .function_constants
                .iter()
                .zip(expected_constants)
                .all(|(c, (index, name, value))| {
                    c.index == index && c.name == name && c.data_type == "bool" && c.value == value
                })
        {
            return Err("unsupported reference function constants".into());
        }
        let parent = path
            .parent()
            .ok_or("manifest has no parent")?
            .canonicalize()
            .map_err(|e| e.to_string())?;
        let mut headers = BTreeMap::new();
        let mut identities = Vec::new();
        let mut total = 0_u64;
        let mut paths = BTreeSet::new();
        for file in &manifest.files {
            if !safe_path(&file.path) || !paths.insert(file.path.clone()) {
                return Err(format!("unsafe/duplicate manifest path {}", file.path));
            }
            let resolved = parent
                .join(&file.path)
                .canonicalize()
                .map_err(|e| e.to_string())?;
            if !resolved.starts_with(&parent) {
                return Err("source escapes artifact root".into());
            }
            let bytes = limited_read(&resolved, 1 << 20)?;
            total = total
                .checked_add(bytes.len() as u64)
                .ok_or("source total overflow")?;
            if total > 2 << 20 {
                return Err("reference asset total exceeds 2MiB diagnostic limit".into());
            }
            verify_payload(file, &bytes)?;
            identities.push(
                serde_json::json!({"path":file.path,"sha256":file.sha256,"bytes":file.bytes}),
            );
            if file.kind == "metal_header" {
                let source = String::from_utf8(bytes).map_err(|e| e.to_string())?;
                let includes = source
                    .lines()
                    .map(quoted_include)
                    .collect::<Result<Vec<_>, _>>()?
                    .into_iter()
                    .flatten()
                    .map(str::to_owned)
                    .collect::<Vec<_>>();
                // Metadata orders includes canonically, whereas expansion retains source order.
                let mut actual = includes;
                actual.sort();
                let mut declared = file.quoted_includes.clone();
                declared.sort();
                if actual != declared {
                    return Err(format!("quoted include closure mismatch: {}", file.path));
                }
                headers.insert(file.path.clone(), source);
            } else if file.kind != "license" {
                return Err("unsupported asset kind".into());
            }
        }
        let mut source = String::new();
        for entry in &manifest.entrypoints {
            if !safe_path(entry) {
                return Err("unsafe entrypoint".into());
            }
            expand(entry, &headers, &mut Vec::new(), &mut source)?;
        }
        source.push_str("\n#line 1 \"ferrum_sdpa_reference_instantiation\"\n");
        source.push_str(&expected_source);
        source.push('\n');
        let source_hash = hash(source.as_bytes());
        let manifest_hash = hash(&raw);
        let fingerprint = implementation_fingerprint(&[
            &raw,
            source.as_bytes(),
            COMPILER.as_bytes(),
            include_str!("steel.rs").as_bytes(),
        ]);
        let report = serde_json::json!({"manifest_sha256":manifest_hash,"flattened_source_sha256":source_hash,"flattened_source_bytes":source.len(),"source_files":identities,"manifest":serde_json::from_slice::<Value>(&raw).map_err(|e|e.to_string())?,"actual_compile_contract":COMPILER,"actual_header_guard_prefix":"FERRUM_SDPA_HEADER_","artifact_fingerprint":fingerprint});
        Ok(Self {
            source,
            fingerprint,
            report,
        })
    }
}

#[test]
#[ignore = "CPU-only official source artifact check; requires explicit reference manifest locator"]
fn official_steel_artifact_manifest_and_include_closure() {
    // This entry point does not create a device, Metal library, or runtime.
    // Missing or invalid artifacts fail instead of skipping the diagnostic.
    let source = Source::from_locator().expect("load the declared official Steel source artifact");
    eprintln!(
        "metal_attention_steel_artifact={}",
        serde_json::json!({
            "manifest_sha256": source.report["manifest_sha256"],
            "flattened_source_sha256": source.report["flattened_source_sha256"],
            "flattened_source_bytes": source.source.len(),
            "artifact_fingerprint": source.fingerprint,
            "source_files": source.report["source_files"],
            "scope": "CPU-only typed manifest, payload hashes, and quoted-include expansion; no Metal compilation or execution"
        })
    );
}

#[test]
fn header_expansion_preserves_conditional_repeated_includes_and_rejects_cycles() {
    let headers=BTreeMap::from([("root.h".into(),"#if 0\n#include \"shared.h\"\n#endif\n#include \"shared.h\"\n#include <metal_stdlib>\n".into()),("shared.h".into(),"#pragma once\nfloat shared();\n".into())]);
    let mut source = String::new();
    expand("root.h", &headers, &mut Vec::new(), &mut source).unwrap();
    assert_eq!(source.matches("float shared();").count(), 2);
    assert!(source.contains("#if 0\n\n#ifndef FERRUM_SDPA_HEADER_"));
    assert!(source.contains("#include <metal_stdlib>"));
    assert!(!source.contains("#pragma once"));
    assert!(!safe_path("../escape.h") && !safe_path("/absolute.h") && !safe_path("a\\b.h"));
    assert!(quoted_include("#include MACRO_HEADER").is_err());
    let cycle = BTreeMap::from([("a.h".into(), "#include \"a.h\"".into())]);
    assert!(expand("a.h", &cycle, &mut Vec::new(), &mut String::new()).is_err());
}

#[test]
fn declared_payload_integrity_rejects_changed_bytes_and_length() {
    let payload = b"#pragma once\nfloat declaration();\n";
    let file = SourceFile {
        path: "fixture.h".into(),
        kind: "metal_header".into(),
        bytes: payload.len() as u64,
        sha256: hash(payload),
        quoted_includes: vec![],
        annotations: BTreeMap::new(),
    };
    verify_payload(&file, payload).unwrap();
    let mut changed = payload.to_vec();
    changed[0] = b'!';
    assert!(verify_payload(&file, &changed)
        .unwrap_err()
        .contains("size/hash mismatch"));
    assert!(verify_payload(&file, &payload[..payload.len() - 1]).is_err());
}
