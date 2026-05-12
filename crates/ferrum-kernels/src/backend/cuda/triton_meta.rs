//! Shared Triton kernel metadata parser.
//!
//! Each PTX file produced by triton-rs ships alongside a small JSON blob
//! describing the kernel's launch requirements (entry name, num_warps,
//! shared_mem, scratch sizes, etc.). We only need a handful of fields,
//! so we keep the parser tiny and stdlib-only — no serde dep.
//!
//! The original parser was authored inline in `triton_rms_norm.rs`; this
//! module hoists it out so the eight other triton wrappers can reuse it.

#![cfg(all(feature = "cuda", feature = "triton-kernels"))]
#![allow(dead_code)]

pub struct Meta {
    pub name: String,
    pub num_warps: u32,
    pub shared_mem: u64,
    pub global_scratch_size: usize,
    pub profile_scratch_size: usize,
}

pub fn parse_meta(json: &str) -> candle_core::Result<Meta> {
    fn pick<'a>(s: &'a str, key: &str) -> Option<&'a str> {
        let needle = format!("\"{key}\":");
        let start = s.find(&needle)? + needle.len();
        let rest = s[start..].trim_start();
        if rest.starts_with('"') {
            let inner = &rest[1..];
            let end = inner.find('"')?;
            Some(&inner[..end])
        } else {
            let end = rest
                .find(|c: char| c == ',' || c == '}' || c.is_whitespace())
                .unwrap_or(rest.len());
            Some(rest[..end].trim())
        }
    }
    let parse_u = |k: &str| -> u64 {
        pick(json, k)
            .and_then(|v| v.trim_matches('"').parse::<u64>().ok())
            .unwrap_or(0)
    };
    let name = pick(json, "name")
        .ok_or_else(|| candle_core::Error::Msg(format!("triton meta: missing name in {json}")))?
        .to_string();
    Ok(Meta {
        name,
        num_warps: parse_u("num_warps") as u32,
        shared_mem: parse_u("shared_mem"),
        global_scratch_size: parse_u("global_scratch_size") as usize,
        profile_scratch_size: parse_u("profile_scratch_size") as usize,
    })
}
