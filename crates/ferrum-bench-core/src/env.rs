//! Bench environment snapshot — hardware + software + config — and
//! the SHA-256 `env_hash` used by `compare-commits.sh` and similar to
//! filter "apples-to-apples" cells.
//!
//! See `docs/bench/PLAYBOOK.md` § 7 schema + § 0.6 vs-vLLM parity:
//! every cell carries one `Env` block and its `env_hash`. Two cells
//! with the same hash are guaranteed comparable.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Snapshot of everything we expect to affect bench outcomes.
///
/// `BTreeMap` (not `HashMap`) for `ferrum_env` so JSON serialization
/// is deterministic — required for `env_hash` reproducibility.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Env {
    /// ferrum git commit short SHA.
    pub commit_sha: String,
    /// Stable hardware identifier (e.g. `rtx-4090`, `m1-max-32gb`).
    pub hw_id: String,
    /// NVIDIA driver version (e.g. `555.42.06`). Omitted on non-CUDA hosts.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub driver: Option<String>,
    /// CUDA toolkit version (e.g. `12.4`). Omitted on non-CUDA hosts.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cuda: Option<String>,
    /// Rust toolchain version (e.g. `1.78.0`).
    pub rust: String,
    /// Cargo features enabled on the ferrum build (sorted).
    pub ferrum_features: Vec<String>,

    /// GPU clock lock value in MHz (`nvidia-smi -lgc <mhz>,<mhz>`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu_clock_lock_mhz: Option<u32>,
    /// Power limit in watts (`nvidia-smi -pl <W>`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu_power_limit_w: Option<u32>,
    /// Persistence mode state (`nvidia-smi -pm 1`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu_persistence_mode: Option<bool>,
    /// Auto-boost state (false ⇒ disabled).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu_auto_boost: Option<bool>,

    /// Selected `FERRUM_*` env vars affecting runtime (sorted by BTreeMap).
    pub ferrum_env: BTreeMap<String, String>,

    /// `vllm serve` effective args, populated only for vLLM cells. None
    /// for ferrum cells — used by the config-parity report block.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vllm_args: Option<Vec<String>>,
}

/// SHA-256 of canonical-JSON-serialized `Env`, prefixed with `sha256:`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EnvHash(pub String);

impl EnvHash {
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for EnvHash {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl Env {
    /// Compute the env_hash. Deterministic given:
    /// - `BTreeMap` (sorted) for `ferrum_env`
    /// - explicit struct field order (serde serializes in declaration order)
    /// - `Vec<String>` fields are caller-sorted (we don't sort here so the
    ///   raw order is preserved for display — callers should sort `ferrum_features`
    ///   before constructing `Env` if they want order-independent hashes).
    pub fn hash(&self) -> EnvHash {
        use sha2::{Digest, Sha256};
        let canonical = serde_json::to_vec(self).expect("Env serialization must not fail");
        let mut hasher = Sha256::new();
        hasher.update(&canonical);
        let digest = hasher.finalize();
        EnvHash(format!("sha256:{:x}", digest))
    }

    /// Capture commit + rust + features + auto-detected hw/driver/cuda.
    /// On CUDA hosts populates GPU driver + CUDA toolkit versions via
    /// `nvidia-smi` and `nvcc --version`; on macOS uses `sysctl`.
    pub fn capture_minimal(commit_sha: String, ferrum_features: Vec<String>) -> Self {
        let mut feat = ferrum_features;
        feat.sort();
        feat.dedup();
        Self {
            commit_sha,
            hw_id: detect_hw_id(),
            driver: detect_nvidia_driver(),
            cuda: detect_cuda_version(),
            rust: detect_rust_version(),
            ferrum_features: feat,
            gpu_clock_lock_mhz: detect_gpu_clock_lock_mhz(),
            gpu_power_limit_w: detect_gpu_power_limit_w(),
            gpu_persistence_mode: detect_gpu_persistence(),
            gpu_auto_boost: None,
            ferrum_env: capture_ferrum_env(),
            vllm_args: None,
        }
    }
}

/// Heuristic hardware ID. On CUDA hosts uses the GPU name (e.g.
/// "rtx-4090"); on macOS uses the CPU brand (e.g. "apple-m1-max").
/// Returns generic "unknown" only when both fail.
pub fn detect_hw_id() -> String {
    // Try nvidia-smi first — most reliable on CUDA hosts.
    if let Some(name) = nvidia_smi_query("name") {
        // "NVIDIA GeForce RTX 4090" → "rtx-4090"
        let normalized = name
            .to_lowercase()
            .replace("nvidia ", "")
            .replace("geforce ", "")
            .trim()
            .replace(' ', "-");
        if !normalized.is_empty() {
            return normalized;
        }
    }
    #[cfg(target_os = "macos")]
    {
        if let Some(brand) = std::process::Command::new("sysctl")
            .args(["-n", "machdep.cpu.brand_string"])
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
        {
            return brand.trim().to_lowercase().replace(' ', "-");
        }
    }
    // Linux fallback: read /proc/cpuinfo "model name" first line.
    if let Ok(content) = std::fs::read_to_string("/proc/cpuinfo") {
        for line in content.lines() {
            if let Some(rest) = line.strip_prefix("model name") {
                if let Some(name) = rest.split(':').nth(1) {
                    return name.trim().to_lowercase().replace(' ', "-");
                }
            }
        }
    }
    "unknown".to_string()
}

/// Best-effort NVIDIA driver version via `nvidia-smi --query-gpu=driver_version`.
/// Returns None on hosts without nvidia-smi.
pub fn detect_nvidia_driver() -> Option<String> {
    nvidia_smi_query("driver_version")
}

/// Best-effort CUDA toolkit version via `nvcc --version` or
/// `nvidia-smi --query-gpu=cuda_version`. Toolkit version (nvcc) is
/// reported when available, else driver-reported runtime version.
pub fn detect_cuda_version() -> Option<String> {
    // Try nvcc first — that's the toolkit (what ferrum compiles against).
    if let Ok(out) = std::process::Command::new("nvcc").arg("--version").output() {
        if let Ok(s) = String::from_utf8(out.stdout) {
            for line in s.lines() {
                if let Some(idx) = line.find("release ") {
                    let rest = &line[idx + 8..];
                    if let Some(comma) = rest.find(',') {
                        return Some(rest[..comma].trim().to_string());
                    }
                }
            }
        }
    }
    // Fall back to driver-reported runtime CUDA via nvidia-smi.
    nvidia_smi_query("cuda_version")
}

/// Query nvidia-smi for a single field. Returns None if nvidia-smi
/// isn't available or the field isn't supported.
fn nvidia_smi_query(field: &str) -> Option<String> {
    let out = std::process::Command::new("nvidia-smi")
        .args([
            &format!("--query-gpu={field}"),
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8(out.stdout).ok()?;
    let first = s.lines().next()?.trim().to_string();
    if first.is_empty() || first == "[Not Supported]" || first == "[N/A]" {
        return None;
    }
    Some(first)
}

/// GPU clock lock state (MHz). Returns the *current* graphics clock —
/// when `nvidia-smi -lgc N,N` is applied this equals the lock value;
/// without lock it equals whatever the GPU is currently doing.
pub fn detect_gpu_clock_lock_mhz() -> Option<u32> {
    nvidia_smi_query("clocks.gr")
        .and_then(|s| s.parse::<u32>().ok())
}

/// GPU power limit in watts.
pub fn detect_gpu_power_limit_w() -> Option<u32> {
    nvidia_smi_query("power.limit")
        .and_then(|s| s.split('.').next()?.parse::<u32>().ok())
}

/// Persistence mode (true ⇒ enabled).
pub fn detect_gpu_persistence() -> Option<bool> {
    nvidia_smi_query("persistence_mode").map(|s| s == "Enabled")
}

/// Best-effort Rust toolchain version. Tries `rustc --version` via the
/// `RUSTC` env (set by cargo) first, then plain `rustc`. Falls back to
/// the compile-time string the binary was compiled with via env var
/// inserted by build.rs (not yet wired — returns `unknown` until then).
pub fn detect_rust_version() -> String {
    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".to_string());
    std::process::Command::new(rustc)
        .arg("--version")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| {
            // "rustc 1.78.0 (9b00956e5 2024-04-29)" → "1.78.0"
            s.split_whitespace().nth(1).map(|v| v.to_string())
        })
        .unwrap_or_else(|| "unknown".to_string())
}

/// Snapshot all `FERRUM_*` env vars in the current process, sorted.
pub fn capture_ferrum_env() -> BTreeMap<String, String> {
    std::env::vars()
        .filter(|(k, _)| k.starts_with("FERRUM_"))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_env() -> Env {
        let mut ferrum_env = BTreeMap::new();
        ferrum_env.insert("FERRUM_KV_MAX_BLOCKS".into(), "2048".into());
        ferrum_env.insert("FERRUM_PREFIX_CACHE".into(), "0".into());
        Env {
            commit_sha: "b769bbd".into(),
            hw_id: "rtx-4090".into(),
            driver: Some("555.42.06".into()),
            cuda: Some("12.4".into()),
            rust: "1.78.0".into(),
            ferrum_features: vec!["cuda".into(), "vllm-moe-marlin".into()],
            gpu_clock_lock_mhz: Some(2520),
            gpu_power_limit_w: Some(350),
            gpu_persistence_mode: Some(true),
            gpu_auto_boost: Some(false),
            ferrum_env,
            vllm_args: None,
        }
    }

    #[test]
    fn env_hash_is_deterministic() {
        let h1 = fixture_env().hash();
        let h2 = fixture_env().hash();
        assert_eq!(h1, h2);
        assert!(h1.0.starts_with("sha256:"));
        assert_eq!(h1.0.len(), "sha256:".len() + 64);
    }

    #[test]
    fn env_hash_changes_on_clock_lock() {
        let h1 = fixture_env().hash();
        let mut e = fixture_env();
        e.gpu_clock_lock_mhz = Some(2400); // different lock value
        let h2 = e.hash();
        assert_ne!(h1, h2);
    }

    #[test]
    fn env_hash_changes_on_ferrum_env() {
        let h1 = fixture_env().hash();
        let mut e = fixture_env();
        e.ferrum_env
            .insert("FERRUM_VLLM_MOE".into(), "1".into());
        let h2 = e.hash();
        assert_ne!(h1, h2);
    }

    #[test]
    fn ferrum_env_order_independent() {
        // BTreeMap sorts by key, so insertion order should not matter.
        let mut e1 = fixture_env();
        e1.ferrum_env.clear();
        e1.ferrum_env.insert("A".into(), "1".into());
        e1.ferrum_env.insert("B".into(), "2".into());

        let mut e2 = fixture_env();
        e2.ferrum_env.clear();
        e2.ferrum_env.insert("B".into(), "2".into());
        e2.ferrum_env.insert("A".into(), "1".into());

        assert_eq!(e1.hash(), e2.hash());
    }
}
