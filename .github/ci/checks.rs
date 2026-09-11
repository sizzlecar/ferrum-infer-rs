//! Compilation and device execution have different consumers. In particular,
//! GPU builds include CPU code, but CPU vNext execution is not a GPU test oracle.
use super::{is_documentation, Outcome};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Check {
    Cpu,
    Metal,
    Cuda,
    MetalRuntime,
    CudaRuntime,
    Windows,
}

impl Check {
    pub const ALL: [Self; 6] = [
        Self::Cpu,
        Self::Metal,
        Self::Cuda,
        Self::MetalRuntime,
        Self::CudaRuntime,
        Self::Windows,
    ];
    pub fn bit(self) -> u8 {
        1 << self as u8
    }
    pub fn name(self) -> &'static str {
        match self {
            Self::Cpu => "CPU (Linux)",
            Self::Metal => "Metal (macOS)",
            Self::Cuda => "CUDA (Linux)",
            Self::MetalRuntime => "GPU runtime (metal)",
            Self::CudaRuntime => "GPU runtime (cuda)",
            Self::Windows => "Windows (MSVC contracts)",
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Checks(pub u8);
impl Checks {
    pub const ALL: Self = Self(63);
    pub fn has(self, check: Check) -> bool {
        self.0 & check.bit() != 0
    }
    fn of(checks: &[Check]) -> Self {
        Self(checks.iter().fold(0, |mask, check| mask | check.bit()))
    }
}

fn valid(path: &str) -> bool {
    !path.contains(['\\', '\n', '\r', ':'])
        && path.split('/').all(|part| !matches!(part, "" | "." | ".."))
}

fn below(path: &str, root: &str) -> bool {
    path.strip_prefix(root).is_some_and(|tail| !tail.is_empty())
}

fn path_checks(path: &str) -> Checks {
    use Check::*;
    if !valid(path) {
        return Checks::ALL;
    }
    if is_documentation(path) {
        return Checks::default();
    }
    let kernel = "crates/ferrum-kernels/src/";
    if let Some(module) = path.strip_prefix(kernel) {
        if below(module, "backend/metal/") {
            return Checks::of(&[Cpu, Metal, MetalRuntime]);
        }
        if below(module, "backend/cuda/") {
            return Checks::of(&[Cpu, Cuda, CudaRuntime]);
        }
        if matches!(
            module,
            "backend/cpu/vnext_ops.rs"
                | "backend/cpu/vnext_runtime.rs"
                | "quant_linear/cpu_dequant.rs"
                | "quant_linear/cpu_gguf.rs"
                | "quant_linear/cpu_marlin_stack.rs"
        ) || below(module, "backend/cpu/vnext_ops/")
            || below(module, "backend/cpu/vnext_runtime/")
        {
            return Checks::of(&[Cpu, Metal, Cuda, Windows]);
        }
        // cpu.rs provides reference answers for GPU differential tests;
        // attention/cpu can be a fallback in Metal products. Neither is isolated.
        return Checks::ALL;
    }
    if matches!(path, "crates/ferrum-bench-core/examples/regression_plan.rs")
        || below(path, "crates/ferrum-bench-core/examples/regression_plan/")
    {
        return Checks::of(&[Cpu]);
    }
    if path == "crates/ferrum-devtools/src/bin/release_delivery.rs"
        || below(path, "crates/ferrum-devtools/src/bin/release_delivery/")
    {
        return Checks::of(&[Cpu, Metal, Windows]);
    }
    if matches!(
        path,
        "crates/ferrum-cli/src/bin/ferrum-launcher.rs"
            | "crates/ferrum-cli/tests/windows_launcher.rs"
            | "crates/ferrum-devtools/tests/windows_bootstrap.rs"
            | "crates/ferrum-bench-core/tests/release_staging_workflows.rs"
    ) || below(path, "crates/ferrum-cli/src/bin/launcher/")
        || matches!(
            path,
            "scripts/install.ps1"
                | "packaging/windows/ferrum.iss"
                | ".github/workflows/release-windows.yml"
        )
    {
        return Checks::of(&[Cpu, Windows]);
    }
    if path == "scripts/install.sh" {
        return Checks::of(&[Cpu, Metal]);
    }
    Checks::ALL
}

pub fn affected(input: &[u8]) -> Checks {
    let Some(input) = input.strip_suffix(&[0]) else {
        return Checks::ALL;
    };
    let Ok(paths) = std::str::from_utf8(input) else {
        return Checks::ALL;
    };
    Checks(
        paths
            .split('\0')
            .fold(0, |mask, path| mask | path_checks(path).0),
    )
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Plan {
    pub required: Checks,
    pub reused: Checks,
}
impl Plan {
    pub fn fresh(required: Checks) -> Self {
        Self {
            required,
            reused: Checks::default(),
        }
    }
    pub fn run(self) -> Checks {
        Checks(self.required.0 & !self.reused.0)
    }
    pub fn encode(self) -> String {
        format!("v1:{}:{}", self.required.0, self.reused.0)
    }
    pub fn parse(value: &str) -> Result<Self, String> {
        let parts: Vec<_> = value.split(':').collect();
        if parts.len() != 3 || parts[0] != "v1" {
            return Err("unknown CI plan format".into());
        }
        let required = parts[1]
            .parse::<u8>()
            .map_err(|_| "invalid required checks")?;
        let reused = parts[2]
            .parse::<u8>()
            .map_err(|_| "invalid reused checks")?;
        if required > Checks::ALL.0 || reused & !required != 0 {
            return Err("invalid CI plan coverage".into());
        }
        Ok(Self {
            required: Checks(required),
            reused: Checks(reused),
        })
    }
    pub fn output(self) {
        let run = self.run();
        println!(
            "scope={}",
            if self.required.0 == 0 { "docs" } else { "code" }
        );
        println!("plan={}", self.encode());
        for (key, check) in [
            ("cpu", Check::Cpu),
            ("metal", Check::Metal),
            ("cuda", Check::Cuda),
            ("windows", Check::Windows),
        ] {
            println!("{key}={}", run.has(check));
        }
        let mut entries = Vec::new();
        if run.has(Check::MetalRuntime) {
            entries.push(
                r#"{"backend":"metal","runner":["self-hosted","macOS","ARM64","ferrum-metal"]}"#,
            );
        }
        if run.has(Check::CudaRuntime) {
            entries.push(r#"{"backend":"cuda","runner":["self-hosted","Linux","X64","ferrum-cuda","cuda-sm89"]}"#);
        }
        println!("gpu={}", !entries.is_empty());
        // A skipped job still needs a syntactically valid matrix at expansion.
        if entries.is_empty() {
            entries.push(
                r#"{"backend":"metal","runner":["self-hosted","macOS","ARM64","ferrum-metal"]}"#,
            );
        }
        println!("gpu_matrix={{\"include\":[{}]}}", entries.join(","));
    }
    pub fn aggregate(self, jobs: [&str; 5]) -> Result<(), String> {
        let run = self.run();
        for ((name, required), result) in [
            ("CPU", run.has(Check::Cpu)),
            ("Metal", run.has(Check::Metal)),
            ("CUDA", run.has(Check::Cuda)),
            (
                "GPU runtime",
                run.has(Check::MetalRuntime) || run.has(Check::CudaRuntime),
            ),
            ("Windows", run.has(Check::Windows)),
        ]
        .into_iter()
        .zip(jobs)
        {
            let expected = if required {
                Outcome::Success
            } else {
                Outcome::Skipped
            };
            let actual = Outcome::parse(result)?;
            if actual != expected {
                return Err(format!("{name} requires {expected:?}, got {actual:?}"));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn paths(paths: &[&str]) -> Vec<u8> {
        let mut bytes = paths.join("\0").into_bytes();
        bytes.push(0);
        bytes
    }
    #[test]
    fn isolated_execution_keeps_shared_feature_compilation() {
        use Check::*;
        let cpu = affected(&paths(&[
            "crates/ferrum-kernels/src/backend/cpu/vnext_runtime.rs",
        ]));
        assert_eq!(cpu, Checks::of(&[Cpu, Metal, Cuda, Windows]));
        let metal = affected(&paths(&[
            "crates/ferrum-kernels/src/backend/metal/vnext_ops/mod.rs",
        ]));
        assert_eq!(metal, Checks::of(&[Cpu, Metal, MetalRuntime]));
        for path in [
            "crates/ferrum-kernels/src/backend/cpu.rs",
            "crates/ferrum-kernels/src/attention/cpu/mod.rs",
            "Cargo.lock",
            "crates/ferrum-engine/src/registry.rs",
        ] {
            assert_eq!(affected(&paths(&[path])), Checks::ALL, "{path}");
        }
    }
    #[test]
    fn complete_diff_unions_shared_changes_and_keeps_rename_sources() {
        for shared in [
            "Cargo.toml",
            "crates/ferrum-types/src/lib.rs",
            "future/backend.rs",
            "docs/fixture.json",
        ] {
            assert_eq!(
                affected(&paths(&[
                    "crates/ferrum-kernels/src/backend/metal/buffer.rs",
                    shared
                ])),
                Checks::ALL
            );
        }
        assert_eq!(
            affected(&paths(&["README.md", "docs/install.md"])),
            Checks::default()
        );
        assert_eq!(
            affected(&paths(&["crates/example.rs", "docs/example.md"])),
            Checks::ALL
        );
        for input in [
            &b""[..],
            b"README.md",
            b"\0",
            b"README.md\0\0",
            b"docs/../README.md\0",
            b"docs/\xff.md\0",
        ] {
            assert_eq!(affected(input), Checks::ALL);
        }
    }
    #[test]
    fn planner_and_windows_installer_do_not_require_device_arithmetic() {
        assert_eq!(
            affected(&paths(&[
                "crates/ferrum-bench-core/examples/regression_plan/native_artifacts.rs"
            ])),
            Checks::of(&[Check::Cpu])
        );
        for path in [
            "scripts/install.ps1",
            "crates/ferrum-cli/src/bin/ferrum-launcher.rs",
            "crates/ferrum-cli/src/bin/launcher/windows.rs",
            "crates/ferrum-devtools/tests/windows_bootstrap.rs",
            "crates/ferrum-bench-core/tests/release_staging_workflows.rs",
        ] {
            assert_eq!(
                affected(&paths(&[path])),
                Checks::of(&[Check::Cpu, Check::Windows]),
                "{path}"
            );
        }
        for path in [
            "crates/ferrum-devtools/src/bin/release_delivery.rs",
            "crates/ferrum-devtools/src/bin/release_delivery/gate.rs",
        ] {
            assert_eq!(
                affected(&paths(&[path])),
                Checks::of(&[Check::Cpu, Check::Metal, Check::Windows]),
                "{path}"
            );
        }
    }
    #[test]
    fn expected_skips_and_reuse_never_hide_selected_job_failure() {
        let plan = Plan {
            required: Checks::ALL,
            reused: Checks::of(&[
                Check::Metal,
                Check::Cuda,
                Check::MetalRuntime,
                Check::CudaRuntime,
            ]),
        };
        assert_eq!(Plan::parse(&plan.encode()).unwrap(), plan);
        assert!(plan
            .aggregate(["success", "skipped", "skipped", "skipped", "success"])
            .is_ok());
        for failure in ["failure", "cancelled", "skipped"] {
            assert!(plan
                .aggregate([failure, "skipped", "skipped", "skipped", "success"])
                .is_err());
        }
        for invalid in ["v1:64:0", "v1:0:1", "v1:1:2", "v1:1", "v2:1:0"] {
            assert!(Plan::parse(invalid).is_err());
        }
    }
}
