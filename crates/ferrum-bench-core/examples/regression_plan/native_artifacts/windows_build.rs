//! Finite Windows Cargo consumption routes, with immutable helper provenance.
//! This does not interpret PowerShell; unknown routes retain native kernel scope.
use super::{digest, require_fragments, Snapshot, WINDOWS_WORKFLOW};
use serde_json::{json, Value};

pub(super) const HELPER: &str = ".github/ci/windows-build.ps1";

pub(super) fn helper_path(workflow: &str) -> Option<&'static str> {
    (workflow.contains(HELPER) || workflow.contains("Invoke-FerrumCargo")).then_some(HELPER)
}

pub(super) fn unchanged_helper(
    before: &Snapshot,
    after: &Snapshot,
) -> Result<Option<Value>, String> {
    let old_path = helper_path(before.get(WINDOWS_WORKFLOW)?);
    let new_path = helper_path(after.get(WINDOWS_WORKFLOW)?);
    if old_path != new_path {
        return Err("Windows Cargo helper route changed".into());
    }
    let Some(path) = new_path else {
        return Ok(None);
    };
    let source = after.get(path)?;
    if before.get(path)? != source {
        return Err(format!("Windows Cargo helper changed: {path}"));
    }
    // Forwarding and failure propagation are part of the reviewed consumption
    // contract. Runtime behavior is separately exercised by Windows helper tests
    // and the mandatory staging gate, not asserted from a source digest.
    require_boundary(
        source,
        r#"
        foreach ($argument in $CargoArguments) {
            if (-not $insertedJobs) {
                if ($argument -match '^(--jobs(?:=|$)|-j)') {
                    throw 'Invoke-FerrumCargo selects --jobs from current build capacity.'
                }
                if ($argument -eq '--') {
                    $arguments.Add('--jobs')
                    $arguments.Add([string]$capacity.selected_jobs)
                    $insertedJobs = $true
                }
            }
            $arguments.Add($argument)
        }
        if (-not $insertedJobs) {
            $arguments.Add('--jobs')
            $arguments.Add([string]$capacity.selected_jobs)
        }
        $previousJobs = $env:CARGO_BUILD_JOBS
        try {
            $env:CARGO_BUILD_JOBS = [string]$capacity.selected_jobs
            & cargo @arguments | Out-Host
            if ($LASTEXITCODE -ne 0) {
                throw "Cargo stage '$Stage' failed with exit code $LASTEXITCODE."
            }
        } finally {
            $env:CARGO_BUILD_JOBS = $previousJobs
        }
        return $record
        }
        "#,
        Boundary::Suffix,
    )?;
    Ok(Some(
        json!({"path": path, "sha256": digest(source.as_bytes())}),
    ))
}

fn active_lines(source: &str) -> Vec<&str> {
    source
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .collect()
}

enum Boundary {
    Prefix,
    Suffix,
}

fn require_boundary(source: &str, reviewed: &str, boundary: Boundary) -> Result<(), String> {
    // This finite route does not admit block comments or here-strings: their
    // bodies could look like commands without executing them.
    if ["<#", "#>", "@'", "@\""].iter().any(|s| source.contains(s)) {
        return Err("unreviewed PowerShell quoting in Windows Cargo route".into());
    }
    let lines = active_lines(source);
    let reviewed = active_lines(reviewed);
    let matches = match boundary {
        Boundary::Prefix => lines.starts_with(&reviewed),
        Boundary::Suffix => lines.ends_with(&reviewed),
    };
    if !matches {
        return Err("unreviewed Windows Cargo helper consumption or failure contract".into());
    }
    Ok(())
}

pub(super) fn verify_executable(script: &str, has_helper: bool) -> Result<(), String> {
    if !script.contains("Invoke-FerrumCargo") {
        // Historical inline commands remain a separate supported route. An
        // inline/helper migration is rejected by whole-workflow equality first.
        return require_fragments(
            script,
            &[
                "FERRUM_NATIVE_OPERATOR_SET_LOCK",
                "cargo build --release --locked -p ferrum-cli --bin ferrum",
                "cuda,vllm-moe-marlin,vllm-paged-attn-v2",
            ],
        );
    }
    if !has_helper {
        return Err("Windows executable calls a helper without immutable input".into());
    }
    // Keep the literal argv attached to each actual call and backend branch.
    // Matching independent feature/package substrings could accept a comment or
    // an unused array while the invoked helper compiled another artifact.
    require_boundary(
        script,
        r#"
        $ErrorActionPreference = 'Stop'
        $PSNativeCommandUseErrorActionPreference = $true
        . './.github/ci/windows-build.ps1'
        if ($env:BACKEND -eq 'cpu') {
            Remove-Item Env:FERRUM_NATIVE_OPERATOR_SET_LOCK -ErrorAction SilentlyContinue
            $dependencies = Invoke-FerrumCargo -Stage 'cpu-executable' -CargoArguments @('build', '--release', '--locked', '-p', 'ferrum-cli', '--bin', 'ferrum', '--timings')
            $link = $dependencies
            $binary = 'target/release/ferrum.exe'
            $profile = 'release'
        } else {
            if (-not (Test-Path $env:FERRUM_NATIVE_OPERATOR_SET_LOCK -PathType Leaf)) { throw 'The native operator set is missing' }
            $dependencies = Invoke-FerrumCargo -Stage 'cuda-dependencies' -CargoArguments @('build', '--profile', 'release-thin', '--locked', '-p', 'ferrum-cli', '--lib', '--features', 'cuda,vllm-moe-marlin,vllm-paged-attn-v2', '--timings')
            $link = Invoke-FerrumCargo -Stage 'cuda-link' -ParallelLink -CargoArguments @('build', '--profile', 'release-thin', '--locked', '-p', 'ferrum-cli', '--bin', 'ferrum', '--features', 'cuda,vllm-moe-marlin,vllm-paged-attn-v2', '--timings')
            $binary = 'target/release-thin/ferrum.exe'
            $profile = 'release-thin'
        "#,
        Boundary::Prefix,
    )
}
