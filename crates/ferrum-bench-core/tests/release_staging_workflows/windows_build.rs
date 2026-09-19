//! Execute the committed capacity helper with process-local memory and Cargo
//! substitutes. Assertions stay in Rust; no build or host configuration changes.
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::{
    fs,
    path::Path,
    process::{Command, Stdio},
};

const GIB: u64 = 1024 * 1024 * 1024;
// Win32_OperatingSystem reports whole KiB, including fractional GiB samples.
const MEMORY_3_8_GIB: u64 = 38 * GIB / 10 / 1024 * 1024;
const MEMORY_7_4_GIB: u64 = 74 * GIB / 10 / 1024 * 1024;

#[derive(Debug, Deserialize, PartialEq, Eq)]
struct Capacity {
    available_memory_bytes: u64,
    logical_cpu_count: u32,
    dependency_jobs: u32,
    selected_jobs: u32,
    job_policy: String,
}

#[derive(Debug, Deserialize)]
struct CapacityRecord {
    stage: String,
    #[serde(flatten)]
    capacity: Capacity,
}

struct Execution<T> {
    result: T,
    directory: tempfile::TempDir,
}

impl<T> Execution<T> {
    fn capacity_records(&self) -> Vec<CapacityRecord> {
        fs::read_to_string(self.directory.path().join("windows-build-capacity.jsonl"))
            .expect("read capacity evidence written by the actual helper")
            .lines()
            .map(|line| serde_json::from_str(line).expect("valid capacity record"))
            .collect()
    }
}

fn execute<T: DeserializeOwned>(script: &str, fixture: &impl Serialize) -> Execution<T> {
    let directory = tempfile::tempdir().unwrap();
    let result_path = directory.path().join("results.json");
    let helper = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.github/ci/windows-build.ps1");
    let command = format!(
        r#"
$ErrorActionPreference = 'Stop'
. $env:FERRUM_WINDOWS_BUILD_HELPER
$fixture = ConvertFrom-Json -InputObject $env:FERRUM_WINDOWS_BUILD_FIXTURE
{script}
ConvertTo-Json -InputObject $result -Depth 10 | Set-Content -LiteralPath $env:FERRUM_WINDOWS_BUILD_RESULTS -Encoding utf8
"#
    );
    let output = Command::new("pwsh.exe")
        .args([
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            &command,
        ])
        .current_dir(directory.path())
        .env("FERRUM_WINDOWS_BUILD_HELPER", helper)
        .env(
            "FERRUM_WINDOWS_BUILD_FIXTURE",
            serde_json::to_string(fixture).unwrap(),
        )
        .env("FERRUM_WINDOWS_BUILD_RESULTS", &result_path)
        .env("RUNNER_TEMP", directory.path())
        .env("GITHUB_STEP_SUMMARY", directory.path().join("summary.md"))
        .stdin(Stdio::null())
        .output()
        .expect("Windows capacity tests require PowerShell 7 as pwsh.exe on PATH");
    assert!(
        output.status.success(),
        "capacity harness failed: status={}; stdout={}; stderr={}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let result = serde_json::from_slice(&fs::read(result_path).expect("read harness result"))
        .expect("deserialize harness result");
    Execution { result, directory }
}

#[derive(Serialize)]
struct CapacityInput {
    available_memory_bytes: u64,
    logical_cpu_count: u32,
    parallel_link: bool,
}

#[test]
fn windows_capacity_respects_memory_reserve_cpu_limit_and_parallel_link() {
    let cases = [
        (MEMORY_3_8_GIB, 20, false, 1, 1),
        (MEMORY_7_4_GIB, 20, false, 3, 3),
        (0, 20, false, 1, 1),
        (GIB, 20, false, 1, 1),
        (5 * GIB - 1024, 20, false, 1, 1),
        (5 * GIB, 20, false, 2, 2),
        (MEMORY_7_4_GIB, 2, false, 2, 2),
        (MEMORY_3_8_GIB, 20, true, 1, 20),
    ];
    let inputs: Vec<_> = cases
        .iter()
        .map(
            |&(available_memory_bytes, logical_cpu_count, parallel_link, _, _)| CapacityInput {
                available_memory_bytes,
                logical_cpu_count,
                parallel_link,
            },
        )
        .collect();
    let execution: Execution<Vec<Capacity>> = execute(
        r#"
$result = @(
    foreach ($case in $fixture) {
        Get-FerrumBuildCapacity -AvailableMemoryBytes $case.available_memory_bytes -LogicalCpuCount $case.logical_cpu_count -ParallelLink:$case.parallel_link
    }
)
"#,
        &inputs,
    );
    let expected: Vec<_> = cases
        .into_iter()
        .map(
            |(
                available_memory_bytes,
                logical_cpu_count,
                parallel_link,
                dependency_jobs,
                selected_jobs,
            )| {
                Capacity {
                    available_memory_bytes,
                    logical_cpu_count,
                    dependency_jobs,
                    selected_jobs,
                    job_policy: if parallel_link {
                        "parallel_link"
                    } else {
                        "memory_budget"
                    }
                    .into(),
                }
            },
        )
        .collect();
    assert_eq!(execution.result, expected);
}

#[derive(Serialize)]
struct CargoStage<'a> {
    stage: &'a str,
    available_memory_bytes: u64,
    arguments: Vec<&'a str>,
    previous_jobs: Option<&'a str>,
    parallel_link: bool,
    exit_code: i32,
    throw_cargo: bool,
}

impl<'a> CargoStage<'a> {
    fn new(stage: &'a str, memory: u64, arguments: Vec<&'a str>) -> Self {
        Self {
            stage,
            available_memory_bytes: memory,
            arguments,
            previous_jobs: Some("17"),
            parallel_link: false,
            exit_code: 0,
            throw_cargo: false,
        }
    }
}

#[derive(Debug, Deserialize)]
struct CargoInvocation {
    stage: String,
    jobs: String,
    arguments: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct CargoOutcome {
    stage: String,
    capacity: Option<Capacity>,
    restored_jobs: Option<String>,
    error: Option<String>,
}

#[derive(Debug, Deserialize)]
struct CargoResults {
    outcomes: Vec<CargoOutcome>,
    invocations: Vec<CargoInvocation>,
}

const CARGO_HARNESS: &str = r#"
$script:invocations = [Collections.Generic.List[object]]::new()
function Get-CimInstance {
    param([string]$ClassName)
    if ($ClassName -ne 'Win32_OperatingSystem') { throw "Unexpected CIM query: $ClassName" }
    [pscustomobject]@{ FreePhysicalMemory = [uint64]($script:activeCase.available_memory_bytes / 1KB) }
}
function cargo {
    $script:invocations.Add([pscustomobject]@{
        stage = $script:activeCase.stage
        jobs = $env:CARGO_BUILD_JOBS
        arguments = @($args)
    })
    Write-Output 'Cargo fixture output must remain a log, not a returned capacity value.'
    $global:LASTEXITCODE = [int]$script:activeCase.exit_code
    if ($script:activeCase.throw_cargo) { throw 'Cargo fixture failed to start.' }
}
$outcomes = [Collections.Generic.List[object]]::new()
foreach ($case in $fixture) {
    $script:activeCase = $case
    if ($null -eq $case.previous_jobs) {
        Remove-Item Env:CARGO_BUILD_JOBS -ErrorAction SilentlyContinue
    } else {
        $env:CARGO_BUILD_JOBS = $case.previous_jobs
    }
    $capacity = $null
    $failure = $null
    try {
        $capacity = Invoke-FerrumCargo -Stage $case.stage -CargoArguments $case.arguments -ParallelLink:$case.parallel_link
    } catch {
        $failure = $_.Exception.Message
    }
    $outcomes.Add([pscustomobject]@{
        stage = $case.stage
        capacity = $capacity
        restored_jobs = [Environment]::GetEnvironmentVariable('CARGO_BUILD_JOBS', 'Process')
        error = $failure
    })
}
$result = [pscustomobject]@{ outcomes = $outcomes.ToArray(); invocations = $script:invocations.ToArray() }
"#;

#[test]
fn windows_cargo_refreshes_capacity_and_records_actual_jobs_without_changing_tail_arguments() {
    let initial = CargoStage::new("initial", MEMORY_3_8_GIB, vec!["check", "--workspace"]);
    let warmed = CargoStage::new(
        "warmed",
        MEMORY_7_4_GIB,
        vec![
            "test",
            "--workspace",
            "--",
            "--test-threads=1",
            "-j",
            "test-argument",
        ],
    );
    let mut pressured = CargoStage::new(
        "pressured",
        3 * GIB,
        vec!["rustc", "--bin", "ferrum", "--", "-C", "codegen-units=1"],
    );
    pressured.previous_jobs = None;
    let mut link = CargoStage::new("link", 3 * GIB, vec!["build", "--release"]);
    link.parallel_link = true;
    let cases = [initial, warmed, pressured, link];
    let execution: Execution<CargoResults> = execute(CARGO_HARNESS, &cases);
    let records = execution.capacity_records();
    for (case, memory_jobs) in cases.iter().zip([1, 3, 1, 1]) {
        let outcome = execution
            .result
            .outcomes
            .iter()
            .find(|run| run.stage == case.stage)
            .unwrap();
        assert_eq!(outcome.error, None, "{}: {:?}", case.stage, outcome.error);
        assert_eq!(outcome.restored_jobs.as_deref(), case.previous_jobs);
        let capacity = outcome
            .capacity
            .as_ref()
            .expect("successful invocation returns capacity");
        assert_eq!(capacity.available_memory_bytes, case.available_memory_bytes);
        assert!(capacity.logical_cpu_count > 0);
        assert_eq!(
            capacity.dependency_jobs,
            memory_jobs.min(capacity.logical_cpu_count)
        );
        let selected_jobs = if case.parallel_link {
            capacity.logical_cpu_count
        } else {
            capacity.dependency_jobs
        };
        assert_eq!(capacity.selected_jobs, selected_jobs);
        assert_eq!(
            capacity.job_policy,
            if case.parallel_link {
                "parallel_link"
            } else {
                "memory_budget"
            }
        );
        let invocation = execution
            .result
            .invocations
            .iter()
            .find(|run| run.stage == case.stage)
            .expect("Cargo was invoked");
        assert_eq!(invocation.jobs, selected_jobs.to_string());
        let jobs = selected_jobs.to_string();
        let expected = match case.stage {
            "initial" => vec!["check", "--workspace", "--jobs", &jobs],
            "warmed" => vec![
                "test",
                "--workspace",
                "--jobs",
                &jobs,
                "--",
                "--test-threads=1",
                "-j",
                "test-argument",
            ],
            "pressured" => vec![
                "rustc",
                "--bin",
                "ferrum",
                "--jobs",
                &jobs,
                "--",
                "-C",
                "codegen-units=1",
            ],
            "link" => vec!["build", "--release", "--jobs", &jobs],
            _ => unreachable!(),
        };
        assert_eq!(invocation.arguments, expected, "{}", case.stage);
        let record = records
            .iter()
            .find(|record| record.stage == case.stage)
            .expect("capacity evidence names this stage");
        assert_eq!(&record.capacity, capacity);
    }
    // Both a nonzero exit and a failure to start must propagate while restoring
    // the caller's previous environment, including an originally absent value.
    let mut nonzero = CargoStage::new("nonzero-exit", MEMORY_7_4_GIB, vec!["check"]);
    nonzero.exit_code = 9;
    let mut throwing = CargoStage::new("failed-start", MEMORY_3_8_GIB, vec!["test"]);
    throwing.throw_cargo = true;
    throwing.previous_jobs = None;
    let cases = [nonzero, throwing];
    let execution: Execution<CargoResults> = execute(CARGO_HARNESS, &cases);
    let records = execution.capacity_records();
    for case in &cases {
        let outcome = execution
            .result
            .outcomes
            .iter()
            .find(|run| run.stage == case.stage)
            .unwrap();
        assert!(
            outcome.error.is_some(),
            "failed Cargo stage must propagate an error"
        );
        assert!(
            outcome.capacity.is_none(),
            "failed Cargo stage must not report success"
        );
        assert_eq!(outcome.restored_jobs.as_deref(), case.previous_jobs);
        let invocation = execution
            .result
            .invocations
            .iter()
            .find(|run| run.stage == case.stage)
            .expect("Cargo failure was exercised");
        let record = records
            .iter()
            .find(|record| record.stage == case.stage)
            .expect("failed stages retain capacity evidence");
        assert_eq!(invocation.jobs, record.capacity.selected_jobs.to_string());
        assert_eq!(
            record.capacity.available_memory_bytes,
            case.available_memory_bytes
        );
    }
}
