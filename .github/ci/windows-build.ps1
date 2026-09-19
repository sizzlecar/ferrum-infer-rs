#requires -Version 7.0
# Shared implementation for Windows staging Cargo invocations. Capacity is
# sampled immediately before each build, not persisted for the entire job.

function Get-FerrumBuildCapacity {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][UInt64]$AvailableMemoryBytes,
        [Parameter(Mandatory)][ValidateRange(1, 2147483647)][int]$LogicalCpuCount,
        [switch]$ParallelLink
    )

    # Retain 1 GiB for the OS and budget 2 GiB per dependency compiler.
    $memoryJobs = [math]::Floor(($AvailableMemoryBytes / 1GB - 1) / 2)
    $dependencyJobs = [int][math]::Max(1, [math]::Min($LogicalCpuCount, $memoryJobs))
    [pscustomobject][ordered]@{
        available_memory_bytes = $AvailableMemoryBytes
        logical_cpu_count = $LogicalCpuCount
        dependency_jobs = $dependencyJobs
        selected_jobs = $(if ($ParallelLink) { $LogicalCpuCount } else { $dependencyJobs })
        job_policy = $(if ($ParallelLink) { 'parallel_link' } else { 'memory_budget' })
    }
}

function Invoke-FerrumCargo {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][ValidateNotNullOrEmpty()][string]$Stage,
        [Parameter(Mandatory)][ValidateNotNullOrEmpty()][string[]]$CargoArguments,
        [switch]$ParallelLink
    )

    $ErrorActionPreference = 'Stop'
    # Check the exit code explicitly while always restoring this process's env.
    $PSNativeCommandUseErrorActionPreference = $false
    $memory = Get-CimInstance Win32_OperatingSystem
    $capacity = Get-FerrumBuildCapacity `
        -AvailableMemoryBytes ([UInt64]$memory.FreePhysicalMemory * 1KB) `
        -LogicalCpuCount ([Environment]::ProcessorCount) -ParallelLink:$ParallelLink
    $record = [pscustomobject][ordered]@{
        schema_version = 1
        stage = $Stage
        sampled_at = [DateTime]::UtcNow.ToString('o')
        available_memory_bytes = $capacity.available_memory_bytes
        logical_cpu_count = $capacity.logical_cpu_count
        dependency_jobs = $capacity.dependency_jobs
        selected_jobs = $capacity.selected_jobs
        job_policy = $capacity.job_policy
    }
    $message = 'Windows build [{0}]: {1} logical CPUs, {2:N2} GiB available RAM, {3} Cargo jobs ({4}).' -f `
        $Stage, $capacity.logical_cpu_count, ($capacity.available_memory_bytes / 1GB), `
        $capacity.selected_jobs, $capacity.job_policy
    Write-Host $message
    if ($env:GITHUB_STEP_SUMMARY) {
        Add-Content -LiteralPath $env:GITHUB_STEP_SUMMARY -Value $message -Encoding utf8
    }
    if ($env:RUNNER_TEMP) {
        $evidence = Join-Path $env:RUNNER_TEMP 'windows-build-capacity.jsonl'
        $record | ConvertTo-Json -Compress | Add-Content -LiteralPath $evidence -Encoding utf8
    }

    # Explicit --jobs also overrides Cargo config files. Keep rustc/test args
    # after their separator intact; callers cannot supply a second job policy.
    $arguments = [Collections.Generic.List[string]]::new()
    $insertedJobs = $false
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
        # Cargo propagates this invocation's jobserver to rustc/codegen/LTO.
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
