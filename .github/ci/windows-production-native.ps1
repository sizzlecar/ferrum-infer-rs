# Shared source staging for the Windows production quantized projection operators.
# The caller verifies the pinned llama.cpp revision before invoking this helper.
function Initialize-FerrumUpstreamNativeSource {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][string]$Workspace,
        [Parameter(Mandatory)][string]$LlamaSource,
        [Parameter(Mandatory)][ValidateSet('upstream-linear', 'upstream-extra-linear', 'upstream-q6-f32-linear')][string]$Operator,
        [Parameter(Mandatory)][string]$Output
    )
    $ErrorActionPreference = 'Stop'
    $definition = Get-Content "$Workspace/native-operators/cuda/source-definitions/$Operator.json" -Raw | ConvertFrom-Json
    New-Item -ItemType Directory $Output | Out-Null
    foreach ($relative in @($definition.translation_units) + @($definition.headers)) {
        if ($relative -match '^kernels/[^/]+$') {
            $source = Join-Path "$Workspace/native-operators/cuda/$Operator" (Split-Path $relative -Leaf)
        } elseif ($relative.StartsWith('llama/') -and $relative -notmatch '(^|/)\.\.(/|$)') {
            $source = Join-Path $LlamaSource $relative.Substring(6)
        } else {
            throw "Unsupported upstream source member: $relative"
        }
        $destination = Join-Path $Output $relative
        New-Item -ItemType Directory -Force (Split-Path $destination) | Out-Null
        Copy-Item -LiteralPath $source -Destination $destination
    }
    foreach ($directory in $definition.include_dirs) {
        New-Item -ItemType Directory -Force (Join-Path $Output $directory) | Out-Null
    }
}

function Write-FerrumWindowsNativePackageDefinition {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][string]$Definition,
        [Parameter(Mandatory)][string]$Output
    )
    $ErrorActionPreference = 'Stop'
    $package = Get-Content $Definition -Raw | ConvertFrom-Json
    # This is a new MSVC/CUDA 12.4 build, not a relabeled Linux archive.
    # package validates these values against the actual source-build receipt.
    $package.compute_capabilities = @('sm_89')
    $package.cuda_toolkit = '12.4'
    $package.cuda_runtime_min = '12.4'
    $package | ConvertTo-Json -Depth 30 | Set-Content $Output -Encoding utf8NoBOM
}
