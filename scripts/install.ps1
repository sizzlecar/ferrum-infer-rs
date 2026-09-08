#requires -Version 5.1
<#
.SYNOPSIS
Installs the official Ferrum Windows x64 CUDA sm89 release for the current user.
.PARAMETER Version
Optional formal version, for example 0.8.9. The default is the latest formal release.
#>
[CmdletBinding()]
param([string]$Version = '')

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function ConvertTo-FerrumVersion {
    param([Parameter(Mandatory=$true)][string]$Value)
    if ($Value -notmatch '^v?((0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*))$') {
        throw 'Version must be a formal version such as 0.8.9.'
    }
    return $Matches[1]
}

function Select-FerrumRelease {
    param([Parameter(Mandatory=$true)][object]$Release, [string]$RequestedVersion = '')
    if ($Release.draft -isnot [bool] -or $Release.prerelease -isnot [bool] -or $Release.draft -or $Release.prerelease) {
        throw 'Ferrum installation requires a published formal release.'
    }
    $resolved = ConvertTo-FerrumVersion -Value ([string]$Release.tag_name)
    if ([string]$Release.tag_name -cne ('v'+$resolved)) { throw 'Release tag is not canonical.' }
    if ($RequestedVersion -and $resolved -cne (ConvertTo-FerrumVersion -Value $RequestedVersion)) { throw 'Release version does not match the request.' }
    $name = 'ferrum-'+$resolved+'-windows-x86_64-cuda-sm89-setup.exe'
    $selected = @()
    foreach ($assetName in @($name, ($name+'.sha256'))) {
        $matchingAssets = @($Release.assets | Where-Object { [string]$_.name -ceq $assetName })
        if ($matchingAssets.Count -ne 1) { throw ('Release v'+$resolved+' does not provide exactly one '+$assetName+'. No Windows installer was selected.') }
        $asset = $matchingAssets[0]
        $expected = 'https://github.com/sizzlecar/ferrum-infer-rs/releases/download/v'+$resolved+'/'+$assetName
        if ([string]$asset.browser_download_url -cne $expected -or [long]$asset.size -le 0) { throw 'Release asset URL or byte size is invalid.' }
        $selected += [pscustomobject]@{name=$assetName;url=$expected;size=[long]$asset.size}
    }
    return [pscustomobject]@{version=$resolved;setup=$selected[0];checksum=$selected[1]}
}

function ConvertFrom-FerrumChecksum {
    param([Parameter(Mandatory=$true)][string]$Text, [Parameter(Mandatory=$true)][string]$AssetName)
    $line = $Text.Trim()
    $match = [regex]::Match($line, '^([0-9a-fA-F]{64})(?:[ \t]+\*?([a-zA-Z0-9._-]+))?$')
    if (-not $match.Success -or ($match.Groups[2].Success -and $match.Groups[2].Value -cne $AssetName)) {
        throw 'Installer checksum is malformed or names a different asset.'
    }
    return $match.Groups[1].Value.ToLowerInvariant()
}

function Confirm-FerrumFile {
    param([Parameter(Mandatory=$true)][string]$Path, [Parameter(Mandatory=$true)][string]$Sha256)
    if ($Sha256 -cnotmatch '^[0-9a-f]{64}$') { throw 'Expected SHA256 is invalid.' }
    $file = Get-Item -LiteralPath $Path -Force
    if ($file.PSIsContainer -or ($file.Attributes -band [IO.FileAttributes]::ReparsePoint)) { throw 'Installer must be a regular file.' }
    if ((Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant() -cne $Sha256) { throw 'Installer SHA256 mismatch; nothing was executed.' }
}

function Invoke-FerrumProcess {
    param([Parameter(Mandatory=$true)][string]$Program, [string[]]$Arguments = @())
    $quoted = @($Arguments | ForEach-Object {
        '"'+[regex]::Replace([regex]::Replace($_, '(\\*)"', '$1$1\"'), '(\\+)$', '$1$1')+'"'
    }) -join ' '
    $info = [Diagnostics.ProcessStartInfo]::new()
    $info.FileName = $Program
    $info.Arguments = $quoted
    $info.UseShellExecute = $false
    $info.RedirectStandardOutput = $true
    $info.RedirectStandardError = $true
    $process = [Diagnostics.Process]::new()
    $process.StartInfo = $info
    try {
        if (-not $process.Start()) { throw 'Could not start the requested program.' }
        $stdout = $process.StandardOutput.ReadToEndAsync()
        $stderr = $process.StandardError.ReadToEndAsync()
        $process.WaitForExit()
        $out = $stdout.GetAwaiter().GetResult()
        $err = $stderr.GetAwaiter().GetResult()
        if ($process.ExitCode -ne 0) { throw ('Program exited '+$process.ExitCode+': '+$Program+"`n"+$err) }
        return [pscustomobject]@{stdout=$out;stderr=$err;exit_code=$process.ExitCode}
    } finally { $process.Dispose() }
}

function Assert-FerrumHardware {
    param([Parameter(Mandatory=$true)][string]$Architecture, [Parameter(Mandatory=$true)][string]$GpuCsv)
    if ($Architecture -cne 'X64') { throw 'This installer supports native Windows x64 only.' }
    $match = [regex]::Match($GpuCsv.Trim(), '^8\.9\s*,\s*([0-9]+\.[0-9]+)$')
    if (-not $match.Success) { throw 'This release requires NVIDIA GPU 0 with CUDA compute capability 8.9 (sm89).' }
    # CUDA 12.4 Update 1 release notes, Table 3: Windows driver >=551.78.
    # Ferrum loads PTX; the lower CUDA 12.x minor-compatibility floor is insufficient.
    # https://docs.nvidia.com/cuda/archive/12.4.1/cuda-toolkit-release-notes/index.html
    # https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html
    if ([version]$match.Groups[1].Value -lt [version]'551.78') { throw 'Update the NVIDIA driver to 551.78 or later: https://www.nvidia.com/drivers/ . CUDA Toolkit is not required.' }
}

function Add-FerrumProcessPath {
    param([Parameter(Mandatory=$true)][string]$Directory)
    if (-not [IO.Directory]::Exists($Directory) -or $Directory.Contains(';')) { throw 'Installed Ferrum directory is not a usable PATH entry.' }
    $key = $Directory.Replace('/','\').TrimEnd([char[]]@('\'))
    $current = [string][Environment]::GetEnvironmentVariable('Path','Process')
    $remaining = [Collections.Generic.List[string]]::new()
    foreach ($entry in $current.Split(';')) {
        if (-not [string]::Equals($entry.Replace('/','\').TrimEnd([char[]]@('\')), $key, [StringComparison]::OrdinalIgnoreCase)) { $remaining.Add($entry) }
    }
    # Inno owns the persistent user PATH. Update only this PowerShell process so
    # this installation wins over older executables without changing other entries.
    $next = $Directory.Replace('/','\').TrimEnd([char[]]@('\'))
    if ($next.EndsWith(':')) { $next += '\' }
    if ($current.Length -gt 0 -and $remaining.Count -gt 0) { $next += ';'+($remaining -join ';') }
    [Environment]::SetEnvironmentVariable('Path', $next, 'Process')
}

function Confirm-FerrumCommand {
    param([Parameter(Mandatory=$true)][string]$Program)
    $command = Get-Command -Name ferrum -ErrorAction SilentlyContinue
    $expected = [IO.Path]::GetFullPath($Program)
    if ($null -eq $command -or $command.CommandType -ne [System.Management.Automation.CommandTypes]::Application -or
        -not [string]::Equals([IO.Path]::GetFullPath($command.Path), $expected, [StringComparison]::OrdinalIgnoreCase)) {
        throw ("Ferrum is installed, but 'ferrum' resolves to another command in this PowerShell session. Your existing command was preserved. Use the installed executable directly: & '"+$expected.Replace("'","''")+"'")
    }
}

function Install-FerrumSetup {
    param([Parameter(Mandatory=$true)][string]$SetupPath, [Parameter(Mandatory=$true)][string]$Sha256, [Parameter(Mandatory=$true)][string]$ExpectedVersion)
    $expected = ConvertTo-FerrumVersion -Value $ExpectedVersion
    Confirm-FerrumFile -Path $SetupPath -Sha256 $Sha256
    $null = Invoke-FerrumProcess -Program $SetupPath -Arguments @('/VERYSILENT','/SUPPRESSMSGBOXES','/NORESTART','/SP-')
    $installed = Join-Path ([Environment]::GetFolderPath('LocalApplicationData')) 'Programs\Ferrum\ferrum.exe'
    $result = Invoke-FerrumProcess -Program $installed -Arguments @('--version')
    if ($result.stdout.Trim() -cne ('ferrum '+$expected)) { throw 'Installed Ferrum version does not match the selected release.' }
    Add-FerrumProcessPath -Directory ([IO.Path]::GetDirectoryName($installed))
    Confirm-FerrumCommand -Program $installed
    Write-Host ('Installed Ferrum '+$expected+' (Windows x64, CUDA sm89).')
    Write-Host 'Ready in this terminal. Example (downloads the model on first use): ferrum run qwen3:0.6b'
    Write-Host 'To select another model, see: ferrum run --help'
}

function Install-FerrumRelease {
    param([string]$RequestedVersion = '')
    if ([Environment]::OSVersion.Platform -ne [PlatformID]::Win32NT) { throw 'This installer requires Windows.' }
    $architecture = [Runtime.InteropServices.RuntimeInformation]::OSArchitecture.ToString()
    if ($architecture -cne 'X64') { throw 'This installer supports native Windows x64 only.' }
    $candidates = @((Join-Path ([Environment]::GetFolderPath('System')) 'nvidia-smi.exe'), (Join-Path ([Environment]::GetFolderPath('ProgramFiles')) 'NVIDIA Corporation\NVSMI\nvidia-smi.exe'))
    $smi = @($candidates | Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } | Select-Object -First 1)
    if ($smi.Count -eq 0) { throw 'NVIDIA driver tools were not found. Install the NVIDIA driver: https://www.nvidia.com/drivers/ . CUDA Toolkit is not required.' }
    $gpu = Invoke-FerrumProcess -Program $smi[0] -Arguments @('-i','0','--query-gpu=compute_cap,driver_version','--format=csv,noheader,nounits')
    Assert-FerrumHardware -Architecture $architecture -GpuCsv $gpu.stdout
    $endpoint = 'https://api.github.com/repos/sizzlecar/ferrum-infer-rs/releases/latest'
    if ($RequestedVersion) { $endpoint = 'https://api.github.com/repos/sizzlecar/ferrum-infer-rs/releases/tags/v'+(ConvertTo-FerrumVersion -Value $RequestedVersion) }
    [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
    $release = Invoke-RestMethod -Uri $endpoint -Headers @{Accept='application/vnd.github+json';'User-Agent'='Ferrum-Windows-Installer'}
    $selected = Select-FerrumRelease -Release $release -RequestedVersion $RequestedVersion
    $temporary = Join-Path ([IO.Path]::GetTempPath()) ('Ferrum-install-'+[Guid]::NewGuid().ToString('N'))
    $null = [IO.Directory]::CreateDirectory($temporary)
    try {
        $ProgressPreference = 'SilentlyContinue'
        foreach ($asset in @($selected.setup,$selected.checksum)) {
            $destination = Join-Path $temporary $asset.name
            Invoke-WebRequest -UseBasicParsing -Uri $asset.url -OutFile $destination
            if ((Get-Item -LiteralPath $destination).Length -ne $asset.size) { throw 'Downloaded release asset byte size differs from GitHub metadata.' }
        }
        $checksumPath = Join-Path $temporary $selected.checksum.name
        if ((Get-Item -LiteralPath $checksumPath).Length -gt 4096) { throw 'Installer checksum file is unexpectedly large.' }
        $sha = ConvertFrom-FerrumChecksum -Text ([IO.File]::ReadAllText($checksumPath)) -AssetName $selected.setup.name
        Install-FerrumSetup -SetupPath (Join-Path $temporary $selected.setup.name) -Sha256 $sha -ExpectedVersion $selected.version
    } finally { Remove-Item -LiteralPath $temporary -Recurse -Force }
}

# Dot-sourcing exposes the same product functions for Rust-driven contract tests.
if ($MyInvocation.InvocationName -ne '.') { Install-FerrumRelease -RequestedVersion $Version }
