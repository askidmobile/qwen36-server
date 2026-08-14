[CmdletBinding()]
param(
    [ValidateSet('Acquire', 'AuditSource', 'PrepareConverter')]
    [string]$Action = 'Acquire',
    [string]$Workspace = 'D:\Projects\yttri-inference',
    [string]$ModelRoot = 'D:\Models\yttri\qwen3.5-4b',
    [string]$ProjectSource,
    [switch]$DownloadWeights
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$Repository = 'Qwen/Qwen3.5-4B'
$ModelRevision = '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'
$LlamaRevision = '8e7f22b67ef4667b4ddd50230771287f328cfb3f'
$TransformersRevision = '00e8e49eb3eda67290f635f6bdf59f236f6adf7e'
$ForbiddenRoot = 'D:\Projects\yttri-build'

function Resolve-NormalizedPath([string]$Path, [switch]$AllowMissing) {
    $expanded = [Environment]::ExpandEnvironmentVariables($Path)
    if (-not $AllowMissing -and -not (Test-Path -LiteralPath $expanded)) {
        throw "Path does not exist: $expanded"
    }
    return [IO.Path]::GetFullPath($expanded).TrimEnd('\')
}

function Assert-UnderRoot([string]$Path, [string[]]$AllowedRoots) {
    $candidate = Resolve-NormalizedPath $Path -AllowMissing
    $forbidden = Resolve-NormalizedPath $ForbiddenRoot -AllowMissing
    if ($candidate.Equals($forbidden, [StringComparison]::OrdinalIgnoreCase) -or
        $candidate.StartsWith($forbidden + '\', [StringComparison]::OrdinalIgnoreCase)) {
        throw "Forbidden workspace path: $candidate"
    }
    foreach ($rootPath in $AllowedRoots) {
        $root = Resolve-NormalizedPath $rootPath -AllowMissing
        if ($candidate.Equals($root, [StringComparison]::OrdinalIgnoreCase) -or
            $candidate.StartsWith($root + '\', [StringComparison]::OrdinalIgnoreCase)) {
            return $candidate
        }
    }
    throw "Path is outside allowed roots: $candidate"
}

$Workspace = Assert-UnderRoot $Workspace @('D:\Projects\yttri-inference')
$ModelRoot = Assert-UnderRoot $ModelRoot @('D:\Models\yttri')
if (-not $ProjectSource) { $ProjectSource = Join-Path $Workspace 'qwen36-server' }
$ProjectSource = Assert-UnderRoot $ProjectSource @($Workspace)
$SourceRoot = Assert-UnderRoot (Join-Path $ModelRoot "source\$ModelRevision") @($ModelRoot)
$ToolsRoot = Assert-UnderRoot (Join-Path $Workspace 'tools') @($Workspace)
$LlamaRoot = Assert-UnderRoot (Join-Path $ToolsRoot "llama.cpp-$LlamaRevision") @($Workspace)
$ReportsRoot = Assert-UnderRoot (Join-Path $Workspace 'bench\qwen35-artifacts') @($Workspace)

function Invoke-Native([string]$FilePath, [string[]]$Arguments, [string]$WorkingDirectory = $Workspace) {
    Push-Location $WorkingDirectory
    try {
        & $FilePath @Arguments
        if ($LASTEXITCODE -ne 0) { throw "$FilePath exited with code $LASTEXITCODE" }
    } finally {
        Pop-Location
    }
}

function Get-HfFile([string]$Name, [string]$Destination) {
    $uri = "https://huggingface.co/Qwen/Qwen3.5-4B/resolve/$ModelRevision/$Name"
    $tmp = "$Destination.partial"
    Remove-Item -LiteralPath $tmp -Force -ErrorAction SilentlyContinue
    Invoke-WebRequest -UseBasicParsing -Uri $uri -OutFile $tmp
    Move-Item -LiteralPath $tmp -Destination $Destination -Force
}

function Acquire-Source {
    New-Item -ItemType Directory -Force -Path $SourceRoot, $ToolsRoot, $ReportsRoot | Out-Null
    $metadataFiles = @(
        'config.json', 'model.safetensors.index.json',
        'preprocessor_config.json', 'video_preprocessor_config.json',
        'tokenizer.json', 'tokenizer_config.json'
    )
    foreach ($name in $metadataFiles) {
        $destination = Assert-UnderRoot (Join-Path $SourceRoot $name) @($ModelRoot)
        if (-not (Test-Path -LiteralPath $destination)) { Get-HfFile $name $destination }
    }
    if ($DownloadWeights) {
        $index = Get-Content -Raw -LiteralPath (Join-Path $SourceRoot 'model.safetensors.index.json') | ConvertFrom-Json
        $shards = @($index.weight_map.PSObject.Properties.Value | Sort-Object -Unique)
        foreach ($name in $shards) {
            $destination = Assert-UnderRoot (Join-Path $SourceRoot $name) @($ModelRoot)
            if (-not (Test-Path -LiteralPath $destination)) { Get-HfFile $name $destination }
        }
    }
    $sourceRecord = [ordered]@{
        repository = $Repository
        revision = $ModelRevision
        transformers_revision = $TransformersRevision
        acquired_utc = [DateTime]::UtcNow.ToString('o')
        files = @()
    }
    Get-ChildItem -File -LiteralPath $SourceRoot | Sort-Object Name | ForEach-Object {
        $sourceRecord.files += [ordered]@{
            name = $_.Name
            bytes = $_.Length
            sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $_.FullName).Hash.ToLowerInvariant()
        }
    }
    $sourceRecord | ConvertTo-Json -Depth 6 | Set-Content -Encoding utf8 -LiteralPath (Join-Path $ReportsRoot 'source-files.json')
}

function Prepare-Converter {
    New-Item -ItemType Directory -Force -Path $ToolsRoot | Out-Null
    if (-not (Test-Path -LiteralPath $LlamaRoot)) {
        Invoke-Native 'git.exe' @('init', $LlamaRoot)
        Invoke-Native 'git.exe' @('-C', $LlamaRoot, 'remote', 'add', 'origin', 'https://github.com/ggml-org/llama.cpp.git')
        Invoke-Native 'git.exe' @('-C', $LlamaRoot, 'fetch', '--depth', '1', 'origin', $LlamaRevision)
        Invoke-Native 'git.exe' @('-C', $LlamaRoot, 'checkout', '--detach', 'FETCH_HEAD')
    }
    $actual = (& git.exe -C $LlamaRoot rev-parse HEAD).Trim()
    if ($LASTEXITCODE -ne 0 -or $actual -ne $LlamaRevision) {
        throw "llama.cpp revision mismatch: expected $LlamaRevision, got $actual"
    }
    Invoke-Native 'git.exe' @('-C', $LlamaRoot, 'reset', '--hard', $LlamaRevision)
    $patchPath = Assert-UnderRoot (Join-Path $ProjectSource 'tools\llama-qwen35-map-report.patch') @($Workspace)
    $reverseCheck = & git.exe -C $LlamaRoot apply --reverse --check --directory=conversion $patchPath 2>$null
    if ($LASTEXITCODE -ne 0) {
        Invoke-Native 'git.exe' @('-C', $LlamaRoot, 'apply', '--check', '--directory=conversion', $patchPath)
        Invoke-Native 'git.exe' @('-C', $LlamaRoot, 'apply', '--directory=conversion', $patchPath)
    }
}

function Audit-Source {
    New-Item -ItemType Directory -Force -Path $ReportsRoot | Out-Null
    $driver = Assert-UnderRoot (Join-Path $ProjectSource 'tools\qwen35-artifacts.py') @($Workspace)
    $index = Assert-UnderRoot (Join-Path $SourceRoot 'model.safetensors.index.json') @($ModelRoot)
    $output = Assert-UnderRoot (Join-Path $ReportsRoot 'source-inventory.json') @($Workspace)
    Invoke-Native 'python.exe' @($driver, '--source-index', $index, '--output', $output)
}

switch ($Action) {
    'Acquire' { Acquire-Source; Audit-Source }
    'AuditSource' { Audit-Source }
    'PrepareConverter' { Prepare-Converter }
}

Write-Host "qwen35_artifacts: $Action PASS"
