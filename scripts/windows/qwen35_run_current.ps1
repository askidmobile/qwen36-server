[CmdletBinding()]
param(
    [string]$Current = 'D:\Models\yttri\qwen3.5-4b\current',
    [string]$EnvFile
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$currentPath = [IO.Path]::GetFullPath($Current)
$root = Split-Path -Parent $currentPath
$pointer = Get-Content -Raw -LiteralPath $currentPath | ConvertFrom-Json
if ($pointer.schema_version -ne 1) { throw "Unsupported current schema: $($pointer.schema_version)" }
if ([IO.Path]::IsPathRooted($pointer.release) -or $pointer.release -split '[\\/]' -contains '..') {
    throw "Unsafe current.release: $($pointer.release)"
}
$release = [IO.Path]::GetFullPath((Join-Path $root $pointer.release))
if (-not $release.StartsWith($root + '\', [StringComparison]::OrdinalIgnoreCase)) {
    throw "current.release escapes model root: $release"
}
$manifest = Join-Path $release 'profile.json'
$hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $manifest).Hash
if ($hash -ine $pointer.manifest_sha256) { throw 'current manifest SHA-256 mismatch' }
$profile = Get-Content -Raw -LiteralPath $manifest | ConvertFrom-Json
$server = [IO.Path]::GetFullPath((Join-Path $release $profile.runtime.server.path))
if (-not $server.StartsWith($release + '\', [StringComparison]::OrdinalIgnoreCase)) {
    throw 'runtime.server escapes release directory'
}
$serverHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $server).Hash
if ($serverHash -ine $profile.runtime.server.sha256) { throw 'runtime.server SHA-256 mismatch' }
$env:PROFILE = $currentPath
if ($EnvFile) { $env:ENV_FILE = [IO.Path]::GetFullPath($EnvFile) }
& $server
exit $LASTEXITCODE
