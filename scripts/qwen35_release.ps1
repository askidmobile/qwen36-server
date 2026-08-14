[CmdletBinding(SupportsShouldProcess)]
param(
    [ValidateSet('Publish', 'Rollback', 'Validate')]
    [string]$Action = 'Validate',
    [string]$ModelRoot = 'D:\Models\yttri\qwen3.5-4b',
    [string]$Release,
    [string]$PreviousRelease
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$ForbiddenRoot = 'D:\Projects\yttri-build'

function Normalize([string]$Path) {
    [IO.Path]::GetFullPath([Environment]::ExpandEnvironmentVariables($Path)).TrimEnd('\')
}

function Assert-Under([string]$Path, [string]$Root) {
    $candidate = Normalize $Path
    $allowed = Normalize $Root
    $forbidden = Normalize $ForbiddenRoot
    if ($candidate.Equals($forbidden, [StringComparison]::OrdinalIgnoreCase) -or
        $candidate.StartsWith($forbidden + '\', [StringComparison]::OrdinalIgnoreCase)) {
        throw "Forbidden workspace path: $candidate"
    }
    if (-not ($candidate.Equals($allowed, [StringComparison]::OrdinalIgnoreCase) -or
        $candidate.StartsWith($allowed + '\', [StringComparison]::OrdinalIgnoreCase))) {
        throw "Path is outside allowed root ${allowed}: $candidate"
    }
    $candidate
}

function Assert-Relative([string]$Path, [string]$Label) {
    if ([IO.Path]::IsPathRooted($Path) -or $Path -split '[\\/]' -contains '..') {
        throw "$Label must be a contained relative path: $Path"
    }
}

$ModelRoot = Assert-Under $ModelRoot 'D:\Models\yttri'
$ReleasesRoot = Assert-Under (Join-Path $ModelRoot 'releases') $ModelRoot
$CurrentPath = Assert-Under (Join-Path $ModelRoot 'current') $ModelRoot

function Test-HashedFile([string]$ReleasePath, $Entry, [string]$Label) {
    if ($null -eq $Entry) { throw "Missing $Label entry" }
    Assert-Relative $Entry.path "$Label path"
    $path = Assert-Under (Join-Path $ReleasePath $Entry.path) $ReleasePath
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Missing ${Label}: $path" }
    $info = Get-Item -LiteralPath $path
    if ($info.Length -ne [uint64]$Entry.bytes) { throw "$Label size mismatch: $path" }
    $hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $path).Hash
    if ($hash -ine $Entry.sha256) { throw "$Label SHA-256 mismatch: $path" }
}

function Get-Profile([string]$ReleasePath) {
    $releasePath = Assert-Under $ReleasePath $ReleasesRoot
    $manifestPath = Join-Path $releasePath 'profile.json'
    if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) { throw "Missing profile.json: $releasePath" }
    $profile = Get-Content -Raw -LiteralPath $manifestPath | ConvertFrom-Json
    if ($profile.schema_version -ne 1) { throw "Unsupported profile schema: $($profile.schema_version)" }
    if ($profile.server_abi -ne 'qwen36-profile-v1') { throw "Incompatible server ABI: $($profile.server_abi)" }
    if (-not $profile.gates -or @($profile.gates | Where-Object { -not $_.passed }).Count -ne 0) {
        throw 'Every mandatory gate must exist and pass'
    }
    foreach ($artifact in @($profile.artifacts.text, $profile.artifacts.vision, $profile.artifacts.mtp)) {
        if ($null -ne $artifact) { Test-HashedFile $releasePath $artifact 'artifact' }
    }
    Test-HashedFile $releasePath $profile.processor.image_config 'processor.image_config'
    Test-HashedFile $releasePath $profile.processor.video_config 'processor.video_config'
    Test-HashedFile $releasePath $profile.runtime.server 'runtime.server'
    Test-HashedFile $releasePath $profile.runtime.media_helper 'runtime.media_helper'
    Test-HashedFile $releasePath $profile.runtime.ffmpeg 'runtime.ffmpeg'
    Test-HashedFile $releasePath $profile.runtime.ffprobe 'runtime.ffprobe'
    foreach ($license in @($profile.runtime.licenses)) { Test-HashedFile $releasePath $license 'runtime.license' }
    foreach ($gate in @($profile.gates)) { Test-HashedFile $releasePath $gate.result 'gate.result' }
    [ordered]@{ Path=$releasePath; Manifest=$manifestPath; Profile=$profile }
}

function Write-AtomicCurrent([string]$ReleasePath, [string]$ManifestPath) {
    $relative = [IO.Path]::GetRelativePath($ModelRoot, $ReleasePath)
    Assert-Relative $relative 'current.release'
    $pointer = [ordered]@{
        schema_version = 1
        release = $relative.Replace('\', '/')
        manifest_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $ManifestPath).Hash.ToLowerInvariant()
    } | ConvertTo-Json -Compress
    $temp = "$CurrentPath.$PID.tmp"
    try {
        $bytes = [Text.UTF8Encoding]::new($false).GetBytes($pointer + "`n")
        $stream = [IO.File]::Open($temp, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::None)
        try {
            $stream.Write($bytes, 0, $bytes.Length)
            $stream.Flush($true)
        } finally {
            $stream.Dispose()
        }
        if (-not ('Qwen35AtomicFile' -as [type])) {
            Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public static class Qwen35AtomicFile {
    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    public static extern bool MoveFileExW(string existing, string replacement, uint flags);
}
'@
        }
        if (-not [Qwen35AtomicFile]::MoveFileExW($temp, $CurrentPath, 0x1 -bor 0x8)) {
            throw "MoveFileExW failed: $([Runtime.InteropServices.Marshal]::GetLastWin32Error())"
        }
    } finally {
        Remove-Item -LiteralPath $temp -Force -ErrorAction SilentlyContinue
    }
}

switch ($Action) {
    'Validate' {
        if (-not $Release) { throw '-Release is required' }
        $validated = Get-Profile $Release
        Write-Output "VALID $($validated.Path)"
    }
    'Publish' {
        if (-not $Release) { throw '-Release is required' }
        $validated = Get-Profile $Release
        if ($PSCmdlet.ShouldProcess($CurrentPath, "publish $($validated.Path)")) {
            Write-AtomicCurrent $validated.Path $validated.Manifest
            Write-Output "PUBLISHED $($validated.Path)"
        }
    }
    'Rollback' {
        if (-not $PreviousRelease) { throw '-PreviousRelease is required' }
        $validated = Get-Profile $PreviousRelease
        if ($PSCmdlet.ShouldProcess($CurrentPath, "rollback to $($validated.Path)")) {
            Write-AtomicCurrent $validated.Path $validated.Manifest
            Write-Output "ROLLED_BACK $($validated.Path)"
        }
    }
}
