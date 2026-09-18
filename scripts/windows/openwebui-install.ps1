# openwebui-install.ps1 -- provision Open WebUI (source checkout) for yforge.
# Part of the standard deployment kit; idempotent, finished stages are skipped.
#
# Stages:
#   1. git clone https://github.com/open-webui/open-webui at -Version (default v0.9.6)
#   2. frontend: npm ci + npm run build      (skipped when build\index.html exists)
#   3. backend : uv venv + uv pip install -r backend\requirements.txt
#   4. .env    : auth ON (login + password, signup OFF), yforge OpenAI connection,
#                admin bootstrap credentials (WEBUI_ADMIN_*), HF_HOME, telemetry opt-out
#   5. task    : \open-webui scheduled task + firewall rule (openwebui-setup.ps1)
#   6. start   : run the task and wait for /health
#
# Usage (as Administrator):
#   powershell -NoLogo -NoProfile -ExecutionPolicy Bypass -File openwebui-install.ps1
#   ... -AdminEmail admin@localhost -AdminPassword 'S3cret...'   # omit -> keep/generate
#   ... -Rebuild                                                 # force npm/pip stages
#   ... -SkipStart                                               # prepare only
#
# Requires: git, node/npm (>=18), uv, Python 3.12 available to uv.

param(
    [string]$Root = 'D:\Projects\yttri-inference',
    [string]$Version = 'v0.9.6',
    [int]$Port = 8080,
    [string]$RepoUrl = 'https://github.com/open-webui/open-webui.git',
    [string]$ServerEnv = '',
    [string]$YforgeBaseUrl = 'http://127.0.0.1:18099/v1',
    [string]$YforgeKeyName = 'web-chat',
    [string]$AdminEmail = 'admin@localhost',
    [string]$AdminPassword = '',
    [string]$AdminName = 'Admin',
    [string]$TaskName = 'open-webui',
    [switch]$Rebuild,
    [switch]$SkipStart
)

$ErrorActionPreference = 'Stop'

$WebuiDir  = Join-Path $Root 'open-webui'
$WebuiEnv  = Join-Path $WebuiDir '.env'
$Python    = Join-Path $WebuiDir '.venv\Scripts\python.exe'
$BuildFile = Join-Path $WebuiDir 'build\index.html'
if (-not $ServerEnv) { $ServerEnv = Join-Path $Root 'qwen36-server\.env' }

function Write-Line([string]$Text) {
    Write-Output ("[{0}] {1}" -f (Get-Date).ToString('yyyy-MM-dd HH:mm:ss'), $Text)
}

function Get-EnvValue([string[]]$Lines, [string]$Name) {
    foreach ($line in $Lines) {
        if ($line -match "^\s*$([regex]::Escape($Name))\s*=") {
            return ($line -split '=', 2)[1].Trim().Trim("'").Trim('"')
        }
    }
    return $null
}

function Write-EnvFile([string]$Path, [string[]]$Existing, $Desired) {
    $written = New-Object System.Collections.Generic.List[string]
    foreach ($line in $Existing) {
        if ($line -notmatch '^\s*[A-Za-z_][A-Za-z0-9_]*\s*=') { $written.Add($line); continue }
        $name = ($line -split '=', 2)[0].Trim()
        if ($Desired.Contains($name)) {
            $written.Add("$name=$($Desired[$name])")
            [void]$Desired.Remove($name)
        }
    }
    foreach ($name in $Desired.Keys) { $written.Add("$name=$($Desired[$name])") }
    [System.IO.File]::WriteAllLines($Path, $written, (New-Object System.Text.UTF8Encoding($false)))
}

function Get-YforgeApiKey([string]$EnvFile, [string]$KeyName) {
    $keysLine = Get-Content -LiteralPath $EnvFile -Encoding UTF8 |
        Where-Object { $_ -match '^\s*API_KEYS\s*=' } | Select-Object -First 1
    if (-not $keysLine) { throw "API_KEYS= line not found in $EnvFile" }
    $json = ($keysLine -split '=', 2)[1].Trim()
    if ($json.StartsWith("'") -and $json.EndsWith("'")) { $json = $json.Substring(1, $json.Length - 2) }
    $keys = ConvertFrom-Json -InputObject $json
    if ($keys -isnot [System.Array]) { $keys = @($keys) }
    $key = ($keys | Where-Object { $_.name -eq $KeyName } | Select-Object -First 1).key
    if (-not $key) { $key = ($keys | Select-Object -First 1).key }
    if (-not $key) { throw "no API key found in $EnvFile" }
    return $key
}

# --- preflight -------------------------------------------------------------
foreach ($tool in @('git', 'npm', 'uv')) {
    if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) {
        throw "required tool not found in PATH: $tool"
    }
}
if (-not (Test-Path -LiteralPath $ServerEnv)) { throw "yforge env not found: $ServerEnv" }
Write-Line "install target: $WebuiDir (open-webui $Version)"

# --- stage 1: sources ------------------------------------------------------
if (-not (Test-Path -LiteralPath (Join-Path $WebuiDir '.git'))) {
    if (Test-Path -LiteralPath $WebuiDir) { throw "$WebuiDir exists but is not a git checkout" }
    Write-Line "stage 1: git clone $RepoUrl ($Version)"
    & git clone --depth 1 --branch $Version $RepoUrl $WebuiDir
    if ($LASTEXITCODE -ne 0) { throw 'git clone failed' }
} else {
    $current = (& git -C $WebuiDir describe --tags --always 2>$null)
    if ($Rebuild -or $current -ne $Version) {
        Write-Line "stage 1: checkout $Version (current: $current)"
        & git -C $WebuiDir fetch --depth 1 origin "refs/tags/$Version`:refs/tags/$Version"
        & git -C $WebuiDir checkout --force $Version
        if ($LASTEXITCODE -ne 0) { throw "git checkout $Version failed" }
    } else {
        Write-Line "stage 1: already at $Version"
    }
}

# --- stage 2: frontend -----------------------------------------------------
if ($Rebuild -or -not (Test-Path -LiteralPath $BuildFile)) {
    Write-Line "stage 2: npm ci"
    Push-Location $WebuiDir
    try {
        & npm ci --loglevel=error
        if ($LASTEXITCODE -ne 0) { throw 'npm ci failed' }
        Write-Line "stage 2: npm run build"
        & npm run build
        if ($LASTEXITCODE -ne 0) { throw 'npm run build failed' }
    } finally { Pop-Location }
} else {
    Write-Line "stage 2: frontend build present"
}

# --- stage 3: backend venv -------------------------------------------------
$needBackendInstall = $Rebuild.IsPresent -or (-not (Test-Path -LiteralPath $Python))
if (-not $needBackendInstall) {
    $probe = & $Python -c "import uvicorn, fastapi; print('probe-ok')" 2>$null
    if ($probe -notcontains 'probe-ok') { $needBackendInstall = $true }
}
if ($needBackendInstall) {
    Push-Location $WebuiDir
    try {
        if (-not (Test-Path -LiteralPath $Python)) {
            Write-Line "stage 3: uv venv --python 3.12"
            & uv venv --python 3.12 .venv
            if ($LASTEXITCODE -ne 0) { throw 'uv venv failed' }
        }
        Write-Line "stage 3: uv pip install -r backend\requirements.txt"
        & uv pip install --python $Python -r (Join-Path $WebuiDir 'backend\requirements.txt')
        if ($LASTEXITCODE -ne 0) { throw 'uv pip install failed' }
    } finally { Pop-Location }
} else {
    Write-Line "stage 3: backend venv ready"
}

# --- stage 4: .env ---------------------------------------------------------
$existing = @()
if (Test-Path -LiteralPath $WebuiEnv) { $existing = @(Get-Content -LiteralPath $WebuiEnv) }

$secret = Get-EnvValue $existing 'WEBUI_SECRET_KEY'
if (-not $secret) { $secret = [guid]::NewGuid().ToString('N') + [guid]::NewGuid().ToString('N') }

$generatedPassword = $false
$password = $AdminPassword
if (-not $password) { $password = Get-EnvValue $existing 'WEBUI_ADMIN_PASSWORD' }
if (-not $password) {
    $bytes = New-Object byte[] 15
    [System.Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($bytes)
    $password = [Convert]::ToBase64String($bytes).TrimEnd('=').Replace('+', '-').Replace('/', '_')
    $generatedPassword = $true
}
if (-not (Get-EnvValue $existing 'WEBUI_ADMIN_EMAIL')) { $AdminEmail = $AdminEmail } else { $AdminEmail = (Get-EnvValue $existing 'WEBUI_ADMIN_EMAIL') }
if (-not (Get-EnvValue $existing 'WEBUI_ADMIN_NAME')) { $AdminName = $AdminName } else { $AdminName = (Get-EnvValue $existing 'WEBUI_ADMIN_NAME') }

$desired = [ordered]@{
    'WEBUI_AUTH'                  = 'true'
    'ENABLE_SIGNUP'               = 'false'
    'WEBUI_SECRET_KEY'            = $secret
    'WEBUI_ADMIN_EMAIL'           = $AdminEmail
    'WEBUI_ADMIN_PASSWORD'        = $password
    'WEBUI_ADMIN_NAME'            = $AdminName
    'ENABLE_OPENAI_API'           = 'true'
    'OPENAI_API_BASE_URLS'        = $YforgeBaseUrl
    'OPENAI_API_KEYS'             = (Get-YforgeApiKey $ServerEnv $YforgeKeyName)
    'ANONYMIZED_TELEMETRY'        = 'false'
    'SCARF_NO_ANALYTICS'          = 'true'
    'DO_NOT_TRACK'                = 'true'
    'ENABLE_VERSION_UPDATE_CHECK' = 'false'
    'HF_HOME'                     = (Join-Path $WebuiDir '.cache\huggingface')
}
Write-EnvFile $WebuiEnv $existing $desired
Write-Line "stage 4: $WebuiEnv written (auth on, signup off, admin $AdminEmail)"

# --- stage 5: task + firewall ---------------------------------------------
$setup = Join-Path $Root 'openwebui-setup.ps1'
if (Test-Path -LiteralPath $setup) {
    Write-Line "stage 5: $setup"
    & powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File $setup -Root $Root -Port $Port -TaskName $TaskName
    if ($LASTEXITCODE -ne 0) { throw 'openwebui-setup.ps1 failed' }
} else {
    Write-Line "WARN: $setup not found - scheduled task and firewall rule were not touched"
}

# --- stage 6: start + health ----------------------------------------------
$healthy = $false
if (-not $SkipStart) {
    Write-Line "stage 6: restart task $TaskName"
    & schtasks /end /tn $TaskName 2>$null | Out-Null
    $listener = Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue |
        Select-Object -First 1 -ExpandProperty OwningProcess
    if ($listener) {
        Stop-Process -Id $listener -Force -ErrorAction SilentlyContinue
        Start-Sleep -Seconds 3
    }
    & schtasks /run /tn $TaskName | Out-Null
    for ($i = 1; $i -le 60; $i++) {
        Start-Sleep -Seconds 2
        try {
            $config = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/api/config" -TimeoutSec 5
            if ($config.status) { $healthy = $true; break }
        } catch { }
    }
    if ($healthy) {
        Write-Line "stage 6: Open WebUI answered, auth=$($config.features.auth), version=$($config.version)"
    } else {
        Write-Line "WARN: no answer on http://127.0.0.1:$Port/api/config after 120 s - check logs\open-webui.log"
    }
}

# --- summary ---------------------------------------------------------------
$lan = ''
try {
    # prefer the interface that owns the default route (ICS/WSL adapters come later)
    $route = Get-NetRoute -DestinationPrefix '0.0.0.0/0' -ErrorAction SilentlyContinue |
        Sort-Object RouteMetric, InterfaceMetric | Select-Object -First 1
    if ($route) {
        $lan = (Get-NetIPAddress -AddressFamily IPv4 -InterfaceIndex $route.InterfaceIndex -ErrorAction SilentlyContinue |
            Select-Object -First 1 -ExpandProperty IPAddress)
    }
    if (-not $lan) {
        $lan = (Get-NetIPAddress -AddressFamily IPv4 |
            Where-Object { $_.IPAddress -notlike '127.*' -and $_.IPAddress -notlike '169.254.*' -and $_.IPAddress -notlike '192.168.137.*' } |
            Select-Object -First 1 -ExpandProperty IPAddress)
    }
} catch { }

Write-Output ''
Write-Output "Open WebUI:  http://$lan`:$Port/   (login required)"
Write-Output "Login:       $AdminEmail"
if ($generatedPassword) {
    Write-Output "Password:    $password    (generated now, stored in $WebuiEnv)"
} else {
    Write-Output "Password:    from $WebuiEnv (WEBUI_ADMIN_PASSWORD)"
}
Write-Output "Env file:    $WebuiEnv"
Write-Output "Log:         $(Join-Path $Root 'logs\open-webui.log')"
if (-not $healthy -and -not $SkipStart) { exit 1 }
exit 0
