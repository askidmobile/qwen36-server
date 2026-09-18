# openwebui-run.ps1 -- launch Open WebUI (source checkout) for the local yforge server.
#
# What it does:
#   1. reads the yforge API key (entry named "web-chat") from qwen36-server\.env;
#   2. waits until yforge answers /v1/models and takes the model ids it serves;
#   3. writes/updates open-webui\.env: auth off, OpenAI connection to yforge,
#      DEFAULT_MODELS = the model actually loaded right now;
#   4. starts uvicorn (Open WebUI backend) on 0.0.0.0:8080 in the foreground.
#
# Source checkout: D:\Projects\yttri-inference\open-webui (tag v0.9.6).
# Scheduled task:  \open-webui  ->  openwebui-run.bat

param(
    [string]$Root = 'D:\Projects\yttri-inference',
    [int]$Port = 8080
)

$ErrorActionPreference = 'Stop'

$WebuiDir  = Join-Path $Root 'open-webui'
$ServerEnv = Join-Path $Root 'qwen36-server\.env'
$WebuiEnv  = Join-Path $WebuiDir '.env'
$Python    = Join-Path $WebuiDir '.venv\Scripts\python.exe'
$Backend   = Join-Path $WebuiDir 'backend'
$BaseUrl   = 'http://127.0.0.1:18099/v1'

function Write-Line([string]$Text) {
    Write-Output ("[{0}] {1}" -f (Get-Date).ToString('yyyy-MM-dd HH:mm:ss'), $Text)
}

if (Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue) {
    Write-Line "port $Port is already listening - Open WebUI seems to be running"
    exit 0
}
foreach ($path in @($ServerEnv, $Python, $Backend)) {
    if (-not (Test-Path -LiteralPath $path)) { throw "not found: $path" }
}

# --- yforge API key --------------------------------------------------------
$keysLine = Get-Content -LiteralPath $ServerEnv -Encoding UTF8 |
    Where-Object { $_ -match '^\s*API_KEYS\s*=' } | Select-Object -First 1
if (-not $keysLine) { throw "API_KEYS= line not found in $ServerEnv" }
$keysJson = ($keysLine -split '=', 2)[1].Trim()
if ($keysJson.StartsWith("'") -and $keysJson.EndsWith("'")) {
    $keysJson = $keysJson.Substring(1, $keysJson.Length - 2)
}
$keys = ConvertFrom-Json -InputObject $keysJson
# PS 5.1: ConvertFrom-Json keeps the array as one object, so normalise it.
if ($keys -isnot [System.Array]) { $keys = @($keys) }
$apiKey = ($keys | Where-Object { $_.name -eq 'web-chat' } | Select-Object -First 1).key
if (-not $apiKey) { $apiKey = ($keys | Select-Object -First 1).key }
if (-not $apiKey) { throw "no API key found in $ServerEnv" }

# --- wait for yforge, collect model ids ------------------------------------
$modelIds = @()
$headers = @{ Authorization = "Bearer $apiKey" }
for ($attempt = 1; $attempt -le 120; $attempt++) {
    try {
        $reply = Invoke-RestMethod -Uri "$BaseUrl/models" -Headers $headers -TimeoutSec 10
        $modelIds = @($reply.data | ForEach-Object { $_.id } | Where-Object { $_ })
        if ($modelIds.Count -gt 0) { break }
    } catch {
        $modelIds = @()
    }
    if ($attempt -eq 1) { Write-Line "waiting for yforge at $BaseUrl ..." }
    Start-Sleep -Seconds 5
}
$modelsText = 'unchanged'
if ($modelIds.Count -gt 0) {
    $modelsText = $modelIds -join ','
    Write-Line ("yforge is up, model ids: " + $modelsText)
} else {
    Write-Line "WARN: yforge did not answer within 10 min, DEFAULT_MODELS is left as is"
}

# --- open-webui .env -------------------------------------------------------
$desired = [ordered]@{
    'WEBUI_AUTH'                  = 'false'
    'ENABLE_OPENAI_API'           = 'true'
    'OPENAI_API_BASE_URLS'        = $BaseUrl
    'OPENAI_API_KEYS'             = $apiKey
    'ANONYMIZED_TELEMETRY'        = 'false'
    'SCARF_NO_ANALYTICS'          = 'true'
    'DO_NOT_TRACK'                = 'true'
    'ENABLE_VERSION_UPDATE_CHECK' = 'false'
    'HF_HOME'                     = (Join-Path $WebuiDir '.cache\huggingface')
}
if ($modelIds.Count -gt 0) { $desired['DEFAULT_MODELS'] = $modelsText }

$existing = @()
if (Test-Path -LiteralPath $WebuiEnv) { $existing = @(Get-Content -LiteralPath $WebuiEnv) }

$secret = $null
foreach ($line in $existing) {
    if ($line -match '^\s*WEBUI_SECRET_KEY\s*=') {
        $secret = ($line -split '=', 2)[1].Trim().Trim("'").Trim('"')
    }
}
if (-not $secret) { $secret = [guid]::NewGuid().ToString('N') + [guid]::NewGuid().ToString('N') }
$desired['WEBUI_SECRET_KEY'] = $secret

$written = New-Object System.Collections.Generic.List[string]
foreach ($line in $existing) {
    if ($line -notmatch '^\s*[A-Za-z_][A-Za-z0-9_]*\s*=') { $written.Add($line); continue }
    $name = ($line -split '=', 2)[0].Trim()
    if ($desired.Contains($name)) {
        $written.Add("$name=$($desired[$name])")
        [void]$desired.Remove($name)
    }
}
foreach ($name in $desired.Keys) { $written.Add("$name=$($desired[$name])") }
[System.IO.File]::WriteAllLines($WebuiEnv, $written, (New-Object System.Text.UTF8Encoding($false)))
Write-Line "open-webui .env updated: $WebuiEnv"

# --- start -----------------------------------------------------------------
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
Write-Line "starting Open WebUI on http://0.0.0.0:$Port/ (auth disabled, default models: $modelsText)"
Set-Location -LiteralPath $Backend
& $Python -m uvicorn open_webui.main:app --host '0.0.0.0' --port $Port --workers 1
exit $LASTEXITCODE
