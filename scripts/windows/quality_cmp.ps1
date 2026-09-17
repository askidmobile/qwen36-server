$ErrorActionPreference = 'Continue'
$root = 'D:\Projects\yttri-inference'
$prompts = @('prompt_omega.txt','prompt_beta.txt','prompt_gamma.txt')
foreach ($cfg in @(@{n='f16'; ovr='KV_POOL_Q8=0;CTX=128000;CONTEXT_LIMIT=128000;GRAPH_WINDOW=131072;PGRAPH=on;PGRAPH_MIN_T=1000000;PREFILL_CHUNK=8192;SAMPLING_LOCK=0'}, @{n='q8'; ovr='KV_POOL_Q8=1;CTX=128000;CONTEXT_LIMIT=128000;GRAPH_WINDOW=131072;PGRAPH=on;PGRAPH_MIN_T=1000000;PREFILL_CHUNK=8192;SAMPLING_LOCK=0'})) {
  Set-Content -Path "$root\bench-overrides.env" -Value ($cfg.ovr -split ';') -Encoding ascii
  & powershell -ExecutionPolicy Bypass -File "$root\scripts\restart_only.ps1" | Select-Object -Last 1 | Out-Null
  foreach ($p in $prompts) {
    $out = & powershell -ExecutionPolicy Bypass -File "$root\scripts\text_dump.ps1" -PromptFile "$root\$p" -Tag "$($cfg.n)-$p" -MaxTokens 64 2>&1
    Write-Output ($out | Select-Object -Last 1)
  }
}
Remove-Item "$root\bench-overrides.env" -ErrorAction SilentlyContinue
