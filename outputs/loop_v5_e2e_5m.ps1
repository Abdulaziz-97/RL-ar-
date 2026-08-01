$ErrorActionPreference = "Continue"
$payloadPath = Join-Path $PSScriptRoot "loop_v5_e2e_prompt.json"
while ($true) {
  Start-Sleep -Seconds 300
  $payload = Get-Content -Raw -Path $payloadPath -Encoding UTF8
  Write-Output ("AGENT_LOOP_TICK_v5_e2e " + $payload.Trim())
}
