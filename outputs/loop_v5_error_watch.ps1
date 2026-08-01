$ErrorActionPreference = "Continue"
$base = "C:\Users\Azooo\arabic-reasoning-rlvr-sota\RL-ar-"
$seen = @{}
while ($true) {
  Start-Sleep -Seconds 15
  $hits = @()
  $fail = Join-Path $base "data_1\outputs\LOCAL_SFT_DATAGEN_FAILED.json"
  if (Test-Path $fail) {
    $key = "failmarker"
    if (-not $seen.ContainsKey($key)) {
      $seen[$key] = 1
      $hits += "LOCAL_SFT_DATAGEN_FAILED"
    }
  }
  $logs = @(
    (Join-Path $base "outputs\local_sft_topup_local_v4_20260801_073920.log"),
    (Join-Path $base "outputs\v5_finish_continue.log"),
    (Join-Path $base "outputs\v5_e2e_heartbeat.log")
  )
  foreach ($log in $logs) {
    if (-not (Test-Path $log)) { continue }
    $tail = Get-Content $log -Tail 50 -ErrorAction SilentlyContinue
    foreach ($line in $tail) {
      if ($line -match "(?i)\b(FAILED|ERROR|Traceback|Exception|Killed|OOM|No space left)\b") {
        $key = ($log + "|" + $line)
        if ($key.Length -gt 200) { $key = $key.Substring(0, 200) }
        if (-not $seen.ContainsKey($key)) {
          $seen[$key] = 1
          $hits += $line
        }
      }
    }
  }
  if ($hits.Count -gt 0) {
    $msg = ($hits | Select-Object -First 5) -join " || "
    $obj = @{
      prompt = "IMMEDIATE ERROR HEARTBEAT. Investigate and fix: $msg. Continue V5 e2e: top-up finish, 4000 RLVR only (no MCQ), Vast SFT+pass8+GRPO. SSH -p 49198 root@8.243.214.78 passphrase aziz."
    }
    Write-Output ("AGENT_LOOP_WAKE_v5_error " + ($obj | ConvertTo-Json -Compress))
  }
}
