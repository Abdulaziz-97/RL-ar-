# Local SFT teacher datagen (DeepSeek Pro). On success writes DONE marker for the monitor loop.
# Usage: powershell -File scripts/local_sft_datagen_v4.ps1
# Resume: $env:RUN_ID='local_v4_...'; $env:SKIP_CANARY='1' (keeps multi_trace.partial.jsonl)
$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Data1 = Join-Path $RepoRoot "data_1"
$RunId = if ($env:RUN_ID) { $env:RUN_ID } else { "local_v4_" + (Get-Date -Format "yyyyMMdd_HHmmss") }
$WorkRoot = Join-Path $Data1 "outputs\v4_regen\$RunId"
$LogDir = Join-Path $RepoRoot "outputs"
$LogFile = Join-Path $LogDir "local_sft_datagen_$RunId.log"
$DoneMarker = Join-Path $WorkRoot "LOCAL_SFT_DATAGEN_DONE.json"
$FailMarker = Join-Path $WorkRoot "LOCAL_SFT_DATAGEN_FAILED.json"

New-Item -ItemType Directory -Force -Path $WorkRoot, $LogDir, (Join-Path $RepoRoot "data") | Out-Null
if (Test-Path $DoneMarker) { Remove-Item -Force $DoneMarker }
if (Test-Path $FailMarker) { Remove-Item -Force $FailMarker }
Remove-Item -Force (Join-Path $Data1 "outputs\LOCAL_SFT_DATAGEN_FAILED.json") -ErrorAction SilentlyContinue

function Write-Log([string]$msg) {
    $line = "[{0}] {1}" -f (Get-Date -Format "o"), $msg
    Write-Host $line
    Add-Content -Path $LogFile -Value $line -Encoding UTF8
}

function Invoke-LoggedPython {
    param([Parameter(Mandatory = $true)][string[]]$ArgumentList)
    # Native stderr (dspy WARNING) must NOT become terminating errors under Stop.
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & python @ArgumentList 2>&1 | ForEach-Object {
        $s = "$_"
        Add-Content -Path $LogFile -Value $s -Encoding UTF8
        Write-Host $s
    }
    $code = $LASTEXITCODE
    $ErrorActionPreference = $prev
    if ($null -eq $code) { $code = 0 }
    if ($code -ne 0) {
        throw ("python exit={0} args={1}" -f $code, ($ArgumentList -join " "))
    }
}

if (-not $env:DEEPSEEK_API_KEY) {
    throw "DEEPSEEK_API_KEY is required for local SFT datagen"
}

$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONPATH = "$Data1;$Data1\src;$Data1\vendor"
$env:USE_OPENROUTER = if ($env:USE_OPENROUTER) { $env:USE_OPENROUTER } else { "0" }
$env:TEACHER_MODEL = if ($env:TEACHER_MODEL) { $env:TEACHER_MODEL } else { "deepseek-v4-pro" }
$env:TEACHER_MAX_TOKENS = if ($env:TEACHER_MAX_TOKENS) { $env:TEACHER_MAX_TOKENS } else { "8192" }
$workers = if ($env:TEACHER_WORKERS) { [int]$env:TEACHER_WORKERS } else { 48 }
$budget = if ($env:SFT_BUDGET_USD) { $env:SFT_BUDGET_USD } else { "200" }
$canaryN = if ($env:DATAGEN_CANARY_N) { [int]$env:DATAGEN_CANARY_N } else { 20 }
$canaryBudget = if ($env:DATAGEN_CANARY_BUDGET_USD) { $env:DATAGEN_CANARY_BUDGET_USD } else { "5" }
$env:DATAGEN_RESUME_MULTI_TRACE = if ($env:DATAGEN_RESUME_MULTI_TRACE) { $env:DATAGEN_RESUME_MULTI_TRACE } else { "1" }
$env:TEACHER_RETRIES = if ($env:TEACHER_RETRIES) { $env:TEACHER_RETRIES } else { "4" }
$skipCanary = ($env:SKIP_CANARY -eq "1")
$fresh = ($env:DATAGEN_FRESH -eq "1")

$SftCfg = Join-Path $Data1 "configs\full_sft_6500.yaml"
$runPipeline = Join-Path $Data1 "scripts\run_pipeline.py"
$selectScript = Join-Path $Data1 "scripts\select_sft_v4_release.py"

Write-Log "START run_id=$RunId workers=$workers max_tokens=$($env:TEACHER_MAX_TOKENS) skip_canary=$skipCanary fresh=$fresh log=$LogFile"
Write-Log "WORK=$WorkRoot"

try {
    if (-not $skipCanary) {
        $CanaryWork = Join-Path $WorkRoot "canary_sft"
        $CanaryCfg = Join-Path $WorkRoot "canary_sft_$canaryN.yaml"
        New-Item -ItemType Directory -Force -Path $CanaryWork | Out-Null
        $prev = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        & python -c @"
from pathlib import Path
import yaml
cfg = yaml.safe_load(Path(r'$SftCfg').read_text(encoding='utf-8'))
cfg['n_families'] = $canaryN
cfg['work_dir'] = r'$CanaryWork'
Path(r'$CanaryCfg').write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding='utf-8')
print(r'$CanaryCfg')
"@
        $ErrorActionPreference = $prev
        Write-Log "CANARY start n=$canaryN"
        Invoke-LoggedPython -ArgumentList @(
            $runPipeline, "--mode", "live", "--config", $CanaryCfg,
            "--work-dir", $CanaryWork, "--track", "sft",
            "--model", $env:TEACHER_MODEL, "--workers", "$workers",
            "--budget-usd", "$canaryBudget", "--no-resume"
        )
        $canaryOut = Join-Path $CanaryWork "release_corpora\sft_train.jsonl"
        if (-not (Test-Path $canaryOut) -or ((Get-Content $canaryOut | Measure-Object -Line).Lines -lt 1)) {
            throw "Canary produced 0 SFT rows"
        }
        Write-Log "CANARY OK rows=$((Get-Content $canaryOut | Measure-Object -Line).Lines)"
    }
    else {
        Write-Log "SKIP_CANARY=1"
    }

    $SftWork = Join-Path $WorkRoot "sft_candidates"
    Write-Log "FULL SFT start work=$SftWork"
    $fullArgs = @(
        $runPipeline, "--mode", "live", "--config", $SftCfg,
        "--work-dir", $SftWork, "--track", "sft",
        "--model", $env:TEACHER_MODEL, "--workers", "$workers",
        "--budget-usd", "$budget"
    )
    if ($fresh) { $fullArgs += "--no-resume" }
    Invoke-LoggedPython -ArgumentList $fullArgs

    $cand = Join-Path $SftWork "release_corpora\sft_train.jsonl"
    $selected = Join-Path $WorkRoot "sft_selected_4000.jsonl"
    Write-Log "SELECT 4k from $cand"
    Invoke-LoggedPython -ArgumentList @(
        $selectScript, "--candidates", $cand, "--out", $selected, "--allow-missing-decontam"
    )

    $dest = Join-Path $RepoRoot "data\arabic_reasoning_coldstart_v5.jsonl"
    Copy-Item -Force $selected $dest
    $n = (Get-Content $dest | Measure-Object -Line).Lines
    Write-Log "WROTE $dest rows=$n"

    $payload = @{
        status = "done"
        run_id = $RunId
        coldstart = $dest
        selected = $selected
        rows = $n
        log = $LogFile
        finished_at = (Get-Date).ToString("o")
    } | ConvertTo-Json
    Set-Content -Path $DoneMarker -Value $payload -Encoding UTF8
    Set-Content -Path (Join-Path $Data1 "outputs\LOCAL_SFT_DATAGEN_DONE.json") -Value $payload -Encoding UTF8
    Write-Log "DONE marker=$DoneMarker"
    exit 0
}
catch {
    $err = $_.Exception.Message
    Write-Log "FAILED $err"
    $payload = @{ status = "failed"; run_id = $RunId; error = $err; log = $LogFile; finished_at = (Get-Date).ToString("o") } | ConvertTo-Json
    Set-Content -Path $FailMarker -Value $payload -Encoding UTF8
    Set-Content -Path (Join-Path $Data1 "outputs\LOCAL_SFT_DATAGEN_FAILED.json") -Value $payload -Encoding UTF8
    exit 1
}
