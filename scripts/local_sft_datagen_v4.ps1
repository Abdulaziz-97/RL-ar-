# Local FULL teacher datagen: SFT (+ Flash Reverse-QA+reteach) then RLVR candidates (+ Flash Reverse-QA).
# Writes coldstart_v5 + rlvr_candidates_v5. Does NOT run pass@8 (needs GPU + SFT ckpt on Vast).
# Vast: SFT train → pass@8 → curate/promote rlvr_v5 → GRPO.
# Usage: powershell -File scripts/local_sft_datagen_v4.ps1
# Resume SFT: $env:RUN_ID='local_v4_...'; $env:SKIP_CANARY='1'
# RLVR-only after SFT ready: $env:SKIP_SFT='1' $env:RUN_ID='...'
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
$env:REVERSE_QA = if ($env:REVERSE_QA) { $env:REVERSE_QA } else { "1" }
$env:REVERSE_QA_MODE = if ($env:REVERSE_QA_MODE) { $env:REVERSE_QA_MODE } else { "full" }
$env:REVERSE_QA_RESOLVE = if ($env:REVERSE_QA_RESOLVE) { $env:REVERSE_QA_RESOLVE } else { "1" }
$env:REVERSE_QA_MODEL = if ($env:REVERSE_QA_MODEL) { $env:REVERSE_QA_MODEL } else { "deepseek-v4-flash" }
# Fast path: Flash for Reverse-QA gen+resolve AND CoT reteach (override with REVERSE_QA_RETEACH_MODEL=deepseek-v4-pro if needed)
$env:REVERSE_QA_RETEACH_MODEL = if ($env:REVERSE_QA_RETEACH_MODEL) { $env:REVERSE_QA_RETEACH_MODEL } else { $env:REVERSE_QA_MODEL }
$skipCanary = ($env:SKIP_CANARY -eq "1")
$fresh = ($env:DATAGEN_FRESH -eq "1")

$SftCfg = Join-Path $Data1 "configs\full_sft_6500.yaml"
$RlvrCfg = Join-Path $Data1 "configs\full_rlvr_8000.yaml"
$runPipeline = Join-Path $Data1 "scripts\run_pipeline.py"
$selectScript = Join-Path $Data1 "scripts\select_sft_v4_release.py"
$rlvrBudget = if ($env:RLVR_BUDGET_USD) { $env:RLVR_BUDGET_USD } else { "80" }
$skipSft = ($env:SKIP_SFT -eq "1")

Write-Log "START run_id=$RunId workers=$workers max_tokens=$($env:TEACHER_MAX_TOKENS) reverse_qa=$($env:REVERSE_QA) skip_canary=$skipCanary skip_sft=$skipSft fresh=$fresh log=$LogFile"
Write-Log "WORK=$WorkRoot"

try {
    $selected = Join-Path $WorkRoot "sft_selected_4000.jsonl"
    $dest = Join-Path $RepoRoot "data\arabic_reasoning_coldstart_v5.jsonl"

    if ($skipSft) {
        Write-Log "SKIP_SFT=1 — reuse existing coldstart/selected"
        if (-not (Test-Path $dest)) { throw "SKIP_SFT=1 but missing $dest" }
        if (-not (Test-Path $selected)) {
            Copy-Item -Force $dest $selected
        }
    }
    elseif (-not $skipCanary) {
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

    if (-not $skipSft) {
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
        Write-Log "SELECT 4k from $cand"
        Invoke-LoggedPython -ArgumentList @(
            $selectScript, "--candidates", $cand, "--out", $selected, "--allow-missing-decontam"
        )

        # Full answer-first Reverse-QA on the selected 4k, then re-teach CoT for new prompts.
        $selectedRqa = Join-Path $WorkRoot "sft_selected_4000_reverse_qa.jsonl"
        Write-Log "FULL REVERSE-QA + reteach on selected 4k (Flash gen+resolve+reteach)"
        $env:REVERSE_QA = "1"
        $env:REVERSE_QA_MODE = "full"
        $env:REVERSE_QA_RESOLVE = "1"
        Invoke-LoggedPython -ArgumentList @(
            (Join-Path $Data1 "scripts\apply_full_reverse_qa_sft.py"),
            "--in", $selected,
            "--out", $selectedRqa,
            "--reteach",
            "--workers", "$workers"
        )
        if (Test-Path $selectedRqa) { $selected = $selectedRqa }

        Copy-Item -Force $selected $dest
        $n = (Get-Content $dest | Measure-Object -Line).Lines
        Write-Log "WROTE $dest rows=$n"
    }

    # ---- RLVR candidates (local API gen; pass@8 stays on Vast GPU after SFT) ----
    $RlvrWork = Join-Path $WorkRoot "rlvr_candidates"
    $RlvrRuntime = Join-Path $WorkRoot "full_rlvr_8000.runtime.yaml"
    Write-Log "FULL RLVR candidates start work=$RlvrWork"
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & python -c @"
from pathlib import Path
import yaml
cfg = yaml.safe_load(Path(r'$RlvrCfg').read_text(encoding='utf-8'))
cfg['decontam_reference_paths'] = [r'$selected']
cfg['work_dir'] = r'$RlvrWork'
Path(r'$RlvrRuntime').write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding='utf-8')
print(r'$RlvrRuntime')
"@
    $ErrorActionPreference = $prev
    # Programmatic RLVR first (REVERSE_QA off in-pipeline); post-pass Flash Reverse-QA below.
    $env:REVERSE_QA = "0"
    $rlvrArgs = @(
        $runPipeline, "--mode", "live", "--config", $RlvrRuntime,
        "--work-dir", $RlvrWork, "--track", "rlvr",
        "--model", $env:TEACHER_MODEL, "--workers", "1",
        "--budget-usd", "$rlvrBudget"
    )
    if ($fresh) { $rlvrArgs += "--no-resume" }
    Invoke-LoggedPython -ArgumentList $rlvrArgs

    $rlvrCand = Join-Path $RlvrWork "release_corpora\rlvr_train.jsonl"
    if (-not (Test-Path $rlvrCand)) { throw "RLVR release missing: $rlvrCand" }
    $rlvrRqa = Join-Path $WorkRoot "rlvr_candidates_reverse_qa.jsonl"
    Write-Log "FULL REVERSE-QA on RLVR candidates (Flash gen+resolve, no reteach)"
    $env:REVERSE_QA = "1"
    $env:REVERSE_QA_MODE = "full"
    $env:REVERSE_QA_RESOLVE = "1"
    Invoke-LoggedPython -ArgumentList @(
        (Join-Path $Data1 "scripts\apply_full_reverse_qa_sft.py"),
        "--in", $rlvrCand,
        "--out", $rlvrRqa,
        "--workers", "$workers"
    )
    if (-not (Test-Path $rlvrRqa)) { throw "RLVR Reverse-QA output missing" }

    $rlvrDest = Join-Path $RepoRoot "data\arabic_reasoning_rlvr_candidates_v5.jsonl"
    Copy-Item -Force $rlvrRqa $rlvrDest
    $nSft = (Get-Content $dest | Measure-Object -Line).Lines
    $nRlvr = (Get-Content $rlvrDest | Measure-Object -Line).Lines
    Write-Log "WROTE $rlvrDest rows=$nRlvr (pass@8+curate on Vast after SFT)"

    $payload = @{
        status = "done"
        run_id = $RunId
        coldstart = $dest
        selected = $selected
        rlvr_candidates = $rlvrDest
        rows_sft = $nSft
        rows_rlvr_candidates = $nRlvr
        log = $LogFile
        finished_at = (Get-Date).ToString("o")
        note = "Local teacher gen complete; Vast must run SFT train then GPU pass@8 then curate/promote rlvr_v5 then GRPO"
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
