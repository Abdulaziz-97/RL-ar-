# Focused top-up for math / math_comp / logic shortages after first merge select.
# Requires DEEPSEEK_API_KEY. Resumes into merge+select+IFEval compose when done.
$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Data1 = Join-Path $RepoRoot "data_1"
$RunId = if ($env:RUN_ID) { $env:RUN_ID } else { "local_v4_20260801_073920" }
$WorkRoot = Join-Path $Data1 "outputs\v4_regen\$RunId"
$LogFile = Join-Path $RepoRoot "outputs\local_sft_shortage_topup_$RunId.log"
$FailMarker = Join-Path $Data1 "outputs\LOCAL_SFT_DATAGEN_FAILED.json"
$DoneMarker = Join-Path $Data1 "outputs\LOCAL_SFT_DATAGEN_DONE.json"

if (-not $env:DEEPSEEK_API_KEY) { throw "DEEPSEEK_API_KEY required" }

New-Item -ItemType Directory -Force -Path $WorkRoot, (Join-Path $RepoRoot "outputs"), (Join-Path $RepoRoot "data") | Out-Null
Remove-Item -Force $FailMarker -ErrorAction SilentlyContinue

function Write-Log([string]$msg) {
    $line = "[{0}] {1}" -f (Get-Date -Format "o"), $msg
    Write-Host $line
    Add-Content -Path $LogFile -Value $line -Encoding UTF8
}

function Invoke-LoggedPython {
    param([Parameter(Mandatory = $true)][string[]]$ArgumentList)
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
    if ($code -ne 0) { throw ("python exit={0} args={1}" -f $code, ($ArgumentList -join " ")) }
}

$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONPATH = "$Data1;$Data1\src;$Data1\vendor"
$env:USE_OPENROUTER = if ($env:USE_OPENROUTER) { $env:USE_OPENROUTER } else { "0" }
$env:TEACHER_MODEL = if ($env:TEACHER_MODEL) { $env:TEACHER_MODEL } else { "deepseek-v4-pro" }
$env:TEACHER_MAX_TOKENS = if ($env:TEACHER_MAX_TOKENS) { $env:TEACHER_MAX_TOKENS } else { "8192" }
$workers = if ($env:TEACHER_WORKERS) { [int]$env:TEACHER_WORKERS } else { 48 }
$env:REVERSE_QA_MODEL = if ($env:REVERSE_QA_MODEL) { $env:REVERSE_QA_MODEL } else { "deepseek-v4-flash" }
$env:REVERSE_QA_RETEACH_MODEL = if ($env:REVERSE_QA_RETEACH_MODEL) { $env:REVERSE_QA_RETEACH_MODEL } else { $env:REVERSE_QA_MODEL }
$env:REVERSE_QA_MODE = "full"
$env:REVERSE_QA_RESOLVE = "1"
$nTop = if ($env:SFT_SHORTAGE_N) { [int]$env:SFT_SHORTAGE_N } else { 4500 }
$topSeed = if ($env:SFT_SHORTAGE_SEED) { [int]$env:SFT_SHORTAGE_SEED } else { 3300 }

$runPipeline = Join-Path $Data1 "scripts\run_pipeline.py"
$selectScript = Join-Path $Data1 "scripts\select_sft_v4_release.py"
$mergeScript = Join-Path $Data1 "scripts\merge_sft_jsonl.py"
$SftCfg = Join-Path $Data1 "configs\full_sft_6500.yaml"
$RlvrCfg = Join-Path $Data1 "configs\full_rlvr_8000.yaml"
$baseRelease = Join-Path $WorkRoot "sft_candidates\release_corpora\sft_train.jsonl"
$top1Release = Join-Path $WorkRoot "sft_topup_6000\release_corpora\sft_train.jsonl"
if (-not (Test-Path $baseRelease)) { throw "Missing base release $baseRelease" }
if (-not (Test-Path $top1Release)) { throw "Missing first top-up release $top1Release" }

Write-Log "SHORTAGE TOPUP START n=$nTop seed=$topSeed domains=math/math_comp/logic"

try {
    $TopUp = Join-Path $WorkRoot "sft_topup_shortage_$nTop"
    $TopCfg = Join-Path $WorkRoot "sft_topup_shortage_$nTop.yaml"
    New-Item -ItemType Directory -Force -Path $TopUp | Out-Null
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & python -c @"
from pathlib import Path
import yaml
cfg = yaml.safe_load(Path(r'$SftCfg').read_text(encoding='utf-8'))
cfg['n_families'] = $nTop
cfg['seed'] = $topSeed
cfg['work_dir'] = r'$TopUp'
cfg['domains'] = ['math', 'math_comp', 'logic', 'math', 'logic', 'math_comp']
Path(r'$TopCfg').write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding='utf-8')
print(r'$TopCfg')
"@
    $ErrorActionPreference = $prev

    $env:REVERSE_QA = "0"
    Invoke-LoggedPython -ArgumentList @(
        $runPipeline, "--mode", "live", "--config", $TopCfg,
        "--work-dir", $TopUp, "--track", "sft",
        "--model", $env:TEACHER_MODEL, "--workers", "$workers",
        "--budget-usd", "$(if ($env:SFT_BUDGET_USD) { $env:SFT_BUDGET_USD } else { '200' })"
    )

    $top2Release = Join-Path $TopUp "release_corpora\sft_train.jsonl"
    if (-not (Test-Path $top2Release)) { throw "Shortage top-up release missing" }

    $merged = Join-Path $WorkRoot "sft_merged_candidates.jsonl"
    Write-Log "MERGE base+topup1+shortage -> $merged"
    Invoke-LoggedPython -ArgumentList @(
        $mergeScript, "--out", $merged, $baseRelease, $top1Release, $top2Release
    )

    $selected = Join-Path $WorkRoot "sft_selected_4000.jsonl"
    Write-Log "SELECT 4k from merged"
    Invoke-LoggedPython -ArgumentList @(
        $selectScript, "--candidates", $merged, "--out", $selected, "--allow-missing-decontam"
    )

    $selectedRqa = Join-Path $WorkRoot "sft_selected_4000_reverse_qa.jsonl"
    Write-Log "FULL REVERSE-QA + reteach on selected 4k"
    $env:REVERSE_QA = "1"
    Invoke-LoggedPython -ArgumentList @(
        (Join-Path $Data1 "scripts\apply_full_reverse_qa_sft.py"),
        "--in", $selected, "--out", $selectedRqa, "--reteach", "--workers", "$workers"
    )
    if (Test-Path $selectedRqa) { $selected = $selectedRqa }

    $ifevalSide = Join-Path $RepoRoot "data\sidecars\ifeval_sft_v5.jsonl"
    if (-not (Test-Path $ifevalSide)) {
        Invoke-LoggedPython -ArgumentList @(
            (Join-Path $Data1 "scripts\generate_ifeval_sidecar_v5.py"),
            "--n", "500", "--seed", "4400", "--out", "$ifevalSide", "--reference", "$selected"
        )
    }

    $mixed = Join-Path $WorkRoot "sft_mixed_core_ifeval.jsonl"
    Write-Log "COMPOSE mixed SFT (core + IFEval; mcq=0)"
    Invoke-LoggedPython -ArgumentList @(
        (Join-Path $Data1 "scripts\compose_mixed_sft_v5.py"),
        "--core", "$selected", "--ifeval", "$ifevalSide", "--out", "$mixed", "--seed", "42"
    )

    $dest = Join-Path $RepoRoot "data\arabic_reasoning_coldstart_v5.jsonl"
    Copy-Item -Force $mixed $dest
    $nSft = (Get-Content $dest | Measure-Object -Line).Lines
    Write-Log "WROTE $dest rows=$nSft"

    $RlvrWork = Join-Path $WorkRoot "rlvr_candidates"
    $RlvrRuntime = Join-Path $WorkRoot "full_rlvr_8000.runtime.yaml"
    $rlvrSeed = if ($env:RLVR_SEED) { [int]$env:RLVR_SEED } else { 4400 }
    Write-Log "FULL RLVR candidates (core only; no MCQ)"
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & python -c @"
from pathlib import Path
import yaml
cfg = yaml.safe_load(Path(r'$RlvrCfg').read_text(encoding='utf-8'))
cfg['seed'] = $rlvrSeed
cfg['decontam_reference_paths'] = [r'$dest', r'$ifevalSide']
cfg['work_dir'] = r'$RlvrWork'
Path(r'$RlvrRuntime').write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding='utf-8')
print(r'$RlvrRuntime')
"@
    $ErrorActionPreference = $prev
    $env:REVERSE_QA = "0"
    Invoke-LoggedPython -ArgumentList @(
        $runPipeline, "--mode", "live", "--config", $RlvrRuntime,
        "--work-dir", $RlvrWork, "--track", "rlvr",
        "--model", $env:TEACHER_MODEL, "--workers", "1",
        "--budget-usd", "$(if ($env:RLVR_BUDGET_USD) { $env:RLVR_BUDGET_USD } else { '80' })"
    )
    $rlvrCand = Join-Path $RlvrWork "release_corpora\rlvr_train.jsonl"
    $rlvrRqa = Join-Path $WorkRoot "rlvr_candidates_reverse_qa.jsonl"
    $env:REVERSE_QA = "1"
    Invoke-LoggedPython -ArgumentList @(
        (Join-Path $Data1 "scripts\apply_full_reverse_qa_sft.py"),
        "--in", $rlvrCand, "--out", $rlvrRqa, "--workers", "$workers"
    )
    $rlvrDest = Join-Path $RepoRoot "data\arabic_reasoning_rlvr_candidates_v5.jsonl"
    Copy-Item -Force $rlvrRqa $rlvrDest
    $nRlvr = (Get-Content $rlvrDest | Measure-Object -Line).Lines

    $payload = @{
        status = "done"
        run_id = $RunId
        coldstart = $dest
        ifeval_sidecar = $ifevalSide
        rlvr_candidates = $rlvrDest
        rows_sft = $nSft
        rows_rlvr_candidates = $nRlvr
        mcq_generated = $false
        log = $LogFile
        finished_at = (Get-Date).ToString("o")
        note = "shortage topup filled 4k core + IFEval; core RLVR only"
    } | ConvertTo-Json
    Set-Content -Path $DoneMarker -Value $payload -Encoding UTF8
    Write-Log "DONE"
    exit 0
}
catch {
    $err = $_.Exception.Message
    Write-Log "FAILED $err"
    $payload = @{ status = "failed"; run_id = $RunId; error = $err; log = $LogFile; finished_at = (Get-Date).ToString("o") } | ConvertTo-Json
    Set-Content -Path $FailMarker -Value $payload -Encoding UTF8
    exit 1
}
