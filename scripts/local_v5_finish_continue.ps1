# Continue after top-up release exists: merge -> select -> RQA -> IFEval compose -> RLVR 4000
$ErrorActionPreference = "Stop"
$RepoRoot = "C:\Users\Azooo\arabic-reasoning-rlvr-sota\RL-ar-"
$Data1 = Join-Path $RepoRoot "data_1"
$RunId = "local_v4_20260801_073920"
$WorkRoot = Join-Path $Data1 "outputs\v4_regen\$RunId"
$LogFile = Join-Path $RepoRoot "outputs\v5_finish_continue.log"
$DoneMarker = Join-Path $Data1 "outputs\LOCAL_SFT_DATAGEN_DONE.json"
$FailMarker = Join-Path $Data1 "outputs\LOCAL_SFT_DATAGEN_FAILED.json"

New-Item -ItemType Directory -Force -Path (Join-Path $RepoRoot "outputs"), (Join-Path $RepoRoot "data") | Out-Null
Remove-Item -Force $FailMarker -ErrorAction SilentlyContinue

function Write-Log([string]$msg) {
    $line = "[{0}] {1}" -f (Get-Date -Format "o"), $msg
    Write-Host $line
    try { Add-Content -Path $LogFile -Value $line -Encoding UTF8 -ErrorAction Stop } catch { }
}
function Invoke-LoggedPython {
    param([string[]]$ArgumentList)
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

if (-not $env:DEEPSEEK_API_KEY) { throw "DEEPSEEK_API_KEY required" }
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONPATH = "$Data1;$Data1\src;$Data1\vendor"
$env:USE_OPENROUTER = if ($env:USE_OPENROUTER) { $env:USE_OPENROUTER } else { "0" }
$workers = if ($env:TEACHER_WORKERS) { [int]$env:TEACHER_WORKERS } else { 48 }
$env:TEACHER_MODEL = if ($env:TEACHER_MODEL) { $env:TEACHER_MODEL } else { "deepseek-v4-pro" }
$env:REVERSE_QA_MODEL = if ($env:REVERSE_QA_MODEL) { $env:REVERSE_QA_MODEL } else { "deepseek-v4-flash" }
$env:REVERSE_QA_RETEACH_MODEL = $env:REVERSE_QA_MODEL
$env:REVERSE_QA_MODE = "full"
$env:REVERSE_QA_RESOLVE = "1"
$rlvrSeed = if ($env:RLVR_SEED) { [int]$env:RLVR_SEED } else { 3300 }

$selectScript = Join-Path $Data1 "scripts\select_sft_v4_release.py"
$uniqScript = Join-Path $Data1 "scripts\assert_prompt_uniqueness_v5.py"
$runPipeline = Join-Path $Data1 "scripts\run_pipeline.py"
$RlvrCfg = Join-Path $Data1 "configs\full_rlvr_8000.yaml"
$baseRelease = Join-Path $WorkRoot "sft_candidates\release_corpora\sft_train.jsonl"
$topRelease = Join-Path $WorkRoot "sft_topup_6000\release_corpora\sft_train.jsonl"

try {
    if (-not (Test-Path $topRelease)) { throw "top-up release missing" }
    $merged = Join-Path $WorkRoot "sft_merged_candidates.jsonl"
    Write-Log "MERGE base+topup (utf-8 no BOM)"
    Invoke-LoggedPython -ArgumentList @(
        "-c",
        ("from pathlib import Path; b=Path(r'{0}').read_text(encoding='utf-8-sig'); t=Path(r'{1}').read_text(encoding='utf-8-sig'); Path(r'{2}').write_text((b.rstrip()+'\n'+t.lstrip()).lstrip('\ufeff'), encoding='utf-8')" -f $baseRelease, $topRelease, $merged)
    )
    Invoke-LoggedPython -ArgumentList @($uniqScript, "--sft", $merged, "--drop-sft-dups")

    $selected = Join-Path $WorkRoot "sft_selected_4000.jsonl"
    Write-Log "SELECT max under quotas (allow shortages; pool may be <4000)"
    Invoke-LoggedPython -ArgumentList @(
        $selectScript, "--candidates", $merged, "--out", $selected,
        "--allow-missing-decontam", "--allow-shortages"
    )

    $selectedRqa = Join-Path $WorkRoot "sft_selected_4000_reverse_qa.jsonl"
    Write-Log "FULL REVERSE-QA + reteach"
    $env:REVERSE_QA = "1"
    Invoke-LoggedPython -ArgumentList @(
        (Join-Path $Data1 "scripts\apply_full_reverse_qa_sft.py"),
        "--in", $selected, "--out", $selectedRqa, "--reteach", "--workers", "$workers"
    )
    if (Test-Path $selectedRqa) { $selected = $selectedRqa }
    Invoke-LoggedPython -ArgumentList @($uniqScript, "--sft", $selected, "--drop-sft-dups")

    $ifevalSide = Join-Path $RepoRoot "data\sidecars\ifeval_sft_v5.jsonl"
    if (-not (Test-Path $ifevalSide)) {
        Invoke-LoggedPython -ArgumentList @(
            (Join-Path $Data1 "scripts\generate_ifeval_sidecar_v5.py"),
            "--n", "500", "--seed", "4400", "--out", $ifevalSide, "--reference", $selected
        )
    }
    $mixed = Join-Path $WorkRoot "sft_mixed_core_ifeval.jsonl"
    Write-Log "COMPOSE mixed SFT (no MCQ)"
    Invoke-LoggedPython -ArgumentList @(
        (Join-Path $Data1 "scripts\compose_mixed_sft_v5.py"),
        "--core", $selected, "--ifeval", $ifevalSide, "--out", $mixed, "--seed", "42"
    )
    Invoke-LoggedPython -ArgumentList @($uniqScript, "--sft", $mixed, "--drop-sft-dups")

    $dest = Join-Path $RepoRoot "data\arabic_reasoning_coldstart_v5.jsonl"
    Copy-Item -Force $mixed $dest
    $nSft = (Get-Content $dest | Measure-Object -Line).Lines
    Write-Log "WROTE coldstart rows=$nSft"

    $RlvrWork = Join-Path $WorkRoot "rlvr_candidates_4000"
    $RlvrRuntime = Join-Path $WorkRoot "full_rlvr_4000.runtime.yaml"
    Write-Log "RLVR candidates n_families=4000 seed=$rlvrSeed (NOT 4500, no MCQ)"
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & python -c @"
from pathlib import Path
import yaml
cfg = yaml.safe_load(Path(r'$RlvrCfg').read_text(encoding='utf-8'))
cfg['n_families'] = 4000
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
    $rlvrRqa = Join-Path $WorkRoot "rlvr_candidates_4000_reverse_qa.jsonl"
    $env:REVERSE_QA = "1"
    Invoke-LoggedPython -ArgumentList @(
        (Join-Path $Data1 "scripts\apply_full_reverse_qa_sft.py"),
        "--in", $rlvrCand, "--out", $rlvrRqa, "--workers", "$workers"
    )
    Invoke-LoggedPython -ArgumentList @(
        $uniqScript, "--sft", $dest, "--rlvr", $rlvrRqa, "--drop-rlvr-dups"
    )
    $rlvrDest = Join-Path $RepoRoot "data\arabic_reasoning_rlvr_candidates_v5.jsonl"
    Copy-Item -Force $rlvrRqa $rlvrDest
    $nRlvr = (Get-Content $rlvrDest | Measure-Object -Line).Lines

    $payload = @{
        status = "done"
        run_id = $RunId
        coldstart = $dest
        rlvr_candidates = $rlvrDest
        rows_sft = $nSft
        rows_rlvr_candidates = $nRlvr
        rlvr_target = 4000
        mcq_generated = $false
        note = "core SFT may be <4000 due to domain shortages; RLVR n_families=4000; no MCQ"
        finished_at = (Get-Date).ToString("o")
    } | ConvertTo-Json
    Set-Content -Path $DoneMarker -Value $payload -Encoding UTF8
    Write-Log "DONE sft=$nSft rlvr=$nRlvr"
    exit 0
}
catch {
    $err = $_.Exception.Message
    Write-Log "FAILED $err"
    (@{ status = "failed"; error = $err; finished_at = (Get-Date).ToString("o") } | ConvertTo-Json) |
        Set-Content -Path $FailMarker -Encoding UTF8
    exit 1
}

