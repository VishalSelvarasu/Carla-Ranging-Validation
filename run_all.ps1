param(
    [int]$From = 0,
    [int]$Only = -1,
    [switch]$Yes,
    [int]$Frames = 500,
    [int]$Seed = 42,
    [string]$Map = "Town10HD_Opt"
)
 
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
 
$Dataset = ".\dataset"
$Results = ".\results"
 
function C38 { conda run --no-capture-output -n carla38 @args }
function Per { conda run --no-capture-output -n percep  @args }
 
function Banner($n, $t) { Write-Host ""; Write-Host "############ PHASE ${n}: $t ############" -ForegroundColor Cyan }
function Runs($n) { if ($Only -ge 0) { return $Only -eq $n } else { return $n -ge $From } }
 
function Gate($msg, $phase) {
    Write-Host ""
    Write-Host "  GATE: $msg" -ForegroundColor Yellow
    if ($Yes) { Write-Host "  (-Yes: continuing)"; return }
    $a = Read-Host "  Passed? [y/N]"
    if ($a -notmatch '^[Yy]$') {
        Write-Host "  Stopped. Fix, then: .\run_all.ps1 -From $phase" -ForegroundColor Red
        exit 1
    }
}
 
# --------------------------------------------------------------------------
if (Runs 0) {
    Banner 0 "determinism"
    Remove-Item -Recurse -Force .\run_a, .\run_b -ErrorAction SilentlyContinue
 
    # 100 frames is enough: broken seeding diverges within a few ticks. No
    # point burning 500 frames to learn that.
    C38 python scripts\carla_capture.py --out .\run_a --seed $Seed --frames 100 --map $Map
    C38 python scripts\carla_capture.py --out .\run_b --seed $Seed --frames 100 --map $Map
 
    # Byte comparison is the WRONG test here, and an earlier version of this
    # script used it. Two reasons it fails on a correctly configured system:
    #   - meta JSON carries carla_frame / sim_timestamp, which count from SERVER
    #     start, so a second run on the same server process cannot match.
    #   - GPU rasterisation is not bit-reproducible at object silhouettes
    #     (~0.01% of pixels, mean difference ~0.0000 m).
    # Neither touches a number this project reports: true_range comes from actor
    # bounding boxes, not the depth buffer. verify_determinism.py checks
    # simulation state instead, which is the property the experiment needs.
    Per python scripts\verify_determinism.py .\run_a .\run_b
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  PASS: simulation state is reproducible." -ForegroundColor Green
        Remove-Item -Recurse -Force .\run_a, .\run_b
    }
    else {
        Write-Host "  FAIL: see the diagnosis above." -ForegroundColor Red
        Write-Host "  run_a and run_b are kept for inspection."
        Write-Host "  run_a is reproducible, run_b is not. Compare the two to find the source of non-determinism."
        Write-Host "  run_b is the one to fix, run_a is the reference."
        exit 1
    }
}
 
# --------------------------------------------------------------------------
if (Runs 1) {
    Banner 1 "condition matrix"
    & .\scripts\run_matrix.ps1 -Out $Dataset -Seed $Seed -Frames $Frames -Map $Map
    if ($LASTEXITCODE -ne 0) { exit 1 }
    $sz = [math]::Round((Get-ChildItem $Dataset -Recurse -File | Measure-Object Length -Sum).Sum / 1GB, 2)
    Write-Host "  dataset size: $sz GB"
    Gate "Trajectory check reported identical ego paths across all conditions?" 1
}
 
# --------------------------------------------------------------------------
if (Runs 2) {
    Banner 2 "labels"
    $base = (Per python -m src.config --baseline).Trim()
    Per python -m src.labels --run (Join-Path $Dataset $base) --debug-frame 42
 
    # Relative paths only. $_.FullName would be absolute, and an absolute path
    # containing spaces (e.g. "D:\Projects\Carla Ranging Validation\dataset\...")
    # is unreliable through `conda run`, which re-quotes arguments on its way to
    # the subprocess. Relative paths sidestep the problem entirely.
    Get-ChildItem $Dataset -Directory | Where-Object {
        Test-Path (Join-Path $_.FullName "run_config.json")
    } | ForEach-Object { Per python -m src.labels --run (Join-Path $Dataset $_.Name) }
 
    Per python -m src.inspect_labels --dataset $Dataset --frames 4
 
    Write-Host @"
 
  Open these two files and LOOK at them:
    $Dataset\$base\debug_000042.png
    $Dataset\contact_sheet.png
 
  Boxes must sit ON the vehicles, nothing drawn on hidden cars, ranges
  plausible, and object counts roughly flat across conditions.
"@
    Gate "Labels visually correct?" 2
}
 
# --------------------------------------------------------------------------
if (Runs 3) {
    Banner 3 "evaluation"
    New-Item -ItemType Directory -Force -Path $Results | Out-Null
    Per python -m src.evaluate --dataset $Dataset --out $Results
    Gate "Degradation trend visible, and at least one condition genuinely fails?" 3
}
 
# --------------------------------------------------------------------------
if (Runs 4) {
    Banner 4 "report"
    Per python -m src.report --results $Results
    Write-Host ""
    Write-Host "  Figures and RESULTS_AUTO.md are in $Results\"
    Write-Host "  Now write results\RESULTS.md by hand -- sections 4, 5 and 6 are the"
    Write-Host "  ones that make this read as engineering rather than coursework."
}
 
Write-Host ""
Write-Host "Pipeline complete." -ForegroundColor Green