param(
    [string]$Out = ".\dataset",
    [int]$Seed = 42,
    [int]$Frames = 300,
    [string]$Map = "Town10HD_Opt",
    [double]$Fov = 60.0,
    [string]$LeadDistances = "90",
    [int]$Port = 2000
)

$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

# Conditions come from configs\conditions.yaml, never from a list in here.
# Two lists drift apart, and the drift surfaces as a results table whose rows
# are silently in the wrong order.
$conditions = conda run --no-capture-output -n percep python -m src.config --names
$conditions = $conditions | Where-Object { $_.Trim() } | ForEach-Object { $_.Trim() }
if (-not $conditions) { Write-Host "No conditions parsed from configs\conditions.yaml" -ForegroundColor Red; exit 1 }

New-Item -ItemType Directory -Force -Path $Out | Out-Null
$log = Join-Path $Out "matrix.log"
"" | Set-Content $log

function Log($m) { Write-Host $m; Add-Content $log $m }

# 90 deg fov puts a car under the 16px height floor past ~37 m, which empties
# the far distance bins. 60 is also closer to a real forward ADAS lens.
# Convoy starts at 30 m -- anything closer makes autopilot crawl, and brake
# latency in metres is meaningless at 3 m/s.
Log "seed=$Seed frames=$Frames map=$Map fov=$Fov lead=$LeadDistances"
Log "conditions ($($conditions.Count)): $($conditions -join ' ')"
Log "---"

foreach ($cond in $conditions) {
    $target = Join-Path $Out $cond
    if (Test-Path (Join-Path $target "run_config.json")) { Log "SKIP  $cond (already present)"; continue }

    # The server has died mid-matrix before. Check first so the failure names
    # itself instead of showing up as a 30 s client timeout.
    if (-not (Test-NetConnection -ComputerName 127.0.0.1 -Port $Port -WarningAction SilentlyContinue -InformationLevel Quiet)) {
        Log "CARLA is not listening on :$Port. Restart it with .\scripts\start_server.ps1"
        Log "Completed conditions are kept -- rerun this script to resume."
        exit 1
    }

    Log "RUN   $cond -> $target"
    $t0 = Get-Date

    # Same seed for every condition. This is the entire experimental design:
    # identical ego and traffic trajectories mean any metric delta is caused by
    # weather alone. Vary the seed and you have confounded your own experiment.
    conda run --no-capture-output -n carla38 python scripts\carla_capture.py `
        --out $target --seed $Seed --frames $Frames --map $Map --weather $cond `
        --fov $Fov --lead-distances $LeadDistances 2>&1 |
    Tee-Object -FilePath $log -Append

    if ($LASTEXITCODE -ne 0) { Log "FAILED on $cond"; exit 1 }
    Log "DONE  $cond in $([int]((Get-Date) - $t0).TotalSeconds)s"
}

Log "---"
Log "Verifying trajectory identity across conditions..."

$verify = @'
import json, os, sys
root, conds = sys.argv[1], sys.argv[2:]
present = [c for c in conds if os.path.isdir(os.path.join(root, c, "meta"))]
if len(present) < 2:
    sys.exit("not enough conditions to compare")

def poses(cond):
    d = os.path.join(root, cond, "meta"); out = []
    for fn in sorted(os.listdir(d)):
        with open(os.path.join(d, fn)) as f:
            out.append(json.load(f)["ego_transform"]["location"])
    return out

ref_name, ref = present[0], poses(present[0])
bad = False
for cond in present[1:]:
    cur = poses(cond); n = min(len(ref), len(cur))
    worst = max(max(abs(a[k]-b[k]) for k in "xyz") for a, b in zip(ref[:n], cur[:n]))
    if worst >= 1e-3: bad = True
    print(f"  {'OK ' if worst < 1e-3 else 'FAIL'} {cond:16s} max ego drift vs {ref_name}: {worst:.6f} m")

if bad:
    print("\n  Trajectories diverge. Any per-condition metric is now confounded")
    print("  by path differences rather than weather. Fix before evaluating.")
    sys.exit(1)
print("\n  Trajectories identical. Weather is the only varying factor.")
'@
$tmp = Join-Path $env:TEMP "verify_traj.py"
$verify | Set-Content -Encoding UTF8 $tmp

conda run --no-capture-output -n percep python $tmp $Out @conditions 2>&1 |
Tee-Object -FilePath $log -Append
if ($LASTEXITCODE -ne 0) { exit 1 }

# Detection counts per condition. These come from geometry and depth, not from
# pixels, so weather should barely move them -- a big drop means the visibility
# test is being affected by rain rather than the ranging being affected.
Log "---"
$summary = @'
import json, os, sys
root = sys.argv[1]
for cond in sys.argv[2:]:
    p = os.path.join(root, cond, "meta")
    if not os.path.isdir(p):
        continue
    speeds = []
    for fn in sorted(os.listdir(p)):
        with open(os.path.join(p, fn)) as f:
            speeds.append(json.load(f)["ego_speed_mps"])
    cfg_path = os.path.join(root, cond, "run_config.json")
    lead = "?"
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            lead = json.load(f).get("lead_spawned", "?")
    print(f"  {cond:16s} frames {len(speeds):4d}  "
          f"ego speed mean {sum(speeds)/len(speeds):5.2f} m/s  lead {lead}")
'@
$tmp2 = Join-Path $env:TEMP "matrix_summary.py"
$summary | Set-Content -Encoding UTF8 $tmp2
conda run --no-capture-output -n percep python $tmp2 $Out @conditions 2>&1 |
Tee-Object -FilePath $log -Append

Log "Matrix complete: $Out"
