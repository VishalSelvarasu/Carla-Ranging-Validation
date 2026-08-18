$ErrorActionPreference = "Continue"
Set-Location (Join-Path $PSScriptRoot "..")

$script:Pass = 0; $script:Warn = 0; $script:Fail = 0
function Ok   ($m) { Write-Host "  [ OK ] $m" -ForegroundColor Green;  $script:Pass++ }
function Warn ($m) { Write-Host "  [WARN] $m" -ForegroundColor Yellow; $script:Warn++ }
function Bad  ($m) { Write-Host "  [FAIL] $m" -ForegroundColor Red;    $script:Fail++ }
function Sec  ($m) { Write-Host ""; Write-Host "== $m ==" -ForegroundColor Cyan }

Sec "1. GPU and driver"
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    $q = (nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader,nounits) -split ',\s*'
    Ok "$($q[0]), driver $($q[1]), $($q[2]) MiB"
    $vram = [int]$q[2]
    if     ($vram -lt 5800) { Bad  "VRAM below CARLA 0.9.15's 6 GB minimum" }
    elseif ($vram -lt 7800) { Warn "6-8 GB: keep -quality-level=Low and 800x600" }
    if ([int](($q[1] -split '\.')[0]) -lt 528) { Warn "driver is old; update via GeForce Experience" }
} else {
    Bad "nvidia-smi not found. Add C:\Windows\System32 to PATH or reinstall the driver."
}

Sec "2. CARLA installation"
$carlaRoot = $env:CARLA_ROOT
if (-not $carlaRoot) {
    foreach ($c in @("C:\CARLA_0.9.15", "D:\CARLA_0.9.15", "$HOME\CARLA_0.9.15")) {
        if (Test-Path (Join-Path $c "CarlaUE4.exe")) { $carlaRoot = $c; break }
    }
}
if ($carlaRoot -and (Test-Path (Join-Path $carlaRoot "CarlaUE4.exe"))) {
    Ok "CarlaUE4.exe found at $carlaRoot"
    if (-not $env:CARLA_ROOT) {
        Warn "CARLA_ROOT not set. Set it permanently:
        [Environment]::SetEnvironmentVariable('CARLA_ROOT','$carlaRoot','User')"
    }
} else {
    Bad "CarlaUE4.exe not found. Download CARLA_0.9.15.zip from
        https://github.com/carla-simulator/carla/releases/tag/0.9.15
        extract it, then set CARLA_ROOT to that folder.
        Do NOT use Docker on Windows -- Vulkan passthrough into WSL2 is the fragile path."
}

Sec "3. Conda environments"
if (Get-Command conda -ErrorAction SilentlyContinue) {
    Ok "conda $((conda --version) -split ' ' | Select-Object -Last 1)"
    $envs = (conda env list) | ForEach-Object { ($_ -split '\s+')[0] }
    foreach ($e in @("carla38", "percep")) {
        if ($envs -contains $e) { Ok "env '$e' exists" }
        else { Bad "env '$e' missing: conda env create -f environment\$e.yml" }
    }
    if ($envs -contains "carla38") {
        $pyv = conda run -n carla38 python -c "import sys;print('%d.%d'%sys.version_info[:2])" 2>$null
        if ($pyv -eq "3.8") { Ok "carla38 runs Python $pyv" }
        else { Bad "carla38 runs Python $pyv -- CARLA 0.9.15 requires 3.8" }
        conda run -n carla38 python -c "import carla" 2>$null
        if ($LASTEXITCODE -eq 0) { Ok "carla module imports" }
        else { Bad "carla missing: conda run -n carla38 pip install carla==0.9.15" }
    }
    if ($envs -contains "percep") {
        $missing = conda run -n percep python -c @"
import importlib.util
mods=['numpy','pandas','cv2','matplotlib','yaml','pyarrow','pytest']
print(','.join(m for m in mods if not importlib.util.find_spec(m)))
"@ 2>$null
        if ([string]::IsNullOrWhiteSpace($missing)) { Ok "percep has all required packages" }
        else { Bad "percep missing: $missing" }
    }
} else {
    Bad "conda not found -- install Miniforge, then run 'conda init powershell' and reopen the terminal"
}

Sec "4. Disk"
$drive = (Get-Item .).PSDrive.Name
$free = [math]::Round((Get-PSDrive $drive).Free / 1GB)
if     ($free -ge 40) { Ok   "$free GB free on ${drive}:" }
elseif ($free -ge 20) { Warn "$free GB free -- tight; CARLA + dataset want ~40 GB" }
else                  { Bad  "$free GB free -- not enough" }

Sec "5. Power and CPU"
$cores = (Get-CimInstance Win32_ComputerSystem).NumberOfLogicalProcessors
if ($cores -ge 6) { Ok "$cores logical cores" } else { Warn "$cores cores -- CARLA is CPU-bound" }
$batt = Get-CimInstance Win32_Battery -ErrorAction SilentlyContinue
if ($batt -and $batt.BatteryStatus -ne 2) {
    Warn "on battery -- clocks drop and the tick rate becomes unstable. Plug in."
} elseif ($batt) { Ok "on mains power" }
$plan = (powercfg /getactivescheme) 2>$null
if ($plan -match "Power saver") { Warn "power plan is 'Power saver' -- switch to High performance" }

Sec "6. Geometry tests (no server needed)"
$out = conda run -n percep python -m pytest tests/ -q 2>&1
if ($LASTEXITCODE -eq 0) { Ok ($out | Select-Object -Last 1) }
else { Bad "geometry tests failing:"; $out | Select-Object -Last 15 | ForEach-Object { Write-Host "        $_" } }

Sec "7. CARLA server (only if already started)"
$conn = Test-NetConnection -ComputerName 127.0.0.1 -Port 2000 -WarningAction SilentlyContinue
if ($conn.TcpTestSucceeded) {
    Ok "something is listening on :2000"
    $ver = conda run -n carla38 python -c @"
import carla
c=carla.Client('127.0.0.1',2000); c.set_timeout(10.0); print(c.get_server_version())
"@ 2>$null
    if ($ver) { Ok "server version $ver" } else { Warn "port open but handshake failed" }

    # This constant changed between 0.9.13 and 0.9.14. A wrong value makes every
    # visibility check return zero without raising anything.
    $tag = conda run -n carla38 python -c "import carla;print(int(carla.CityObjectLabel.Vehicles))" 2>$null
    if ($tag) {
        $cur = (Select-String -Path src\labels.py -Pattern 'SEM_TAG_VEHICLE\s*=\s*(\d+)').Matches[0].Groups[1].Value
        if ($tag -eq $cur) { Ok "SEM_TAG_VEHICLE=$cur matches this build" }
        else { Bad "SEM_TAG_VEHICLE is $cur but this build reports $tag -- edit src\labels.py" }
    }
} else {
    Warn "no server on :2000 -- start it with .\scripts\start_server.ps1"
}

Write-Host ""
Write-Host "================================"
Write-Host "  pass $script:Pass   warn $script:Warn   fail $script:Fail"
Write-Host "================================"
if ($script:Fail -gt 0) { Write-Host "Fix the FAIL items before running the pipeline." -ForegroundColor Red; exit 1 }
Write-Host "Preflight clean. Next: .\scripts\start_server.ps1, then .\run_all.ps1" -ForegroundColor Green
