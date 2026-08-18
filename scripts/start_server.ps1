param(
    [switch]$Stop,
    [string]$Quality = "Low",
    [int]$Port = 2000,
    [int]$TimeoutSec = 240
)

$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

if ($Stop) {
    $p = Get-Process CarlaUE4-Win64-Shipping, CarlaUE4 -ErrorAction SilentlyContinue
    if ($p) { $p | Stop-Process -Force; Write-Host "Stopped CARLA." }
    else    { Write-Host "Not running." }
    exit 0
}

if ((Test-NetConnection -ComputerName 127.0.0.1 -Port $Port -WarningAction SilentlyContinue).TcpTestSucceeded) {
    Write-Host "CARLA already listening on :$Port" -ForegroundColor Green
    exit 0
}

$carlaRoot = $env:CARLA_ROOT
if (-not $carlaRoot) {
    foreach ($c in @("C:\CARLA_0.9.15", "D:\CARLA_0.9.15", "$HOME\CARLA_0.9.15")) {
        if (Test-Path (Join-Path $c "CarlaUE4.exe")) { $carlaRoot = $c; break }
    }
}
$exe = Join-Path $carlaRoot "CarlaUE4.exe"
if (-not (Test-Path $exe)) {
    Write-Host "CarlaUE4.exe not found. Set CARLA_ROOT to your extracted CARLA folder:" -ForegroundColor Red
    Write-Host '  [Environment]::SetEnvironmentVariable("CARLA_ROOT","D:\CARLA_0.9.15","User")'
    exit 1
}

# -RenderOffScreen skips the window entirely. There is no spectator view to
# watch, so rendering one wastes GPU on frames nobody sees.
$carlaArgs = @("-quality-level=$Quality", "-RenderOffScreen", "-nosound", "-carla-server",
               "-world-port=$Port", "-carla-rpc-port=$Port")

Write-Host "Starting CARLA (quality=$Quality) from $carlaRoot ..."
$proc = Start-Process -FilePath $exe -ArgumentList $carlaArgs -PassThru -WorkingDirectory $carlaRoot

Write-Host -NoNewline "Waiting for RPC on :$Port "
for ($i = 1; $i -le $TimeoutSec; $i++) {
    if ((Test-NetConnection -ComputerName 127.0.0.1 -Port $Port -WarningAction SilentlyContinue -InformationLevel Quiet)) {
        Write-Host " ready after ${i}s" -ForegroundColor Green
        Write-Host "  PID $($proc.Id)"
        Write-Host "  Stop with: .\scripts\start_server.ps1 -Stop"
        exit 0
    }
    if ($proc.HasExited) {
        Write-Host ""
        Write-Host "CARLA exited with code $($proc.ExitCode)." -ForegroundColor Red
        Write-Host "Check $carlaRoot\CarlaUE4\Saved\Logs\CarlaUE4.log"
        Write-Host "Common causes: GPU driver too old, or another CARLA already holding the port."
        exit 1
    }
    Write-Host -NoNewline "."
    Start-Sleep -Seconds 1
}

Write-Host " TIMEOUT after ${TimeoutSec}s" -ForegroundColor Red
Write-Host "Check $carlaRoot\CarlaUE4\Saved\Logs\CarlaUE4.log"
exit 1
