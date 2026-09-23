# Run upstream PolyFEM_bin on foxgirl_oracle.json and record wall time and
# peak working set (sampled every 100 ms, plus the OS peak counter read just
# before exit). Usage: run_oracle.ps1 -Threads 16 -OutDir C:/b/cf-up-out16
# The json's asset paths are relative to the repo root, so the process runs
# there. -Exe defaults to $env:CF_BUILD/PolyFEM_bin.exe, else build-native.
param(
    [int]$Threads = 16,
    [string]$OutDir = "C:/b/cf-up-out16",
    [string]$Exe = "",
    [string]$Json = "$PSScriptRoot/foxgirl_oracle.json"
)
$root = (Resolve-Path "$PSScriptRoot/../..").Path
if (-not $Exe) {
    $build = $env:CF_BUILD
    if (-not $build) { $build = "$root/build-native" }
    $Exe = "$build/PolyFEM_bin.exe"
}
if (Test-Path $OutDir) { Remove-Item -Recurse -Force $OutDir -Confirm:$false }
$log = "$OutDir.log"
$err = "$OutDir.err.log"
$args_ = @("-j", "`"$Json`"", "-o", "`"$OutDir`"", "--max_threads", "$Threads")
$sw = [System.Diagnostics.Stopwatch]::StartNew()
$p = Start-Process -FilePath $Exe -ArgumentList $args_ -PassThru -NoNewWindow `
    -WorkingDirectory $root `
    -RedirectStandardOutput $log -RedirectStandardError $err
$null = $p.Handle
$peak = 0L
$peakPriv = 0L
while (-not $p.HasExited) {
    try {
        $p.Refresh()
        if ($p.PeakWorkingSet64 -gt $peak) { $peak = $p.PeakWorkingSet64 }
        if ($p.PeakPagedMemorySize64 -gt $peakPriv) { $peakPriv = $p.PeakPagedMemorySize64 }
        $cpu = $p.TotalProcessorTime.TotalSeconds
    } catch {}
    Start-Sleep -Milliseconds 100
}
$p.WaitForExit()
$sw.Stop()
$user = $p.UserProcessorTime.TotalSeconds
$kern = $p.PrivilegedProcessorTime.TotalSeconds
$res = [ordered]@{
    threads = $Threads
    out_dir = $OutDir
    exit_code = $p.ExitCode
    wall_s = [math]::Round($sw.Elapsed.TotalSeconds, 3)
    cpu_s_last_sample = [math]::Round($cpu, 3)
    cpu_user_s = [math]::Round($user, 3)
    cpu_kernel_s = [math]::Round($kern, 3)
    peak_working_set_mb = [math]::Round($peak / 1MB, 1)
    peak_pagefile_mb = [math]::Round($peakPriv / 1MB, 1)
}
$res | ConvertTo-Json | Out-File -Encoding utf8 "$OutDir.result.json"
$res | ConvertTo-Json
