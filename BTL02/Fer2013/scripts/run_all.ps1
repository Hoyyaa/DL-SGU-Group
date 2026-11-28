# Chạy train -> evaluate -> visualize cho nhiều config, tự bỏ qua bước đã xong (chạy lại để tiếp tục).
#   .\scripts\run_all.ps1                                   # mọi configs\e*.yaml (e1 -> e15)
#   .\scripts\run_all.ps1 -Wandb                            # ghi log lên W&B
#   .\scripts\run_all.ps1 -Configs e1_small,e2_medium,e3_medium01,e4_large -Wandb
#   .\scripts\run_all.ps1 -Seed 1 -Configs e5_large_bn
# Một config lỗi không làm dừng cả loạt: ghi lại rồi chạy config tiếp theo, cuối cùng in bảng tổng kết.
# Log từng config ở logs\<name>_s<seed>.log.
param(
    [string[]]$Configs = @(),
    [int]$Seed = 42,
    [switch]$Wandb,
    [switch]$NoViz
)

$ErrorActionPreference = "Continue"
$env:PYTHONIOENCODING = "utf-8"
Set-Location (Join-Path $PSScriptRoot "..")
New-Item -ItemType Directory -Force logs | Out-Null

if ($Configs.Count -eq 0) {
    # sắp theo số thứ tự (e1, e2, ..., e10) thay vì ABC (e1, e10, e11, ..., e2)
    $Configs = Get-ChildItem configs\e*.yaml |
        Sort-Object { [int]($_.BaseName -replace '^e(\d+).*', '$1') } |
        ForEach-Object { $_.BaseName }
}
# "-File ... -Configs a,b" truyền vào một chuỗi "a,b" nên tách lại; chấp nhận cả configs\e1_small.yaml
$Configs = $Configs -split ',' | ForEach-Object { $_.Trim() } | Where-Object { $_ } |
    ForEach-Object { [IO.Path]::GetFileNameWithoutExtension($_) }

function Invoke-Step([string]$label, [string[]]$pyArgs, [string]$log) {
    Write-Host "  -> $label" -ForegroundColor Cyan
    $t = Get-Date
    # Tee-Object: vừa hiện tiến trình trên màn hình, vừa ghi log
    # Out-Host: nếu không, các dòng log sẽ lẫn vào giá trị trả về của hàm
    & python @pyArgs 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $log -Append | Out-Host
    $ok = ($LASTEXITCODE -eq 0)
    $min = [math]::Round(((Get-Date) - $t).TotalMinutes, 1)
    if (-not $ok) { Write-Host "  !! $label lỗi (exit $LASTEXITCODE), xem $log" -ForegroundColor Red }
    return @{ ok = $ok; min = $min }
}

$wb = @(); if ($Wandb) { $wb = @("--wandb") }
$summary = @()
$start = Get-Date
Write-Host "Sẽ chạy $($Configs.Count) config, seed $Seed`: $($Configs -join ', ')" -ForegroundColor Yellow

foreach ($name in $Configs) {
    $cfg = "configs\$name.yaml"
    $run = "runs\${name}_s$Seed"
    $res = "results\${name}_s$Seed"
    $log = "logs\${name}_s$Seed.log"
    $row = [ordered]@{ config = $name; train = "-"; evaluate = "-"; visualize = "-"; minutes = 0 }
    Write-Host "`n===== $name  ($(Get-Date -Format 'HH:mm')) =====" -ForegroundColor Yellow

    if (-not (Test-Path $cfg)) {
        Write-Host "  !! không thấy $cfg" -ForegroundColor Red
        $row.train = "không có config"; $summary += [pscustomobject]$row; continue
    }

    # train: có last.pt thì --resume (train.py tự bỏ qua nếu run đã xong)
    $trainArgs = @("-m", "fer.train", "--config", $cfg, "--seed", "$Seed") + $wb
    if (Test-Path "$run\last.pt") { $trainArgs += "--resume" }
    $r = Invoke-Step "train" $trainArgs $log
    $row.train = $(if ($r.ok) { "ok" } else { "LỖI" }); $row.minutes += $r.min
    if (-not $r.ok -or -not (Test-Path "$run\best.pt")) { $summary += [pscustomobject]$row; continue }

    # evaluate: bỏ qua nếu metrics.json mới hơn best.pt
    $metrics = "$res\metrics.json"
    if ((Test-Path $metrics) -and ((Get-Item $metrics).LastWriteTime -gt (Get-Item "$run\best.pt").LastWriteTime)) {
        $row.evaluate = "đã có"
    } else {
        $r = Invoke-Step "evaluate" @("-m", "fer.evaluate", "--run", $run) $log
        $row.evaluate = $(if ($r.ok) { "ok" } else { "LỖI" }); $row.minutes += $r.min
    }

    if ($NoViz) { $row.visualize = "bỏ qua" }
    elseif ((Test-Path "$res\viz\gradcam.png") -and ((Get-Item "$res\viz\gradcam.png").LastWriteTime -gt (Get-Item "$run\best.pt").LastWriteTime)) {
        $row.visualize = "đã có"
    } else {
        $r = Invoke-Step "visualize" @("-m", "fer.visualize", "--run", $run) $log
        $row.visualize = $(if ($r.ok) { "ok" } else { "LỖI" }); $row.minutes += $r.min
    }
    $summary += [pscustomobject]$row
}

$total = [math]::Round(((Get-Date) - $start).TotalHours, 2)
Write-Host "`n===== Tổng kết ($total giờ) =====" -ForegroundColor Yellow
$summary | Format-Table -AutoSize
if (Test-Path results\summary.md) { Get-Content results\summary.md -Encoding utf8 }
