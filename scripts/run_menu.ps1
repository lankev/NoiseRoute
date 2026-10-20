$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..')
$PythonPath = Join-Path (Get-Location) '.venv/Scripts/python.exe'
if (-not (Test-Path $PythonPath)) {
    Write-Error "Environnement Python absent: $PythonPath. Exécute d'abord .\scripts\setup_windows.cmd -Cuda, puis relance .\scripts\run_menu.cmd."
    exit 2
}
$Python = (Resolve-Path $PythonPath).Path
$Stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
Write-Host '1 Tests | 2 Benchmark 20 seeds | 3 Demo | 4 Train RTX + SABRE gate | 5 Smoke pipeline | 6 NPU latency | 7 Diagnostic RTX'
$Choice = Read-Host 'Choice'
switch ($Choice) {
    '1' { & $Python -m pytest -q }
    '2' { & $Python -m noiseroute bench --config configs/bench-small20.yaml --out "runs/bench-$Stamp" }
    '3' { & $Python -m streamlit run examples/demo.py }
    '4' {
        & $Python scripts/train_competitive.py
    }
    '5' { & $Python scripts/reproduce.py --smoke --out "runs/repro-$Stamp" }
    '6' { & $Python -m noiseroute latency --model models/smoke-policy.onnx --engine openvino --device NPU --out "runs/npu-$Stamp.json" }
    '7' { & $Python scripts/train_diagnostic.py }
    default { throw 'Choose 1 to 7.' }
}
if ($LASTEXITCODE -ne 0) { throw 'Command failed; inspect the error above.' }
