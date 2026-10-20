param([switch]$Cuda, [switch]$Rust)
$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..')
function Run-Python { & $script:Python @args; if ($LASTEXITCODE -ne 0) { throw "Python failed: $args" } }
if (-not (Test-Path '.venv/Scripts/python.exe')) {
    $launcher = $null
    if (Get-Command py -ErrorAction SilentlyContinue) {
        & py -3.12 --version *> $null
        if ($LASTEXITCODE -eq 0) { $launcher = 'py' }
    }
    if (-not $launcher -and (Get-Command python -ErrorAction SilentlyContinue)) {
        $version = (& python --version 2>&1 | Out-String).Trim()
        if ($version -match '^Python 3\.(11|12|13)(\.|$)') { $launcher = 'python' }
    }
    if (-not $launcher) {
        throw 'Python 3.11, 3.12 ou 3.13 est requis. Installe-le avec: winget install --id Python.Python.3.12 -e --source winget ; puis ferme et rouvre PowerShell.'
    }
    if ($launcher -eq 'py') { & py -3.12 -m venv .venv } else { & python -m venv .venv }
    if ($LASTEXITCODE -ne 0) { throw 'Impossible de créer .venv. Vérifie l installation de Python.' }
}
$script:Python = (Resolve-Path '.venv/Scripts/python.exe').Path
Run-Python -m pip install --upgrade pip
if ($Cuda) {
    Run-Python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
} else {
    Run-Python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
}
Run-Python -m pip install -e '.[train,bench,export,demo,dev]'
if ($Rust) {
    if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) { throw 'Install Rust and Visual Studio C++ Build Tools, then run again with -Rust.' }
    Run-Python -m pip install ./rust
}
Run-Python -m noiseroute doctor --out runs/doctor.json
Write-Host 'Ready. Run: ./scripts/run_menu.ps1'
