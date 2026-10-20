# Installation et matériel

## Environnement principal

Python 3.12 recommandé (package : 3.11–3.13). Dans NoiseRoute, PowerShell :

Si l’écran affiche `No suitable Python runtime found`, installe Python avec `winget` :

```powershell
winget install --id Python.Python.3.12 -e --source winget
```

Ferme ensuite PowerShell, ouvre une nouvelle fenêtre dans le dossier NoiseRoute et vérifie :

```powershell
py --list
py -3.12 --version
```

Lanceur `py` absent ? Le script accepte aussi une commande `python` correspondant aux versions 3.11, 3.12 ou 3.13. L’installeur Python officiel doit alors avoir l’option `Add python.exe to PATH` activée.

```powershell
py -3.12 -m venv .venv
.venv/Scripts/python.exe -m pip install --upgrade pip
.venv/Scripts/python.exe -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/python.exe -m pip install -e '.[train,bench,export,demo,dev]'
.venv/Scripts/python.exe -m noiseroute doctor --out runs/doctor.json
.venv/Scripts/python.exe -m pytest -q
```

Si PowerShell refuse les fichiers avec `PSSecurityException`/« fichier non signé », les fichiers `.cmd` inclus lancent les mêmes scripts avec une politique temporaire limitée au processus :

```powershell
./scripts/setup_windows.cmd -Cuda
./scripts/run_menu.cmd
```

Tu peux aussi débloquer uniquement les scripts du projet avec `Get-ChildItem .\scripts\*.ps1 | Unblock-File`, puis relancer les commandes `.ps1`. Évite de modifier la stratégie globale `LocalMachine`.

La roue CUDA 12.8 nécessite un pilote NVIDIA compatible. Si CUDA n'est pas détecté, vérifiez `nvidia-smi` et PyTorch. Pour un essai CPU, remplacez l'index `cu128` par `cpu` et `torch_device: cuda` par `cpu` dans une copie du YAML. L'entraînement n'utilise pas le NPU. L'Arc Intel n'est pas le GPU CUDA.

Les versions exactes des résultats livrés sont dans `results/smoke/doctor.json` et les manifestes. Les plages de dépendances ne sont pas un lock universel. Les scripts installent PyTorch 2.8.0 ; la validation livrée utilisait la version CPU indiquée dans son manifeste. Revalidez les tests après une mise à jour.

## Rust facultatif

Installez Rust via rustup et un compilateur C/C++. Windows : Visual Studio Build Tools, composant Desktop development with C++, SDK Windows. Linux : build-essential. Puis :

```bash
python -m pip install ./rust
python -m noiseroute doctor
```

`NOISEROUTE_CORE=auto` choisit Rust s'il est installé, sinon Python. `NOISEROUTE_CORE=rust` exige Rust ; utilisez-le pour comparer le coût de l'environnement. `NOISEROUTE_CORE=python` force la référence.

## AIRouting local isolé

`qiskit-ibm-transpiler==0.18.0` dépend d'API Qiskit plus anciennes et impose NetworkX 2.8.5, incompatible avec certaines dépendances du corpus. Ne mélangez pas l'extra `ai` avec tous les extras.

```bash
python scripts/setup_ai.py
export NOISEROUTE_AI_PYTHON="$PWD/.venv-ai/bin/python"
export NOISEROUTE_AI_TIMEOUT=120
python -m noiseroute bench --config configs/bench-ai-smoke.yaml --out runs/ai-001
```

Après le script d'installation, PowerShell :

```powershell
$env:NOISEROUTE_AI_PYTHON = (Resolve-Path '.venv-ai/Scripts/python.exe').Path
$env:NOISEROUTE_AI_TIMEOUT = '120'
.venv/Scripts/python.exe -m noiseroute bench --config configs/bench-ai-smoke.yaml --out runs/ai-001
```

Le worker utilise Qiskit 2.3.1 et le helper IBM qui construit AIRouting en `local_mode=True`. Aucun token IBM n'est nécessaire. Le premier usage peut nécessiter des ressources/modèles upstream et du réseau. Une erreur ou un timeout reste visible dans le CSV. Le format QPY transmet les circuits compilés à l'environnement principal.

`compile_ms` inclut processus et chargement ; `compile_ms_inner` ne mesure que `pm.run`. Ces temps diffèrent de ceux d'un service persistant. La seed Qiskit n'est pas déclarée contrôler l'aléa interne AIRouting.

## NPU et OpenVINO

Essayez Windows natif avec le pilote Intel NPU adapté. Sa présence dans le Gestionnaire des tâches ne garantit pas sa disponibilité dans Python/WSL. Lancez `doctor`, puis :

```bash
python -m noiseroute latency --model models/smoke-policy.onnx --engine openvino --device NPU --out runs/npu.json
```

L'export a une forme statique, batch 1. ONNX ne garantit pas que toutes les opérations seront acceptées par le pilote NPU. Une erreur de compilation doit être conservée et résolue ; elle n'est pas masquée. Mesurez séparément le modèle final 128 nœuds et le petit modèle smoke.

Variables du plugin : `NOISEROUTE_MODEL`, `NOISEROUTE_ENGINE=torch|onnx|openvino`, `NOISEROUTE_ACCELERATOR=CPU|NPU|GPU`, `NOISEROUTE_BEAM=1|4|...`, `NOISEROUTE_DEVICE` pour un snapshot correspondant au target. Le wrapper `.pt` infère sur CPU ; la configuration CUDA concerne l'entraînement.

## Docker et pip

```bash
docker build -t noiseroute .
docker run --rm -v "$PWD/docker-results:/results" noiseroute
python -m build
python -m pip install dist/noiseroute-0.1.0-py3-none-any.whl
```

Docker fournit un profil CPU. L'image et les workflows sont fournis, sans exécution distante déclenchée. Le moteur Rust est un deuxième package. La roue principale seule utilise l'oracle Python et ne contient pas les poids ni les corpus de l'archive. Les scripts Windows sont fournis ; cette livraison a été vérifiée sous Linux.
