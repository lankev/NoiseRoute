# NoiseRoute

Plateforme de recherche pour le **placement et le routage quantiques sensibles au bruit** : politique partagée GATv2 + transformer, environnement Rust/PyO3, plugin Qiskit, export ONNX/OpenVINO.

**Version 0.1.0 expérimentale.** Le pipeline est exécutable et testé. Les poids fournis proviennent d'un entraînement très court de validation. Aucune supériorité sur SABRE/AIRouting, généralisation zéro-shot ou latence NPU sur votre PC n'est revendiquée. Les cas perdants sont conservés dans les CSV.

![Démo de l'interface NoiseRoute](assets/demo.jpg)

**Nouveau parcours comparatif** : imitation à `5e-5`, PPO à `1e-5`, référence
d'imitation gelée, pénalité KL, récompense `Δ log(ESP)` face à SABRE et validation
obligatoire à 20 seeds avant création de `accepted.pt`. Lancer
`python scripts/train_competitive.py` ou l'option **4** du menu Windows.
Le premier profil couvre 5 qubits ; les critères exacts, la mise à jour Windows
et la reprise sont décrits dans [docs/COMPETITIVE_TRAINING.md](docs/COMPETITIVE_TRAINING.md).

## Démarrage sur votre PC

Configuration prévue : Core Ultra 9 185H, 96 Go de RAM, RTX 4060 Laptop 8 Go. Python 3.12 recommandé. Depuis le dossier extrait, dans PowerShell :

Si Python n’est pas encore installé : `winget install --id Python.Python.3.12 -e --source winget`, puis ferme et rouvre PowerShell.

```powershell
./scripts/setup_windows.ps1 -Cuda
./scripts/run_menu.ps1
```

Si Windows affiche `PSSecurityException` ou « fichier non signé », utilise les lanceurs inclus :

```powershell
./scripts/setup_windows.cmd -Cuda
./scripts/run_menu.cmd
```

Alternative dans la session PowerShell courante : `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force`, puis relance les deux commandes `.ps1`. La portée `Process` disparaît quand la fenêtre est fermée.

Le menu lance les tests, le benchmark à 20 seeds, la démo, l'entraînement ou la mesure NPU. Si PowerShell bloque les scripts, utilisez les commandes Python manuelles de [docs/INSTALL.md](docs/INSTALL.md). Pour accélérer le moteur, installez Rust et les outils C++ de Visual Studio, puis relancez `./scripts/setup_windows.ps1 -Cuda -Rust`. Sans extension compilée, un moteur Python de référence assure le fonctionnement.

Sous Linux/WSL :

```bash
bash scripts/setup_linux.sh cuda
source .venv/bin/activate
python -m pytest -q
python -m streamlit run examples/demo.py
```

## Ce qui est livré

| Fonction | Code et état |
|---|---|
| Circuits | Générateurs QFT, Grover, ansatz, random, arithmétique ; imports MQT Bench/QASMBench ; 10 circuits externes figés |
| Devices | Ligne, anneau, grille, heavy-hex, graphe géométrique ; snapshot calibré FakeBrisbane 127 qubits |
| Baselines | SABRE niveau 3, SABRE avec placement sensible au bruit, adaptateur AIRouting local isolé |
| Environnement | Placement injectif puis SWAP masqués ; DAG, permutations et transitions Rust ; oracle Python |
| Politique unique | GATv2 du device + transformer de la fenêtre DAG ; têtes placement/SWAP et critique partagés |
| Apprentissage | Imitation de traces SABRE, PPO masqué, curriculum 5→20→50, baseline SB3 MaskablePPO |
| Recherche | Beam search configurable ; secours par plus court chemin et comptage des recours |
| Bruit | ESP native avec erreurs de portes, lecture et idles ASAP ; simulation Aer et Hellinger |
| Échelle | Encodeur hiérarchique ; vérification structurelle d'un routage 100→127 qubits |
| Intégration | Distillation, ONNX statique, OpenVINO CPU/NPU explicite, plugins Qiskit layout/routing |
| Reproduction | CSV, QPY, snapshots, hashes, tests, CI, Docker, scripts Windows/Linux, démo locale |
| Communication | Brouillon LaTeX et fiche modèle ; publication et campagne scientifique à réaliser |

## Exemple Qiskit

```python
from qiskit import QuantumCircuit, transpile
from noiseroute.device import synthetic

qc = QuantumCircuit(5)
qc.h(0)
qc.cx(0, 4)
qc.cx(0, 2)
qc.measure_all()
device = synthetic("ring", 5, seed=7)
result = transpile(
    qc, target=device.target(), optimization_level=3,
    layout_method="noiseroute", routing_method="noiseroute",
)
print(result)
```

Sans checkpoint, le plugin utilise une heuristique. Pour employer la même politique apprise dans les deux stages :

```powershell
$env:NOISEROUTE_MODEL = 'models/smoke-policy.onnx'
$env:NOISEROUTE_ENGINE = 'onnx'
$env:NOISEROUTE_BEAM = '1'
.venv/Scripts/python.exe examples/plugin.py
```

Les modèles `smoke` acceptent au plus 8 nœuds et 16 arêtes ; le profil laptop, 128 nœuds et 384 arêtes. Un dépassement est une erreur explicite. Utilisez QPY pour préserver les layouts ; le QASM seul ne conserve pas cette information.

## Entraîner et exporter

Pour l'expérience comparative avec acceptation conditionnelle, utiliser le
nouveau parcours ci-dessus. Les commandes ci-dessous restent le curriculum
historique et ne certifient pas les checkpoints :

```bash
python -m noiseroute imitate --config configs/train-laptop.yaml --out runs/laptop-bc
python -m noiseroute train --config configs/train-laptop.yaml --checkpoint runs/laptop-bc/imitation.pt --out runs/laptop-ppo
python -m noiseroute distill --config configs/train-laptop.yaml --checkpoint runs/laptop-ppo/last.pt --episodes 128 --out runs/student.pt
python -m noiseroute export --checkpoint runs/student.pt --out runs/student.onnx
python -m noiseroute latency --model runs/student.onnx --engine openvino --device CPU --out runs/latency-cpu.json
python -m noiseroute latency --model runs/student.onnx --engine openvino --device NPU --out runs/latency-npu.json
```

Le profil laptop utilise CUDA, batch 16 et dimension 64 : point de départ pour 8 Go de VRAM, à mesurer avec `nvidia-smi`. Les 96 Go de RAM ne deviennent pas de la VRAM CUDA. La mention « 70 Go NPU » n'est pas traitée comme une mémoire NPU dédiée. `noiseroute doctor` rapporte les accélérateurs réellement accessibles. Aucun basculement CPU silencieux ne transforme un test NPU échoué en réussite.

Pour reprendre un PPO interrompu sans refaire l’imitation, voir [progression et reprise](docs/TRAINING_RECOVERY.md). Le terminal affiche maintenant les phases et les compteurs sauvegardés.

Après le diagnostic des premiers checkpoints, un [profil court et contrôlé](docs/PPO_DIAGNOSTIC.md) est disponible avec `python scripts/train_diagnostic.py` (option 7 du menu Windows). Il ne démontre pas encore un gain sur SABRE.

## Benchmarks reproductibles

```bash
make bench OUT=runs/bench-001
python -m noiseroute bench --config configs/bench-corpus.yaml --out runs/corpus-001
python -m noiseroute bench --config configs/bench-zeroshot.yaml --out runs/zeroshot-001
make reproduce OUT=runs/reproduction-001
```

`make bench` recalcule les 600 compilations de `results/small20` : 5 circuits × 2 devices × 3 algorithmes × 20 seeds. `bench-corpus` recalcule les 1 800 compilations du corpus externe. `make reproduce` ajoute imitation, PPO court, export, latences CPU et simulation bruitée. Un dossier contenant déjà un CSV n'est pas écrasé. Les durées dépendent de la machine, des versions et de la charge.

Les résultats initiaux de l'heuristique sont **moins bons que SABRE** sur plusieurs cas. Voir [docs/VALIDATION.md](docs/VALIDATION.md), les CSV complets et [docs/PROTOCOL.md](docs/PROTOCOL.md). Les répétitions d'un algorithme déterministe ne constituent pas 20 observations indépendantes. Le petit benchmark contrôle le pipeline, sans prouver sa généralisation.

## Lire ensuite

- [Installation, AIRouting et accélérateurs](docs/INSTALL.md)
- [Architecture et limites](docs/ARCHITECTURE.md)
- [Protocole, métriques et équivalence](docs/PROTOCOL.md)
- [Validations réalisées](docs/VALIDATION.md)
- [Plan des 26 semaines](docs/ROADMAP.md)
- [Sources techniques](docs/SOURCES.md)
- [Fiche des modèles fournis](models/README.md)

Licence MIT pour le code NoiseRoute. Les fichiers QASMBench conservent leur licence et leur NOTICE. Rien n'est publié sur PyPI, Hugging Face ou arXiv par cette archive.
