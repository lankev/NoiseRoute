# Validation de la livraison 0.1.0

Mesures réalisées le 25 septembre 2026 dans un environnement Linux CPU. Elles ne proviennent pas du Core Ultra 9 / RTX 4060 / NPU de l'utilisateur. Versions et limites dans `results/smoke/doctor.json`. Le backend Rust a été compilé et utilisé.

## Tests et intégration

- Livraison initiale : 44 tests pytest passaient sous Linux, couvrant parité Rust/Python, masques, dépendances et permutations, équivalence unitaire de cinq familles sur quatre topologies, plugins aux niveaux 0–3, conservation des mesures/registres, budget d'échec, métriques, GAE, gradients, imitation/PPO, import QASM, hashes et export ONNX.
- Deux avertissements PyTorch signalent l'exporteur ONNX historique ; il fonctionne dans la version testée, mais devra être migré avant son retrait.
- Ruff et construction de la roue Python réussis. Les entry points ont été testés depuis l'installation locale.
- La démo Streamlit charge et affiche ses 12 métriques après compilation. Les appels natifs sont isolés dans un worker : un plantage Qiskit lors des reruns dans un thread a été constaté puis contourné ainsi.
- Les scripts Windows, le NPU réel, la CI distante et l'image Docker n'ont pas été exécutés sur le matériel cible.

### Correctif Windows du 25 septembre 2026

Le retour utilisateur a révélé un défaut d'écriture du QASM sous Windows : conversion automatique LF→CRLF après calcul du hash. Le test de lecture et de détection d'altération est désormais paramétré pour les deux conventions. Avant correction : un cas passe, le cas CRLF échoue avec la même erreur que la capture. Après écriture des mêmes octets UTF-8 que ceux hachés : les deux cas passent, y compris le rejet d'un fichier altéré. La validation ciblée est conservée dans `results/smoke/windows-hash-fix.txt`. Elle simule les écritures Windows sous Linux ; la suite complète historique n'a pas été relancée pour ce correctif limité. Une tâche CI Windows est configurée mais n'a pas été exécutée à distance. Voir [WINDOWS_HASH_FIX.md](WINDOWS_HASH_FIX.md).

## Campagnes à 20 seeds

Les nombres ci-dessous concernent l'heuristique NoiseRoute, pas le modèle appris. `results/small20/raw.csv` contient 600 lignes sans erreur. `results/corpus20/raw.csv` contient 1 800 lignes sans erreur : dix circuits MQT Bench/QASMBench, trois devices, trois algorithmes, vingt seeds.

| Suite | Algorithme | SWAP médian global | Profondeur 2Q médiane | Compilation médiane, ms |
|---|---|---:|---:|---:|
| Built-in 5 qubits | SABRE | 11 | 36,5 | 18,82 |
| Built-in 5 qubits | SABRE + layout bruit | 14 | 36 | 17,33 |
| Built-in 5 qubits | NoiseRoute heuristique | 12 | 44 | 29,05 |
| Corpus externe | SABRE | 2 | 11 | 18,66 |
| Corpus externe | SABRE + layout bruit | 8 | 11,5 | 17,73 |
| Corpus externe | NoiseRoute heuristique | 4 | 15 | 25,91 |

Comparaison appariée en ESP, après médiane des seeds par circuit/device :

| Suite | Paires | NoiseRoute gagne / égalité / perd face à noise-SABRE | Delta log ESP médian |
|---|---:|---|---:|
| Built-in | 10 | 1 / 2 / 7 | −0,08959 |
| Externe | 30 | 2 / 13 / 15 | −0,00345 |

Les résultats ne démontrent pas une amélioration. Une médiane globale plus basse en SWAP ne signifie pas une meilleure fidélité appariée. Les intervalles exploratoires figurent dans les `summary.json`; les corrélations entre cas limitent leur portée.

## AIRouting local et politique smoke

`results/ai-smoke` contient six compilations sans erreur : random et QFT, cinq qubits, une ligne, SABRE/noise-SABRE/AIRouting, une seed. Le worker utilise Qiskit 2.3.1, IBM transpiler 0.18.0 et NetworkX 2.8.5. Ses deux compilations AIRouting ont pris environ 38,3 s et 11,4 s de bout en bout, dont 34,3 s et 7,1 s dans `pm.run`. Ce démarrage froid et ces deux exemples ne forment pas une campagne comparative complète.

`results/learned-smoke` contient six compilations sans erreur. Le modèle ONNX brièvement entraîné utilise cinq puis neuf pas de secours et donne des ESP de 0,4963 et 0,1046 sur random et QFT. Ces résultats sont inférieurs aux baselines de ces exemples ; les poids sont exclusivement des artefacts de validation technique.

## Apprentissage, distillation et export

L'imitation a conservé 32 décisions issues de huit épisodes, zéro épisode rejeté, sur deux epochs. PPO a exécuté 128 pas sur les paliers 3 puis 5 qubits, avec 23 épisodes terminés dans ces rollouts et aucun échec enregistré. Cela ne prouve ni convergence ni qualité de routage. La baseline SB3 a exécuté 64 pas. La distillation a utilisé 515 observations de deux épisodes.

L'export ONNX a été vérifié sur des observations de placement/routage : écart absolu maximal `3,28e-7` pour la politique et `1,49e-7` pour l'élève. Les checkpoints, ONNX et JSON sont dans `models`, les logs dans `results/smoke`.

| Moteur CPU | Chargement, ms | Décision médiane, ms | Décision p95, ms | Routage d'exemple, ms |
|---|---:|---:|---:|---:|
| ONNX Runtime | 53,79 | 0,102 | 0,465 | 11,71 |
| OpenVINO | 352,85 | 0,406 | 0,642 | 16,61 |

Cent mesures après vingt warmups, même observation de placement à cinq qubits, bucket 8/16/16. La construction des features est exclue des décisions mais incluse dans le routage. Chaque routage d'exemple utilise un pas de secours. Une décision sous 10 ms ne démontre pas une compilation complète sous 10 ms. Aucun NPU n'était accessible : `doctor` ne liste que CPU pour OpenVINO.

## Échelle et bruit

Un ansatz peu profond de 100 qubits est routé sur FakeBrisbane 127 qubits : 46 SWAP, zéro secours, environ 328 ms et vérification structurelle réussie. Ce test concerne l'heuristique. Une compilation native d'un petit circuit utilise quatre ECR calibrés et conserve cinq mesures sur le target à 127 qubits. Voir `scale127.json` et `fake-native.json`.

Le microbenchmark isolé des transitions effectue 500 actions : médiane 16,39 ms en Python contre 0,444 ms en Rust. Il exclut observations, réseau et compilation et ne mesure pas l'entraînement complet.

Une simulation Aer à 1024 shots, cinq qubits, donne des Hellinger d'environ 0,9531 / 0,9281 / 0,9269 pour SABRE / noise-SABRE / NoiseRoute heuristique. Ce contrôle unique ne valide pas statistiquement l'ESP. Les données exactes sont dans `results/smoke/noise.json`.

## Ce qui reste à établir

Entraînement long sur RTX, campagne AI complète, zéro-shot appris, robustesse aux calibrations, compression sans perte de qualité, performances NPU et tests Windows réels, mesures QPU et résultats de publication. Aucun niveau bronze/argent/or n'est revendiqué. Les sources et commandes de reproduction sont dans les autres documents.


## Correctif de reprise du PPO (25 septembre 2026)

Le test ciblé `pytest tests/test_training_recovery.py tests/test_learning.py -q`
vérifie l’entraînement court, les reprises depuis des checkpoints PPO anciens et
nouveaux, les compteurs d’optimiseur, les frontières de stages et les interruptions
pendant une mise à jour ou une sauvegarde. Le résultat brut est conservé dans
`results/smoke/training-recovery-tests.txt`. Exécution locale sur CPU Linux avec
PyTorch 2.8.0+cpu et Qiskit 2.5.2. Le scénario de relance emploie aussi des chemins
contenant espaces et parenthèses, sans constituer un test Windows natif.

Le correctif transfère les observations une fois par rollout et ajoute les temps
de collecte et d’optimisation. Aucun gain de vitesse RTX n’est revendiqué sans
mesure sur le matériel cible. La reprise repart d’un épisode neuf ; elle ne
restaure pas exactement RNG, environnement et buffer. Le projet garde les
résultats de recherche antérieurs inchangés.


## Diagnostic PPO du 26 septembre 2026

Les logs et checkpoints utilisateurs ont révélé une couverture d’imitation
corrélée à la taille, aucune démonstration acceptée à 50 qubits dans le run
initial, puis 32/45 et 20/22 épisodes échoués aux stages 20/50. Les réglages
expérimentaux et limites du correctif sont décrits dans `PPO_DIAGNOSTIC.md`.
Les validations ciblées se trouvent dans `results/smoke/training-quality-tests.txt`
et `training-quality-workflow.txt`. Les réglages ne constituent pas une
validation de supériorité sur SABRE.
