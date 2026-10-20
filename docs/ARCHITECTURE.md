# Architecture

## Une politique, deux phases

`RoutingEnv` place les qubits logiques par ordre décroissant de participation aux interactions sur des positions libres. L'action de placement est l'indice du nœud. Ensuite, les portes exécutables du front sont vidées automatiquement et l'action devient un SWAP sur une arête. L'espace contient `Nmax + Emax` actions ; des masques séparent les phases et sélectionnent les actions utiles.

L'observation comprend huit attributs par nœud, l'adjacence pondérée par la calibration, les arêtes, la phase, les masques, une fenêtre de portes restantes avec indication du front, les positions courantes et une partition BFS du device. Les tokens décrivent les portes d'une fenêtre DAG, pas toutes les couches du circuit. Voir `env.py::observe`.

`model.py` partage l'encodeur entre têtes de placement, SWAP et critique. Le device passe dans des couches GATv2 denses ; la fenêtre de portes utilise un transformer. La variante hiérarchique regroupe les nœuds par paquets BFS de 16, encode le graphe agrégé et réinjecte le contexte. C'est une hiérarchie d'encodage, pas un planificateur optimal entre régions.

Les poids ne dépendent pas de la taille du graphe ; les tenseurs exportés ont des buckets fixes. L'attention dense est quadratique. Des indices physiques figurent dans les tokens : invariance aux permutations et zéro-shot doivent être mesurés. Un test à 127 qubits ne démontre pas une montée en charge arbitraire.

## Rust

`rust/src/lib.rs` gère les dépendances, le front, les permutations, le drainage, les masques et le clonage. `core.py` expose un oracle Python de la même API. Les tests rejouent les mêmes transitions dans les deux moteurs.

La normalisation Qiskit, les observations, l'ESP terminal et les rollouts restent en Python. Le pipeline utilise un environnement sériel, sans saturer automatiquement une RTX. `scripts/benchmark_core.py` isole les transitions ; son gain n'est pas un gain garanti pour l'entraînement complet.

## Apprentissage

L'imitation produit un layout SABRE, embarque le circuit et rejoue les SWAP de SabreSwap. Les trajectoires incomplètes/non rejouables sont rejetées et comptées. PPO utilise actions masquées, GAE, clipping, critique, entropie et clipping des gradients. Les terminaisons coupent le bootstrap de GAE. Le profil laptop effectue 600 000 pas sur 5→20→50 qubits.

La baseline SB3 MaskablePPO utilise MultiInputPolicy et le premier palier du curriculum ; le curriculum complet concerne le PPO GATv2/transformer. La distillation minimise une KL sur les observations visitées par le professeur. L'élève (dimension 32, une couche, sans hiérarchie) est plus petit que le professeur laptop, mais plus grand que le minuscule professeur smoke dimension 16.

Les checkpoints incluent poids, configuration et provenance. Les checkpoints PPO récents incluent aussi Adam et le numéro d’épisode. `train --resume --checkpoint ...` restaure le compteur et ignore les stages achevés ; les anciens checkpoints reprennent avec Adam neuf. Un épisode neuf est créé : RNG, DAG en cours et buffer ne sont pas restaurés. `--checkpoint` seul reste une initialisation par les poids. Voir `TRAINING_RECOVERY.md` pour la progression, les sauvegardes atomiques et les limites de reprise. Les longs runs doivent être surveillés et évalués séparément.

`examples/hybridcompute_adapter.py` expose une frontière Python simple pour un dispatcher : chargement OpenVINO explicite CPU/NPU, appel observation→logits, attribut de forme. Le code ou contrat d'API de HybridCompute n'étant pas fourni, aucune intégration à ce logiciel externe n'est déclarée testée.

## Routage et plugin

Le routage glouton ou beam ne renvoie jamais de circuit partiel. Après stagnation, un pas de plus court chemin aide à progresser ; `fallback_steps` le signale, y compris pour la politique apprise. `beam=1` ne fait pas de recherche ; comparez les largeurs supérieures à budget temporel égal.

Qiskit impose des stages distincts. Les entry points layout/routing chargent le même checkpoint dans deux wrappers. Le premier fixe le placement ; le second applique les SWAP et compose `final_layout`. Les identités des bits classiques, les registres, les mesures et la phase globale sont préservés. Le double chargement du modèle est inclus dans le temps de compilation ; son amortissement dans un service persistant reste une optimisation future.

## Périmètre

Paramètres liés, opérations normalisables et mesures terminales sont pris en charge. Reset, délais explicites, contrôle classique, contrôle de flux et mesures intermédiaires suivies d'une opération sur le même qubit sont rejetés. Les lectures terminales sont retardées pour éviter tout SWAP après mesure.

Le graphe « neutral » est un modèle géométrique statique, sans transport d'atomes, blockade, crosstalk ni pertes. Les calibrations synthétiques ne représentent pas un vrai device. Les fake backends sont des snapshots de test, pas des calibrations en temps réel.
