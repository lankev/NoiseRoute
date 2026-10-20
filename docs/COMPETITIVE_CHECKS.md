# Vérification de la mise à jour comparative — 26 septembre 2026

Vérifications réalisées dans l'environnement de développement **Linux CPU**,
Python 3.12.14, PyTorch 2.8.0+cpu, Qiskit 2.5.2, NumPy 2.5.3. Ces mesures ne sont
pas celles de la RTX 4060, du CPU ni du NPU de l'utilisateur.

- Suite complète : **69 tests réussis, 2 ignorés**, 12,68 secondes.
- Tests ignorés : extension Rust `noiseroute_core` et export/runtime ONNX,
  dépendances absentes dans cet environnement de vérification.
- Ruff : vérification et formatage conformes.
- Construction du wheel Python : réussie.
- Une tâche CI Windows est ajoutée pour les tests d'apprentissage et de reprise ;
  elle n'a pas été exécutée depuis cet environnement Linux.

Le test intégré a effectué une imitation très courte, puis 64 pas PPO avec les
deux learning rates demandés, la référence gelée et la récompense comparative.
La validation finale couvre **12 cas × 20 seeds × 3 méthodes = 720 lignes** :

- 0 erreur de compilation ou d'équivalence ;
- 0 mesure de SWAP manquante ;
- 160 lignes du candidat utilisent le routage de secours ;
- des régressions sont mesurées face aux baselines ;
- **candidat refusé, aucun `accepted.pt` produit**.

Ce test vérifie le fonctionnement et le refus d'un modèle insuffisant. Il ne
représente pas un entraînement visant des résultats de recherche. La configuration
livrée pour le PC utilise une imitation et un entraînement plus longs ; ses
résultats restent à mesurer.

Les nouveaux tests couvrent notamment la séparation des taux, les gradients KL
sur actions masquées, le maintien de la référence après reprise et suppression
du fichier d'imitation initial, la somme exacte des récompenses comparatives,
l'absence de bonus pour un épisode raté, les seeds manquantes ou dupliquées,
les régressions, les mesures manquantes, l'équivalence avec permutations, et le
refus de promotion si le candidat ou l'écriture du rapport est altéré.

Dans l'archive complète, `results/competitive-check/` conserve les journaux,
la configuration de vérification, les CSV, les rapports et les artefacts du test.
Les poids de ce dossier sont des **poids de test refusés**.
