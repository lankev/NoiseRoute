# Entraînement comparatif contre SABRE

Cette mise à jour implémente les cinq changements demandés. Elle ne fournit pas
un modèle déjà supérieur à SABRE. Le premier objectif mesurable est une victoire
sur le corpus de développement à 5 qubits décrit ci-dessous.

## Installer la mise à jour dans le projet existant

Garder le dossier qui contient déjà `.venv` et les résultats `runs`. Arrêter un
entraînement avant de remplacer son code. Depuis ce dossier dans PowerShell :

```powershell
Expand-Archive -LiteralPath "$env:USERPROFILE\Downloads\NoiseRoute-SABRE-update.zip" -DestinationPath . -Force
.\.venv\Scripts\python.exe -m pip install --no-deps -e .
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts/train_competitive.py
```

Le ZIP de mise à jour contient directement `src`, `configs`, `scripts`, etc.
Il ne contient ni `.venv`, ni `runs`, ni modèles de remplacement.
Si `.venv\Scripts\python.exe` est absent, exécuter d'abord
`.\scripts\setup_windows.cmd -Cuda` dans ce dossier. Ne pas copier un environnement
virtuel depuis un autre emplacement. Le menu Windows, option **4**, lance aussi
le nouvel entraînement. Les profils `smoke`, `diagnostic` et l'ancien curriculum
`train-laptop` restent des expériences distinctes, sans promotion automatique.

Chaque lancement crée un dossier `runs/competitive-...` : imitation puis PPO.
La configuration exacte est copiée dans ce dossier. Les mises à jour PPO et la
validation affichent leur progression. Une comparaison comporte 720 compilations
mesurées : 12 cas × 20 seeds × 3 méthodes, plus les échauffements et les contrôles
d'équivalence. Elle peut être longue sur le PC ; aucune durée n'est présupposée.

Pour réutiliser une imitation existante :

```powershell
.\.venv\Scripts\python.exe scripts/train_competitive.py --imitation "runs/bc-20260925-214456/imitation.pt"
```

Le lanceur reprend alors la taille des entrées et l'architecture de ce checkpoint.
Cela peut augmenter la latence par rapport au petit modèle par défaut. Les
paramètres `5e-5` ne réentraînent pas rétroactivement une imitation réutilisée.
Un checkpoint PPO ne peut pas servir de référence d'imitation.

## Objectif d'apprentissage

Dans `configs/train-competitive.yaml` :

```yaml
imitation_learning_rate: 0.00005
ppo_learning_rate: 0.00001
reference_kl_coef: 0.05
reward:
  mode: sabre_delta_log_esp
  seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
```

Les optimisateurs d'imitation et de PPO lisent des clés différentes. L'ancienne
clé partagée `learning_rate` est uniquement un repli de compatibilité pour les
anciens fichiers de configuration. Une clé spécifique a priorité.

La politique de référence est une copie indépendante, en mode évaluation,
`requires_grad=False`, absente de l'optimiseur. Sa distribution est calculée sur
les **mêmes observations et mêmes actions autorisées** que celle de PPO.

La perte est :

\[
L = L_{PPO} + c_v L_{valeur} - c_H H(\pi)
    + 0.05\,D_{KL}(\pi_{PPO}\,\Vert\,\pi_{imitation}).
\]

Le coefficient `0.05` est un point de départ expérimental, pas une valeur optimale
démontrée. C'est une pénalité souple : elle n'impose pas une borne mathématique
absolue sur l'écart. `target_kl: 0.02` reste un arrêt précoce distinct, relatif à
la politique qui a collecté le rollout. Le journal rapporte `approx_kl` et
`imitation_kl` séparément, cette dernière après la mise à jour.

Pour un épisode réussi, la somme des récompenses brutes est exactement :

\[
\Delta\log ESP = \log ESP(C_{politique})
 - \operatorname{médiane}_{s=0..19}\log ESP(C_{SABRE,s}).
\]

Un gain positif signifie une meilleure ESP estimée. Les coûts intermédiaires des
SWAP sont corrigés au dernier pas pour que la somme soit exacte. La compilation
terminale et SABRE utilisent le même device et le niveau d'optimisation 3. Le
placement/routage de l'agent est conservé lors de la compilation terminale.
Le chemin d'inférence Qiskit complet est évalué séparément pendant la validation.
Un épisode raté conserve sa pénalité `-10000`, sans bonus SABRE ; `gamma=1`
est obligatoire pour cette interprétation épisodique. La perte de valeur Huber
limite l'effet des grandes erreurs de retour sur les gradients.

Le score SABRE est indépendant des actions de l'agent : sa soustraction donne
une référence comparative et peut aider le critique, mais ne change pas à elle
seule l'ordre des solutions pour un circuit. Elle ne garantit donc pas une
victoire. Le signal d'apprentissage optimise l'ESP ; les autres critères sont
contrôlés par la sélection des checkpoints.

## Acceptation obligatoire

Le profil par défaut valide tous les 4096 pas et à la fin :

- 5 qubits, lignes/anneaux/grilles ; circuits random et ansatz, profondeur 4 ;
- deux seeds de circuits `200000, 200001`, réservés à la validation ;
- snapshot synthétique fixé avec `device_seed=177` ;
- **20 seeds distincts du transpileur par cas et par méthode** ;
- SABRE niveau 3 et SABRE avec placement sensible au bruit ;
- politique Torch sur CPU, un thread Torch, compilation complète avec chargement
  du modèle ; échauffement exclu du temps, ordre des méthodes alterné ;
- contrôle exact des projecteurs de mesure pour tous les états d'entrée, avec
  permutations initiale/finale et couplage vérifiés, pour chaque compilation.

Le contrôle de projecteurs accepte les optimisations de phases finales qui
préservent toutes les probabilités mesurées. Ce n'est pas une simulation bruitée
ni une simple comparaison sur un état choisi. La version actuelle de ce contrôle
accepte 2 à 8 qubits, avec autant de qubits physiques que logiques.

Pour obtenir `accepted.pt`, il faut simultanément :

1. Toutes les lignes présentes, finies, équivalentes et sans routage de secours.
2. Sur **chaque cas**, aucune régression des médianes en log(ESP), SWAP ajoutés,
   profondeur 2Q et temps de compilation, face à **chacune** des deux baselines.
3. Un gain strict moyen sur l'ensemble des cas pour **chacun des quatre critères**,
   face à chacune des baselines. Une égalité générale ne suffit pas. Seul log(ESP)
   bénéficie d'une tolérance numérique de `1e-9` ; aucune marge de régression
   temporelle n'est accordée.

Les médianes de log(ESP) sont comparées directement, avec un cas = un poids.
Les 20 seeds sont des répétitions de compilation, pas 20 modèles indépendamment
entraînés. Ce critère ne prétend pas battre le meilleur tirage SABRE sur chaque
circuit ni établir une différence statistiquement significative de latence.
Les mesures de temps dépendent aussi de la charge et de l'état thermique du PC.

Fichiers dans `ppo` :

| Fichier | Signification |
|---|---|
| `last.pt`, `stage_0.pt` | Reprise de l'entraînement, sans certification de supériorité |
| `train.json` | Pertes, KL, récompenses et écart log(ESP) des épisodes terminés |
| `validation.json` | Historique des acceptations/refus, avec chemins des rapports |
| `validation-.../raw.csv` | Toutes les mesures et erreurs, y compris les cas perdants |
| `validation-.../report.json` | Critères, gains par cas, raisons de refus et hashes |
| `validation-.../candidate.pt` | Poids d'inférence effectivement évalués, sans état Adam |
| `accepted.pt`, `accepted.json` | Dernier candidat qui a passé le protocole, s'il en existe un |

Un candidat refusé ne remplace pas un précédent candidat accepté. Si tous sont
refusés, `accepted.pt` n'existe pas. Les CSV, QPY, calibrations et hashes restent
disponibles. Il ne faut pas renommer `last.pt` en `accepted.pt`.

L'utilisation répétée de ce corpus pour sélectionner des checkpoints en fait un
**corpus de développement**. Pour une revendication de recherche, prévoir ensuite
un test final indépendant : nouveaux circuits, calibrations, topologies et seeds
d'entraînement, intervalles d'incertitude, validation bruitée et protocole figé.
`research_validated` reste `false`, y compris pour un candidat accepté ici.

## Reprise et validation manuelle

Le checkpoint de reprise inclut les poids gelés d'imitation, leur identité SHA-256
et l'état Adam. Déplacer/supprimer le fichier original d'imitation ne change pas
la référence lors de `--resume`. Il faut conserver la même configuration :

```powershell
.\.venv\Scripts\python.exe -m noiseroute train --config "runs/competitive-XXX/config.yaml" --checkpoint "runs/competitive-XXX/ppo/last.pt" --resume --out "runs/competitive-resume-01"
```

Une reprise démarre un nouvel épisode ; elle n'est pas bit à bit identique à une
exécution ininterrompue. Un candidat d'inférence accepté n'inclut pas l'état de
reprise. Utiliser `last.pt` pour continuer PPO.

La commande autonome de validation ne dispose d'aucune exemption « smoke » :

```powershell
.\.venv\Scripts\python.exe -m noiseroute accept-checkpoint --checkpoint "runs/competitive-XXX/ppo/last.pt" --config configs/train-competitive.yaml --out "runs/validation-manuelle-01"
```

Code de sortie `0` : accepté ; `2` : comparaison terminée mais candidat refusé.
Les erreurs de protocole interrompent l'opération sans promotion. L'entraînement
normal continue après un refus comparatif pour permettre d'évaluer les étapes
suivantes ; son code de sortie `0` signifie qu'il a fini, pas qu'il a battu SABRE.
