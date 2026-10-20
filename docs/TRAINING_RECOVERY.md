# Progression et reprise du PPO

Une trace `KeyboardInterrupt` indique une interruption, généralement Ctrl+C.
Elle ne démontre ni erreur CUDA ni épuisement de mémoire. Le temps écoulé
entre lancement et interruption ne permet pas d'isoler les durées de
l'imitation et du PPO.

Le PPO initial enregistrait déjà `last.pt` après chaque mise à jour, mais sans
affichage de progression. Son option `--checkpoint` rechargeait uniquement les
poids et recommençait le curriculum.

## Appliquer le correctif Windows

Extraire `NoiseRoute-correctif-entrainement.zip` directement dans la racine du
projet, celle qui contient `src`, `scripts`, `tests` et `pyproject.toml`.
Accepter le remplacement des fichiers. Le ZIP ne contient aucun environnement,
modèle, résultat ou checkpoint personnel. L'installation éditable existante
prendra les modifications en compte au prochain lancement Python.

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_training_recovery.py tests/test_learning.py -q
.\.venv\Scripts\python.exe scripts/resume_training.py --run "runs/ppo-20260925-214456" --imitation "runs/bc-20260925-214456/imitation.pt"
```

Le script choisit `last.pt`, sinon le dernier checkpoint de stage disponible,
sinon le checkpoint d'imitation explicitement fourni. Une erreur de lecture
d'un checkpoint présent est signalée : elle ne provoque pas une reprise
silencieuse depuis un modèle plus ancien. Le script écrit dans un nouveau
dossier `runs/ppo-resume-<date-heure>` et affiche le chemin choisi.

Pour les reprises suivantes, remplacer `--run` par le dossier effectivement
affiché lors de cette relance. L'option 4 du menu démarre toujours un nouvel
entraînement avec imitation ; employer le script ci-dessus pour reprendre.

## Ce qui est restauré

- `--checkpoint` seul : initialisation par les poids, compteur à zéro.
- `--checkpoint ... --resume` : poids et compteur PPO, stages terminés ignorés.
- Checkpoints du correctif : état Adam et numéro d'épisode également restaurés.
- Anciens checkpoints : Adam neuf et numéro d'épisode dérivé du compteur de pas.

La configuration et le curriculum doivent rester identiques pour `--resume`.
Seuls le périphérique PyTorch, son nombre de threads et la fréquence d'affichage
peuvent changer. La forme des observations est vérifiée dans tous les cas.

La reprise démarre un nouvel épisode. Elle ne restaure ni le DAG en cours,
ni le buffer, ni l'état des générateurs aléatoires. Elle n'est donc pas une
continuation identique bit à bit. Cette limite figure dans le terminal.

## Suivre l'avancement

Le terminal affiche le démarrage, les stages, les phases de collecte et
d'optimisation, puis chaque sauvegarde. Dans une phase, la boucle affiche un
point environ toutes les 10 secondes (`progress_interval_seconds` dans YAML).
Une opération individuelle longue peut retarder cet affichage.

Le champ `step` compte les interactions collectées ; `saved_step` indique
le compteur du dernier checkpoint complet. `steps_per_s` est le débit moyen
de la session après les mises à jour terminées. Ce n'est pas une estimation
garantie de la durée des prochains stages, qui contiennent plus de qubits.

`status.json` expose la dernière phase ; `train.json` contient les mises à jour
de la session, leur récompense, leur perte et les durées collecte/optimisation.

Le buffer d'observations est maintenant transféré une seule fois par rollout
sur le périphérique, puis indexé pour les minibatches. Cela évite des copies
répétées ; le gain réel doit être mesuré sur la RTX Windows. L'environnement
reste CPU-bound et le modèle utilise toujours les dimensions rembourrées du
profil laptop (128 nœuds), même au stage 5 qubits.

## Interrompre

Un Ctrl+C conserve le dernier checkpoint complet et affiche son chemin.
Les modifications d'une mise à jour inachevée sont abandonnées. Avant la
première sauvegarde PPO, le point de reprise reste le checkpoint d'imitation.
L'écriture d'un checkpoint se fait dans un fichier temporaire, fermé puis
remplacé atomiquement ; une interruption pendant la sérialisation n'écrase
donc pas l'ancien `last.pt`.

Les tests automatisés du correctif vérifient la reprise de checkpoints anciens
et nouveaux, les frontières de stages, les compteurs Adam, l'interruption et
la préservation du fichier précédent. Les essais locaux sont exécutés sur CPU
Linux, pas sur la RTX ni sous Windows.
