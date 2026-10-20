# Diagnostic du premier entraînement PPO

Les logs fournis enregistrent 8 121 épisodes réussis / 0 échoué au stage 5 qubits,
13 / 32 à 20 qubits et 2 / 20 à 50 qubits. Les épisodes encore ouverts aux
frontières de stages ne sont pas inclus dans ces comptes. Le compteur atteint
bien 600 000 interactions ; il ne représente pas 600 000 circuits réussis.

Le rejeu de l'imitation initiale reproduit exactement les 838 exemples et
74 rejets. Le choix simultané `episode % 3` pour taille et topologie associait
5/ligne, 20/anneau et 50/grille. Aucun des 42 épisodes 50/grille n'était accepté :
le rejeu rencontrait un SWAP masqué. Le correctif ne supprime pas les masques
pour forcer l'acceptation d'une démonstration.

Les 32 échecs du stage 20 coïncident avec 32 pics de loss > 10 000, avec un maximum
proche de 1,99 million. L'environnement impose -10 000 à un échec et le critique
utilisait une MSE. Cette échelle explique des cibles et gradients potentiellement
très grands ; les logs ne séparent pas les composantes de loss et ne prouvent pas
que c'est l'unique cause de la dégradation.

## Corrections livrées

- L'imitation visite désormais le produit cartésien taille/topologie/famille.
  `coverage.json` enregistre tentatives, épisodes acceptés et nombre d'actions
  par groupe. `require_demo_coverage: true` empêche d'entraîner si un groupe
  configuré n'a aucune démonstration complète.
- Le PPO accepte une échelle de récompense constante et positive. Le nouveau
  profil utilise 0,01, conserve les récompenses brutes dans `mean_reward` et
  expose `mean_training_reward`. La pénalité brute reste -10 000 ; elle n'est
  ni supprimée ni rendue positive. Le score ESP du benchmark ne change pas.
- Une loss Huber pour le critique est disponible ; le profil court l'active.
  Les learning rates sont désormais séparés : imitation `5e-5`, PPO `1e-5`.
  L'entropie passe à 0,001 et `target_kl: 0.02` interrompt
  les mises à jour du rollout lorsque la divergence approchée dépasse 0,03.
- Les logs séparent acteur, critique, entropie, KL approchée, fraction clippée
  et nombre réel de mises à jour d'optimiseur. Des valeurs non finies arrêtent
  l'entraînement avant d'écraser la sauvegarde précédente.
- Un contrôle optionnel examine les derniers épisodes terminés du stage. Dans
  le profil court, au moins 30 épisodes sont nécessaires, parmi les 100 derniers.
  Plus de 25 % d'échecs entraîne un arrêt ; la fin du stage exige au moins 90 %
  de réussites. Un nombre insuffisant d'épisodes bloque aussi la transition.
  Le checkpoint est sauvegardé avant cet arrêt, avec le résultat du contrôle.

Ces paramètres constituent une proposition expérimentale de stabilisation, pas
une preuve d'amélioration. Le contrôle porte sur la complétion des épisodes
stochastiques d'entraînement, pas sur l'ESP d'une politique gloutonne en validation.
Un contrôle réussi n'autorise donc pas à revendiquer une supériorité sur SABRE.

Pour préserver la reproduction, les options de stabilité sont désactivées par
défaut quand elles ne figurent pas dans la configuration. `train-laptop.yaml`
reste le profil historique. La correction d'échantillonnage concerne les
nouvelles imitations. Les anciens checkpoints restent lisibles et inchangés.

## Essai Windows court

Extraire le contenu du correctif dans la racine du projet et accepter les
remplacements. Depuis ce dossier :

```powershell
.\.venv\Scripts\python.exe scripts/train_diagnostic.py
```

L'option **7 — Diagnostic RTX** du menu appelle la même commande. Un nouveau
répertoire `runs/diagnostic-<date-heure>` contient 72 tentatives de démonstration
à 5 qubits, cinq époques d'imitation, puis seulement **4 096 pas de PPO à 5 qubits**.
Aucun passage automatique à 20 ou 50 qubits n'est prévu dans ce profil.

Fichiers à examiner après cet essai :

- `bc/coverage.json` et `bc/imitation.pt` ;
- `ppo/train.json`, `ppo/status.json` et `ppo/last.pt`.

Le script peut utiliser `--checkpoint chemin/imitation.pt` pour démarrer depuis
des poids existants, en sautant la nouvelle imitation. Dans ce cas, les anciens
biais du jeu de démonstrations ne sont pas corrigés. Les poids sont une
initialisation ; cet essai ne reprend pas les 600 000 pas précédents.

Ne pas utiliser `--resume` pour changer l'échelle des récompenses ou le curriculum :
une telle modification définit un nouvel essai et un optimiseur neuf. Le contrôle
refuse de reprendre directement un checkpoint déjà arrêté pour insuffisance de
qualité ; analyser le log avant de définir une nouvelle expérience.

## Validation du correctif

Les tests ciblés couvrent l'indépendance des dimensions d'imitation, la détection
d'une couverture vide, la conservation de l'ordre des récompenses, les gradients
Huber, l'arrêt lors d'échecs répétés, les sauvegardes, la reprise et le franchissement
normal des stages. Voir `results/smoke/training-quality-tests.txt`.

Un essai d'intégration CPU réduit utilise les 72 tentatives de démonstration du
profil, une seule époque d'imitation et 256 pas PPO. Son log est dans
`results/smoke/training-quality-workflow.txt`. Il vérifie le fonctionnement, pas
la qualité du routage après un entraînement long ni les performances Windows/CUDA.
