# De cette livraison à un résultat de recherche

Le code couvre les étapes du plan ; les expériences longues, la qualité scientifique et la publication demandent encore du travail mesuré. Aucun niveau bronze/argent/or n'est déclaré atteint.

| Phase | Disponible | Validation ou travail restant |
|---|---|---|
| 0 · semaine 1 | Package, CI, corpus, snapshots, métriques | Étendre et figer les splits ; choisir la distribution d'usage |
| 1 · semaines 2–3 | Deux campagnes à 20 seeds, SABRE et noise-SABRE, adaptateur AIRouting | Campagne complète AIRouting et comparaison à budget identique |
| 2 · semaines 3–6 | Moteur Rust/Python, Gym, masques, tests, curriculum | Profiler tout l'environnement, vectoriser si nécessaire |
| 3 · semaines 7–10 | SB3, GATv2 + transformer, imitation, PPO, beam | Entraînement long, sélection sur validation, plusieurs seeds d'entraînement |
| 4 · semaines 11–14 | ESP et Aer/Hellinger | Calibrations multiples, sensibilité du proxy, validation QPU si accès autorisé |
| 5 · semaines 15–18 | Hiérarchie d'encodage, test structurel 127, distillation, export | Politique réellement entraînée à l'échelle, tests zéro-shot, mesures laptop/NPU |
| 6 · semaines 19–22 | Plugins, pip local, Docker, documentation, démo | Installation Windows réelle, CI distante, publication PyPI/HF après revue |
| 7 · semaines 23–26 | Brouillon d'article et données préliminaires | Compléter résultats/ablations, atelier, arXiv, contribution upstream

## Premier parcours sur votre machine

1. Installer, exécuter `doctor`, tests et démo. Conserver versions et inventaire matériel.
2. Refaire `make bench`, puis le corpus externe. Comparer les médianes et les erreurs aux résultats livrés.
3. Installer AIRouting séparément, vérifier un smoke puis une campagne à 20 répétitions, en conservant les échecs.
4. Lancer imitation puis PPO laptop. Contrôler mémoire GPU, débit, retour, entropie, taux de succès et recours au secours. Commencer par un YAML réduit avant les 600 000 pas.
5. Comparer le checkpoint sur validation, sans ajuster sur le test. Répéter plusieurs seeds d'entraînement.
6. Figer le modèle puis lancer test et zéro-shot. Tester aussi les cas défavorables.
7. Distiller/exporter et mesurer qualité après compression, CPU puis NPU. Inclure le routage complet.

## Critères à préenregistrer

Bronze : équivalence et validité à 100 % sur le périmètre annoncé ; nombre de SWAP apparié au moins comparable à SABRE sur petits circuits, avec marge d'équivalence définie avant l'étude.

Argent : delta de log ESP positif sur une majorité de couples circuit/device du test, intervalle d'incertitude compatible avec une amélioration réelle, profondeur/temps publiés et proximité d'AIRouting quantifiée à budget comparable.

Or : argent plus zéro-shot sur topologies et calibrations non vues, budget `<10 ms` défini précisément (décision ou compilation complète), mesure sur le vrai matériel et plugin publié avec reproduction.

Ces seuils ne sont pas ajustés a posteriori aux premiers résultats. Le benchmark actuel montre que l'heuristique seule ne suffit pas.

## Ablations nécessaires

Comparer : placement appris seul, routage appris seul, politique jointe ; GAT sans transformer, sans hiérarchie ; récompense SWAP vs bruit ; sans imitation ; glouton vs beam à budgets de temps égaux ; avant/après distillation ; calibrations vues vs déplacées ; avec et sans mécanisme de secours. Publier la proportion de décisions de secours pour éviter d'attribuer leur qualité au réseau.

Les tâches de visibilité sont préparatoires. Aucun article, package ou modèle n'est publié et aucun chercheur n'est contacté par les scripts. Les quotas QPU, conditions et accès doivent être vérifiés au moment de préparer des jobs réels ; aucun quota gratuit n'est promis.
