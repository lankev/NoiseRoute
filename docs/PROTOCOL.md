# Protocole expérimental

## Circuits et séparation

Les fichiers normalisés utilisent `rz, sx, x, cx`. Les corpus combinent générateurs locaux et adaptateurs MQT Bench/QASMBench. `benchmarks/corpus/manifest.json` fige dix circuits externes avec hashes, provenance, taille et split. Les sources QASMBench et leur licence sont conservées. L'arithmétique MQT exige une taille paire : la demande 5 produit 6 qubits, déclarés dans le manifeste. Le générateur random MQT dispose de sa propre seed : des seeds de liaison différentes ne garantissent pas des circuits distincts. Les hashes sont dédupliqués.

Fixez des manifestes train/validation/test avant de régler les hyperparamètres. `scripts/check_splits.py` détecte les doublons exacts, sans remplacer la séparation par familles, paramètres et tailles. `train_manifest` permet d'utiliser un manifeste marqué train. Le profil livré entraîne sur random/ansatz et ligne/anneau/grille ; heavy-hex et neutral sont candidats de test hors distribution. Les poids smoke ne sont pas un modèle zéro-shot validé.

## Comparaisons

Même circuit normalisé, target, mesures et niveau Qiskit 3 pour tous. SABRE reçoit au moins 20 seeds. `noise_sabre` fixe un layout heuristique pondérant erreurs d'arêtes, lecture, interactions et distance, puis emploie SabreSwap. Ce layout ne garantit pas une amélioration sur SABRE.

`noiseroute` désigne l'heuristique ; `learned` exige un checkpoint et enregistre son SHA256. AIRouting emploie un worker local isolé ; sa version Qiskit différente doit être mentionnée dans l'analyse. Il ne reçoit pas nécessairement les calibrations comme objectif interne. Ses répétitions ne sont pas déclarées indépendantes sous le contrôle de la seed Qiskit.

Les SWAP sont comptés par callback après le routage et avant la décomposition native, jamais par `CX/3`. Les SWAP présents dans l'entrée sont décomposés à la normalisation. Les optimisations aval peuvent annuler des portes ; rapportez aussi nombre de portes natives et profondeur 2Q.

Les CSV conservent chaque seed, échec, temps, ESP et profondeur. Un timeout reste une erreur. Les circuits trop grands pour un device sont ignorés par compatibilité ; inventoriez ces exclusions avant toute étude à grande échelle.

## ESP et récompense

Après traduction native, le proxy logarithmique est :

\[
\log\mathrm{ESP}=\sum_g\log(1-\epsilon_g)-\sum_q t_{\mathrm{idle},q}\left[T_{1,q}^{-1}+\max(T_{2,q}^{-1}-(2T_{1,q})^{-1},0)\right].
\]

La lecture est une instruction avec erreur. L'idle découle d'un ordonnancement ASAP simplifié, jusqu'à la lecture ou, pour les qubits actifs non mesurés, la fin du circuit. Les ancillas jamais actives sont exclues. Ce modèle ignore les contraintes matérielles complètes, les erreurs corrélées, le crosstalk et la dérive.

Une calibration native manquante provoque une erreur. Les instructions d'erreur égale à 1 sont retirées du snapshot et listées dans `source` ; le graphe doit rester connecté. Le snapshot FakeBrisbane livré exclut `ecr(25, 24)`. Les sens des portes et la traduction native passent par le target Qiskit.

À l'entraînement, le coût provisoire de chaque SWAP est `3 log(1-erreur_arête)`. Le dernier pas corrige ce cumul vers l'ESP du circuit natif à optimisation 0. Avec `gamma=1`, la somme des récompenses égale le log ESP terminal. Avec un autre gamma, cette équivalence disparaît. Le benchmark évalue le niveau 3 : optimisations aval et objectif terminal d'entraînement peuvent donc classer certains routages différemment. Un échec reçoit une pénalité et aucun circuit partiel n'est accepté.

L'ESP est une **estimation heuristique**, pas une fidélité d'état ou une probabilité d'exécution garantie. Les erreurs moyennes publiées sont utilisées telles quelles ; leur produit n'est pas une observation d'événements indépendants porte par porte.

## Bruit simulé

`validate-noise` utilise Aer density_matrix et est limité à 12 qubits physiques. Le dépolarisant est ajusté à l'infidélité moyenne, la lecture est symétrique et les idles ASAP reçoivent des canaux thermiques. On n'ajoute pas une seconde décohérence pendant les portes déjà décrites par leur erreur totale. Les synchronisations des barrières sont prises en compte.

La Hellinger compare la distribution idéale logique aux fréquences classiques du circuit compilé : `(sum_i sqrt(p_i q_i))**2`. Le mapping des mesures doit être conservé. Les 1024 shots de démonstration fluctuent ; une étude doit augmenter les shots, varier les seeds de simulation et estimer l'incertitude. Quelques distributions d'entrée ne démontrent pas l'équivalence unitaire générale.

## Équivalence

À chaque sortie de `route`, la trace vérifie : toutes les portes exactement une fois, dépendances respectées, SWAP sur le graphe, mapping cohérent, permutation finale correcte. Sur petits circuits, les tests comparent les unitaires en corrigeant permutations initiale/finale. Les tests du plugin prennent en compte le layout Qiskit et les bits classiques. La vérification structurelle passe à 127 qubits sans matrice exponentielle.

## Statistiques et temps

Le résumé agrège les seeds par médiane pour chaque couple circuit/device puis compare les écarts appariés de log ESP à `noise_sabre`. Le bootstrap de 2000 rééchantillonnages est exploratoire : les corrélations entre familles et topologies nécessitent une analyse stratifiée pour un article. Les médianes globales de SWAP/temps ne sont pas des comparaisons appariées complètes. Une politique gloutonne déterministe peut reproduire la même sortie pour 20 seeds.

Séparez chargement du modèle, décision réseau, construction de l'observation, routage et compilation complète. `latency` fait 20 warmups puis N appels d'une même observation 5 qubits, avec un routage d'exemple séparé. Cela ne certifie pas `<10 ms` sur tous les circuits ou sur NPU. Fixez CPU, threads, pilote, alimentation et température du laptop ; rapportez tests à froid/à chaud, tailles de graphes et nombre de décisions.
