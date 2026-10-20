# Correctif du hash QASM sous Windows

## Cause

Le test `test_dataset_roundtrip_and_tamper` échouait avant toute modification du circuit. L'ancien `save_circuit` calculait le SHA256 du texte UTF-8 avec des fins de ligne LF, puis appelait `Path.write_text` avec les paramètres par défaut. Windows traduisait les LF en CRLF sur disque. `load_manifest`, qui contrôle correctement les octets réellement enregistrés, détectait donc une différence.

## Correction

Le circuit est encodé une fois en UTF-8. Le SHA256 et `Path.write_bytes` utilisent exactement ce même contenu. Le contrôle d'intégrité reste strict : aucune erreur n'est ignorée et aucune modification du fichier n'est normalisée à la lecture.

Le test de lecture aller-retour et de détection d'altération est maintenant paramétré pour LF et CRLF. Une simulation des écritures Windows permet de reproduire le problème même sous Linux. Un job Windows ciblé est ajouté à la CI ; sa configuration ne constitue pas une exécution effective sur Windows. Les attributs Git préservent aussi les octets des QASM lors des checkouts.

## Appliquer le petit correctif

Extraire `NoiseRoute-correctif-Windows.zip` directement dans le dossier du projet qui contient `pyproject.toml`, `src`, `tests` et `.venv`, et accepter le remplacement des deux fichiers. Aucun environnement, checkpoint ou résultat n'est inclus dans ce petit ZIP.

Depuis ce dossier, lancer :

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_io_export.py -k dataset_roundtrip -q
.\.venv\Scripts\python.exe -m pytest -q
```

L'installation editable créée par le setup utilise directement le fichier corrigé dans `src`. Il n'est pas nécessaire de réinstaller PyTorch ou les autres dépendances.

Le QASM défectueux montré dans la capture est un fichier temporaire de pytest : le prochain test le régénère. Si d'autres corpus ont été générés avec l'ancienne version sous Windows, les régénérer dans un nouveau dossier avec leur commande et leur configuration initiales ; ne pas réécrire aveuglément les hashes de manifestes pour masquer une différence.

Les deux avertissements ONNX n'expliquent pas ce bug. Un test Rust ignoré est possible quand l'extension facultative n'est pas installée ; `pytest -rs` indique la raison exacte des tests ignorés.
