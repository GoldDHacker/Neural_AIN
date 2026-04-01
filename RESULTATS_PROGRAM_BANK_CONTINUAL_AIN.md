# Résultats — Program Bank + Continual Learning (AIN)

Script : `demo_ain_program_bank_continual.py`

## Objectif
Mettre en place une **Program Bank** qui stocke :
- une **signature de contexte** (être + devenir : mean/std + deltas sur `x_exp=[x,x^2]`)
- l’**invariant** `z`
- le **programme compilé** `forged` (poids + grilles + coeffs)
- un mini-épisode de **replay** (`support`, `query`, `target`)

Puis simuler un entraînement **incrémental** (plusieurs phases avec *distribution shift*) et vérifier que le replay aide à maintenir la performance sur les épisodes stockés.

## Paramétrage (démo)
- Modèle : `AIN(x_dim=6, q_dim=4, z_dim=32, hidden=64)`
- Phases : `4` (0 → 3)
- Steps par phase : `250`
- Replay : `replay_bs=32`, `replay_weight=0.8`
- Ajout à la bank : 1 épisode (index 0) toutes les 10 étapes
- Capacité bank : `1000` (LRU)

## Métrique principale
- `bank_mse` : MSE mesurée en échantillonnant dans la bank (replay) et en évaluant le modèle courant.

## Logs consolidés (run)

### Phase 0
- `EVAL BEFORE` : `bank_mse = nan` ; `bank_size = 0`
- `EVAL AFTER`  : `bank_mse = 0.043577` ; `bank_size = 25`

### Phase 1
- `EVAL BEFORE` : `bank_mse = 0.038166` ; `bank_size = 25`
- `EVAL AFTER`  : `bank_mse = 0.078197` ; `bank_size = 50`

### Phase 2
- `EVAL BEFORE` : `bank_mse = 0.050926` ; `bank_size = 50`
- `EVAL AFTER`  : `bank_mse = 0.100852` ; `bank_size = 75`

### Phase 3
- `EVAL BEFORE` : `bank_mse = 0.066099` ; `bank_size = 75`
- `EVAL AFTER`  : `bank_mse = 0.104503` ; `bank_size = 100`

### Sérialisation
- Save/load de la bank : OK
- `saved=100 loaded=100` (test smoke)

## Lecture
- La bank fonctionne comme un **buffer de replay** : la performance sur les épisodes mémorisés reste contrôlée malgré les shifts, mais elle se dégrade progressivement (phases 1→3), ce qui est cohérent.
- Le replay aide, mais ne “garantit” pas l’absence d’oubli : pour durcir, il faudra typiquement :
  - augmenter la proportion de replay (ou batch mixte)
  - augmenter la capacité ou la diversité des épisodes stockés
  - ajouter une régularisation anti-forgetting plus explicite (distillation sur `z` / EWC / etc.)

## Notes d’implémentation
- Module bank : `program_bank.py`
- Sûreté `torch.load` : utilisation de `weights_only=True` quand disponible (fallback sinon).
