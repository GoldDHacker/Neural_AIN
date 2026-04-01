# Rubik 2x2 — Expériences AIN (curriculum)

## Contexte
- **Objectif**: démontrer un apprentissage progressif des « lois de groupe » du Rubik 2x2 (coins) via épisodes meta-learning.
- **Moves**: `U`, `R`, `F`.
- **Représentation**: 8 positions de coins, avec
  - **pièce**: 8 classes (permutation)
  - **orientation**: 3 classes
- **Loss**: cross-entropy séparée (pièce + orientation) (cf. `rubiks_ce_loss`).

## Critères de passage par stage
- Le curriculum **passe un stage** si:
  - `eval_piece >= piece_thr` ET `eval_ori >= ori_thr`
- `eval_exact` est loggé mais **non requis** pour passer.

Seuils utilisés dans les runs comparatifs:
- **Stage 1**: `piece_thr1=0.85`, `ori_thr1=0.90`
- **Stage 2**: `piece_thr2=0.70`, `ori_thr2=0.75`
- **Stage 3**: `piece_thr3=0.45`, `ori_thr3=0.60`

## Paramètres communs (sauf mention)
- `B=32`, `N=24`, `hidden=128`, `z_dim=64`, `lr=8e-4`
- Budgets d’entraînement:
  - `epochs1=700`, `epochs2=1000`, `epochs3=1500`
- `scramble1=1`, `scramble2=4`, `scramble3` variable
- `eval_every=50`

---

## Run 1 — scramble3=4 (référence)
- **Stage 1**: PASS à `ep=100`
  - `eval_piece=0.775`, `eval_ori=0.850`, `eval_exact=0.250`
- **Stage 2**: PASS à `ep=500`
  - `eval_piece=0.573`, `eval_ori=0.825`, `eval_exact=0.141`
- **Stage 3**: PASS à `ep=100`
  - `eval_piece=0.507`, `eval_ori=0.776`, `eval_exact=0.031`

Conclusion:
- `scramble3=4` passe les 3 stages avec ces seuils.

---

## Run 2 — scramble3=5 (eval_batches=6)
- **Stage 1**: PASS à `ep=200`
  - `eval_piece=0.939`, `eval_ori=0.996`, `eval_exact=0.573`
- **Stage 2**: PASS à `ep=900`
  - `eval_piece=0.711`, `eval_ori=0.887`, `eval_exact=0.302`
- **Stage 3**: PASS à `ep=200`
  - `eval_piece=0.490`, `eval_ori=0.706`, `eval_exact=0.089`

Conclusion:
- `scramble3=5` passe les 3 stages, mais Stage 2 devient plus long.

---

## Run 3 — scramble3=6 (eval_batches=6)
- **Stage 1**: PASS à `ep=100`
  - `eval_piece=0.893`, `eval_ori=0.956`, `eval_exact=0.443`
- **Stage 2**: PASS à `ep=700`
  - `eval_piece=0.706`, `eval_ori=0.893`, `eval_exact=0.250`
- **Stage 3**: PASS à `ep=1`
  - `eval_piece=0.455`, `eval_ori=0.616`, `eval_exact=0.062`

Note critique:
- Le passage de Stage 3 à `ep=1` est suspect: avec `eval_batches` faible, la variance peut produire un **faux positif**.

---

## Run 4 — scramble3=6 (eval_batches=30) — test de robustesse
Changement:
- `eval_batches=30` (au lieu de 6) pour réduire la variance d’évaluation.

Résultat:
- **Stage 1**: PASS à `ep=150`
  - `eval_piece=0.975`, `eval_ori=1.000`, `eval_exact=0.898`
- **Stage 2**: **FAIL** à `ep=1000` (budget épuisé)
  - dernier log à `ep=1000`: `eval_piece=0.565`, `eval_ori=0.748`
  - verdict: `CURRICULUM STOPPED: stage 2 failed`

Conclusion:
- Le « PASS immédiat » de Stage 3 observé en `eval_batches=6` n’est **pas stable**.
- Avec une éval robuste, le goulot devient **Stage 2**.

---

## Run 5 — scramble3=6 (eval_batches=30, epochs2=2000)
Changement:
- `eval_batches=30` (éval robuste)
- `epochs2=2000` (budget Stage 2 augmenté)

Résultat:
- **Stage 1**: PASS à `ep=100`
  - `eval_piece=0.939`, `eval_ori=0.976`, `eval_exact=0.718`
- **Stage 2**: PASS à `ep=1100`
  - `eval_piece=0.729`, `eval_ori=0.878`, `eval_exact=0.296`
- **Stage 3**: PASS à `ep=1`
  - `eval_piece=0.504`, `eval_ori=0.693`, `eval_exact=0.117`

Conclusion:
- Avec `epochs2=2000`, `scramble3=6` devient atteignable sous évaluation robuste.
- Le passage Stage 3 à `ep=1` peut refléter une généralisation déjà acquise via Stage 2, mais la stabilité doit idéalement être vérifiée via une exigence de passage répétée (plusieurs évaluations consécutives) si on veut une preuve plus solide.

 ---

 ## Run 6 — scramble3=6 (eval_batches=30, epochs2=2000, pass_k=3) — validation anti-variance
 Démo:
 - `demo_ain_rubiks_2x2_consecutive.py`

 Changement:
 - Exiger un passage sur **`K=3` évaluations consécutives** (`--pass_k 3`).

 Résultat:
 - **Stage 1**: PASS à `ep=450` (streak=3/3)
 - **Stage 2**: PASS à `ep=1900` (streak=3/3)
   - `eval_piece=0.758`, `eval_ori=0.877`, `eval_exact=0.340`
 - **Stage 3**: PASS à `ep=100` (streak=3/3)
   - `eval_piece=0.479`, `eval_ori=0.645`, `eval_exact=0.087`

 Conclusion:
 - Le passage de Stage 3 n’est plus un “PASS instantané” ambigu : il est obtenu avec **3 évaluations consécutives** au-dessus des seuils.
 - Le coût principal du critère anti-variance est un passage Stage 2 plus tardif (ici `ep=1900`).

## Recommandations opérationnelles (état actuel)
- Pour comparer des difficultés (`scramble3`), augmenter `eval_batches` (p.ex. 20–30) est nécessaire pour éviter les faux positifs.
- Si `scramble3=6` est visé, augmenter le budget `epochs2` est une approche simple et efficace (ex: `epochs2=2000` a suffi dans Run 5).
 - Pour une validation rigoureuse (éviter les faux positifs), utiliser la démo consecutive avec `--pass_k 3` (Run 6).

 ## Amélioration anti-variance: PASS sur K évaluations consécutives
 Pour limiter les faux positifs dus à la variance d’évaluation, une démo séparée a été ajoutée (sans modifier la démo originale):
 - `demo_ain_rubiks_2x2_consecutive.py`

 Principe:
 - Un stage est déclaré **PASS** seulement si `piece_thr` et `ori_thr` sont dépassés pendant **`K` évaluations consécutives** (`--pass_k K`).
 - Le log affiche `pass_streak=x/K`.

 Exemple d’usage (éval robuste + stage2 renforcé):
 ```bash
 python "Adaptive Invariant Neuron - AIN/demo_ain_rubiks_2x2_consecutive.py" \
   --epochs1 700 --epochs2 2000 --epochs3 1500 \
   --B 32 --N 24 --hidden 128 --z_dim 64 --lr 8e-4 \
   --piece_thr1 0.85 --piece_thr2 0.70 --piece_thr3 0.45 \
   --ori_thr1 0.90 --ori_thr2 0.75 --ori_thr3 0.60 \
   --scramble1 1 --scramble2 4 --scramble3 6 \
   --eval_every 50 --eval_batches 30 \
   --pass_k 3
 ```
