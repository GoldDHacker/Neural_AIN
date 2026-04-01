# Résultats — Compilation Test (AIN)

## Objectif
Tester le triptyque :
- Support → **Invariant `z`** (représentation de loi)
- `z` → **Forge** (compilation : génération des paramètres d’un effecteur)
- **Muscle forgé** → exécution sur de multiples queries sans ré-encoder le support

Critère clé : la Forge doit produire un exécuteur **réutilisable** (intra-épisode) et **spécifique à la loi** (swap-law doit échouer).

---

## Test 1 — Compilation “facile” (paramètres observables dans le support)
Script : `demo_ain_compilation.py`

### Protocole
- La loi par épisode est paramétrée par `theta0`, `theta1`.
- Le support contient des observations bruitées de ces paramètres (feature0/feature1).
- On forge une seule fois, puis on exécute sur `K` queries.

### Résultats (log)
- `mse_in_episode` : `0.010176`
- `mse_cross_support_a` : `0.013868`
- `mse_cross_support_b` : `0.013627`
- `z_cos_cross_support` : `0.998517`
- `mse_swap_12` : `1.729182`
- `mse_swap_21` : `1.727348`

### Lecture
- La compilation est **réutilisable** (MSE bas sur K queries).
- La représentation `z` est **très stable** entre deux supports d’une même loi (`cos≈0.999`).
- Le contrôle **swap-law** échoue fortement (MSE élevé) → la compilation est **spécifique**.

Limite : ce test reste relativement simple car le support donne un accès direct (bruité) aux paramètres de loi.

---

## Test 2 — Compilation “hard” (support = exemples (xᵢ,yᵢ), pas de paramètres)
Script : `demo_ain_compilation_hard.py`

### Protocole
- La loi `f_episode` est latente et tirée aléatoirement par épisode.
- Le support ne contient que des **paires d’exemples** `(xᵢ, yᵢ)` avec `yᵢ = f_episode(xᵢ)`.
- Le modèle doit inférer la loi uniquement via ces exemples.
- Tests :
  - **in-episode reuse**
  - **cross-support same law** (deux supports différents pour la même `f_episode`)
  - **swap-law control**

### Résultats (log)
- `mse_in_episode` : `0.312410`
- `mse_zero` (baseline prédire 0) : `0.512144`
- `mse_cross_support_a` : `0.311756`
- `mse_cross_support_b` : `0.312259`
- `z_cos_cross_support` : `0.964641`
- `mse_swap_12` : `0.816240`
- `mse_swap_21` : `0.770138`

### Lecture
- Le modèle fait mieux que le baseline trivial (`mse_in_episode < mse_zero`) → il apprend une loi exploitable.
- Cross-support stable → la performance dépend de la loi, pas d’un support particulier.
- Swap-law significativement pire que in-episode (`~0.77–0.82` vs `~0.31`) → la compilation est **discriminante**.

Limite : la précision reste modérée (MSE ~0.31). Ce test est plus proche d’une inférence de fonction (few-shot) et il est normal qu’il soit plus difficile.

---

## Conclusion
- AIN réalise bien le schéma **Support → z → Forge → Exécution** dans les deux cas.
- Le test hard confirme que la Forge peut compiler une loi **à partir d’exemples**, sans accès direct à des paramètres.
- Le contrôle swap-law valide que la compilation est **spécifique** à l’épisode (pas un simple fit moyen).
