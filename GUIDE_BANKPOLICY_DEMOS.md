# Guide — Utilisation de `ProgramBank` + `BankPolicy` + `ContextualAIN` dans les démos

Ce guide décrit **le pattern standard** pour utiliser la bank de programmes compilés en mode **inférence** (reuse vs recompile vs refresh), sans modifier `ain_neuron.py`.

## 1) Rappels : rôles des composants

- **`ProgramBank`** (base de données)
  - Stocke : `signature`, `z`, `forged` (+ option replay)
  - Recherche : similarité cosinus sur `z` (+ fallback signature)
  - Gère : LRU / éviction / `add()` / `refresh()` / save/load

- **`BankPolicy` / `BankPolicyV2`** (politique de décision)
  - Décide : `REUSE` / `RECOMPILE` / `REUSE_THEN_REFRESH`
  - Critères typiques :
    - similarité `z` (`best_z`)
    - marge `margin_z` (séparation top1-top2)
    - fallback signature (`best_sig`)
    - stabilité de `z` sous perturbation légère du support (`z_stability`)
    - optionnel : un probe supervisé (`probe_mse`) si une `target` est disponible
  - Arbitrage **qualité vs coût** via des coûts explicites (reuse/recompile/refresh)

  - **`BankPolicy` (v1)**
    - heuristique (seuils + règles)
    - simple à régler, mais nécessite du tuning manuel

  - **`BankPolicyV2` (v2)**
    - contextual bandit (petit MLP) avec **replay buffer** et updates sur **mini-batch**
    - arbitre dynamiquement coût vs qualité via des **lambdas adaptatifs** :
      - `lambda_cost` (budget de coût)
      - `lambda_quality` (contrainte qualité basée sur MSE)
      - `lambda_z` (contrôle de confiance sur le matching `z`)

- **`ContextualAIN`** (orchestrateur)
  - Orchestration : `support -> signature + z -> bank -> policy -> forged -> pred`
  - Réalise :
    - `RECOMPILE` : compile via Forge + `bank.add(...)`
    - `REUSE` : exécute le `forged` cache
    - `REUSE_THEN_REFRESH` : exécute puis recompile + `bank.refresh(...)`
  - Renvoie : `pred` + `logs` standardisés

## 2) Pattern de câblage (wiring) dans une démo

### Imports

```python
from ain_neuron import AIN
from program_bank import (
    ProgramBank,
    BankPolicy,
    BankPolicyConfig,
    BankPolicyV2,
    BankPolicyV2Config,
    ContextualAIN,
)
```

### Initialisation de la bank

```python
bank = ProgramBank(
    capacity=1000,
    z_threshold=0.93,
    signature_threshold=0.65,
    enable_signature_fallback=True,
    device=device,
)
```

### Politique v2 (apprentissage en ligne + lambdas adaptatifs)

```python
policy = BankPolicyV2(
    config=BankPolicyV2Config(
        epsilon=0.10,
        lr=2e-3,
        lambda_cost=0.15,

        adaptive_lambda=True,
        target_expected_cost=0.25,
        lambda_lr=5e-3,

        adaptive_quality=True,
        quality_delta=0.02,
        lambda_quality_lr=5e-3,

        adaptive_z=True,
        z_target_best=0.90,
        z_target_margin=0.02,
        z_target_stability=0.85,
        z_violation_target=0.0,
        z_violation_ema_beta=0.95,
        lambda_z_lr=5e-3,

        buffer_capacity=5000,
        batch_size=64,
        warmup=200,
        update_every=5,
        updates_per_step=2,
    ),
    device=device,
)
```

### Politique (config)

```python
policy = BankPolicy(
    config=BankPolicyConfig(
        z_reuse_threshold=bank.z_threshold,
        signature_reuse_threshold=bank.signature_threshold,
        enable_signature_fallback=bank.enable_signature_fallback,
        enable_probe=False,  # typiquement en vraie inférence
        z_margin_threshold=0.01,
        z_stability_threshold=0.90,
        # Coûts (unités arbitraires) :
        cost_reuse=0.05,
        cost_recompile=1.00,
        cost_refresh=1.00,
        quality_vs_cost=0.50,
    )
)
```

### Wrapper `ContextualAIN`

Tu dois fournir **3 callables** :

- `infer_z(support) -> z`
- `compile_forged(z) -> forged`
- `execute(query, forged) -> pred`

Exemple AIN :

```python
model = AIN(...)

contextual = ContextualAIN(
    bank=bank,
    policy=policy,
    infer_z=lambda support: model.eye(support),
    compile_forged=lambda z: model.forge(z),
    execute=lambda query, forged: model.effector(query, forged),
    device=device,
)
```

## 3) Exécution standard (un épisode)

Tu appelles :

```python
pred, logs = contextual.run(
    support=support,
    query=query,
    target=target_or_none,
    topk=5,
    costs={
        "reuse": 0.05,
        "recompile": 1.00,
        "refresh": 1.00,
    },
)
```

### Remarques importantes

- **`target`**
  - Si tu passes `target`, la policy peut activer le **probe supervisé** (`probe_mse`) et décider `REUSE_THEN_REFRESH`.
  - En production (inférence réelle), **`target` est souvent indisponible** : passe `target=None` et désactive `enable_probe`.
  - Avec **`BankPolicyV2`**, l'**apprentissage en ligne** (replay buffer + updates) ne se fait que si `ContextualAIN` peut appeler `policy.observe(...)`. Dans la version actuelle, cela nécessite que `target` soit fourni (pour calculer la MSE et construire la reward interne).

- **Batching de `forged`**
  - La bank stocke généralement `forged` en batch `1`.
  - `ContextualAIN` expand automatiquement pour exécuter sur un batch de queries.

## 4) Logs utiles à tracer (diagnostic)

Le dictionnaire `logs` contient typiquement :

- `best_z` : meilleure similarité cosinus (0..1)
- `margin_z` : séparation entre top1 et top2
- `best_sig` : score signature (fallback)
- `z_stability` : stabilité de `z` sous légère perturbation
- `probe_mse` : MSE du probe (si `target` fourni)
- `reused` / `recompiled` / `refreshed` : indicateurs 0/1
- `expected_cost` : coût attendu de la décision

Si la policy expose des lambdas adaptatifs (v2), `ContextualAIN` ajoute :

- `lambda_cost`
- `lambda_quality`
- `lambda_z`

Et des signaux de contrainte (v2) :

- `mse_baseline` : baseline MSE (EMA des épisodes `RECOMPILE`)
- `quality_violation` : violation (MSE au-dessus de baseline + delta)
- `z_violation` : violation instantanée de confiance `z` (best/margin/stability)
- `z_violation_ema` : EMA de `z_violation` (utile pour comprendre l'évolution de `lambda_z`)

## 5) Réglages recommandés (pragmatiques)

- Si la bank réutilise **trop** (et fait beaucoup de refresh) :
  - augmenter `z_reuse_threshold`
  - augmenter `z_margin_threshold`
  - augmenter `z_stability_threshold`

Avec `BankPolicyV2`, tu peux aussi :

- augmenter `lambda_z_lr` ou augmenter les cibles (`z_target_best`, `z_target_stability`) pour rendre le reuse plus strict
- augmenter `target_expected_cost` si tu veux autoriser plus de recompiles (donc réduire la pression de coût)

- Si la bank recompile **trop** (bank grossit vite, faible hit-rate) :
  - diminuer `z_reuse_threshold`
  - diminuer `z_margin_threshold`
  - diminuer `z_stability_threshold`
  - augmenter `cost_recompile` (ou diminuer `cost_reuse`) pour favoriser la réutilisation dans les cas “near match”

Avec `BankPolicyV2`, tu peux aussi :

- diminuer `z_target_best` / `z_target_stability` (moins strict) ou augmenter `z_violation_target` (tolérance)
- diminuer `lambda_z_lr` si `lambda_z` devient trop agressif

## 6) Exemple : démo existante

Le script `demo_ain_program_bank_inference.py` illustre le pattern complet :
- pré-train AIN
- stream d’épisodes avec lois récurrentes
- comparaison MSE bank vs compile systématique
- tracking `hit_rate`, `refresh_rate` et `expected_cost`
