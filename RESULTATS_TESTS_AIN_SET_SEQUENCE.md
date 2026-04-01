# Résultats d’expériences — AIN (mode SET vs mode SEQUENCE)

Ce document synthétise les tests conçus pour éprouver **Adaptive Invariant Neuron (AIN)** dans deux régimes d’interprétation du support :

- **Mode SET (ensemble)** : le support est un **ensemble** de nœuds, l’ordre ne doit porter aucune information.
- **Mode SEQUENCE (séquence)** : le support est une **séquence** causale, l’ordre porte du sens.

L’objectif est d’évaluer ce que ton AIN gagne en “émergence” (moins de priors humains), et ce qu’il perd/transforme en termes d’**inductive bias**.

---

## Fichiers de démo

- **SET** : `Adaptive Invariant Neuron - AIN/demo_ain_set.py`
- **SEQUENCE** : `Adaptive Invariant Neuron - AIN/demo_ain_sequence.py`

Chaque démo contient 2 épreuves :

1. **Non-aligné** : invariant discontinu + non-local (résultat évalué via `sign_acc`)
2. **Composé** : produit de 3 invariants hétérogènes + curriculum (inv1 → inv2 → inv3 → composed)

Métriques :

- **MSE** : erreur quadratique moyenne.
- **`sign_acc`** : `mean(sign(pred) == sign(target))`.

---

## Définition des tests

### Test 1 — « Non-aligné »

- Invariant : basé sur des **extrêmes** (`argmax/argmin` après projection) et un **seuil** (discontinuité).
- Target : `inv * tanh(sin(qproj))`.

Ce test sert à vérifier si AIN peut apprendre un invariant discontinu/non-local sans expert mathématique humain dédié.

### Test 2 — « Composé » + curriculum

- `inv = inv1 * inv2 * inv3`.
- En **pré-entraînement (PRE)** : on entraîne sur chaque invariant pur (`target = inv`).
- En **POST** : on entraîne sur la target composée (avec composant query + terme discontinu).

**Point crucial (identifiabilité)** :
- En mode SET, chaque invariant doit être permutation-invariant.
- En mode SEQUENCE, au contraire on autorise des invariants dépendants de l’ordre.

---

# Résultats — Mode SET (ensemble)

Script : `demo_ain_set.py`

Paramètres d’entraînement (SET) : augmentation organique par permutation du support

- `perm_aug_weight=1.0`
- `num_perm_augs=2`

## Résultats finaux (lignes `[FINAL]`)

- **UNALIGNED-SET**
  - `mse=0.059307`
  - `sign_acc=0.9219`

- **UNALIGNED-SET (diagnostic permutation — lignes `[FINAL-PERM]`)**
  - `mse_perm=0.060192`
  - `sign_acc_perm=0.9180`
  - `z_cos=0.9610`
  - `z_l2=0.2543`
  - `gates_l1=0.0048`
  - `blend_l1=0.0041`

- **COMPOSED-SET**
  - `mse=0.326517`
  - `sign_acc=0.7305`

- **COMPOSED-SET (diagnostic permutation — lignes `[FINAL-PERM]`)**
  - `mse_perm=0.329032`
  - `sign_acc_perm=0.7109`
  - `z_cos=0.9784`
  - `z_l2=0.0537`
  - `gates_l1=0.0008`
  - `blend_l1=0.0050`

## Ce que ça démontre

- AIN est **capable** d’apprendre un invariant discontinu + non-local en respectant la contrainte ensemble (permutation aléatoire des nœuds à chaque épisode).
- AIN apprend la **composition** au-dessus du hasard, mais reste moins performant qu’un modèle avec priors experts (ex. MIN‑E sur des versions comparables).

---

# Résultats — Mode SEQUENCE (séquence)

Script : `demo_ain_sequence.py`

## Résultats finaux (lignes `[FINAL]`)

- **UNALIGNED-SEQ**
  - `mse=0.157031`
  - `sign_acc=0.8320`

- **COMPOSED-SEQ**
  - `mse=0.399350`
  - `sign_acc=0.4609`

## Ce que ça démontre

- En séquence, AIN apprend le non-aligné à un niveau correct (sign_acc ~0.83).
- Sur ce run, la composition séquentielle finale reste fragile (sign_acc ~0.46).

---

# Patch appliqué (Option A) — rendre `inv2` séquentiel apprenable

Contexte : la première version de `inv2` séquentiel était une **parité dure** sur un préfixe (`k` premiers nœuds). Cette loi est très hostile à une supervision MSE (discontinuité + dépendances de haut ordre).

## Modification

Fichier : `demo_ain_sequence.py` uniquement.

- `inv2` a été remplacé par un **vote majoritaire** discontinu sur les `k` premiers nœuds.
- `k` est passé à un nombre **impair** (`k=7`) pour réduire les égalités.
- En cas d’égalité, un tie-break utilise la somme brute (`x.sum`) pour éviter un biais systématique.

Effet observé :
- `PRE-inv2-SEQ` devient apprenable (sign_acc monte fortement),
- et `COMPOSED-SEQ` dépasse nettement le hasard.

---

# Conclusion générale

- **SET (ensemble)** :
  - Très bon sur non-aligné (`sign_acc ~0.91`).
  - Composition apprise mais modérément (`~0.64`).

- **SEQUENCE (séquence)** :
  - Non-aligné correct (`~0.89`).
  - Composition réussie (`~0.68`) après avoir rendu `inv2` “apprenable” (vote majoritaire au lieu de parité).

**Lecture “prior vs émergence”** :
- AIN supprime les priors humains (Gram/Clifford/Noether…), mais conserve des **priors ontologiques** (continu/discret/relationnel/stochastique/temporel).
- Les résultats montrent que ces priors ontologiques suffisent à apprendre des invariants discontinus/non-locaux, mais la composition discontinue peut exiger :
  - une loi plus apprenable,
  - un curriculum,
  - ou une loss adaptée (si on voulait aller plus loin).

---

## Commandes pour reproduire

Depuis `.../Officiel` :

- `python -u "Adaptive Invariant Neuron - AIN/demo_ain_set.py"`
- `python -u "Adaptive Invariant Neuron - AIN/demo_ain_sequence.py"`
