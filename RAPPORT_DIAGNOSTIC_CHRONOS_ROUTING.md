# Rapport — Diagnostic Chronos / Routing (AIN)

## Objectif
Établir rigoureusement :
- si la voie **Chronos** (`SplineFlow`) est réellement utile (au-delà de `chronos_mean`),
- si les tâches “séquence” testées nécessitent vraiment l’ordre,
- et si la suppression de Chronos dégrade une loi **strictement dépendante de l’ordre**.

## Contexte
Après passage à un **router conditionné par l’entrée**, puis ajout d’une “signature de réalité” :
- **Être** : `mean/std` sur le support enrichi `x_exp = [x, x^2]`
- **Devenir** : `mean/std` sur les **deltas** `x_exp[:,1:]-x_exp[:,:-1]`

Le routeur reçoit :
```python
router_input = cat([x_mean, x_std, delta_mean, delta_std], dim=-1)
```

## Expérimentation
### Scripts modifiés
- `demo_ain_sequence.py`
  - Ajout d’une épreuve **ORDER-ONLY** (identifiabilité stricte à l’ordre).
  - Ajout d’un test `FINAL-PERM` : même batch, support **permuté**, mêmes `query/target`.
  - Ajout d’une ablation **Option 2** : entraînement + évaluation d’un modèle **sans Chronos** (`AINNoChronos`).
  - Ajout d’un logging avancé sur ORDER-ONLY :
    - quantiles de `gate_time` (distribution par sample)
    - contributions par voie via `mean(||gate_i * z_i||)`

### Épreuve ORDER-ONLY (identifiabilité stricte)
Construction :
- feature0 contient un multiset fixe de signes : `N/2` positifs et `N/2` négatifs.
- l’ordre est aléatoire par batch.
- invariant = signe de la somme pondérée positionnelle (rampe) :
  - `inv = sign( sum_t w_t * sign(x_t0) )`
- `target = inv` (pas de modulation par `query`) pour isoler la dépendance à l’ordre.

### Épreuve DYNAMICS (séquentielle “réaliste”, identifiabilité stricte)
Construction :
- On génère une trajectoire via un état latent scalaire `s_t` :
  - `s_{t+1} = tanh(theta * s_t + eps)`
  - `theta` (par batch) est le paramètre de loi.
- Le support ne contient **pas** de deltas explicites (sinon la tâche redevient résoluble sans ordre) :
  - feature0 = `s_t`
  - autres features = bruit indépendant
- Target : `inv = sign(theta)`.
- Test `FINAL-PERM` : on permute les noeuds du support sans changer le target.

## Résultats (logs consolidés)

### Baseline (AIN avec Chronos)
ORDER-ONLY :
- `ORDER-ONLY-SEQ FINAL` :
  - `mse=0.225646`
  - `sign_acc=0.9414`
  - `chronos_mean=0.133`
- `ORDER-ONLY-SEQ FINAL-PERM` :
  - `mse=1.786785`
  - `sign_acc=0.5352` (≈ hasard)

Logging avancé :
- `ORDER-ONLY-SEQ GATE_TIME_Q` :
  - `min=0.127 p10=0.130 p50=0.133 p90=0.136 max=0.139`
- `ORDER-ONLY-SEQ CONTRIB_NORM` :
  - `spline=0.725 spin=0.533 attn=2.246 var=0.070 time=3.618`
- `ORDER-ONLY-SEQ FINAL-PERM CONTRIB_NORM` :
  - `spline=0.728 spin=0.533 attn=2.243 var=0.070 time=3.589`

### Ablation (AINNoChronos, 4 voies)
ORDER-ONLY :
- `ORDER-ONLY-NOCHRONOS FINAL` :
  - `mse=0.997036`
  - `sign_acc=0.5312` (≈ hasard)
- `ORDER-ONLY-NOCHRONOS FINAL-PERM` :
  - `mse=0.997289`
  - `sign_acc=0.5234` (≈ hasard)

Logging avancé :
- `ORDER-ONLY-NOCHRONOS CONTRIB_NORM` :
  - `spline=1.764 spin=0.502 attn=0.480 var=0.476`

### Observations sur les anciennes épreuves “SEQ”
Sur `UNALIGNED-SEQ` et `COMPOSED-SEQ` (dans ce script), on observe typiquement :
- `FINAL ≈ FINAL-PERM`
- `chronos_mean` très faible

Cela montre qu’elles sont **résolubles sans ordre** (proxy set-like), donc Chronos est rationnellement éteint.

### Résultats DYNAMICS (logs consolidés)

Baseline (avec Chronos) :
- `DYNAMICS-SEQ FINAL` :
  - `mse=0.018451`
  - `sign_acc=0.9961`
  - `chronos_mean=0.008`
- `DYNAMICS-SEQ FINAL-PERM` :
  - `mse=1.697718`
  - `sign_acc=0.5391` (≈ hasard)

Logging avancé (baseline) :
- `DYNAMICS-SEQ GATE_TIME_Q` :
  - `min=0.006 p10=0.007 p50=0.008 p90=0.009 max=0.010`
- `DYNAMICS-SEQ CONTRIB_NORM` :
  - `spline=4.591 spin=0.913 attn=0.466 var=0.024 time=0.106`

Ablation (AINNoChronos, 4 voies) :
- `DYNAMICS-NOCHRONOS FINAL` :
  - `mse=0.046027`
  - `sign_acc=0.9922`
- `DYNAMICS-NOCHRONOS FINAL-PERM` :
  - `mse=1.675217`
  - `sign_acc=0.5781`

## Analyse

### 1) `chronos_mean` faible n’implique pas “Chronos inutile”
Sur ORDER-ONLY, `chronos_mean≈0.13` est relativement bas, mais :
- la permutation détruit la performance (acc ~0.94 → ~0.53)
- l’ablation **sans Chronos** retombe au hasard même sans permutation

Donc Chronos est **causalement nécessaire**, même si le mélange ne lui donne pas 90% du poids.

### 2) Distribution de `gate_time` (pas de mode “0 ou 1”)
Les quantiles de `gate_time` sont serrés (~0.127–0.139), ce qui indique :
- pas une sélection “hard” de Chronos sur quelques samples,
- mais une contribution **faible et stable** sur l’ensemble du batch.

Ce point est cohérent avec un rôle de Chronos comme composant “faiblement pondéré mais décisif”.

### 3) Contribution réelle `||gate_i * z_i||`
Sur ORDER-ONLY (baseline), la contribution **time** est la plus grande :
- `time ≈ 3.618` > `attn ≈ 2.246` > `spline/spin/var`.

C’est crucial :
- même avec `gate_time≈0.13`, le latent `z_time` a une amplitude/structure telle que sa contribution au `z` final est dominante.
- cela explique pourquoi Chronos est “peu présent” en poids, mais **très présent** en impact.

### 4) Pourquoi ORDER-ONLY-PERM ne change pas les contributions par voie ?
Sur `FINAL-PERM`, les contributions restent proches, mais la performance s’effondre.
Interprétation plausible :
- Chronos continue de produire une représentation temporelle “forte” (norme élevée),
- mais celle-ci n’est plus alignée avec la loi (car l’ordre a été détruit), donc l’effecteur se trompe.

### 5) Pourquoi les anciennes tâches SEQ n’activent pas Chronos ?
La preuve `FINAL≈FINAL-PERM` indique que l’ordre n’est pas nécessaire pour gagner sur ces tasks.
Donc, même avec une signature delta, le routeur n’a aucune raison d’investir dans Chronos.

### 6) Point critique : la “signature delta” rend le modèle global non permutation-invariant
Même **sans Chronos**, dès lors que le routeur consomme `delta_mean/delta_std`, le gating (et donc le `z` final) dépend de l’ordre.

Conséquence :
- un `FINAL-PERM` peut se dégrader **même si Chronos est absent**, simplement parce que le routeur (et donc le mélange des voies set-like) a changé.

Ce point explique pourquoi :
- ORDER-ONLY : sans Chronos, on retombe au hasard même sans permutation (incapacité à extraire la loi d’ordre).
- DYNAMICS : sans Chronos, on peut encore réussir sur l’original (probablement via statistiques/structures restantes), tout en étant sensible à la permutation via le routeur.

## Conclusion
- Il **ne faut pas retirer Chronos** si l’objectif inclut des lois réellement séquentielles/causales.
- `chronos_mean` est un indicateur trompeur :
  - Chronos peut avoir un poids moyen faible,
  - mais une contribution dominante au latent final (via `||gate_i * z_i||`).
- Les tâches “SEQ” existantes doivent être refondues si l’on veut tester le temps :
  - elles doivent être **non résolubles** sans ordre (identifiabilité).

## Recommandations
- Ajouter/maintenir au moins une épreuve de type ORDER-ONLY (ou semi-order) pour valider la causalité.
- Si l’objectif est de voir `chronos_mean` monter, il faut soit :
  - des lois qui exigent l’ordre,
  - soit une exploration dirigée (warmup Chronos, régularisation ciblée), ce qui est un choix méthodologique.

### Option (plus “propre” ontologiquement, sans flag)
Pour éviter que le routeur "hallucine" du temps sur des données set-like, on peut ajouter une **régularisation de cohérence sous permutation** (dans les démos / l’entraînement, pas dans `ain_neuron.py`) :

- Prendre un support `X` et une version permutée `perm(X)`
- Pénaliser la différence des sorties du routeur :
  - `||gate_logits(X) - gate_logits(perm(X))||`
  - et/ou `||blend(X) - blend(perm(X))||`

- pas de `mode=set`
- pas de tri
- juste une contrainte informationnelle : *si l’ordre n’est pas stable, n’en dépends pas.*

## ÉPILOGUE : L'Évolution du Routeur (De la Pénalité à l'Attracteur Cognitif)

Les diagnostics initiaux avaient mis en exergue un double problème : "L'Atemporalité du Routeur" et la "Tyrannie du Soft Gating".

La **version initiale** de la solution avait consisté à introduire un *TemporalRouter* récurrent et une *Pénalité d'Indécision* (`blend_penalty`) forçant artificiellement le routeur vers le Hardmax. Bien que fonctionnelle, cette approche reposait sur un prior humain (forcer l'entropie binaire) et limitait les capacités de superposition du réseau (par exemple, pour la Courbure ou le Hasard qui exigent un "vrai" Softmax). 

> [!WARNING]
> **Pourquoi `blend_penalty` a été définitivement supprimée et pourquoi vous ne devriez JAMAIS l'ajouter dans vos boucles d'entraînement (démos) :**
> L'ajout d'une pénalité d'entropie (comme `lam * blend_penalty`) est une béquille humaine qui force le réseau à choisir de manière binaire. Cela détruit la capacité de superposition du routeur combinatoire (qui a besoin d'être "soft" pour fusionner certaines propriétés mathématiques) et empêche l'émergence naturelle du Routing. La Loss principale (MSE ou Cross-Entropy) est mathématiquement suffisante pour enseigner au réseau quand être certain (Hardmax) et quand être nuancé (Softmax).

### La Révolution du Gating (Architecture Finale)

L'architecture du Cerveau Gating a donc été totalement refondue autour de quatre piliers purs, **sans aucune pénalité artificielle (Curriculum naturel dirigé par la Loss)** :

1. **Gating Hybride Libre** : La MSE seule décide de l'équilibre Soft/Hard.
2. **Matrice Combinatoire Creuse** : Au lieu de moyenner les experts, le réseau opère par Superposition Mathématique Pure.
3. **Anti-Cécité (1%)** : Survie garantie des hypothèses mineures pour sauver le gradient.
4. **La Forge Récurrente** : Pensée analytique itérative (System 2 Thinking) permettant au réseau de corriger ses choix.

### Anatomie d'une Décision : Les 8 Étapes du Routeur Combinatoire Récurrent

Voici exactement comment l'Octogone choisit ses coalitions d'experts, étape par étape :

**Étape 1 : Le Routeur propose ses 256 votes**
Le routeur ne vote pas directement sur les 8 experts. Il vote sur les **256 recettes**. Chaque recette est une combinaison binaire des 8 experts. Par exemple :
- Recette 0 = `[0,0,0,0,0,0,0,0]` (personne)
- Recette 7 = `[0,0,0,0,0,1,1,1]` (Géomètre + Cartographe + Jardin)
- Recette 255 = `[1,1,1,1,1,1,1,1]` (tout le monde)
Le routeur produit 256 logits bruts : *"Je vote 3.2 pour la Recette 42, 1.1 pour la Recette 7..."*

**Étape 2 : Softmax (La moyenne douce)**
`soft_gates = softmax(logits)`
→ Distribution floue d'exploration : "Recette 42 = 8%, Recette 7 = 5%, etc."

**Étape 3 : Hardmax (Le choix binaire)**
`hard_gates = one_hot(argmax(logits))`
→ Choix radical : "Recette 42 = 100%, toutes les autres = 0%"

**Étape 4 : Le Blend (Le Curseur d'Exploitation)**
`gates_256 = blend × soft_gates + (1 - blend) × hard_gates`
La MSE pousse naturellement le curseur `blend` vers le Soft ou le Hard selon la nature du problème mathématique à résoudre.

**⭐ Étape 5 : LA COMBINATOIRE (Le Choix de Coalition)**
C'est ici que la magie opère. Comment passer des 256 recettes aux 8 experts ?
`gates_8 = gates_256 × matrice_combinatoire(256, 8)`
La matrice dit : *"La Recette 42, en binaire c'est `[0,0,1,0,1,0,1,0]`, donc elle veut le Tisserand + le Chronos + le Cartographe."*
Si le réseau a mis 100% de son poids sur la Recette 42, alors `gates_8` sera exactement `[0, 0, 1, 0, 1, 0, 1, 0]` → seuls 3 experts sont allumés !
Le réseau ne choisit pas "60% Géomètre et 40% Spin". Il choisit **une coalition solidaire**, empêchant la funeste moyenne arithmétique.

**⭐ Étape 6 : L'ANTI-CÉCITÉ (La Survie Hérétique)**
`gates_8 = clamp(gates_8, min=0.01)`
Les experts éteints sont remontés à 1%. Le `[0, 0, 1, 0, 1, 0, 1, 0]` devient `[0.01, 0.01, 1.0, 0.01, 1.0, 0.01, 1.0, 0.01]`.
*Pourquoi ?* Si le routeur s'est trompé à l'époque 10 (ex: il a éteint le Géomètre par erreur sur la Chiralité), ce petit 1% assure que le gradient atteint toujours l'expert. À l'époque 11, il pourra le réveiller. Sans ce 1%, un expert éteint resterait mort à jamais.

**⭐ Étape 7 : LA FORGE RÉCURRENTE (L'Attracteur, T=3)**
Le calcul des étapes 1 à 6 ne se fait pas qu'une seule fois. Le `CombinatorialRouter` reçoit en *feedback* le vecteur macroscopique $Z$ qu'il est en train de construire (`z_proj`).
Le réseau boucle 3 fois : le brouillon $Z^{(0)}$ donne de nouveaux indices au routeur pour l'étape 1, qui re-calcule les votes, corrigeant ainsi dynamiquement ses a priori (System 2 Thinking). Un Géomètre à 1% au tour 1 peut grimper à 100% au tour 3.

**Étape 8 : La Matrice Creuse (Sparse Graph)**
`z_sparse = gates_8 × z_stack` (Multiplication, *PAS* addition).
Chaque expert calcule sa dimension `slot_dim` (ex: 4). Les propositions sont multipliées par le masque `gates_8`. L'espace `z_flat` résultant est une **concaténation orthogonale** géante de `8 * slot_dim = z_dim` paramètres. L'interférence est mathématiquement vaincue.
