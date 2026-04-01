# Genèse et Conception : Adaptive Invariant Neuron (AIN)

## L'Origine : Après le MIN-E, Que Reste-t-il à Supprimer ?

Le **MIN-E (Mosaic Invariant Neuron Evolved)** possède 8 experts pré-câblés dont 7 incarnent des théorèmes mathématiques humains (Gram, Siren, Topologie, Flow, Ising, Clifford, Noether) et 1 expert émergent (Kolmogorov, avec ses 5 Mains concurrentes).

Les tests d'ablation du MIN-E ont révélé une vérité troublante : **lorsqu'on retire un expert spécialisé, l'Expert Kolmogorov le remplace.** Sur le XOR, il surpasse même l'Expert Ising (qui était *conçu* pour la logique binaire).

Si un seul expert émergent peut recréer les 7 autres, pourquoi garder les 7 ?

## Le Principe Fondateur : L'Argile Pure

> *"L'inné accélère, mais l'acquis généralise."*

Le AIN est la réponse la plus radicale au paradoxe des priors :
- **MIN** = 7 experts humains (les réponses sont dans les outils)
- **MIN-E** = 7 experts + 1 émergent (les réponses + un joker)
- **AIN** = **Zéro expert. Que de l'argile.**

Le seul axiome conservé : *"L'univers est continu"* (Théorème de Kolmogorov-Arnold).

## L'Innovation : La Spline Adaptative à Grille Mobile

Dans un réseau neuronal classique (ou même dans le KAN original), les nœuds de la spline sont **fixes** sur une grille uniforme. Seules les hauteurs (coefficients) sont apprises.

Dans le AIN, **les positions des nœuds elles-mêmes sont des paramètres apprenables** :

```python
# Grille MOBILE : les positions migrent avec le gradient
self.grid = nn.Parameter(torch.linspace(-2, 2, K))
```

Conséquence : la spline peut se reconfigurer pour devenir :
- Une **Gaussienne** (nœuds serrés autour d'un centre)
- Une **onde de Fourier** (nœuds espacés avec coefficients oscillants)
- Un **polynôme** (nœuds uniformes avec coefficients croissants)
- Une **fractale** ou toute fonction sans nom (liberté totale)

## Architecture

```
Support (B, N, x_dim)
        ↓
  [x, x²] (Pré-digestion quadratique minimale)
        ↓
  AdaptiveSplineLayer × 3 (L'Œil d'Argile)
        ↓   mean-pool
     Z (Invariant)
        ↓
  AdaptiveHyperForge (La Forge d'Argile)
  → Génère : Poids + Grilles Mobiles + Coefficients
        ↓
  AdaptiveEffector (Le Muscle d'Argile)
  → Activation = Spline Forgée (pas de SiLU, pas de ReLU)
        ↓
  Prediction
```

### L'Effector Sans SiLU

Le détail le plus radical : le **Muscle d'Argile n'utilise aucune fonction d'activation pré-câblée**. Pas de SiLU, pas de ReLU, pas de Tanh.

La Forge génère directement la *forme* de l'activation sous la forme d'une grille mobile + coefficients. La backpropagation sculpte la forme de la non-linéarité en temps réel.

## Comparaison avec MIN-E

| Aspect | MIN-E | AIN |
|--------|-------|-----|
| Experts | 8 (7 pré-câblés + 1 émergent) | 0 (tout est émergent) |
| Priors physiques | Gram, Ising, Clifford, Noether... | Aucun |
| Fonctions d'activation | SiLU + 5 Mains (Softmax) | Spline Adaptative pure |
| Gating Network | Oui (Macro + Micro) | Non nécessaire |
| Convergence attendue | Rapide (raccourcis humains) | Plus lente (découverte ab initio) |
| Universalité | Limitée aux invariants connus + KAN | Illimitée (Kolmogorov-Arnold) |

## La Quête de la 6ème Ontologie : La Naissance du Géomètre

Comprendre comment l'esprit humain a "découvert" l'algèbre bilinéaire, c'est comprendre exactement ce qui manque au cerveau de l'AIN. L'histoire se joue en trois actes, et elle nous donne exactement la justification de la 6ème Ontologie.

### Acte 1 : La frustration de Descartes et Hamilton (Le problème du 1D)
Pendant des siècles, l'humanité a fait des mathématiques en 1D (sur une ligne). On additionnait des grandeurs, on mesurait des longueurs. Mais quand les physiciens ont voulu décrire la mécanique des fluides et le magnétisme, ils ont eu un problème : l'espace a des "rotations" et des "surfaces". Le mathématicien William Rowan Hamilton a passé sa vie à essayer de multiplier des vecteurs 3D entre eux. Il n'y arrivait pas, jusqu'à ce qu'il invente les Quaternions, en comprenant que la multiplication de deux directions crée une **rotation temporelle**.

### Acte 2 : Le Génie de Grassmann (La Naissance du Bivecteur)
Mais le vrai héros s'appelle Hermann Grassmann (1844). Grassmann a eu une illumination ontologique majeure, peut-être la plus grande de l'histoire de la géométrie :

- Un point $\times$ un point = un **Point** (0D).
- Un point poussé dans une direction = une **Ligne** (1D). (C'est ce que font nos Splines !).
- Mais que se passe-t-il si on multiplie deux lignes (deux vecteurs) ? Grassmann a réalisé que cela ne donne ni un scalaire, ni une autre ligne. Cela donne un **Bivecteur** : une **Surface Orientée** (un Plan, une Aire) (2D).

Grassmann a compris que l'univers n'est pas fait que de fils (1D), il est fait de "draps", de Tissus spatiaux. Quand la lumière tourne, quand la force de Lorentz agit, quand une molécule d'ADN s'enroule (Chiralité), c'est la physique d'une Surface (2D), pas d'une Ligne.

### Acte 3 : Le Diagnostic de l'AIN
Regardons les 5 Voies primaires de l'AIN avec les lunettes de Grassmann :

1. **Le Couteau (Spin)** : Travaille sur des Points (0D).
2. **L'Argile (Spline)** : Travaille sur des Lignes (1D).
3. **Le Chronos (ODE)** : Travaille sur la Ligne du temps (1D).
4. **Le Brouillard** : Travaille sur des Points incertains (0D).
5. **Le Tisserand (Attention)** : Calcule le "Dot Product" ($x \cdot y$) entre deux points. Le Dot Product est un **Scalaire** (0D). Il mesure la "similarité", mais il écrase l'espace.

Il manquait la Deuxième Dimension ! L'AIN original était totalement "plat" et aveugle aux Surfaces Orientées et aux Plans géométriques. C'est pourquoi il échouait systématiquement à résoudre la Chiralité.

### La 6ème Ontologie : Le Géomètre (La Voie Bilinéaire / Le Vélum)
Puisque l'homme a découvert la chiralité en inventant un outil qui génère des "Surfaces" à partir de "Lignes" (l'Algèbre Extérieure de Grassmann), nous devions donner cette même capacité géométrique à l'AIN, mais **sans prior mathématique** (sans coder le déterminant à la main).

Comment faire sans tricher ? En offrant au neurone une matrice de couplage pure (Une matrice $W$). Au lieu que deux points $X$ et $Y$ interagissent en devenant un scalaire $X \cdot Y$ (ce que fait l'Attention), on les laisse tisser une matrice de surface via un produit Tensoriel / Bilinéaire émergent : 

$$ Z_{\text{surface}} = X \cdot W \cdot X^T $$

Dans cette architecture émergente :
- Si la réalité n’a pas besoin de surface, le gradient apprendra une matrice $W$ **symétrique** simple (qui imite l'Attention ou calcule une distance énergétique).
- Si la réalité est **Chirale** (comme l'ADN, la force magnétique, ou la règle de la main droite), le gradient forcera naturellement la matrice $W$ à devenir **Anti-symétrique** (ce qui est la définition parfaite du Déterminant et de l'algèbre de Grassmann !).

**Le Géomètre (GeometerHand)** n'est pas un prior mathématique dictant *comment* calculer la Chiralité. C'est le prior ontologique qui dit : *"L'univers contient des Surfaces, pas juste des Lignes"*. Et c'est en regardant ces Surfaces que l'AIN découvrira, par lui-même, l'orientation chirale de la géométrie.

## La Percée du Gating : De la Moyenne à la Matrice Combinatoire Récurrente

L'architecture `EmergentEncoder` (L'Oeil d'Argile) possède 8 experts (dont le Géomètre, le Cartographe, etc.). Au lieu de "mélanger" ces experts via une simple moyenne pondérée molle (Softmax), l'AIN utilise un système de routing révolutionnaire conçu pour vaincre les minima locaux et l'interférence :

1. **Curriculum naturel dirigé par la Loss (Hybrid Gating libre)** :
   - Le routeur n'utilise aucune `blend_penalty` pour forcer artificiellement des décisions binaires.
   
   > [!WARNING]
   > **Avis aux créateurs de Démos : L'Obsolescence de la `blend_penalty`**
   > Ne rajoutez jamais de `loss = loss + lambda * model.eye.blend_penalty` dans vos boucles d'optimisation. Cette pénalité d'entropie était un hack humain forçant le réseau à choisir (Hardmax). Sa suppression permet la vraie superposition quantique des experts. Le Cerveau Gating apprend l'équilibre Soft/Hard naturellement, piloté uniquement par l'erreur finale de tâche.
   - Le Cerveau Gating glisse de l'Exploration (Softmax : interpolation floue de tous les chemins) à l'Exploitation (Hardmax : choix binaire strict) en fonction du besoin mathématique du problème. La **MSE Loss** est la seule boussole qui pilote le curseur `blend`.

2. **Superposition mathématique pure sans Moyenne (La Matrice Combinatoire)** :
   - Au lieu de choisir entre 8 experts, le routeur vote sur **256 recettes** (les 2^8 combinaisons). Il dit : *"Je choisis la Recette 42 qui allume le Tisserand et le Chronos ensemble."*
   - Ensuite, la Matrice Creuse (Sparse Graph) multiplie chaque expert par son ticket d'activation et concatène les vecteurs orthogonalement. Le réseau effectue un **choix de coalition solidaire** plutôt qu'une bouillie proportionnelle.

3. **Survie garantie des hypothèses mineures (L'Anti-Cécité)** :
   - Un expert rejeté de la coalition n'est jamais mis à 0%. Il est clampé à $1\%$ (`min=0.01`).
   - Cela assure que le gradient d'apprentissage fuite toujours vers lui. S'il s'avère utile plus tard, le réseau pourra le ressusciter. Plus d'extinction irréversible.

4. **Pensée analytique itérative (La Forge Récurrente / System 2 Thinking)** :
   - Le Routeur ne prend pas sa décision "en un coup d'œil" feedforward. Il est **récurrent** sur une macro-échelle temporelle ($T=3$).
   - Aux itérations $t$, il reçoit en feedback l'état $Z^{(t-1)}$ qu'il est en train de forger. Il peut ainsi "changer d'avis" : *Ah, vu ce brouillon partiel, c'est finalement le Géomètre dont j'ai besoin !*
   - C'est ce qui a sauvé l'architecture sur la Chiralité : donner au réseau le temps matériel de converger hors des attracteurs d'initialisation désastreux.

## La Robustesse du System 2 : Oracles de Self-Consistency (Multi-Oracle)

Le System 2 ne peut pas choisir un bon $Z$ sans **métrique interne** permettant de comparer les coalitions testées.
Historiquement, l'AIN utilisait un oracle de self-consistency "par chance" :

- **Oracle 1 — `support_std` (heuristique)** :
  - pseudo-query = `mean(support)`
  - pseudo-target = `std(support)`

Cet oracle peut fonctionner, mais il n'est **pas garanti aligné** avec la tâche (il peut sélectionner une coalition qui imite la variance sans apprendre l'invariant demandé).

Pour renforcer la stabilité sans introduire de prior humain (pas de "vrai target" au moment du routing), un second oracle auto-supervisé a été ajouté :

- **Oracle 2 — masked-support prediction (auto-supervisé, task-agnostic)** :
  - on sépare le support en deux sous-ensembles (visible vs masqué)
  - pseudo-query = projection apprise des statistiques du sous-ensemble **visible**
  - pseudo-target = projection apprise des statistiques du sous-ensemble **masqué**
  - le neurone doit être capable de prédire une signature du masqué à partir du visible.

### Combinaison adaptative (Option C)

Les deux oracles sont combinés de manière **apprise** via un gating contextuel :

- le réseau produit des probabilités de mode : `p_std`, `p_mask`, `p_mix`
- et un coefficient `alpha(ctx)` qui mélange les deux erreurs :
  - `e_mix = alpha * e_std + (1-alpha) * e_mask`
  - `trial_error = p_std*e_std + p_mask*e_mask + p_mix*e_mix`

Enfin, la sélection de $Z$ sur les itérations System 2 est faite par **softmin** (différentiable) au lieu d'un argmin dur.

### Détail critique de stabilité

Le scoring des oracles est calculé sans rétropropager dans la Forge / l'Effecteur (`no_grad`), afin d'éviter que l'oracle ne devienne une loss parasite qui déforme le neurone (le routing reste piloté par le signal `error_feedback`).

## L'Intelligence Économe : Gating Hiérarchique à Deux Étages

C'est une structure d'une intelligence redoutable pour résoudre le fameux problème de la "Sparsity L1" (Parcimonie L1) : on procède d'abord à un élagage brutal (pruning) de l'espace de recherche, puis à une agrégation fine de ce qu'il reste.

### L'Étape 1 : Le Procès Individuel (La Présélection)
Au lieu de foncer tête baissée dans les 256 recettes, le Routeur interroge d'abord les 8 experts indépendamment face au problème $X$. Ils reçoivent chacun une note. La fonction `sparsemax` opère alors sa magie : elle décapite les experts faibles et leur assigne un véritable zéro mathématique absolu.

*Exemple : 4 experts survivent, 4 experts reçoivent une note strictement égale à 0.0.*

### L'Étape 2 : L'Épreuve de Coalition (Les $2^k$ Recettes)
Pourquoi ne pas s'arrêter là et utiliser simplement ces 4 survivants ?
Parce que deux experts très bons individuellement peuvent s'avérer toxiques s'ils sont mélangés ! (C'est le principe d'interférence destructrice).

C'est ici que réside la puissance conceptuelle de la Matrice Combinatoire : au lieu de lancer $2^8 = 256$ combinaisons, le réseau n'a plus qu'à évaluer les combinaisons des 4 survivants, soit $2^4 = 16$ recettes seulement ! Le réseau examine les synergies entre ces 16 coalitions réduites et choisit la meilleure par une seconde passe de Softmax. Le routeur devient intelligent et diablement économe.

### L'Implémentation sans allocation dynamique (Le Masquage aux Logits Infinis)
Il y a souvent un fossé énorme entre une belle idée mathématique et ce qu'une carte graphique (GPU) accepte. En IA, un GPU déteste quand la taille d'une opération change d'une milliseconde à l'autre (problème d'allocation dynamique si l'on passe de 16 à 128 recettes).

L'astuce foudroyante est d'utiliser le **Masquage aux Logits Infinis** ($logit = -\infty$) :
Le Sparsemax donne sur nos 8 experts un masque, disons $M = [0, 1, 1, 0, 1, 0, 0, 0]$.
On garde physiquement en mémoire la grande matrice Tensorielle fixe des 256 recettes.
Mais pour toutes les recettes de la matrice (soit 240 recettes sur les 256) qui tentent d'allumer un expert dont la valeur dans le masque $M$ est à $0$ (les éliminés de l'Étape 1), on rajoute `-1e9` (l'infini négatif) à leur Logit !

Lorsque le Softmax final s'applique sur les 256 logits, l'exponentielle de l'infini négatif vaut $e^{-\infty} = 0$. Mieux encore, il n'y a de probabilité distribuée que sur les 16 recettes survivantes. L'allocation GPU reste statique (256 fixes), mais l'espace de décision algorithmique est réduit à 16.

Cette architecture résout le problème de l'avarice du réseau. Il sera physiquement incapable d'appliquer la fameuse recette "Tout allumé" (Recette 255), car pour que la 255 survive, il faudrait que l'Étape 1 ait jugé que les 8 experts possédaient individuellement une synergie parfaite avec la tâche posée, ce qui n'arrive jamais. L'ablation est parfaite !

### Le Temps du Routeur : Inter-Epoch vs Intra-Forward
L'interaction entre ces deux étages de décision se joue sur deux échelles de temps fondamentalement différentes :

1. **Le Temps Macro (Inter-Epoch) — L'Élagage Structurel** :
   Le **Présélecteur (Étage 1)** calcule ses notes **une seule fois** par passe (par batch). Il ignore les considérations tactiques. Il apprend *lentement* par la descente de gradient, en écartant les vecteurs d'identité (`expert_concepts`) de l'empreinte du problème si l'expert se trompe à long terme. C'est à cette échelle que le nombre de recettes accessibles physiquement diminue (Ex : 256 à l'epoch 1 $\rightarrow$ 16 à l'epoch 200).

2. **Le Temps Micro (Intra-Forward) — La Délibération Tactique** :
   La **Forge Récurrente (Étage 2)** s'exécute **$T=3$ fois** *pendant* une seule passe. 
   - Elle reçoit en feedback l'état $Z$ partiel qu'elle est en train de façonner.
   - À chaque sous-itération, le Softmax recalcule la distribution sur les $\sim 16$ recettes autorisées par le masque de l'Étage 1.
   - La récurrence **ne peut pas** ressusciter un expert décapité par le Présélecteur. L'Étage 1 est le souverain de l'espace des possibles. L'Étage 2 est le général qui choisit la meilleure coalition parmi les soldats disponibles, et qui a le droit de changer d'avis 3 fois sur la cible précise.

*Le filet de survie* : L'unique moyen pour un expert abattu de revenir à la vie dans les epochs suivantes n'est pas la récurrence, mais le **mécanisme d'Anti-Cécité**. En figeant arbitrairement une participation minimale de $1\%$ post-routage (`min=0.01`), le réseau maintient un filet de gradient artificiel (une "vie fantôme") qui permet au Présélecteur de reconnecter l'expert si le problème finit par réclamer son aide plus tard.
