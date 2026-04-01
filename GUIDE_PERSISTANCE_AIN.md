# Guide de la Persistance et de la Mémoire Native (AIN)

Ce document explique le fonctionnement du système de **Persistance Automatique** du neurone d'argile (AIN), et montre à quel point il simplifie l'écriture des démos tout en permettant l'apprentissage incrémental continu.

---

## 1. Le Changement de Paradigme

Auparavant, la mémoire du neurone (le `ProgramBank`) était gérée **à l'extérieur** de l'architecture. Chaque démo devait créer sa propre banque, archiver manuellement les lois, s'occuper du chargement et de la sauvegarde.

Maintenant, **le ProgramBank est un organe interne natif du AIN**. La mémoire est indissociable du cerveau.

---

## 2. Ce qu'il faut faire dans ta démo (La seule étape !)

Pour activer la persistance globale, il suffit de donner un nom à ton neurone lors de son instanciation avec le paramètre **`model_name`** :

```python
from ain_neuron import AIN

# C'est la seule chose à faire !
model = AIN(
    x_dim=6, 
    z_dim=32, 
    query_dim=4, 
    out_dim=1, 
    model_name="mon_cerveau_global" # <-- Active la magie
)
```

Que se passe-t-il exactement quand tu ajoutes ce paramètre ? Le cycle de vie devient autonome :

### Étape A : Le Réveil (Auto-Load)
Si le fichier `mon_cerveau_global.pth` existe déjà dans le dossier parent de `ain_neuron.py`, **il est automatiquement téléchargé dans `model`**. Le neurone se réveille avec l'intégralité de ses poids (cerveau) et de son `ProgramBank` (mémoire) issus des sessions précédentes. Si le fichier n'existe pas, un nouveau cerveau vierge est créé.

### Étape B : L'Observation (Auto-Archive)
Pendant la boucle :
- Si `model.train()` est actif : Chaque passage par la fonction `forward(...)` **archive automatiquement** la nouvelle loi découverte (vecteur Z, signature de l'environnement, poids forgés) dans `model.bank`.
- Si `model.eval()` est actif : L'archivage est suspendu. Le neurone exécute les tâches sans remplir inutilement sa mémoire de choses qu'il ne doit pas apprendre.

### Étape C : Le Sommeil (Auto-Save `atexit`)
Lorsque ton script Python se termine (même si la boucle est finie, même si tu fais Ctrl+C, ou même en cas de crash non-fatal), le neurone intercepte la fermeture du programme et **sauvegarde automatiquement son état combiné (poids + mémoire) dans le fichier `mon_cerveau_global.pth`**. Tu ne perdras donc jamais une session d'entraînement.

---

## 3. Optionnel : Gérer la "Best Loss" (Meilleur Checkpoint)

Comme tu l'as remarqué, la fin du script n'est pas toujours le moment où le neurone était le meilleur (overfitting, perte instable). Le neurone offre désormais une méthode native pour gérer le "save_best_checkpoint".

Dans ta boucle de traning, juste après avoir calculé la _loss_, appelle cette fonction :

```python
for ep in range(epochs):
    # ... forward, loss, backward, step ...
    loss = crit(pred, target)
    
    # Dit au neurone de sauvegarder sur le disque UNIQUEMENT si cette loss 
    # bat la meilleure loss historique qu'il ait jamais vue pour ce model_name.
    model.save_if_best(current_loss=loss.item())
```

Ainsi, le fichier `.pth` sur le disque dur représentera toujours la version mathématiquement la plus pure de l'apprentissage en cours.

---

## 4. Le Graal : L'Apprentissage Incrémental Multi-Démos

Grâce à cette architecture, l'apprentissage continu (Continual Learning) devient transparent.

Si tu as deux démos complètement différentes :
- `demo_climat.py`
- `demo_finance.py`

Si dans les deux fichiers tu écris le même code d'instanciation :
```python
model = AIN(..., model_name="super_ain")
```

Alors **les deux scripts vont partager le même fichier `super_ain.pth`**. 
1. `demo_climat` va apprendre 120 lois de météorologie.
2. Plus tard, tu lances `demo_finance`. Le AIN se réveillera avec les lois de `demo_climat` déjà en mémoire. Face à un problème financier qui possède la même géométrie mathématique (signature Méta-Z) qu'une météo chaotique, la `BankPolicy` déclenchera un Reuse ("Hit !") et **réutilisera une loi météorologique pour prédire la finance**, sans rien avoir à recompiler.

C'est l'essence même d'une intelligence générale unifiée !
