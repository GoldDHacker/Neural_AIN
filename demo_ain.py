"""
==============================================================================
  DEMO DIAGNOSTIQUE — ADAPTIVE INVARIANT NEURON (AIN)
  Le Neurone d'Argile : Zero Prior, Splines Adaptatives a Grille Mobile
  4 Epreuves : Affine | XOR | Chiralite | Anti-Fuite
==============================================================================
"""

import sys, os
import math
import copy
import argparse
from datetime import datetime


class TeeWriter:
    """Ecrit simultanement dans la console ET dans un fichier .txt."""
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, 'w', encoding='utf-8')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.log.close()
        sys.stdout = self.terminal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim
from ain_neuron import AIN

# --- Generateurs de donnees synthétiques ---

def gen_affine(B, N, x_dim, q_dim):
    """Loi affine : target = sum(mean(support) * query)"""
    support = torch.randn(B, N, x_dim)
    query = torch.randn(B, q_dim)
    mu = support.mean(dim=1)  # (B, x_dim)
    target = (mu[:, :q_dim] * query).sum(dim=1, keepdim=True)
    return support, query, target

def gen_xor(B, N, x_dim, q_dim):
    """XOR scalaire : target = xor(sign projections) -> {0, 1}"""
    support = torch.randn(B, N, x_dim)
    query = torch.randn(B, q_dim)
    
    mu = support.mean(dim=1)
    a = (mu[:, 0] > 0).float()
    b = (mu[:, 1] > 0).float() if x_dim >= 2 else (mu[:, 0] > 0.5).float()
    target = (a + b - 2 * a * b).unsqueeze(1)  # XOR
    return support, query, target

def gen_chirality(B, N, x_dim, q_dim):
    """Chiralite pure (Set topology) : spirale d'Archimede 2D.
    
    50% Spirale Droite (+1), 50% Spirale Gauche (-1).
    L'objet subit une rotation SO(2) aléatoire et un shuffling des noeuds
    pour s'assurer qu'aucune information de séquence (Chronos) ne puisse aider.
    Seule l'asymetrie intrinseque de la forme permet de resoudre.
    """
    assert x_dim >= 2
    support = torch.zeros(B, N, x_dim)
    query = torch.randn(B, q_dim)
    
    # +1 ou -1
    labels = (torch.rand(B) > 0.5).float() * 2 - 1.0
    target = labels.unsqueeze(1)
    
    # Generer la spirale
    t = torch.linspace(0.5, 3.0 * math.pi, N).view(1, N)
    r = t
    
    x = r * torch.cos(t)
    y = r * torch.sin(t)
    
    x = x.expand(B, N).clone()
    y = y.expand(B, N).clone()
    
    # Symetrie miroir sur Y pour obtenir le chiral oppose
    y = y * labels.view(B, 1)
    
    # Rotation aleatoire SO(2)
    theta = torch.rand(B) * 2 * math.pi
    cos_t = torch.cos(theta).view(B, 1)
    sin_t = torch.sin(theta).view(B, 1)
    
    x_rot = x * cos_t - y * sin_t
    y_rot = x * sin_t + y * cos_t
    
    support[:, :, 0] = x_rot
    support[:, :, 1] = y_rot
    
    # Bruit structurel
    if x_dim > 2:
        support[:, :, 2:] = torch.randn(B, N, x_dim - 2) * 0.1
    support += torch.randn_like(support) * 0.1
    
    # Shuffling essentiel pour detruire la sequence temporelle
    for b in range(B):
        idx = torch.randperm(N)
        support[b] = support[b, idx]
        
    return support, query, target

def gen_antileak(B, N, x_dim, q_dim):
    """Anti-fuite : loi non-lineaire + locale (tanh sur sous-ensemble de noeuds)"""
    support = torch.randn(B, N, x_dim)
    
    n_local = max(1, N // 3)
    local = support[:, :n_local, :]
    
    cross = (local[:, :, 0] * local[:, :, 1]).mean(dim=1) if x_dim >= 2 else local.mean(dim=(1, 2))
    local_mean = local.mean(dim=(1, 2))
    
    cross = cross.view(B, 1)
    local_mean = local_mean.view(B, 1)
    
    factor = torch.tanh(0.75 * local_mean + 0.25 * cross)
    
    query = torch.randn(B, q_dim)
    w = torch.linspace(-1.0, 1.0, q_dim).view(1, -1)
    qproj = (query * w).sum(dim=1, keepdim=True)
    target = torch.sin(qproj) * factor + (factor ** 3)
    
    return support, query, target


# --- Epreuve 5 : Topologie Riemannienne (Le Cartographe) ---

def gen_curvature(B, N, x_dim, q_dim):
    """Manifold Detection : Sphère (Courbe) vs Plan (Plat) en 3D
    Teste si le réseau peut extraire la courbure intrinsèque (indépendante
    de la rotation et de la position dans l'espace 3D environnant).
    """
    assert x_dim >= 3, "Curvature needs x_dim >= 3 for 3D embedding"
    
    support = torch.zeros(B, N, x_dim)
    labels = torch.zeros(B, 1)
    
    half_B = B // 2
    
    # 1. PLAN (Courbure intrinsèque = 0)
    # Coordonnées x, y aléatoires entre -1 et 1
    plane_xy = torch.rand(half_B, N, 2) * 2 - 1.0
    support[:half_B, :, :2] = plane_xy
    labels[:half_B] = 0.0
    
    # 2. SPHERE (Courbure positive > 0)
    # Distribution uniforme sur la sphère (R=1)
    sphere_pts = torch.randn(B - half_B, N, 3)
    sphere_pts = sphere_pts / sphere_pts.norm(dim=-1, keepdim=True)
    support[half_B:, :, :3] = sphere_pts
    labels[half_B:] = 1.0
    
    # Rotation 3D aleatoire (Holonomie) pour detruire les reperes (Plan incline)
    for b in range(B):
        yaw, pitch, roll = torch.rand(3) * 2 * math.pi
        cy, sy = math.cos(yaw), math.sin(yaw)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cr, sr = math.cos(roll), math.sin(roll)
        
        Rz = torch.tensor([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        Ry = torch.tensor([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        Rx = torch.tensor([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
        Rot3D = Rz @ Ry @ Rx
        
        support[b, :, :3] = support[b, :, :3] @ Rot3D.T
        
    # Bruit d'epaisseur (la surface n'est pas infiniment fine)
    support[:, :, :3] += torch.randn(B, N, 3) * 0.05
    
    if x_dim > 3:
        support[:, :, 3:] = torch.randn(B, N, x_dim - 3) * 0.1
        
    # Shuffling (Topology pure, pas de sequence)
    for b in range(B):
        idx = torch.randperm(N)
        support[b] = support[b, idx]
        
    query = torch.randn(B, q_dim)
    
    return support, query, labels


# --- Epreuve 6 : Graphes de Probabilite (Le Jardin des Chemins) ---

def gen_probability(B, N, x_dim, q_dim):
    """Detection de l'Aleatoire Ontologique : Deterministe vs Arbre Branchant
    Compare une sequence deterministe (ligne droite) a un processus de
    Monte Carlo (Random Walk Binomial avec bifurcations).
    """
    support = torch.zeros(B, N, x_dim)
    labels = torch.zeros(B, 1)
    
    half_B = B // 2
    
    # 1. DETERMINISTE (Target 0.0) -> Chemin unique, pas de bifurcation
    for b in range(half_B):
        # Point de depart aleatoire
        start = torch.randn(x_dim)
        # Vecteur de vitesse constant
        velocity = torch.randn(x_dim) * 0.1
        for t in range(N):
            support[b, t] = start + velocity * t
    labels[:half_B] = 0.0
    
    # 2. STOCHASTIQUE BRANCHANT (Target 1.0) -> Arbre Binomial de Possibles
    for b in range(half_B, B):
        start = torch.randn(x_dim)
        velocity = torch.randn(x_dim) * 0.1
        support[b, 0] = start
        for t in range(1, N):
            # Le hasard ontologique : +V ou -V (Bifurcation a multi=2)
            sign = torch.randint(0, 2, (x_dim,)).float() * 2.0 - 1.0
            support[b, t] = support[b, t-1] + velocity * sign
    labels[half_B:] = 1.0
    
    # Bruit de mesure infime pour ne pas confondre avec le Brouillard
    support += torch.randn_like(support) * 0.01
    
    query = torch.randn(B, q_dim)
    
    return support, query, labels



# --- Epreuve 8 : Automate Cellulaire Complexe (Majorite iteree) ---

def gen_automaton_hard(B, N, x_dim, q_dim):
    """
    Automate Cellulaire Majoritaire itere 5 fois.
    Pour chaque cellule (avec conditions periodiques), le nouvel etat
    est 1 si la cellule et ses deux voisins forment une majorite de 1.
    
    C'est lineairement separable par etape (faisable par W_trans de la Forge),
    mais impossible a fitter globalement pour une simple Spline continue en 1 passe
    a cause de l'effet papillon unrolled (effet domino discret sur 5 etapes).
    """
    q = torch.randint(0, 2, (B, q_dim)).float()
    
    def apply_majority(state):
        left = torch.roll(state, shifts=1, dims=1)
        right = torch.roll(state, shifts=-1, dims=1)
        # Somme des 3 voisins >= 2
        return ((state + left + right) >= 2.0).float()
    
    state = q.clone()
    for _ in range(5):
        state = apply_majority(state)
        
    # Target : le premier bit (ou la somme modulo 2)
    target = state[:, 0:1]
    
    support = torch.zeros(B, N, x_dim)
    for i in range(N):
        s_init = torch.randint(0, 2, (B, q_dim)).float()
        s_state = s_init.clone()
        for _ in range(5):
            s_state = apply_majority(s_state)
        s_target = s_state[:, 0:1]
        
        # Ajout de bruit au support pour forcer Z a generaliser
        support[:, i, :q_dim] = s_init + 0.05 * torch.randn_like(s_init)
        if x_dim > q_dim:
            support[:, i, q_dim:q_dim+1] = s_target
            
    return support, q, target




# --- Epreuve 11 : Automate 32 Bits (Anti-Spline) ---

def gen_automaton_32bit(B, N, x_dim, q_dim):
    """
    Automate Cellulaire (XOR-shift x3) sur 32 bits.
    La dimension Q_DIM est 32, ce qui donne 2^32 = 4.29 milliards d'etats possibles.
    Une memoire Spline continue (64 neurones) echouera mathematiquement a 
    memoriser la table de verite. 
    Seul le "Constructeur" (9eme Expert) peut survivre en detectant la signature 
    algorithmique discrete et iteree de la donnee !
    """
    assert q_dim == 32, "Automate32 necessite q_dim=32"
    q = torch.randint(0, 2, (B, q_dim)).float()
    
    state = q.clone()
    for _ in range(3):
        shifted_l = torch.roll(state, shifts=1, dims=1)
        state = ((state.long() ^ shifted_l.long())).float()
        
    target = state.sum(dim=1, keepdim=True) % 2
    
    support = torch.zeros(B, N, x_dim)
    for i in range(N):
        s_init = torch.randint(0, 2, (B, q_dim)).float()
        s_state = s_init.clone()
        for _ in range(3):
            shifted_l = torch.roll(s_state, shifts=1, dims=1)
            s_state = ((s_state.long() ^ shifted_l.long())).float()
        s_target = s_state.sum(dim=1, keepdim=True) % 2
        
        support[:, i, :q_dim] = s_init + 0.05 * torch.randn_like(s_init)
        if x_dim > q_dim:
            support[:, i, q_dim:q_dim+1] = s_target
            
    return support, q, target


# --- Boucle d'entrainement avec diagnostics ---

def train_with_debug(model, gen_fn, name, epochs=500, B=32, N=32,
                     x_dim=4, q_dim=4, lr=1e-3, device='cpu'):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    model.to(device)
    model.train()
    
    best_loss = float('inf')
    best_state = None
    final_loss = 0.0
    
    for epoch in range(1, epochs + 1):
        support, query, target = gen_fn(B, N, x_dim, q_dim)
        support, query, target = support.to(device), query.to(device), target.to(device)
        
        optimizer.zero_grad()
        pred, z = model(support, query)
        loss = criterion(pred, target)
        loss.backward()
        
        # Gradient clipping pour la stabilite
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        current_loss = loss.item()
        final_loss = current_loss
        
        # Sauvegarde du meilleur cerveau
        if current_loss < best_loss:
            best_loss = current_loss
            # Ne sauvegarde pas toutes les epochs pour eviter overhead, juste les percées
            best_state = copy.deepcopy(model.state_dict())
        
        if epoch == 1 or epoch % 50 == 0 or epoch == epochs:
            z_norm = z.detach().norm(dim=1).mean().item()
            print(f"\n--- Epoch {epoch:03d} (LR: {lr:.2e}) ---")
            print(f"  Loss MSE    : {current_loss:.6f}")
            print(f"  ||Z|| moyen : {z_norm:.4f}")
            if epoch == 1 and hasattr(model.eye, '_debug_num_alive'):
                n_alive = model.eye._debug_num_alive.mean().item()
                print(f"  [Loterie Initiale] Experts survivants a l'Epoch 1 : {n_alive:.2f}/8")
    
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"\n  >> Restauration du meilleur checkpoint (Loss: {best_loss:.6f})")
        final_loss = best_loss
    
    print(f"\n  [RESULTAT FINAL - {name}]")
    print(f"  Loss Finale  : {final_loss:.6f}")
    
    return final_loss


def diagnose_routing(model, gen_fn, name, x_dim, q_dim, device):
    """Diagnostic : montre quelle combinaison d'experts le reseau a choisi."""
    EXPERT_NAMES = ['Argile', 'Couteau', 'Tisserand', 'Brouillard',
                    'Chronos', 'Geometre', 'Cartographe', 'Jardin', 'Constructeur']
    SPLINE_NAMES = ['Basique', 'Hermite', 'Rationnelle']
    
    model.eval()
    with torch.no_grad():
        support, query, target = gen_fn(64, 8, x_dim, q_dim)
        support, query = support.to(device), query.to(device)
        pred, z = model(support, query)
    
    # --- PRESELECTEUR CONTRASTIF (Etage 1) ---
    if hasattr(model.eye, '_debug_preselector'):
        ps = model.eye._debug_preselector.mean(dim=0)  # (8,)
        n_alive = model.eye._debug_num_alive.mean().item()
        n_recipes = model.eye._debug_num_valid_recipes.mean().item()
        print(f"\n  [DIAGNOSTIC PRESELECTEUR — {name}]")
        print(f"  Experts Survivants : {n_alive:.1f}/{len(EXPERT_NAMES)}  |  Recettes Valides : {n_recipes:.0f}/{model.eye.router.num_combos}")
        print(f"  {'Expert':<15s} {'Score':>8s}  {'Statut':>10s}")
        print(f"  {'-'*15} {'-'*8}  {'-'*10}")
        for i, (nom, val) in enumerate(zip(EXPERT_NAMES, ps.tolist())):
            statut = 'VIVANT' if val > 0 else 'ELIMINE'
            print(f"  {nom:<15s} {val:>8.4f}  {statut:>10s}")
    
    # --- L'OCTOGONE (Experts Dynamiques) ---
    if hasattr(model.eye, '_debug_gates_voies'):
        g8 = model.eye._debug_gates_voies.mean(dim=0)  # (num_voies,) moyenne sur le batch
        blend_oct = model.eye._debug_blend.mean().item()
        print(f"\n  [DIAGNOSTIC OCTOGONE — {name}]")
        print(f"  Blend (Soft/Hard) : {blend_oct:.4f}  (0=Hard pur, 1=Soft pur)")
        print(f"  {'Expert':<15s} {'Poids':>8s}  {'Barre':>20s}")
        print(f"  {'-'*15} {'-'*8}  {'-'*20}")
        for i, (nom, val) in enumerate(zip(EXPERT_NAMES, g8.tolist())):
            bar = '#' * int(val * 20)
            status = ' ACTIF' if val > 0.1 else ''
            print(f"  {nom:<15s} {val:>8.4f}  {bar:<20s}{status}")
    
    # --- L'ARGILE POLYMATHE (3 Splines) ---
    if hasattr(model.eye, 'argile') and hasattr(model.eye.argile, '_debug_gates_3'):
        g3 = model.eye.argile._debug_gates_3.mean(dim=0)  # (3,)
        blend_arg = model.eye.argile._debug_blend.mean().item()
        print(f"\n  [DIAGNOSTIC ARGILE — {name}]")
        print(f"  Blend (Soft/Hard) : {blend_arg:.4f}")
        print(f"  {'Spline':<15s} {'Poids':>8s}  {'Barre':>20s}")
        print(f"  {'-'*15} {'-'*8}  {'-'*20}")
        for i, (nom, val) in enumerate(zip(SPLINE_NAMES, g3.tolist())):
            bar = '#' * int(val * 20)
            status = ' ACTIF' if val > 0.1 else ''
            print(f"  {nom:<15s} {val:>8.4f}  {bar:<20s}{status}")
    
    model.train()


def run_single_seed(seed: int, device: str, args):
    """Execute les epreuves selectionnees pour un seed donne."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    X_DIM, Q_DIM, Z_DIM = 4, 4, 36
    results = {}
    
    all_epreuves = [
        ('Affine',    'LOI AFFINE (Continu)',                gen_affine),
        ('AntiFuite', 'ANTI-FUITE (Non-lineaire + Locale)',  gen_antileak),
        ('XOR',       'XOR (Logique Booleenne)',             gen_xor),
        ('Chiralite',   'CHIRALITE (Orientation / Signe)',     gen_chirality),
        ('Courbure',    'COURBURE RIEMANNIENNE (Manifold)',    gen_curvature),
        ('Hasard',      'ARBRES DE PROBABILITE (Monte Carlo)', gen_probability),
        ('AutomateHard','AUTOMATE COMPLEXE (Majorite Iteree)', gen_automaton_hard),
        ('Automate32',  'AUTOMATE 32-BITS (Anti-Spline)',      gen_automaton_32bit),
    ]
    
    epreuves = []
    for key, label, gen_fn in all_epreuves:
        if args.only and args.only.lower() != key.lower(): continue
        if key == 'Affine' and args.no_affine: continue
        if key == 'AntiFuite' and args.no_antileak: continue
        if key == 'XOR' and args.no_xor: continue
        if key == 'Chiralite' and args.no_chirality: continue
        if key == 'Courbure' and args.no_curvature: continue
        if key == 'Hasard' and args.no_probability: continue
        if key == 'AutomateHard' and args.no_automatehard: continue
        if key == 'Automate32' and args.no_automate32: continue
        epreuves.append((key, label, gen_fn))
        
    if not epreuves:
        print("  [!] Aucune épreuve sélectionnée.")
        return {}
    
    for key, label, gen_fn in epreuves:
        print(f"\n{'='*70}")
        print(f"  EPREUVE: {label}")
        print(f"{'='*70}")
        
        # L'espace 32 bits a besoin de 33 pour le support (input + target)
        cur_q_dim = Q_DIM
        cur_x_dim = X_DIM
        if key == 'Automate32':
            cur_q_dim, cur_x_dim = 32, 33
            
        cur_z_dim = 36  # Toujours 36 (divisible par 9 voies)
        
        model = AIN(x_dim=cur_x_dim, z_dim=Z_DIM, query_dim=cur_q_dim, out_dim=1, hidden=64)
        results[key] = train_with_debug(
            model, gen_fn, key.upper(), epochs=500, x_dim=cur_x_dim, q_dim=cur_q_dim, device=device
        )
        # Diagnostic : quelle combinaison a-t-il choisi ?
        diagnose_routing(model, gen_fn, key, cur_x_dim, cur_q_dim, device)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Demo AIN - Tests Selection")
    parser.add_argument('--seeds', type=int, default=5, help="Nombre de seeds à executer")
    parser.add_argument('--log', action='store_true', help="Sauvegarder les logs dans un fichier .txt")
    parser.add_argument('--only', type=str, default="", help="Executer UNIQUEMENT ce test (ex: 'xor', 'chiralite')")
    parser.add_argument('--no-affine', action='store_true', help="Desactiver le test Affine")
    parser.add_argument('--no-antileak', action='store_true', help="Desactiver le test AntiFuite")
    parser.add_argument('--no-xor', action='store_true', help="Desactiver le test XOR")
    parser.add_argument('--no-chirality', action='store_true', help="Desactiver le test Chiralite")
    parser.add_argument('--no-curvature', action='store_true', help="Desactiver le test Courbure")
    parser.add_argument('--no-probability', action='store_true', help="Desactiver le test Hasard")
    parser.add_argument('--no-automatehard', action='store_true', help="Desactiver le test AutomateHard")
    parser.add_argument('--no-automate32', action='store_true', help="Desactiver le test Automate 32 Bits")
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # === LOG FICHIER ===
    tee = None
    log_path = None
    if args.log:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_name = f"demo_ain_log_{timestamp}_{args.seeds}seeds.txt"
        log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), log_name)
        tee = TeeWriter(log_path)
        sys.stdout = tee
    
    # =========================================================================
    # NOMBRE DE SEEDS (Reproductibilite Statistique)
    # =========================================================================
    SEED_COUNT = args.seeds
    
    print("=" * 70)
    print("  DEMO DIAGNOSTIQUE — ADAPTIVE INVARIANT NEURON (AIN)")
    print("  Le Neurone d'Argile : Zero Prior, Matrice Combinatoire Creuse")
    print(f"  Nombre de Seeds : {SEED_COUNT}")
    if args.only: print(f"  Filtre : Epreuve '{args.only}'")
    print("=" * 70)
    
    all_results = []
    
    for s in range(SEED_COUNT):
        seed = 42 + s * 1000
        print(f"\n{'#'*70}")
        print(f"  SEED {s+1}/{SEED_COUNT} (torch.manual_seed={seed})")
        print(f"{'#'*70}")
        results = run_single_seed(seed, device, args)
        all_results.append(results)
        
        # Tableau pour ce seed
        print(f"\n{'='*70}")
        print(f"  TABLEAU SEED {s+1} (seed={seed})")
        print(f"{'='*70}")
        print(f"  {'Epreuve':<25s} {'Loss':>10s}")
        print(f"  {'-'*25} {'-'*10}")
        for name, loss in results.items():
            print(f"  {name:<25s} {loss:>10.6f}")
    
    # =========================================================================
    # TABLEAU RECAPITULATIF MULTI-SEEDS
    # =========================================================================
    epreuve_keys = list(all_results[0].keys())
    
    print(f"\n\n{'='*70}")
    print(f"  TABLEAU RECAPITULATIF MULTI-SEEDS ({SEED_COUNT} seeds)")
    print(f"{'='*70}")
    
    # Header
    header = f"  {'Epreuve':<15s}"
    for s in range(SEED_COUNT):
        header += f" {'Seed '+str(s+1):>10s}"
    header += f" {'Moyenne':>10s}"
    print(header)
    print(f"  {'-'*15} " + " ".join(['-'*10] * (SEED_COUNT + 1)))
    
    for key in epreuve_keys:
        row = f"  {key:<15s}"
        vals = [all_results[s][key] for s in range(SEED_COUNT)]
        for v in vals:
            row += f" {v:>10.6f}"
        avg = sum(vals) / len(vals)
        row += f" {avg:>10.6f}"
        print(row)
    
    print(f"\n[ANALYSE]")
    print(f"  - Le AIN n'a AUCUN expert pre-cable. L'Octogone est emergent.")
    print(f"  - Blend Penalty : DESACTIVEE (La MSE seule guide le Curriculum).")
    print(f"  - Architecture : Gating Hybride + Combinatoire 2^N + Anti-Cecite 1%")
    
    # === FERMETURE DU LOG ===
    if tee is not None:
        print(f"\n[LOG] Fichier sauvegarde : {log_path}")
        tee.close()

if __name__ == '__main__':
    main()
