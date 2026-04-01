"""
==============================================================================
  ADAPTIVE INVARIANT NEURON (AIN) — Le Neurone d'Argile
  
  5 Mains Ontologiques. Zero Expert Pre-cable.
  
  Architecture :
  1. Oeil d'Argile (EmergentEncoder) : 5 voies ontologiques
     - L'Argile (Spline Adaptive)    → le Continu
     - Le Couteau (Spin/Ising)       → le Discret
     - Le Tisserand (Attention)      → le Relationnel
     - Le Brouillard (Variationnel)  → le Stochastique
     - Le Chronos (Flux Temporel)    → le Dynamique
     - Un Micro-Gating (Softmax + Temperature) dose le melange
  2. Forge Adaptive (AdaptiveHyperForge) : Z -> Poids + Grilles + Coefficients
  3. Muscle d'Argile (AdaptiveEffector) : Execution avec fonctions forgees
  
  5 priors ontologiques minimaux :
  - "L'univers contient du continu"      (Kolmogorov-Arnold)
  - "L'univers contient du discret"      (Ising / Spins)
  - "Les choses interagissent"           (Attention par paires)
  - "L'univers contient du flou"         (Encodage variationnel)
  - "L'information s'ecoule causalement" (Flux temporel / Prigogine)
==============================================================================
"""

import atexit
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from program_bank import ProgramBank, compute_context_signature


# ==============================================================================
#   1. LA MAIN D'ARGILE : AdaptiveSplineLayer (+ Masque Dynamique)
# ==============================================================================

class AdaptiveSplineLayer(nn.Module):
    """Couche a Spline Adaptative — La Main d'Argile.
    
    Chaque connexion est une courbe de Spline dont :
    - Les POSITIONS des noeuds (grille) sont apprises
    - Les HAUTEURS des noeuds (coefficients) sont apprises
    - Les MASQUES de vitalite (knot_alive) sont appris (Naissance/Mort dynamique)
    
    La grille est triee a chaque forward pour garantir la monotonie.
    Les noeuds dont le masque sigmoid tombe sous le seuil sont 'eteints' :
    leur coefficient est force a zero, simulant la mort du noeud.
    """
    def __init__(self, in_features: int, out_features: int, num_knots: int = 12):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_knots = num_knots
        
        # --- Grille MOBILE : les positions des noeuds sont apprises ---
        self.grid = nn.Parameter(
            torch.linspace(-2.0, 2.0, num_knots).unsqueeze(0).expand(in_features, num_knots).clone()
        )  # (in_features, K)
        
        # --- Coefficients appris : la hauteur de chaque noeud ---
        self.coeffs = nn.Parameter(
            torch.randn(in_features, num_knots) * 0.1
        )  # (in_features, K)
        
        # --- Masque de Vitalite : chaque noeud peut naitre ou mourir ---
        # Initialise a +3 (sigmoid ~= 0.95, tous vivants au depart)
        self.knot_alive = nn.Parameter(
            torch.ones(in_features, num_knots) * 3.0
        )  # (in_features, K)
        
        # --- Projection lineaire vers l'espace de sortie ---
        self.proj = nn.Linear(in_features, out_features, bias=True)
        
        # --- Couche residuelle lineaire (pour la stabilite du gradient) ---
        self.residual = nn.Linear(in_features, out_features, bias=False)
    
    def _eval_adaptive_spline(self, x: torch.Tensor) -> torch.Tensor:
        """Evalue la Spline Adaptative via interpolation lineaire + masque.
        
        x: (..., in_features) -> retourne (..., in_features)
        """
        # Trier la grille pour garantir l'ordre croissant
        sorted_grid, sort_idx = self.grid.sort(dim=1)  # (in_features, K)
        
        # Trier les coefficients et masques dans le meme ordre
        sorted_coeffs = self.coeffs.gather(1, sort_idx)  # (in_features, K)
        sorted_alive = self.knot_alive.gather(1, sort_idx)  # (in_features, K)
        
        # Masque de Vitalite : sigmoid -> [0, 1]. Les noeuds morts tendent vers 0.
        alive_mask = torch.sigmoid(sorted_alive)  # (in_features, K)
        masked_coeffs = sorted_coeffs * alive_mask  # Noeuds morts -> coeffs ~= 0
        
        # Sauvegarder la forme originale
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.in_features)  # (M, in_features)
        M = x_flat.shape[0]
        
        # Clamp x entre les bornes de la grille
        grid_min = sorted_grid[:, 0]
        grid_max = sorted_grid[:, -1]
        x_clamped = torch.max(torch.min(x_flat, grid_max.unsqueeze(0)), grid_min.unsqueeze(0))
        
        K = self.num_knots
        grid_range = (grid_max - grid_min).clamp(min=1e-6)
        x_norm = (x_clamped - grid_min.unsqueeze(0)) / grid_range.unsqueeze(0) * (K - 1)
        
        idx_l = x_norm.long().clamp(0, K - 2)
        frac = x_norm - idx_l.float()
        
        sc_ex = masked_coeffs.unsqueeze(0).expand(M, -1, -1)
        c_left = sc_ex.gather(2, idx_l.unsqueeze(-1)).squeeze(-1)
        c_right = sc_ex.gather(2, (idx_l + 1).unsqueeze(-1)).squeeze(-1)
        
        result = c_left + frac * (c_right - c_left)
        
        return result.reshape(orig_shape)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (..., in_features) -> (..., out_features)
        """
        spline_out = self._eval_adaptive_spline(x)
        return self.proj(spline_out) + self.residual(x)


# ==============================================================================
#   1b. L'ARGILE VISQUEUSE : HermiteSplineLayer (Tangentes Apprenables)
# ==============================================================================

class HermiteSplineLayer(nn.Module):
    """Couche a Spline Cubique d'Hermite — L'Argile Visqueuse.
    
    Comme AdaptiveSplineLayer, mais apprend AUSSI la TANGENTE (pente)
    en chaque noeud. L'interpolation est cubique (lisse, C1-continue),
    pas lineaire (zigzag).
    
    Chaque connexion apprend :
    - grid     : Position X du noeud (mobile)
    - coeffs   : Hauteur Y du noeud
    - tangents : Pente M du noeud (derivee locale)
    - knot_alive : Masque de Vitalite (Naissance/Mort)
    
    Interpolation d'Hermite entre p0 et p1 :
        h(t) = (2t^3 - 3t^2 + 1)*p0 + (t^3 - 2t^2 + t)*m0*dx
             + (-2t^3 + 3t^2)*p1     + (t^3 - t^2)*m1*dx
    
    Cela permet de sculpter des courbes parfaitement lisses avec ~ 5 noeuds
    la ou la Spline lineaire basique en aurait besoin de ~ 20.
    """
    def __init__(self, in_features: int, out_features: int, num_knots: int = 12):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_knots = num_knots
        
        # Grille mobile
        self.grid = nn.Parameter(
            torch.linspace(-2.0, 2.0, num_knots).unsqueeze(0).expand(in_features, num_knots).clone()
        )
        # Hauteurs
        self.coeffs = nn.Parameter(
            torch.randn(in_features, num_knots) * 0.1
        )
        # TANGENTES (pentes apprenables en chaque noeud)
        self.tangents = nn.Parameter(
            torch.zeros(in_features, num_knots)
        )
        # Masque de Vitalite
        self.knot_alive = nn.Parameter(
            torch.ones(in_features, num_knots) * 3.0
        )
        
        self.proj = nn.Linear(in_features, out_features, bias=True)
        self.residual = nn.Linear(in_features, out_features, bias=False)
    
    def _eval_hermite_spline(self, x: torch.Tensor) -> torch.Tensor:
        """Evalue la Spline Cubique d'Hermite.
        
        x: (..., in_features) -> (..., in_features)
        """
        sorted_grid, sort_idx = self.grid.sort(dim=1)
        sorted_coeffs = self.coeffs.gather(1, sort_idx)
        sorted_tangents = self.tangents.gather(1, sort_idx)
        sorted_alive = self.knot_alive.gather(1, sort_idx)
        
        alive_mask = torch.sigmoid(sorted_alive)
        masked_coeffs = sorted_coeffs * alive_mask
        masked_tangents = sorted_tangents * alive_mask
        
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.in_features)
        M = x_flat.shape[0]
        K = self.num_knots
        
        grid_min = sorted_grid[:, 0]
        grid_max = sorted_grid[:, -1]
        x_clamped = torch.max(torch.min(x_flat, grid_max.unsqueeze(0)), grid_min.unsqueeze(0))
        
        grid_range = (grid_max - grid_min).clamp(min=1e-6)
        x_norm = (x_clamped - grid_min.unsqueeze(0)) / grid_range.unsqueeze(0) * (K - 1)
        
        idx_l = x_norm.long().clamp(0, K - 2)
        t = x_norm - idx_l.float()  # fraction in [0, 1]
        
        # Recuperer coefficients et tangentes gauche/droite
        mc_ex = masked_coeffs.unsqueeze(0).expand(M, -1, -1)
        mt_ex = masked_tangents.unsqueeze(0).expand(M, -1, -1)
        
        p0 = mc_ex.gather(2, idx_l.unsqueeze(-1)).squeeze(-1)        # hauteur gauche
        p1 = mc_ex.gather(2, (idx_l + 1).unsqueeze(-1)).squeeze(-1)  # hauteur droite
        m0 = mt_ex.gather(2, idx_l.unsqueeze(-1)).squeeze(-1)        # tangente gauche
        m1 = mt_ex.gather(2, (idx_l + 1).unsqueeze(-1)).squeeze(-1)  # tangente droite
        
        # Pas local (distance entre noeuds adjacents), normalise
        sg_ex = sorted_grid.unsqueeze(0).expand(M, -1, -1)
        x_left = sg_ex.gather(2, idx_l.unsqueeze(-1)).squeeze(-1)
        x_right = sg_ex.gather(2, (idx_l + 1).unsqueeze(-1)).squeeze(-1)
        dx = (x_right - x_left).clamp(min=1e-6)  # (M, in)
        
        # Polynomes de base d'Hermite
        t2 = t * t
        t3 = t2 * t
        h00 = 2*t3 - 3*t2 + 1   # val au noeud gauche
        h10 = t3 - 2*t2 + t      # tang au noeud gauche
        h01 = -2*t3 + 3*t2       # val au noeud droit
        h11 = t3 - t2             # tang au noeud droit
        
        result = h00 * p0 + h10 * m0 * dx + h01 * p1 + h11 * m1 * dx
        
        return result.reshape(orig_shape)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (..., in_features) -> (..., out_features)
        """
        spline_out = self._eval_hermite_spline(x)
        return self.proj(spline_out) + self.residual(x)


# ==============================================================================
#   1c. L'ARGILE RATIONNELLE : RationalSplineLayer (Masse Apprenable / NURBS)
# ==============================================================================

class RationalSplineLayer(nn.Module):
    """Couche a Spline Rationnelle — L'Argile Gravitationnelle.
    
    Comme HermiteSplineLayer, mais apprend UN POIDS (Masse W) par noeud.
    L'interpolation est une fraction rationnelle : le numerateur est
    la courbe d'Hermite ponderee, le denominateur est la somme des poids.
    
    Chaque connexion apprend :
    - grid      : Position X du noeud (mobile)
    - coeffs    : Hauteur Y du noeud
    - tangents  : Pente M du noeud (derivee locale)
    - weights   : Masse W du noeud (attraction gravitationnelle)
    - knot_alive: Masque de Vitalite (Naissance/Mort)
    
    La division par les poids permet de representer :
    - Des asymptotes (W -> 0 : singularite)
    - Des angles DURS (W >> 1 : attraction extreme, angle net)
    - Des coniques parfaites (cercles, ellipses, hyperboles)
    
    Prior ontologique : 'La continuite possede une densite variable.'
    """
    def __init__(self, in_features: int, out_features: int, num_knots: int = 12):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_knots = num_knots
        
        # Grille mobile
        self.grid = nn.Parameter(
            torch.linspace(-2.0, 2.0, num_knots).unsqueeze(0).expand(in_features, num_knots).clone()
        )
        # Hauteurs
        self.coeffs = nn.Parameter(
            torch.randn(in_features, num_knots) * 0.1
        )
        # Tangentes (pentes apprenables)
        self.tangents = nn.Parameter(
            torch.zeros(in_features, num_knots)
        )
        # POIDS / MASSE (la nouveaute NURBS) — init a 1.0 = neutre
        self.weights = nn.Parameter(
            torch.ones(in_features, num_knots)
        )
        # Masque de Vitalite
        self.knot_alive = nn.Parameter(
            torch.ones(in_features, num_knots) * 3.0
        )
        
        self.proj = nn.Linear(in_features, out_features, bias=True)
        self.residual = nn.Linear(in_features, out_features, bias=False)
    
    def _eval_rational_spline(self, x: torch.Tensor) -> torch.Tensor:
        """Evalue la Spline Rationnelle (NURBS 1D).
        
        La courbe rationnelle = sum(w_i * H_i * B_i) / sum(w_i * B_i)
        ou H_i sont les valeurs d'Hermite et B_i les fonctions de base.
        
        x: (..., in_features) -> (..., in_features)
        """
        sorted_grid, sort_idx = self.grid.sort(dim=1)
        sorted_coeffs = self.coeffs.gather(1, sort_idx)
        sorted_tangents = self.tangents.gather(1, sort_idx)
        sorted_weights = self.weights.gather(1, sort_idx)
        sorted_alive = self.knot_alive.gather(1, sort_idx)
        
        alive_mask = torch.sigmoid(sorted_alive)
        masked_coeffs = sorted_coeffs * alive_mask
        masked_tangents = sorted_tangents * alive_mask
        # Poids positifs + masque de vitalite
        positive_weights = F.softplus(sorted_weights) * alive_mask + 1e-8
        
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.in_features)
        M = x_flat.shape[0]
        K = self.num_knots
        
        grid_min = sorted_grid[:, 0]
        grid_max = sorted_grid[:, -1]
        x_clamped = torch.max(torch.min(x_flat, grid_max.unsqueeze(0)), grid_min.unsqueeze(0))
        
        grid_range = (grid_max - grid_min).clamp(min=1e-6)
        x_norm = (x_clamped - grid_min.unsqueeze(0)) / grid_range.unsqueeze(0) * (K - 1)
        
        idx_l = x_norm.long().clamp(0, K - 2)
        t = x_norm - idx_l.float()  # fraction in [0, 1]
        
        # Recuperer tous les parametres gauche/droite
        mc_ex = masked_coeffs.unsqueeze(0).expand(M, -1, -1)
        mt_ex = masked_tangents.unsqueeze(0).expand(M, -1, -1)
        pw_ex = positive_weights.unsqueeze(0).expand(M, -1, -1)
        
        p0 = mc_ex.gather(2, idx_l.unsqueeze(-1)).squeeze(-1)
        p1 = mc_ex.gather(2, (idx_l + 1).unsqueeze(-1)).squeeze(-1)
        m0 = mt_ex.gather(2, idx_l.unsqueeze(-1)).squeeze(-1)
        m1 = mt_ex.gather(2, (idx_l + 1).unsqueeze(-1)).squeeze(-1)
        w0 = pw_ex.gather(2, idx_l.unsqueeze(-1)).squeeze(-1)
        w1 = pw_ex.gather(2, (idx_l + 1).unsqueeze(-1)).squeeze(-1)
        
        # Pas local
        sg_ex = sorted_grid.unsqueeze(0).expand(M, -1, -1)
        x_left = sg_ex.gather(2, idx_l.unsqueeze(-1)).squeeze(-1)
        x_right = sg_ex.gather(2, (idx_l + 1).unsqueeze(-1)).squeeze(-1)
        dx = (x_right - x_left).clamp(min=1e-6)
        
        # Polynomes de base d'Hermite
        t2 = t * t
        t3 = t2 * t
        h00 = 2*t3 - 3*t2 + 1
        h10 = t3 - 2*t2 + t
        h01 = -2*t3 + 3*t2
        h11 = t3 - t2
        
        # Interpolation d'Hermite classique
        hermite_val = h00 * p0 + h10 * m0 * dx + h01 * p1 + h11 * m1 * dx
        
        # Pondération rationnelle (NURBS) :
        # Le poids interpole lineairement entre w0 et w1
        w_interp = (1 - t) * w0 + t * w1  # (M, in_features)
        
        # Numerateur = hermite * poids interpole
        # Denominateur = poids interpole (normalisation)
        # Cela permet de "tirer" la courbe vers les noeuds lourds
        result = hermite_val * w_interp / w_interp.clamp(min=1e-8)
        
        # NOTE : sous cette forme simplifiee, w s'annule numerateur/denominateur
        # La vraie puissance vient du fait que w modifie le gradient de la grille
        # et permet au reseau de creer des concentrations de noeuds
        # Pour la version complete, on pondère chaque contribution de base séparément :
        num = h00 * p0 * w0 + h10 * m0 * dx * w0 + h01 * p1 * w1 + h11 * m1 * dx * w1
        den = (h00 + h10) * w0 + (h01 + h11) * w1
        den = den.clamp(min=1e-8)
        result = num / den
        
        return result.reshape(orig_shape)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (..., in_features) -> (..., out_features)
        """
        spline_out = self._eval_rational_spline(x)
        return self.proj(spline_out) + self.residual(x)


# ==============================================================================
#   1d. LE ROUTEUR ONTOLOGIQUE : OntologicalRouter (Gating Contextuel)
# ==============================================================================

class OntologicalRouter(nn.Module):
    """Routeur Ontologique — Gating Contextuel Dynamique.
    
    Au lieu d'apprendre une preference globale (statique) sur tout le dataset,
    ce routeur prend un Support (pool) en entree et decide A LA VOLEE :
    - gate_logits : quelles mains activer (Softmax/Hard dynamique)
    - gate_blend  : s'il faut etre nuancé (Soft) ou exclusif (Hard)
    
    Cela resout le paradoxe Set vs Sequence : le reseau voit par lui-meme
    si la donnee se comporte comme un Set chaotique (et eteint le Chronos)
    ou comme une Sequence causale.
    """
    def __init__(self, in_features: int, num_voies: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.LayerNorm(in_features // 2),
            nn.SiLU(),
            nn.Linear(in_features // 2, num_voies + 1) # N logits + 1 blend
        )
        
    def forward(self, x_pool: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x_pool: (B, in_features)
        Returns:
            gate_logits: (B, num_voies)
            gate_blend: (B, 1) -> avant sigmoid
        """
        out = self.net(x_pool)
        return out[:, :-1], out[:, -1:]


# ==============================================================================
#   1e. L'ARGILE POLYMATHE : PolymathArgile (Basique vs Visqueuse vs Rationnelle)
# ==============================================================================

class PolymathArgile(nn.Module):
    """L'Argile Polymathe — La Main Emergente Ultime.
    
    Met en competition 3 variantes de la continuite (La Trinite) :
    1. Spline Adaptative (Argile Basique)     : interpolation lineaire, rapide
    2. Hermite Spline   (Argile Visqueuse)    : interpolation cubique, lisse
    3. Rational Spline  (Argile Rationnelle)  : interpolation NURBS, singularites
    
    Un Gating Hybride interne (Soft + Hard) decide dynamiquement
    quelle forme de continuite sert le mieux la donnee.
    
    Trinite de la Continuite :
    - Basique     = la ligne droite (efficacite brute)
    - Visqueuse   = la courbe lisse (cinematique fluide)
    - Rationnelle = la courbe a masse (singularites, asymptotes, angles durs)
    """
    NUM_VARIANTS = 3
    
    def __init__(self, in_features: int, hidden: int, out_features: int, num_knots: int = 12):
        super().__init__()
        
        # Variante A : Argile Basique (interpolation lineaire par morceaux)
        self.base_s1 = AdaptiveSplineLayer(in_features, hidden, num_knots)
        self.base_s2 = AdaptiveSplineLayer(hidden, hidden, num_knots)
        self.base_s3 = AdaptiveSplineLayer(hidden, out_features, num_knots)
        self.base_n1 = nn.LayerNorm(hidden)
        self.base_n2 = nn.LayerNorm(hidden)
        
        # Variante B : Argile Visqueuse (interpolation cubique d'Hermite)
        self.herm_s1 = HermiteSplineLayer(in_features, hidden, num_knots)
        self.herm_s2 = HermiteSplineLayer(hidden, hidden, num_knots)
        self.herm_s3 = HermiteSplineLayer(hidden, out_features, num_knots)
        self.herm_n1 = nn.LayerNorm(hidden)
        self.herm_n2 = nn.LayerNorm(hidden)
        
        # Variante C : Argile Rationnelle (NURBS — masses apprenables)
        self.rat_s1 = RationalSplineLayer(in_features, hidden, num_knots)
        self.rat_s2 = RationalSplineLayer(hidden, hidden, num_knots)
        self.rat_s3 = RationalSplineLayer(hidden, out_features, num_knots)
        self.rat_n1 = nn.LayerNorm(hidden)
        self.rat_n2 = nn.LayerNorm(hidden)
        
        # Gating Hybride CONTEXTUEL Combinatoire (2^3 = 8 recettes)
        self.num_combos = 2 ** self.NUM_VARIANTS  # 8
        self.router = OntologicalRouter(in_features=in_features * 4, num_voies=self.num_combos)
        self.argile_temp = nn.Parameter(torch.ones(1) * 1.0)
        
        # Generation de la matrice des 8 combinaisons binaires (8, 3)
        combo_indices = torch.arange(self.num_combos).unsqueeze(1)
        bits = 2 ** torch.arange(self.NUM_VARIANTS - 1, -1, -1)
        self.register_buffer('combinations', combo_indices.bitwise_and(bits).bool().float())
        
        # Initialisation Pylint des variables de debug
        self._debug_gates_3 = None
        self._debug_blend = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (..., in_features) -> (..., out_features)
        """
        # Variante A : Argile Basique
        h_a = self.base_s1(x)
        h_a = self.base_n1(h_a)
        h_a = self.base_s2(h_a)
        h_a = self.base_n2(h_a)
        z_base = self.base_s3(h_a)
        
        # Variante B : Argile Visqueuse (Hermite)
        h_b = self.herm_s1(x)
        h_b = self.herm_n1(h_b)
        h_b = self.herm_s2(h_b)
        h_b = self.herm_n2(h_b)
        z_herm = self.herm_s3(h_b)
        
        # Variante C : Argile Rationnelle (NURBS)
        h_c = self.rat_s1(x)
        h_c = self.rat_n1(h_c)
        h_c = self.rat_s2(h_c)
        h_c = self.rat_n2(h_c)
        z_rat = self.rat_s3(h_c)
        
        # Gating Hybride Contextuel Interne (Trinite)
        x_mean = x.mean(dim=1)
        x_std = x.std(dim=1, unbiased=False)
        if x.shape[1] >= 2:
            x_delta = x[:, 1:, :] - x[:, :-1, :]
            delta_mean = x_delta.mean(dim=1)
            delta_std = x_delta.std(dim=1, unbiased=False)
        else:
            delta_mean = torch.zeros_like(x_mean)
            delta_std = torch.zeros_like(x_std)
        router_input = torch.cat([x_mean, x_std, delta_mean, delta_std], dim=-1)
        argile_logits, argile_blend = self.router(router_input)  # (B, 8), (B, 1)
        
        temp = torch.clamp(self.argile_temp, min=0.01)
        
        soft_g = F.softmax(argile_logits / temp, dim=-1)  # (B, 8)
        
        hard_idx = torch.argmax(argile_logits, dim=-1)
        hard_g_raw = F.one_hot(hard_idx, num_classes=self.num_combos).float().to(x.device)
        hard_g = hard_g_raw - soft_g.detach() + soft_g  # (B, 8)
        
        blend = torch.sigmoid(argile_blend)  # (B, 1)
        
        gates_8 = blend * soft_g + (1.0 - blend) * hard_g  # (B, 8)
        
        # Projection Combinatoire (8 Recettes -> 3 Masques Binaires)
        gates_3 = torch.matmul(gates_8, self.combinations.to(x.device))  # (B, 3)
        
        # === ANTI-CÉCITÉ : 1% de survie pour les splines non-selectionnees ===
        gates_3 = torch.clamp(gates_3, min=0.01)
        
        # Debug : stocker les masques pour inspection
        self._debug_gates_3 = gates_3.detach()  # pylint: disable=attribute-defined-outside-init
        self._debug_blend = blend.detach()      # pylint: disable=attribute-defined-outside-init
 
        # Broadcast safe : z_* sont (B, N, out_features), gates_3 est (B, 3)
        gates = gates_3.unsqueeze(1)  # (B, 1, 3)
 
        return (gates[:, :, 0:1] * z_base + 
                gates[:, :, 1:2] * z_herm + 
                gates[:, :, 2:3] * z_rat)


# ==============================================================================
#   2. LE COUTEAU : SpinInteraction (Champ d'Ising Mean-Field)
# ==============================================================================


class EmergentSpinGlass(nn.Module):
    """Le Couteau V2 — Verre de Spin Emergent (Ising Pair-a-Pair).
    
    L'ancien Couteau n'utilisait qu'un Champ Moyen (chaque noeud ressentant
    identiquement la moyenne globale). Cela elimine la possibilite de "frustration"
    asymetrique critique.
    
    Ici, on cree un couplage bilineaire complet :
    1. Les noeuds sont projetes en spins continus.
    2. Une matrice d'interaction J dicte si deux spins s'attirent (ferromagnetique)
       ou se repoussent (anti-ferromagnetique).
    3. Chaque spin s_i ressent le champ local H_i = sum_j J_{ij} s_j.
    4. La dynamique locale = tanh(beta * (s_i + H_i)).
    
    Prior ontologique : Le "Discret" emerge de la polarisation concurrente (+1/-1)
    sous contraintes de frustration (metastabilite de type spin glass).
    """
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        
        # Projection vers le repere des spins
        self.spin_proj = nn.Linear(in_features, out_features)
        
        # Couplage global (Champ Moyen Externe)
        self.global_coupling = nn.Linear(out_features, out_features, bias=False)
        
        # Matrice de couplage emergent J (interaction Energie Pair-a-Pair)
        self.J_coupling = nn.Linear(out_features, out_features, bias=False)
        
        # Initialisation a Zero : 
        # Demarre comme un Pur Modele d'Ising Mean-Field (affinites uniformes)
        # Laisse la structure s'apprendre doucement vers un Verre de Spin.
        with torch.no_grad():
            self.J_coupling.weight.zero_()
        
        # Inverse Temperature (beta / force de cristallisation)
        self.beta = nn.Parameter(torch.ones(1) * 1.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, in_features) -> (B, N, out_features)
        """
        # 1. Projeter chaque noeud spatial en espace de spin
        s = self.spin_proj(x)  # (B, N, D), D = out_features
        
        # 2. Champ Moyen Global (Polarisation collective h_ext)
        # Indispensable pour briser la symetrie sur des problemes globaux comme XOR
        mean_field = s.mean(dim=1, keepdim=True)  # (B, 1, D)
        global_influence = self.global_coupling(mean_field)  # (B, 1, D)
        
        # 3. Calcul du Couplage par Paires (Frustration J_ij)
        # Pour interagir spatialement de maniere permutation-invariante, 
        # le champ est propulse a travers la connectivite J sur l'aggregation globale.
        #
        # Note : Pour echapper au pur 'Mean Field' tout en gardant o(N), 
        # on calcule les affinites energetiques s @ J @ s^T
        #
        # energy_matrix = s @ J  -> (B, N, D)
        energy_fields = self.J_coupling(s)  # (B, N, D)
        
        # affinities[i, j] = etat d'influence du noeud j sur le noeud i
        affinities = torch.bmm(energy_fields, s.transpose(1, 2))  # (B, N, N)
        
        # Normalisation pour equilibrer l'energie si N varie (Softmax thermodynamique)
        # Transforme l'affinite brute en vritable distribution d'energie de Boltzmann
        affinities = torch.softmax(affinities / (s.shape[2] ** 0.5), dim=-1)
        
        # Champ local ressenti par chaque spin
        # local_field[i] = sum_j affinities[i, j] * s[j]
        # L'utilisation du softmax normalise le champ a une valeur intensive O(1).
        local_field = torch.bmm(affinities, s)  # (B, N, D)
        
        # 4. Thermodynamique (Dynamique des etats)
        # Hamiltonien physique complet : s_i + h_ext + sum(J_ij s_j)
        # Activation : force le signal vers des etats discrets polarises (+1/-1)
        result = torch.tanh(self.beta * (s + global_influence + local_field))  # (B, N, D)
        
        return result

# ==============================================================================
#   3. LE TISSERAND : PairwiseAttention (Interactions Relationnelles)
# ==============================================================================

class PhysicalWeaver(nn.Module):
    """Le Tisserand Physique — Interaction par paires avec portee apprise.
    
    A l'inverse de l'Attention dense (Transformer) qui connecte tout a tout (artefact humain),
    le Tisserand respecte la physique : l'influence decroit avec la distance semantique.
    
    - Distance euclidienne apprise
    - Decroissance exponentielle (facteur lambda appris)
    - Sparsification de seuil (vrais zeros = pas d'action a distance infinie)
    """
    def __init__(self, in_features: int, out_features: int, head_dim: int = 16):
        super().__init__()
        self.head_dim = head_dim
        
        # Projections pour l'espace semantique
        self.proj_q = nn.Linear(in_features, head_dim, bias=False)
        self.proj_k = nn.Linear(in_features, head_dim, bias=False)
        self.proj_v = nn.Linear(in_features, out_features, bias=False)
        
        # Longueur d'ecrantage (portee de l'interaction), log_lambda pour stabilite
        self.log_lambda = nn.Parameter(torch.zeros(1))
        
        # Seuil d'interaction minimale
        self.threshold = 0.05
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, in_features) -> (B, N, out_features)
        """
        Q = self.proj_q(x)  # (B, N, head_dim)
        K = self.proj_k(x)  # (B, N, head_dim)
        V = self.proj_v(x)  # (B, N, out_features)
        
        # Matrice de distance euclidienne (plus stable via cdist)
        dist = torch.cdist(Q, K)  # (B, N, N)
        
        # Le carre de la distance
        dist_sq = dist ** 2
        
        # Decroissance exponentielle avec la distance (affinite douce)
        lambda_range = torch.exp(self.log_lambda).clamp(min=1e-3)
        logits = -dist_sq / (2.0 * lambda_range**2)
        
        # Normalisation locale (loi de conservation)
        # On utilise F.softmax car c'est mathematiquement identique a 
        # exp(logits) / sum(exp(logits)) mais numeriquement stable (evite l'underflow FP32).
        attn = F.softmax(logits, dim=-1)
        
        # Ponderation relationnelle
        out = torch.bmm(attn, V)  # (B, N, out_features)
        
        return out


# ==============================================================================
#   4. LE BROUILLARD : VariationalPath (Encodage Stochastique)
# ==============================================================================

class VariationalPath(nn.Module):
    """Le Brouillard — Encodage Variationnel.
    
    Au lieu de produire un Z deterministe, cette voie encode
    l'invariant comme une DISTRIBUTION (moyenne + variance).
    Puis echantillonne via le reparameterization trick.
    
    C'est la reconnaissance que certains invariants ne sont pas
    des points fixes mais des ZONES de l'espace latent.
    
    Prior ontologique : 'L'univers contient de l'incertitude irreductible.'
    """
    def __init__(self, in_features: int, hidden: int, z_dim: int):
        super().__init__()
        
        # Encodeur vers l'espace latent
        self.encoder = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU()
        )
        
        # Deux tetes : moyenne et log-variance
        self.mu_head = nn.Linear(hidden, z_dim)
        self.logvar_head = nn.Linear(hidden, z_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, in_features) -> z: (B, z_dim)
        
        Effectue le mean-pool puis encode en distribution.
        """
        # Mean-pool d'abord (invariance par permutation)
        x_pool = x.mean(dim=1)  # (B, in_features)
        
        h = self.encoder(x_pool)  # (B, hidden)
        
        mu = self.mu_head(h)             # (B, z_dim)
        logvar = self.logvar_head(h)     # (B, z_dim)
        
        # Reparameterization trick
        if self.training:
            std = (0.5 * logvar).exp()
            eps = torch.randn_like(std)
            z = mu + eps * std
        else:
            z = mu  # En inference, on utilise la moyenne pure
        
        return z


# ==============================================================================
#   5. LE CHRONOS : SplineFlow (ODE Temporelle Sculptee par l'Argile)
# ==============================================================================

class SplineFlow(nn.Module):
    """Le Chronos — Flux Temporel Pur, Sculpte par l'Argile.
    
    Traite les noeuds du support comme une SEQUENCE causale.
    Un etat cache (memoire) s'accumule noeud apres noeud,
    capturant les dependances temporelles et les trajectoires.
    
    ZERO porte artificielle (pas de Reset, pas de Update, pas de GRU).
    La mecanique du temps est ENTIEREMENT sculptee par des Splines Adaptatives.
    
    A chaque pas temporel t :
        dH = AdaptiveSpline([H_{t-1}, x_t])   <-- La Spline sculpte le changement
        H_t = H_{t-1} + dt * dH               <-- Integration d'Euler (ODE)
    
    Si la Spline apprend a produire des zeros -> la memoire est conservee (= Update gate)
    Si la Spline apprend un gradient negatif  -> la memoire est oubliee (= Reset gate)
    Si la Spline apprend une oscillation      -> la memoire pulse (= aucun RNN ne fait ca)
    
    Prior ontologique : 'L'information s'ecoule causalement (Passe -> Futur).'
    Le SEUL prior ici est la Fleche du Temps (Euler). La forme de l'ecoulement est libre.
    """
    def __init__(self, in_features: int, hidden: int, z_dim: int):
        super().__init__()
        self.hidden_dim = hidden
        
        # Projection d'entree
        self.input_proj = nn.Linear(in_features, hidden)
        
        # La Spline qui sculpte la dynamique temporelle
        # Elle recoit [H_{t-1}, x_t] concatenes et produit dH (le changement)
        self.flow_spline = AdaptiveSplineLayer(hidden * 2, hidden, num_knots=12)
        
        # Pas temporel apprenable : le Chronos decouvre sa propre vitesse
        self.dt = nn.Parameter(torch.ones(1) * 0.5)
        
        # Projection finale vers z_dim
        self.output_proj = nn.Linear(hidden, z_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, in_features) -> z: (B, z_dim)
        
        Scanne les N noeuds sequentiellement comme un flux temporel.
        """
        B, N, _ = x.shape
        
        # Projeter l'entree
        x_proj = self.input_proj(x)  # (B, N, hidden)
        
        # Etat cache initial : vierge (le Chronos nait sans memoire)
        h_t = torch.zeros(B, self.hidden_dim, device=x.device)  # (B, hidden)
        
        # Pas temporel (clamp pour la stabilite)
        dt = torch.clamp(self.dt, min=0.01, max=2.0)
        
        # Scanner les noeuds un par un (la fleche du temps)
        for t in range(N):
            x_t = x_proj[:, t, :]  # (B, hidden)
            
            # Concatenation [memoire_passee, perception_presente]
            combined = torch.cat([h_t, x_t], dim=1)  # (B, 2*hidden)
            
            # La Spline sculpte le changement (dH) — ZERO porte, ZERO sigmoid
            dH = self.flow_spline(combined)  # (B, hidden)
            
            # Integration d'Euler : le seul axiome est la fleche du temps
            h_t = h_t + dt * dH  # (B, hidden)
        
        # L'etat final contient toute la memoire causale
        z = self.output_proj(h_t)  # (B, z_dim)
        
        return z


# ==============================================================================
#   6. LE GEOMETRE : GeometerHand (Voie Bilineaire / Surfaces)
# ==============================================================================

class GeometerHand(nn.Module):
    """Le Geometre — La Voie des Surfaces (Couplage Bilineaire).
    
    L'AIN voit des Lignes (Splines 1D), des Points (Spin), des Liens (Attention).
    Mais l'univers contient aussi des SURFACES : des plans orientes, des aires,
    des rotations. Le Geometre donne au neurone la capacite de tisser
    des surfaces a partir de ses dimensions.
    
    Mecanisme : une matrice W apprenable couple les dimensions entre elles
    via un produit bilineaire h * (W @ h). Chaque sortie i calcule :
        z_i = h_i * sum_j(W_ij * h_j)
    
    Ce couplage est une forme bilineaire generalisee :
    - Si W apprend a etre SYMETRIQUE  -> le reseau voit des distances, energies
    - Si W apprend a etre ANTISYMETRIQUE -> le reseau voit des ORIENTATIONS, 
      des chiralites, des rotations (c'est le produit exterieur de Grassmann)
    - Si W reste quelconque -> melange libre (emergence pure)
    
    Prior ontologique : 'L'univers contient des Surfaces, pas juste des Lignes.'
    La NATURE de ces surfaces (symetrique ou chirale) est emergente.
    """
    def __init__(self, in_features: int, hidden: int, z_dim: int):
        super().__init__()
        
        # Projection par noeud vers un espace compact
        self.proj = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU()
        )
        
        # LA MATRICE DES SURFACES : W (hidden x hidden)
        # Initialisee proche de zero, le gradient decidera
        # si elle doit etre symetrique, antisymetrique, ou mixte.
        self.W = nn.Parameter(torch.randn(hidden, hidden) * 0.01)
        
        # Projection finale vers z_dim
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, z_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, in_features) -> z_geom: (B, z_dim)
        """
        # Projeter chaque noeud dans l'espace compact
        h = self.proj(x)  # (B, N, hidden)
        
        # Couplage bilineaire : h_i * (W @ h_i) pour chaque noeud
        # Wh[b,n,:] = sum_j W[i,j] * h[b,n,j]  pour chaque i
        Wh = torch.einsum('bni,ij->bnj', h, self.W)  # (B, N, hidden)
        
        # Surface = produit de Hadamard (element-wise)
        # surface[b,n,i] = h[b,n,i] * sum_j(W[i,j] * h[b,n,j])
        # C'est la forme bilineaire generalisee de Grassmann
        surface = h * Wh  # (B, N, hidden)
        
        # Aggregation invariante par permutation (Set topology)
        z_surface = surface.mean(dim=1)  # (B, hidden)
        
        # Projection vers z_dim
        z_geom = self.output_proj(z_surface)  # (B, z_dim)
        
        return z_geom


# ==============================================================================
#   7. LE CARTOGRAPHE : RiemannianWeaver (Courbure Emergente)
# ==============================================================================

class HolonomyCartographer(nn.Module):
    """Le Cartographe V2 — Courbure par transport parallele pur.
    
    L'ancien Cartographe (RiemannianWeaver) utilisait un GRUCell (artefact 
    temporel d'ingenierie) pour accumuler l'holonomie. 
    Ici, la courbure est detectee purement de maniere geometrique via le defaut 
    de fermeture d'une boucle : (Barycentre -> Noeud -> Barycentre)
    
    Mecanisme (Zero Prior Humain) :
    1. Barycentre spatial
    2. Vecteur de reference lie au centre
    3. Reseau de jauge (Network Gauge) pour simuler le transport sur la courbure
    4. C'est strict : pas de sequences LSTM/GRU.
    """
    def __init__(self, in_features: int, z_dim: int):
        super().__init__()
        self.z_dim = z_dim
        
        # Point central de reference
        self.tangent_proj = nn.Linear(in_features, in_features)
        
        # Reseau de connexion de Levi-Civita emergent (Comment le repere se tord).
        # Plus general et robuste qu'un literal tenseur 3D einsum.
        self.connection_gauge = nn.Sequential(
            nn.Linear(in_features * 2, in_features),
            nn.LayerNorm(in_features),
            nn.SiLU(),
            nn.Linear(in_features, in_features)
        )
        
        # Projection finale de l'holonomie vers l'espace invariant
        self.output_proj = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.SiLU(),
            nn.Linear(in_features, z_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, in_features) -> z_holonomy: (B, z_dim)
        """
        B, N, D = x.shape
        
        # 1. Barycentre (point de base de la topologie locale)
        center = x.mean(dim=1, keepdim=True)  # (B, 1, D)
        
        # 2. Vecteur de reference dans l'espace tangent au centre
        ref_vec = self.tangent_proj(center)  # (B, 1, D)
        
        # Le 'delta' : vecteur d'ecart
        delta = x - center  # (B, N, D)
        
        # 3. Transport geodesique (Barycentre -> Noeud)
        # Le transport parallele tord le vecteur en fonction du deplacement !
        gauge_input = torch.cat([delta, ref_vec.expand(-1, N, -1)], dim=-1)  # (B, N, 2*D)
        transport_step_1 = self.connection_gauge(gauge_input)  # (B, N, D)
        
        transported = ref_vec.expand(-1, N, -1) + transport_step_1  # (B, N, D)
        
        # Retour (Noeud -> Barycentre), un espace parfaitement plat inversera l'effet.
        # Sur un tore ou sphere, il y aura un defaut d'alignement.
        # delta s'inverse au retour (-delta)
        gauge_input_return = torch.cat([-delta, transported], dim=-1)
        transport_step_2 = self.connection_gauge(gauge_input_return)
        
        final_vector = transported + transport_step_2
        
        # 4. Holonomie V1 : Defaut de cloture de la boucle (Direction d'Anomalie)
        # Sur un espace plat (euclidien), final_vector revient inchange au ref_vec originel.
        holonomy = final_vector - ref_vec.expand(-1, N, -1)  # (B, N, D)
        holonomy_direction = holonomy.mean(dim=1)  # (B, D)
        
        # 5. Lemme de Gauss V2 : Spheroide de Ricci (Magnitude Absolue)
        # Aucun parametre appris ! Variance brute des distances au barycentre.
        # Dans un espace courbe, le volume (donc la variance radiale) devie du cas plat.
        dist_center = torch.norm(delta, dim=2)  # (B, N)
        ricci_scalar = torch.var(dist_center, dim=1, keepdim=True)  # (B, 1)
        
        # 6. V3 : Fusion Ontologique (Geometrie Complete = Scalaire x Direction)
        # Le vecteur indique comment la variete se tord (la forme de l'invariant).
        # Le scalaire module cette courbure proportionnellement a l'anomalie volumique pure (la densite topologique).
        z_riemann = self.output_proj(holonomy_direction) * ricci_scalar  # (B, z_dim)
        
        return z_riemann


# ==============================================================================
#   8. LE JARDIN DES CHEMINS : PathGarden (Arbre des Possibles)
# ==============================================================================

class PathGarden(nn.Module):
    """Le Jardin des Chemins — La Voie de la Combinatoire Probabiliste.
    
    L'AIN voit le Continu (Argile), l'Incertitude Gaussiene (Brouillard).
    Mais le VRAI hasard physique et logique est discret et ramifie
    (arbres de probabilite, Monte Carlo, jeux, mechanique statistique).
    
    Cette Voie detecte "L'Univers qui se Deplie" :
    Au lieu de coder les regles de Bayes ou des factorielles, elle
    regarde si le dataset contient des **bifurcations** (les points de choix),
    des **multiplicites** (le nombre de branches), et des **convergences**
    (la symetrie des resultats).
    
    Z_garden = La "Geometrie" de l'arbre des possibles.
    """
    def __init__(self, in_features: int, hidden: int, z_dim: int, 
                 max_branching: int = 8):
        super().__init__()
        self.max_branching = max_branching
        
        # Graines : points de divergence potentiels dans l'espace des données
        self.seeds = nn.Parameter(torch.randn(max_branching, in_features) * 0.3)
        
        # Détecteur de bifurcation : choix ou consequence ?
        # in = seed (in_features) + local_mean (in_features) + local_var (1) = 2*in_features + 1
        self.bifurcation_detector = nn.Sequential(
            nn.Linear(in_features * 2 + 1, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2)  # [score_bifurcation, multiplicité]
        )
        
        # Détecteur de convergence : plusieurs chemins convergent-ils ?
        self.convergence_detector = nn.MultiheadAttention(
            embed_dim=in_features, num_heads=2 if in_features % 2 == 0 else 1, batch_first=True
        )
        
        # Estimateur de volume (densité globale)
        self.volume_estimator = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1)
        )
        
        # Synthese de l'arbre
        self.tree_fusion = nn.Linear(max_branching * 3 + 1, z_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        
        bifurcation_scores = []
        multiplicities = []
        
        for i in range(self.max_branching):
            seed = self.seeds[i]
            s = seed.unsqueeze(0).unsqueeze(0).expand(B, N, -1)  # (B, N, D)
            
            # Distance aux données (pertinence de la graine)
            dists = torch.norm(x - s, dim=-1, keepdim=True)  # (B, N, 1)
            
            # Quantile pour determiner le voisinage
            near = (dists < dists.quantile(0.3, dim=1, keepdim=True)).float()
            local_count = near.sum(dim=1).clamp(min=1)
            
            local_mean = (x * near).sum(dim=1) / local_count  # (B, D)
            local_var  = (((x - local_mean.unsqueeze(1)) ** 2) * near).sum(dim=1).sum(dim=-1, keepdim=True) / local_count  # (B, 1)
            
            # Détection proprement dite
            s_flat = seed.unsqueeze(0).expand(B, -1)  # (B, D)
            bif_input = torch.cat([s_flat, local_mean, local_var], dim=-1)  # (B, 2D + 1)
            bif, mult = self.bifurcation_detector(bif_input).chunk(2, dim=-1)  # (B, 1)
            
            bifurcation_scores.append(torch.sigmoid(bif))
            multiplicities.append(F.softplus(mult) + 1.0)  # Multiplicité >= 1
            
        bif_stack = torch.cat(bifurcation_scores, dim=-1)  # (B, max_branching)
        mult_stack = torch.cat(multiplicities, dim=-1)    # (B, max_branching)
        
        # Convergence (Symétrie des probabilités)
        attn_out, attn_weights = self.convergence_detector(x, x, x)
        convergence_score = attn_weights.max(dim=-1)[0].mean(dim=-1, keepdim=True)  # (B, 1)
        
        # Densité (inverse du volume)
        global_density = torch.sigmoid(self.volume_estimator(x.mean(dim=1)))  # (B, 1)
        
        # Synthèse Z_garden
        tree_features = torch.cat([
            bif_stack,
            mult_stack,
            convergence_score.expand(-1, self.max_branching),
            global_density
        ], dim=-1)  # (B, 3 * max_branching + 1)
        
        z_garden = self.tree_fusion(tree_features)  # (B, z_dim)
        return z_garden


# ==============================================================================
#   9a. SPARSEMAX : Projection Euclidienne sur le Simplexe (Vrais Zeros)
# ==============================================================================

def sparsemax(z: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Sparsemax (Martins & Astudillo, 2016).
    
    Contrairement au Softmax qui ne produit jamais de vrais zeros,
    le Sparsemax projette le vecteur sur le simplexe de probabilites
    via la distance Euclidienne. Les composantes faibles sont ecrasees
    a exactement 0.0, permettant une ablation chirurgicale.
    """
    z_sorted, _ = z.sort(dim=dim, descending=True)
    cumsum = z_sorted.cumsum(dim=dim)
    d = z.shape[dim]
    k = torch.arange(1, d + 1, device=z.device, dtype=z.dtype)
    # Reshape k pour le broadcasting
    shape = [1] * z.dim()
    shape[dim] = d
    k = k.view(shape)
    support = (1 + k * z_sorted > cumsum).float()
    k_z = support.sum(dim=dim, keepdim=True)
    # tau = (sum des elements dans le support - 1) / k_z
    tau_sum = (z_sorted * support).sum(dim=dim, keepdim=True)
    tau = (tau_sum - 1.0) / k_z.clamp(min=1)
    return torch.clamp(z - tau, min=0)


# ==============================================================================
#   9b. LE PRESELECTEUR CONTRASTIF : ContrastivePreselector (Metric Learning)
# ==============================================================================

class ContrastivePreselector(nn.Module):
    """Preselecteur Contrastif — Evalue les experts individuellement.
    
    Chaque expert possede un vecteur-concept apprenable (sa Carte d'Identite).
    Le module encode le probleme X en un vecteur-requete, puis calcule la
    similarite cosinus entre la requete et les 8 concepts experts.
    Sparsemax met a 0.0 strict les experts non-qualifies.
    
    C'est l'Etage 1 du Gating Hierarchique : l'elagage brutal.
    """
    def __init__(self, in_features: int, num_voies: int, concept_dim: int = 32):
        super().__init__()
        self.num_voies = num_voies
        self.concept_dim = concept_dim
        
        # Encodeur de la "question" (le probleme X)
        self.query_proj = nn.Sequential(
            nn.Linear(in_features, concept_dim),
            nn.SiLU(),
        )
        
        # Carte d'identite conceptuelle de chaque expert (apprenable)
        self.expert_concepts = nn.Parameter(
            torch.randn(num_voies, concept_dim) * 0.1
        )
        
        # Temperature Cosinus apprenable
        self.cos_temp = nn.Parameter(torch.ones(1) * 0.5)
    
    def forward(self, x_summary: torch.Tensor) -> torch.Tensor:
        """x_summary: (B, in_features) -> expert_scores: (B, num_voies)"""
        query = F.normalize(self.query_proj(x_summary), dim=-1)   # (B, concept_dim)
        concepts = F.normalize(self.expert_concepts, dim=-1)       # (num_voies, concept_dim)
        temp = torch.clamp(self.cos_temp, min=0.01)
        similarity = torch.matmul(query, concepts.T) / temp        # (B, num_voies)
        return sparsemax(similarity, dim=-1)                        # (B, num_voies) avec vrais 0


# ==============================================================================
#   9c. LE CERVEAU COMBINATOIRE : CombinatorialRouter (Matrice Creuse Anti-Cécité)
# ==============================================================================

class CombinatorialRouter(nn.Module):
    """Le Cerveau Gating — Routeur Combinatoire Recurrent.
    
    Evalue dynamiquement les 2^N combinaisons (recettes) possibles d'ontologies 
    (où N=num_voies) et assigne des probabilites a ces "recettes" entieres
    plutot qu'aux experts individuels. Le but est de trouver des synergies.
    Le Z retourne est une Matrice Creuse (Sparse Graph) aplatie.
    Les experts non-selectionnes gardent 1% de signal (Anti-Cecite)
    pour eviter la starvation du gradient.
    
    RECURRENT : Accepte un etat Z courant pour raffiner ses decisions
    a travers plusieurs iterations (Routing by Agreement).
    
    SYSTEM 2 : Accepte un signal d'erreur (error_feedback) provenant de
    l'Effecteur pour penaliser les coalitions echouees et explorer de
    nouvelles recettes (Trial-and-Error algorithmique).
    """
    def __init__(self, in_features: int, num_voies: int, hidden: int = 32, z_dim: int = 32):
        super().__init__()
        self.hidden_dim = hidden
        self.num_voies = num_voies
        self.num_combos = 2 ** num_voies  # 2^N dynamique
        
        self.wx = nn.Linear(in_features, hidden)
        self.wh = nn.Linear(hidden, hidden, bias=False)
        self.act = nn.SiLU()
        
        # Feedback Z : projection de Z_flat dans l'espace du routeur
        self.z_proj = nn.Linear(z_dim, hidden, bias=False)
        
        # System 2 : Conscience des experts vivants
        # Projette le masque binaire (num_voies,) dans l'espace cache du routeur
        # pour concentrer la reflexion sur les sous-arbres legaux.
        self.proj_alive = nn.Linear(num_voies, hidden, bias=False)
        
        # System 2 : Feedback d'erreur de l'Effecteur
        # Projette le signal d'erreur (scalaire par batch -> broadcast)
        # dans l'espace cache pour repousser le routeur hors de l'attracteur echoue.
        self.error_proj = nn.Linear(1, hidden, bias=False)
        
        self.head_gates = nn.Linear(hidden, self.num_combos)
        self.head_blend = nn.Linear(hidden, 1)
        
        # Generation de la matrice des combinaisons binaires (num_combos, num_voies)
        # Ex pour 8 voies, ligne 3 = [0, 0, 0, 0, 0, 0, 1, 1]
        combo_indices = torch.arange(self.num_combos).unsqueeze(1)
        bits = 2 ** torch.arange(num_voies - 1, -1, -1)
        self.combinations = combo_indices.bitwise_and(bits).bool().float()  # (num_combos, num_voies)
        
    def forward(self, x_seq: torch.Tensor, expert_alive: torch.Tensor = None,
                z_current: torch.Tensor = None, error_feedback: torch.Tensor = None):
        B, N, _ = x_seq.shape
        h = torch.zeros(B, self.hidden_dim, device=x_seq.device)
        for t in range(N):
            h = self.act(self.wx(x_seq[:, t, :]) + self.wh(h))
        
        # System 2 : Injecter la conscience des experts vivants dans h
        # Avant meme de reflechir aux recettes, le routeur sait quels outils
        # sont disponibles — comme un chirurgien qui voit ses instruments.
        if expert_alive is not None:
            h = h + self.proj_alive(expert_alive)
            h = self.act(h)
        
        # Feedback recurrent : integrer l'etat Z courant dans la decision
        if z_current is not None:
            h = h + self.z_proj(z_current)
            h = self.act(h)
        
        # System 2 : Feedback d'erreur de l'Effecteur (Trial-and-Error)
        # Si l'erreur est elevee, le routeur doit etre pousse hors de son
        # attracteur actuel pour explorer une nouvelle coalition.
        if error_feedback is not None:
            h = h + self.error_proj(error_feedback)
            h = self.act(h)
        
        gate_logits = self.head_gates(h) # (B, num_combos)
        gate_blend = self.head_blend(h)  # (B, 1)
        return gate_logits, gate_blend


# ==============================================================================
#   7b. LE CONSTRUCTEUR : EmergentCellularAutomaton (9eme Main Ontologique)
# ==============================================================================

class EmergentCellularAutomaton(nn.Module):
    """Le Constructeur — Automate Cellulaire sur Graphe Emergent.
    
    Prior ontologique : "L'information evolue par regles LOCALES sur un substrat."
    (von Neumann, modele 2 : l'automate cellulaire)
    
    Zero-prior HUMAIN : Pas de grille 2D, pas de Conv2d, pas de voisinage impose.
    Le voisinage est EMERGENT : decouvert dans l'espace des features par k-NN.
    
    Distinction cruciale avec les autres experts :
    - Chronos  = Temps continu, fleche causale, ODE (s_{t+1} = s_t + dt * f(s_t))
    - Tisserand = Relations GLOBALES pair-a-pair (attention dense, pas de localite)
    - Constructeur = Espace discret, LOCALITE, regles synchrones
                     (s_{t+1}[i] = R(s_t[i], Voisins(i)))
    
    La localite est le prior irreductible qui manquait aux 8 autres.
    """
    def __init__(self, in_features: int, hidden: int, out_features: int,
                 num_steps: int = 3, k_neighbors: int = 4):
        super().__init__()
        self.num_steps = num_steps
        self.k = k_neighbors  # Degre de connectivite locale (pas une geometrie !)
        
        # Detecteur de voisinage EMERGENT (pas spatial, semantique)
        # La "distance" est apprise dans l'espace des features
        self.neighbor_proj = nn.Linear(in_features, hidden)
        
        # Regle de transition locale (comme Rule 110, mais apprise)
        # Entree : [moi, mes voisins agreges] -> Sortie : mon nouvel etat
        self.local_rule = nn.Sequential(
            nn.Linear(in_features + hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, in_features)
        )
        
        # Projection vers l'espace du slot
        self.out_proj = nn.Linear(in_features, out_features)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, in_features) — un SET, pas une grille !
        
        Retourne: (B, out_features)
        """
        B, N, D = x.shape
        state = x  # (B, N, D)
        
        # Limiter k au nombre de points disponibles
        k = min(self.k, N)
        
        for step in range(self.num_steps):
            # Etape 1 : Calculer la "proximite informationnelle" (emergente)
            # Pas de coordonnees (x,y) imposees ! Juste une metrique apprise.
            keys = self.neighbor_proj(state)  # (B, N, hidden)
            
            # Similarite cosinus normalisee pour trouver les voisins
            keys_norm = F.normalize(keys, dim=-1)
            sim = torch.bmm(keys_norm, keys_norm.transpose(1, 2))  # (B, N, N)
            
            # Trouver les k plus proches voisins (connectivite LOCALE dynamique)
            # C'est la traduction "sans prior" du "voisinage de Moore"
            _, topk_idx = torch.topk(sim, k=k, dim=-1)  # (B, N, k)
            
            # Agreger les voisins par moyenne
            idx_expanded = topk_idx.unsqueeze(-1).expand(-1, -1, -1, D)  # (B, N, k, D)
            state_expanded = state.unsqueeze(1).expand(-1, N, -1, -1)     # (B, N, N, D)
            gathered = torch.gather(state_expanded, dim=2, index=idx_expanded)  # (B, N, k, D)
            
            # Projection locale des voisins dans l'espace hidden
            neighbor_agg = self.neighbor_proj(gathered.mean(dim=2))  # (B, N, hidden)
            
            # Etape 2 : Regle de transition locale (synchrone)
            # new_state[i] = f(state[i], voisins[i])
            # C'est PUR von Neumann : chaque cellule ne voit que son voisinage local
            combined = torch.cat([state, neighbor_agg], dim=-1)  # (B, N, D + hidden)
            update = self.local_rule(combined)  # (B, N, D)
            
            # Mise a jour synchrone (toutes les cellules en meme temps)
            # Contrairement au Chronos (sequentiel temporel), c'est spatial parallele
            state = state + update  # Residuel pour la stabilite
        
        # Pooling final pour l'invariant
        return self.out_proj(state.mean(dim=1))  # (B, out_features)



# ==============================================================================
#   8. L'OEIL D'ARGILE : EmergentEncoder (9 Voies Ontologiques)
# ==============================================================================

class EmergentEncoder(nn.Module):
    """L'Oeil d'Argile — Encodeur a 9 Voies Ontologiques (Le Nonagone Combinatoire).
    
    9 voies paralleles qui couvrent la totalite de l'information observable.
    Matrice Combinatoire Creuse : Le routeur evalue 2^N recettes possibles.
    Chaque expert produit un vecteur dans son espace reserve.
    Les experts non-selectionnes sont etouffes a 1% (Anti-Cecite).
    
    Support (B, N, x_dim) -> Z_flat (B, z_dim)
    """
    NUM_VOIES = 9
    
    def __init__(self, x_dim: int, hidden: int = 64, z_dim: int = 36):
        super().__init__()
        
        # Matrice Combinatoire Creuse : chaque expert a son espace reserve
        assert z_dim % self.NUM_VOIES == 0, "z_dim doit etre un multiple de 9 (Sparse Combinatorial)"
        self.slot_dim = z_dim // self.NUM_VOIES  # 4
        
        # Pre-digestion quadratique minimale
        self.expand_dim = x_dim * 2
        
        # === VOIE 1 : ARGILE POLYMATHE (le Continu — Basique vs Hermite vs Rationnelle) ===
        self.argile = PolymathArgile(self.expand_dim, hidden, self.slot_dim, num_knots=12)
        
        # === VOIE 2 : SPIN (Le Couteau — le Verre de Spin Emergent) ===
        self.spin1 = EmergentSpinGlass(self.expand_dim, hidden)
        self.spin2 = EmergentSpinGlass(hidden, self.slot_dim)
        self.norm_p1 = nn.LayerNorm(hidden)
        
        # === VOIE 3 : INTERACTION PHYSIQUE (Le Tisserand — localite relationnelle) ===
        self.attn1 = PhysicalWeaver(self.expand_dim, hidden, head_dim=16)
        self.attn2 = PhysicalWeaver(hidden, self.slot_dim, head_dim=16)
        self.norm_a1 = nn.LayerNorm(hidden)
        
        # === VOIE 4 : VARIATIONNEL (Le Brouillard — le Stochastique Continu) ===
        self.variational = VariationalPath(self.expand_dim, hidden, self.slot_dim)
        
        # === VOIE 5 : CAUSALITE (Le Chronos — le Dynamique) ===
        self.chronos = SplineFlow(self.expand_dim, hidden, self.slot_dim)
        
        # === VOIE 6 : GEOMETRIE (Le Geometre — le Chiral/Surfaces) ===
        self.geometer = GeometerHand(self.expand_dim, hidden, self.slot_dim)
        
        # === VOIE 7 : COURBURE (Le Cartographe — l'Holonomie emergente) ===
        self.weaver = HolonomyCartographer(self.expand_dim, self.slot_dim)
        
        # === VOIE 8 : GRAPHES DE HASARD (Le Jardin — la Combinatoire) ===
        self.garden = PathGarden(self.expand_dim, hidden, self.slot_dim)
        
        # === VOIE 9 : ALGORITHME (Le Constructeur — le Cellulaire Emergent) ===
        self.constructor = EmergentCellularAutomaton(self.expand_dim, hidden, self.slot_dim)
        
        # === PRESELECTEUR CONTRASTIF (Etage 1 du Gating Hierarchique) ===
        self.preselector = ContrastivePreselector(
            in_features=self.expand_dim * 2,  # mean + std
            num_voies=self.NUM_VOIES,
            concept_dim=32
        )
        
        # === ROUTEUR COMBINATOIRE RECURRENT (2^N Recettes x Anti-Cécité x T iterations) ===
        self.router = CombinatorialRouter(in_features=self.expand_dim, num_voies=self.NUM_VOIES, z_dim=z_dim)
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)
        self.T = 3  # Nombre d'iterations de reflexion (Routing by Agreement)
        
        # Initialisation Pylint des variables de debug
        self._debug_gates_voies = None
        self._debug_blend = None
        self._debug_preselector = None
        self._debug_num_alive = None
        self._debug_num_valid_recipes = None

    def prepare_voies(self, x: torch.Tensor) -> dict:
        """Etape 1 : Calculer les Z de chaque voie + System 1 (Preselecteur).
        
        Cette methode est appelee UNE SEULE FOIS par forward pass.
        Elle retourne un dictionnaire contenant tout le contexte necessaire
        pour que step_routing puisse etre appele iterativement.
        
        x: (B, N, x_dim)
        Retourne: dict avec z_stack, x_exp, expert_alive, recipe_mask, etc.
        """
        B, N, D = x.shape
        
        # Pre-digestion : [x, x^2]
        x_exp = torch.cat([x, x ** 2], dim=-1)  # (B, N, 2*D)
        
        # --- Voie 1 : Argile Polymathe (Basique vs Hermite) ---
        z_argile_raw = self.argile(x_exp)   # (B, N, slot_dim)
        z_spline = z_argile_raw.mean(dim=1) # (B, slot_dim)
        
        # --- Voie 2 : Spin (Le Couteau) ---
        h_p = self.spin1(x_exp)
        h_p = self.norm_p1(h_p)
        h_p = self.spin2(h_p)       # (B, N, slot_dim)
        z_spin = h_p.mean(dim=1)    # (B, slot_dim)
        
        # --- Voie 3 : Attention (Le Tisserand) ---
        h_a = self.attn1(x_exp)
        h_a = self.norm_a1(h_a)
        h_a = self.attn2(h_a)       # (B, N, slot_dim)
        z_attn = h_a.mean(dim=1)    # (B, slot_dim)
        
        # --- Voie 4 : Variationnel (Le Brouillard) ---
        z_var = self.variational(x_exp)  # (B, slot_dim)
        
        # --- Voie 5 : Temporel (Le Chronos) ---
        z_time = self.chronos(x_exp)  # (B, slot_dim)
        
        # --- Voie 6 : Surface (Le Géomètre) ---
        z_geom = self.geometer(x_exp)  # (B, slot_dim)
        
        # --- Voie 7 : Courbure (Le Cartographe) ---
        z_riemann = self.weaver(x_exp)  # (B, slot_dim)
        
        # --- Voie 8 : Graphe de Hasard (Le Jardin) ---
        z_garden = self.garden(x_exp)  # (B, slot_dim)
        
        # --- Voie 9 : Algorithme (Le Constructeur) ---
        z_algo = self.constructor(x_exp)  # (B, slot_dim)
        
        # L'Encyclopédie des 9 Experts -> Dimension : (B, 9, slot_dim)
        z_stack = torch.stack([z_spline, z_spin, z_attn, z_var, z_time, z_geom, z_riemann, z_garden, z_algo], dim=1)
        
        z_dim = self.NUM_VOIES * self.slot_dim
        
        # =================================================================
        # ÉTAGE 1 : PRÉSÉLECTION CONTRASTIVE (System 1 — Sparsemax)
        # =================================================================
        x_summary = torch.cat([x_exp.mean(dim=1), x_exp.std(dim=1)], dim=-1)  # (B, expand_dim*2)
        expert_scores = self.preselector(x_summary)   # (B, num_voies)
        expert_alive = (expert_scores > 0).float()     # (B, num_voies) masque binaire
        
        # Quelles recettes sont legales ?
        combinations = self.router.combinations.to(x.device)  # (num_combos, num_voies)
        needed = combinations.unsqueeze(0)                     # (1, num_combos, num_voies)
        alive_expanded = expert_alive.unsqueeze(1)             # (B, 1, num_voies)
        recipe_valid = ((needed * (1.0 - alive_expanded)).sum(dim=-1) == 0).float()  # (B, num_combos)
        recipe_mask = torch.where(recipe_valid > 0, 0.0, -1e9)  # (B, num_combos)
        # Interdire la recette vide (combo 0 = aucun expert) : attracteur degenerate
        # qui force gates_voies -> 0 puis clamp -> 0.01 partout.
        recipe_mask[:, 0] = -1e9
        
        # Etat initial du Z : concatenation brute (pas encore filtre)
        z_flat = z_stack.view(B, z_dim)
        
        return {
            'x_exp': x_exp,
            'z_stack': z_stack,
            'z_flat': z_flat,
            'z_dim': z_dim,
            'expert_alive': expert_alive,
            'expert_scores': expert_scores,
            'recipe_valid': recipe_valid,
            'recipe_mask': recipe_mask,
            'device': x.device,
        }
    
    def step_routing(self, ctx: dict, error_feedback: torch.Tensor = None) -> torch.Tensor:
        """Etape 2 : Une iteration de routing (System 2).
        
        Peut etre appelee plusieurs fois avec un error_feedback different
        pour implementer le Trial-and-Error algorithmique.
        
        ctx: dictionnaire retourne par prepare_voies
        error_feedback: (B, 1) signal d'erreur de l'Effecteur (None au premier appel)
        
        Retourne: z_flat (B, z_dim) — le vecteur Z filtre par le gating
        """
        x_exp = ctx['x_exp']
        z_stack = ctx['z_stack']
        z_flat = ctx['z_flat']
        z_dim = ctx['z_dim']
        expert_alive = ctx['expert_alive']
        recipe_mask = ctx['recipe_mask']
        device = ctx['device']
        
        temp = torch.clamp(self.temperature, min=0.01)
        num_combos = self.router.num_combos
        combinations = self.router.combinations.to(device)
        
        # System 2 : Le routeur recoit le feedback d'erreur de l'Effecteur
        gate_logits, gate_blend = self.router(
            x_exp, expert_alive=expert_alive,
            z_current=z_flat, error_feedback=error_feedback
        )
        
        # Masquage des recettes illegales (System 1)
        gate_logits = gate_logits + recipe_mask
        
        soft_gates = F.softmax(gate_logits / temp, dim=-1)  # (B, num_combos)
        
        hard_idx = torch.argmax(gate_logits, dim=-1)  # (B,)
        hard_gates_raw = F.one_hot(hard_idx, num_classes=num_combos).float().to(device)
        hard_gates = hard_gates_raw - soft_gates.detach() + soft_gates
        
        blend = torch.sigmoid(gate_blend)  # (B, 1)
        
        gates_combos = blend * soft_gates + (1.0 - blend) * hard_gates  # (B, num_combos)
        
        gates_voies = torch.matmul(gates_combos, combinations)  # (B, num_voies)
        
        # === ANTI-CÉCITÉ (filet de securite) ===
        gates_voies = torch.clamp(gates_voies, min=0.01)
        
        # Mise a jour de Z
        z_sparse = gates_voies.unsqueeze(-1) * z_stack  # (B, num_voies, slot_dim)
        z_flat_new = z_sparse.view(z_flat.shape[0], z_dim)
        
        # Mettre a jour le contexte pour le prochain appel
        ctx['z_flat'] = z_flat_new
        
        # Debug : stocker les masques pour inspection
        self._debug_gates_voies = gates_voies.detach()
        self._debug_blend = blend.detach()
        self._debug_preselector = ctx['expert_scores'].detach()
        self._debug_num_alive = ctx['expert_alive'].sum(dim=-1).detach()
        self._debug_num_valid_recipes = ctx['recipe_valid'].sum(dim=-1).detach()
        
        return z_flat_new
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Interface backward-compatible : prepare_voies + T iterations de step_routing.
        
        x: (B, N, x_dim) -> z: (B, z_dim)
        """
        ctx = self.prepare_voies(x)
        
        # Boucle de coherence interne (sans error_feedback = mode classique)
        for t_iter in range(self.T):
            z_flat = self.step_routing(ctx, error_feedback=None)
        
        return z_flat


# ==============================================================================
#   7. LA FORGE D'ARGILE : AdaptiveHyperForge
# ==============================================================================

class AdaptiveHyperForge(nn.Module):
    """La Forge d'Argile — Genere Poids + Grilles + Coefficients pour l'effecteur.
    
    A partir de l'invariant Z, cette forge imprime en temps reel :
    - Les poids lineaires (w1, b1, w2, b2) du muscle
    - Les POSITIONS des noeuds de la Spline du muscle (grille mobile forgee)
    - Les COEFFICIENTS de la Spline du muscle (hauteurs des noeuds)
    
    La Forge du AIN ne genere pas de fonctions d'activation figees (SiLU) :
    elle genere la FORME MEME de l'activation via la grille et les coefficients.
    """
    def __init__(self, z_dim: int, input_dim: int, output_dim: int,
                 hidden_p: int = 32, num_knots: int = 12):
        super().__init__()
        self.hidden_p = hidden_p
        self.num_knots = num_knots
        
        # Backbone de la forge
        self.backbone = nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU()
        )
        
        # Generateurs de parametres
        self.w1_gen = nn.Linear(128, hidden_p * input_dim)
        self.b1_gen = nn.Linear(128, hidden_p)
        self.w2_gen = nn.Linear(128, output_dim * hidden_p)
        self.b2_gen = nn.Linear(128, output_dim)
        
        # Generateurs de la Spline Forgee (grille mobile + coefficients)
        self.grid_gen = nn.Linear(128, hidden_p * num_knots)
        self.coeffs_gen = nn.Linear(128, hidden_p * num_knots)
    
    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        B = z.shape[0]
        
        feat = self.backbone(z)  # (B, 128)
        
        # Poids lineaires
        w1 = self.w1_gen(feat).view(B, self.hidden_p, -1)  # (B, hp, input)
        b1 = self.b1_gen(feat)                                # (B, hp)
        w2 = self.w2_gen(feat).view(B, -1, self.hidden_p)    # (B, out, hp)
        b2 = self.b2_gen(feat)                                # (B, out)
        
        # Grille mobile forgee (positions des noeuds)
        forged_grid = self.grid_gen(feat).view(B, self.hidden_p, self.num_knots)
        # Trier pour garantir l'ordre croissant
        forged_grid, sort_idx = forged_grid.sort(dim=2)
        
        # Coefficients forges (hauteurs aux noeuds), tries dans le meme ordre
        raw_coeffs = self.coeffs_gen(feat).view(B, self.hidden_p, self.num_knots)
        forged_coeffs = raw_coeffs.gather(2, sort_idx)
        
        return {
            'w1': w1, 'b1': b1,
            'w2': w2, 'b2': b2,
            'grid': forged_grid,       # (B, hp, K)
            'coeffs': forged_coeffs    # (B, hp, K)
        }


# ==============================================================================
#   8. LE MUSCLE D'ARGILE : AdaptiveEffector
# ==============================================================================

class AdaptiveEffector(nn.Module):
    """Le Muscle d'Argile — Effecteur a Spline Forgee.
    
    N'a AUCUN parametre interne. Ses poids ET ses fonctions d'activation
    lui sont entierement injectes par la Forge.
    
    Activation = Spline(h, grille_forgee, coeffs_forges)
    
    Pas de SiLU. Pas de ReLU. La forme de l'activation est sculptee
    par le gradient a travers la Forge.
    """
    def __init__(self):
        super().__init__()
    
    def _eval_forged_spline(self, x: torch.Tensor,
                             grid: torch.Tensor,
                             coeffs: torch.Tensor) -> torch.Tensor:
        """Evalue la Spline Forgee via interpolation lineaire.
        
        x:      (B, hidden_p)     -- les pre-activations
        grid:   (B, hidden_p, K)  -- la grille forgee (DEJA triee)
        coeffs: (B, hidden_p, K)  -- les coefficients forges (DEJA tries)
        
        Retourne: (B, hidden_p)
        """
        K = grid.shape[2]
        
        # Clamp dans les bornes de la grille (par batch et par neurone)
        grid_min = grid[:, :, 0]   # (B, hp)
        grid_max = grid[:, :, -1]  # (B, hp)
        x_clamped = torch.max(torch.min(x, grid_max), grid_min)
        
        # Normaliser dans [0, K-1]
        grid_range = (grid_max - grid_min).clamp(min=1e-6)
        x_norm = (x_clamped - grid_min) / grid_range * (K - 1)
        
        idx_l = x_norm.long().clamp(0, K - 2)       # (B, hp)
        frac = x_norm - idx_l.float()                 # (B, hp)
        
        # Gather les coefficients
        c_left = coeffs.gather(2, idx_l.unsqueeze(-1)).squeeze(-1)
        c_right = coeffs.gather(2, (idx_l + 1).unsqueeze(-1)).squeeze(-1)
        
        return c_left + frac * (c_right - c_left)
    
    def forward(self, query_x: torch.Tensor,
                forged: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        query_x: (B, input_dim)
        forged:  dict avec w1, b1, w2, b2, grid, coeffs
        """
        w1, b1 = forged['w1'], forged['b1']
        w2, b2 = forged['w2'], forged['b2']
        grid = forged['grid']
        coeffs = forged['coeffs']
        
        # Couche 1 : Transformation lineaire forgee
        h = torch.bmm(w1, query_x.unsqueeze(-1)).squeeze(-1) + b1  # (B, hp)
        
        # Activation : Spline Forgee PURE (ni SiLU, ni ReLU, ni rien de pre-cable)
        h = self._eval_forged_spline(h, grid, coeffs)  # (B, hp)
        
        # Connexion residuelle pour la stabilite
        h = h + b1 * 0.1  # Micro-skip
        
        # Couche 2 : Sortie lineaire
        out = torch.bmm(w2, h.unsqueeze(-1)).squeeze(-1) + b2  # (B, out_dim)
        
        return out


# ==============================================================================
#   9. L'ASSEMBLAGE FINAL : AIN (Adaptive Invariant Neuron)
# ==============================================================================

class AIN(nn.Module):
    """Adaptive Invariant Neuron (AIN) — Le Neurone d'Argile.
    
    5 Mains Ontologiques, Zero Expert Pre-cable.
    
    Architecture :
    1. Oeil d'Argile (EmergentEncoder) : 5 voies
       - L'Argile (Spline)       -> Continu
       - Le Couteau (Spin)        -> Discret
       - Le Tisserand (Attention) -> Relationnel
       - Le Brouillard (Var.)     -> Stochastique
       - Le Chronos (GRU)         -> Dynamique
       - Micro-Gating (Softmax + Temperature)
    2. Forge d'Argile (AdaptiveHyperForge) : Z -> Poids + Grilles + Coefficients
    3. Muscle d'Argile (AdaptiveEffector) : Execute avec fonctions forgees pures
    
    5 priors ontologiques minimaux :
    - 'L'univers contient du continu'
    - 'L'univers contient du discret'
    - 'Les choses interagissent'
    - 'L'univers contient du flou'
    - 'L'information s'ecoule causalement'
    """
    def __init__(self, x_dim: int, z_dim: int, query_dim: int, out_dim: int,
                 hidden: int = 64, bank_capacity: int = 1000,
                 model_name: Optional[str] = None,
                 use_multi_oracle: bool = True,
                 oracle_hidden: int = 32,
                 oracle_tau: float = 0.5):
        super().__init__()
        
        # L'Oeil d'Argile
        self.eye = EmergentEncoder(x_dim=x_dim, hidden=hidden, z_dim=z_dim)
        
        # La Forge d'Argile
        self.forge = AdaptiveHyperForge(
            z_dim=z_dim, input_dim=query_dim,
            output_dim=out_dim, hidden_p=32, num_knots=12
        )
        
        # Le Muscle d'Argile
        self.effector = AdaptiveEffector()

        self._out_dim = int(out_dim)
        self._query_dim = int(query_dim)
        self._use_multi_oracle = bool(use_multi_oracle)
        self._oracle_tau = float(oracle_tau)
        self._oracle_gate = nn.Sequential(
            nn.Linear(4 * x_dim, oracle_hidden),
            nn.SiLU(),
            nn.Linear(oracle_hidden, 4),
        )

        self._oracle_mask_q = nn.Sequential(
            nn.Linear(4 * x_dim, oracle_hidden),
            nn.SiLU(),
            nn.Linear(oracle_hidden, int(query_dim)),
        )
        self._oracle_mask_t = nn.Sequential(
            nn.Linear(4 * x_dim, oracle_hidden),
            nn.SiLU(),
            nn.Linear(oracle_hidden, int(out_dim)),
        )
        
        # === MEMOIRE A LONG TERME (Le ProgramBank organique) ===
        self.bank = ProgramBank(capacity=bank_capacity)
        self._auto_archive = True  # Desactivable pour les benchmarks purs
        
        # === PERSISTANCE AUTOMATIQUE ===
        self._best_loss = float('inf')
        self._model_name = model_name
        self._auto_save_path = None  # Chemin pour la sauvegarde automatique
        
        if model_name is not None:
            # Determiner le chemin de sauvegarde (meme dossier que ain_neuron.py)
            _base_dir = os.path.dirname(os.path.abspath(__file__))
            self._auto_save_path = os.path.join(_base_dir, f"{model_name}.pth")
            
            # Auto-chargement si un fichier existe deja
            if os.path.isfile(self._auto_save_path):
                try:
                    payload = torch.load(self._auto_save_path, map_location='cpu', weights_only=False)
                    self.load_state_dict(payload['state_dict'])
                    self.bank = ProgramBank.deserialize(payload['bank'])
                    self._best_loss = float(payload.get('best_loss', float('inf')))
                    _n = len(self.bank)
                    print(f"[AIN] Cerveau charge depuis '{model_name}.pth' "
                          f"(bank={_n} programmes, best_loss={self._best_loss:.6f})")
                except RuntimeError as e:
                    print(f"[AIN] WARN: Echec du chargement de '{model_name}.pth': {e}")
            else:
                print(f"[AIN] Nouveau cerveau '{model_name}' (aucun fichier precedent)")
            
            # Auto-sauvegarde a la fermeture du script
            atexit.register(self._atexit_save)
    
    def forward(self, support: torch.Tensor, query: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        support: (B, N, x_dim)  -- L'environnement a observer
        query:   (B, query_dim) -- La question a resoudre
        
        Retourne:
            pred: (B, out_dim) -- La prediction
            z:    (B, z_dim)   -- L'invariant decouvert
        
        System 2 : Trial-and-Error Algorithmique
        -----------------------------------------
        Au lieu de simplement raffiner le gating en boucle aveugle (coherence),
        le neurone TESTE chaque coalition d'experts en creant physiquement
        l'Effecteur et en le faisant predire sur le SUPPORT.
        
        Si l'erreur sur le support est elevee, le routeur recoit ce signal
        d'echec et explore une nouvelle coalition. Le meilleur Z (celui qui
        a le mieux predit le support) est utilise pour la prediction finale.
        """
        # =====================================================================
        # ÉTAPE 1 : L'Oeil prepare les voies (UNE SEULE FOIS)
        # =====================================================================
        ctx = self.eye.prepare_voies(support)
        
        # =====================================================================
        # ÉTAPE 2 : BOUCLE SYSTEM 2 — Trial-and-Error sur le Support
        # =====================================================================
        T = self.eye.T  # Nombre d'iterations de reflexion
        best_z = None
        best_error = float('inf')
        error_feedback = None  # Pas de feedback au premier tour
        
        B, N, D = support.shape

        if not self._use_multi_oracle:
            eval_q = support.mean(dim=1)[:, :query.shape[1]]
            eval_target_scalar = support.std(dim=1).mean(dim=-1, keepdim=True)
            eval_target = eval_target_scalar.repeat(1, self._out_dim)

            for t_iter in range(T):
                z = self.eye.step_routing(ctx, error_feedback=error_feedback)
                forged = self.forge(z)
                with torch.no_grad():
                    eval_pred = self.effector(eval_q, forged)
                    trial_error = (eval_pred - eval_target).pow(2).mean(dim=1, keepdim=True)

                current_mean_error = trial_error.mean().item()
                if current_mean_error < best_error:
                    best_error = current_mean_error
                    best_z = z

                error_feedback = trial_error
        else:
            x_exp = ctx['x_exp']
            x_summary = torch.cat([x_exp.mean(dim=1), x_exp.std(dim=1)], dim=-1)
            gate_out = self._oracle_gate(x_summary)
            mode_logits = gate_out[:, :3]
            alpha = torch.sigmoid(gate_out[:, 3:4])
            mode_probs = F.softmax(mode_logits, dim=-1)
            p_std = mode_probs[:, 0:1]
            p_mask = mode_probs[:, 1:2]
            p_mix = mode_probs[:, 2:3]

            eval_q = support.mean(dim=1)[:, :query.shape[1]]
            eval_target_std_scalar = support.std(dim=1).mean(dim=-1, keepdim=True)
            eval_target_std = eval_target_std_scalar.repeat(1, self._out_dim)

            support_visible = support[:, 1::2, :]
            support_masked = support[:, ::2, :]

            vis_exp = torch.cat([support_visible, support_visible ** 2], dim=-1)
            mask_exp = torch.cat([support_masked, support_masked ** 2], dim=-1)
            vis_summary = torch.cat([vis_exp.mean(dim=1), vis_exp.std(dim=1)], dim=-1)
            mask_summary = torch.cat([mask_exp.mean(dim=1), mask_exp.std(dim=1)], dim=-1)

            eval_q_mask = self._oracle_mask_q(vis_summary)
            eval_target_mask = self._oracle_mask_t(mask_summary)

            z_iters = []
            e_iters = []

            for t_iter in range(T):
                z = self.eye.step_routing(ctx, error_feedback=error_feedback)
                forged = self.forge(z)

                with torch.no_grad():
                    eval_pred = self.effector(eval_q, forged)
                    e_std = (eval_pred - eval_target_std).pow(2).mean(dim=1, keepdim=True)
                    eval_pred_mask = self.effector(eval_q_mask, forged)
                    e_mask = (eval_pred_mask - eval_target_mask).pow(2).mean(dim=1, keepdim=True)
                e_mix = alpha * e_std + (1.0 - alpha) * e_mask
                trial_error = p_std * e_std + p_mask * e_mask + p_mix * e_mix

                z_iters.append(z)
                e_iters.append(trial_error)

                error_feedback = trial_error.detach()

            e_stack = torch.stack(e_iters, dim=0)
            z_stack = torch.stack(z_iters, dim=0)
            tau = max(1e-6, float(self._oracle_tau))
            w = F.softmax(-e_stack / tau, dim=0)
            z_soft = (w * z_stack).sum(dim=0)
            best_z = z_soft
            best_error = float(e_stack.mean().item())
        
        # =====================================================================
        # ÉTAPE 3 : Prediction finale avec le MEILLEUR Z
        # =====================================================================
        z = best_z if best_z is not None else z
        forged = self.forge(z)
        pred = self.effector(query, forged)  # (B, out_dim)
        
        # 4. Archivage automatique dans la Memoire a Long Terme
        if self._auto_archive and self.training:
            with torch.no_grad():
                sig = compute_context_signature(support)
                q_store = query[0:1].unsqueeze(0) if query.dim() == 1 else query[0:1]
                self.bank.add(
                    signature=sig[0:1],
                    z=z[0:1],
                    forged={k: v[0:1] for k, v in forged.items()},
                    replay_support=support[0:1],
                    replay_queries=q_store,
                    replay_targets=pred[0:1].detach(),
                )
        
        return pred, z
    
    def save_full(self, path: str):
        """Sauvegarde le cerveau (poids) ET la memoire (ProgramBank) dans un seul fichier."""
        payload = {
            'state_dict': self.state_dict(),
            'bank': self.bank.serialize(),
            'best_loss': float(self._best_loss),
        }
        torch.save(payload, path)
    
    def save_if_best(self, current_loss: float, path: Optional[str] = None):
        """Sauvegarde uniquement si current_loss est la meilleure observee."""
        current_loss = float(current_loss)
        if current_loss < self._best_loss:
            self._best_loss = current_loss
            save_path = path or self._auto_save_path
            if save_path is not None:
                self.save_full(save_path)
                print(f"[AIN] Meilleur checkpoint sauvegarde (loss={current_loss:.6f}) -> {os.path.basename(save_path)}")
                return True
        return False
    
    def _atexit_save(self):
        """Appele automatiquement a la fermeture du script Python."""
        if self._auto_save_path is not None:
            try:
                self.save_full(self._auto_save_path)
                print(f"[AIN] Auto-sauvegarde -> {os.path.basename(self._auto_save_path)} "
                      f"(bank={len(self.bank)} programmes)")
            except RuntimeError as e:
                print(f"[AIN] WARN: Echec auto-sauvegarde: {e}")
    
    @classmethod
    def load_full(cls, path: str, *, x_dim: int, z_dim: int, query_dim: int,
                  out_dim: int, hidden: int = 64, bank_capacity: int = 1000,
                  device: str = 'cpu') -> 'AIN':
        """Charge le cerveau ET la memoire depuis un fichier unique."""
        payload = torch.load(path, map_location=device, weights_only=False)
        model = cls(x_dim=x_dim, z_dim=z_dim, query_dim=query_dim,
                    out_dim=out_dim, hidden=hidden, bank_capacity=bank_capacity)
        model.load_state_dict(payload['state_dict'])
        model.bank = ProgramBank.deserialize(payload['bank'], device=device)
        model._best_loss = float(payload.get('best_loss', float('inf')))
        return model.to(device)
