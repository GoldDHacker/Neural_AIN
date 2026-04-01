"""
==============================================================================
  DEMO — AIN en mode SEQUENCE (ordre causal)

  Objectif:
  - Tester AIN quand le support est une SEQUENCE: l'ordre des noeuds porte du sens.
  - 2 epreuves:
    1) NON-ALIGNE-SEQUENCE (discontinu + non-local + dependance a la position)
    2) COMPOSE-SEQUENCE (produit de 3 invariants dependants de l'ordre) + curriculum

  Metriques:
  - MSE
  - sign_acc = mean(sign(pred) == sign(target))
==============================================================================
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ain_neuron import (
    AIN,
    PolymathArgile,
    EmergentSpinGlass,
    PhysicalWeaver,
    VariationalPath,
    GeometerHand,
    HolonomyCartographer,
    PathGarden,
    EmergentCellularAutomaton,
    CombinatorialRouter,
    AdaptiveHyperForge,
    AdaptiveEffector,
)


class EmergentEncoderNoChronos(nn.Module):
    NUM_VOIES = 8
    def __init__(self, x_dim: int, hidden: int = 64, z_dim: int = 28):
        super().__init__()
        assert z_dim % self.NUM_VOIES == 0, "z_dim doit etre un multiple de 8 pour l'ablation"
        self.slot_dim = z_dim // self.NUM_VOIES
        self.expand_dim = x_dim * 2

        self.argile = PolymathArgile(self.expand_dim, hidden, self.slot_dim, num_knots=12)

        self.spin1 = EmergentSpinGlass(self.expand_dim, hidden)
        self.spin2 = EmergentSpinGlass(hidden, self.slot_dim)
        self.norm_p1 = nn.LayerNorm(hidden)

        self.attn1 = PhysicalWeaver(self.expand_dim, hidden, head_dim=16)
        self.attn2 = PhysicalWeaver(hidden, self.slot_dim, head_dim=16)
        self.norm_a1 = nn.LayerNorm(hidden)

        self.variational = VariationalPath(self.expand_dim, hidden, self.slot_dim)
        
        self.geometer = GeometerHand(self.expand_dim, hidden, self.slot_dim)
        self.weaver = HolonomyCartographer(self.expand_dim, self.slot_dim)
        self.garden = PathGarden(self.expand_dim, hidden, self.slot_dim)

        self.constructor = EmergentCellularAutomaton(self.expand_dim, hidden, self.slot_dim)

        self.router = CombinatorialRouter(in_features=self.expand_dim, num_voies=self.NUM_VOIES, z_dim=z_dim)
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)
        self.T = 3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        x_exp = torch.cat([x, x ** 2], dim=-1)

        z_argile_raw = self.argile(x_exp)
        z_spline = z_argile_raw.mean(dim=1)

        h_p = self.spin1(x_exp)
        h_p = self.norm_p1(h_p)
        h_p = self.spin2(h_p)
        z_spin = h_p.mean(dim=1)

        h_a = self.attn1(x_exp)
        h_a = self.norm_a1(h_a)
        h_a = self.attn2(h_a)
        z_attn = h_a.mean(dim=1)

        z_var = self.variational(x_exp)
        z_geom = self.geometer(x_exp)
        z_riemann = self.weaver(x_exp)
        z_garden = self.garden(x_exp)
        z_algo = self.constructor(x_exp)

        # L'Encyclopédie des 8 Experts (Chronos ablati)
        z_stack = torch.stack([z_spline, z_spin, z_attn, z_var, z_geom, z_riemann, z_garden, z_algo], dim=1)
        
        temp = torch.clamp(self.temperature, min=0.01)
        z_dim = self.NUM_VOIES * self.slot_dim
        
        z_flat = z_stack.view(B, z_dim)
        
        for t_iter in range(self.T):
            gate_logits, gate_blend = self.router(x_exp, z_current=z_flat)
            
            soft_gates = F.softmax(gate_logits / temp, dim=-1)
            
            hard_idx = torch.argmax(gate_logits, dim=-1)
            hard_gates_raw = F.one_hot(hard_idx, num_classes=self.router.num_combos).float().to(x.device)
            hard_gates = hard_gates_raw - soft_gates.detach() + soft_gates
            
            blend = torch.sigmoid(gate_blend)
            gates_combos = blend * soft_gates + (1.0 - blend) * hard_gates
            
            gates_8 = torch.matmul(gates_combos, self.router.combinations.to(x.device))
            gates_8 = torch.clamp(gates_8, min=0.01)
            
            z_sparse = gates_8.unsqueeze(-1) * z_stack
            z_flat = z_sparse.view(B, z_dim)
            
        self._debug_gates_voies = gates_8.detach()
        self._debug_blend = blend.detach()

        return z_flat


class AINNoChronos(nn.Module):
    def __init__(self, x_dim: int, z_dim: int, query_dim: int, out_dim: int, hidden: int = 64):
        super().__init__()
        # On force z_dim_ablation a etre multiple de 8
        z_dim_ablation = 32
        self.eye = EmergentEncoderNoChronos(x_dim=x_dim, hidden=hidden, z_dim=z_dim_ablation)
        self.forge = AdaptiveHyperForge(
            z_dim=z_dim_ablation, input_dim=query_dim, output_dim=out_dim, hidden_p=32, num_knots=12
        )
        self.effector = AdaptiveEffector()

    def forward(self, support: torch.Tensor, query: torch.Tensor):
        z = self.eye(support)
        forged = self.forge(z)
        pred = self.effector(query, forged)
        return pred, z


def _permute_nodes(support: torch.Tensor) -> torch.Tensor:
    """Permute les noeuds par batch (diagnostic: sensibilite a l'ordre)."""
    B, N, D = support.shape
    idx = torch.stack([torch.randperm(N, device=support.device) for _ in range(B)], dim=0)  # (B,N)
    idx_exp = idx.unsqueeze(-1).expand(-1, -1, D)
    return torch.gather(support, dim=1, index=idx_exp)


def _inv_pos_extremes_sequence(support: torch.Tensor) -> torch.Tensor:
    """Invariant discontinu + non-local + dependance a la position.

    - max sur la premiere moitie
    - min sur la seconde moitie
    Cela est intentionnellement NON permutation-invariant.
    """
    B, N, D = support.shape
    x = support[..., :4]
    v = torch.tensor([0.61, -0.37, 0.19, 0.08], device=support.device, dtype=support.dtype)
    proj = (x * v.view(1, 1, -1)).sum(dim=-1)  # (B,N)

    half = max(1, int(N) // 2)
    p_first = proj[:, :half]
    p_second = proj[:, half:]

    i_max = p_first.argmax(dim=1)  # (B,)
    i_min_local = p_second.argmin(dim=1)  # (B,)
    i_min = i_min_local + half

    x_max = x[torch.arange(B, device=support.device), i_max]
    x_min = x[torch.arange(B, device=support.device), i_min]

    dot = (x_max * x_min).sum(dim=-1, keepdim=True)
    inv = torch.where(dot >= 0.0, torch.ones_like(dot), -torch.ones_like(dot))
    return inv


def _inv_parity_firstk_sequence(support: torch.Tensor, *, k: int = 7) -> torch.Tensor:
    """Vote majoritaire discontinu, dependante de l'ordre: uniquement les k premiers noeuds.

    Remplace la parite (tres hostile a MSE) par une regle toujours discontinue,
    mais plus "apprenable" : signe de la somme des signes sur un prefix.
    """
    x = support[:, : min(int(k), int(support.shape[1])), 0]  # (B,k)
    votes = torch.sign(x)
    score = votes.sum(dim=1, keepdim=True)

    # Tie-break sans biais : si score == 0, utiliser la somme brute (continue)
    tie = (score == 0.0)
    raw_score = x.sum(dim=1, keepdim=True)
    score = torch.where(tie, raw_score, score)

    # Si encore exactement 0 (rare), fixer a +1 pour eviter un label neutre
    score = torch.where(score == 0.0, torch.ones_like(score), score)
    inv = torch.where(score >= 0.0, torch.ones_like(score), -torch.ones_like(score))
    return inv


def _inv_dynamics_sequence(theta: torch.Tensor) -> torch.Tensor:
    """Invariant dynamique : signe du paramètre de loi theta (B,1)."""
    theta = theta.view(-1, 1)
    theta = torch.where(theta == 0.0, torch.ones_like(theta), theta)
    inv = torch.where(theta >= 0.0, torch.ones_like(theta), -torch.ones_like(theta))
    return inv


def _inv_energy_prefix_vs_suffix_sequence(support: torch.Tensor) -> torch.Tensor:
    """Invariant: compare energie prefixe vs suffixe (ordre causal)."""
    B, N, D = support.shape
    half = max(1, int(N) // 2)
    e_first = (support[:, :half, :] ** 2).sum(dim=(1, 2), keepdim=False).view(B, 1)
    e_second = (support[:, half:, :] ** 2).sum(dim=(1, 2), keepdim=False).view(B, 1)
    inv = torch.where(e_first >= e_second, torch.ones_like(e_first), -torch.ones_like(e_first))
    return inv


def _inv_positional_vote_sequence(support: torch.Tensor) -> torch.Tensor:
    """Invariant strictement dependante de l'ordre via ponderation positionnelle.

    Multiset constant (marges identiques) mais label depend de l'assignation
    des signes aux positions.
    """
    B, N, _ = support.shape
    s = torch.sign(support[:, :, 0])  # (B,N)
    w = torch.linspace(-1.0, 1.0, int(N), device=support.device, dtype=support.dtype).view(1, -1)
    score = (s * w).sum(dim=1, keepdim=True)

    # Tie-break (rare) : utiliser la derniere position pour eviter un label neutre
    tie = (score == 0.0)
    last = s[:, -1:].clone()
    last = torch.where(last == 0.0, torch.ones_like(last), last)
    score = torch.where(tie, last, score)

    inv = torch.where(score >= 0.0, torch.ones_like(score), -torch.ones_like(score))
    return inv


def generate_unaligned_episode(B: int, N: int, x_dim: int, q_dim: int):
    support = torch.randn(B, N, x_dim)
    query = torch.randn(B, q_dim)

    inv = _inv_pos_extremes_sequence(support)

    w = torch.linspace(-1.0, 1.0, int(q_dim)).view(1, -1)
    qproj = (query * w).sum(dim=1, keepdim=True)
    target = inv * torch.tanh(torch.sin(qproj))

    return support, query, target


def generate_composed_episode(B: int, N: int, x_dim: int, q_dim: int, *, mode: str = "composed"):
    support = torch.randn(B, N, x_dim)
    query = torch.randn(B, q_dim)

    if mode == "inv1":
        inv = _inv_pos_extremes_sequence(support)
    elif mode == "inv2":
        inv = _inv_parity_firstk_sequence(support, k=7)
    elif mode == "inv3":
        inv = _inv_energy_prefix_vs_suffix_sequence(support)
    elif mode == "composed":
        inv = _inv_pos_extremes_sequence(support) * _inv_parity_firstk_sequence(support, k=7) * _inv_energy_prefix_vs_suffix_sequence(support)
    else:
        raise ValueError(f"mode inconnu: {mode}")

    inv = inv.view(B, 1)

    if mode != "composed":
        return support, query, inv

    w = torch.linspace(-1.0, 1.0, int(q_dim)).view(1, -1)
    qproj = (query * w).sum(dim=1, keepdim=True)
    disc = torch.where(
        query[:, :1] >= 0.0,
        torch.ones((B, 1), device=query.device, dtype=query.dtype),
        -torch.ones((B, 1), device=query.device, dtype=query.dtype),
    )
    target = inv * (torch.sin(0.8 * qproj) + 0.25 * disc)
    return support, query, target


def generate_orderonly_episode(B: int, N: int, x_dim: int, q_dim: int):
    """Episode SEQ identifiabilite pure : impossible sans ordre.

    - Le multiset de signes sur feature0 est fixe : N/2 +, N/2 - (par batch)
    - L'ordre est aleatoire et porte TOUTE l'information.
    - Invariant = vote positionnel (rampe w_t) sur les signes.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Construire une sequence de signes avec marges constantes
    half = max(1, int(N) // 2)
    signs = torch.cat([
        torch.ones((B, half), device=device),
        -torch.ones((B, int(N) - half), device=device),
    ], dim=1)

    # Permutation par batch : l'ordre est la variable latente
    idx = torch.stack([torch.randperm(int(N), device=device) for _ in range(int(B))], dim=0)
    signs = signs.gather(1, idx)

    support = torch.randn((B, N, x_dim), device=device) * 0.15
    support[:, :, 0] = 1.5 * signs  # eviter les zeros -> signe stable

    query = torch.randn((B, q_dim), device=device)
    inv = _inv_positional_vote_sequence(support)

    # Identifiabilite pure : target depend UNIQUEMENT de l'ordre (pas du query)
    target = inv
    return support, query, target


def generate_dynamics_episode(B: int, N: int, x_dim: int, q_dim: int):
    """Episode SEQ réaliste et apprenable : loi latente + trajectoire.

    On génère une séquence par un état latent s_t :
        s_{t+1} = tanh(theta * s_t + eps)
    où theta (par batch) est la "loi". Le target est sign(theta).

    L'ordre est nécessaire car les couples (s_t, s_{t+1}) et les deltas ds_t
    perdent leur cohérence si on permute la séquence.
    """
    # Loi par batch
    theta = torch.empty((B, 1)).uniform_(-1.0, 1.0)
    theta = theta + 0.15 * torch.sign(theta)  # eviter theta trop proche de 0

    # Etat initial
    s = torch.empty((B, 1)).uniform_(-1.0, 1.0)

    support = torch.zeros((B, N, x_dim), dtype=theta.dtype)
    noise_scale = 0.03

    for t in range(int(N)):
        eps = noise_scale * torch.randn((B, 1), dtype=theta.dtype)
        s_next = torch.tanh(theta * s + eps)

        # Encodage observationnel : NE PAS fournir les deltas / features d'ordre 1
        # sinon la tache devient resoluble sans ordre (chaque noeud contient deja (s, ds)).
        support[:, t, 0:1] = s
        if x_dim > 1:
            support[:, t, 1:] = 0.05 * torch.randn((B, x_dim - 1), dtype=theta.dtype)

        s = s_next

    query = torch.randn((B, q_dim), dtype=theta.dtype)
    inv = _inv_dynamics_sequence(theta)
    target = inv
    return support, query, target


def _sign_acc(pred: torch.Tensor, target: torch.Tensor) -> float:
    return (torch.sign(pred) == torch.sign(target)).float().mean().item()


def _routing_stats(model: AIN, support: torch.Tensor):
    """Extrait gates/blend du micro-gating."""
    if hasattr(model.eye, '_debug_gates_voies'):
        gates_mean = model.eye._debug_gates_voies.mean(dim=0)
        blend_mean = model.eye._debug_blend.mean()
        # Chronos est la 5eme voie (index 4) si num_voies >= 5
        chronos_mean = gates_mean[4] if len(gates_mean) > 4 else torch.zeros((), device=support.device)
    else:
        gates_mean = torch.zeros(9, device=support.device)
        blend_mean = torch.zeros((), device=support.device)
        chronos_mean = torch.zeros((), device=support.device)
    return gates_mean, blend_mean, chronos_mean


def _routing_gates(model, support: torch.Tensor):
    """Retourne gates par-sample (utile pour quantiles)."""
    if hasattr(model.eye, '_debug_gates_voies'):
        gates = model.eye._debug_gates_voies
        blend = model.eye._debug_blend
    else:
        B = support.shape[0]
        gates = torch.zeros(B, 9, device=support.device)
        blend = torch.zeros(B, 1, device=support.device)
    return gates, gates, blend


def _pathway_latents(model, support: torch.Tensor):
    """Recalcule z_i (par voie) depuis l'Eye, pour mesurer les contributions.

    Retourne un dict {name: z_i} où chaque z_i est (B, z_dim).
    """
    eye = model.eye

    x_exp = torch.cat([support, support ** 2], dim=-1)

    # Argile
    z_argile_raw = eye.argile(x_exp)
    z_spline = z_argile_raw.mean(dim=1)

    # Spin
    h_p = eye.spin1(x_exp)
    h_p = eye.norm_p1(h_p)
    h_p = eye.spin2(h_p)
    z_spin = h_p.mean(dim=1)

    # Attention
    h_a = eye.attn1(x_exp)
    h_a = eye.norm_a1(h_a)
    h_a = eye.attn2(h_a)
    z_attn = h_a.mean(dim=1)

    # Variationnel
    z_var = eye.variational(x_exp)

    out = {
        'spline': z_spline,
        'spin': z_spin,
        'attn': z_attn,
        'var': z_var,
    }

    # Chronos si present
    if hasattr(eye, 'chronos'):
        out['time'] = eye.chronos(x_exp)
        
    if hasattr(eye, 'geometer'):
        out['geom'] = eye.geometer(x_exp)
    if hasattr(eye, 'weaver'):
        out['riemann'] = eye.weaver(x_exp)
    if hasattr(eye, 'garden'):
        out['garden'] = eye.garden(x_exp)

    return out


def _format_quantiles(x_1d: torch.Tensor) -> str:
    x = x_1d.detach().float().view(-1).cpu()
    qs = [0.0, 0.1, 0.5, 0.9, 1.0]
    vals = torch.quantile(x, torch.tensor(qs))
    return f"min={vals[0]:.3f} p10={vals[1]:.3f} p50={vals[2]:.3f} p90={vals[3]:.3f} max={vals[4]:.3f}"


def _format_contribs(gates: torch.Tensor, latents: dict) -> str:
    # gates: (B, V), latents: dict -> (B,z)
    names = ['spline', 'spin', 'attn', 'var']
    if 'time' in latents: names.append('time')
    if 'geom' in latents: names.append('geom')
    if 'riemann' in latents: names.append('riemann')
    if 'garden' in latents: names.append('garden')

    parts = []
    for i, name in enumerate(names):
        if i < gates.shape[1]:
            g = gates[:, i:i+1]
            z = latents[name]
            c = (g * z).norm(dim=1).mean().item()
            parts.append(f"{name}={c:.3f}")
    return " ".join(parts)


def train(
    model: AIN,
    gen_fn,
    *,
    epochs: int,
    lr: float,
    device: str,
    label: str,
    log_every: int = 50,
    entropy_coef: float = 0.02,
    entropy_anneal_frac: float = 0.6,
):
    opt = optim.Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()
    model.to(device)
    model.train()

    best = float('inf')

    for ep in range(1, epochs + 1):
        support, query, target = gen_fn()
        support, query, target = support.to(device), query.to(device), target.to(device)

        opt.zero_grad(set_to_none=True)
        pred, z = model(support, query)
        loss = crit(pred, target)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()

        best = min(best, float(loss.item()))

        if ep == 1 or ep % log_every == 0 or ep == epochs:
            acc = _sign_acc(pred.detach(), target)
            z_norm = z.detach().norm(dim=1).mean().item()
            with torch.no_grad():
                gates_mean, blend_mean, chronos_mean = _routing_stats(model, support)
            gm = ",".join([f"{v:.3f}" for v in gates_mean.detach().cpu().tolist()])
            print(
                f"[{label} EP {ep:03d}] loss={loss.item():.6f} best={best:.6f} sign_acc={acc:.4f} ||Z||={z_norm:.4f} "
                f"gates_mean=[{gm}] blend_mean={float(blend_mean):.3f} chronos_mean={float(chronos_mean):.3f}"
            )


def eval_model(model: AIN, gen_fn, *, device: str, label: str):
    crit = nn.MSELoss()
    model.eval()
    with torch.no_grad():
        support, query, target = gen_fn()
        support, query, target = support.to(device), query.to(device), target.to(device)
        pred, z = model(support, query)
        mse = crit(pred, target).item()
        acc = _sign_acc(pred, target)
        gates_mean, blend_mean, chronos_mean = _routing_stats(model, support)
        gm = ",".join([f"{v:.3f}" for v in gates_mean.detach().cpu().tolist()])
        print(
            f"\n[{label} FINAL] mse={mse:.6f} sign_acc={acc:.4f} "
            f"gates_mean=[{gm}] blend_mean={float(blend_mean):.3f} chronos_mean={float(chronos_mean):.3f}"
        )

        if ("ORDER-ONLY" in str(label)) or ("DYNAMICS" in str(label)):
            gates_full, soft_full, blend_full = _routing_gates(model, support)
            lat = _pathway_latents(model, support)
            if gates_full.shape[1] >= 5:
                print(f"[{label} GATE_TIME_Q] {_format_quantiles(gates_full[:, 4])}")
            print(f"[{label} CONTRIB_NORM] {_format_contribs(gates_full, lat)}")

        # Diagnostic : sensibilite a la permutation de l'ordre
        support_p = _permute_nodes(support)
        pred_p, _ = model(support_p, query)
        mse_p = crit(pred_p, target).item()
        acc_p = _sign_acc(pred_p, target)
        gates_mean_p, blend_mean_p, chronos_mean_p = _routing_stats(model, support_p)
        gm_p = ",".join([f"{v:.3f}" for v in gates_mean_p.detach().cpu().tolist()])
        print(
            f"[{label} FINAL-PERM] mse={mse_p:.6f} sign_acc={acc_p:.4f} "
            f"gates_mean=[{gm_p}] blend_mean={float(blend_mean_p):.3f} chronos_mean={float(chronos_mean_p):.3f}"
        )

        if ("ORDER-ONLY" in str(label)) or ("DYNAMICS" in str(label)):
            gates_full_p, soft_full_p, blend_full_p = _routing_gates(model, support_p)
            lat_p = _pathway_latents(model, support_p)
            if gates_full_p.shape[1] >= 5:
                print(f"[{label} FINAL-PERM GATE_TIME_Q] {_format_quantiles(gates_full_p[:, 4])}")
            print(f"[{label} FINAL-PERM CONTRIB_NORM] {_format_contribs(gates_full_p, lat_p)}")
    return mse, acc


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("=" * 70)
    print("  DEMO AIN — MODE SEQUENCE (ordre causal)")
    print("=" * 70)
    print(f"[INFO] device={device}")

    torch.manual_seed(0)

    B = 64
    N = 28
    x_dim = 8
    q_dim = 6
    z_dim = 36
    hidden = 64
    lr = 2e-3

    results = {}

    print("\n" + "=" * 70)
    print("  EPREUVE 0: ORDER-ONLY (identifiabilite pure, dependance stricte a l'ordre)")
    print("=" * 70)
    model_o = AIN(x_dim=x_dim, z_dim=z_dim, query_dim=q_dim, out_dim=1, hidden=hidden)
    train(
        model_o,
        lambda: generate_orderonly_episode(B, N, x_dim, q_dim),
        epochs=250,
        lr=lr,
        device=device,
        label="ORDER-ONLY-SEQ",
        log_every=50,
    )
    results["ORDER-ONLY-SEQ"] = eval_model(
        model_o,
        lambda: generate_orderonly_episode(256, N, x_dim, q_dim),
        device=device,
        label="ORDER-ONLY-SEQ",
    )

    print("\n" + "=" * 70)
    print("  EPREUVE 0b: ORDER-ONLY — ABLATION SANS CHRONOS")
    print("=" * 70)
    model_o_nc = AINNoChronos(x_dim=x_dim, z_dim=z_dim, query_dim=q_dim, out_dim=1, hidden=hidden)
    train(
        model_o_nc,
        lambda: generate_orderonly_episode(B, N, x_dim, q_dim),
        epochs=250,
        lr=lr,
        device=device,
        label="ORDER-ONLY-NOCHRONOS",
        log_every=50,
    )
    results["ORDER-ONLY-NC"] = eval_model(
        model_o_nc,
        lambda: generate_orderonly_episode(256, N, x_dim, q_dim),
        device=device,
        label="ORDER-ONLY-NOCHRONOS",
    )

    print("\n" + "=" * 70)
    print("  EPREUVE 0c: DYNAMICS (dynamique récurrente, ordre strict)")
    print("=" * 70)
    model_d = AIN(x_dim=x_dim, z_dim=z_dim, query_dim=q_dim, out_dim=1, hidden=hidden)
    train(
        model_d,
        lambda: generate_dynamics_episode(B, N, x_dim, q_dim),
        epochs=250,
        lr=lr,
        device=device,
        label="DYNAMICS-SEQ",
        log_every=50,
    )
    results["DYNAMICS-SEQ"] = eval_model(
        model_d,
        lambda: generate_dynamics_episode(256, N, x_dim, q_dim),
        device=device,
        label="DYNAMICS-SEQ",
    )

    print("\n" + "=" * 70)
    print("  EPREUVE 0d: DYNAMICS — ABLATION SANS CHRONOS")
    print("=" * 70)
    model_d_nc = AINNoChronos(x_dim=x_dim, z_dim=z_dim, query_dim=q_dim, out_dim=1, hidden=hidden)
    train(
        model_d_nc,
        lambda: generate_dynamics_episode(B, N, x_dim, q_dim),
        epochs=250,
        lr=lr,
        device=device,
        label="DYNAMICS-NOCHRONOS",
        log_every=50,
    )
    results["DYNAMICS-NC"] = eval_model(
        model_d_nc,
        lambda: generate_dynamics_episode(256, N, x_dim, q_dim),
        device=device,
        label="DYNAMICS-NOCHRONOS",
    )

    print("\n" + "=" * 70)
    print("  EPREUVE 1: NON-ALIGNE-SEQUENCE")
    print("=" * 70)
    model_u = AIN(x_dim=x_dim, z_dim=z_dim, query_dim=q_dim, out_dim=1, hidden=hidden)
    train(
        model_u,
        lambda: generate_unaligned_episode(B, N, x_dim, q_dim),
        epochs=250,
        lr=lr,
        device=device,
        label="UNALIGNED-SEQ",
        log_every=50,
    )
    results["UNALIGNED-SEQ"] = eval_model(
        model_u,
        lambda: generate_unaligned_episode(256, N, x_dim, q_dim),
        device=device,
        label="UNALIGNED-SEQ",
    )

    print("\n" + "=" * 70)
    print("  EPREUVE 2: COMPOSE-SEQUENCE (curriculum inv1->inv2->inv3 puis composed)")
    print("=" * 70)
    model_c = AIN(x_dim=x_dim, z_dim=z_dim, query_dim=q_dim, out_dim=1, hidden=hidden)

    for mode in ["inv1", "inv2", "inv3"]:
        train(
            model_c,
            lambda m=mode: generate_composed_episode(B, N, x_dim, q_dim, mode=m),
            epochs=250,
            lr=lr,
            device=device,
            label=f"PRE-{mode}-SEQ",
            log_every=50,
        )

    train(
        model_c,
        lambda: generate_composed_episode(B, N, x_dim, q_dim, mode="composed"),
        epochs=250,
        lr=lr,
        device=device,
        label="POST-COMPOSED-SEQ",
        log_every=50,
    )
    results["COMPOSED-SEQ"] = eval_model(
        model_c,
        lambda: generate_composed_episode(256, N, x_dim, q_dim, mode="composed"),
        device=device,
        label="COMPOSED-SEQ",
    )

    print("\n" + "=" * 70)
    print("  TABLEAU RECAPITULATIF — AIN SEQUENCE")
    print("=" * 70)
    print(f"  {'Test':<20s} {'MSE':>10s} {'sign_acc':>10s}")
    print(f"  {'-'*20} {'-'*10} {'-'*10}")
    for name, (mse, acc) in results.items():
        print(f"  {name:<20s} {mse:>10.6f} {acc:>10.4f}")


if __name__ == '__main__':
    main()
