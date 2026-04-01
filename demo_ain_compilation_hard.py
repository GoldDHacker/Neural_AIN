"""
==============================================================================
  DEMO — AIN COMPILATION TEST (HARD)

  Test plus dur que demo_ain_compilation.py :
  - Le support ne contient PAS les parametres de loi (theta).
  - Le support ne contient que des EXEMPLES (x_i, y_i) issus d'une fonction latente
    aleatoire par episode.

  Objectif strict :
  support (exemples) -> z -> forge -> muscle reutilisable

  Tests:
  - IN-EPISODE REUSE : forger une fois, evaluer K queries (meme loi)
  - CROSS-SUPPORT SAME LAW : deux supports differents mais meme loi (meme f)
  - SWAP-LAW CONTROL : utiliser la compilation d'une loi sur les queries d'une autre
==============================================================================
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ain_neuron import AIN


def _sample_law(B: int, q_dim: int, hidden: int, *, device: str):
    """Echantillonne une fonction latente par episode.

    On definie f(x) = sum_k w1_k * sin( (W0 x + b0)_k ) + b1

    W0: (B, hidden, q_dim)
    b0: (B, hidden)
    w1: (B, hidden)
    b1: (B, 1)
    """
    # Bornes d'amplitude pour eviter des targets enormes (instabilite au debut)
    # MAIS suffisamment de diversite pour que SWAP-LAW soit réellement penalise.
    W0 = torch.randn((B, hidden, q_dim), device=device) * 0.45
    b0 = torch.randn((B, hidden), device=device) * 0.30
    w1 = torch.randn((B, hidden), device=device) * 0.35

    # Scale/offset par episode pour eviter des sorties trop centrees ~0
    scale = torch.empty((B, 1), device=device).uniform_(0.6, 1.8)
    offset = torch.empty((B, 1), device=device).uniform_(-0.6, 0.6)
    return W0, b0, w1, scale, offset


def _law_eval(W0, b0, w1, scale, offset, x: torch.Tensor) -> torch.Tensor:
    """x: (B, K, q_dim) -> y: (B, K, 1)"""
    # W0: (B, hidden, q_dim)
    # x:  (B, K, q_dim)
    # -> (B, K, hidden)
    proj = torch.einsum('bkq,bhq->bkh', x, W0) + b0.unsqueeze(1)
    h = torch.sin(proj)
    raw = (h * w1.unsqueeze(1)).sum(dim=-1, keepdim=True)
    # Borner la non-linearite, puis appliquer une variation d'amplitude + offset
    y = scale.unsqueeze(1) * torch.tanh(raw) + offset.unsqueeze(1)
    return y


def _make_support_from_law(
    W0,
    b0,
    w1,
    scale,
    offset,
    *,
    B: int,
    N: int,
    q_dim: int,
    x_dim: int,
    device: str,
):
    """Support: (B,N,x_dim) contenant (x_i, y_i, bruit...)."""
    x = torch.randn((B, N, q_dim), device=device) * 0.8
    y = _law_eval(W0, b0, w1, scale, offset, x)
    y = y + 0.01 * torch.randn_like(y)

    support = torch.zeros((B, N, x_dim), device=device)
    support[:, :, :q_dim] = x
    support[:, :, q_dim:q_dim + 1] = y
    if x_dim > q_dim + 1:
        support[:, :, q_dim + 1:] = 0.05 * torch.randn((B, N, x_dim - (q_dim + 1)), device=device)

    return support


def _make_queries_targets(
    W0,
    b0,
    w1,
    scale,
    offset,
    *,
    B: int,
    K: int,
    q_dim: int,
    device: str,
):
    queries = torch.randn((B, K, q_dim), device=device) * 0.8
    targets = _law_eval(W0, b0, w1, scale, offset, queries)
    return queries, targets


def compiled_execute(effector: nn.Module, forged: dict, queries: torch.Tensor) -> torch.Tensor:
    """Execute le muscle forge sur K queries sans re-forger.

    queries: (B,K,q_dim)
    returns: (B,K,1)
    """
    B, K, q_dim = queries.shape
    q_flat = queries.reshape(B * K, q_dim)

    forged_flat = {}
    for k, v in forged.items():
        forged_flat[k] = v.repeat_interleave(K, dim=0)

    out_flat = effector(q_flat, forged_flat)
    out = out_flat.reshape(B, K, -1)
    return out


def train_compilation_hard(model: AIN, *, device: str, epochs: int = 600, lr: float = 1e-3,
                           B: int = 32, N: int = 64, x_dim: int = 6, q_dim: int = 4, K: int = 16,
                           law_hidden: int = 10):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.to(device)
    model.train()

    best = float('inf')

    for ep in range(1, epochs + 1):
        W0, b0, w1, scale, offset = _sample_law(B, q_dim, law_hidden, device=device)
        support = _make_support_from_law(
            W0, b0, w1, scale, offset,
            B=B, N=N, q_dim=q_dim, x_dim=x_dim, device=device
        )
        queries, targets = _make_queries_targets(
            W0, b0, w1, scale, offset,
            B=B, K=K, q_dim=q_dim, device=device
        )

        optimizer.zero_grad()

        z = model.eye(support)
        forged = model.forge(z)
        pred = compiled_execute(model.effector, forged, queries)

        loss = criterion(pred, targets)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        best = min(best, float(loss.item()))

        if ep == 1 or ep % 150 == 0 or ep == epochs:
            z_norm = float(z.detach().norm(dim=1).mean().item())
            print(f"[TRAIN EP {ep:03d}] loss={loss.item():.6f} best={best:.6f} ||z||={z_norm:.3f}")

    return best


@torch.no_grad()
def eval_compilation_hard(model: AIN, *, device: str, B: int = 128, N: int = 64,
                          x_dim: int = 6, q_dim: int = 4, K: int = 32, law_hidden: int = 10):
    criterion = nn.MSELoss()
    model.to(device)
    model.eval()

    # 1) In-episode reuse
    W0, b0, w1, scale, offset = _sample_law(B, q_dim, law_hidden, device=device)
    support = _make_support_from_law(
        W0, b0, w1, scale, offset,
        B=B, N=N, q_dim=q_dim, x_dim=x_dim, device=device
    )
    queries, targets = _make_queries_targets(
        W0, b0, w1, scale, offset,
        B=B, K=K, q_dim=q_dim, device=device
    )

    z = model.eye(support)
    forged = model.forge(z)
    pred = compiled_execute(model.effector, forged, queries)
    mse_in = float(criterion(pred, targets).item())

    mse_zero = float(criterion(torch.zeros_like(targets), targets).item())

    # 2) Cross-support same law
    support_a = _make_support_from_law(
        W0, b0, w1, scale, offset,
        B=B, N=N, q_dim=q_dim, x_dim=x_dim, device=device
    )
    support_b = _make_support_from_law(
        W0, b0, w1, scale, offset,
        B=B, N=N, q_dim=q_dim, x_dim=x_dim, device=device
    )

    z_a = model.eye(support_a)
    z_b = model.eye(support_b)

    forged_a = model.forge(z_a)
    forged_b = model.forge(z_b)

    pred_a = compiled_execute(model.effector, forged_a, queries)
    pred_b = compiled_execute(model.effector, forged_b, queries)

    mse_cs_a = float(criterion(pred_a, targets).item())
    mse_cs_b = float(criterion(pred_b, targets).item())

    z_cos = F.cosine_similarity(z_a, z_b, dim=-1).mean().item()

    # 3) Swap-law control
    B2 = max(2, B // 2)
    W01, b01, w11, scale1, offset1 = _sample_law(B2, q_dim, law_hidden, device=device)
    W02, b02, w12, scale2, offset2 = _sample_law(B2, q_dim, law_hidden, device=device)

    support_1 = _make_support_from_law(
        W01, b01, w11, scale1, offset1,
        B=B2, N=N, q_dim=q_dim, x_dim=x_dim, device=device
    )
    support_2 = _make_support_from_law(
        W02, b02, w12, scale2, offset2,
        B=B2, N=N, q_dim=q_dim, x_dim=x_dim, device=device
    )

    queries_1, targets_1 = _make_queries_targets(
        W01, b01, w11, scale1, offset1,
        B=B2, K=K, q_dim=q_dim, device=device
    )
    queries_2, targets_2 = _make_queries_targets(
        W02, b02, w12, scale2, offset2,
        B=B2, K=K, q_dim=q_dim, device=device
    )

    z1 = model.eye(support_1)
    z2 = model.eye(support_2)

    forged_1 = model.forge(z1)
    forged_2 = model.forge(z2)

    pred_swap_12 = compiled_execute(model.effector, forged_1, queries_2)
    pred_swap_21 = compiled_execute(model.effector, forged_2, queries_1)

    mse_swap_12 = float(criterion(pred_swap_12, targets_2).item())
    mse_swap_21 = float(criterion(pred_swap_21, targets_1).item())

    return {
        "mse_in_episode": mse_in,
        "mse_zero": mse_zero,
        "mse_cross_support_a": mse_cs_a,
        "mse_cross_support_b": mse_cs_b,
        "z_cos_cross_support": float(z_cos),
        "mse_swap_12": mse_swap_12,
        "mse_swap_21": mse_swap_21,
    }


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 70)
    print("  DEMO — AIN COMPILATION TEST (HARD)")
    print("  support = exemples (x_i, y_i) / loi latente inconnue")
    print("=" * 70)

    x_dim, q_dim, z_dim = 6, 4, 32
    hidden = 64

    model = AIN(x_dim=x_dim, z_dim=z_dim, query_dim=q_dim, out_dim=1, hidden=hidden)

    print("\n" + "=" * 70)
    print("  ENTRAINEMENT")
    print("=" * 70)
    train_compilation_hard(model, device=device, epochs=800, lr=3e-4, B=32, N=64, x_dim=x_dim, q_dim=q_dim, K=16)

    print("\n" + "=" * 70)
    print("  EVALUATION")
    print("=" * 70)
    metrics = eval_compilation_hard(model, device=device, B=128, N=64, x_dim=x_dim, q_dim=q_dim, K=32)

    for k, v in metrics.items():
        print(f"{k:<24s} : {v:.6f}" if isinstance(v, float) else f"{k:<24s} : {v}")

    print("\n[INTERPRETATION]")
    print("- mse_in_episode bas => compilation reutilisable sur K queries")
    print("- mse_cross_support_* bas => z capture la loi via exemples, pas le support lui-meme")
    print("- mse_swap_* haut => compilation specifique a la loi")


if __name__ == '__main__':
    main()
