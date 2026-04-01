"""
==============================================================================
  DEMO — AIN COMPILATION TEST (z -> forge -> muscle reutilisable)

  Objectif:
  Verifier que AIN ne fait pas seulement "predire", mais :
  1) infere une loi compacte z depuis un support
  2) compile z via la Forge (parametres d'un effecteur)
  3) execute cette loi sur de multiples queries sans re-encoder le support

  Tests:
  - IN-EPISODE REUSE : forger une fois, evaluer K queries (meme support)
  - CROSS-SUPPORT SAME LAW : deux supports differents mais meme loi theta
    -> les deux compilations doivent reussir pareil
  - SWAP-LAW CONTROL : utiliser la compilation d'un episode sur les queries d'un autre
    -> doit s'effondrer
==============================================================================
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim

from ain_neuron import AIN


def _law(theta0: torch.Tensor, theta1: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
    """Loi par episode.

    theta0, theta1: (B,1)
    query: (B,K,q_dim)
    returns: (B,K,1)
    """
    # Projection fixe (prior minimal: un repere de mesure constant)
    q0 = query[..., 0:1]
    q1 = query[..., 1:2] if query.shape[-1] >= 2 else query[..., 0:1]

    # Une loi mixte (affine + trig) assez expressive pour motiver la Forge
    out = theta0.unsqueeze(1) * (0.6 * q0 + 0.4 * q1) + torch.sin(q0 + theta1.unsqueeze(1))
    return out


def generate_compilation_episode(B: int, N: int, x_dim: int, q_dim: int, K: int, *, device: str):
    """Episode : un meme support encode une loi (theta0, theta1) et on echantillonne K queries."""

    theta0 = torch.empty((B, 1), device=device).uniform_(-1.5, 1.5)
    theta1 = torch.empty((B, 1), device=device).uniform_(-3.14, 3.14)

    # Support = observations noisy de la loi (pas le code direct)
    support = torch.randn((B, N, x_dim), device=device) * 0.20
    support[:, :, 0:1] = theta0.unsqueeze(1) + 0.15 * torch.randn((B, N, 1), device=device)
    if x_dim >= 2:
        support[:, :, 1:2] = theta1.unsqueeze(1) + 0.15 * torch.randn((B, N, 1), device=device)

    queries = torch.randn((B, K, q_dim), device=device)
    targets = _law(theta0, theta1, queries)

    return support, queries, targets, theta0, theta1


def generate_cross_support_same_law(B: int, N: int, x_dim: int, q_dim: int, K: int, *, device: str):
    """Deux supports differents mais meme (theta0, theta1)."""

    theta0 = torch.empty((B, 1), device=device).uniform_(-1.5, 1.5)
    theta1 = torch.empty((B, 1), device=device).uniform_(-3.14, 3.14)

    def make_support():
        s = torch.randn((B, N, x_dim), device=device) * 0.20
        s[:, :, 0:1] = theta0.unsqueeze(1) + 0.15 * torch.randn((B, N, 1), device=device)
        if x_dim >= 2:
            s[:, :, 1:2] = theta1.unsqueeze(1) + 0.15 * torch.randn((B, N, 1), device=device)
        return s

    support_a = make_support()
    support_b = make_support()

    queries = torch.randn((B, K, q_dim), device=device)
    targets = _law(theta0, theta1, queries)

    return support_a, support_b, queries, targets


def compiled_execute(effector: nn.Module, forged: dict, queries: torch.Tensor) -> torch.Tensor:
    """Execute le muscle forge sur K queries sans re-forger.

    forged: dict de tenseurs (B, ...)
    queries: (B,K,q_dim)
    returns: (B,K,out)
    """
    B, K, q_dim = queries.shape

    q_flat = queries.reshape(B * K, q_dim)

    forged_flat = {}
    for k, v in forged.items():
        # v: (B, ...) -> (B*K, ...)
        forged_flat[k] = v.repeat_interleave(K, dim=0)

    out_flat = effector(q_flat, forged_flat)
    out = out_flat.reshape(B, K, -1)
    return out


def train_compilation(model: AIN, *, device: str, epochs: int = 400, lr: float = 1e-3,
                      B: int = 32, N: int = 32, x_dim: int = 4, q_dim: int = 4, K: int = 16):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.to(device)
    model.train()

    best = float('inf')

    for ep in range(1, epochs + 1):
        support, queries, targets, _, _ = generate_compilation_episode(B, N, x_dim, q_dim, K, device=device)

        optimizer.zero_grad()

        # 1) infere z (code source latent)
        z = model.eye(support)

        # 2) compile en muscle
        forged = model.forge(z)

        # 3) execute sur K queries
        pred = compiled_execute(model.effector, forged, queries)

        loss = criterion(pred, targets)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        best = min(best, float(loss.item()))

        if ep == 1 or ep % 100 == 0 or ep == epochs:
            z_norm = float(z.detach().norm(dim=1).mean().item())
            print(f"[TRAIN EP {ep:03d}] loss={loss.item():.6f} best={best:.6f} ||z||={z_norm:.3f}")

    return best


@torch.no_grad()
def eval_compilation(model: AIN, *, device: str, B: int = 128, N: int = 32, x_dim: int = 4,
                     q_dim: int = 4, K: int = 32):
    criterion = nn.MSELoss()
    model.to(device)
    model.eval()

    # -----------------
    # 1) In-episode reuse
    # -----------------
    support, queries, targets, _, _ = generate_compilation_episode(B, N, x_dim, q_dim, K, device=device)
    z = model.eye(support)
    forged = model.forge(z)
    pred = compiled_execute(model.effector, forged, queries)
    mse_in = float(criterion(pred, targets).item())

    # -----------------
    # 2) Cross-support same law
    # -----------------
    support_a, support_b, queries_cs, targets_cs = generate_cross_support_same_law(
        B, N, x_dim, q_dim, K, device=device
    )

    z_a = model.eye(support_a)
    z_b = model.eye(support_b)

    forged_a = model.forge(z_a)
    forged_b = model.forge(z_b)

    pred_a = compiled_execute(model.effector, forged_a, queries_cs)
    pred_b = compiled_execute(model.effector, forged_b, queries_cs)

    mse_cs_a = float(criterion(pred_a, targets_cs).item())
    mse_cs_b = float(criterion(pred_b, targets_cs).item())

    # proximite des invariants (diagnostic, pas un objectif)
    z_cos = torch.nn.functional.cosine_similarity(z_a, z_b, dim=-1).mean().item()

    # -----------------
    # 3) Swap-law control
    # -----------------
    # Batch split en deux groupes; on swap les compilations
    B2 = max(2, B // 2)
    support_1, queries_1, targets_1, _, _ = generate_compilation_episode(B2, N, x_dim, q_dim, K, device=device)
    support_2, queries_2, targets_2, _, _ = generate_compilation_episode(B2, N, x_dim, q_dim, K, device=device)

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
        "mse_cross_support_a": mse_cs_a,
        "mse_cross_support_b": mse_cs_b,
        "z_cos_cross_support": float(z_cos),
        "mse_swap_12": mse_swap_12,
        "mse_swap_21": mse_swap_21,
    }


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 70)
    print("  DEMO — AIN COMPILATION TEST")
    print("  support -> z -> forge -> muscle reutilisable")
    print("=" * 70)

    # Hyperparams
    x_dim, q_dim, z_dim = 4, 4, 16
    hidden = 64

    model = AIN(x_dim=x_dim, z_dim=z_dim, query_dim=q_dim, out_dim=1, hidden=hidden)

    print("\n" + "=" * 70)
    print("  ENTRAINEMENT (apprendre a compiler)")
    print("=" * 70)
    train_compilation(model, device=device, epochs=400, lr=1e-3, B=32, N=32, x_dim=x_dim, q_dim=q_dim, K=16)

    print("\n" + "=" * 70)
    print("  EVALUATION")
    print("=" * 70)
    metrics = eval_compilation(model, device=device, B=128, N=32, x_dim=x_dim, q_dim=q_dim, K=32)

    for k, v in metrics.items():
        print(f"{k:<24s} : {v:.6f}" if isinstance(v, float) else f"{k:<24s} : {v}")

    print("\n[INTERPRETATION]")
    print("- mse_in_episode doit etre bas: le muscle forge est reutilisable sur K queries")
    print("- mse_cross_support_* doit rester bas: z capture la loi, pas le support")
    print("- mse_swap_* doit etre haut: une compilation ne doit pas marcher sur une autre loi")


if __name__ == '__main__':
    main()
