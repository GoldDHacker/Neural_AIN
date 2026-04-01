"""\
==============================================================================
  DEMO — Program Bank (AIN) : Inference reuse vs recompile

  Objectif:
  - Montrer l'usage "inference" de la ProgramBank:
    1) Soit on recompile a chaque episode (support -> z -> forge)
    2) Soit on REUTILISE un programme (forged) si z matche un programme connu

  Metriques:
  - hit_rate : proportion d'episodes ou un programme est reutilise
  - mse_reuse : MSE obtenu en mode bank (reuse / refresh)
  - mse_always_compile : MSE si on compile systematiquement
  - refresh_rate : proportion de hits qui declenchent une recompilation (mismatch)

  Notes:
  - Ne modifie pas ain_neuron.py.
  - La bank stocke z + forged + signature.
==============================================================================
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from ain_neuron import AIN
from program_bank import BankPolicy, BankPolicyConfig, BankPolicyV2, BankPolicyV2Config, ContextualAIN, ProgramBank


@dataclass
class Law:
    W0: torch.Tensor
    b0: torch.Tensor
    w1: torch.Tensor
    scale: torch.Tensor
    offset: torch.Tensor


def _make_random_law(q_dim: int, hidden: int, *, device: str) -> Law:
    W0 = torch.randn((hidden, q_dim), device=device) * 0.45
    b0 = torch.randn((hidden,), device=device) * 0.30
    w1 = torch.randn((hidden,), device=device) * 0.35
    scale = torch.empty((1,), device=device).uniform_(0.6, 1.8)
    offset = torch.empty((1,), device=device).uniform_(-0.6, 0.6)
    return Law(W0=W0, b0=b0, w1=w1, scale=scale, offset=offset)


def _law_eval(law: Law, x: torch.Tensor) -> torch.Tensor:
    """x: (..., q_dim) -> (..., 1)"""
    proj = torch.einsum('...q,hq->...h', x, law.W0) + law.b0
    h = torch.sin(proj)
    raw = (h * law.w1).sum(dim=-1, keepdim=True)
    y = law.scale * torch.tanh(raw) + law.offset
    return y


def generate_episode_from_law(
    law: Law,
    *,
    B: int,
    N: int,
    x_dim: int,
    q_dim: int,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """support=(x_i,y_i), query=x, target=y"""
    x = torch.randn((B, N, q_dim), device=device) * 0.8
    y = _law_eval(law, x)
    y = y + 0.01 * torch.randn_like(y)

    support = torch.zeros((B, N, x_dim), device=device)
    support[:, :, :q_dim] = x
    support[:, :, q_dim:q_dim + 1] = y
    if x_dim > q_dim + 1:
        support[:, :, q_dim + 1:] = 0.05 * torch.randn((B, N, x_dim - (q_dim + 1)), device=device)

    query = torch.randn((B, q_dim), device=device) * 0.8
    target = _law_eval(law, query)
    return support, query, target


def effector_execute(effector: nn.Module, forged: dict, query: torch.Tensor) -> torch.Tensor:
    """Query: (B,q_dim), forged: dict (B,...) -> pred: (B,1)"""
    return effector(query, forged)


def _expand_forged_to_batch(forged: dict, B: int) -> dict:
    out = {}
    for k, v in forged.items():
        if v.shape[0] == B:
            out[k] = v
        elif v.shape[0] == 1:
            out[k] = v.expand(B, *v.shape[1:])
        else:
            raise RuntimeError(f"forged['{k}'] batch mismatch: {v.shape[0]} vs expected {B}")
    return out


def pretrain(model: AIN, *, device: str, epochs: int = 600, lr: float = 8e-4,
             B: int = 32, N: int = 64, x_dim: int = 6, q_dim: int = 4,
             law_pool: int = 64, law_hidden: int = 10):
    opt = optim.Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()

    # Pool fixe de lois pour stabiliser l'apprentissage
    pool: List[Law] = [_make_random_law(q_dim, law_hidden, device=device) for _ in range(law_pool)]

    model.to(device)
    model.train()

    best = float('inf')

    for ep in range(1, epochs + 1):
        law = pool[int(torch.randint(low=0, high=len(pool), size=(1,)).item())]
        support, query, target = generate_episode_from_law(law, B=B, N=N, x_dim=x_dim, q_dim=q_dim, device=device)

        opt.zero_grad()
        pred, z = model(support, query)
        loss = crit(pred, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()

        best = min(best, float(loss.item()))
        if ep == 1 or ep % 150 == 0 or ep == epochs:
            z_norm = float(z.detach().norm(dim=1).mean().item())
            print(f"[PRETRAIN {ep:03d}] loss={loss.item():.6f} best={best:.6f} ||z||={z_norm:.3f}")


@torch.no_grad()
def run_inference_stream(
    model: AIN,
    bank: ProgramBank,
    *,
    device: str,
    episodes: int = 300,
    B: int = 16,
    N: int = 64,
    x_dim: int = 6,
    q_dim: int = 4,
    law_hidden: int = 10,
    law_pool: int = 50,
    recurrence_p: float = 0.70,
    refresh_mse_threshold: float = 0.25,
    cost_reuse: float = 0.05,
    cost_recompile: float = 1.00,
    cost_refresh: float = 1.00,
    policy_mode: str = "v1",
    quality_tol: float = 0.02,
):
    crit = nn.MSELoss()

    if str(policy_mode).lower() == "v2":
        policy = BankPolicyV2(
            config=BankPolicyV2Config(
                epsilon=0.10,
                lr=2e-3,
                lambda_cost=0.15,
            ),
            device=device,
        )
    else:
        policy = BankPolicy(
            config=BankPolicyConfig(
                z_reuse_threshold=float(bank.z_threshold),
                signature_reuse_threshold=float(bank.signature_threshold),
                probe_mse_refresh_threshold=float(refresh_mse_threshold),
                enable_signature_fallback=bool(bank.enable_signature_fallback),
                enable_probe=True,
            )
        )
    contextual = ContextualAIN(
        bank=bank,
        policy=policy,
        infer_z=lambda support: model.eye(support),
        compile_forged=lambda z: model.forge(z),
        execute=lambda query, forged: effector_execute(model.effector, forged, query),
        device=device,
    )

    pool: List[Law] = [_make_random_law(q_dim, law_hidden, device=device) for _ in range(law_pool)]

    hits = 0
    refreshes = 0

    mse_reuse_sum = 0.0
    mse_compile_sum = 0.0

    expected_cost_sum = 0.0
    reward_sum = 0.0

    # Loi courante (avec recurrence)
    current_idx = int(torch.randint(low=0, high=len(pool), size=(1,)).item())

    model.to(device)
    model.eval()

    for t in range(1, episodes + 1):
        if torch.rand(()) > recurrence_p:
            current_idx = int(torch.randint(low=0, high=len(pool), size=(1,)).item())
        law = pool[current_idx]

        support, query, target = generate_episode_from_law(law, B=B, N=N, x_dim=x_dim, q_dim=q_dim, device=device)

        # Always-compile baseline
        z_compile = model.eye(support)
        forged_compile = model.forge(z_compile)
        pred_compile = effector_execute(model.effector, forged_compile, query)
        mse_compile = float(crit(pred_compile, target).item())
        mse_compile_sum += mse_compile

        # Bank mode
        pred_bank, logs = contextual.run(
            support=support,
            query=query,
            target=target,
            topk=5,
            costs={
                "reuse": float(cost_reuse),
                "recompile": float(cost_recompile),
                "refresh": float(cost_refresh),
            },
        )
        mse_bank = float(crit(pred_bank, target).item())
        hits += int(logs.get("reused", 0.0) > 0.5)
        refreshes += int(logs.get("refreshed", 0.0) > 0.5)
        expected_cost_sum += float(logs.get("expected_cost", 0.0))
        if float(logs.get("reward", -1.0)) != -1.0:
            reward_sum += float(logs.get("reward", 0.0))

        mse_reuse_sum += mse_bank

        if t == 1 or t % 50 == 0 or t == episodes:
            hit_rate = hits / max(t, 1)
            refresh_rate = refreshes / max(hits, 1)
            lam_c = float(logs.get("lambda_cost", -1.0))
            lam_q = float(logs.get("lambda_quality", -1.0))
            lam_z = float(logs.get("lambda_z", -1.0))
            mse_b = float(logs.get("mse_baseline", -1.0))
            qv = float(logs.get("quality_violation", -1.0))
            zv = float(logs.get("z_violation", -1.0))
            mse_bank_avg = mse_reuse_sum / max(t, 1)
            over = 0.0
            if mse_b >= 0.0:
                over = float(max(0.0, float(mse_bank_avg) - (float(mse_b) + float(quality_tol))))
            print(
                f"[STREAM {t:03d}] "
                f"hit_rate={hit_rate:.3f} refresh_rate={refresh_rate:.3f} "
                f"mse_bank={mse_bank_avg:.4f} mse_compile={mse_compile_sum / t:.4f} "
                f"bank={len(bank)} best_z={logs.get('best_z', 0.0):.3f} exp_cost={expected_cost_sum / t:.3f} reward={reward_sum / t:.3f} "
                f"lam_c={lam_c:.3f} lam_q={lam_q:.3f} lam_z={lam_z:.3f} mse_b={mse_b:.3f} qv={qv:.3f} zv={zv:.3f}"
            )
            if over > 0.0:
                print(f"[QUALITY WARN] mse_bank_avg - (mse_baseline + tol) = {over:.6f} (tol={float(quality_tol):.6f})")

    return {
        "episodes": episodes,
        "hit_rate": hits / max(episodes, 1),
        "refresh_rate": refreshes / max(hits, 1),
        "mse_bank": mse_reuse_sum / max(episodes, 1),
        "mse_always_compile": mse_compile_sum / max(episodes, 1),
        "expected_cost": expected_cost_sum / max(episodes, 1),
        "reward": reward_sum / max(episodes, 1),
        "bank_size": len(bank),
    }


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    policy_mode = "v1"
    quality_tol = 0.02
    for i, a in enumerate(sys.argv):
        if a == "--policy" and i + 1 < len(sys.argv):
            policy_mode = str(sys.argv[i + 1]).strip().lower()
        if a == "--quality_tol" and i + 1 < len(sys.argv):
            quality_tol = float(sys.argv[i + 1])

    print("=" * 70)
    print("  DEMO — ProgramBank Inference (reuse vs recompile)")
    print("=" * 70)

    x_dim, q_dim, z_dim = 6, 4, 36

    model = AIN(x_dim=x_dim, z_dim=z_dim, query_dim=q_dim, out_dim=1, hidden=64)

    print("\n" + "=" * 70)
    print("  PRETRAIN")
    print("=" * 70)
    pretrain(model, device=device, epochs=600, lr=8e-4, B=32, N=64, x_dim=x_dim, q_dim=q_dim)

    # On recupere la banque nativement remplie pendant le pretrain
    bank = model.bank
    bank.z_threshold = 0.93
    bank.signature_threshold = 0.65
    bank.enable_signature_fallback = True
    bank.device = device

    print("\n" + "=" * 70)
    print("  INFERENCE STREAM")
    print("=" * 70)
    metrics = run_inference_stream(model, bank, device=device, episodes=300, policy_mode=policy_mode, quality_tol=float(quality_tol))

    print("\n" + "=" * 70)
    for k, v in metrics.items():
        print(f"{k:<18s}: {v:.6f}" if isinstance(v, float) else f"{k:<18s}: {v}")
    print("=" * 70)


if __name__ == '__main__':
    main()
