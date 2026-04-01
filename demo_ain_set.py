"""
==============================================================================
  DEMO — AIN en mode ENSEMBLE (permutation-invariant)

  Objectif:
  - Tester AIN quand le support est un SET: on permute les noeuds a chaque episode.
  - 2 epreuves:
    1) NON-ALIGNE (discontinu + non-local)
    2) COMPOSE (produit de 3 invariants permutation-invariants) + curriculum

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

from ain_neuron import AIN


def _permute_nodes(support: torch.Tensor) -> torch.Tensor:
    """Permute les noeuds par batch (renforce la contrainte ensemble)."""
    B, N, D = support.shape
    idx = torch.stack([torch.randperm(N, device=support.device) for _ in range(B)], dim=0)  # (B,N)
    idx_exp = idx.unsqueeze(-1).expand(-1, -1, D)
    return torch.gather(support, dim=1, index=idx_exp)


def _inv_extreme_pair_set(support: torch.Tensor) -> torch.Tensor:
    """Invariant discontinu + non-local, permutation-invariant."""
    v = torch.tensor([0.61, -0.37, 0.19, 0.08], device=support.device, dtype=support.dtype)
    x = support[..., :4]
    proj = (x * v.view(1, 1, -1)).sum(dim=-1)  # (B,N)
    i_max = proj.argmax(dim=1)
    i_min = proj.argmin(dim=1)
    x_max = x[torch.arange(x.shape[0], device=support.device), i_max]
    x_min = x[torch.arange(x.shape[0], device=support.device), i_min]
    dot = (x_max * x_min).sum(dim=-1, keepdim=True)
    inv = torch.where(dot >= 0.0, torch.ones_like(dot), -torch.ones_like(dot))
    return inv


def _inv_parity_topk_set(support: torch.Tensor, *, k: int = 5) -> torch.Tensor:
    """Invariant discontinu (parite), permutation-invariant."""
    x = support[..., 0]
    vals, _ = x.abs().topk(k=min(int(k), int(x.shape[1])), dim=1)
    bits = (vals > 1.0).to(dtype=support.dtype)
    parity = (bits.sum(dim=1, keepdim=True) % 2.0)
    inv = torch.where(parity > 0.5, torch.ones_like(parity), -torch.ones_like(parity))
    return inv


def _inv_energy_split_by_median_set(support: torch.Tensor) -> torch.Tensor:
    """Invariant "mixte" mais permutation-invariant: split par projection vs median, puis compare energies."""
    B, N, D = support.shape
    x = support[..., :4]
    v = torch.tensor([0.17, 0.41, -0.29, 0.08], device=support.device, dtype=support.dtype)
    proj = (x * v.view(1, 1, -1)).sum(dim=-1)  # (B,N)
    med = proj.median(dim=1, keepdim=True).values  # (B,1)

    mask_a = (proj >= med).to(dtype=support.dtype)
    mask_b = 1.0 - mask_a

    count_a = mask_a.sum(dim=1, keepdim=True)
    need_fix = (count_a < 1.0).to(dtype=support.dtype)
    if float(need_fix.max().item()) > 0.0:
        i = proj.argmax(dim=1)
        mask_a = mask_a.clone()
        mask_a[torch.arange(B, device=support.device), i] = 1.0
        mask_b = 1.0 - mask_a

    node_energy = (support ** 2).sum(dim=-1)  # (B,N)
    ea = (node_energy * mask_a).sum(dim=1, keepdim=True)
    eb = (node_energy * mask_b).sum(dim=1, keepdim=True)
    diff = ea - eb
    inv = torch.where(diff >= 0.0, torch.ones_like(diff), -torch.ones_like(diff))
    return inv


def generate_unaligned_episode(B: int, N: int, x_dim: int, q_dim: int):
    support = torch.randn(B, N, x_dim)
    query = torch.randn(B, q_dim)

    support = _permute_nodes(support)

    inv = _inv_extreme_pair_set(support)

    w = torch.linspace(-1.0, 1.0, int(q_dim)).view(1, -1)
    qproj = (query * w).sum(dim=1, keepdim=True)
    target = inv * torch.tanh(torch.sin(qproj))

    return support, query, target


def generate_composed_episode(B: int, N: int, x_dim: int, q_dim: int, *, mode: str = "composed"):
    support = torch.randn(B, N, x_dim)
    query = torch.randn(B, q_dim)

    support = _permute_nodes(support)

    if mode == "inv1":
        inv = _inv_extreme_pair_set(support)
    elif mode == "inv2":
        inv = _inv_parity_topk_set(support, k=5)
    elif mode == "inv3":
        inv = _inv_energy_split_by_median_set(support)
    elif mode == "composed":
        inv = _inv_extreme_pair_set(support) * _inv_parity_topk_set(support, k=5) * _inv_energy_split_by_median_set(support)
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


def _sign_acc(pred: torch.Tensor, target: torch.Tensor) -> float:
    return (torch.sign(pred) == torch.sign(target)).float().mean().item()


def _routing_stats(model: AIN, support: torch.Tensor):
    """Extrait gates/blend du micro-gating contextuel (sans toucher au neurone)."""
    if hasattr(model.eye, '_debug_gates_voies'):
        gates_mean = model.eye._debug_gates_voies.mean(dim=0)
        blend_mean = model.eye._debug_blend.mean() if hasattr(model.eye, '_debug_blend') else torch.zeros((), device=support.device)
        chronos_mean = gates_mean[4] if int(gates_mean.shape[0]) > 4 else torch.zeros((), device=support.device)
    else:
        gates_mean = torch.zeros(9, device=support.device)
        blend_mean = torch.zeros((), device=support.device)
        chronos_mean = torch.zeros((), device=support.device)
    return gates_mean, blend_mean, chronos_mean


def train(
    model: AIN,
    gen_fn,
    *,
    epochs: int,
    lr: float,
    device: str,
    label: str,
    log_every: int = 50,
    perm_aug_weight: float = 1.0,
    num_perm_augs: int = 1,
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
        loss_task = crit(pred, target)

        # Augmentation SET organique : permuter le support mais garder la meme loi/target
        # (on n'impose pas z==z_perm explicitement; on impose seulement la meme performance)
        z_cos = torch.zeros((), device=device)
        gates_l1 = torch.zeros((), device=device)
        blend_l1 = torch.zeros((), device=device)

        loss_perm = torch.zeros((), device=device)
        if float(perm_aug_weight) > 0.0:
            gates_a = model.eye._debug_gates_voies if hasattr(model.eye, '_debug_gates_voies') else None
            blend_a = model.eye._debug_blend if hasattr(model.eye, '_debug_blend') else None

            support_p = _permute_nodes(support)
            pred_p, z_p = model(support_p, query)
            loss_perm = crit(pred_p, target)

            z_cos = F.cosine_similarity(z, z_p, dim=-1).mean()
            if gates_a is not None and hasattr(model.eye, '_debug_gates_voies'):
                gates_l1 = (gates_a - model.eye._debug_gates_voies).abs().mean()
            if blend_a is not None and hasattr(model.eye, '_debug_blend'):
                blend_l1 = (blend_a - model.eye._debug_blend).abs().mean()

            if int(num_perm_augs) >= 2:
                support_p2 = _permute_nodes(support)
                pred_p2, _z_p2 = model(support_p2, query)
                loss_perm = loss_perm + crit(pred_p2, target)

        loss = loss_task + float(perm_aug_weight) * loss_perm
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()

        best = min(best, float(loss_task.item()))

        if ep == 1 or ep % log_every == 0 or ep == epochs:
            acc = _sign_acc(pred.detach(), target)
            z_norm = z.detach().norm(dim=1).mean().item()
            with torch.no_grad():
                gates_mean, blend_mean, chronos_mean = _routing_stats(model, support)
            gm = ",".join([f"{v:.3f}" for v in gates_mean.detach().cpu().tolist()])
            print(
                f"[{label} EP {ep:03d}] loss={loss_task.item():.6f} best={best:.6f} sign_acc={acc:.4f} ||Z||={z_norm:.4f} "
                f"gates_mean=[{gm}] blend_mean={float(blend_mean):.3f} chronos_mean={float(chronos_mean):.3f}"
            )

            if float(perm_aug_weight) > 0.0:
                print(
                    f"[{label} PERM-AUG] loss_perm={float(loss_perm.detach().item()):.6f} z_cos={float(z_cos.detach().item()):.4f} "
                    f"gates_l1={float(gates_l1.detach().item()):.6f} blend_l1={float(blend_l1.detach().item()):.6f} "
                    f"(perm_aug_weight={float(perm_aug_weight):.3f} num_perm_augs={int(num_perm_augs)})"
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

        support_p = _permute_nodes(support)
        pred_p, z_p = model(support_p, query)
        mse_p = crit(pred_p, target).item()
        acc_p = _sign_acc(pred_p, target)
        gates_mean_p, blend_mean_p, chronos_mean_p = _routing_stats(model, support_p)
        gm_p = ",".join([f"{v:.3f}" for v in gates_mean_p.detach().cpu().tolist()])

        z_cos = F.cosine_similarity(z, z_p, dim=-1).mean().item()
        z_l2 = (z - z_p).pow(2).sum(dim=-1).sqrt().mean().item()
        gates_l1 = (gates_mean - gates_mean_p).abs().mean().item()
        blend_l1 = float((blend_mean - blend_mean_p).abs().item())

        print(
            f"[{label} FINAL-PERM] mse={mse_p:.6f} sign_acc={acc_p:.4f} "
            f"gates_mean=[{gm_p}] blend_mean={float(blend_mean_p):.3f} chronos_mean={float(chronos_mean_p):.3f} "
            f"| z_cos={z_cos:.4f} z_l2={z_l2:.4f} gates_l1={gates_l1:.4f} blend_l1={blend_l1:.4f}"
        )
    return mse, acc


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("=" * 70)
    print("  DEMO AIN — MODE ENSEMBLE (SET / permutation-invariant)")
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
    print("  EPREUVE 1: NON-ALIGNE (discontinu + non-local)")
    print("=" * 70)
    model_u = AIN(x_dim=x_dim, z_dim=z_dim, query_dim=q_dim, out_dim=1, hidden=hidden)
    train(
        model_u,
        lambda: generate_unaligned_episode(B, N, x_dim, q_dim),
        epochs=250,
        lr=lr,
        device=device,
        label="UNALIGNED",
        log_every=50,
        perm_aug_weight=1.0,
        num_perm_augs=2,
    )
    results["UNALIGNED-SET"] = eval_model(
        model_u,
        lambda: generate_unaligned_episode(256, N, x_dim, q_dim),
        device=device,
        label="UNALIGNED",
    )

    print("\n" + "=" * 70)
    print("  EPREUVE 2: COMPOSE (curriculum inv1->inv2->inv3 puis composed)")
    print("=" * 70)
    model_c = AIN(x_dim=x_dim, z_dim=z_dim, query_dim=q_dim, out_dim=1, hidden=hidden)

    for mode in ["inv1", "inv2", "inv3"]:
        train(
            model_c,
            lambda m=mode: generate_composed_episode(B, N, x_dim, q_dim, mode=m),
            epochs=250,
            lr=lr,
            device=device,
            label=f"PRE-{mode}",
            log_every=50,
            perm_aug_weight=1.0,
            num_perm_augs=2,
        )

    train(
        model_c,
        lambda: generate_composed_episode(B, N, x_dim, q_dim, mode="composed"),
        epochs=250,
        lr=lr,
        device=device,
        label="POST-COMPOSED",
        log_every=50,
        perm_aug_weight=1.0,
        num_perm_augs=2,
    )
    results["COMPOSED-SET"] = eval_model(
        model_c,
        lambda: generate_composed_episode(256, N, x_dim, q_dim, mode="composed"),
        device=device,
        label="COMPOSED",
    )

    print("\n" + "=" * 70)
    print("  TABLEAU RECAPITULATIF — AIN SET")
    print("=" * 70)
    print(f"  {'Test':<20s} {'MSE':>10s} {'sign_acc':>10s}")
    print(f"  {'-'*20} {'-'*10} {'-'*10}")
    for name, (mse, acc) in results.items():
        print(f"  {name:<20s} {mse:>10.6f} {acc:>10.4f}")


if __name__ == '__main__':
    main()
