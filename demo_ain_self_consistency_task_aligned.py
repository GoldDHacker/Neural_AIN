import torch
import torch.nn as nn
import torch.nn.functional as F

from ain_neuron import AIN
from demo_ain_set import (
    _permute_nodes,
    _routing_stats,
    generate_composed_episode,
    generate_unaligned_episode,
    train,
)


def _system2_rollout_with_oracle(
    model: AIN,
    support: torch.Tensor,
    query: torch.Tensor,
    target: torch.Tensor,
    *,
    oracle: str,
):
    ctx = model.eye.prepare_voies(support)

    T = int(model.eye.T)
    best_z = None
    best_error = float("inf")
    best_t = -1
    error_feedback = None

    B, N, D = support.shape

    if oracle == "support_std":
        eval_q = support.mean(dim=1)[:, : query.shape[1]]
        eval_target = support.std(dim=1)[:, :1]
    elif oracle == "task":
        eval_q = query
        eval_target = target
    else:
        raise ValueError(f"oracle inconnu: {oracle}")

    for t_iter in range(T):
        z = model.eye.step_routing(ctx, error_feedback=error_feedback)
        forged = model.forge(z)

        with torch.no_grad():
            eval_pred = model.effector(eval_q, forged)
            trial_error = (eval_pred - eval_target).pow(2).mean(dim=1, keepdim=True)

        current_mean_error = float(trial_error.mean().item())
        if current_mean_error < best_error:
            best_error = current_mean_error
            best_z = z
            best_t = int(t_iter)

        error_feedback = trial_error

    z = best_z if best_z is not None else z
    forged = model.forge(z)
    pred = model.effector(query, forged)
    return pred, z, best_error, best_t


def _sign_acc(pred: torch.Tensor, target: torch.Tensor) -> float:
    return (pred.sign() == target.sign()).float().mean().item()


def _l1(x: torch.Tensor, y: torch.Tensor) -> float:
    return float((x - y).abs().mean().item())


def _cos(x: torch.Tensor, y: torch.Tensor) -> float:
    return float(F.cosine_similarity(x, y, dim=-1).mean().item())


def _l2(x: torch.Tensor, y: torch.Tensor) -> float:
    return float((x - y).pow(2).sum(dim=-1).sqrt().mean().item())


def _hist_to_str(hist: dict, *, T: int) -> str:
    total = sum(int(v) for v in hist.values())
    parts = []
    for t in range(T):
        c = int(hist.get(t, 0))
        frac = (c / max(1, total))
        parts.append(f"{t}:{c}({frac:.2f})")
    return " ".join(parts)


def _eval_many_batches(model: AIN, gen_fn, *, device: str, label: str, batches: int):
    crit = nn.MSELoss()
    model.eval()

    T = int(model.eye.T)

    mse_native_sum = 0.0
    acc_native_sum = 0.0

    mse_std_sum = 0.0
    acc_std_sum = 0.0
    trial_std_sum = 0.0
    best_t_std_hist = {}

    mse_task_sum = 0.0
    acc_task_sum = 0.0
    trial_task_sum = 0.0
    best_t_task_hist = {}

    z_cos_std_task_sum = 0.0
    z_l2_std_task_sum = 0.0

    gates_l1_std_task_sum = 0.0
    blend_l1_std_task_sum = 0.0

    for _ in range(int(batches)):
        with torch.no_grad():
            support, query, target = gen_fn()
            support, query, target = support.to(device), query.to(device), target.to(device)

            pred_native, z_native = model(support, query)
            mse_native_sum += float(crit(pred_native, target).item())
            acc_native_sum += _sign_acc(pred_native, target)
            gates_native, blend_native, _ = _routing_stats(model, support)

            pred_std, z_std, err_std, t_std = _system2_rollout_with_oracle(
                model, support, query, target, oracle="support_std"
            )
            mse_std_sum += float(crit(pred_std, target).item())
            acc_std_sum += _sign_acc(pred_std, target)
            trial_std_sum += float(err_std)
            best_t_std_hist[int(t_std)] = int(best_t_std_hist.get(int(t_std), 0)) + 1
            gates_std, blend_std, _ = _routing_stats(model, support)

            pred_task, z_task, err_task, t_task = _system2_rollout_with_oracle(
                model, support, query, target, oracle="task"
            )
            mse_task_sum += float(crit(pred_task, target).item())
            acc_task_sum += _sign_acc(pred_task, target)
            trial_task_sum += float(err_task)
            best_t_task_hist[int(t_task)] = int(best_t_task_hist.get(int(t_task), 0)) + 1
            gates_task, blend_task, _ = _routing_stats(model, support)

            z_cos_std_task_sum += _cos(z_std, z_task)
            z_l2_std_task_sum += _l2(z_std, z_task)

            gates_l1_std_task_sum += _l1(gates_std, gates_task)
            blend_l1_std_task_sum += float((blend_std - blend_task).abs().item())

    den = float(max(1, int(batches)))
    out = {
        "native": {
            "mse": mse_native_sum / den,
            "sign_acc": acc_native_sum / den,
        },
        "support_std": {
            "mse": mse_std_sum / den,
            "sign_acc": acc_std_sum / den,
            "trial_err": trial_std_sum / den,
            "best_t_hist": best_t_std_hist,
        },
        "task": {
            "mse": mse_task_sum / den,
            "sign_acc": acc_task_sum / den,
            "trial_err": trial_task_sum / den,
            "best_t_hist": best_t_task_hist,
        },
        "std_vs_task": {
            "z_cos": z_cos_std_task_sum / den,
            "z_l2": z_l2_std_task_sum / den,
            "gates_l1": gates_l1_std_task_sum / den,
            "blend_l1": blend_l1_std_task_sum / den,
        },
        "meta": {
            "T": T,
            "batches": int(batches),
            "label": str(label),
        },
    }
    return out


def _print_eval(out: dict):
    T = int(out["meta"]["T"])
    batches = int(out["meta"]["batches"])
    label = str(out["meta"]["label"])

    n = out["native"]
    s = out["support_std"]
    t = out["task"]
    d = out["std_vs_task"]

    print("\n" + "=" * 70)
    print(f"  EVAL MULTI-BATCHES — {label} (batches={batches})")
    print("=" * 70)
    print(f"[NATIVE forward]        mse={n['mse']:.6f} sign_acc={n['sign_acc']:.4f} best_t=N/A")
    print(
        f"[ORACLE support_std]    mse={s['mse']:.6f} sign_acc={s['sign_acc']:.4f} "
        f"trial_err={s['trial_err']:.6f} best_t_hist={_hist_to_str(s['best_t_hist'], T=T)}"
    )
    print(
        f"[ORACLE task-aligned]   mse={t['mse']:.6f} sign_acc={t['sign_acc']:.4f} "
        f"trial_err={t['trial_err']:.6f} best_t_hist={_hist_to_str(t['best_t_hist'], T=T)}"
    )
    print(
        f"[STD vs TASK] z_cos={d['z_cos']:.4f} z_l2={d['z_l2']:.4f} "
        f"gates_l1={d['gates_l1']:.6f} blend_l1={d['blend_l1']:.6f}"
    )


def eval_oracles(model: AIN, gen_fn, *, device: str, label: str, batches: int):
    out = _eval_many_batches(model, gen_fn, device=device, label=label, batches=batches)
    _print_eval(out)
    return out


def _run_seed(seed: int, *, device: str, batches_eval: int):
    torch.manual_seed(int(seed))

    B = 64
    N = 28
    x_dim = 8
    q_dim = 6
    z_dim = 36
    hidden = 64
    lr = 2e-3

    print("\n" + "#" * 70)
    print(f"SEED={seed} device={device}")
    print("#" * 70)

    model_u = AIN(x_dim=x_dim, z_dim=z_dim, query_dim=q_dim, out_dim=1, hidden=hidden)
    train(
        model_u,
        lambda: generate_unaligned_episode(B, N, x_dim, q_dim),
        epochs=250,
        lr=lr,
        device=device,
        label=f"S{seed}-UNALIGNED",
        log_every=50,
        perm_aug_weight=1.0,
        num_perm_augs=2,
    )

    out_u = eval_oracles(
        model_u,
        lambda: generate_unaligned_episode(256, N, x_dim, q_dim),
        device=device,
        label=f"S{seed}-UNALIGNED",
        batches=batches_eval,
    )

    model_c = AIN(x_dim=x_dim, z_dim=z_dim, query_dim=q_dim, out_dim=1, hidden=hidden)
    for mode in ["inv1", "inv2", "inv3"]:
        train(
            model_c,
            lambda m=mode: generate_composed_episode(B, N, x_dim, q_dim, mode=m),
            epochs=250,
            lr=lr,
            device=device,
            label=f"S{seed}-PRE-{mode}",
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
        label=f"S{seed}-POST-COMPOSED",
        log_every=50,
        perm_aug_weight=1.0,
        num_perm_augs=2,
    )

    out_c = eval_oracles(
        model_c,
        lambda: generate_composed_episode(256, N, x_dim, q_dim, mode="composed"),
        device=device,
        label=f"S{seed}-COMPOSED",
        batches=batches_eval,
    )

    return {"unaligned": out_u, "composed": out_c}


def _avg_metrics(outs: list, path: tuple):
    s = 0.0
    for o in outs:
        cur = o
        for k in path:
            cur = cur[k]
        s += float(cur)
    return s / float(max(1, len(outs)))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    seeds = [0, 1, 2]
    batches_eval = 20

    outs = []
    for s in seeds:
        outs.append(_run_seed(int(s), device=device, batches_eval=batches_eval))

    unaligned = [o["unaligned"] for o in outs]
    composed = [o["composed"] for o in outs]

    print("\n" + "=" * 70)
    print("  RESUME AGREGÉ (moyenne sur 3 seeds)")
    print("=" * 70)
    print(
        "[UNALIGNED] native_mse={:.6f} native_acc={:.4f} | task_mse={:.6f} task_acc={:.4f} | std_vs_task z_cos={:.4f} gates_l1={:.6f}".format(
            _avg_metrics(unaligned, ("native", "mse")),
            _avg_metrics(unaligned, ("native", "sign_acc")),
            _avg_metrics(unaligned, ("task", "mse")),
            _avg_metrics(unaligned, ("task", "sign_acc")),
            _avg_metrics(unaligned, ("std_vs_task", "z_cos")),
            _avg_metrics(unaligned, ("std_vs_task", "gates_l1")),
        )
    )
    print(
        "[COMPOSED ] native_mse={:.6f} native_acc={:.4f} | task_mse={:.6f} task_acc={:.4f} | std_vs_task z_cos={:.4f} gates_l1={:.6f}".format(
            _avg_metrics(composed, ("native", "mse")),
            _avg_metrics(composed, ("native", "sign_acc")),
            _avg_metrics(composed, ("task", "mse")),
            _avg_metrics(composed, ("task", "sign_acc")),
            _avg_metrics(composed, ("std_vs_task", "z_cos")),
            _avg_metrics(composed, ("std_vs_task", "gates_l1")),
        )
    )


if __name__ == "__main__":
    main()
