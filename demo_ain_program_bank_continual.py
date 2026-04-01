"""\
==============================================================================
  DEMO — AIN Program Bank + Continual Learning (Replay)

  Objectif:
  - Maintenir une bank de "programmes" : signature + z + forged + episodes de replay
  - Entrainement incrémental sur une suite de distributions de lois (phases)
  - Mesurer l'oubli et l'effet du replay depuis la bank

  Notes:
  - On n'edite pas ain_neuron.py. La bank est un outil d'entrainement/inference.
==============================================================================
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim

from ain_neuron import AIN
from program_bank import ProgramBank, compute_context_signature


def _sample_law(B: int, q_dim: int, hidden: int, phase: int, *, device: str):
    # Phases: changer distributions => risque d'oubli
    W0 = torch.randn((B, hidden, q_dim), device=device) * (0.35 + 0.05 * phase)
    b0 = torch.randn((B, hidden), device=device) * (0.25 + 0.05 * phase)
    w1 = torch.randn((B, hidden), device=device) * (0.30 + 0.03 * phase)

    scale = torch.empty((B, 1), device=device).uniform_(0.6, 1.6 + 0.2 * phase)
    offset = torch.empty((B, 1), device=device).uniform_(-0.7 - 0.1 * phase, 0.7 + 0.1 * phase)
    return W0, b0, w1, scale, offset


def _law_eval(W0, b0, w1, scale, offset, x: torch.Tensor) -> torch.Tensor:
    proj = torch.einsum('bq,bhq->bh', x, W0) + b0
    h = torch.sin(proj)
    raw = (h * w1).sum(dim=-1, keepdim=True)
    y = scale * torch.tanh(raw) + offset
    return y


def generate_episode(B: int, N: int, x_dim: int, q_dim: int, phase: int, *, device: str):
    W0, b0, w1, scale, offset = _sample_law(B, q_dim, hidden=10, phase=phase, device=device)

    # Support: (x_i, y_i)
    x = torch.randn((B, N, q_dim), device=device) * 0.8
    # Pour evaluer sur support (B,N,...) on flatten les lois dans law_eval pointwise
    # y_i
    y = []
    for t in range(N):
        y.append(_law_eval(W0, b0, w1, scale, offset, x[:, t, :]))
    y = torch.stack(y, dim=1)  # (B,N,1)

    support = torch.zeros((B, N, x_dim), device=device)
    support[:, :, :q_dim] = x
    support[:, :, q_dim:q_dim + 1] = y
    if x_dim > q_dim + 1:
        support[:, :, q_dim + 1:] = 0.05 * torch.randn((B, N, x_dim - (q_dim + 1)), device=device)

    # Query/target
    query = torch.randn((B, q_dim), device=device) * 0.8
    target = _law_eval(W0, b0, w1, scale, offset, query)

    return support, query, target


@torch.no_grad()
def eval_on_bank(model: AIN, bank: ProgramBank, *, device: str, batch_size: int = 64) -> float:
    sample = bank.sample_replay(batch_size=batch_size)
    if sample is None:
        return float('nan')
    support, query, target = sample
    support = support.to(device)
    query = query.to(device)
    target = target.to(device)

    pred, _ = model(support, query)
    mse = nn.functional.mse_loss(pred, target).item()
    return float(mse)


def train_continual(*, seed: int = 0):
    torch.manual_seed(seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Model
    x_dim, q_dim, z_dim = 6, 4, 36
    model = AIN(x_dim=x_dim, z_dim=z_dim, query_dim=q_dim, out_dim=1, hidden=64).to(device)

    opt = optim.Adam(model.parameters(), lr=8e-4)
    crit = nn.MSELoss()

    bank = model.bank
    bank.device = device

    phases = 4
    steps_per_phase = 250

    B_new = 32
    N = 64

    replay_bs = 32
    replay_weight = 0.8

    print("=" * 70)
    print("  DEMO — ProgramBank Continual Learning (AIN)")
    print("=" * 70)

    for phase in range(phases):
        print("\n" + "=" * 70)
        print(f"  PHASE {phase} (distribution shift)")
        print("=" * 70)

        mse_before = eval_on_bank(model, bank, device=device, batch_size=64) if len(bank) > 0 else float('nan')
        if len(bank) > 0:
            print(f"[EVAL BEFORE] bank_mse={mse_before:.6f} bank_size={len(bank)}")
        else:
            print(f"[EVAL BEFORE] bank_mse=nan bank_size={len(bank)}")

        for step in range(1, steps_per_phase + 1):
            model.train()

            support, query, target = generate_episode(B_new, N, x_dim, q_dim, phase, device=device)

            opt.zero_grad()
            pred, z = model(support, query)
            loss_new = crit(pred, target)

            loss = loss_new

            if len(bank) > 0:
                replay = bank.sample_replay(batch_size=replay_bs)
                if replay is not None:
                    rs, rq, rt = replay
                    rs = rs.to(device)
                    rq = rq.to(device)
                    rt = rt.to(device)
                    pred_r, _ = model(rs, rq)
                    loss_replay = crit(pred_r, rt)
                    loss = loss_new + replay_weight * loss_replay

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            # L'archivage est desormais 100% automatique via model.bank
            # Plus besoin de bank.add(...) manuel ici !

            if step == 1 or step % 50 == 0 or step == steps_per_phase:
                z_norm = float(z.detach().norm(dim=1).mean().item())
                print(f"[PH{phase} STEP {step:03d}] loss={loss.item():.6f} loss_new={loss_new.item():.6f} ||z||={z_norm:.3f} bank={len(bank)}")

        mse_after = eval_on_bank(model, bank, device=device, batch_size=64)
        print(f"[EVAL AFTER] bank_mse={mse_after:.6f} bank_size={len(bank)}")

    # Quick serialization smoke test
    tmp_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_program_bank_tmp.pt")
    bank.save(tmp_path)
    bank2 = ProgramBank.load(tmp_path, device=device)
    print("\n" + "=" * 70)
    print(f"[SERIALIZATION] saved={len(bank)} loaded={len(bank2)} path={tmp_path}")
    print("=" * 70)


def main():
    train_continual(seed=0)


if __name__ == '__main__':
    main()
