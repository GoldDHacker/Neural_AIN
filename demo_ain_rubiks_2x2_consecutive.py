import os
import sys
import math
from typing import List, Tuple

import torch
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ain_neuron import AIN
from demo_ain_rubiks_2x2 import (
    decode_sticker_accuracy,
    exact_state_accuracy,
    generate_episode,
    rubiks_ce_loss,
    validate_move_tables,
)


def train_curriculum_consecutive(
    *,
    device: str,
    epochs1: int,
    epochs2: int,
    epochs3: int,
    B: int,
    N: int,
    hidden: int,
    z_dim: int,
    lr: float,
    threshold1: float,
    threshold2: float,
    threshold3: float,
    piece_thr1: float,
    piece_thr2: float,
    piece_thr3: float,
    ori_thr1: float,
    ori_thr2: float,
    ori_thr3: float,
    scramble1: int,
    scramble2: int,
    scramble3: int,
    eval_every: int,
    eval_batches: int,
    pass_k: int,
) -> None:
    x_dim = 2 * (8 * 11)
    q_dim = 8 * 11
    out_dim = 8 * 11

    model = AIN(x_dim=x_dim, z_dim=z_dim, query_dim=q_dim, out_dim=out_dim, hidden=hidden).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    crit = rubiks_ce_loss

    def _eval(*, max_seq_len: int, scramble_max_len: int) -> Tuple[float, float, float, float]:
        model.eval()
        losses: List[float] = []
        eaccs: List[float] = []
        paccs: List[float] = []
        oaccs: List[float] = []
        with torch.no_grad():
            for _ in range(int(eval_batches)):
                support, query, target, _seq = generate_episode(
                    B=B,
                    N=N,
                    device=device,
                    max_seq_len=max_seq_len,
                    scramble_max_len=int(scramble_max_len),
                )
                pred, _z = model(support, query)
                losses.append(float(crit(pred, target).item()))
                pacc, oacc = decode_sticker_accuracy(pred, target)
                eaccs.append(exact_state_accuracy(pred, target))
                paccs.append(float(pacc))
                oaccs.append(float(oacc))
        model.train()
        return (
            float(sum(losses) / max(len(losses), 1)),
            float(sum(eaccs) / max(len(eaccs), 1)),
            float(sum(paccs) / max(len(paccs), 1)),
            float(sum(oaccs) / max(len(oaccs), 1)),
        )

    def _train_stage(
        *,
        stage_id: int,
        max_seq_len: int,
        epochs: int,
        threshold_exact: float,
        threshold_piece: float,
        threshold_ori: float,
        scramble_max_len: int,
    ) -> bool:
        ema_loss = None
        consecutive = 0

        print("\n" + "=" * 70)
        print(
            f"  STAGE {stage_id}/3 — max_seq_len={max_seq_len} — "
            f"targets: piece>={threshold_piece:.2f} ori>={threshold_ori:.2f} (exact>={threshold_exact:.2f} optional)"
        )
        print(f"  scramble_max_len={int(scramble_max_len)} — pass_k={int(pass_k)}")
        print("=" * 70)

        for ep in range(1, int(epochs) + 1):
            support, query, target, seq = generate_episode(
                B=B,
                N=N,
                device=device,
                max_seq_len=max_seq_len,
                scramble_max_len=int(scramble_max_len),
            )
            opt.zero_grad()
            pred, _z = model(support, query)
            loss = crit(pred, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            l = float(loss.item())
            if not math.isfinite(l):
                print(f"[WARN] Non-finite loss detected at stage={stage_id} ep={ep}: loss={l} -> skipping EMA update")
                continue
            if ema_loss is None:
                ema_loss = l
            else:
                prev_ema = float(ema_loss)
                if not math.isfinite(prev_ema):
                    print(
                        f"[WARN] Non-finite EMA detected at stage={stage_id} ep={ep}: ema={prev_ema} (loss={l}) -> reset EMA"
                    )
                    ema_loss = l
                else:
                    ema_loss = 0.97 * prev_ema + 0.03 * l
                    if (not math.isfinite(float(ema_loss))) or (abs(float(ema_loss)) > 1e6):
                        print(
                            f"[WARN] EMA became invalid/huge at stage={stage_id} ep={ep}: ema={float(ema_loss)} (prev={prev_ema}, loss={l}) -> reset EMA"
                        )
                        ema_loss = l

            if ep == 1 or ep % int(eval_every) == 0 or ep == int(epochs):
                ev_loss, ev_exact, ev_piece, ev_ori = _eval(max_seq_len=max_seq_len, scramble_max_len=int(scramble_max_len))
                seq_str = "".join(seq)

                passed_once = (float(ev_piece) >= float(threshold_piece)) and (float(ev_ori) >= float(threshold_ori))
                if passed_once:
                    consecutive += 1
                else:
                    consecutive = 0

                print(
                    f"[S{stage_id} EP {ep:04d}] loss={l:.6f} ema={float(ema_loss):.6f} seq={seq_str:<8s} "
                    f"eval_loss={ev_loss:.6f} eval_exact={ev_exact:.3f} eval_piece={ev_piece:.3f} eval_ori={ev_ori:.3f} "
                    f"pass_streak={int(consecutive)}/{int(pass_k)}"
                )

                if int(consecutive) >= int(pass_k):
                    print(
                        f"  >> STAGE {stage_id} PASSED at ep={ep} "
                        f"(eval_piece={ev_piece:.3f} eval_ori={ev_ori:.3f} eval_exact={ev_exact:.3f}; "
                        f"streak={int(consecutive)}/{int(pass_k)})"
                    )
                    return True

        ev_loss, ev_exact, ev_piece, ev_ori = _eval(max_seq_len=max_seq_len, scramble_max_len=int(scramble_max_len))
        print(
            f"  !! STAGE {stage_id} NOT PASSED "
            f"(eval_piece={ev_piece:.3f} < {threshold_piece:.2f} or eval_ori={ev_ori:.3f} < {threshold_ori:.2f}; "
            f"eval_exact={ev_exact:.3f}; streak_end={int(consecutive)}/{int(pass_k)})"
        )
        return False

    ok1 = _train_stage(
        stage_id=1,
        max_seq_len=1,
        epochs=epochs1,
        threshold_exact=threshold1,
        threshold_piece=piece_thr1,
        threshold_ori=ori_thr1,
        scramble_max_len=int(scramble1),
    )
    if not bool(ok1):
        print("\n" + "=" * 70)
        print("  CURRICULUM STOPPED: stage 1 failed")
        print("=" * 70)
        return

    ok2 = _train_stage(
        stage_id=2,
        max_seq_len=3,
        epochs=epochs2,
        threshold_exact=threshold2,
        threshold_piece=piece_thr2,
        threshold_ori=ori_thr2,
        scramble_max_len=int(scramble2),
    )
    if not bool(ok2):
        print("\n" + "=" * 70)
        print("  CURRICULUM STOPPED: stage 2 failed")
        print("=" * 70)
        return

    _train_stage(
        stage_id=3,
        max_seq_len=8,
        epochs=epochs3,
        threshold_exact=threshold3,
        threshold_piece=piece_thr3,
        threshold_ori=ori_thr3,
        scramble_max_len=int(scramble3),
    )


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    validate_move_tables()

    epochs1 = 700
    epochs2 = 1000
    epochs3 = 1500
    threshold1 = 0.85
    threshold2 = 0.75
    threshold3 = 0.55

    piece_thr1 = 0.85
    piece_thr2 = 0.70
    piece_thr3 = 0.60

    ori_thr1 = 0.90
    ori_thr2 = 0.75
    ori_thr3 = 0.65

    scramble1 = 2
    scramble2 = 6
    scramble3 = 12

    B = 32
    N = 24
    hidden = 128
    z_dim = 64
    lr = 8e-4

    eval_every = 50
    eval_batches = 6
    pass_k = 3

    for i, a in enumerate(sys.argv):
        if a == "--epochs1" and i + 1 < len(sys.argv):
            epochs1 = int(sys.argv[i + 1])
        if a == "--epochs2" and i + 1 < len(sys.argv):
            epochs2 = int(sys.argv[i + 1])
        if a == "--epochs3" and i + 1 < len(sys.argv):
            epochs3 = int(sys.argv[i + 1])
        if a == "--threshold1" and i + 1 < len(sys.argv):
            threshold1 = float(sys.argv[i + 1])
        if a == "--threshold2" and i + 1 < len(sys.argv):
            threshold2 = float(sys.argv[i + 1])
        if a == "--threshold3" and i + 1 < len(sys.argv):
            threshold3 = float(sys.argv[i + 1])
        if a == "--piece_thr1" and i + 1 < len(sys.argv):
            piece_thr1 = float(sys.argv[i + 1])
        if a == "--piece_thr2" and i + 1 < len(sys.argv):
            piece_thr2 = float(sys.argv[i + 1])
        if a == "--piece_thr3" and i + 1 < len(sys.argv):
            piece_thr3 = float(sys.argv[i + 1])
        if a == "--ori_thr1" and i + 1 < len(sys.argv):
            ori_thr1 = float(sys.argv[i + 1])
        if a == "--ori_thr2" and i + 1 < len(sys.argv):
            ori_thr2 = float(sys.argv[i + 1])
        if a == "--ori_thr3" and i + 1 < len(sys.argv):
            ori_thr3 = float(sys.argv[i + 1])
        if a == "--scramble1" and i + 1 < len(sys.argv):
            scramble1 = int(sys.argv[i + 1])
        if a == "--scramble2" and i + 1 < len(sys.argv):
            scramble2 = int(sys.argv[i + 1])
        if a == "--scramble3" and i + 1 < len(sys.argv):
            scramble3 = int(sys.argv[i + 1])
        if a == "--B" and i + 1 < len(sys.argv):
            B = int(sys.argv[i + 1])
        if a == "--N" and i + 1 < len(sys.argv):
            N = int(sys.argv[i + 1])
        if a == "--hidden" and i + 1 < len(sys.argv):
            hidden = int(sys.argv[i + 1])
        if a == "--z_dim" and i + 1 < len(sys.argv):
            z_dim = int(sys.argv[i + 1])
        if a == "--lr" and i + 1 < len(sys.argv):
            lr = float(sys.argv[i + 1])
        if a == "--eval_every" and i + 1 < len(sys.argv):
            eval_every = int(sys.argv[i + 1])
        if a == "--eval_batches" and i + 1 < len(sys.argv):
            eval_batches = int(sys.argv[i + 1])
        if a == "--pass_k" and i + 1 < len(sys.argv):
            pass_k = int(sys.argv[i + 1])

    print("=" * 70)
    print("  DEMO — AIN x Rubik 2x2 (curriculum, PASS K consecutive evals)")
    print("=" * 70)

    train_curriculum_consecutive(
        device=device,
        epochs1=epochs1,
        epochs2=epochs2,
        epochs3=epochs3,
        B=B,
        N=N,
        hidden=hidden,
        z_dim=z_dim,
        lr=lr,
        threshold1=threshold1,
        threshold2=threshold2,
        threshold3=threshold3,
        piece_thr1=piece_thr1,
        piece_thr2=piece_thr2,
        piece_thr3=piece_thr3,
        ori_thr1=ori_thr1,
        ori_thr2=ori_thr2,
        ori_thr3=ori_thr3,
        scramble1=scramble1,
        scramble2=scramble2,
        scramble3=scramble3,
        eval_every=eval_every,
        eval_batches=eval_batches,
        pass_k=pass_k,
    )


if __name__ == "__main__":
    main()
