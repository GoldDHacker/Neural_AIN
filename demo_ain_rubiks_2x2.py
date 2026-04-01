import os
import sys
import random
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ain_neuron import AIN


CORNER_NAMES = ["URF", "UFL", "ULB", "UBR", "DFR", "DLF", "DBL", "DRB"]


@dataclass
class Cube2x2:
    cp: List[int]
    co: List[int]


def cube_solved() -> Cube2x2:
    return Cube2x2(cp=list(range(8)), co=[0] * 8)


MOVE_TABLES = {
    "U": {
        "cp": [3, 0, 1, 2, 4, 5, 6, 7],
        "co_delta": [0, 0, 0, 0, 0, 0, 0, 0],
    },
    "R": {
        "cp": [3, 1, 2, 7, 0, 5, 6, 4],
        "co_delta": [2, 0, 0, 1, 1, 0, 0, 2],
    },
    "F": {
        "cp": [4, 0, 2, 3, 5, 1, 6, 7],
        "co_delta": [1, 2, 0, 0, 2, 1, 0, 0],
    },
}


def apply_move(c: Cube2x2, mv: str) -> Cube2x2:
    t = MOVE_TABLES[mv]
    cp_map = t["cp"]
    co_delta = t["co_delta"]

    new_cp = [0] * 8
    new_co = [0] * 8
    for pos in range(8):
        src = cp_map[pos]
        new_cp[pos] = c.cp[src]
        new_co[pos] = (c.co[src] + co_delta[pos]) % 3
    return Cube2x2(cp=new_cp, co=new_co)


def apply_moves(c: Cube2x2, seq: List[str]) -> Cube2x2:
    out = c
    for mv in seq:
        out = apply_move(out, mv)
    return out


def random_scramble_state(max_len: int = 12) -> Cube2x2:
    c = cube_solved()
    L = random.randint(0, max_len)
    moves = [random.choice(["U", "R", "F"]) for _ in range(L)]
    return apply_moves(c, moves)


def cube_to_onehot(c: Cube2x2, *, device: str) -> torch.Tensor:
    x = torch.zeros(8, 11, device=device)
    for pos in range(8):
        piece = int(c.cp[pos])
        ori = int(c.co[pos])
        x[pos, piece] = 1.0
        x[pos, 8 + ori] = 1.0
    return x.reshape(-1)


def batch_cube_to_onehot(cubes: List[Cube2x2], *, device: str) -> torch.Tensor:
    return torch.stack([cube_to_onehot(c, device=device) for c in cubes], dim=0)


def decode_sticker_accuracy(pred: torch.Tensor, target: torch.Tensor) -> Tuple[float, float]:
    B = int(pred.shape[0])
    pred = pred.view(B, 8, 11)
    target = target.view(B, 8, 11)

    pred_piece = torch.argmax(pred[:, :, 0:8], dim=-1)
    tgt_piece = torch.argmax(target[:, :, 0:8], dim=-1)
    pred_ori = torch.argmax(pred[:, :, 8:11], dim=-1)
    tgt_ori = torch.argmax(target[:, :, 8:11], dim=-1)

    piece_acc = (pred_piece == tgt_piece).float().mean().item()
    ori_acc = (pred_ori == tgt_ori).float().mean().item()
    return float(piece_acc), float(ori_acc)


def exact_state_accuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
    B = int(pred.shape[0])
    pred = pred.view(B, 8, 11)
    target = target.view(B, 8, 11)

    pred_piece = torch.argmax(pred[:, :, 0:8], dim=-1)
    tgt_piece = torch.argmax(target[:, :, 0:8], dim=-1)
    pred_ori = torch.argmax(pred[:, :, 8:11], dim=-1)
    tgt_ori = torch.argmax(target[:, :, 8:11], dim=-1)

    ok = (pred_piece == tgt_piece) & (pred_ori == tgt_ori)
    return float(ok.all(dim=1).float().mean().item())


def rubiks_ce_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Cross-entropy sur les coins: piece (8 classes) + orientation (3 classes)."""
    B = int(pred.shape[0])
    pred = pred.view(B, 8, 11)
    target = target.view(B, 8, 11)

    tgt_piece = torch.argmax(target[:, :, 0:8], dim=-1)  # (B, 8)
    tgt_ori = torch.argmax(target[:, :, 8:11], dim=-1)  # (B, 8)

    piece_logits = pred[:, :, 0:8].reshape(B * 8, 8)
    ori_logits = pred[:, :, 8:11].reshape(B * 8, 3)

    loss_piece = F.cross_entropy(piece_logits, tgt_piece.reshape(B * 8))
    loss_ori = F.cross_entropy(ori_logits, tgt_ori.reshape(B * 8))
    return loss_piece + loss_ori


def _validate_perm(p: List[int], *, size: int) -> None:
    if len(p) != size:
        raise ValueError(f"Bad perm length: expected {size}, got {len(p)}")
    if sorted(p) != list(range(size)):
        raise ValueError(f"Bad perm: not a bijection over [0..{size-1}] -> {p}")


def validate_move_tables() -> None:
    required = ["U", "R", "F"]
    for mv in required:
        if mv not in MOVE_TABLES:
            raise ValueError(f"Missing move table: {mv}")
        t = MOVE_TABLES[mv]
        if "cp" not in t or "co_delta" not in t:
            raise ValueError(f"Move table {mv} must define 'cp' and 'co_delta'")
        _validate_perm(list(t["cp"]), size=8)
        if len(list(t["co_delta"])) != 8:
            raise ValueError(f"Move table {mv} co_delta must have length 8")
        for d in list(t["co_delta"]):
            if int(d) not in (0, 1, 2):
                raise ValueError(f"Move table {mv} has invalid co_delta element: {d}")

    c = cube_solved()
    for mv in required:
        c = apply_move(c, mv)
        if sorted(c.cp) != list(range(8)):
            raise ValueError(f"After move {mv}, cp not a permutation: {c.cp}")
        for o in c.co:
            if int(o) not in (0, 1, 2):
                raise ValueError(f"After move {mv}, invalid orientation: {c.co}")
        if (sum(int(o) for o in c.co) % 3) != 0:
            raise ValueError(f"After move {mv}, orientation sum invariant violated: {c.co}")


def generate_episode(
    *,
    B: int,
    N: int,
    device: str,
    max_seq_len: int = 3,
    scramble_max_len: int = 12,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
    seq_len = random.randint(1, max_seq_len)
    seq = [random.choice(["U", "R", "F"]) for _ in range(seq_len)]

    support_states: List[Cube2x2] = [random_scramble_state(max_len=scramble_max_len) for _ in range(B * N)]
    support_next: List[Cube2x2] = [apply_moves(s, seq) for s in support_states]

    query_states: List[Cube2x2] = [random_scramble_state(max_len=scramble_max_len) for _ in range(B)]
    query_next: List[Cube2x2] = [apply_moves(s, seq) for s in query_states]

    x_s = batch_cube_to_onehot(support_states, device=device).view(B, N, -1)
    y_s = batch_cube_to_onehot(support_next, device=device).view(B, N, -1)

    support = torch.cat([x_s, y_s], dim=-1)
    query = batch_cube_to_onehot(query_states, device=device)
    target = batch_cube_to_onehot(query_next, device=device)

    return support, query, target, seq


def train_rubiks_demo(
    *,
    device: str,
    epochs: int = 1500,
    B: int = 32,
    N: int = 24,
    hidden: int = 128,
    z_dim: int = 64,
    lr: float = 8e-4,
    max_seq_len: int = 3,
    scramble_max_len: int = 12,
) -> None:
    x_dim = 2 * (8 * 11)
    q_dim = 8 * 11
    out_dim = 8 * 11

    model = AIN(x_dim=x_dim, z_dim=z_dim, query_dim=q_dim, out_dim=out_dim, hidden=hidden).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    crit = rubiks_ce_loss

    ema_loss = None

    for ep in range(1, epochs + 1):
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
        if ema_loss is None:
            ema_loss = l
        else:
            ema_loss = 0.97 * ema_loss + 0.03 * l

        if ep == 1 or ep % 100 == 0 or ep == epochs:
            with torch.no_grad():
                pacc, oacc = decode_sticker_accuracy(pred.detach(), target.detach())
                eacc = exact_state_accuracy(pred.detach(), target.detach())
            seq_str = "".join(seq)
            print(
                f"[EP {ep:04d}] loss={l:.6f} ema={float(ema_loss):.6f} seq={seq_str:<6s} piece_acc={pacc:.3f} ori_acc={oacc:.3f} exact={eacc:.3f}"
            )


def train_curriculum(
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
    eval_every: int = 50,
    eval_batches: int = 6,
):
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
        print("\n" + "=" * 70)
        print(
            f"  STAGE {stage_id}/3 — max_seq_len={max_seq_len} — "
            f"targets: piece>={threshold_piece:.2f} ori>={threshold_ori:.2f} (exact>={threshold_exact:.2f} optional)"
        )
        print(f"  scramble_max_len={int(scramble_max_len)}")
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
            if ema_loss is None:
                ema_loss = l
            else:
                ema_loss = 0.97 * float(ema_loss) + 0.03 * l

            if ep == 1 or ep % int(eval_every) == 0 or ep == int(epochs):
                ev_loss, ev_exact, ev_piece, ev_ori = _eval(max_seq_len=max_seq_len, scramble_max_len=int(scramble_max_len))
                seq_str = "".join(seq)
                print(
                    f"[S{stage_id} EP {ep:04d}] loss={l:.6f} ema={float(ema_loss):.6f} seq={seq_str:<8s} "
                    f"eval_loss={ev_loss:.6f} eval_exact={ev_exact:.3f} eval_piece={ev_piece:.3f} eval_ori={ev_ori:.3f}"
                )
                passed = (float(ev_piece) >= float(threshold_piece)) and (float(ev_ori) >= float(threshold_ori))
                if passed:
                    print(
                        f"  >> STAGE {stage_id} PASSED at ep={ep} "
                        f"(eval_piece={ev_piece:.3f} eval_ori={ev_ori:.3f} eval_exact={ev_exact:.3f})"
                    )
                    return True

        ev_loss, ev_exact, ev_piece, ev_ori = _eval(max_seq_len=max_seq_len, scramble_max_len=int(scramble_max_len))
        print(
            f"  !! STAGE {stage_id} NOT PASSED "
            f"(eval_piece={ev_piece:.3f} < {threshold_piece:.2f} or eval_ori={ev_ori:.3f} < {threshold_ori:.2f}; "
            f"eval_exact={ev_exact:.3f})"
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


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    validate_move_tables()

    max_seq_len = 3
    epochs = 1500
    B = 32
    N = 24
    hidden = 128
    z_dim = 64
    lr = 8e-4
    curriculum = True

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

    eval_every = 50
    eval_batches = 6
    for i, a in enumerate(sys.argv):
        if a == "--max_seq_len" and i + 1 < len(sys.argv):
            max_seq_len = int(sys.argv[i + 1])
        if a == "--epochs" and i + 1 < len(sys.argv):
            epochs = int(sys.argv[i + 1])
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
        if a == "--no_curriculum":
            curriculum = False
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
        if a == "--eval_every" and i + 1 < len(sys.argv):
            eval_every = int(sys.argv[i + 1])
        if a == "--eval_batches" and i + 1 < len(sys.argv):
            eval_batches = int(sys.argv[i + 1])

    print("=" * 70)
    print("  DEMO — AIN x Rubik 2x2 (group laws / permutation + orientation)")
    print("=" * 70)
    if bool(curriculum):
        train_curriculum(
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
            scramble1=int(scramble1),
            scramble2=int(scramble2),
            scramble3=int(scramble3),
            eval_every=eval_every,
            eval_batches=eval_batches,
        )
    else:
        train_rubiks_demo(
            device=device,
            epochs=epochs,
            B=B,
            N=N,
            hidden=hidden,
            z_dim=z_dim,
            lr=lr,
            max_seq_len=max_seq_len,
            scramble_max_len=12,
        )


if __name__ == "__main__":
    main()
