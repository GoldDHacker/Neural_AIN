import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


def compute_context_signature(support: torch.Tensor) -> torch.Tensor:
    x_exp = torch.cat([support, support ** 2], dim=-1)
    x_mean = x_exp.mean(dim=1)
    x_std = x_exp.std(dim=1, unbiased=False)

    if x_exp.shape[1] >= 2:
        x_delta = x_exp[:, 1:, :] - x_exp[:, :-1, :]
        delta_mean = x_delta.mean(dim=1)
        delta_std = x_delta.std(dim=1, unbiased=False)
    else:
        delta_mean = torch.zeros_like(x_mean)
        delta_std = torch.zeros_like(x_std)

    sig = torch.cat([x_mean, x_std, delta_mean, delta_std], dim=-1)
    return sig


def detach_to_cpu(x: torch.Tensor) -> torch.Tensor:
    return x.detach().to('cpu')


def detach_dict_to_cpu(d: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k, v in d.items():
        out[k] = detach_to_cpu(v)
    return out


@dataclass
class ProgramEntry:
    key: str
    created_ts: float
    last_used_ts: float
    uses: int

    signature: torch.Tensor
    z: torch.Tensor
    forged: Dict[str, torch.Tensor]

    replay_support: Optional[torch.Tensor] = None
    replay_queries: Optional[torch.Tensor] = None
    replay_targets: Optional[torch.Tensor] = None


class ProgramBank:
    def __init__(
        self,
        *,
        capacity: int = 1000,
        z_threshold: float = 0.92,
        signature_threshold: float = 0.65,
        enable_signature_fallback: bool = True,
        device: Optional[str] = None,
    ):
        self.capacity = int(capacity)
        self.z_threshold = float(z_threshold)
        self.signature_threshold = float(signature_threshold)
        self.enable_signature_fallback = bool(enable_signature_fallback)
        self.device = device

        self._entries: "OrderedDict[str, ProgramEntry]" = OrderedDict()
        self._counter = 0

    def __len__(self) -> int:
        return len(self._entries)

    def _make_key(self) -> str:
        self._counter += 1
        return f"prog_{self._counter:08d}"

    def _lru_touch(self, key: str):
        entry = self._entries[key]
        entry.last_used_ts = time.time()
        entry.uses += 1
        self._entries.move_to_end(key)

    def _evict_if_needed(self):
        while len(self._entries) > self.capacity:
            self._entries.popitem(last=False)

    def add(
        self,
        *,
        signature: torch.Tensor,
        z: torch.Tensor,
        forged: Dict[str, torch.Tensor],
        replay_support: Optional[torch.Tensor] = None,
        replay_queries: Optional[torch.Tensor] = None,
        replay_targets: Optional[torch.Tensor] = None,
    ) -> str:
        key = self._make_key()
        now = time.time()

        entry = ProgramEntry(
            key=key,
            created_ts=now,
            last_used_ts=now,
            uses=0,
            signature=detach_to_cpu(signature),
            z=detach_to_cpu(z),
            forged=detach_dict_to_cpu(forged),
            replay_support=detach_to_cpu(replay_support) if replay_support is not None else None,
            replay_queries=detach_to_cpu(replay_queries) if replay_queries is not None else None,
            replay_targets=detach_to_cpu(replay_targets) if replay_targets is not None else None,
        )

        self._entries[key] = entry
        self._lru_touch(key)
        self._evict_if_needed()
        return key

    def _cosine_score(self, z_new: torch.Tensor, z_bank: torch.Tensor) -> float:
        z_new = z_new.view(1, -1)
        z_bank = z_bank.view(1, -1)
        cos = torch.nn.CosineSimilarity(dim=-1)
        return float(cos(z_new, z_bank).item())

    def _signature_score(self, sig_new: torch.Tensor, sig_bank: torch.Tensor) -> float:
        sig_new = sig_new.view(1, -1)
        sig_bank = sig_bank.view(1, -1)

        # score dans [0,1] approx: 1/(1+||a-b||)
        dist = torch.norm(sig_new - sig_bank, p=2, dim=-1)
        return float((1.0 / (1.0 + dist)).item())

    def match(
        self,
        *,
        signature: torch.Tensor,
        z: torch.Tensor,
    ) -> Tuple[Optional[ProgramEntry], Dict[str, float]]:
        if len(self._entries) == 0:
            return None, {"best_z": 0.0, "best_sig": 0.0}

        sig_cpu = detach_to_cpu(signature)
        z_cpu = detach_to_cpu(z)

        best_key = None
        best_z = -1.0
        best_sig = -1.0

        for key, entry in self._entries.items():
            z_score = self._cosine_score(z_cpu, entry.z)
            if z_score > best_z:
                best_z = z_score
                best_key = key

        if best_key is not None and best_z >= self.z_threshold:
            self._lru_touch(best_key)
            return self._entries[best_key], {"best_z": best_z, "best_sig": 0.0}

        if self.enable_signature_fallback:
            for key, entry in self._entries.items():
                sig_score = self._signature_score(sig_cpu, entry.signature)
                if sig_score > best_sig:
                    best_sig = sig_score
                    best_key = key

            if best_key is not None and best_sig >= self.signature_threshold:
                self._lru_touch(best_key)
                return self._entries[best_key], {"best_z": best_z, "best_sig": best_sig}

        return None, {"best_z": best_z, "best_sig": best_sig}

    def match_topk(
        self,
        *,
        signature: torch.Tensor,
        z: torch.Tensor,
        k: int = 5,
    ) -> Tuple[List[ProgramEntry], Dict[str, float]]:
        if len(self._entries) == 0:
            return [], {"best_z": 0.0, "best_sig": 0.0, "margin_z": 0.0}

        sig_cpu = detach_to_cpu(signature)
        z_cpu = detach_to_cpu(z)

        scored: List[Tuple[float, float, str]] = []
        for key, entry in self._entries.items():
            z_score = self._cosine_score(z_cpu, entry.z)
            sig_score = self._signature_score(sig_cpu, entry.signature)
            scored.append((z_score, sig_score, key))

        scored.sort(key=lambda t: t[0], reverse=True)

        best_z = float(scored[0][0])
        margin_z = float(best_z - float(scored[1][0])) if len(scored) >= 2 else float(best_z)
        best_sig = float(max(t[1] for t in scored)) if len(scored) > 0 else 0.0

        out_entries: List[ProgramEntry] = []
        for i in range(min(int(k), len(scored))):
            key = scored[i][2]
            out_entries.append(self._entries[key])

        return out_entries, {"best_z": best_z, "best_sig": best_sig, "margin_z": margin_z}

    def get_forged_for(self, *, signature: torch.Tensor, z: torch.Tensor) -> Tuple[Optional[Dict[str, torch.Tensor]], Dict[str, float]]:
        entry, scores = self.match(signature=signature, z=z)
        if entry is None:
            return None, scores

        forged = entry.forged
        if self.device is not None:
            forged = {k: v.to(self.device) for k, v in forged.items()}
        return forged, scores

    def refresh(
        self,
        *,
        key: str,
        signature: torch.Tensor,
        z: torch.Tensor,
        forged: Dict[str, torch.Tensor],
        replay_support: Optional[torch.Tensor] = None,
        replay_queries: Optional[torch.Tensor] = None,
        replay_targets: Optional[torch.Tensor] = None,
    ):
        entry = self._entries[key]
        entry.signature = detach_to_cpu(signature)
        entry.z = detach_to_cpu(z)
        entry.forged = detach_dict_to_cpu(forged)
        if replay_support is not None:
            entry.replay_support = detach_to_cpu(replay_support)
        if replay_queries is not None:
            entry.replay_queries = detach_to_cpu(replay_queries)
        if replay_targets is not None:
            entry.replay_targets = detach_to_cpu(replay_targets)
        self._lru_touch(key)

    def sample_replay(self, *, batch_size: int) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        candidates = [e for e in self._entries.values() if e.replay_support is not None]
        if len(candidates) == 0:
            return None

        idx = torch.randint(low=0, high=len(candidates), size=(int(batch_size),))

        supports = []
        queries = []
        targets = []
        for i in idx.tolist():
            e = candidates[i]
            supports.append(e.replay_support)
            queries.append(e.replay_queries)
            targets.append(e.replay_targets)

        support = torch.cat(supports, dim=0)
        query = torch.cat(queries, dim=0)
        target = torch.cat(targets, dim=0)

        return support, query, target

    def save(self, path: str):
        entries = []
        for e in self._entries.values():
            entries.append(
                {
                    "key": e.key,
                    "created_ts": float(e.created_ts),
                    "last_used_ts": float(e.last_used_ts),
                    "uses": int(e.uses),
                    "signature": e.signature,
                    "z": e.z,
                    "forged": e.forged,
                    "replay_support": e.replay_support,
                    "replay_queries": e.replay_queries,
                    "replay_targets": e.replay_targets,
                }
            )

        payload = {
            "capacity": int(self.capacity),
            "z_threshold": float(self.z_threshold),
            "signature_threshold": float(self.signature_threshold),
            "enable_signature_fallback": bool(self.enable_signature_fallback),
            "counter": int(self._counter),
            "entries": entries,
        }
        torch.save(payload, path)

    @staticmethod
    def load(path: str, *, device: Optional[str] = None) -> "ProgramBank":
        try:
            payload = torch.load(path, map_location='cpu', weights_only=True)
        except TypeError:
            payload = torch.load(path, map_location='cpu')
        bank = ProgramBank(
            capacity=int(payload["capacity"]),
            z_threshold=float(payload["z_threshold"]),
            signature_threshold=float(payload["signature_threshold"]),
            enable_signature_fallback=bool(payload["enable_signature_fallback"]),
            device=device,
        )
        bank._counter = int(payload.get("counter", 0))
        entries = payload.get("entries", [])
        for d in entries:
            e = ProgramEntry(
                key=str(d["key"]),
                created_ts=float(d["created_ts"]),
                last_used_ts=float(d["last_used_ts"]),
                uses=int(d["uses"]),
                signature=d["signature"],
                z=d["z"],
                forged=d["forged"],
                replay_support=d.get("replay_support", None),
                replay_queries=d.get("replay_queries", None),
                replay_targets=d.get("replay_targets", None),
            )
            bank._entries[e.key] = e
        return bank

    def serialize(self) -> dict:
        """Retourne un dict serialisable contenant toute la banque (pour integration dans save_full)."""
        entries = []
        for e in self._entries.values():
            entries.append(
                {
                    "key": e.key,
                    "created_ts": float(e.created_ts),
                    "last_used_ts": float(e.last_used_ts),
                    "uses": int(e.uses),
                    "signature": e.signature,
                    "z": e.z,
                    "forged": e.forged,
                    "replay_support": e.replay_support,
                    "replay_queries": e.replay_queries,
                    "replay_targets": e.replay_targets,
                }
            )
        return {
            "capacity": int(self.capacity),
            "z_threshold": float(self.z_threshold),
            "signature_threshold": float(self.signature_threshold),
            "enable_signature_fallback": bool(self.enable_signature_fallback),
            "counter": int(self._counter),
            "entries": entries,
        }

    @staticmethod
    def deserialize(data: dict, *, device: Optional[str] = None) -> "ProgramBank":
        """Reconstruit un ProgramBank a partir d'un dict serialise (pour integration dans load_full)."""
        bank = ProgramBank(
            capacity=int(data["capacity"]),
            z_threshold=float(data["z_threshold"]),
            signature_threshold=float(data["signature_threshold"]),
            enable_signature_fallback=bool(data["enable_signature_fallback"]),
            device=device,
        )
        bank._counter = int(data.get("counter", 0))
        for d in data.get("entries", []):
            e = ProgramEntry(
                key=str(d["key"]),
                created_ts=float(d["created_ts"]),
                last_used_ts=float(d["last_used_ts"]),
                uses=int(d["uses"]),
                signature=d["signature"],
                z=d["z"],
                forged=d["forged"],
                replay_support=d.get("replay_support", None),
                replay_queries=d.get("replay_queries", None),
                replay_targets=d.get("replay_targets", None),
            )
            bank._entries[e.key] = e
        return bank


@dataclass
class BankDecision:
    action: str
    key: Optional[str] = None


@dataclass
class BankPolicyConfig:
    z_reuse_threshold: float = 0.92
    z_margin_threshold: float = 0.01
    signature_reuse_threshold: float = 0.65
    z_stability_threshold: float = 0.90

    cost_reuse: float = 0.05
    cost_recompile: float = 1.00
    cost_refresh: float = 1.00
    quality_vs_cost: float = 0.50

    enable_signature_fallback: bool = True
    enable_probe: bool = True

    probe_batch: int = 4
    probe_mse_refresh_threshold: float = 0.25


class BankPolicy:
    def __init__(self, *, config: Optional[BankPolicyConfig] = None):
        self.config = config if config is not None else BankPolicyConfig()

    def decide(
        self,
        *,
        scores: Dict[str, float],
        candidates: List[ProgramEntry],
        z_stability: Optional[float] = None,
        probe_mse: Optional[float] = None,
        costs: Optional[Dict[str, float]] = None,
    ) -> BankDecision:
        best_z = float(scores.get("best_z", 0.0))
        best_sig = float(scores.get("best_sig", 0.0))
        margin_z = float(scores.get("margin_z", 0.0))

        cost_reuse = float(self.config.cost_reuse)
        cost_recompile = float(self.config.cost_recompile)
        cost_refresh = float(self.config.cost_refresh)
        if costs is not None:
            cost_reuse = float(costs.get("reuse", cost_reuse))
            cost_recompile = float(costs.get("recompile", cost_recompile))
            cost_refresh = float(costs.get("refresh", cost_refresh))

        stable_ok = True
        if z_stability is not None:
            stable_ok = float(z_stability) >= float(self.config.z_stability_threshold)

        probe_refresh = False
        if self.config.enable_probe and probe_mse is not None:
            probe_refresh = float(probe_mse) > float(self.config.probe_mse_refresh_threshold)

        reuse_ok = (
            best_z >= float(self.config.z_reuse_threshold)
            and margin_z >= float(self.config.z_margin_threshold)
            and stable_ok
        )

        if (not reuse_ok) and self.config.enable_signature_fallback:
            reuse_ok = best_sig >= float(self.config.signature_reuse_threshold)

        if len(candidates) == 0:
            return BankDecision(action="RECOMPILE", key=None)

        # Arbitrage coût-vs-qualité:
        # - Si reuse_ok -> reuse (sauf probe_refresh)
        # - Si reuse_ok est faux mais très proche du seuil, on peut choisir REUSE si recompile est coûteux.
        if not reuse_ok:
            near_match = (best_z >= (float(self.config.z_reuse_threshold) - 0.01)) and stable_ok
            if near_match and cost_recompile > (1.0 + float(self.config.quality_vs_cost)) * cost_reuse:
                reuse_ok = True
            else:
                return BankDecision(action="RECOMPILE", key=None)

        key = candidates[0].key
        if probe_refresh:
            return BankDecision(action="REUSE_THEN_REFRESH", key=key)
        return BankDecision(action="REUSE", key=key)


@dataclass
class BankPolicyV2Config:
    epsilon: float = 0.10
    lr: float = 2e-3
    lambda_cost: float = 0.15

    adaptive_lambda: bool = True
    target_expected_cost: float = 0.25
    lambda_lr: float = 5e-3
    lambda_min: float = 0.0
    lambda_max: float = 5.0

    adaptive_quality: bool = True
    quality_delta: float = 0.02
    lambda_quality_lr: float = 5e-3
    lambda_quality_min: float = 0.0
    lambda_quality_max: float = 5.0

    adaptive_z: bool = True
    z_target_best: float = 0.90
    z_target_margin: float = 0.02
    z_target_stability: float = 0.85
    z_violation_target: float = 0.0
    z_violation_ema_beta: float = 0.95
    lambda_z_lr: float = 5e-3
    lambda_z_min: float = 0.0
    lambda_z_max: float = 5.0

    ema_beta: float = 0.98
    normalize_mse: bool = True

    buffer_capacity: int = 5000
    batch_size: int = 64
    warmup: int = 200
    update_every: int = 5
    updates_per_step: int = 2

    z_reuse_hard_floor: float = 0.75
    signature_reuse_hard_floor: float = 0.55


class BankPolicyV2:
    def __init__(self, *, config: Optional[BankPolicyV2Config] = None, device: Optional[str] = None):
        self.config = config if config is not None else BankPolicyV2Config()
        self.device = device

        self.model = torch.nn.Sequential(
            torch.nn.Linear(4, 32),
            torch.nn.Tanh(),
            torch.nn.Linear(32, 3),
        )
        if self.device is not None:
            self.model.to(self.device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=float(self.config.lr))
        self._steps = 0

        self._lambda_cost = float(self.config.lambda_cost)
        self._lambda_quality = 0.0
        self._lambda_z = 0.0
        self._ema_mse = 1.0
        self._ema_mse_baseline: Optional[float] = None

        self._last_quality_violation = 0.0
        self._last_z_violation = 0.0
        self._ema_z_violation = 0.0

        self._buffer_x: List[torch.Tensor] = []
        self._buffer_a: List[int] = []
        self._buffer_r: List[float] = []

    def _features_tensor(self, *, scores: Dict[str, float], z_stability: float) -> torch.Tensor:
        x = torch.tensor(
            [
                float(scores.get("best_z", 0.0)),
                float(scores.get("margin_z", 0.0)),
                float(scores.get("best_sig", 0.0)),
                float(z_stability),
            ],
            dtype=torch.float32,
        ).view(1, 4)
        if self.device is not None:
            x = x.to(self.device)
        return x

    def get_lambda_cost(self) -> float:
        return float(self._lambda_cost)

    def get_lambda_quality(self) -> float:
        return float(self._lambda_quality)

    def get_mse_baseline(self) -> float:
        if self._ema_mse_baseline is None:
            return -1.0
        return float(self._ema_mse_baseline)

    def get_lambda_z(self) -> float:
        return float(self._lambda_z)

    def get_last_z_violation(self) -> float:
        return float(self._last_z_violation)

    def get_z_violation_ema(self) -> float:
        return float(self._ema_z_violation)

    def get_last_quality_violation(self) -> float:
        return float(self._last_quality_violation)

    def _update_lambda(self, *, expected_cost: float):
        if not bool(self.config.adaptive_lambda):
            self._lambda_cost = float(self.config.lambda_cost)
            return

        # Dual ascent: pousse lambda vers le haut si le coût dépasse le budget.
        # Si le coût est sous le budget, lambda diminue.
        err = float(expected_cost) - float(self.config.target_expected_cost)
        self._lambda_cost = float(self._lambda_cost) + float(self.config.lambda_lr) * err
        self._lambda_cost = float(max(float(self.config.lambda_min), min(float(self.config.lambda_max), self._lambda_cost)))

    def _normalize_mse(self, mse: float) -> float:
        if not bool(self.config.normalize_mse):
            return float(mse)

        beta = float(self.config.ema_beta)
        self._ema_mse = beta * float(self._ema_mse) + (1.0 - beta) * float(mse)
        denom = float(self._ema_mse) + 1e-8
        return float(mse) / denom

    def _update_quality_baseline(self, *, mse: float):
        beta = float(self.config.ema_beta)
        if self._ema_mse_baseline is None:
            self._ema_mse_baseline = float(mse)
        else:
            self._ema_mse_baseline = beta * float(self._ema_mse_baseline) + (1.0 - beta) * float(mse)

    def _quality_violation(self, *, mse: float) -> float:
        if self._ema_mse_baseline is None:
            return 0.0
        # Violation si la qualité est significativement plus mauvaise que la baseline EMA.
        thresh = float(self._ema_mse_baseline) + float(self.config.quality_delta)
        return float(max(0.0, float(mse) - thresh))

    def _update_lambda_quality(self, *, violation: float):
        if not bool(self.config.adaptive_quality):
            return
        # Dual ascent vers violation ~ 0.
        self._lambda_quality = float(self._lambda_quality) + float(self.config.lambda_quality_lr) * float(violation)
        self._lambda_quality = float(max(float(self.config.lambda_quality_min), min(float(self.config.lambda_quality_max), self._lambda_quality)))

    def _z_violation(self, *, scores: Dict[str, float], z_stability: float) -> float:
        best_z = float(scores.get("best_z", 0.0))
        margin_z = float(scores.get("margin_z", 0.0))
        v_best = max(0.0, float(self.config.z_target_best) - best_z)
        v_margin = max(0.0, float(self.config.z_target_margin) - margin_z)
        v_stab = max(0.0, float(self.config.z_target_stability) - float(z_stability))
        return float(v_best + v_margin + v_stab)

    def _update_lambda_z(self, *, violation: float):
        if not bool(self.config.adaptive_z):
            return
        beta = float(self.config.z_violation_ema_beta)
        self._ema_z_violation = beta * float(self._ema_z_violation) + (1.0 - beta) * float(violation)

        # Dual ascent centrée sur une cible: lambda peut monter ET descendre.
        err = float(self._ema_z_violation) - float(self.config.z_violation_target)
        self._lambda_z = float(self._lambda_z) + float(self.config.lambda_z_lr) * float(err)
        self._lambda_z = float(max(float(self.config.lambda_z_min), min(float(self.config.lambda_z_max), self._lambda_z)))

    def _action_to_idx(self, action: str) -> Optional[int]:
        if action == "REUSE":
            return 0
        if action == "RECOMPILE":
            return 1
        if action == "REUSE_THEN_REFRESH":
            return 2
        return None

    def _trim_buffer(self):
        cap = int(self.config.buffer_capacity)
        if cap <= 0:
            self._buffer_x = []
            self._buffer_a = []
            self._buffer_r = []
            return
        while len(self._buffer_x) > cap:
            self._buffer_x.pop(0)
            self._buffer_a.pop(0)
            self._buffer_r.pop(0)

    def _update_minibatch(self):
        n = len(self._buffer_x)
        bs = int(self.config.batch_size)
        if n < max(2, bs):
            return

        idx = torch.randint(low=0, high=n, size=(bs,))
        xs = torch.cat([self._buffer_x[i] for i in idx.tolist()], dim=0)
        a = torch.tensor([self._buffer_a[i] for i in idx.tolist()], dtype=torch.long)
        r = torch.tensor([self._buffer_r[i] for i in idx.tolist()], dtype=torch.float32).view(-1, 1)

        if self.device is not None:
            xs = xs.to(self.device)
            a = a.to(self.device)
            r = r.to(self.device)

        q = self.model(xs)
        pred = q.gather(1, a.view(-1, 1))
        loss = F.mse_loss(pred, r)

        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
        self.opt.step()

    def decide(
        self,
        *,
        scores: Dict[str, float],
        candidates: List[ProgramEntry],
        z_stability: float,
        costs: Optional[Dict[str, float]] = None,
    ) -> BankDecision:
        if len(candidates) == 0:
            return BankDecision(action="RECOMPILE", key=None)

        best_z = float(scores.get("best_z", 0.0))
        best_sig = float(scores.get("best_sig", 0.0))

        # Garde-fous: éviter des reuse manifestement mauvais.
        if best_z < float(self.config.z_reuse_hard_floor) and best_sig < float(self.config.signature_reuse_hard_floor):
            return BankDecision(action="RECOMPILE", key=None)

        cost_reuse = 0.0
        cost_recompile = 0.0
        cost_refresh = 0.0
        if costs is not None:
            cost_reuse = float(costs.get("reuse", 0.0))
            cost_recompile = float(costs.get("recompile", 0.0))
            cost_refresh = float(costs.get("refresh", 0.0))

        x = self._features_tensor(scores=scores, z_stability=z_stability)
        z_v = self._z_violation(scores=scores, z_stability=float(z_stability))
        with torch.no_grad():
            q = self.model(x).view(-1)
            # Q-values bruts -> on soustrait un coût (penalty) pour approximer Q_net
            lam = float(self.get_lambda_cost())
            lam_z = float(self.get_lambda_z())
            q_reuse = float(q[0].item()) - lam * cost_reuse
            q_recompile = float(q[1].item()) - lam * cost_recompile
            q_refresh = float(q[2].item()) - lam * (cost_reuse + cost_refresh)

            q_reuse = float(q_reuse) - lam_z * float(z_v)
            q_refresh = float(q_refresh) - lam_z * float(z_v)

        self._steps += 1
        if float(torch.rand(()).item()) < float(self.config.epsilon):
            r = float(torch.rand(()).item())
            if r < (1.0 / 3.0):
                pick = "REUSE"
            elif r < (2.0 / 3.0):
                pick = "RECOMPILE"
            else:
                pick = "REUSE_THEN_REFRESH"
        else:
            vals = {
                "REUSE": q_reuse,
                "RECOMPILE": q_recompile,
                "REUSE_THEN_REFRESH": q_refresh,
            }
            pick = max(vals.items(), key=lambda kv: kv[1])[0]

        if pick == "REUSE":
            return BankDecision(action="REUSE", key=candidates[0].key)
        if pick == "REUSE_THEN_REFRESH":
            return BankDecision(action="REUSE_THEN_REFRESH", key=candidates[0].key)
        return BankDecision(action="RECOMPILE", key=None)

    def observe(
        self,
        *,
        scores: Dict[str, float],
        z_stability: float,
        action: str,
        reward: Optional[float] = None,
        mse: Optional[float] = None,
        expected_cost: Optional[float] = None,
    ):
        # Apprentissage supervisé bandit : on pousse Q(action) -> reward.
        if action not in ("REUSE", "RECOMPILE", "REUSE_THEN_REFRESH"):
            return

        if expected_cost is not None:
            self._update_lambda(expected_cost=float(expected_cost))

        z_v = self._z_violation(scores=scores, z_stability=float(z_stability))
        if action.startswith("REUSE"):
            self._update_lambda_z(violation=float(z_v))
        self._last_z_violation = float(z_v)

        if mse is not None:
            mse_n = self._normalize_mse(float(mse))
            if action == "RECOMPILE":
                self._update_quality_baseline(mse=float(mse))
            violation = self._quality_violation(mse=float(mse))
            self._update_lambda_quality(violation=float(violation))
            self._last_quality_violation = float(violation)

            # Reward interne: qualité normalisée + pénalité coût + pénalité violation qualité
            c = float(expected_cost) if expected_cost is not None else 0.0
            z_pen = float(self.get_lambda_z()) * float(z_v) if action.startswith("REUSE") else 0.0
            reward_val = -float(mse_n) - float(self.get_lambda_cost()) * c - float(self.get_lambda_quality()) * float(violation) - float(z_pen)
        else:
            reward_val = float(reward) if reward is not None else 0.0

        a_idx = self._action_to_idx(action)
        if a_idx is None:
            return

        x = self._features_tensor(scores=scores, z_stability=z_stability)
        self._buffer_x.append(x.detach().to("cpu"))
        self._buffer_a.append(int(a_idx))
        self._buffer_r.append(float(reward_val))
        self._trim_buffer()

        self._steps += 1

        # NOTE: observe peut être appelée depuis une boucle externe sous torch.no_grad().
        # On force ici l'activation des gradients pour le modèle de policy.
        with torch.enable_grad():
            if len(self._buffer_x) >= int(self.config.warmup) and (self._steps % int(self.config.update_every) == 0):
                for _ in range(int(self.config.updates_per_step)):
                    self._update_minibatch()


class ContextualAIN:
    def __init__(
        self,
        *,
        bank: ProgramBank,
        policy: Optional[BankPolicy] = None,
        infer_z: Callable[[torch.Tensor], torch.Tensor],
        compile_forged: Callable[[torch.Tensor], Dict[str, torch.Tensor]],
        execute: Callable[[torch.Tensor, Dict[str, torch.Tensor]], torch.Tensor],
        device: Optional[str] = None,
    ):
        self.bank = bank
        self.policy = policy if policy is not None else BankPolicy()
        self.infer_z = infer_z
        self.compile_forged = compile_forged
        self.execute = execute
        self.device = device

    def _z_stability(self, support: torch.Tensor) -> float:
        z1 = self.infer_z(support)
        noise = 0.02 * torch.randn_like(support)
        z2 = self.infer_z(support + noise)
        z1 = z1.view(1, -1)
        z2 = z2.view(1, -1)
        cos = torch.nn.CosineSimilarity(dim=-1)
        return float(cos(z1, z2).item())

    def _get_costs(self, costs: Optional[Dict[str, float]]) -> Tuple[float, float, float]:
        # Defaults: si policy v1 (BankPolicy), elle expose config.cost_*.
        cost_reuse = 0.0
        cost_recompile = 1.0
        cost_refresh = 1.0

        if hasattr(self.policy, "config"):
            cfg = getattr(self.policy, "config")
            if hasattr(cfg, "cost_reuse"):
                cost_reuse = float(getattr(cfg, "cost_reuse"))
            if hasattr(cfg, "cost_recompile"):
                cost_recompile = float(getattr(cfg, "cost_recompile"))
            if hasattr(cfg, "cost_refresh"):
                cost_refresh = float(getattr(cfg, "cost_refresh"))

        if costs is not None:
            cost_reuse = float(costs.get("reuse", cost_reuse))
            cost_recompile = float(costs.get("recompile", cost_recompile))
            cost_refresh = float(costs.get("refresh", cost_refresh))

        return cost_reuse, cost_recompile, cost_refresh

    def _lambda_cost(self) -> float:
        if hasattr(self.policy, "get_lambda_cost"):
            try:
                return float(self.policy.get_lambda_cost())
            except TypeError:
                pass
        if hasattr(self.policy, "config") and hasattr(getattr(self.policy, "config"), "lambda_cost"):
            return float(getattr(getattr(self.policy, "config"), "lambda_cost"))
        return 0.0

    def _mse_baseline(self) -> float:
        if hasattr(self.policy, "get_mse_baseline"):
            try:
                return float(self.policy.get_mse_baseline())
            except TypeError:
                pass
        return -1.0

    def _quality_violation(self) -> float:
        if hasattr(self.policy, "get_last_quality_violation"):
            try:
                return float(self.policy.get_last_quality_violation())
            except TypeError:
                pass
        return -1.0

    def _z_violation(self) -> float:
        if hasattr(self.policy, "get_last_z_violation"):
            try:
                return float(self.policy.get_last_z_violation())
            except TypeError:
                pass
        return -1.0

    def _z_violation_ema(self) -> float:
        if hasattr(self.policy, "get_z_violation_ema"):
            try:
                return float(self.policy.get_z_violation_ema())
            except TypeError:
                pass
        return -1.0

    def _lambda_quality(self) -> float:
        if hasattr(self.policy, "get_lambda_quality"):
            try:
                return float(self.policy.get_lambda_quality())
            except TypeError:
                pass
        return 0.0

    def _lambda_z(self) -> float:
        if hasattr(self.policy, "get_lambda_z"):
            try:
                return float(self.policy.get_lambda_z())
            except TypeError:
                pass
        return 0.0

    def run(
        self,
        *,
        support: torch.Tensor,
        query: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        topk: int = 5,
        costs: Optional[Dict[str, float]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if self.device is not None:
            support = support.to(self.device)
            query = query.to(self.device)
            if target is not None:
                target = target.to(self.device)

        signature = compute_context_signature(support)
        z = self.infer_z(support)

        candidates, scores = self.bank.match_topk(signature=signature[0:1], z=z[0:1], k=int(topk))
        z_stability = self._z_stability(support[0:1])

        probe_mse = None
        enable_probe = False
        if hasattr(self.policy, "config"):
            enable_probe = bool(getattr(getattr(self.policy, "config"), "enable_probe", False))

        if enable_probe and target is not None and len(candidates) > 0:
            with torch.no_grad():
                key0 = candidates[0].key
                forged0 = self.bank._entries[key0].forged
                if self.device is not None:
                    forged0 = {k: v.to(self.device) for k, v in forged0.items()}
                forged0_b = {k: v.expand(query.shape[0], *v.shape[1:]) if v.shape[0] == 1 else v for k, v in forged0.items()}
                pred0 = self.execute(query, forged0_b)
                probe_mse = float(torch.mean((pred0 - target) ** 2).item())

        if hasattr(self.policy, "decide"):
            try:
                decision = self.policy.decide(
                    scores=scores,
                    candidates=candidates,
                    z_stability=z_stability,
                    probe_mse=probe_mse,
                    costs=costs,
                )
            except TypeError:
                decision = self.policy.decide(
                    scores=scores,
                    candidates=candidates,
                    z_stability=z_stability,
                    costs=costs,
                )
        else:
            decision = BankDecision(action="RECOMPILE", key=None)

        cost_reuse, cost_recompile, cost_refresh = self._get_costs(costs)

        expected_cost = cost_reuse if decision.action.startswith("REUSE") else cost_recompile
        if decision.action == "REUSE_THEN_REFRESH":
            expected_cost = cost_reuse + cost_refresh

        logs: Dict[str, float] = {
            "best_z": float(scores.get("best_z", 0.0)),
            "best_sig": float(scores.get("best_sig", 0.0)),
            "margin_z": float(scores.get("margin_z", 0.0)),
            "z_stability": float(z_stability),
            "probe_mse": float(probe_mse) if probe_mse is not None else -1.0,
            "action": 0.0,
            "reused": 1.0 if decision.action.startswith("REUSE") else 0.0,
            "recompiled": 1.0 if decision.action == "RECOMPILE" else 0.0,
            "refreshed": 1.0 if decision.action == "REUSE_THEN_REFRESH" else 0.0,
            "cost_reuse": float(cost_reuse),
            "cost_recompile": float(cost_recompile),
            "cost_refresh": float(cost_refresh),
            "expected_cost": float(expected_cost),
            "reward": -1.0,
            "lambda_cost": float(self._lambda_cost()),
            "lambda_quality": float(self._lambda_quality()),
            "lambda_z": float(self._lambda_z()),
            "mse_baseline": float(self._mse_baseline()),
            "quality_violation": float(self._quality_violation()),
            "z_violation": float(self._z_violation()),
            "z_violation_ema": float(self._z_violation_ema()),
        }

        if decision.action == "RECOMPILE":
            forged = self.compile_forged(z)
            pred = self.execute(query, forged)
            self.bank.add(
                signature=signature[0:1],
                z=z[0:1],
                forged={k: v[0:1] for k, v in forged.items()},
                replay_support=support[0:1],
                replay_queries=query[0:1],
                replay_targets=target[0:1] if target is not None else None,
            )
            if target is not None and hasattr(self.policy, "observe"):
                mse = float(torch.mean((pred - target) ** 2).item())
                try:
                    self.policy.observe(
                        scores=scores,
                        z_stability=z_stability,
                        action="RECOMPILE",
                        mse=mse,
                        expected_cost=float(expected_cost),
                    )
                except TypeError:
                    reward = -float(mse) - float(self._lambda_cost()) * float(expected_cost)
                    try:
                        self.policy.observe(scores=scores, z_stability=z_stability, action="RECOMPILE", reward=reward)
                    except TypeError:
                        pass
                    logs["reward"] = float(reward)
                else:
                    # reward implicite (normalisée et lambda dynamique) non exposée ici.
                    reward = -float(mse) - float(self._lambda_cost()) * float(expected_cost)
                    logs["reward"] = float(reward)

                logs["lambda_cost"] = float(self._lambda_cost())
                logs["lambda_quality"] = float(self._lambda_quality())
                logs["lambda_z"] = float(self._lambda_z())
                logs["mse_baseline"] = float(self._mse_baseline())
                logs["quality_violation"] = float(self._quality_violation())
                logs["z_violation"] = float(self._z_violation())
                logs["z_violation_ema"] = float(self._z_violation_ema())
            return pred, logs

        key = decision.key
        entry = self.bank._entries[key]
        forged_cached = entry.forged
        if self.device is not None:
            forged_cached = {k: v.to(self.device) for k, v in forged_cached.items()}
        forged_cached_b = {k: v.expand(query.shape[0], *v.shape[1:]) if v.shape[0] == 1 else v for k, v in forged_cached.items()}
        pred_cached = self.execute(query, forged_cached_b)

        if decision.action == "REUSE_THEN_REFRESH":
            forged_new = self.compile_forged(z)
            pred_new = self.execute(query, forged_new)
            self.bank.refresh(
                key=key,
                signature=signature[0:1],
                z=z[0:1],
                forged={k: v[0:1] for k, v in forged_new.items()},
                replay_support=support[0:1],
                replay_queries=query[0:1],
                replay_targets=target[0:1] if target is not None else None,
            )
            if target is not None and hasattr(self.policy, "observe"):
                mse = float(torch.mean((pred_new - target) ** 2).item())
                try:
                    self.policy.observe(
                        scores=scores,
                        z_stability=z_stability,
                        action="REUSE_THEN_REFRESH",
                        mse=mse,
                        expected_cost=float(expected_cost),
                    )
                except TypeError:
                    reward = -float(mse) - float(self._lambda_cost()) * float(expected_cost)
                    try:
                        self.policy.observe(scores=scores, z_stability=z_stability, action="REUSE_THEN_REFRESH", reward=reward)
                    except TypeError:
                        pass
                    logs["reward"] = float(reward)
                else:
                    reward = -float(mse) - float(self._lambda_cost()) * float(expected_cost)
                    logs["reward"] = float(reward)

                logs["lambda_cost"] = float(self._lambda_cost())
                logs["lambda_quality"] = float(self._lambda_quality())
                logs["lambda_z"] = float(self._lambda_z())
                logs["mse_baseline"] = float(self._mse_baseline())
                logs["quality_violation"] = float(self._quality_violation())
                logs["z_violation"] = float(self._z_violation())
                logs["z_violation_ema"] = float(self._z_violation_ema())
            return pred_new, logs

        if target is not None and hasattr(self.policy, "observe"):
            mse = float(torch.mean((pred_cached - target) ** 2).item())
            try:
                self.policy.observe(
                    scores=scores,
                    z_stability=z_stability,
                    action="REUSE",
                    mse=mse,
                    expected_cost=float(expected_cost),
                )
            except TypeError:
                reward = -float(mse) - float(self._lambda_cost()) * float(expected_cost)
                try:
                    self.policy.observe(scores=scores, z_stability=z_stability, action="REUSE", reward=reward)
                except TypeError:
                    pass
                logs["reward"] = float(reward)
            else:
                reward = -float(mse) - float(self._lambda_cost()) * float(expected_cost)
                logs["reward"] = float(reward)

            logs["lambda_cost"] = float(self._lambda_cost())
            logs["lambda_quality"] = float(self._lambda_quality())
            logs["mse_baseline"] = float(self._mse_baseline())
            logs["quality_violation"] = float(self._quality_violation())

        return pred_cached, logs
