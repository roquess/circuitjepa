"""CircuitJEPA training loop with ablation variants."""

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch

from circuitjepa.ansatz import N_ENC_PARAMS, N_PRED_PARAMS
from circuitjepa.data import make_dataset
from circuitjepa.model import embed
from circuitjepa.sim import cosine_torch

Variant = Literal["full", "no_ema", "no_predictor", "untrained"]


@dataclass
class TrainConfig:
    variant: Variant
    seed: int
    n_train: int = 256
    n_test: int = 64
    train_data_seed: int = 1234
    test_data_seed: int = 5678
    delta: float = 0.25
    t0_max: int = 3
    epochs: int = 150
    batch_size: int = 32
    lr: float = 0.02
    tau: float = 0.99


@dataclass
class RunResult:
    config: TrainConfig
    theta: list[float]
    theta_bar: list[float]
    phi: list[float]
    history: list[float]
    wall_seconds: float


def loss_fn(pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    return 1 - cosine_torch(pred, tgt).mean()


def ema_update(theta_bar: torch.Tensor, theta: torch.Tensor, tau: float) -> torch.Tensor:
    return tau * theta_bar + (1 - tau) * theta


def train(cfg: TrainConfig) -> RunResult:
    t_start = time.perf_counter()
    gen = torch.Generator().manual_seed(cfg.seed)
    two_pi = 2 * math.pi
    theta = (torch.rand(N_ENC_PARAMS, generator=gen, dtype=torch.float64) * two_pi).requires_grad_()
    use_pred = cfg.variant != "no_predictor"
    n_phi = N_PRED_PARAMS if use_pred else 0
    phi = (torch.rand(n_phi, generator=gen, dtype=torch.float64) * two_pi).requires_grad_()
    theta_bar = theta.detach().clone()
    history: list[float] = []

    if cfg.variant != "untrained":
        data = make_dataset(cfg.n_train, cfg.train_data_seed, cfg.t0_max)
        params = [theta, phi] if use_pred else [theta]
        opt = torch.optim.Adam(params, lr=cfg.lr)
        phi_arg = phi if use_pred else None
        for _ in range(cfg.epochs):
            perm = torch.randperm(len(data), generator=gen).tolist()
            total, count = 0.0, 0
            for i in range(0, len(perm), cfg.batch_size):
                batch = [data[j] for j in perm[i : i + cfg.batch_size]]
                pred = embed(batch, "context", theta, phi_arg, cfg.delta, use_pred)
                if cfg.variant == "no_ema":
                    tgt = embed(batch, "target", theta, None, cfg.delta)
                else:
                    with torch.no_grad():
                        tgt = embed(batch, "target", theta_bar, None, cfg.delta)
                loss = loss_fn(pred, tgt)
                opt.zero_grad()
                loss.backward()
                opt.step()
                if cfg.variant != "no_ema":
                    theta_bar = ema_update(theta_bar, theta.detach(), cfg.tau)
                total += loss.item() * len(batch)
                count += len(batch)
            history.append(total / count)

    th = theta.detach()
    tb = th.clone() if cfg.variant == "no_ema" else theta_bar
    return RunResult(
        config=cfg,
        theta=th.tolist(),
        theta_bar=tb.tolist(),
        phi=phi.detach().tolist(),
        history=history,
        wall_seconds=time.perf_counter() - t_start,
    )


def save_run(r: RunResult, path: Path) -> None:
    Path(path).write_text(json.dumps(asdict(r)))


def load_run(path: Path) -> RunResult:
    d = json.loads(Path(path).read_text())
    d["config"] = TrainConfig(**d["config"])
    return RunResult(**d)
