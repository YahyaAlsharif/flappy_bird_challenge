"""Shared Dueling DQN network and checkpoint I/O for training and inference.

Both the training notebook and the standalone inference viewers import this
module so the network architecture and checkpoint schema are defined exactly
once instead of duplicated between training and playback code.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

EXPECTED_ALGORITHM = "Dueling Double DQN with Prioritized Experience Replay"


class CheckpointError(RuntimeError):
    """Raised when a checkpoint is missing, incompatible, or corrupt."""


class DuelingQNetwork(nn.Module):
    """Two-stream Dueling DQN: shared trunk, then value and advantage heads."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 192) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.features = nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(self.hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(self.hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_dim),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        features = self.features(states)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        return values + advantages - advantages.mean(dim=1, keepdim=True)

    @torch.no_grad()
    def choose_actions(
        self,
        states,
        epsilon: float = 0.0,
        exploration_flap_probability: float = 0.08,
    ) -> np.ndarray:
        state_tensor = torch.as_tensor(
            np.asarray(states, dtype=np.float32),
            device=next(self.parameters()).device,
        )
        greedy_actions = self(state_tensor).argmax(dim=1).cpu().numpy()
        if epsilon <= 0:
            return greedy_actions.astype(np.int64)
        explore_mask = np.random.random(len(greedy_actions)) < epsilon
        exploratory_actions = (
            np.random.random(len(greedy_actions)) < exploration_flap_probability
        ).astype(np.int64)
        return np.where(explore_mask, exploratory_actions, greedy_actions).astype(np.int64)


def soft_update(online: nn.Module, target: nn.Module, tau: float) -> None:
    """Polyak-average the target network towards the online network in place."""

    with torch.no_grad():
        for target_param, online_param in zip(target.parameters(), online.parameters()):
            target_param.data.mul_(1.0 - tau).add_(online_param.data, alpha=tau)


def atomic_torch_save(payload: dict, path: os.PathLike | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_torch_payload(path: os.PathLike | str, device="cpu", weights_only: bool = False):
    path = Path(path)
    if not path.is_file():
        raise CheckpointError(f"Checkpoint file not found: {path}")
    try:
        return torch.load(path, map_location=device, weights_only=weights_only)
    except TypeError:
        return torch.load(path, map_location=device)
    except Exception as exc:
        raise CheckpointError(
            f"Could not load checkpoint {path}. It may be corrupt or an "
            f"unsupported checkpoint type: {type(exc).__name__}: {exc}"
        ) from exc


def _required_positive_int(config: Mapping, key: str, checkpoint_path) -> int:
    if key not in config:
        raise CheckpointError(
            f"Checkpoint configuration is missing required key {key!r}: {checkpoint_path}"
        )
    try:
        value = int(config[key])
    except (TypeError, ValueError) as exc:
        raise CheckpointError(
            f"Checkpoint configuration key {key!r} must be an integer: {checkpoint_path}"
        ) from exc
    if value <= 0:
        raise CheckpointError(
            f"Checkpoint configuration key {key!r} must be positive, got {value}: {checkpoint_path}"
        )
    return value


def build_model_config(
    state_dim: int,
    action_dim: int,
    hidden_dim: int,
    pipe_gap: int,
    extra: Mapping | None = None,
) -> dict:
    config = {
        "state_dim": int(state_dim),
        "action_dim": int(action_dim),
        "hidden_dim": int(hidden_dim),
        "pipe_gap": int(pipe_gap),
    }
    if extra:
        config.update(dict(extra))
    return config


def validate_checkpoint_metadata(checkpoint: Mapping, path) -> dict:
    """Validate a checkpoint payload and return its model_config mapping."""

    if not isinstance(checkpoint, Mapping):
        raise CheckpointError(
            f"Expected checkpoint {path} to contain a mapping, got {type(checkpoint).__name__}."
        )
    algorithm = checkpoint.get("algorithm")
    if algorithm != EXPECTED_ALGORITHM:
        raise CheckpointError(
            f"Checkpoint algorithm mismatch in {path}. Expected {EXPECTED_ALGORITHM!r}, "
            f"got {algorithm!r}."
        )
    config = checkpoint.get("model_config", checkpoint.get("config"))
    if not isinstance(config, Mapping):
        raise CheckpointError(
            f"Checkpoint {path} does not contain a valid 'model_config' mapping."
        )
    for key in ("state_dim", "action_dim", "hidden_dim", "pipe_gap"):
        _required_positive_int(config, key, path)
    if "online_state_dict" not in checkpoint or not isinstance(
        checkpoint["online_state_dict"], Mapping
    ):
        raise CheckpointError(f"Checkpoint {path} is missing a valid 'online_state_dict'.")
    return config


def load_model_from_checkpoint(
    path: os.PathLike | str,
    device: str | torch.device = "cpu",
) -> tuple[DuelingQNetwork, dict]:
    """Load an inference-ready DuelingQNetwork plus the raw checkpoint payload."""

    path = Path(path).expanduser().resolve()
    checkpoint = load_torch_payload(path, device=device, weights_only=False)
    config = validate_checkpoint_metadata(checkpoint, path)

    network = DuelingQNetwork(
        state_dim=int(config["state_dim"]),
        action_dim=int(config["action_dim"]),
        hidden_dim=int(config["hidden_dim"]),
    ).to(device)
    try:
        network.load_state_dict(checkpoint["online_state_dict"], strict=True)
    except RuntimeError as exc:
        raise CheckpointError(
            f"Checkpoint/model shape or key mismatch in {path}; strict loading failed: {exc}"
        ) from exc
    network.eval()
    network.requires_grad_(False)
    return network, checkpoint
