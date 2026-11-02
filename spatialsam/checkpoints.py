"""Strict model initialization from current and historical checkpoints."""

from collections.abc import Mapping
from pathlib import Path

import torch
from torch import nn


def _model_state(payload: Mapping) -> Mapping[str, torch.Tensor]:
    """Extract model weights without treating optimizer state as initialization."""
    for key in ("model", "model_state_dict"):
        if key in payload:
            state = payload[key]
            break
    else:
        state = payload
    if not isinstance(state, Mapping) or not state:
        raise ValueError(
            "Checkpoint does not contain a nonempty model state dictionary"
        )
    if not all(
        isinstance(key, str) and isinstance(value, torch.Tensor)
        for key, value in state.items()
    ):
        raise ValueError("Model state must map parameter names to tensors")
    return state


def _remove_wrapper_prefixes(
    state: Mapping[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """Remove uniform training wrapper prefixes without rewriting layer names."""
    state = dict(state)
    while True:
        prefix = next(
            (
                prefix
                for prefix in ("module.", "net.")
                if all(key.startswith(prefix) for key in state)
            ),
            None,
        )
        if prefix is None:
            return state
        state = {key[len(prefix) :]: value for key, value in state.items()}


def load_model_weights(
    model: nn.Module, checkpoint_path: str | Path | None
) -> nn.Module:
    """Initialize a model, rejecting incompatible architectures before mutation.

    Args:
        model: Model whose architecture must match all checkpoint parameters.
        checkpoint_path: Trusted local checkpoint, or None to keep initialization.

    Returns:
        The initialized model. Optimizer and scheduler state are not restored.

    Raises:
        ValueError: The checkpoint layout, parameter names, or shapes differ.
    """
    if checkpoint_path is None:
        return model
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("Checkpoint must contain a mapping")
    state = _remove_wrapper_prefixes(_model_state(payload))
    expected = model.state_dict()
    missing = sorted(expected.keys() - state.keys())
    unexpected = sorted(state.keys() - expected.keys())
    mismatched = [
        key
        for key in expected.keys() & state.keys()
        if expected[key].shape != state[key].shape
    ]
    if missing or unexpected or mismatched:
        raise ValueError(
            f"Incompatible model checkpoint {checkpoint_path}: "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}, "
            f"shape_mismatches={sorted(mismatched)[:5]}. "
            "The earlier refactored UNet has a different architecture; its weights "
            "cannot be converted by renaming parameters."
        )
    model.load_state_dict(state, strict=True)
    return model
