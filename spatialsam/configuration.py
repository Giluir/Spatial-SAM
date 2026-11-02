"""Shared configuration resolvers without training or device initialization."""

import math
from pathlib import Path

import hydra
from omegaconf import OmegaConf


def validate_output_directory(output, *inputs):
    """Keep generated artifacts separate from read-only dataset directories."""
    destination = Path(str(output)).expanduser().resolve()
    for value in inputs:
        if not value:
            continue
        source = Path(str(value)).expanduser().resolve()
        if destination.is_relative_to(source) or source.is_relative_to(destination):
            raise ValueError(
                f"Output and input data directories must be separate: {destination} / {source}"
            )


def register_omegaconf_resolvers():
    """Register idempotently so each stage can compose its recipe independently."""
    resolvers = {
        "get_method": hydra.utils.get_method,
        "get_class": hydra.utils.get_class,
        "add": lambda x, y: x + y,
        "times": lambda *values: math.prod(values),
        "divide": lambda x, y: x / y,
        "pow": lambda x, y: x**y,
        "subtract": lambda x, y: x - y,
        "range": lambda value: list(range(value)),
        "int": int,
        "ceil_int": lambda value: int(math.ceil(value)),
        "merge": lambda *values: OmegaConf.merge(*values),
    }
    for name, resolver in resolvers.items():
        if not OmegaConf.has_resolver(name):
            OmegaConf.register_new_resolver(name, resolver)
