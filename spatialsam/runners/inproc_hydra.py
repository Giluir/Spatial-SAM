from __future__ import annotations

from pathlib import Path
from contextlib import contextmanager
from typing import List, Optional

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf, open_dict
from spatialsam.configuration import register_omegaconf_resolvers


@contextmanager
def local_config_context(config_dir: Path):
    """Isolate composition from SAM2's package-level Hydra initialization."""
    global_hydra = GlobalHydra.instance()
    previous = global_hydra.hydra
    global_hydra.clear()
    try:
        with initialize_config_dir(
            config_dir=str(config_dir.resolve()), version_base=None
        ):
            yield
    finally:
        global_hydra.hydra = previous


def compose_main_config(
    project_root: Path,
    task_name: str,
    overrides: Optional[List[str]] = None,
) -> DictConfig:
    register_omegaconf_resolvers()
    project_root = project_root.resolve()
    all_overrides = [
        f'root="{project_root}"',
        f"+task={task_name}",
    ]
    if overrides:
        all_overrides.extend(overrides)

    with local_config_context(project_root / "config"):
        cfg = compose(config_name="config", overrides=all_overrides)
    if "shared" in cfg.task:
        with open_dict(cfg):
            cfg.shared = OmegaConf.create(
                OmegaConf.to_container(cfg.task.shared, resolve=False)
            )
    return cfg
