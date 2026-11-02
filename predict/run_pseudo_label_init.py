from __future__ import annotations

from omegaconf import OmegaConf

from spatialsam.data.io import build_pseudo_label_samples
from spatialsam.configuration import register_omegaconf_resolvers

register_omegaconf_resolvers()


def run(cfg) -> None:
    from predict.pseudo_label_orchestrator import PseudoLabelInitializationOrchestrator

    register_omegaconf_resolvers()
    OmegaConf.resolve(cfg)
    task_cfg = cfg.task

    samples = build_pseudo_label_samples(task_cfg)
    if not samples:
        raise ValueError("No bootstrap samples; check image volumes and annotations")

    orchestrator = PseudoLabelInitializationOrchestrator(task_cfg)
    orchestrator.run_all(samples)
