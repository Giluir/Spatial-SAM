from __future__ import annotations

from omegaconf import OmegaConf

from spatialsam.data.io import build_inference_samples
from spatialsam.configuration import register_omegaconf_resolvers


register_omegaconf_resolvers()


def run(cfg) -> None:
    from predict.inference_orchestrator import InferenceOrchestrator

    register_omegaconf_resolvers()
    OmegaConf.resolve(cfg)
    task_cfg = cfg.task

    samples = build_inference_samples(task_cfg)
    if not samples:
        raise ValueError("No inference samples; check the image and output directories")

    orchestrator = InferenceOrchestrator(task_cfg)
    orchestrator.run_all(samples)
