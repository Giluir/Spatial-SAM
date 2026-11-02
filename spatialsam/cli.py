"""One command line for the existing training and prediction implementations."""

import argparse
import json
from pathlib import Path

from omegaconf import OmegaConf

from spatialsam.runners.inproc_hydra import compose_main_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_parser():
    parser = argparse.ArgumentParser(description="Spatial-SAM 3D volume segmentation")
    commands = parser.add_subparsers(dest="command", required=True)
    for command, task in (
        ("train", "train"),
        ("predict", "predict"),
        ("bootstrap", "bootstrap"),
    ):
        descriptions = {
            "train": "Bootstrap pseudo-labels and run all SDF/SAM training rounds",
            "predict": "Segment image volumes with trained checkpoints",
            "bootstrap": "Generate initial pseudo-labels only; use train for the complete pipeline",
        }
        sub = commands.add_parser(
            command, help=descriptions[command], description=descriptions[command]
        )
        sub.add_argument("--task", default=task, help="Task recipe in config/task")
        sub.add_argument("--images", type=Path)
        sub.add_argument("--output", type=Path)
        sub.add_argument("--sam-checkpoint", type=Path)
        sub.add_argument("--mean", type=float)
        sub.add_argument("--std", type=float)
        sub.add_argument(
            "--dry-run",
            action="store_true",
            help="Validate inputs and print configuration without loading models",
        )
        sub.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
        if command in ("train", "bootstrap"):
            supervision = sub.add_mutually_exclusive_group()
            supervision.add_argument(
                "--labels",
                type=Path,
                help="Complete ground-truth volumes for simulated sparse supervision",
            )
            supervision.add_argument(
                "--annotations",
                type=Path,
                help="JSON manifest of complete 2D slice masks; all listed slices are used",
            )
        if command == "train":
            sub.add_argument("--rounds", type=int)
            sub.add_argument(
                "--seed",
                type=int,
                help="Base random seed for both models (default: 123)",
            )
            sub.add_argument("--gpu", help="Single GPU index")
        else:
            sub.add_argument("--device", choices=("cpu", "cuda"))
        if command == "predict":
            sub.add_argument("--sdf-checkpoint", type=Path)
            sub.add_argument("--output-mode", choices=("semantic", "instance"))
    return parser


def configure(args):
    cfg = compose_main_config(PROJECT_ROOT, args.task, args.set)
    training = args.command == "train"
    paths = "task.shared.paths." if training else "task."
    runtime = "task.shared.runtime." if training else "task."
    fields = {
        "images": paths + "volume_dir",
        "labels": paths + "true_dir",
        "output": paths + "output_root",
        "sam_checkpoint": paths + "sam2_model_path",
        "sdf_checkpoint": "task.sdf_model_path",
        "rounds": "task.num_rounds",
        "mean": runtime + "mean",
        "std": runtime + "std",
        "gpu": "task.cuda_devices",
        "device": "task.device",
        "annotations": paths + "annotations",
    }
    if (args.mean is None) != (args.std is None):
        raise ValueError("Set --mean and --std together using your dataset statistics")
    for argument, field in fields.items():
        value = getattr(args, argument, None)
        if value is not None:
            if isinstance(value, Path):
                value = str(value.expanduser().resolve())
            OmegaConf.update(cfg, field, value, force_add=True)
    if args.command in ("train", "bootstrap"):
        if args.annotations is not None:
            OmegaConf.update(cfg, paths + "true_dir", None, force_add=True)
        elif args.labels is not None and OmegaConf.select(cfg, paths + "annotations"):
            OmegaConf.update(cfg, paths + "annotations", None, force_add=True)
        if args.command == "bootstrap" and args.output is not None:
            cfg.task.output_root = str(args.output.expanduser().resolve() / "sdf")
    if training:
        if args.seed is not None:
            for stage in ("sdf_train", "sam_train"):
                OmegaConf.update(
                    cfg,
                    f"task.pipeline.stages.{stage}.trainer.seed_value",
                    args.seed,
                    force_add=True,
                )
        # Stage recipes refer to the shared copy populated during composition.
        cfg.shared = OmegaConf.create(
            OmegaConf.to_container(cfg.task.shared, resolve=False)
        )
    elif args.command == "predict" and args.output_mode is not None:
        OmegaConf.update(
            cfg, "task.to_instance", args.output_mode == "instance", force_add=True
        )
    required = {
        "--images": paths + "volume_dir",
        "--output": paths + "output_root",
        "--mean": runtime + "mean",
        "--std": runtime + "std",
    }
    if args.command != "predict" or cfg.task.get("mode") != "sdf_only":
        required["--sam-checkpoint"] = paths + "sam2_model_path"
    if args.command == "predict":
        required["--sdf-checkpoint"] = "task.sdf_model_path"
    missing = [
        flag for flag, key in required.items() if OmegaConf.select(cfg, key) is None
    ]
    if missing:
        raise ValueError(
            "Provide " + ", ".join(missing) + " via CLI or task configuration"
        )
    if args.command in ("train", "bootstrap"):
        if not OmegaConf.select(cfg, paths + "true_dir") and not OmegaConf.select(
            cfg, paths + "annotations"
        ):
            raise ValueError(
                "Provide --annotations for labeled slices or --labels for complete volumes"
            )
    return cfg


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        cfg = configure(args)
        if args.command == "train":
            from spatialsam.pipeline import build_stages, run_pipeline

            stages = build_stages(cfg)
            if args.dry_run:
                if stages[0].config.get("annotations"):
                    from spatialsam.data.io import build_pseudo_label_samples
                    from spatialsam.data.annotations import validate_annotation_samples

                    bootstrap = OmegaConf.create(stages[0].config)
                    validate_annotation_samples(
                        build_pseudo_label_samples(bootstrap), bootstrap.resolution
                    )
                print(
                    json.dumps(
                        [
                            {
                                "name": stage.name,
                                "inputs": list(map(str, stage.inputs)),
                                "outputs": {
                                    k: str(v) for k, v in stage.outputs.items()
                                },
                                "config": stage.config,
                            }
                            for stage in stages
                        ],
                        indent=2,
                        ensure_ascii=False,
                    )
                )
            else:
                run_pipeline(cfg)
        else:
            from predict.configuration import validate_options

            OmegaConf.resolve(cfg)
            validate_options(cfg.task, bootstrap=args.command == "bootstrap")
            if args.dry_run:
                if args.command == "bootstrap" and cfg.task.get("annotations"):
                    from spatialsam.data.io import build_pseudo_label_samples
                    from spatialsam.data.annotations import validate_annotation_samples

                    validate_annotation_samples(
                        build_pseudo_label_samples(cfg.task), cfg.task.resolution
                    )
                print(OmegaConf.to_yaml(cfg.task, resolve=True))
            elif args.command == "predict":
                from predict.run_inference import run

                run(cfg)
            else:
                from predict.run_pseudo_label_init import run

                run(cfg)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
