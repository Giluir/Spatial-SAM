"""Build the optional SAM2 CUDA extension locally without using a GPU."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda-home", type=Path, required=True)
    parser.add_argument("--arch", default="8.6")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source = root / "sam2/csrc/connected_components.cu"
    build = root / "build/sam2_cuda"
    build.mkdir(parents=True, exist_ok=True)
    os.environ["CUDA_HOME"] = str(args.cuda_home.resolve())
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["TORCH_CUDA_ARCH_LIST"] = args.arch
    os.environ.setdefault("MAX_JOBS", "2")

    import torch
    from torch.utils.cpp_extension import load

    extension = load(
        name="_C",
        sources=[str(source)],
        build_directory=str(build),
        extra_cuda_cflags=[
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ],
        verbose=True,
    )
    destination = root / "sam2/_C.so"
    temporary = destination.with_suffix(".so.tmp")
    shutil.copyfile(extension.__file__, temporary)
    temporary.replace(destination)
    report = {
        "build_succeeded": True,
        "gpu_validation_executed": False,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "extension_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "extension": str(destination),
        "cuda_home": str(args.cuda_home.resolve()),
        "cuda_arch": args.arch,
        "nvcc": subprocess.check_output(
            [str(args.cuda_home / "bin/nvcc"), "--version"], text=True
        ).strip(),
        "pytorch": torch.__version__,
        "pytorch_cuda": torch.version.cuda,
        "host_cxx": os.environ.get("CXX", "c++"),
    }
    report_path = root / "build/sam2_extension_build.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
