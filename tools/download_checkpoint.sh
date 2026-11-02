#!/usr/bin/env bash
set -euo pipefail
checkpoint_dir="${1:-checkpoints}"
mkdir -p "$checkpoint_dir"
curl --fail --location --retry 3 \
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt \
  --output "$checkpoint_dir/sam2.1_hiera_tiny.pt"
