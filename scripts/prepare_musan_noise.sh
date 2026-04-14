#!/usr/bin/env bash
# Download a subset of MUSAN (noise + music) for augmentation on H100.
# Run from repo root: bash scripts/prepare_musan_noise.sh
set -euo pipefail
NOISE_DIR="${NOISE_DIR:-data/noise}"
mkdir -p "$NOISE_DIR"
cd "$NOISE_DIR"
if [ ! -d "musan" ]; then
    echo "[noise] downloading MUSAN (~10 GB)…"
    curl -L -O https://www.openslr.org/resources/17/musan.tar.gz
    tar -xf musan.tar.gz
    rm musan.tar.gz
fi
# Flatten noise + music subsets into one dir for NoiseBank
mkdir -p flat
find musan/noise musan/music -name '*.wav' -print0 | xargs -0 -I{} cp -n "{}" flat/
echo "[noise] total files: $(ls flat | wc -l)"
