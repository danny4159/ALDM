# Repository Guidelines

## Project Structure & Module Organization
- Root folders: `VQ-GAN/` (training code), `asset/` (figures), `LDM/` (related models). Work inside `VQ-GAN/` unless you know you need the others.
- Key files: `main.py` (Lightning entrypoint), `configs/*.yaml` (experiment configs), `taming/` (model, data, losses), `data/` (local assets), `environment.yaml` (conda spec), `setup.py` (minimal package install).
- Datasets are not tracked; point config paths to your local BraTS data directory.

## Setup, Build, and Run Commands
- Create env: `conda env create -f environment.yaml && conda activate vqgan`.
- Stage 1 train (reconstruction): `CUDA_VISIBLE_DEVICES=<GPU_ID> python main.py -b configs/brats_vqgan_stage1.yaml -t True --gpus 0`.
- Stage 2 train (after stage1 ckpt): `CUDA_VISIBLE_DEVICES=<GPU_ID> python main.py -b configs/brats_vqgan_stage2.yaml -t True --gpus 0`.
- Inference: `python scripts/samples.py -b configs/brats_vqgan.yaml --outdir=/outdir -r vqgan.ckpt`.
- Adjust `data_path` in configs to your dataset root (e.g., `/data/BraTS2024/TrainingData`).

## Data Layout Expectations
- Each subject folder under `TrainingData/` should contain `*_t1n.nii.gz`, `*_t1c.nii.gz`, `*_t2w.nii.gz`, `*_t2f.nii.gz`. Update `csv_path` in configs if you use a custom split; otherwise all subjects train, last 10 validate/test.

## Coding Style & Naming Conventions
- Python 3.8+, 4-space indentation, snake_case for variables/functions, PascalCase for classes.
- Prefer explicit config entries over hard-coded constants; mirror existing YAML patterns.
- Keep imports local to modules (no wildcard imports). Add brief comments only for non-obvious logic.

## Testing & Verification
- No dedicated unit test suite yet. Sanity check a small run: set `batch_size` low in config and run `main.py` to ensure data loading works and losses decrease for a few steps.
- When modifying data code, verify file discovery by logging the resolved `data_path` and subject count (see `taming/data/brats.py`).

## Commit & Pull Request Guidelines
- Commits: concise imperative subject (`Add 3D VQ-GAN loader fix`), group related changes; avoid bundling env and training outputs.
- PRs: include purpose, key config changes, sample commands used, and links to any issues. Add screenshots or logged metrics paths when altering training or inference behavior.

## Language Preference
- 문서와 커뮤니케이션은 한국어로 작성해주세요. 필요 시 영어 용어는 그대로 사용해도 됩니다.
