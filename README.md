# MCG-HGT

## Introduction

![MCG-HGT overview](assets/mcg-hgt-overview.png)

MCG-HGT is a multimodal heterogeneous graph framework for herbal
ingredient-target interaction prediction. It integrates pre-trained molecular
and protein representations, a similarity-augmented ingredient-target
heterogeneous graph, Heterogeneous Graph Transformer (HGT) message passing,
residual/semantic gates, and a gated bilinear scorer for interaction
prediction.

![MCG-HGT architecture](assets/mcg-hgt-architecture.png)

This repository is the publication release for:

> MCG-HGT: Multimodal Heterogeneous Graph Learning for Herbal
> Ingredient-Target Interaction Prediction.

The repository follows the compact release style of Multi-ITI and iCAM-Net:
source code, runnable examples, pre-training utilities, HIT data manifests, and
documentation are kept in GitHub, while full HIT preprocessed inputs and
pretrained weights are distributed through the `v1.0.0` GitHub Release assets.

## Environment

The experiments were run with Python 3.8, PyTorch 1.13.1, and DGL 1.1.1.

```bash
conda env create -f environment.yml
conda activate mcg-hgt
```

Alternatively:

```bash
python -m pip install -r requirements.txt
```

For CUDA builds, install the PyTorch and DGL wheels matching your local CUDA
runtime if the pinned packages are not available from your package indexes.

## Repository Structure

```text
MCG-HGT/
  assets/                  Overview and architecture figures used in the manuscript/repository
  mcg_hgt/                 Main MCG-HGT model, graph, training, evaluation, inference code
  scripts/                 Reproducibility, smoke-test, and release helper scripts
  configs/                 HIT CVS1 configuration
  examples/smoke/          Tiny example for command/interface checks
  pre-training/            Ingredient and target representation pre-training utilities
  manifests/               HIT release asset manifest
  docs/                    Data/code availability and upstream attribution notes
```

The `pre-training/` directory mirrors the Multi-ITI release layout and contains
the ingredient and target feature-pretraining notebooks/utilities used by the
inherited workflow. Large pretrained encoders and generated feature matrices are
not committed to Git history.

## Data

This GitHub-only release uses the HIT dataset as the public full-data example.
Download the HIT preprocessed inputs and HIT pretrained MCG-HGT checkpoints from
the `v1.0.0` GitHub Release:

[https://github.com/jjjsun4-design/MCG-HGT/releases/tag/v1.0.0](https://github.com/jjjsun4-design/MCG-HGT/releases/tag/v1.0.0)

Download, extract, and verify:

```bash
python scripts/download_release_assets.py --tag v1.0.0 --out . --extract --verify
```

After extracting the release assets, arrange files as:

```text
data/
  HIT/
    edges.txt
    ingredient_similarity.txt
    target_similarity.txt
    ingredients_embeddings.csv
    protein_embeddings.csv
checkpoints/
  HIT/CVS1_MCG-HGT/
    best_fold1_auprc.pt
    ...
```

If you use the download helper with `--extract`, the assets unpack as
`preprocessed/HIT/` and `weights/HIT/CVS1_MCG-HGT/`; move or symlink them to the
`data/HIT/` and `checkpoints/HIT/CVS1_MCG-HGT/` paths shown above before running
the training commands.

See `manifests/release_manifest.tsv` and `docs/DATA_AVAILABILITY.md` for the
expected HIT asset contents.

## Usage

Run a help-level smoke check first:

```bash
python scripts/smoke_test.py
```

Train MCG-HGT on HIT CVS1 from the provided config:

```bash
python scripts/train_from_config.py --config configs/hit_cvs1.yaml
```

Equivalent explicit command:

```bash
python -m mcg_hgt.train \
  --device cuda:0 \
  --k_fold 10 \
  --batch_size 4096 \
  --lr 1e-4 \
  --wd 1e-5 \
  --num_epochs 250 \
  --graph_struct 3 \
  --method 5 \
  --data_dir data/HIT \
  --ligand_embed data/HIT/ingredients_embeddings.csv \
  --target_embed data/HIT/protein_embeddings.csv \
  --in_dim 512 \
  --h_dim 1024 \
  --out_dim 512 \
  --hgt_heads 8 \
  --num_layers 3 \
  --fanout 15 \
  --neg_k 5 \
  --cv_mode CVS1 \
  --input_gate_type glu \
  --score_gate gmu \
  --semantic_gate etype \
  --head_gate \
  --residual_gate \
  --checkpoint_dir checkpoints/HIT/CVS1_MCG-HGT
```

Run inference for a pair list:

```bash
python -m mcg_hgt.inference \
  --data_dir data/HIT \
  --ligand_embed data/HIT/ingredients_embeddings.csv \
  --target_embed data/HIT/protein_embeddings.csv \
  --checkpoint checkpoints/HIT/CVS1_MCG-HGT/best_fold1_auprc.pt \
  --pairs examples/smoke/pairs.csv \
  --output outputs/hit_scores.csv \
  --device cuda:0 \
  --in_dim 512 \
  --h_dim 1024 \
  --out_dim 512 \
  --hgt_heads 8 \
  --num_layers 3 \
  --input_gate_type glu \
  --score_gate gmu \
  --semantic_gate etype \
  --head_gate \
  --residual_gate
```

In an environment with PyTorch and DGL installed, run a one-epoch toy training
check:

```bash
python scripts/smoke_test.py --train --device cpu --epochs 1
```

## Pre-Training

The `pre-training/` directory contains the inherited representation-learning
utilities:

```text
pre-training/
  esm/
  src/
  ingredient_pre-training.ipynb
  target_pre-training.ipynb
```

These notebooks document the ingredient and target feature generation workflow.
They are included for reproducibility and attribution; the full generated HIT
feature matrices are provided as release assets.

## Release Asset Preparation

Prepare the full artifact directory outside the Git repository, then bundle it
into GitHub Release assets:

```bash
python scripts/make_release_bundle.py \
  --source /path/to/MCG-HGT-HIT-release-artifacts \
  --output-dir release-assets
```

Upload the resulting files to the `v1.0.0` GitHub Release. Release assets are
used for HIT data and weights because ordinary Git repository files should
remain small.

## Citation

If you use MCG-HGT, please cite the accompanying manuscript and this repository.

```bibtex
@software{mcg_hgt_2026,
  title = {MCG-HGT: Multimodal Heterogeneous Graph Learning for Herbal Ingredient-Target Interaction Prediction},
  author = {Sun, Jiehui and Li, Jinyu},
  year = {2026},
  url = {https://github.com/jjjsun4-design/MCG-HGT}
}
```

## Acknowledgement

This release preserves attribution to the open-source tools and prior work used
in the project:

- DGL: https://www.dgl.ai/
- KPGT: https://github.com/lihan97/KPGT
- ESM: https://github.com/facebookresearch/esm
- Multi-ITI: https://github.com/Xudong-Liang/Multi-ITI
- iCAM-Net: https://github.com/qunshanxingyun/iCAM-Net

The original working tree was derived from a Multi-ITI-style heterogeneous graph
workflow and then extended with residual-gated HGT propagation, semantic gates,
gated scoring, and InfoNCE ranking supervision for MCG-HGT.

## License

This repository is released under the MIT License.
