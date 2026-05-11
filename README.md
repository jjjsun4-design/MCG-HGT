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

Source code, runnable examples, pre-training utilities, HIT data manifests, and
documentation are kept in GitHub. Full HIT preprocessed inputs and pretrained
weights are distributed through the `v1.0.0` GitHub Release assets.

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
  docs/                    Data/code availability notes
```

The `pre-training/` directory contains the ingredient and target
feature-pretraining notebooks/utilities. Large pretrained encoders and generated
feature matrices are not committed to Git history.

## Data

HIT preprocessed inputs and pretrained MCG-HGT checkpoints are available from
the `v1.0.0` GitHub Release:

[https://github.com/jjjsun4-design/MCG-HGT/releases/tag/v1.0.0](https://github.com/jjjsun4-design/MCG-HGT/releases/tag/v1.0.0)

## Usage

Train MCG-HGT on HIT CVS1 with the publication defaults:

```bash
python -m mcg_hgt.train
```

Common overrides stay available when needed:

```bash
python -m mcg_hgt.train --device cuda:1 --num_epochs 50
```

Run inference for a pair list:

```bash
python -m mcg_hgt.inference
```

Use a custom pair list or output file with:

```bash
python -m mcg_hgt.inference --pairs path/to/pairs.csv --output outputs/scores.csv
```

## Pre-Training

The `pre-training/` directory contains the inherited representation-learning
utilities:

```text
pre-training/
  esm/
  scripts/
  src/
  ingredient_pre-training.ipynb
  target_pre-training.ipynb
```

The local Python scripts used for feature and graph preparation are also kept in
`pre-training/scripts/`:

- `extract_molmcl_molecule_embeddings.py`: generate ingredient molecular
  embeddings from SMILES with a MolMCL checkpoint.
- `extract_esm_protein_embeddings.py`: extract and align target protein
  embeddings with ESM.
- `build_graph_from_csv.py`: construct `edges.txt`,
  `ingredient_similarity.txt`, and `target_similarity.txt` from curated CSV
  files.
- `pretrain_relation_alignment.py`: relation-similarity contrastive
  pre-training for molecule and protein embeddings.

The generated HIT feature matrices are provided as release assets.

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

This release uses open-source scientific machine-learning tools, including:

- DGL: https://www.dgl.ai/
- KPGT: https://github.com/lihan97/KPGT
- ESM: https://github.com/facebookresearch/esm

## License

This repository is released under the MIT License.
