# MCG-HGT

MCG-HGT is a multimodal heterogeneous graph framework for
ingredient-target interaction prediction. It combines frozen molecular and
protein representations, similarity-augmented heterogeneous graphs,
residual-gated HGT propagation, and a gated bilinear scorer trained with an
InfoNCE-style ranking objective.

This repository is the publication release for:

> MCG-HGT: Multimodal Heterogeneous Graph Learning for Herbal
> Ingredient-Target Interaction Prediction.

## Repository Scope

The GitHub repository contains source code, configuration files, small smoke-test
inputs, and data manifests. Large pretrained checkpoints and full preprocessed
input matrices are intentionally distributed through Zenodo, not Git.

Important: `10.5281/zenodo.15088340` currently resolves to a DSSA-PPI record, not
an MCG-HGT record. Replace `ZENODO_RECORD_ID` below with the final MCG-HGT Zenodo
record after publication.

## Installation

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

## Data Layout

After downloading the Zenodo artifact, arrange files as:

```text
data/
  HIT/
    edges.txt
    ingredient_similarity.txt
    target_similarity.txt
    ingredients_embeddings.csv
    protein_embeddings.csv
  BindingDB/
  BioSNAP/
checkpoints/
  HIT_CVS1_MCG-HGT/
  BindingDB_CVS1_MCG-HGT/
  BioSNAP_CVS1_MCG-HGT/
```

Download helper:

```bash
python scripts/download_zenodo.py --record ZENODO_RECORD_ID --out data
```

See `manifests/zenodo_manifest.tsv` and `docs/DATA_AVAILABILITY.md` for the
expected Zenodo archive contents.

## Training

Run from a config:

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

For cold-start settings, set `--cv_mode CVS2` for unseen ingredients or
`--cv_mode CVS3` for unseen targets. Add `--strict_cold_start` to remove held-out
nodes from the training graph.

## Inference

```bash
python -m mcg_hgt.inference \
  --data_dir data/HIT \
  --ligand_embed data/HIT/ingredients_embeddings.csv \
  --target_embed data/HIT/protein_embeddings.csv \
  --checkpoint checkpoints/HIT_CVS1_MCG-HGT/best_fold1_auprc.pt \
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

## Smoke Checks

The help-level smoke check does not require PyTorch or DGL:

```bash
python scripts/smoke_test.py
```

In an environment with PyTorch and DGL installed, run a one-epoch toy training
check:

```bash
python scripts/smoke_test.py --train --device cpu --epochs 1
```

## Zenodo Bundle Preparation

Prepare the full artifact directory outside the Git repository, then bundle it:

```bash
python scripts/make_zenodo_bundle.py \
  --source /path/to/MCG-HGT-zenodo-artifacts \
  --output MCG-HGT-zenodo-artifacts.tar.gz
```

Upload the resulting archive to a Zenodo record titled for MCG-HGT and use that
DOI in the manuscript.

## Acknowledgements

This release preserves attribution to the open-source tools and prior work used
in the project, including DGL, KPGT, ESM, and Multi-ITI. The original working
tree was derived from a Multi-ITI-style heterogeneous graph workflow and then
extended with residual-gated HGT propagation, semantic gates, gated scoring, and
InfoNCE ranking supervision for MCG-HGT.

## License

This repository is released under the MIT License.
