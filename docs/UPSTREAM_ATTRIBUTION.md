# Upstream Attribution

MCG-HGT builds on a Multi-ITI-style ingredient-target interaction workflow and
uses public scientific machine-learning tooling. The public release keeps this
lineage explicit so readers can separate inherited components from the MCG-HGT
extensions.

## Multi-ITI

- Repository: https://github.com/Xudong-Liang/Multi-ITI
- Reused release pattern: compact top-level README, overview figure, HIT data
  workflow, and `pre-training/` utilities for ingredient/target representation
  generation.
- MCG-HGT extensions: residual-gated HGT propagation, semantic gates, gated
  scoring, and InfoNCE-style ranking supervision.

## iCAM-Net

- Repository: https://github.com/qunshanxingyun/iCAM-Net
- Reused release pattern: clear data/code organization, simple command-line
  entry points, case/inference documentation style, and citation-oriented
  release notes.

## Core Libraries

- DGL: https://www.dgl.ai/
- KPGT: https://github.com/lihan97/KPGT
- ESM: https://github.com/facebookresearch/esm

The files in this repository are released under the MIT License unless a file
or upstream dependency states otherwise.
