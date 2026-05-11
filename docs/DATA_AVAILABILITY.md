# Data And Code Availability

The source code, data manifests, HIT pretrained MCG-HGT model weights, and HIT
preprocessed model input data are distributed through the public GitHub
repository and its `v1.0.0` release assets:

https://github.com/jjjsun4-design/MCG-HGT

Recommended manuscript wording:

> The source code, data manifests, HIT pretrained MCG-HGT model weights, and HIT
> preprocessed model input data that support the findings of this manuscript are
> openly available in the GitHub repository
> https://github.com/jjjsun4-design/MCG-HGT and its associated release assets.

Recommended GitHub Release asset layout:

```text
weights/
  HIT/CVS1_MCG-HGT/
    best_fold1_auprc.pt
    best_fold2_auprc.pt
    best_fold3_auprc.pt
    best_fold4_auprc.pt
    best_fold5_auprc.pt
preprocessed/
  HIT/
    edges.txt
    ingredient_similarity.txt
    target_similarity.txt
    ingredients_embeddings.csv
    protein_embeddings.csv
manifests/
  checksums.txt
  release_manifest.tsv
README_RELEASE_ASSETS.md
```

Large model checkpoints and full preprocessed feature matrices are intentionally
kept out of Git history and uploaded as GitHub Release assets.
