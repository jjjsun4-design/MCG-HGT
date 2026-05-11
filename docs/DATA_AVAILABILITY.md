# Data Availability

The GitHub repository intentionally excludes pretrained weights and large
preprocessed input matrices. Those files should be distributed through a Zenodo
record dedicated to MCG-HGT.

The DOI `10.5281/zenodo.15088340` was checked during release preparation and
currently resolves to a record titled `DSSA-PPI data`; it should not be used for
MCG-HGT unless the Zenodo metadata and files are corrected.

Recommended manuscript wording after the MCG-HGT Zenodo record is published:

> The pretrained MCG-HGT model weights and preprocessed model input data are
> available on Zenodo at [correct MCG-HGT DOI]. The source code, data manifests,
> and reproduction scripts are openly available at
> https://github.com/jjjsun4-design/MCG-HGT.

Recommended Zenodo archive layout:

```text
weights/
  HIT_CVS1_MCG-HGT/
  BindingDB_CVS1_MCG-HGT/
  BioSNAP_CVS1_MCG-HGT/
preprocessed/
  HIT/
  BindingDB/
  BioSNAP/
manifests/
  checksums.txt
  zenodo_manifest.tsv
README_ZENODO.md
```
