# Data Usage

This folder intentionally does not duplicate the existing datasets.

Use the curated data already present in:

```text
../multimodal_datapipeline/data/processed/
../multimodal_datapipeline/dataset_pipeline_output/
```

Primary table for this deep-learning project:

```text
../multimodal_datapipeline/data/processed/baseline_4_molecule_protein.csv
```

Primary AlphaFold structures:

```text
../multimodal_datapipeline/dataset_pipeline_output/alphafold/structures/
```

Optional image tables:

```text
../multimodal_datapipeline/data/processed/baseline_3_image_only.csv
../multimodal_datapipeline/data/processed/baseline_5_molecule_image.csv
```

Keeping the data in one canonical location avoids stale copies and keeps this folder focused on the deep-learning use case.

