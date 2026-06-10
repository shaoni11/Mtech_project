# Baseline 6: Protein + Image

Flow:

```text
protein sequence -> protein encoder \
                                    -> fusion -> prediction
cell image -> image encoder --------/
```

Current model components:

```text
src/multimodal_datapipeline/models/protein_encoder.py
src/multimodal_datapipeline/models/image_encoder.py
src/multimodal_datapipeline/models/fusion.py
```

Recommended next script:

```text
train_protein_image_baseline.py
```

This is exploratory until image records can be mapped to targets or target-like labels. Keep it optional unless the dataset alignment supports it.

