# Baseline 3: Image Only

Flow:

```text
BBBC021 cell image -> image encoder -> prediction
```

Current model component:

```text
src/multimodal_datapipeline/models/image_encoder.py
```

Planned input table:

```text
image_path
compound
moa_label
metadata fields
```

Recommended next script:

```text
train_image_only_baseline.py
```

The image encoder uses pretrained DINOv2 embeddings by default.

