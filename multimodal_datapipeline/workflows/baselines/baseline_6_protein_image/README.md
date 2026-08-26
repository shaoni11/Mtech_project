# Baseline 6: Protein + Image

Flow:

```text
protein sequence -> protein encoder \
                                    -> fusion -> prediction
cell image -> image encoder --------/
```

Current model components:

```text
package/multimodal_datapipeline/models/protein_encoder.py
package/multimodal_datapipeline/models/image_encoder.py
package/multimodal_datapipeline/models/fusion.py
```

Recommended next script:

```text
train_protein_image_baseline.py
```

This is exploratory until image records can be mapped to targets or target-like labels. Keep it optional unless the dataset alignment supports it.

Commands:

```bash
python workflows/baselines/baseline_6_protein_image/curate_data.py
python workflows/baselines/baseline_6_protein_image/validate_inputs.py
python workflows/baselines/baseline_6_protein_image/train.py
```
