# Experiment 4 Comparison Report

## Question

Can approximate 3D molecular geometry, represented as an atom point cloud, predict molecule activity?

## Course Syllabus Link

- Unit III: 3D Point Cloud
- Unit III: Volumetric/structure representation, through normalized xyz coordinates and atom-wise geometric features
- Geometry concept: translation normalization, scale normalization, Euclidean coordinates, and rotation augmentation

## Model

```text
SMILES -> RDKit ETKDG 3D conformer -> atom point cloud -> PointNet-style classifier -> active/inactive
```

## Test Metrics

| ROC-AUC | PR-AUC | F1 | Balanced Accuracy |
|---:|---:|---:|---:|
| 0.6782 | 0.8791 | 0.8130 | 0.6048 |

## Interpretation

This is a direct 3D Vision & Geometry course-topic experiment because the input is a 3D point cloud.
The model is intentionally lightweight; compare it against Experiment 1 to see whether 3D geometry adds useful signal beyond 2D fingerprints.
