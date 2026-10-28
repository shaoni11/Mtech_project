"""Model components for molecule, protein, image, and fusion branches."""

from multimodal_datapipeline.models.fusion import (
    ConcatenationFusion,
    CrossAttentionFusion,
    FusionHead,
    GatedFusion,
    TwoModalityFusion,
)
from multimodal_datapipeline.models.graph import Graph, GraphBatch, GraphEncoder, batch_graphs, to_dense_batch
from multimodal_datapipeline.models.image_encoder import DINOv2ImageEncoder, ImageEncoder
from multimodal_datapipeline.models.molecule_encoder import (
    MoleculeEncoder,
    MoleculeFingerprintMLP,
    MoleculeGNNEncoder,
    MoleculeGraphFeaturizer,
    MorganFingerprintConfig,
    MorganFingerprintFeaturizer,
)
from multimodal_datapipeline.models.protein_encoder import ESM2ProteinEncoder, ProteinEncoder
from multimodal_datapipeline.models.protein_structure_encoder import (
    ProteinGraphConfig,
    ProteinStructureEncoder,
    ProteinStructureGraphBuilder,
    parse_alphafold_pdb,
)

__all__ = [
    "ConcatenationFusion",
    "CrossAttentionFusion",
    "DINOv2ImageEncoder",
    "ESM2ProteinEncoder",
    "FusionHead",
    "GatedFusion",
    "Graph",
    "GraphBatch",
    "GraphEncoder",
    "ImageEncoder",
    "MoleculeEncoder",
    "MoleculeFingerprintMLP",
    "MoleculeGNNEncoder",
    "MoleculeGraphFeaturizer",
    "MorganFingerprintConfig",
    "MorganFingerprintFeaturizer",
    "ProteinEncoder",
    "ProteinGraphConfig",
    "ProteinStructureEncoder",
    "ProteinStructureGraphBuilder",
    "TwoModalityFusion",
    "batch_graphs",
    "parse_alphafold_pdb",
    "to_dense_batch",
]
