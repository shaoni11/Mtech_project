# References for the research proposal

The proposal PDF (`research_proposal_multimodal_drug_discovery.pdf`) contains
placeholder markers `[cite:193]` to `[cite:198]` and no reference list. The table
below proposes the paper behind each marker, matched to how the proposal text
describes it. Check each entry against the PDF/DOI before submitting; confidence
reflects how closely the found paper matches the proposal's description.

| Marker | Proposed reference | Confidence |
|---|---|---|
| [cite:193] | MolIG — *MultiModal-Learning for Predicting Molecular Properties: A Framework Based on Image and Graph Structures.* arXiv:2311.16666 (2023). https://arxiv.org/abs/2311.16666 | High: graph + image contrastive pretraining, evaluated on MoleculeNet and ADMET, as the proposal says |
| [cite:194] | KEDD — Luo, Y. et al. *Toward Unified AI Drug Discovery with Multimodal Knowledge.* Health Data Science (2024); preprint arXiv:2305.01523. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10886071/ | High: structures + knowledge graph + text, DTI/DDI/property/PPI tasks, sparse attention and modality masking |
| [cite:195] | STAMP-DPI (preprint name X-DPI) — *X-DPI: A structure-aware multi-modal deep learning model for drug-protein interactions prediction.* bioRxiv 10.1101/2021.06.17.448780 (2021). https://www.biorxiv.org/content/10.1101/2021.06.17.448780 | Medium: structure-aware contact-map GNN + pretrained drug/protein embeddings + transformer decoder matches; the Davis/GalaxyDB/AstraZeneca evaluation and the final journal version were not confirmed — cite the journal version if the preprint lists one |
| [cite:196] | Borkakoti, N. & Thornton, J. M. *AlphaFold2 protein structure prediction: Implications for drug discovery.* Current Opinion in Structural Biology 78, 102526 (2023). doi:10.1016/j.sbi.2022.102526 | Medium: a well-regarded review of AlphaFold2 across drug-discovery stages; confirm it covers the specific limitations the proposal lists (induced fit, multimers, cryptic pockets) |
| [cite:197] | Krentzel, D., Shorte, S. L. & Zimmer, C. *Deep learning in image-based phenotypic drug discovery.* Trends in Cell Biology 33(7), 538–554 (2023). doi:10.1016/j.tcb.2022.11.011 | High: the review of deep learning for image-based phenotypic drug discovery |
| [cite:198] | Jumper, J. et al. *Highly accurate protein structure prediction with AlphaFold.* Nature 596, 583–589 (2021). doi:10.1038/s41586-021-03819-2 | High: the AlphaFold paper ("near-experimental accuracy") |

## Datasets used in the code (cite in the thesis)

- Davis, M. I. et al. *Comprehensive analysis of kinase inhibitor selectivity.* Nature Biotechnology 29, 1046–1051 (2011). Used via the DeepPurpose plain-text release.
- Caie, P. D. et al. *High-content phenotypic profiling of drug response signatures across distinct cancer cells.* Molecular Cancer Therapeutics 9, 1913–1926 (2010); Ljosa, V. et al. *Comparison of methods for image-based profiling of cellular morphological responses to small-molecule treatment.* J. Biomol. Screen. 18, 1321–1329 (2013). BBBC021, https://bbbc.broadinstitute.org/BBBC021
- Zdrazil, B. et al. *The ChEMBL Database in 2023.* Nucleic Acids Research 52, D1180–D1192 (2024).
- Varadi, M. et al. *AlphaFold Protein Structure Database.* Nucleic Acids Research 50, D439–D444 (2022).
- Oquab, M. et al. *DINOv2: Learning Robust Visual Features without Supervision.* arXiv:2304.07193 (2023).
- Lin, Z. et al. *Evolutionary-scale prediction of atomic-level protein structure with a language model* (ESM-2). Science 379, 1123–1130 (2023).
