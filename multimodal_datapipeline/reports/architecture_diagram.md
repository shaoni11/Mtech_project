# Multimodal Data Pipeline Architecture

```mermaid
flowchart TB
    %% External sources
    subgraph S["External Data Sources"]
        CHEMBL["ChEMBL API<br/>bioactivity, molecules, targets"]
        AF["AlphaFold DB<br/>protein structures"]
        BBBC["BBBC021<br/>cellular phenotypic images + metadata"]
        WEB["Optional web table<br/>scraped metadata"]
    end

    %% Entrypoints
    subgraph E["Entrypoints"]
        CLI["data/acquire_data.py<br/>dataset acquisition CLI"]
        WRAPPERS["workflows/<br/>baseline workflows"]
    end

    %% Core ingestion pipeline
    subgraph P["data/pipelines"]
        PIPE["dataset_pipeline.py<br/>orchestrates downloads and manifest"]
    end

    %% Data acquisition helpers
    subgraph H["data/sources"]
        CHEM_MOD["chembl.py<br/>target lookup + activity fetch"]
        AF_MOD["alphafold.py<br/>PDB structure download"]
        BBBC_MOD["bbbc021.py<br/>metadata, ZIP download, extraction"]
        SCRAPE_MOD["scrape.py<br/>HTML table extraction"]
    end

    %% Shared package helpers
    subgraph U["package/multimodal_datapipeline"]
        IO["utils/io.py<br/>directory, CSV, JSON helpers"]
    end

    %% Output stores
    subgraph O["Dataset Pipeline Output"]
        OUT["dataset_pipeline_output/manifest.json"]
        CHEM_OUT["dataset_pipeline_output/chembl/<br/>target_mapping.csv<br/>activities.csv<br/>activities_multitarget.csv<br/>activities_by_target/*.csv"]
        CYTO_OUT["dataset_pipeline_output/chembl/<br/>cytoskeleton_activities.csv"]
        AF_OUT["dataset_pipeline_output/alphafold/<br/>metadata.csv<br/>structures/*.pdb"]
        BBBC_OUT["dataset_pipeline_output/bbbc021/<br/>BBBC021 metadata CSVs<br/>download_manifest.csv<br/>images/<br/>zips/"]
        SCRAPE_OUT["dataset_pipeline_output/scrape/<br/>scraped_table.csv"]
    end

    %% Curated data layer
    subgraph D["Trainable Data Layer"]
        RAW["data/raw/"]
        INTERIM["data/interim/"]
        PROCESSED["data/processed/<br/>curated tables and features"]
    end

    %% Model package
    subgraph M["Model Components"]
        MOL["models/molecule_encoder.py<br/>SMILES / molecule features"]
        PROT["models/protein_encoder.py<br/>protein sequence / structure features"]
        IMG["models/image_encoder.py<br/>cell image features"]
        FUSION["models/fusion.py<br/>concatenation fusion head"]
    end

    %% Experiments
    subgraph X["Baseline and Fusion Experiments"]
        B1["baseline_1_molecule_only<br/>SMILES -> molecule encoder -> prediction"]
        B2["baseline_2_protein_only<br/>protein -> protein encoder -> prediction"]
        B3["baseline_3_image_only<br/>image -> image encoder -> prediction"]
        B4["baseline_4_molecule_protein<br/>molecule + protein fusion"]
        B5["baseline_5_molecule_image<br/>molecule + image fusion"]
        B6["baseline_6_protein_image<br/>protein + image fusion"]
        B7["baseline_7_molecule_protein_image<br/>full multimodal fusion"]
    end

    %% Results
    subgraph R["Experiment Outputs"]
        EXP["results/<br/>metrics.json<br/>test_predictions.csv<br/>model artifacts"]
        REPORTS["reports/<br/>figures, tables, thesis exports"]
        NOTEBOOKS["exploration/<br/>scratch notebooks only"]
    end

    CHEMBL --> CLI
    AF --> CLI
    BBBC --> CLI
    WEB --> CLI
    CLI --> PIPE
    CLI --> CYTO_OUT

    PIPE --> CHEM_MOD
    PIPE --> AF_MOD
    PIPE --> BBBC_MOD
    PIPE --> SCRAPE_MOD
    PIPE --> IO

    CHEM_MOD --> CHEM_OUT
    AF_MOD --> AF_OUT
    BBBC_MOD --> BBBC_OUT
    SCRAPE_MOD --> SCRAPE_OUT
    IO --> OUT

    CHEM_OUT --> RAW
    AF_OUT --> RAW
    BBBC_OUT --> RAW
    SCRAPE_OUT --> RAW
    RAW --> INTERIM --> PROCESSED

    PROCESSED --> WRAPPERS
    WRAPPERS --> B1
    WRAPPERS --> B2
    WRAPPERS --> B3
    WRAPPERS --> B4
    WRAPPERS --> B5
    WRAPPERS --> B6
    WRAPPERS --> B7

    MOL --> B1
    PROT --> B2
    IMG --> B3
    MOL --> B4
    PROT --> B4
    MOL --> B5
    IMG --> B5
    PROT --> B6
    IMG --> B6
    MOL --> B7
    PROT --> B7
    IMG --> B7
    FUSION --> B4
    FUSION --> B5
    FUSION --> B6
    FUSION --> B7

    B1 --> EXP
    B2 --> EXP
    B3 --> EXP
    B4 --> EXP
    B5 --> EXP
    B6 --> EXP
    B7 --> EXP
    EXP --> REPORTS
    PROCESSED --> NOTEBOOKS
```

## Main Flow

1. `data/acquire_data.py` is the unified data acquisition script.
2. `data/acquire_data.py` calls `data/pipelines/dataset_pipeline.py` for ChEMBL, AlphaFold, BBBC021, and scraping.
3. The `cytoskeleton-chembl` mode downloads actin/tubulin ChEMBL activity records.
4. Downloaded artifacts are written under `dataset_pipeline_output/`.
5. Curated and trainable datasets are organized through `data/raw/`, `data/interim/`, and `data/processed/`.
6. Baseline workflows consume processed data and route it through molecule, protein, image, or fusion model components.
7. Experiment outputs are stored under `results/`, while thesis-ready artifacts belong in `reports/`.
