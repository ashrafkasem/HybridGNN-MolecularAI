# Hybrid GNN-based IC50 Prediction

A Hybrid Graph Neural Network (GNN) model that combines molecular graph representations with calculated molecular descriptors for predicting IC50 values.

## Project Structure

```
GNN/
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
├── src/                         # Source code
│   ├── __init__.py
│   ├── train.py                 # Main training script (Hybrid Model)
│   ├── model.py                 # Hybrid GNN + Descriptors architecture
│   ├── descriptors.py           # Descriptor calculation utilities
│   ├── utils.py                 # Graph utilities
│   ├── plotting.py              # Plotting utilities
│   ├── dataset.py               # Dataset class (with caching)
│   ├── evaluate.py              # Evaluation script
│   ├── predict.py               # Single molecule inference script
│   └── legacy/                  # Archived/Unused scripts
├── data/                        # Datasets (stores .pt descriptor caches)
│   ├── raw/                     # Original/raw datasets
│   └── processed/               # Processed/split datasets
├── models/                      # Trained model files
└── results/                     # Training results and outputs
```

## Installation

1. Create and activate a Python virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Hybrid Model

The main script `src/train.py` trains a model that fuses GNN embeddings with molecular descriptors (Morgan fingerprints + physicochemical properties).
*Note: Descriptors are automatically cached to disk (`.pt` files) alongside your datasets to significantly speed up future runs.*

#### Train on All Datasets (Recommended)

```bash
python src/train.py \
    --data_dir data/raw \
    --epochs 100 \
    --batch_size 32 \
    --model_name best_model_hybrid.pt
```

#### Train on Specific Datasets

```bash
python src/train.py \
    --datasets CHEMBL204 CHEMBL206 \
    --epochs 100
```

### Key Options

- `--use_fingerprints`: Include Morgan fingerprints (default: True)
- `--fp_bits`: Number of fingerprint bits (default: 512)
- `--gnn_hidden_dim`: Hidden dimension for GNN (default: 128)
- `--regressor_hidden_dim`: Hidden dimension for regressor (default: 64)
- `--oversample`: Oversample training data to handle imbalance

### Evaluation

Evaluate a trained model on new datasets using `src/evaluate.py`.

```bash
python src/evaluate.py \
    --model_path models/saved/best_model_hybrid.pt \
    --data_path data/raw/CHEMBL204_IC50.csv \
    --output_dir evaluation_results \
    --generate_plots
```

**Arguments:**
- `--model_path`: Path to the trained model checkpoint.
- `--data_path`: Path to a CSV file or a directory containing CSV files.
- `--output_dir`: Directory where predictions and plots will be saved.
- `--generate_plots`: Flag to generate scatter plots and performance metrics.

### Single Molecule Inference

Use `src/predict.py` to predict the IC50 for a specific molecule using either its SMILES string or ChEMBL ID.

**Using a SMILES string:**
```bash
python src/predict.py \
    --model_path models/saved/best_model_hybrid.pt \
    --smiles "CC(=O)Oc1ccccc1C(=O)O"
```

**Using a ChEMBL ID (automatically fetches SMILES via REST API):**
```bash
python src/predict.py \
    --model_path models/saved/best_model_hybrid.pt \
    --chembl_id CHEMBL25
```

## Model Architecture

The **HybridGNNModel** consists of two branches:

1.  **GNN Branch**:
    - Graph Attention Network (GAT) layers
    - Global mean and max pooling
    - Extracts structural features from the molecular graph

2.  **Descriptor Branch**:
    - MLP processing vector of molecular descriptors
    - Descriptors include: Morgan fingerprints, Molecular Weight, LogP, TPSA, H-bond donors/acceptors, etc.

3.  **Fusion**:
    - Concatenates GNN embeddings, processed descriptors, and molecule size (N)
    - Passed through a final regressor MLP to predict log(IC50)

## Outputs

- **Models**: Saved in `models/saved/`
- **Plots**: Saved in `results/plots/` (Training curves, Parity plots, Per-dataset metrics)
- **Metrics**: Saved in `results/logs/`

