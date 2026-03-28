"""
Inference script for single molecule prediction
Accepts either a SMILES string or a ChEMBL ID
"""
import torch
import numpy as np
import argparse
import os
import sys

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import HybridGNNModel
from descriptors import calculate_all_descriptors
from utils import ATOM_COMBINED_TYPES, BOND_TYPES, smiles_to_data

def get_smiles_from_chembl(chembl_id):
    """Fetch Canonical SMILES from ChEMBL API given a ChEMBL ID."""
    print(f"Fetching SMILES for {chembl_id} from ChEMBL...")
    try:
        from chembl_webresource_client.new_client import new_client
    except ImportError:
        print("Error: chembl_webresource_client is not installed. Please run 'pip install chembl-webresource-client'")
        sys.exit(1)
        
    molecule = new_client.molecule
    try:
        m = molecule.filter(chembl_id=chembl_id).only(['molecule_structures'])
        if len(m) == 0:
            print(f"Error: Could not find molecule with ID {chembl_id}")
            sys.exit(1)
            
        structures = m[0].get('molecule_structures')
        if structures and 'canonical_smiles' in structures:
            return structures['canonical_smiles']
        else:
            print(f"Error: No SMILES structure available for {chembl_id}")
            sys.exit(1)
    except Exception as e:
        print(f"Error fetching from ChEMBL API: {e}")
        sys.exit(1)

def predict(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")
    
    if args.smiles:
        smiles = args.smiles
        print(f"Using provided SMILES: {smiles}")
    elif args.chembl_id:
        smiles = get_smiles_from_chembl(args.chembl_id)
        print(f"Resolved to SMILES: {smiles}")
    else:
        print("Error: Must provide either --smiles or --chembl_id")
        return
        
    if not os.path.exists(args.model_path):
        print(f"Error: Model not found at {args.model_path}")
        return
        
    print(f"Loading model from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location=device)
    
    use_fingerprints = checkpoint.get('use_fingerprints', True)
    fp_bits = checkpoint.get('fp_bits', 512)
    descriptor_size = checkpoint.get('descriptor_size', 544)
    desc_mean = checkpoint.get('desc_mean')
    desc_std = checkpoint.get('desc_std')

    if desc_mean is not None and desc_std is not None:
        desc_mean = desc_mean.to(device)
        desc_std = desc_std.to(device)
        print("Loaded descriptor normalization stats from checkpoint")
    else:
        print("Warning: No descriptor normalization stats found in checkpoint")
    
    # Initialize model
    model = HybridGNNModel(
        node_vocab_size=len(ATOM_COMBINED_TYPES),
        num_node_features=11,
        edge_feature_size=len(BOND_TYPES) + 2,
        descriptor_size=descriptor_size,
        gnn_hidden_dim=args.gnn_hidden_dim,
        gnn_num_layers=args.gnn_num_layers,
        gnn_num_heads=args.gnn_num_heads,
        regressor_hidden_dim=args.regressor_hidden_dim,
        regressor_num_hidden_layers=args.regressor_num_hidden_layers
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("Preparing molecule data...")
    # Calculate descriptors
    descriptors = calculate_all_descriptors(smiles, use_fingerprints=use_fingerprints, fp_bits=fp_bits)
    descriptors = descriptors.unsqueeze(0).to(device) # Batch size 1

    # Reuse train-split normalization statistics saved in checkpoint.
    if desc_mean is not None and desc_std is not None:
        descriptors = (descriptors - desc_mean) / desc_std
    
    # Convert to graph data
    data = smiles_to_data(smiles, y=0.0) # Dummy target
    if data is None:
        print("Error: Could not convert SMILES to a valid graph representation.")
        return
        
    # Batch structure wrapper for single item
    batch = torch.zeros(data.x.size(0), dtype=torch.long, device=device)
    
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    edge_attr = data.edge_attr.to(device)
    N = data.N.to(device)
    
    with torch.no_grad():
        print("Running prediction...")
        pred_log_ic50 = model(x, edge_index, edge_attr, batch, N, descriptors).squeeze().item()
        pred_ic50 = 10 ** pred_log_ic50
        
    print("\n" + "="*40)
    print("PREDICTION RESULTS")
    print("="*40)
    print(f"Input SMILES:      {smiles}")
    if args.chembl_id:
        print(f"ChEMBL ID:         {args.chembl_id}")
    print(f"Predicted logIC50: {pred_log_ic50:.4f}")
    print(f"Predicted IC50:    {pred_ic50:.4f} nM (appx)")
    print("="*40 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict IC50 for a single molecule')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model (.pt)')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--smiles', type=str, help='SMILES string of the molecule')
    group.add_argument('--chembl_id', type=str, help='ChEMBL ID to automatically lookup SMILES')
    
    parser.add_argument('--cpu', action='store_true', help='Force CPU usage')
    
    # Model architecture args (should match training)
    parser.add_argument('--gnn_hidden_dim', type=int, default=128)
    parser.add_argument('--gnn_num_layers', type=int, default=2)
    parser.add_argument('--gnn_num_heads', type=int, default=2)
    parser.add_argument('--regressor_hidden_dim', type=int, default=64)
    parser.add_argument('--regressor_num_hidden_layers', type=int, default=1)
    
    args = parser.parse_args()
    predict(args)
