"""
Evaluation script for Hybrid GNN + Descriptors Model
"""
import torch
import pandas as pd
import numpy as np
import argparse
import os
import sys
import glob
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import HybridGNNModel
from dataset import IC50Dataset_WithDescriptors
from utils import ATOM_COMBINED_TYPES, BOND_TYPES
from plotting import plot_prediction_scatter, plot_residuals, plot_per_dataset_performance


def evaluate_model(args):
    """
    Evaluate trained hybrid model on new datasets.
    """
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")
    
    # Load model checkpoint
    if not os.path.exists(args.model_path):
        print(f"Error: Model not found at {args.model_path}")
        return
        
    print(f"Loading model from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location=device)
    
    # Extract config from checkpoint
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
    
    print(f"Model config: Fingerprints={use_fingerprints}, Bits={fp_bits}")
    
    # Initialize model structure (assuming default architecture if not in args)
    # Ideally, these should be saved in checkpoint, but for now we use defaults/args
    # or we could save args in checkpoint during training.
    # For now, we'll rely on the standard architecture defined in train_combined_hybrid.py
    
    node_vocab_size = len(ATOM_COMBINED_TYPES)
    num_node_features = 11
    edge_feature_size = len(BOND_TYPES) + 2
    
    model = HybridGNNModel(
        node_vocab_size=node_vocab_size,
        num_node_features=num_node_features,
        edge_feature_size=edge_feature_size,
        descriptor_size=descriptor_size,
        gnn_hidden_dim=args.gnn_hidden_dim,
        gnn_num_layers=args.gnn_num_layers,
        gnn_num_heads=args.gnn_num_heads,
        regressor_hidden_dim=args.regressor_hidden_dim,
        regressor_num_hidden_layers=args.regressor_num_hidden_layers
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Identify datasets
    dataset_files = []
    if os.path.isdir(args.data_path):
        dataset_files = glob.glob(os.path.join(args.data_path, "*.csv"))
    elif os.path.isfile(args.data_path):
        dataset_files = [args.data_path]
    else:
        print(f"Error: Data path not found: {args.data_path}")
        return
        
    print(f"Found {len(dataset_files)} datasets to evaluate")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    if args.generate_plots:
        os.makedirs(os.path.join(args.output_dir, 'plots'), exist_ok=True)
    
    all_results = []
    per_dataset_metrics = {}
    
    for csv_file in dataset_files:
        dataset_name = os.path.basename(csv_file).replace('.csv', '').replace('_IC50', '')
        print(f"\nProcessing {dataset_name}...")
        
        try:
            dataset = IC50Dataset_WithDescriptors(
                csv_file, 
                dataset_name=dataset_name,
                use_fingerprints=use_fingerprints,
                fp_bits=fp_bits,
                cache_dir=args.cache_dir
            )
        except Exception as e:
            print(f"  ⚠️  Skipping {dataset_name}: {str(e)}")
            continue
            
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        
        y_true = []
        y_pred = []
        smiles_list = dataset.df['smiles'].tolist()
        
        with torch.no_grad():
            for batch in tqdm(loader, desc="  Inference"):
                # Handle descriptors
                batch_size = batch.num_graphs
                desc_size = batch.descriptor_size[0].item() if hasattr(batch.descriptor_size, '__getitem__') else descriptor_size
                descriptors = batch.descriptors.clone().view(batch_size, desc_size).to(device)
                if desc_mean is not None and desc_std is not None:
                    descriptors = (descriptors - desc_mean) / desc_std
                
                del batch.descriptors
                del batch.descriptor_size
                
                batch = batch.to(device)
                pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.N, descriptors).squeeze()
                
                y_true.extend(batch.y.view(-1).tolist())
                y_pred.extend(pred.view(-1).tolist())
        
        # Calculate metrics
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        
        print(f"  Results: R²={r2:.3f}, RMSE={rmse:.3f}, MAE={mae:.3f}")
        
        per_dataset_metrics[dataset_name] = {
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'n_samples': len(y_true),
            'y_true': y_true,
            'y_pred': y_pred
        }
        
        # Save predictions
        results_df = pd.DataFrame({
            'SMILES': smiles_list[:len(y_true)], # Ensure length match if any dropped
            'Actual_Log_IC50': y_true,
            'Predicted_Log_IC50': y_pred,
            'Predicted_IC50': [10**x for x in y_pred]
        })
        
        output_file = os.path.join(args.output_dir, f"{dataset_name}_predictions.csv")
        results_df.to_csv(output_file, index=False)
        print(f"  Saved predictions to {output_file}")
        
        # Generate individual plots
        if args.generate_plots:
            plot_path = os.path.join(args.output_dir, 'plots', f"{dataset_name}_scatter.png")
            plot_prediction_scatter(y_true, y_pred, split=dataset_name, save_path=plot_path)
            
            resid_path = os.path.join(args.output_dir, 'plots', f"{dataset_name}_residuals.png")
            plot_residuals(y_true, y_pred, split=dataset_name, save_path=resid_path)

    # Generate summary plots if multiple datasets
    if args.generate_plots and len(per_dataset_metrics) > 0:
        print("\nGenerating summary plots...")
        plot_per_dataset_performance(per_dataset_metrics, save_dir=os.path.join(args.output_dir, 'plots'))

    print(f"\nEvaluation complete. Results saved to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate Hybrid GNN Model')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint (.pt)')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to CSV file or directory of CSV files')
    parser.add_argument('--cache_dir', type=str, default='data/processed/descriptors',
                       help='Directory to store/load cached descriptor tensors (default: data/processed/descriptors)')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='Directory to save results')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--generate_plots', action='store_true',
                       help='Generate performance plots')
    parser.add_argument('--cpu', action='store_true',
                       help='Force CPU usage')
    
    # Model architecture args (should match training)
    parser.add_argument('--gnn_hidden_dim', type=int, default=128)
    parser.add_argument('--gnn_num_layers', type=int, default=2)
    parser.add_argument('--gnn_num_heads', type=int, default=2)
    parser.add_argument('--regressor_hidden_dim', type=int, default=64)
    parser.add_argument('--regressor_num_hidden_layers', type=int, default=1)
    
    args = parser.parse_args()
    evaluate_model(args)
