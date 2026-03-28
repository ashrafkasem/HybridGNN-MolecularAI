"""
Combined training script with Hybrid Model (GNN + Descriptors)
Train on multiple ChEMBL datasets together with molecular descriptors
"""
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split, ConcatDataset
from tqdm import tqdm
import argparse
import os
import json
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import sys
import glob
from bisect import bisect_right

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import HybridGNNModel
from utils import smiles_to_data, ATOM_COMBINED_TYPES, BOND_TYPES
from descriptors import calculate_all_descriptors
from plotting import plot_training_curves, plot_prediction_scatter, plot_comparison, plot_per_dataset_performance
from dataset import IC50Dataset_WithDescriptors
import matplotlib.pyplot as plt
import seaborn as sns


def train_epoch(model, train_loader, optimizer, loss_fn, device):
    """Train for one epoch with hybrid model."""
    model.train()
    y_true, y_pred = [], []
    total_loss = 0
    
    for batch in tqdm(train_loader, desc="Training"):
        # Extract and reshape descriptors (PyTorch Geometric concatenates them)
        batch_size = batch.num_graphs
        descriptor_size = batch.descriptor_size[0].item() if hasattr(batch.descriptor_size, '__getitem__') else 544
        descriptors = batch.descriptors.clone().view(batch_size, descriptor_size).to(device)
        
        # Remove descriptor attributes before moving batch
        del batch.descriptors
        del batch.descriptor_size
        
        batch = batch.to(device)
        optimizer.zero_grad()
        
        # Forward pass with descriptors
        pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.N, descriptors).squeeze()
        loss = loss_fn(pred, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        y_true += batch.y.view(-1).tolist()
        y_pred += pred.view(-1).tolist()
    
    avg_loss = total_loss / len(train_loader)
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    return avg_loss, r2, rmse, mae, y_true, y_pred


def evaluate(model, data_loader, device, desc="Evaluating", return_per_dataset=False):
    """Evaluate hybrid model on a dataset."""
    model.eval()
    y_true, y_pred = [], []
    dataset_results = {}  # Store results per dataset
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc=desc):
            # Extract and reshape descriptors
            batch_size = batch.num_graphs
            descriptor_size = batch.descriptor_size[0].item() if hasattr(batch.descriptor_size, '__getitem__') else 544
            descriptors = batch.descriptors.clone().view(batch_size, descriptor_size).to(device)
            
            # Remove descriptor attributes
            del batch.descriptors
            del batch.descriptor_size
            
            batch = batch.to(device)
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.N, descriptors).squeeze()
            
            batch_y_true = batch.y.view(-1).tolist()
            batch_y_pred = pred.view(-1).tolist()
            
            y_true += batch_y_true
            y_pred += batch_y_pred
            
            # Track per-dataset results if dataset names are available
            if return_per_dataset and hasattr(batch, 'dataset_name'):
                for i, dataset_name in enumerate(batch.dataset_name):
                    if dataset_name not in dataset_results:
                        dataset_results[dataset_name] = {'y_true': [], 'y_pred': []}
                    dataset_results[dataset_name]['y_true'].append(batch_y_true[i])
                    dataset_results[dataset_name]['y_pred'].append(batch_y_pred[i])
    
    # Calculate overall metrics
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    if return_per_dataset and dataset_results:
        # Calculate per-dataset metrics
        per_dataset_metrics = {}
        for dataset_name, data in dataset_results.items():
            if len(data['y_true']) > 1:  # Need at least 2 samples for R²
                ds_r2 = r2_score(data['y_true'], data['y_pred'])
                ds_rmse = np.sqrt(mean_squared_error(data['y_true'], data['y_pred']))
                ds_mae = mean_absolute_error(data['y_true'], data['y_pred'])
                per_dataset_metrics[dataset_name] = {
                    'r2': ds_r2,
                    'rmse': ds_rmse,
                    'mae': ds_mae,
                    'n_samples': len(data['y_true']),
                    'y_true': data['y_true'],  # Keep for plotting
                    'y_pred': data['y_pred']   # Keep for plotting
                }
        return r2, rmse, mae, y_true, y_pred, per_dataset_metrics
    
    return r2, rmse, mae, y_true, y_pred


def compute_train_descriptor_stats(train_subset, concat_dataset, eps=1e-8):
    """Compute descriptor normalization stats from train split only."""
    train_indices = train_subset.indices
    cumulative_sizes = concat_dataset.cumulative_sizes
    source_datasets = concat_dataset.datasets

    train_descriptors = []
    for global_idx in train_indices:
        dataset_idx = bisect_right(cumulative_sizes, global_idx)
        prev_cum_size = 0 if dataset_idx == 0 else cumulative_sizes[dataset_idx - 1]
        sample_idx = global_idx - prev_cum_size
        train_descriptors.append(source_datasets[dataset_idx].descriptors[sample_idx])

    train_descriptors = torch.stack(train_descriptors)
    desc_mean = train_descriptors.mean(dim=0, keepdim=True)
    desc_std = train_descriptors.std(dim=0, keepdim=True) + eps
    return desc_mean, desc_std


def apply_descriptor_normalization(datasets, desc_mean, desc_std):
    """Apply precomputed descriptor normalization to a list of datasets."""
    for dataset in datasets:
        dataset.descriptors = (dataset.descriptors - desc_mean) / desc_std
        dataset.desc_mean = desc_mean
        dataset.desc_std = desc_std


def train_combined(args):
    """Main training function for combined datasets with hybrid model."""
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")
    print(f"Using {'fingerprints + descriptors' if args.use_fingerprints else 'descriptors only'}")
    print(f"Fingerprint size: {args.fp_bits if args.use_fingerprints else 0} bits")
    
    # Load all datasets
    print(f"\nLoading datasets from {args.data_dir}...")
    dataset_files = []
    
    if args.datasets:
        # Use specified datasets
        dataset_files = [os.path.join(args.data_dir, f"{ds}_IC50.csv") for ds in args.datasets]
    else:
        # Use all CSV files in directory, excluding certain files
        all_files = glob.glob(os.path.join(args.data_dir, "*_IC50.csv"))
        # Exclude training files and generic IC50.csv (only keep CHEMBL target-specific files)
        exclude_patterns = ['train', 'test']
        dataset_files = []
        for f in all_files:
            basename = os.path.basename(f)
            # Only include files that start with CHEMBL followed by digits
            if basename.startswith('CHEMBL') and any(c.isdigit() for c in basename.split('_')[0]):
                if not any(pattern.lower() in basename.lower() for pattern in exclude_patterns):
                    dataset_files.append(f)
            # Also exclude standalone IC50.csv
            elif basename == 'IC50.csv':
                continue
    
    if not dataset_files:
        print(f"Error: No dataset files found in {args.data_dir}")
        return
    
    print(f"Found {len(dataset_files)} datasets:")
    for f in sorted(dataset_files):
        print(f"  - {os.path.basename(f)}")
    datasets = []
    dataset_names = []
    
    for csv_file in sorted(dataset_files):
        if os.path.exists(csv_file):
            dataset_name = os.path.basename(csv_file).replace('_IC50.csv', '')
            try:
                dataset = IC50Dataset_WithDescriptors(
                    csv_file, 
                    oversample=args.oversample, 
                    dataset_name=dataset_name,
                    use_fingerprints=args.use_fingerprints,
                    fp_bits=args.fp_bits,
                    cache_dir=args.cache_dir
                )
                dataset_names.append(dataset_name)
                datasets.append(dataset)
            except (ValueError, KeyError) as e:
                print(f"  ⚠️  Skipping {dataset_name}: {str(e)}")
    
    if not datasets:
        print("Error: No valid datasets loaded")
        return
    
    # Combine all datasets
    combined_dataset = ConcatDataset(datasets)
    total_samples = len(combined_dataset)
    print(f"\nCombined dataset: {total_samples} total samples from {len(datasets)} datasets")
    
    # Split combined dataset
    train_size = int((1 - args.test_size) * total_samples)
    test_size = total_samples - train_size
    train_dataset, test_dataset = random_split(
        combined_dataset, 
        [train_size, test_size],
        generator=torch.Generator().manual_seed(args.random_state)
    )

    # Normalize descriptors using train split statistics only.
    desc_mean, desc_std = compute_train_descriptor_stats(train_dataset, combined_dataset)
    apply_descriptor_normalization(datasets, desc_mean, desc_std)

    

    print(f"Train set: {len(train_dataset)} samples ({len(train_dataset)/total_samples*100:.1f}%)")
    print(f"Test set: {len(test_dataset)} samples ({len(test_dataset)/total_samples*100:.1f}%)")
    
    # Create data loaders (descriptors will be reshaped in training loop)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # Model configuration
    node_vocab_size = len(ATOM_COMBINED_TYPES)
    num_node_features = 11  # Updated: added chirality, ring size, implicit valence
    edge_feature_size = len(BOND_TYPES) + 2
    descriptor_size = (args.fp_bits if args.use_fingerprints else 0) + 32  # fingerprints + basic descriptors
    
    # Create hybrid model
    print(f"\nCreating hybrid model...")
    print(f"  GNN hidden dim: {args.gnn_hidden_dim}")
    print(f"  Descriptor size: {descriptor_size}")
    
    model = HybridGNNModel(
        node_vocab_size=node_vocab_size,
        num_node_features=num_node_features,
        edge_feature_size=edge_feature_size,
        descriptor_size=descriptor_size,
        gnn_hidden_dim=args.gnn_hidden_dim,
        gnn_num_layers=args.gnn_num_layers,
        gnn_dropout=args.gnn_dropout,
        gnn_num_heads=args.gnn_num_heads,
        regressor_hidden_dim=args.regressor_hidden_dim,
        regressor_num_hidden_layers=args.regressor_num_hidden_layers,
        regressor_dropout=args.regressor_dropout
    ).to(device)
    
    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.SmoothL1Loss()
    
    # Training metrics
    train_metrics = {'loss': [], 'r2': [], 'rmse': [], 'mae': []}
    test_metrics = {'loss': [], 'r2': [], 'rmse': [], 'mae': []}
    
    # Training loop
    print("\nStarting training...")
    best_test_r2 = float('-inf')
    
    for epoch in range(args.epochs):
        # Train
        train_loss, train_r2, train_rmse, train_mae, train_y_true, train_y_pred = train_epoch(
            model, train_loader, optimizer, loss_fn, device
        )
        
        # Evaluate on test set with per-dataset breakdown
        eval_results = evaluate(
            model, test_loader, device, desc="Evaluating", return_per_dataset=True
        )
        
        if len(eval_results) == 6:
            test_r2, test_rmse, test_mae, test_y_true, test_y_pred, per_dataset_metrics = eval_results
        else:
            test_r2, test_rmse, test_mae, test_y_true, test_y_pred = eval_results
            per_dataset_metrics = {}
        
        # Calculate test loss
        test_loss = loss_fn(torch.tensor(test_y_pred), torch.tensor(test_y_true)).item()
        
        # Store metrics
        train_metrics['loss'].append(train_loss)
        train_metrics['r2'].append(train_r2)
        train_metrics['rmse'].append(train_rmse)
        train_metrics['mae'].append(train_mae)
        
        test_metrics['loss'].append(test_loss)
        test_metrics['r2'].append(test_r2)
        test_metrics['rmse'].append(test_rmse)
        test_metrics['mae'].append(test_mae)
        
        print(f"Epoch {epoch + 1:>3}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} | Train R²: {train_r2:.4f} | Train RMSE: {train_rmse:.4f} | "
              f"Test R²: {test_r2:.4f} | Test RMSE: {test_rmse:.4f}")
        
        # Print per-dataset performance every 10 epochs
        if per_dataset_metrics and (epoch + 1) % 10 == 0:
            print(f"  Per-dataset Test Performance:")
            for ds_name, metrics in sorted(per_dataset_metrics.items()):
                print(f"    {ds_name}: R²={metrics['r2']:.3f}, RMSE={metrics['rmse']:.3f}, n={metrics['n_samples']}")
        
        # Save best model
        if test_r2 > best_test_r2:
            best_test_r2 = test_r2
            os.makedirs(args.model_dir, exist_ok=True)
            model_path = os.path.join(args.model_dir, args.model_name)
            
            # Save model with descriptor normalization stats
            torch.save({
                'model_state_dict': model.state_dict(),
                'use_fingerprints': args.use_fingerprints,
                'fp_bits': args.fp_bits,
                'descriptor_size': descriptor_size,
                'desc_mean': desc_mean.cpu(),
                'desc_std': desc_std.cpu()
            }, model_path)
            print(f"  ✓ Saved best model (Test R²: {test_r2:.4f})")
    
    print(f"\nTraining complete! Best Test R²: {best_test_r2:.4f}")
    
    # Final evaluation with per-dataset breakdown
    print("\n" + "=" * 70)
    print("FINAL TEST SET PERFORMANCE BY DATASET (HYBRID MODEL)")
    print("=" * 70)
    
    final_eval = evaluate(model, test_loader, device, desc="Final Evaluation", return_per_dataset=True)
    if len(final_eval) == 6:
        final_test_r2, final_test_rmse, final_test_mae, _, _, final_per_dataset = final_eval
        
        print(f"\nOverall Combined Test Performance:")
        print(f"  R²:   {final_test_r2:.4f}")
        print(f"  RMSE: {final_test_rmse:.4f}")
        print(f"  MAE:  {final_test_mae:.4f}")
        
        if final_per_dataset:
            print(f"\nPer-Dataset Test Performance:")
            print(f"{'Dataset':<15} {'N Samples':<12} {'R²':<10} {'RMSE':<10} {'MAE':<10}")
            print("-" * 60)
            for ds_name, metrics in sorted(final_per_dataset.items()):
                print(f"{ds_name:<15} {metrics['n_samples']:<12} "
                      f"{metrics['r2']:<10.4f} {metrics['rmse']:<10.4f} {metrics['mae']:<10.4f}")
    print("=" * 70)
    
    # Create output directories
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(os.path.join(args.results_dir, 'plots'), exist_ok=True)
    os.makedirs(os.path.join(args.results_dir, 'logs'), exist_ok=True)
    
    # Generate plots
    print("\nGenerating plots...")
    
    # Training curves
    plot_training_curves(
        train_metrics, test_metrics,
        save_path=os.path.join(args.results_dir, 'plots', 'training_curves_combined_hybrid.png')
    )
    
    # Comparison plot
    plot_comparison(
        train_y_true, train_y_pred, test_y_true, test_y_pred,
        save_path=os.path.join(args.results_dir, 'plots', 'train_test_comparison_combined_hybrid.png')
    )
    
    # Per-dataset performance plots
    if final_per_dataset:
        print("\nGenerating per-dataset performance plots...")
        plot_per_dataset_performance(final_per_dataset, save_dir=os.path.join(args.results_dir, 'plots'))
    
    # Save metrics
    metrics = {
        'model_type': 'hybrid_gnn_descriptors_combined',
        'descriptor_info': {
            'use_fingerprints': args.use_fingerprints,
            'fp_bits': args.fp_bits,
            'total_descriptor_size': descriptor_size
        },
        'datasets': dataset_names,
        'train': {
            'final_r2': train_r2,
            'final_rmse': train_rmse,
            'final_mae': train_mae,
            'final_loss': train_loss
        },
        'test': {
            'overall': {
                'final_r2': final_test_r2,
                'final_rmse': final_test_rmse,
                'final_mae': final_test_mae,
                'best_r2': best_test_r2
            },
            'per_dataset': {
                ds_name: {
                    'r2': metrics['r2'],
                    'rmse': metrics['rmse'],
                    'mae': metrics['mae'],
                    'n_samples': metrics['n_samples']
                }
                for ds_name, metrics in final_per_dataset.items()
            } if final_per_dataset else {}
        },
        'training_history': {
            'train': train_metrics,
            'test': test_metrics
        }
    }
    
    metrics_file = os.path.join(args.results_dir, 'logs', 'training_metrics_combined_hybrid.json')
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nMetrics saved to {metrics_file}")
    print(f"Model saved to {os.path.join(args.model_dir, args.model_name)}")
    print(f"All plots saved to {os.path.join(args.results_dir, 'plots')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Hybrid GNN + Descriptors on combined ChEMBL datasets')
    parser.add_argument('--data_dir', type=str, default='data/raw',
                       help='Directory containing ChEMBL IC50 CSV files (default: data/raw)')
    parser.add_argument('--cache_dir', type=str, default='data/processed/descriptors',
                       help='Directory to store cached descriptor tensors (default: data/processed/descriptors)')
    parser.add_argument('--datasets', type=str, nargs='+', default=None,
                       help='Specific datasets to use (default: all CHEMBL*.csv files)')
    parser.add_argument('--model_dir', type=str, default='models/saved',
                       help='Directory to save models (default: models/saved)')
    parser.add_argument('--model_name', type=str, default='combined_model_hybrid.pt',
                       help='Model filename (default: combined_model_hybrid.pt)')
    parser.add_argument('--results_dir', type=str, default='results',
                       help='Directory to save results (default: results)')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Test set proportion (default: 0.2)')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size (default: 32)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs (default: 100)')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate (default: 1e-3)')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay (default: 1e-5)')
    
    # Model architecture
    parser.add_argument('--gnn_hidden_dim', type=int, default=128,
                       help='GNN hidden dimension (default: 128)')
    parser.add_argument('--gnn_num_layers', type=int, default=2,
                       help='Number of GNN layers (default: 2)')
    parser.add_argument('--gnn_dropout', type=float, default=0.2,
                       help='GNN dropout (default: 0.2)')
    parser.add_argument('--gnn_num_heads', type=int, default=2,
                       help='Number of attention heads (default: 2)')
    parser.add_argument('--regressor_hidden_dim', type=int, default=64,
                       help='Regressor hidden dimension (default: 64)')
    parser.add_argument('--regressor_num_hidden_layers', type=int, default=1,
                       help='Number of regressor hidden layers (default: 1)')
    parser.add_argument('--regressor_dropout', type=float, default=0.2,
                       help='Regressor dropout (default: 0.2)')
    
    # Descriptor options
    parser.add_argument('--use_fingerprints', action='store_true', default=True,
                       help='Use Morgan fingerprints (default: True)')
    parser.add_argument('--no_fingerprints', action='store_false', dest='use_fingerprints',
                       help='Disable Morgan fingerprints')
    parser.add_argument('--fp_bits', type=int, default=512,
                       help='Number of fingerprint bits (default: 512)')
    
    parser.add_argument('--oversample', action='store_true',
                       help='Oversample training data')
    parser.add_argument('--cpu', action='store_true',
                       help='Force CPU usage')
    
    args = parser.parse_args()
    train_combined(args)

