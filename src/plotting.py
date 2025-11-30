"""
Publication-quality plotting utilities for model performance visualization
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import seaborn as sns
import os


# Set publication-quality style
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Helvetica'],
    'axes.linewidth': 1.2,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18,
    'lines.linewidth': 2,
    'lines.markersize': 8,
    'grid.alpha': 0.3,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})

# Color palette for publication
COLORS = {
    'train': '#2E86AB',  # Blue
    'test': '#A23B72',   # Purple
    'prediction': '#F18F01',  # Orange
    'actual': '#C73E1D',  # Red
    'diagonal': '#000000'  # Black
}


def plot_training_curves(train_metrics, test_metrics=None, save_path=None, figsize=(15, 5)):
    """
    Plot training curves with publication quality.
    
    Args:
        train_metrics: dict with keys 'loss', 'r2', 'rmse', 'mae' (lists)
        test_metrics: dict with keys 'loss', 'r2', 'rmse', 'mae' (lists, optional)
        save_path: Path to save figure (optional)
        figsize: Figure size tuple
    """
    epochs = np.arange(1, len(train_metrics['loss']) + 1)
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Loss plot
    axes[0].plot(epochs, train_metrics['loss'], color=COLORS['train'], 
                label='Train', marker='o', markersize=4, linewidth=2)
    if test_metrics:
        axes[0].plot(epochs, test_metrics['loss'], color=COLORS['test'], 
                    label='Test', marker='s', markersize=4, linewidth=2)
    axes[0].set_xlabel('Epoch', fontweight='bold')
    axes[0].set_ylabel('Loss', fontweight='bold')
    axes[0].set_title('Training Loss', fontweight='bold')
    axes[0].grid(True, alpha=0.3, linestyle='--')
    axes[0].legend(frameon=True, fancybox=True, shadow=True)
    
    # R² plot
    axes[1].plot(epochs, train_metrics['r2'], color=COLORS['train'], 
                label='Train', marker='o', markersize=4, linewidth=2)
    if test_metrics:
        axes[1].plot(epochs, test_metrics['r2'], color=COLORS['test'], 
                    label='Test', marker='s', markersize=4, linewidth=2)
    axes[1].set_xlabel('Epoch', fontweight='bold')
    axes[1].set_ylabel('R² Score', fontweight='bold')
    axes[1].set_title('R² Score', fontweight='bold')
    axes[1].grid(True, alpha=0.3, linestyle='--')
    axes[1].legend(frameon=True, fancybox=True, shadow=True)
    
    # RMSE plot
    axes[2].plot(epochs, train_metrics['rmse'], color=COLORS['train'], 
                label='Train', marker='o', markersize=4, linewidth=2)
    if test_metrics:
        axes[2].plot(epochs, test_metrics['rmse'], color=COLORS['test'], 
                    label='Test', marker='s', markersize=4, linewidth=2)
    axes[2].set_xlabel('Epoch', fontweight='bold')
    axes[2].set_ylabel('RMSE', fontweight='bold')
    axes[2].set_title('RMSE', fontweight='bold')
    axes[2].grid(True, alpha=0.3, linestyle='--')
    axes[2].legend(frameon=True, fancybox=True, shadow=True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to {save_path}")
    
    return fig


def plot_prediction_scatter(y_true, y_pred, split='Test', save_path=None, figsize=(8, 8)):
    """
    Plot predicted vs actual values scatter plot with publication quality.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        split: Dataset split name (e.g., 'Train', 'Test')
        save_path: Path to save figure (optional)
        figsize: Figure size tuple
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calculate metrics
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Scatter plot with transparency
    ax.scatter(y_true, y_pred, alpha=0.6, s=50, color=COLORS['prediction'], 
               edgecolors='white', linewidth=0.5, zorder=2)
    
    # Perfect prediction line
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    ax.plot([min_val, max_val], [min_val, max_val], 
            color=COLORS['diagonal'], linestyle='--', linewidth=2, 
            label='Perfect Prediction', zorder=1)
    
    # Add metrics text
    textstr = f'R² = {r2:.3f}\nRMSE = {rmse:.3f}\nMAE = {mae:.3f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)
    
    ax.set_xlabel(f'Actual log(IC50) ({split})', fontweight='bold')
    ax.set_ylabel(f'Predicted log(IC50) ({split})', fontweight='bold')
    ax.set_title(f'Predicted vs Actual log(IC50) - {split} Set', fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(frameon=True, fancybox=True, shadow=True)
    
    # Equal aspect ratio
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Prediction scatter plot saved to {save_path}")
    
    return fig


def plot_residuals(y_true, y_pred, split='Test', save_path=None, figsize=(10, 5)):
    """
    Plot residual analysis with publication quality.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        split: Dataset split name
        save_path: Path to save figure (optional)
        figsize: Figure size tuple
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Residuals vs predicted
    axes[0].scatter(y_pred, residuals, alpha=0.6, s=50, color=COLORS['prediction'],
                   edgecolors='white', linewidth=0.5)
    axes[0].axhline(y=0, color=COLORS['diagonal'], linestyle='--', linewidth=2)
    axes[0].set_xlabel('Predicted log(IC50)', fontweight='bold')
    axes[0].set_ylabel('Residuals', fontweight='bold')
    axes[0].set_title(f'Residuals vs Predicted ({split})', fontweight='bold')
    axes[0].grid(True, alpha=0.3, linestyle='--')
    
    # Residuals histogram
    axes[1].hist(residuals, bins=30, color=COLORS['prediction'], alpha=0.7, 
               edgecolor='black', linewidth=1.2)
    axes[1].axvline(x=0, color=COLORS['diagonal'], linestyle='--', linewidth=2)
    axes[1].set_xlabel('Residuals', fontweight='bold')
    axes[1].set_ylabel('Frequency', fontweight='bold')
    axes[1].set_title(f'Residual Distribution ({split})', fontweight='bold')
    axes[1].grid(True, alpha=0.3, linestyle='--', axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Residual plots saved to {save_path}")
    
    return fig


def plot_comparison(y_true_train, y_pred_train, y_true_test, y_pred_test, 
                   save_path=None, figsize=(12, 5)):
    """
    Plot side-by-side comparison of train and test predictions.
    
    Args:
        y_true_train: True values for training set
        y_pred_train: Predicted values for training set
        y_true_test: True values for test set
        y_pred_test: Predicted values for test set
        save_path: Path to save figure (optional)
        figsize: Figure size tuple
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    for idx, (y_true, y_pred, split, color) in enumerate([
        (y_true_train, y_pred_train, 'Train', COLORS['train']),
        (y_true_test, y_pred_test, 'Test', COLORS['test'])
    ]):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
        axes[idx].scatter(y_true, y_pred, alpha=0.6, s=50, color=color,
                         edgecolors='white', linewidth=0.5, zorder=2)
        
        min_val = min(np.min(y_true), np.min(y_pred))
        max_val = max(np.max(y_true), np.max(y_pred))
        axes[idx].plot([min_val, max_val], [min_val, max_val],
                      color=COLORS['diagonal'], linestyle='--', linewidth=2, zorder=1)
        
        textstr = f'R² = {r2:.3f}\nRMSE = {rmse:.3f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        axes[idx].text(0.05, 0.95, textstr, transform=axes[idx].transAxes, fontsize=11,
                       verticalalignment='top', bbox=props)
        
        axes[idx].set_xlabel(f'Actual log(IC50)', fontweight='bold')
        axes[idx].set_ylabel(f'Predicted log(IC50)', fontweight='bold')
        axes[idx].set_title(f'{split} Set', fontweight='bold')
        axes[idx].grid(True, alpha=0.3, linestyle='--')
        axes[idx].set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to {save_path}")
    
    return fig


def plot_per_dataset_performance(per_dataset_metrics, save_dir='results/plots'):
    """
    Generate comprehensive per-dataset performance visualizations.
    
    Args:
        per_dataset_metrics: Dictionary with per-dataset metrics
        save_dir: Directory to save plots
    """
    datasets = sorted(per_dataset_metrics.keys())
    r2_scores = [per_dataset_metrics[ds]['r2'] for ds in datasets]
    rmse_scores = [per_dataset_metrics[ds]['rmse'] for ds in datasets]
    mae_scores = [per_dataset_metrics[ds]['mae'] for ds in datasets]
    n_samples = [per_dataset_metrics[ds]['n_samples'] for ds in datasets]
    
    # Set publication-quality style
    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'sans-serif',
        'axes.linewidth': 1.2,
        'figure.dpi': 300
    })
    
    # 1. Bar chart comparison of metrics
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(datasets)))
    
    # R² scores
    bars1 = axes[0].bar(range(len(datasets)), r2_scores, color=colors, edgecolor='black', linewidth=1.2)
    axes[0].set_xlabel('Dataset', fontweight='bold', fontsize=14)
    axes[0].set_ylabel('R² Score', fontweight='bold', fontsize=14)
    axes[0].set_title('R² Score by Dataset', fontweight='bold', fontsize=16)
    axes[0].set_xticks(range(len(datasets)))
    axes[0].set_xticklabels(datasets, rotation=45, ha='right')
    axes[0].grid(True, alpha=0.3, axis='y')
    axes[0].axhline(y=np.mean(r2_scores), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(r2_scores):.3f}')
    axes[0].legend()
    
    # Add value labels on bars
    for bar, score in zip(bars1, r2_scores):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{score:.3f}', ha='center', va='bottom', fontsize=10)
    
    # RMSE scores
    bars2 = axes[1].bar(range(len(datasets)), rmse_scores, color=colors, edgecolor='black', linewidth=1.2)
    axes[1].set_xlabel('Dataset', fontweight='bold', fontsize=14)
    axes[1].set_ylabel('RMSE', fontweight='bold', fontsize=14)
    axes[1].set_title('RMSE by Dataset', fontweight='bold', fontsize=16)
    axes[1].set_xticks(range(len(datasets)))
    axes[1].set_xticklabels(datasets, rotation=45, ha='right')
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].axhline(y=np.mean(rmse_scores), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(rmse_scores):.3f}')
    axes[1].legend()
    
    for bar, score in zip(bars2, rmse_scores):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{score:.3f}', ha='center', va='bottom', fontsize=10)
    
    # MAE scores
    bars3 = axes[2].bar(range(len(datasets)), mae_scores, color=colors, edgecolor='black', linewidth=1.2)
    axes[2].set_xlabel('Dataset', fontweight='bold', fontsize=14)
    axes[2].set_ylabel('MAE', fontweight='bold', fontsize=14)
    axes[2].set_title('MAE by Dataset', fontweight='bold', fontsize=16)
    axes[2].set_xticks(range(len(datasets)))
    axes[2].set_xticklabels(datasets, rotation=45, ha='right')
    axes[2].grid(True, alpha=0.3, axis='y')
    axes[2].axhline(y=np.mean(mae_scores), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(mae_scores):.3f}')
    axes[2].legend()
    
    for bar, score in zip(bars3, mae_scores):
        height = bar.get_height()
        axes[2].text(bar.get_x() + bar.get_width()/2., height,
                    f'{score:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'per_dataset_metrics.png'), dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: per_dataset_metrics.png")
    plt.close()
    
    # 2. R² vs Sample Size scatter plot
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(n_samples, r2_scores, c=r2_scores, cmap='RdYlGn', 
                        s=200, edgecolor='black', linewidth=1.5, alpha=0.8)
    
    # Add dataset labels
    for i, ds in enumerate(datasets):
        ax.annotate(ds, (n_samples[i], r2_scores[i]), 
                   xytext=(5, 5), textcoords='offset points', fontsize=10)
    
    ax.set_xlabel('Number of Test Samples', fontweight='bold', fontsize=14)
    ax.set_ylabel('R² Score', fontweight='bold', fontsize=14)
    ax.set_title('R² Score vs Dataset Size', fontweight='bold', fontsize=16)
    ax.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('R² Score', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'r2_vs_sample_size.png'), dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: r2_vs_sample_size.png")
    plt.close()
    
    # 3. Heatmap of metrics
    fig, ax = plt.subplots(figsize=(10, len(datasets) * 0.5 + 2))
    
    metrics_data = np.array([r2_scores, rmse_scores, mae_scores]).T
    
    im = ax.imshow(metrics_data, cmap='RdYlGn', aspect='auto')
    
    # Set ticks
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['R²', 'RMSE', 'MAE'], fontweight='bold')
    ax.set_yticks(range(len(datasets)))
    ax.set_yticklabels(datasets)
    
    # Add values to heatmap
    for i in range(len(datasets)):
        for j in range(3):
            text = ax.text(j, i, f'{metrics_data[i, j]:.3f}',
                          ha="center", va="center", color="black", fontsize=10)
    
    ax.set_title('Per-Dataset Performance Heatmap', fontweight='bold', fontsize=16)
    plt.colorbar(im, ax=ax, label='Metric Value')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'metrics_heatmap.png'), dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: metrics_heatmap.png")
    plt.close()
    
    # 4. Individual scatter plots for each dataset
    n_datasets = len(datasets)
    n_cols = 3
    n_rows = (n_datasets + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    for idx, ds in enumerate(datasets):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        metrics = per_dataset_metrics[ds]
        y_true = metrics['y_true']
        y_pred = metrics['y_pred']
        r2 = metrics['r2']
        rmse = metrics['rmse']
        
        # Scatter plot
        ax.scatter(y_true, y_pred, alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
        
        # Add diagonal line
        min_val = min(min(y_true), min(y_pred))
        max_val = max(max(y_true), max(y_pred))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction')
        
        # Add metrics text
        textstr = f'R² = {r2:.3f}\nRMSE = {rmse:.3f}\nn = {len(y_true)}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=props)
        
        ax.set_xlabel('Actual log(IC50)', fontweight='bold')
        ax.set_ylabel('Predicted log(IC50)', fontweight='bold')
        ax.set_title(ds, fontweight='bold', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
    
    # Hide empty subplots
    for idx in range(n_datasets, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'per_dataset_predictions_all.png'), dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: per_dataset_predictions_all.png")
    plt.close()
    
    print(f"All per-dataset plots saved to {save_dir}")
