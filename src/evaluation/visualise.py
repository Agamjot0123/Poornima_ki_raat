"""
Visualisation — Side-by-side 3D renderings and metric plots.

Renders predicted vs ground-truth meshes and generates 
error visualisation figures.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pathlib import Path
from typing import Optional


def render_mesh_matplotlib(vertices: np.ndarray, faces: np.ndarray,
                           ax=None, color='steelblue', alpha=0.7,
                           title: str = '') -> None:
    """Render a mesh using matplotlib 3D.
    
    Args:
        vertices: (N, 3) vertex positions
        faces: (M, 3) face indices
        ax: matplotlib 3D axis (created if None)
        color: Face color
        alpha: Transparency
        title: Plot title
    """
    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
    
    # Create face collections
    face_verts = vertices[faces]
    collection = Poly3DCollection(face_verts, alpha=alpha, 
                                   facecolor=color, edgecolor='gray',
                                   linewidth=0.1)
    ax.add_collection3d(collection)
    
    # Set axis limits
    max_range = np.max(np.abs(vertices)) * 1.2
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(-max_range, max_range)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)


def render_side_by_side(pred_mesh, gt_mesh, 
                        save_path: Optional[str] = None,
                        title: str = 'Reconstruction vs Ground Truth') -> Figure:
    """Render predicted and ground-truth meshes side by side.
    
    Args:
        pred_mesh: Predicted trimesh.Trimesh
        gt_mesh: Ground-truth trimesh.Trimesh
        save_path: Path to save the figure
        title: Overall title
    
    Returns:
        matplotlib Figure
    """
    fig = plt.figure(figsize=(16, 7))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    # Predicted mesh
    ax1 = fig.add_subplot(121, projection='3d')
    render_mesh_matplotlib(
        np.array(pred_mesh.vertices), np.array(pred_mesh.faces),
        ax=ax1, color='#4CAF50', alpha=0.6, title='Predicted'
    )
    
    # Ground truth mesh
    ax2 = fig.add_subplot(122, projection='3d')
    render_mesh_matplotlib(
        np.array(gt_mesh.vertices), np.array(gt_mesh.faces),
        ax=ax2, color='#2196F3', alpha=0.6, title='Ground Truth'
    )
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    return fig


def plot_metrics_comparison(metrics: dict, 
                            save_path: Optional[str] = None) -> Figure:
    """Create a bar chart of evaluation metrics.
    
    Args:
        metrics: Dict with metric name → value
        save_path: Path to save the figure
    
    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Split into minimise and maximise metrics
    minimise = {k: v for k, v in metrics.items() 
                if k in ['hausdorff_distance', 'chamfer_distance', 'rmse']}
    maximise = {k: v for k, v in metrics.items()
                if k in ['volumetric_iou', 'completeness']}
    
    # Plot minimise metrics
    if minimise:
        names = [k.replace('_', '\n') for k in minimise.keys()]
        values = list(minimise.values())
        bars = axes[0].bar(names, values, color=['#FF6B6B', '#FF8E8E', '#FFB4B4'])
        axes[0].set_title('Minimise ↓', fontweight='bold')
        axes[0].set_ylabel('Value')
        for bar, val in zip(bars, values):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f'{val:.4f}', ha='center', va='bottom', fontsize=9)
    
    # Plot maximise metrics
    if maximise:
        names = [k.replace('_', '\n') for k in maximise.keys()]
        values = list(maximise.values())
        bars = axes[1].bar(names, values, color=['#4CAF50', '#66BB6A'])
        axes[1].set_title('Maximise ↑', fontweight='bold')
        axes[1].set_ylabel('Value')
        for bar, val in zip(bars, values):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f'{val:.4f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_training_history(history: dict, 
                          save_path: Optional[str] = None) -> Figure:
    """Plot training and validation loss curves.
    
    Args:
        history: Dict with 'train_loss' and 'val_loss' lists
        save_path: Path to save the figure
    
    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    ax.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training History')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig
