"""
Visualization utilities for asteroid shape reconstruction.
3D mesh rendering, light-curve plots, loss history, metric charts.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import trimesh
from typing import Dict, List, Optional


def _render_mesh_on_axis(ax, mesh, color='steelblue', alpha=0.7):
    """Render a mesh on a matplotlib 3D axis."""
    verts = mesh.vertices
    faces = mesh.faces
    max_faces = 5000
    if len(faces) > max_faces:
        idx = np.random.choice(len(faces), max_faces, replace=False)
        faces = faces[idx]
    polys = Poly3DCollection(verts[faces], alpha=alpha, facecolor=color,
                             edgecolor='gray', linewidth=0.1)
    ax.add_collection3d(polys)
    ext = np.max(np.abs(verts)) * 1.1
    ax.set_xlim(-ext, ext); ax.set_ylim(-ext, ext); ax.set_zlim(-ext, ext)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_box_aspect([1, 1, 1])


def plot_mesh_comparison(mesh_pred, mesh_gt=None, title="Shape Reconstruction",
                         save_path="shape_comparison.png", elev=25, azim=45):
    """Side-by-side 3D view of predicted and ground-truth meshes."""
    n = 2 if mesh_gt else 1
    fig = plt.figure(figsize=(8*n, 7))
    ax1 = fig.add_subplot(1, n, 1, projection='3d')
    _render_mesh_on_axis(ax1, mesh_pred)
    ax1.set_title("Reconstructed", fontsize=14, fontweight='bold')
    ax1.view_init(elev=elev, azim=azim)
    if mesh_gt:
        ax2 = fig.add_subplot(1, n, 2, projection='3d')
        _render_mesh_on_axis(ax2, mesh_gt, color='coral')
        ax2.set_title("Ground Truth", fontsize=14, fontweight='bold')
        ax2.view_init(elev=elev, azim=azim)
    fig.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout(); plt.savefig(save_path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  Saved: {save_path}")


def plot_lightcurve_fit(obs_phases, obs_flux, pred_phases, pred_flux,
                        title="Light Curve Fit", save_path="lightcurve_fit.png"):
    """Plot observed vs predicted light curves with residuals."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), gridspec_kw={'height_ratios': [3, 1]})
    ax1.scatter(np.rad2deg(obs_phases), obs_flux, s=20, alpha=0.7, color='coral', label='Observed')
    ax1.plot(np.rad2deg(pred_phases), pred_flux, color='steelblue', lw=2, label='Predicted')
    ax1.set_ylabel('Relative Flux'); ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.legend(); ax1.grid(True, alpha=0.3)
    pred_i = np.interp(obs_phases, pred_phases, pred_flux)
    res = obs_flux - pred_i
    ax2.scatter(np.rad2deg(obs_phases), res, s=15, alpha=0.7, color='gray')
    ax2.axhline(0, color='black', ls='--', lw=0.8)
    ax2.set_xlabel('Phase (deg)'); ax2.set_ylabel('Residual'); ax2.grid(True, alpha=0.3)
    ax2.set_title(f'RMS = {np.sqrt(np.mean(res**2)):.4f}', fontsize=11)
    plt.tight_layout(); plt.savefig(save_path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  Saved: {save_path}")


def plot_loss_history(history, title="Optimization History", save_path="loss_history.png"):
    """Plot optimization loss history (4-panel)."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    items = [('total_loss', 'Total', 'tab:blue'), ('photo_loss', 'Photometric', 'tab:red'),
             ('smooth_loss', 'Smoothness', 'tab:green'), ('volume_loss', 'Volume', 'tab:orange')]
    for ax, (k, lbl, c) in zip(axes.ravel(), items):
        if k in history and history[k]:
            ax.plot(history[k], color=c, lw=1); ax.set_title(lbl, fontweight='bold')
            ax.set_xlabel('Iter'); ax.set_ylabel('Loss'); ax.set_yscale('log'); ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout(); plt.savefig(save_path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  Saved: {save_path}")


def plot_metrics_bar(metrics, title="Evaluation Metrics", save_path="metrics_bar.png"):
    """Bar chart of evaluation metrics."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    dist = {k: metrics[k] for k in ['hausdorff', 'chamfer', 'rmse'] if k in metrics}
    if dist:
        bars = ax1.bar(dist.keys(), dist.values(), color=['#e74c3c','#e67e22','#f39c12'], edgecolor='k', lw=0.5)
        ax1.set_title("Distance (lower=better)", fontweight='bold')
        for b, v in zip(bars, dist.values()):
            ax1.text(b.get_x()+b.get_width()/2, b.get_height(), f'{v:.4f}', ha='center', va='bottom')
    qual = {k: metrics[k] for k in ['iou', 'completeness'] if k in metrics}
    if qual:
        bars = ax2.bar(qual.keys(), qual.values(), color=['#27ae60','#2980b9'], edgecolor='k', lw=0.5)
        ax2.set_title("Quality (higher=better)", fontweight='bold')
        for b, v in zip(bars, qual.values()):
            ax2.text(b.get_x()+b.get_width()/2, b.get_height(), f'{v:.4f}', ha='center', va='bottom')
    fig.suptitle(title, fontweight='bold')
    plt.tight_layout(); plt.savefig(save_path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  Saved: {save_path}")


def plot_multi_view(mesh, title="Multi-View", save_path="multi_view.png"):
    """Render mesh from 4 viewing angles."""
    fig = plt.figure(figsize=(14, 14))
    views = [(25,45,"Front-Right"),(25,135,"Back-Right"),(25,225,"Back-Left"),(90,0,"Top-Down")]
    for i, (el, az, lbl) in enumerate(views):
        ax = fig.add_subplot(2, 2, i+1, projection='3d')
        _render_mesh_on_axis(ax, mesh)
        ax.set_title(lbl, fontweight='bold'); ax.view_init(elev=el, azim=az)
    fig.suptitle(title, fontweight='bold')
    plt.tight_layout(); plt.savefig(save_path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  Saved: {save_path}")
