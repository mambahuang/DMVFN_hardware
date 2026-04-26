#!/usr/bin/env python3
"""
analyze_psnr_heatmap.py — HW PSNR analysis with heatmaps
=========================================================
Two modes:
  A) PSNR log analysis  (always runs — only needs *_psnr.txt)
     - Bar charts, gap analysis (fp32 - hw), folder/pred_idx breakdown
  B) Frame-level spatial heatmap
     - With --gt_root: compares hw_pred (cropped from composite PNG) vs true GT
       GT layout: data/{dataset}/test/{folder_id}/im1.png im2.png ... (sorted)
       pred_idx k  →  gt_frames[k]  (same logic as make_video.py)
     - Without --gt_root: compares hw_pred vs src0 from the composite (fallback)

Usage (from repo root):
  # With GT dataset (recommended, run remotely):
  python scripts/analyze_psnr_heatmap.py --dataset cityscapes \\
      --gt_root ./data/cityscapes/test --out_dir ./analysis

  python scripts/analyze_psnr_heatmap.py --dataset kitti \\
      --gt_root ./data/KITTI/test --out_dir ./analysis

  # Without GT (local, uses src0 as proxy):
  python scripts/analyze_psnr_heatmap.py --dataset both --out_dir ./analysis

  --n_folders N  : only process first N scene folders (default: all)
"""

import os
import glob
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image

ROOT     = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
VID_ROOT = os.path.join(ROOT, 'video_output', 'video_output')

parser = argparse.ArgumentParser()
parser.add_argument('--dataset',   default='both', choices=['cityscapes', 'kitti', 'both'])
parser.add_argument('--out_dir',   default=os.path.join(ROOT, 'analysis'))
parser.add_argument('--gt_root',   default=None,
                    help='Path to GT test dir, e.g. ./data/cityscapes/test')
parser.add_argument('--n_folders', type=int, default=None,
                    help='Max scene folders to process (default: all)')
parser.add_argument('--vid_root',  default=VID_ROOT)
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)

SRC_W = {'cityscapes': 1024, 'kitti': 832}
SRC_H = {'cityscapes':  512, 'kitti': 256}
GAP   = 4

# ---------------------------------------------------------------------------
def load_rgb(path):
    return np.array(Image.open(path).convert('RGB'), dtype=np.float32)

def psnr_val(a, b):
    mse = np.mean((a - b) ** 2)
    return float('inf') if mse == 0 else 10 * np.log10(255.**2 / mse)

# ---------------------------------------------------------------------------
# Part A: PSNR log analysis
# ---------------------------------------------------------------------------
def parse_psnr_log(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('folder') or line.startswith('AVERAGE'):
                continue
            parts = line.split('\t')
            if len(parts) < 4:
                continue
            records.append({
                'folder':    parts[0],
                'pred_idx':  int(parts[1]),
                'psnr_hw':   float(parts[2]),
                'psnr_fp32': float(parts[3]),
                'gap':       float(parts[3]) - float(parts[2]),
            })
    return records

def plot_psnr_log(dataset, records, out_dir):
    folders   = sorted(set(r['folder']    for r in records))
    pred_idxs = sorted(set(r['pred_idx'] for r in records))
    labels    = [f"{r['folder']} p{r['pred_idx']}" for r in records]
    hw_vals   = [r['psnr_hw']   for r in records]
    fp_vals   = [r['psnr_fp32'] for r in records]
    gap_vals  = [r['gap']       for r in records]
    x         = np.arange(len(records))

    # --- 1. HW vs fp32 bar chart ---
    fig, ax = plt.subplots(figsize=(max(12, len(records) * 0.55), 4))
    w = 0.38
    ax.bar(x - w/2, hw_vals, w, label='HW sim',  color='#4292c6')
    ax.bar(x + w/2, fp_vals, w, label='FP32 ref', color='#fd8d3c', alpha=0.7)
    ax.axhline(np.mean(hw_vals), color='#2171b5', linestyle='--', linewidth=1,
               label=f'HW avg={np.mean(hw_vals):.2f} dB')
    ax.axhline(np.mean(fp_vals), color='#d94801', linestyle='--', linewidth=1,
               label=f'FP32 avg={np.mean(fp_vals):.2f} dB')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=60, ha='right', fontsize=7)
    ax.set_ylabel('PSNR (dB)')
    ax.set_title(f'{dataset.upper()} — HW vs FP32 PSNR  (gap = fp32 − hw)')
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(out_dir, f'{dataset}_psnr_bar.png')
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f'  Saved: {p}')

    # --- 2. Gap heatmap: rows=folder, cols=pred_idx ---
    gap_matrix = np.full((len(folders), len(pred_idxs)), np.nan)
    fi_map = {f: i for i, f in enumerate(folders)}
    pi_map = {p: i for i, p in enumerate(pred_idxs)}
    for r in records:
        gap_matrix[fi_map[r['folder']], pi_map[r['pred_idx']]] = r['gap']

    fig, ax = plt.subplots(figsize=(max(5, len(pred_idxs) * 1.2),
                                    max(4, len(folders) * 0.55)))
    vmax = np.nanmax(gap_matrix)
    im   = ax.imshow(gap_matrix, cmap='RdYlGn_r', aspect='auto',
                     vmin=0, vmax=vmax)
    plt.colorbar(im, ax=ax, label='FP32 − HW PSNR (dB)  [larger = worse HW]')
    ax.set_xticks(range(len(pred_idxs)))
    ax.set_xticklabels([f'pred {p}' for p in pred_idxs])
    ax.set_yticks(range(len(folders)))
    ax.set_yticklabels(folders, fontsize=8)
    # annotate cells
    for fi in range(len(folders)):
        for pi in range(len(pred_idxs)):
            v = gap_matrix[fi, pi]
            if not np.isnan(v):
                ax.text(pi, fi, f'{v:.1f}', ha='center', va='center',
                        fontsize=8, color='black')
    ax.set_title(f'{dataset.upper()} — PSNR Gap heatmap (fp32 − hw) per folder × pred_idx')
    fig.tight_layout()
    p = os.path.join(out_dir, f'{dataset}_gap_heatmap.png')
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f'  Saved: {p}')

    # --- 3. Per-folder mean HW PSNR ---
    folder_hw = {f: [] for f in folders}
    for r in records:
        folder_hw[r['folder']].append(r['psnr_hw'])
    f_means = [np.mean(folder_hw[f]) for f in folders]

    fig, ax = plt.subplots(figsize=(max(8, len(folders) * 0.8), 3.5))
    colors  = ['#d73027' if v < 18 else '#fc8d59' if v < 22 else '#4dac26'
               for v in f_means]
    ax.bar(range(len(folders)), f_means, color=colors)
    ax.axhline(np.mean(f_means), color='k', linestyle='--',
               label=f'avg={np.mean(f_means):.2f} dB')
    ax.set_xticks(range(len(folders)))
    ax.set_xticklabels(folders, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Mean HW PSNR (dB)')
    ax.set_title(f'{dataset.upper()} — Mean HW PSNR per folder')
    ax.legend()
    fig.tight_layout()
    p = os.path.join(out_dir, f'{dataset}_per_folder_psnr.png')
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f'  Saved: {p}')

    # --- 4. pred_idx trend ---
    pidx_hw  = {p: [] for p in pred_idxs}
    pidx_fp  = {p: [] for p in pred_idxs}
    for r in records:
        pidx_hw[r['pred_idx']].append(r['psnr_hw'])
        pidx_fp[r['pred_idx']].append(r['psnr_fp32'])
    mean_hw = [np.mean(pidx_hw[p]) for p in pred_idxs]
    mean_fp = [np.mean(pidx_fp[p]) for p in pred_idxs]

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(pred_idxs, mean_hw, 'o-', color='#4292c6', label='HW sim')
    ax.plot(pred_idxs, mean_fp, 's--', color='#fd8d3c', label='FP32 ref')
    ax.fill_between(pred_idxs, mean_hw, mean_fp, alpha=0.15, color='gray',
                    label='gap')
    ax.set_xlabel('Predicted frame index')
    ax.set_ylabel('Mean PSNR (dB)')
    ax.set_title(f'{dataset.upper()} — PSNR vs prediction distance')
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = os.path.join(out_dir, f'{dataset}_psnr_trend.png')
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f'  Saved: {p}')

# ---------------------------------------------------------------------------
# Part B: Spatial heatmap from frame PNGs
# ---------------------------------------------------------------------------
def _parse_frame_filename(basename):
    """Return (folder_idx, pred_idx) from 'folder{F}_f{P}_hw_pred.png', or None."""
    import re
    m = re.match(r'folder(\d+)_f(\d+)_hw_pred\.png$', basename)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None


def _load_gt_frame(gt_root, folder_idx, pred_idx):
    """
    Return GT image (H×W×3 float32) for the given folder/pred_idx pair.
    Folder selection: sorted(listdir(gt_root))[folder_idx].
    Frame selection : sorted image files in that folder, index = pred_idx.
    Returns None if anything is missing.
    """
    try:
        folders = sorted(
            d for d in os.listdir(gt_root)
            if os.path.isdir(os.path.join(gt_root, d))
        )
        if folder_idx >= len(folders):
            return None
        folder_path = os.path.join(gt_root, folders[folder_idx])
        exts = ('.png', '.jpg', '.jpeg', '.ppm')
        frames = sorted(
            f for f in os.listdir(folder_path)
            if os.path.splitext(f)[1].lower() in exts
        )
        if pred_idx >= len(frames):
            return None
        img_path = os.path.join(folder_path, frames[pred_idx])
        return load_rgb(img_path)
    except Exception as e:
        print(f'    [WARN] GT load failed (folder_idx={folder_idx}, '
              f'pred_idx={pred_idx}): {e}')
        return None


def spatial_analysis(dataset, out_dir, n_folders, gt_root=None):
    frames_dir = os.path.join(args.vid_root, f'{dataset}_frames')
    if not os.path.isdir(frames_dir):
        print(f'  [SKIP] frames dir not found: {frames_dir}')
        return

    hw_pngs = sorted(glob.glob(os.path.join(frames_dir, '*_hw_pred.png')))
    if not hw_pngs:
        print(f'  [SKIP] no hw_pred PNGs in {frames_dir}')
        return

    # Apply n_folders limit: keep only PNGs whose folder_idx < n_folders
    if n_folders is not None:
        filtered = []
        for png in hw_pngs:
            parsed = _parse_frame_filename(os.path.basename(png))
            if parsed is None or parsed[0] < n_folders:
                filtered.append(png)
        hw_pngs = filtered

    W = SRC_W[dataset]
    H = SRC_H[dataset]
    total_W = W * 3 + GAP * 2   # composite width

    use_gt  = gt_root is not None and os.path.isdir(gt_root)
    ref_label = 'GT' if use_gt else 'src0'

    err_accum = np.zeros((H, W), dtype=np.float64)
    ch_accum  = np.zeros((H, W, 3), dtype=np.float64)
    count = 0
    skipped_gt = 0

    sample_records = []

    for png in hw_pngs:
        basename  = os.path.basename(png)
        composite = load_rgb(png)

        # Crop hw_pred from composite right panel
        if composite.shape[1] == total_W and composite.shape[0] == H:
            src0 = composite[:, :W]
            hw   = composite[:, total_W - W: total_W]
        elif composite.shape[1] >= W:
            src0 = composite[:H, :W]
            hw   = composite[:H, composite.shape[1] - W:]
        else:
            continue

        # Choose reference: true GT or src0 fallback
        if use_gt:
            parsed = _parse_frame_filename(basename)
            if parsed is not None:
                folder_idx, pred_idx = parsed
                gt_img = _load_gt_frame(gt_root, folder_idx, pred_idx)
            else:
                gt_img = None

            if gt_img is None:
                skipped_gt += 1
                continue  # skip frame if GT unavailable

            # Resize GT to match HW pred dimensions if needed
            if gt_img.shape[0] != H or gt_img.shape[1] != W:
                gt_pil  = Image.fromarray(gt_img.astype(np.uint8))
                gt_pil  = gt_pil.resize((W, H), Image.BICUBIC)
                gt_img  = np.array(gt_pil, dtype=np.float32)

            ref = gt_img
        else:
            ref = src0

        err  = np.mean(np.abs(hw - ref), axis=2)
        ch_e = np.abs(hw - ref)
        err_accum += err
        ch_accum  += ch_e
        count += 1

        p = psnr_val(hw, ref)
        sample_records.append({
            'label': basename,
            'psnr':  p,
            'hw':    hw.astype(np.uint8),
            'ref':   ref.astype(np.uint8),
            'err':   err,
        })

    if skipped_gt:
        print(f'  [WARN] {skipped_gt} frames skipped (GT not found)')
    if count == 0:
        print('  [SKIP] no valid composite frames parsed')
        return

    avg_err = err_accum / count
    avg_ch  = ch_accum  / count

    print(f'  Spatial analysis ({ref_label}): {count} frames, '
          f'mean_err={avg_err.mean():.2f}, max_err={avg_err.max():.1f}')

    # --- Average error heatmap ---
    fig, ax = plt.subplots(figsize=(10, 3.5))
    im = ax.imshow(avg_err, cmap='hot', vmin=0,
                   vmax=min(avg_err.max() * 1.1, 30))
    plt.colorbar(im, ax=ax, label=f'Mean |HW − {ref_label}| (pixel)')
    ax.set_title(f'{dataset.upper()} — Spatial avg-error heatmap  '
                 f'(HW pred vs {ref_label}, {count} frames)')
    ax.axis('off')
    fig.tight_layout()
    p = os.path.join(out_dir, f'{dataset}_spatial_avg_heatmap.png')
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f'  Saved: {p}')

    # --- Per-channel heatmap ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 3))
    vmax = avg_ch.max()
    for ci, (name, cmap) in enumerate(zip(['Red', 'Green', 'Blue'],
                                           ['Reds', 'Greens', 'Blues'])):
        im = axes[ci].imshow(avg_ch[:, :, ci], cmap=cmap, vmin=0, vmax=vmax)
        plt.colorbar(im, ax=axes[ci], label='|err|')
        axes[ci].set_title(name); axes[ci].axis('off')
    fig.suptitle(f'{dataset.upper()} — Per-channel spatial error (HW vs {ref_label})')
    fig.tight_layout()
    p = os.path.join(out_dir, f'{dataset}_spatial_channel_heatmap.png')
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f'  Saved: {p}')

    # --- Row / col profiles ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 3))
    axes[0].plot(avg_err.mean(axis=1))
    axes[0].set_xlabel('Row (y)'); axes[0].set_ylabel('Mean |err|')
    axes[0].set_title(f'{dataset.upper()} — Row-wise mean error')
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(avg_err.mean(axis=0))
    axes[1].set_xlabel('Col (x)'); axes[1].set_ylabel('Mean |err|')
    axes[1].set_title(f'{dataset.upper()} — Column-wise mean error')
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    p = os.path.join(out_dir, f'{dataset}_spatial_profile.png')
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f'  Saved: {p}')

    # --- Detail panels: worst / median / best ---
    sorted_rec = sorted(sample_records, key=lambda r: r['psnr'])
    picks = {
        'worst':  sorted_rec[0],
        'median': sorted_rec[len(sorted_rec) // 2],
        'best':   sorted_rec[-1],
    }
    for tag, rec in picks.items():
        fig = plt.figure(figsize=(15, 3.5))
        gs  = gridspec.GridSpec(1, 4, figure=fig, wspace=0.05)
        ax0 = fig.add_subplot(gs[0])
        ax0.imshow(rec['ref']); ax0.set_title(ref_label); ax0.axis('off')
        ax1 = fig.add_subplot(gs[1])
        ax1.imshow(rec['hw']);  ax1.set_title('HW pred'); ax1.axis('off')
        ax2 = fig.add_subplot(gs[2])
        diff = np.clip(
            np.abs(rec['hw'].astype(np.float32) -
                   rec['ref'].astype(np.float32)) * 4,
            0, 255).astype(np.uint8)
        ax2.imshow(diff); ax2.set_title('|Diff| ×4'); ax2.axis('off')
        ax3 = fig.add_subplot(gs[3])
        im = ax3.imshow(rec['err'], cmap='hot',
                        vmin=0, vmax=max(rec['err'].max(), 1))
        plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04, label='|err|')
        ax3.set_title('Error heatmap'); ax3.axis('off')
        fig.suptitle(f'{dataset.upper()} [{tag}]  {rec["label"]}  '
                     f'PSNR(hw/{ref_label})={rec["psnr"]:.2f} dB', fontsize=10)
        fig.tight_layout()
        p = os.path.join(out_dir, f'{dataset}_spatial_detail_{tag}.png')
        fig.savefig(p, dpi=150); plt.close(fig)
        print(f'  Saved: {p}')

# ---------------------------------------------------------------------------
def analyze(dataset):
    print(f'\n{"="*60}')
    print(f'  Dataset: {dataset.upper()}')
    print(f'{"="*60}')

    log_path = os.path.join(args.vid_root, f'{dataset}_psnr.txt')
    if os.path.exists(log_path):
        print(f'  [A] PSNR log analysis: {log_path}')
        records = parse_psnr_log(log_path)
        if records:
            print(f'      {len(records)} records  '
                  f'hw_avg={np.mean([r["psnr_hw"] for r in records]):.2f} dB  '
                  f'fp32_avg={np.mean([r["psnr_fp32"] for r in records]):.2f} dB  '
                  f'gap_avg={np.mean([r["gap"] for r in records]):.2f} dB')
            plot_psnr_log(dataset, records, args.out_dir)
        else:
            print('      [WARN] no records parsed')
    else:
        print(f'  [SKIP A] psnr log not found: {log_path}')

    print(f'  [B] Spatial heatmap from frame PNGs')
    spatial_analysis(dataset, args.out_dir, args.n_folders, gt_root=args.gt_root)

# ---------------------------------------------------------------------------
datasets = ['cityscapes', 'kitti'] if args.dataset == 'both' else [args.dataset]
for ds in datasets:
    analyze(ds)

print('\nDone. Results in:', args.out_dir)
