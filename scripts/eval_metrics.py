#!/usr/bin/env python3
"""
eval_metrics.py — Compute PSNR / SSIM / LPIPS from make_video.py output frames.

Usage (in Colab, from /content/DMVFN_hardware/):
  python scripts/eval_metrics.py \
      --dataset cityscapes \
      --out_dir ./video_output

  python scripts/eval_metrics.py \
      --dataset kitti \
      --out_dir ./video_output
"""

import os
import sys
import argparse
import glob
import math

import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity as sk_ssim
import lpips

# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--dataset',  required=True, choices=['cityscapes', 'kitti'])
parser.add_argument('--out_dir',  default='./video_output')
parser.add_argument('--no_lpips', action='store_true',
                    help='Skip LPIPS (faster, no GPU needed)')
args = parser.parse_args()

ROOT     = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
IMG_ROOT = os.path.join(ROOT, 'data',
                        'cityscapes' if args.dataset == 'cityscapes' else 'KITTI',
                        'test')
FRAMES_DIR = os.path.join(args.out_dir, f'{args.dataset}_frames')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------------------------------------------------------------------------
def load_rgb(path):
    return np.array(Image.open(path).convert('RGB'), dtype=np.uint8)

def compute_psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64))**2)
    return -10 * np.log10(max(mse / (255.**2), 1e-12))

def compute_ssim(a, b):
    return sk_ssim(a, b, data_range=255, channel_axis=2)

def to_lpips_tensor(img_uint8):
    t = torch.from_numpy(img_uint8).permute(2, 0, 1).unsqueeze(0).float() / 255.
    return (t * 2 - 1).to(device)   # LPIPS expects [-1, 1]

# ---------------------------------------------------------------------------
def main():
    if not os.path.isdir(FRAMES_DIR):
        print(f'ERROR: frames dir not found: {FRAMES_DIR}')
        print('Run make_video.py first.')
        sys.exit(1)

    # Load LPIPS model
    lpips_fn = None
    if not args.no_lpips:
        print('Loading LPIPS model...')
        lpips_fn = lpips.LPIPS(net='alex').to(device)

    # Collect all hw_pred PNGs: folder{FFF}_f{PP}_hw_pred.png
    pred_pngs = sorted(glob.glob(os.path.join(FRAMES_DIR, '*_hw_pred.png')))
    if not pred_pngs:
        print(f'ERROR: no *_hw_pred.png found in {FRAMES_DIR}')
        sys.exit(1)
    print(f'Found {len(pred_pngs)} predicted frames to evaluate.\n')

    records = []   # (folder_name, pred_idx, psnr, ssim, lpips)

    for pred_png in pred_pngs:
        fname = os.path.basename(pred_png)
        # parse folder index and pred_idx from filename
        # format: folder{FFF}_f{PP}_hw_pred.png
        try:
            parts      = fname.split('_')
            folder_idx = int(parts[0].replace('folder', ''))
            pred_idx   = int(parts[1].replace('f', ''))
        except (IndexError, ValueError):
            print(f'  SKIP (unexpected filename): {fname}')
            continue

        # Find corresponding GT folder
        all_folders = sorted(f for f in glob.glob(os.path.join(IMG_ROOT, '*'))
                             if os.path.isdir(f))
        if folder_idx >= len(all_folders):
            print(f'  SKIP: folder_idx {folder_idx} out of range')
            continue
        folder = all_folders[folder_idx]
        folder_name = os.path.basename(folder)

        # GT frame index = pred_idx (frame[pred_idx] is the ground truth)
        img_paths = sorted(glob.glob(os.path.join(folder, '*.png')) +
                           glob.glob(os.path.join(folder, '*.jpg')))
        if pred_idx >= len(img_paths):
            print(f'  SKIP: GT frame[{pred_idx}] not found in {folder_name}')
            continue

        gt = load_rgb(img_paths[pred_idx])

        # Load the side-by-side PNG and extract the hw_pred panel (rightmost third)
        # The side-by-side is: src0(W) | gap(4) | src1(W) | gap(4) | hw_pred(W)
        sbs = load_rgb(pred_png)
        H, W_total, _ = sbs.shape
        W = (W_total - 8) // 3   # each panel width (gap=4, two gaps)
        hw_pred = sbs[:, W_total - W:, :]   # rightmost panel

        # Resize GT to match panel size if needed
        if gt.shape[:2] != (H, W):
            gt = np.array(Image.fromarray(gt).resize((W, H), Image.BILINEAR))

        psnr_val = compute_psnr(hw_pred, gt)
        ssim_val = compute_ssim(hw_pred, gt)

        lpips_val = float('nan')
        if lpips_fn is not None:
            with torch.no_grad():
                lpips_val = lpips_fn(
                    to_lpips_tensor(hw_pred),
                    to_lpips_tensor(gt)
                ).item()

        records.append((folder_name, pred_idx, psnr_val, ssim_val, lpips_val))
        print(f'  {folder_name} pred[{pred_idx}]'
              f'  PSNR={psnr_val:.2f} dB'
              f'  SSIM={ssim_val:.4f}'
              + (f'  LPIPS={lpips_val:.4f}' if lpips_fn else ''))

    if not records:
        print('No valid records to write.')
        return

    # Write results
    out_path = os.path.join(args.out_dir, f'{args.dataset}_metrics.txt')
    with open(out_path, 'w') as f:
        header = 'folder\tpred_idx\tpsnr(dB)\tssim\tlpips\n'
        f.write(header)
        for folder_name, pred_idx, psnr_val, ssim_val, lpips_val in records:
            f.write(f'{folder_name}\t{pred_idx}\t'
                    f'{psnr_val:.4f}\t{ssim_val:.4f}\t'
                    f'{lpips_val:.4f}\n')

        avg_psnr  = sum(r[2] for r in records) / len(records)
        avg_ssim  = sum(r[3] for r in records) / len(records)
        avg_lpips = sum(r[4] for r in records if not math.isnan(r[4]))
        avg_lpips = avg_lpips / len(records) if avg_lpips else float('nan')
        f.write(f'\nAVERAGE\t-\t{avg_psnr:.4f}\t{avg_ssim:.4f}\t{avg_lpips:.4f}\n')

    print(f'\n{"="*55}')
    print(f' AVERAGE  PSNR={avg_psnr:.3f} dB'
          f'  SSIM={avg_ssim:.4f}'
          + (f'  LPIPS={avg_lpips:.4f}' if lpips_fn else ''))
    print(f' Results saved: {out_path}')
    print(f'{"="*55}')


if __name__ == '__main__':
    main()
