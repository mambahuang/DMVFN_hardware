#!/usr/bin/env python3
"""
make_video.py — End-to-end HW-sim video generation pipeline (Colab-ready)
===========================================================================
For each folder (scene), predicts frames 2~6 using GT frame pairs:
  frame[0] + frame[1] -> pred[2]
  frame[1] + frame[2] -> pred[3]
  frame[2] + frame[3] -> pred[4]
  frame[3] + frame[4] -> pred[5]
  frame[4] + frame[5] -> pred[6]

Output video frames (per scene): frame[0], frame[1], pred[2], ..., pred[6]
Each video frame layout: GT_left | HW_sim_pred | GT_right

Usage (in Colab, from /content/DMVFN_hardware/):
  python scripts/make_video.py \
      --dataset cityscapes \
      --load_path pretrained_models/dmvfn_city.pkl \
      --n_folders 10 \
      --fps 5 \
      --out_dir ./video_output

  python scripts/make_video.py \
      --dataset kitti \
      --load_path pretrained_models/dmvfn_kitti.pkl \
      --n_folders 10 \
      --fps 5 \
      --out_dir ./video_output
"""

import os
import sys
import time
import random
import argparse
import importlib
import shutil
import glob

import cv2
import numpy as np
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Paths (all relative to repo root = cwd in Colab)
# ---------------------------------------------------------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from utils.util import *
from model.model import Model

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--dataset',   required=True, choices=['cityscapes', 'kitti'])
parser.add_argument('--load_path', required=True)
parser.add_argument('--n_folders', type=int, default=10,
                    help='Number of scene folders to process')
parser.add_argument('--fps',       type=int, default=5)
parser.add_argument('--out_dir',   default='./video_output')
args = parser.parse_args()

DATASET  = args.dataset
IS_CITY  = DATASET == 'cityscapes'
PREFIX   = 'city' if IS_CITY else 'p'
GEN_SCRIPT = os.path.join(ROOT, 'cityscapes' if IS_CITY else 'KITTI',
                           'gen_city_stim.py' if IS_CITY else 'gen_kitti_stim.py')
DATA_DIR   = os.path.join(ROOT, 'cityscapes' if IS_CITY else 'KITTI')
# Image folders: data/cityscapes/test/000000/ or data/KITTI/test/000000/
IMG_ROOT   = os.path.join(ROOT, 'data', 'cityscapes' if IS_CITY else 'KITTI', 'test')

SRC_H = 512 if IS_CITY else 256
SRC_W = 1024 if IS_CITY else 832

os.makedirs(args.out_dir, exist_ok=True)

# ---------------------------------------------------------------------------
# Seed
# ---------------------------------------------------------------------------
seed = 1234
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------------------------------------------------------------------------
# Image I/O helpers
# ---------------------------------------------------------------------------
def load_img_uint8(path):
    """Load PNG as HWC uint8 RGB."""
    return np.array(Image.open(path).convert('RGB'), dtype=np.uint8)

def img_to_tensor(img_uint8):
    """HWC uint8 -> [1, 3, H, W] float32 in [0,1] on device."""
    t = torch.from_numpy(img_uint8).permute(2, 0, 1).unsqueeze(0).float() / 255.
    return t.to(device)

def tensor_to_uint8(t):
    """[1, 3, H, W] or [3, H, W] float32 -> HWC uint8."""
    if t.dim() == 4:
        t = t[0]
    arr = t.detach().cpu().numpy().transpose(1, 2, 0)
    return np.clip(np.round(arr * 255), 0, 255).astype(np.uint8)

# ---------------------------------------------------------------------------
# Stim dump helpers (copied from test.py)
# ---------------------------------------------------------------------------
def _q(val):
    return int(round(float(val) * 256))

def _pad_weight_tconv2d(w_np, target_ic, target_oc):
    ic, oc, kh, kw = w_np.shape
    out = np.zeros((target_ic, target_oc, kh, kw), dtype=w_np.dtype)
    out[:min(ic, target_ic), :min(oc, target_oc)] = w_np[:target_ic, :target_oc]
    return out

def write_weight_tconv2d_txt(filepath, w_int):
    n_ic, n_oc, kh, kw = w_int.shape
    with open(filepath, 'w') as f:
        for k in range(kh * kw):
            ky, kx = k // kw, k % kw
            for oc in range(n_oc):
                f.write(' '.join(str(int(w_int[ic, oc, ky, kx]))
                                 for ic in range(n_ic)) + '\n')

def dump_img_banks(img_hwc_uint8, bank_prefix, out_dir):
    H, W, _ = img_hwc_uint8.shape
    banks = [[] for _ in range(4)]
    for y in range(H):
        for x in range(W):
            b_idx = (y % 2) * 2 + (x % 2)
            # BGR order matching test.py convention
            banks[b_idx].append(
                f"{int(img_hwc_uint8[y,x,2]):02x} "
                f"{int(img_hwc_uint8[y,x,1]):02x} "
                f"{int(img_hwc_uint8[y,x,0]):02x}")
    for n in range(4):
        with open(os.path.join(out_dir, f"{bank_prefix}_bank{n}.txt"), 'w') as f:
            f.write('\n'.join(banks[n]) + '\n')

def _dump_fm_and_weights(out_dir, prefix, bidx, fm_tensor, model, macpe=32):
    wt_dir = os.path.join(out_dir, f'block{bidx}')
    os.makedirs(wt_dir, exist_ok=True)
    block = getattr(model.dmvfn, f'block{bidx}')

    if fm_tensor is None:
        w_np    = block.lastconv.weight.detach().cpu().numpy().astype(np.float32)
        H       = 256 if prefix == 'city' else 128
        W_fm    = 512 if prefix == 'city' else 416
        actual_ic = w_np.shape[0]
        fm_q    = np.zeros((actual_ic, H, W_fm), dtype=np.int32)
        w_np    = np.zeros_like(w_np)
        bias_np = (np.zeros_like(block.lastconv.bias.detach().cpu().numpy())
                   if getattr(block.lastconv, 'bias', None) is not None else None)
    else:
        _, IC, H, W_fm = fm_tensor.shape
        fm_q    = np.clip(
            np.round(fm_tensor[0].numpy() * 256.0).astype(np.int32),
            -32768, 32767)
        w_np    = block.lastconv.weight.detach().cpu().numpy().astype(np.float32)
        bias_np = (block.lastconv.bias.detach().cpu().numpy().astype(np.float32)
                   if getattr(block.lastconv, 'bias', None) is not None else None)

    actual_ic = fm_q.shape[0]
    with open(os.path.join(out_dir, f'{prefix}_b{bidx}_L10_input.txt'), 'w') as f:
        for c in range(actual_ic):
            for y in range(H):
                f.write(' '.join(str(int(fm_q[c, y, x])) for x in range(W_fm)) + '\n')

    ic_sl = ((w_np.shape[0] + macpe - 1) // macpe) * macpe
    w_int = np.vectorize(_q)(_pad_weight_tconv2d(w_np, target_ic=ic_sl, target_oc=8))
    write_weight_tconv2d_txt(
        os.path.join(wt_dir, f'{prefix}_b{bidx}_L10_weight.txt'), w_int)

    if bias_np is not None:
        bias_q = np.round(bias_np * 256.0).astype(np.int32)
        with open(os.path.join(wt_dir, f'{prefix}_b{bidx}_L10_bias.txt'), 'w') as f:
            f.write(' '.join(str(int(b)) for b in bias_q) + '\n')

def dump_all_phases(img0_uint8, img1_uint8, model, fp_lastconv_fm):
    out15 = os.path.join(ROOT, 'data', 'phase15')
    os.makedirs(out15, exist_ok=True)
    _dump_fm_and_weights(out15, PREFIX, 0, fp_lastconv_fm.get(0), model)
    dump_img_banks(img0_uint8, 'p15_img0', out15)
    dump_img_banks(img1_uint8, 'p15_img1', out15)

    out16 = os.path.join(ROOT, 'data', 'phase16')
    os.makedirs(out16, exist_ok=True)
    for bidx in [1, 2]:
        _dump_fm_and_weights(out16, PREFIX, bidx, fp_lastconv_fm.get(bidx), model)

    out17 = os.path.join(ROOT, 'data', 'phase17')
    os.makedirs(out17, exist_ok=True)
    for bidx in [3, 4, 5]:
        _dump_fm_and_weights(out17, PREFIX, bidx, fp_lastconv_fm.get(bidx), model)

    out18 = os.path.join(ROOT, 'data', 'phase18')
    os.makedirs(out18, exist_ok=True)
    for bidx in [6, 7, 8]:
        _dump_fm_and_weights(out18, PREFIX, bidx, fp_lastconv_fm.get(bidx), model)

    ref = [1 if i in fp_lastconv_fm else 0 for i in range(9)]
    with open(os.path.join(out15, 'routing_ref.txt'), 'w') as f:
        f.write(' '.join(str(r) for r in ref) + '\n')

# ---------------------------------------------------------------------------
# GPU inference: img0_t + img1_t -> pred uint8, and dump phase stim
# ---------------------------------------------------------------------------
def infer_and_dump(img0_uint8, img1_uint8, model):
    """Run DMVFN on (img0, img1), dump stim, return pred as HWC uint8."""
    img0_t = img_to_tensor(img0_uint8)
    img1_t = img_to_tensor(img1_uint8)

    fp_lastconv_fm = {}
    hooks = []
    def _make_hook(bidx):
        def hook(_m, inp, _o):
            fp_lastconv_fm[bidx] = inp[0].detach().cpu()
        return hook

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    for bidx in range(9):
        hooks.append(
            getattr(model.dmvfn, f'block{bidx}').lastconv
            .register_forward_hook(_make_hook(bidx)))

    with torch.no_grad():
        merged = model.dmvfn(
            torch.cat((img0_t, img1_t), 1),
            scale=[4, 4, 4, 2, 2, 2, 1, 1, 1], training=False)

    for h in hooks:
        h.remove()

    pred_uint8 = tensor_to_uint8(merged[-1])
    dump_all_phases(img0_uint8, img1_uint8, model, fp_lastconv_fm)
    return pred_uint8

# ---------------------------------------------------------------------------
# sim_dmvfn inline runner
# ---------------------------------------------------------------------------
def run_sim(dataset_key, out_dir):
    sim_module_path = os.path.join(ROOT, 'python', 'sim_dmvfn.py')
    spec = importlib.util.spec_from_file_location('sim_dmvfn', sim_module_path)
    sim  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sim)
    return sim.simulate(dataset_key, out_dir, use_routing=False)

# ---------------------------------------------------------------------------
# Copy phase dirs to where gen_stim expects them, then run gen_stim
# ---------------------------------------------------------------------------
def run_gen_stim():
    for ph in [15, 16, 17, 18]:
        src = os.path.join(ROOT, 'data', f'phase{ph}')
        dst = os.path.join(DATA_DIR, f'phase{ph}')
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)

    spec = importlib.util.spec_from_file_location('gen_stim', GEN_SCRIPT)
    gen  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    gen.main()

# ---------------------------------------------------------------------------
# Build one video frame: GT_left | hw_pred | GT_right
# ---------------------------------------------------------------------------
def make_frame(src0, src1, hw_pred):
    """All inputs are HWC uint8 RGB. Returns HWC uint8 BGR for cv2.
    Layout: src0 | src1 | hw_pred"""
    gap = np.full((SRC_H, 4, 3), 40, dtype=np.uint8)
    row = np.concatenate([src0, gap, src1, gap, hw_pred], axis=1)
    return cv2.cvtColor(row, cv2.COLOR_RGB2BGR)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f'=== make_video.py  dataset={DATASET}  n_folders={args.n_folders} ===')

    # Load model
    model = Model(load_path=args.load_path, training=False)
    model.dmvfn.eval()

    # Collect scene folders, sorted
    folders = sorted(glob.glob(os.path.join(IMG_ROOT, '*')))
    folders = [f for f in folders if os.path.isdir(f)]
    if len(folders) == 0:
        print(f'ERROR: no folders found under {IMG_ROOT}'); return
    folders = folders[:args.n_folders]
    print(f'Found {len(folders)} folders, processing first {len(folders)}')

    # Video writer
    video_w = SRC_W * 3 + 4 * 2
    fourcc  = cv2.VideoWriter_fourcc(*'mp4v')
    video_path = os.path.join(args.out_dir, f'{DATASET}_hw_sim.mp4')
    video = cv2.VideoWriter(video_path, fourcc, args.fps, (video_w, SRC_H))

    frames_dir = os.path.join(args.out_dir, f'{DATASET}_frames')
    os.makedirs(frames_dir, exist_ok=True)

    psnr_log_path = os.path.join(args.out_dir, f'{DATASET}_psnr.txt')
    psnr_records = []  # list of (folder_name, pred_idx, psnr_hw, psnr_fp32)

    t_total = time.time()

    for folder_idx, folder in enumerate(folders):
        print(f'\n[Folder {folder_idx+1}/{len(folders)}] {os.path.basename(folder)}')

        # Load all PNGs in this folder, sorted
        img_paths = sorted(glob.glob(os.path.join(folder, '*.png')) +
                           glob.glob(os.path.join(folder, '*.jpg')))
        if len(img_paths) < 6:
            print(f'  SKIP: only {len(img_paths)} images, need at least 6')
            continue

        # Load frames [0..5] as uint8 RGB
        gt_frames = [load_img_uint8(p) for p in img_paths[:6]]

        # Resize to target resolution if needed
        gt_frames = [
            np.array(Image.fromarray(f).resize((SRC_W, SRC_H), Image.BILINEAR))
            if f.shape[:2] != (SRC_H, SRC_W) else f
            for f in gt_frames
        ]

        # Write frame[0] and frame[1] directly (no prediction)
        # Layout: frame[fi] | frame[fi+1] | frame[fi] (placeholder for pred)
        for fi in range(2):
            f0 = gt_frames[fi]
            f1 = gt_frames[fi + 1] if (fi + 1) < len(gt_frames) else f0
            frame_bgr = make_frame(f0, f1, f0)
            video.write(frame_bgr)
            png_path = os.path.join(frames_dir,
                f'folder{folder_idx:03d}_f{fi:02d}_gt.png')
            cv2.imwrite(png_path, frame_bgr)

        # Predict frames [2..6]: frame[i] + frame[i+1] -> pred[i+2]
        for pred_idx in range(2, 7):
            i = pred_idx - 2   # 0,1,2,3,4
            img0_uint8 = gt_frames[i]
            img1_uint8 = gt_frames[i + 1]
            gt_pred    = gt_frames[i + 2] if (i + 2) < len(gt_frames) else None

            t0 = time.time()

            # Step 1: GPU inference + dump stim
            fp32_pred = infer_and_dump(img0_uint8, img1_uint8, model)
            print(f'  pred[{pred_idx}]: infer done ({time.time()-t0:.1f}s)')

            # Step 2: gen_stim
            t0 = time.time()
            run_gen_stim()
            print(f'  pred[{pred_idx}]: gen_stim done ({time.time()-t0:.1f}s)')

            # Step 3: sim_dmvfn
            t0 = time.time()
            sim_out_dir = os.path.join(args.out_dir,
                f'sim_{DATASET}_f{folder_idx:03d}_p{pred_idx}')
            os.makedirs(sim_out_dir, exist_ok=True)
            hw_pred = run_sim(DATASET, sim_out_dir)
            print(f'  pred[{pred_idx}]: sim done ({time.time()-t0:.1f}s)')

            if hw_pred is None:
                print(f'  WARNING: sim returned None for pred[{pred_idx}], using fp32')
                hw_pred = fp32_pred
            else:
                # sim_dmvfn stores channels in BGR order due to dump convention;
                # swap R/B back to RGB before display
                hw_pred = hw_pred[:, :, ::-1].copy()

            # Step 4: compose frame — layout: src0 | src1 | hw_pred
            frame_bgr = make_frame(img0_uint8, img1_uint8, hw_pred)
            video.write(frame_bgr)

            # Save individual PNGs
            png_path = os.path.join(frames_dir,
                f'folder{folder_idx:03d}_f{pred_idx:02d}_hw_pred.png')
            cv2.imwrite(png_path, frame_bgr)

            if gt_pred is not None:
                def psnr(a, b):
                    if a.shape != b.shape:
                        b = np.array(Image.fromarray(b).resize(
                            (a.shape[1], a.shape[0]), Image.BILINEAR))
                    mse = np.mean((a.astype(np.float64) - b.astype(np.float64))**2)
                    return -10 * np.log10(max(mse / (255.**2), 1e-12))
                p_hw  = psnr(hw_pred, gt_pred)
                p_fp  = psnr(fp32_pred, gt_pred)
                print(f'  pred[{pred_idx}] PSNR: hw={p_hw:.2f} dB  fp32={p_fp:.2f} dB')
                psnr_records.append((os.path.basename(folder), pred_idx, p_hw, p_fp))

        print(f'  Folder done.')

    video.release()

    # Write PSNR log
    with open(psnr_log_path, 'w') as f:
        f.write(f'folder\tpred_idx\tpsnr_hw(dB)\tpsnr_fp32(dB)\n')
        for folder_name, pred_idx, p_hw, p_fp in psnr_records:
            f.write(f'{folder_name}\t{pred_idx}\t{p_hw:.4f}\t{p_fp:.4f}\n')
        if psnr_records:
            avg_hw  = sum(r[2] for r in psnr_records) / len(psnr_records)
            avg_fp  = sum(r[3] for r in psnr_records) / len(psnr_records)
            f.write(f'\nAVERAGE\t-\t{avg_hw:.4f}\t{avg_fp:.4f}\n')

    print(f'\n{"="*60}')
    print(f' Done in {time.time()-t_total:.1f}s')
    print(f' Video ({args.fps} fps): {video_path}')
    print(f' Frames: {frames_dir}/')
    print(f' PSNR log: {psnr_log_path}')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()
