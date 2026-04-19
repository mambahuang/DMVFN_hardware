#!/usr/bin/env python3
"""
dmvfn_simplified.py — Sum-of-flows fp32 reference matching the HW algorithm
==============================================================================
Aligned to the actual repo (model/arch.py, model/model.py).

Your hardware does NOT execute the original DMVFN's iterative refinement.
It executes a "sum-of-flows" approximation:

    for i in 0..8:
        ΔF^i, Δm^i = block_i( ... )       # only the last TConv really runs in HW
    F_final = sum(ΔF^i)
    m_final = sum(Δm^i)
    output  = sigmoid(m_final) * warp(I_{t-1}, F_final[:2])
            + (1 - sigmoid(m_final)) * warp(I_t,     F_final[2:4])

If you compare HW output with the ORIGINAL DMVFN's pred.png you'll always
see "residuals" — that gap is the algorithmic difference, not a HW bug.

This file gives you a fp32 PyTorch reference that runs the SAME sum-of-flows
algorithm, so you can measure:

    HW(int)          vs  Simplified(fp32)   ← shows fixed-point loss only
    Simplified(fp32) vs  Original(fp32)     ← algorithmic gap (a fixed cost)

Usage
-----
Quick A/B (no test.py edits):
  python scripts/dmvfn_simplified.py \
      --val_dataset KittiValDataset \
      --load_path  pretrained_models/dmvfn_kitti.pkl \
      --n_samples  20

Drop-in inside test.py:
  from dmvfn_simplified import Model_Simplified as Model
"""

import os
import sys
import math
import argparse
import importlib

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Re-use the repo's own warp() so behaviour matches bit-for-bit
# (border padding mode, identity grid construction, align_corners=True).
# ---------------------------------------------------------------------------
def _import_repo_warp():
    here = os.path.dirname(os.path.abspath(__file__))
    for p in (os.path.dirname(here), here, os.getcwd()):
        if p and p not in sys.path:
            sys.path.insert(0, p)
    from model.arch import warp as _warp
    return _warp


# ===========================================================================
# Sum-of-flows DMVFN
# ===========================================================================
class DMVFN_Simplified(nn.Module):
    """Wrap a real DMVFN; replace forward() with the HW's sum-of-flows path."""

    # 'sum_of_flows'      → matches gen_kitti_stim.py exactly:
    #                       every block sees zero warped imgs and zero prev-flow.
    # 'sum_no_warp_iter'  → diagnostic: keep inter-block warping but drop routing.
    MODE = 'sum_of_flows'

    SCALE_LIST = [4, 4, 4, 2, 2, 2, 1, 1, 1]
    NUM_BLOCKS = 9

    def __init__(self, dmvfn_orig):
        super().__init__()
        # The real DMVFN exposes block0..block8 as separate attributes
        self.blocks = nn.ModuleList([
            getattr(dmvfn_orig, f'block{i}') for i in range(self.NUM_BLOCKS)
        ])
        # Borrow routing for diagnostics (we ignore it; HW always runs all blocks)
        self.routing = dmvfn_orig.routing
        self.l1      = dmvfn_orig.l1
        self._warp   = _import_repo_warp()

    @torch.no_grad()
    def forward(self, x, scale=None, training=False):
        """
        x: (B, 6, H, W) — concat([img_{t-1}, img_t]) in [0, 1]
        Returns: list with one (B, 3, H, W) tensor — predicted img_{t+1}.
                 (List format mirrors original DMVFN.forward so Model.eval()
                  glue code stays identical.)
        """
        scale = scale or self.SCALE_LIST
        B, _, H, W = x.shape
        device = x.device
        img0 = x[:, :3]   # I_{t-1}
        img1 = x[:, 3:6]  # I_t

        # Accumulators
        flow = torch.zeros(B, 4, H, W, device=device)
        mask = torch.zeros(B, 1, H, W, device=device)

        # Per-block context
        if self.MODE == 'sum_of_flows':
            warped_img0 = torch.zeros_like(img0)
            warped_img1 = torch.zeros_like(img1)
            blk_prev_flow = torch.zeros_like(flow)
            blk_prev_mask = torch.zeros_like(mask)
        elif self.MODE == 'sum_no_warp_iter':
            warped_img0 = img0.clone()
            warped_img1 = img1.clone()
            blk_prev_flow = flow
            blk_prev_mask = mask
        else:
            raise ValueError(f"Unknown MODE: {self.MODE}")

        # Run every block (no Bernoulli routing, all 9 always active)
        for i in range(self.NUM_BLOCKS):
            # MVFB.forward(x_13ch, prev_flow_4ch, scale)
            #   x_13ch = [img0, img1, warped_img0, warped_img1, mask]
            mvfb_x = torch.cat(
                (img0, img1, warped_img0, warped_img1, blk_prev_mask), dim=1
            )
            flow_d, mask_d = self.blocks[i](mvfb_x, blk_prev_flow, scale=scale[i])
            flow = flow + flow_d
            mask = mask + mask_d

            if self.MODE == 'sum_no_warp_iter':
                warped_img0 = self._warp(img0, flow[:, :2])
                warped_img1 = self._warp(img1, flow[:, 2:4])
                blk_prev_flow = flow
                blk_prev_mask = mask

        # Final one-shot warp + blend
        warped_img0 = self._warp(img0, flow[:, :2])
        warped_img1 = self._warp(img1, flow[:, 2:4])
        m = torch.sigmoid(mask)
        merged = warped_img0 * m + warped_img1 * (1.0 - m)
        merged = torch.clamp(merged, 0.0, 1.0)
        return [merged]


# ===========================================================================
# Drop-in replacement for model.model.Model
# ===========================================================================
class Model_Simplified:
    """Public surface mirrors model.model.Model (only eval() is needed)."""

    def __init__(self, local_rank=-1, resume_path=None, resume_epoch=0,
                 load_path=None, training=False):
        from model.model import Model as _OriginalModel
        assert not training, "Model_Simplified is eval-only."
        assert load_path is not None, "load_path is required."
        self._orig = _OriginalModel(load_path=load_path, training=False)
        self._simp = DMVFN_Simplified(self._orig.dmvfn)
        device = next(self._orig.dmvfn.parameters()).device
        self._simp = self._simp.to(device)
        self._simp.eval()

    @torch.no_grad()
    def eval(self, imgs, name='CityValDataset',
             scale_list=(4, 4, 4, 2, 2, 2, 1, 1, 1)):
        """Bit-for-bit copy of model.model.Model.eval, but uses self._simp."""
        scale_list = list(scale_list)
        b, n, c, h, w = imgs.shape
        preds = []
        if name == 'CityValDataset':
            assert n == 14
            img0, img1 = imgs[:, 2], imgs[:, 3]
            for _ in range(5):
                merged = self._simp(torch.cat((img0, img1), 1),
                                    scale=scale_list, training=False)
                pred = merged[-1] if len(merged) > 0 else img0
                preds.append(pred)
                img0, img1 = img1, pred
            assert len(preds) == 5
        elif name in ('KittiValDataset', 'DavisValDataset'):
            assert n == 9
            img0, img1 = imgs[:, 2], imgs[:, 3]
            for _ in range(5):
                merged = self._simp(torch.cat((img0, img1), 1),
                                    scale=scale_list, training=False)
                pred = merged[-1] if len(merged) > 0 else img0
                preds.append(pred)
                img0, img1 = img1, pred
            assert len(preds) == 5
        elif name == 'VimeoValDataset':
            assert n == 3
            merged = self._simp(torch.cat((imgs[:, 0], imgs[:, 1]), 1),
                                scale=scale_list, training=False)
            pred = merged[-1] if len(merged) > 0 else imgs[:, 0]
            preds.append(pred)
            assert len(preds) == 1
        elif name == 'single_test':
            merged = self._simp(imgs[0], scale=scale_list, training=False)
            return merged[-1] if len(merged) > 0 else imgs[:, 0]
        else:
            raise ValueError(f"Unknown dataset: {name}")
        return torch.stack(preds, 1)

    def device(self):
        self._orig.device()


# ===========================================================================
# CLI: Original vs Simplified A/B comparison
# ===========================================================================
def _cli_compare():
    ap = argparse.ArgumentParser()
    ap.add_argument('--val_dataset', default='KittiValDataset',
                    choices=['CityValDataset', 'KittiValDataset',
                             'DavisValDataset', 'VimeoValDataset'])
    ap.add_argument('--load_path', required=True)
    ap.add_argument('--n_samples', type=int, default=20)
    ap.add_argument('--mode', default='sum_of_flows',
                    choices=['sum_of_flows', 'sum_no_warp_iter'])
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.dirname(here))
    sys.path.insert(0, here)

    from torch.utils.data import DataLoader
    from model.model import Model as _OriginalModel

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    DSet = getattr(importlib.import_module('dataset.dataset'), args.val_dataset)
    val_loader = DataLoader(DSet(), batch_size=1, num_workers=1, pin_memory=True)

    model_orig = _OriginalModel(load_path=args.load_path, training=False)
    model_simp = Model_Simplified(load_path=args.load_path, training=False)
    DMVFN_Simplified.MODE = args.mode

    print("=" * 70)
    print(f" A/B  dataset={args.val_dataset}  mode={args.mode}")
    print(f" load_path={args.load_path}")
    print("=" * 70)

    psnr_o, psnr_s, psnr_d = [], [], []

    for idx, (data_gpu, _name) in enumerate(val_loader):
        if idx >= args.n_samples:
            break
        data_gpu = data_gpu.to(device, non_blocking=True) / 255.

        po = model_orig.eval(data_gpu, args.val_dataset)[0]
        ps = model_simp.eval(data_gpu, args.val_dataset)[0]
        gt = data_gpu[0]

        n_pred = po.shape[0]
        for j in range(n_pred):
            if args.val_dataset == 'VimeoValDataset':
                gt_idx = 2
            else:
                gt_idx = 4 + j
            mse_o = torch.mean((gt[gt_idx] - po[j]) ** 2).item()
            mse_s = torch.mean((gt[gt_idx] - ps[j]) ** 2).item()
            mse_d = torch.mean((po[j]      - ps[j]) ** 2).item()
            psnr_o.append(-10 * math.log10(max(mse_o, 1e-12)))
            psnr_s.append(-10 * math.log10(max(mse_s, 1e-12)))
            psnr_d.append(-10 * math.log10(max(mse_d, 1e-12)))

        if (idx + 1) % 5 == 0:
            n = len(psnr_o)
            print(f"  [{idx+1:3d}/{args.n_samples}] "
                  f"orig={sum(psnr_o)/n:5.2f}dB "
                  f"simp={sum(psnr_s)/n:5.2f}dB "
                  f"o-vs-s={sum(psnr_d)/n:5.2f}dB")

    n = max(len(psnr_o), 1)
    print()
    print("=" * 70)
    print(f" RESULTS  ({n} predictions across {min(args.n_samples, idx+1)} samples)")
    print("=" * 70)
    print(f"  Original DMVFN    vs GT : {sum(psnr_o)/n:6.3f} dB")
    print(f"  Simplified        vs GT : {sum(psnr_s)/n:6.3f} dB"
          f"   ← your HW's PSNR target")
    print(f"  Original vs Simplified  : {sum(psnr_d)/n:6.3f} dB"
          f"   ← algorithmic gap (HW cannot exceed this)")
    print()
    print("  How to read this:")
    print("  • If your HW vs Simplified PSNR > ~35 dB → HW is correct;")
    print("    the visible 'residual/distortion' is the sum-of-flows cost.")
    print("  • Otherwise: a fixed-point/tiling/bilinear bug remains.")
    print("=" * 70)


if __name__ == '__main__':
    _cli_compare()
