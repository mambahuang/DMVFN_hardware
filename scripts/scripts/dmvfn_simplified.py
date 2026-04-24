#!/usr/bin/env python3
"""
dmvfn_simplified.py — fp32 PyTorch reference that matches what your HW is
actually doing (as encoded in test.py's fixed_point_forward() + HW sum-of-ΔF).
==============================================================================

Context (read this first)
-------------------------
After reviewing test.py's fixed_point_forward(), it is clear that the HW
pipeline does exactly the following:

  1. PyTorch (fp32) runs all 9 MVFBs iteratively with BETWEEN-BLOCK warping
     (with fixed-point bilinear warp).  The 'L10 input' dumped to
     p{15..18}_b{i}_L10_input.txt is the tensor right BEFORE MVFB_i.lastconv.

  2. On the RTL side, the HW only replays MVFB_i.lastconv (TConv 4x4 s=2) on
     that dumped L10 tensor, for every block i in 0..8, then sums the resulting
     ΔF^i and Δmask^i, and does ONE final warp+blend on the summed flow.

So the HW's algorithmic difference from the original DMVFN (eval mode) is:
  - No Bernoulli routing (all 9 blocks contribute).
  - Only ONE final warp+blend, not 9 progressive warp+blends.
    (The 9 per-block warps still happen — inside PyTorch — to build the
    per-block L10 inputs that get dumped. But the HW itself does only one.)

This file gives you two fp32 baselines so you can locate any PSNR loss:

  'orig'        : DMVFN.forward(training=False) — Bernoulli routing + per-block
                  progressive warp.  The paper's published pipeline.
  'no_routing'  : Same as 'orig' but ref=[1]*9 (all blocks active).  Isolates
                  the cost of dropping Bernoulli routing.
  'hw_algo'     : The HW's algorithm: iterate all 9 blocks to produce per-block
                  ΔF^i, SUM them, final warp once.  The fixed-point Q11.10 /
                  tile / bilinear HW should match THIS fp32 target (plus a few
                  dB of quantization loss).  **This is your HW's PSNR target.**

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
import random
import argparse
import importlib

import numpy as np
import torch
import torch.nn as nn

import torchvision.utils as vutils


# ---------------------------------------------------------------------------
# Helper: put project root on sys.path and import the repo's warp() verbatim
# ---------------------------------------------------------------------------
def _import_repo_warp():
    here = os.path.dirname(os.path.abspath(__file__))
    for p in (os.path.dirname(here), here, os.getcwd()):
        if p and p not in sys.path:
            sys.path.insert(0, p)
    from model.arch import warp as _warp
    return _warp


# ===========================================================================
# The simplified forward paths
# ===========================================================================
class DMVFN_Simplified(nn.Module):
    """
    Wraps a real DMVFN (loaded with pretrained weights).  Replaces forward()
    with one of three explicit variants so you can pinpoint PSNR losses.
    """

    # 'orig'       — full eval forward, with Bernoulli routing (reproduces arch.py).
    # 'no_routing' — eval forward but ref=[1]*9 (all blocks active).
    # 'hw_algo'    — iterative per-block forward (like test.py fixed_point_forward,
    #                but float warp instead of fp warp), then SUM ΔF^i across
    #                blocks and final-warp once.  Matches HW's summation algorithm.
    # 'hw_faithful' — routing-aware iterative forward to build L10 state, but SUM
    #                ΔF^i across all blocks and final-warp once, matching what HW
    MODE = 'hw_faithful'

    SCALE_LIST = [4, 4, 4, 2, 2, 2, 1, 1, 1]
    NUM_BLOCKS = 9

    def __init__(self, dmvfn_orig):
        super().__init__()
        self.dmvfn = dmvfn_orig
        self.blocks = nn.ModuleList([
            getattr(dmvfn_orig, f'block{i}') for i in range(self.NUM_BLOCKS)
        ])
        self._warp = _import_repo_warp()

    @torch.no_grad()
    def forward(self, x, scale=None, training=False):
        """
        x: (B, 6, H, W) in [0, 1]
        Returns list of one (B, 3, H, W) tensor (to match arch.py eval output shape).
        """
        scale = scale or self.SCALE_LIST

        if self.MODE == 'orig':
            # Straight passthrough to the real DMVFN
            merged = self.dmvfn(x, scale=scale, training=False)
            return merged if len(merged) > 0 else [x[:, :3]]

        if self.MODE == 'no_routing':
            return self._forward_no_routing(x, scale)

        if self.MODE == 'hw_algo':
            return self._forward_hw_algo(x, scale)

        if self.MODE == 'hw_faithful':
            return self._forward_hw_faithful(x, scale)

        raise ValueError(f"Unknown MODE: {self.MODE}")

    # ----------------------------------------------------------------
    def _forward_no_routing(self, x, scale):
        """Original eval forward but ref is forced to [1]*9."""
        B, _, H, W = x.shape
        dev = x.device
        img0 = x[:, :3]
        img1 = x[:, 3:6]
        flow = torch.zeros(B, 4, H, W, device=dev)
        mask = torch.zeros(B, 1, H, W, device=dev)
        warped_img0 = img0.clone()
        warped_img1 = img1.clone()

        for i in range(self.NUM_BLOCKS):
            x_in = torch.cat((img0, img1, warped_img0, warped_img1, mask), 1)
            flow_d, mask_d = self.blocks[i](x_in, flow, scale=scale[i])
            flow = flow + flow_d
            mask = mask + mask_d
            warped_img0 = self._warp(img0, flow[:, :2])
            warped_img1 = self._warp(img1, flow[:, 2:4])

        m = torch.sigmoid(mask)
        merged = warped_img0 * m + warped_img1 * (1.0 - m)
        return [torch.clamp(merged, 0.0, 1.0)]

    # ----------------------------------------------------------------
    def _forward_hw_faithful(self, x, scale):
        """
        Faithfully reproduces what the HW actually computes when fed dump data
        from test.py's routing-enabled forward:

        1. Run arch.py's eval path (with Bernoulli routing) to produce the
           per-block L10 inputs — exactly as test.py does.  Blocks skipped by
           routing contribute zero ΔF/Δmask (matching the all-zero FM dump).
        2. For each executed block, run only lastconv (TConv) to get ΔF^i.
        3. SUM all ΔF^i and Δmask^i (including zeros from skipped blocks).
        4. ONE final warp + blend on the summed flow/mask.

        This should match 'orig' PSNR very closely (< 0.01 dB difference,
        only from Bernoulli randomness).  Any gap between this and your HW
        integer sim is purely quantization + tiling error.
        """
        from model.arch import RoundSTE
        B, _, H, W = x.shape
        dev = x.device
        img0 = x[:, :3]
        img1 = x[:, 3:6]

        # --- Step 1: compute routing ref (same as arch.py eval) ---
        routing_vector = self.dmvfn.routing(x[:, :6]).reshape(B, -1)
        routing_vector = torch.sigmoid(self.dmvfn.l1(routing_vector))
        routing_vector = routing_vector / (routing_vector.sum(1, True) + 1e-6) * 4.5
        routing_vector = torch.clamp(routing_vector, 0, 1)
        ref = RoundSTE.apply(routing_vector)

        # --- Step 2: iterative forward (respecting routing) to build L10 state ---
        flow_feed = torch.zeros(B, 4, H, W, device=dev)
        mask_feed = torch.zeros(B, 1, H, W, device=dev)
        warped_img0 = img0.clone()
        warped_img1 = img1.clone()

        # HW accumulator: sum of ΔF from all blocks (zeros for skipped)
        flow_sum = torch.zeros(B, 4, H, W, device=dev)
        mask_sum = torch.zeros(B, 1, H, W, device=dev)

        for i in range(self.NUM_BLOCKS):
            if ref[0, i]:
                x_in = torch.cat((img0, img1, warped_img0, warped_img1, mask_feed), 1)
                flow_d, mask_d = self.blocks[i](x_in, flow_feed, scale=scale[i])

                # HW accumulator
                flow_sum = flow_sum + flow_d
                mask_sum = mask_sum + mask_d

                # Update feed state (only when block is active, matching arch.py eval)
                flow_feed = flow_feed + flow_d
                mask_feed = mask_feed + mask_d
                warped_img0 = self._warp(img0, flow_feed[:, :2])
                warped_img1 = self._warp(img1, flow_feed[:, 2:4])
            # else: block skipped, no state change, no ΔF contribution

        # --- Step 3: ONE final warp + blend (HW behavior) ---
        w0_final = self._warp(img0, flow_sum[:, :2])
        w1_final = self._warp(img1, flow_sum[:, 2:4])
        m_final = torch.sigmoid(mask_sum)
        merged = w0_final * m_final + w1_final * (1.0 - m_final)
        return [torch.clamp(merged, 0.0, 1.0)]

    # ----------------------------------------------------------------
    def _forward_hw_algo(self, x, scale):
        """
        The HW's algorithmic path.

        Same iterative forward as test.py's fixed_point_forward, which gives
        us each block's L10 input (implicitly, we don't need to hook it —
        just re-run the forward but discard the per-block refinements and
        use only the summed ΔF).

        Trick: to match the HW's 'sum of 9 ΔF's, final warp once' behaviour,
        we accumulate flow_d/mask_d separately and do the final warp+blend
        just once at the end.  The per-block warped_img0/1 that feeds each
        block still has to be generated iteratively (that's what the dump
        captures).
        """
        B, _, H, W = x.shape
        dev = x.device
        img0 = x[:, :3]
        img1 = x[:, 3:6]

        # State used to feed MVFBs — iterative (matches test.py)
        flow_feed = torch.zeros(B, 4, H, W, device=dev)
        mask_feed = torch.zeros(B, 1, H, W, device=dev)
        warped_img0 = img0.clone()
        warped_img1 = img1.clone()

        # Summation of ΔF^i and Δmask^i — this is the HW-side sum
        flow_sum = torch.zeros(B, 4, H, W, device=dev)
        mask_sum = torch.zeros(B, 1, H, W, device=dev)

        for i in range(self.NUM_BLOCKS):
            x_in = torch.cat((img0, img1, warped_img0, warped_img1, mask_feed), 1)
            flow_d, mask_d = self.blocks[i](x_in, flow_feed, scale=scale[i])

            # HW's accumulator
            flow_sum = flow_sum + flow_d
            mask_sum = mask_sum + mask_d

            # Feed state for next block (same as test.py fixed_point_forward)
            flow_feed = flow_feed + flow_d
            mask_feed = mask_feed + mask_d
            warped_img0 = self._warp(img0, flow_feed[:, :2])
            warped_img1 = self._warp(img1, flow_feed[:, 2:4])

        # HW's final blend (ONE warp on the summed flow, not per-block warps)
        w0_final = self._warp(img0, flow_sum[:, :2])
        w1_final = self._warp(img1, flow_sum[:, 2:4])
        # HW uses linear sigmoid approximation: (mask >> MASK_SHIFT) + 512, clamped to [0, 1024]
        # Matches sim_dmvfn.py: mask_sh = sm_val >> MASK_SHIFT; mask_q = clamp(mask_sh + 512, 0, 1024)
        MASK_SHIFT = 8
        m_final = torch.clamp(mask_sum / (1 << MASK_SHIFT) + 0.5, 0.0, 1.0)
        merged   = w0_final * m_final + w1_final * (1.0 - m_final)
        return [torch.clamp(merged, 0.0, 1.0)]
    
    def _forward_hw_actual_per_layer(self, x, scale):
        """
        這是在模擬『硬體每一層都進行 Warp + Blend』的真實行為。
        對標你的 RTL：每過一個 Block 就出一個新的 warped_img 和 merged 結果。
        """
        B, _, H, W = x.shape
        dev = x.device
        img0 = x[:, :3]
        img1 = x[:, 3:6]

        # 初始化狀態
        flow = torch.zeros(B, 4, H, W, device=dev)
        mask = torch.zeros(B, 1, H, W, device=dev)
        warped_img0 = img0.clone()
        warped_img1 = img1.clone()
        
        # 用來存放每一層產出的融合影像 (如果你硬體每層都有輸出點)
        layer_outputs = []

        for i in range(self.NUM_BLOCKS):
            # 1. 準備輸入特徵 (17 channels)
            x_in = torch.cat((img0, img1, warped_img0, warped_img1, mask), 1)
            
            # 2. MVFB 計算殘差 (Delta Flow / Delta Mask)
            flow_d, mask_d = self.blocks[i](x_in, flow, scale=scale[i])

            # 3. 更新累加器 (Accumulator)
            flow = flow + flow_d
            mask = mask + mask_d

            # 4. 立即進行 Warp (這對應你硬體裡的 Warping Engine)
            warped_img0 = self._warp(img0, flow[:, :2])
            warped_img1 = self._warp(img1, flow[:, 2:4])

            # 5. 立即進行 Blend (這對應你硬體裡的 Alpha Blending 單元)
            m = torch.sigmoid(mask)
            merged = warped_img0 * m + warped_img1 * (1.0 - m)
            layer_outputs.append(torch.clamp(merged, 0.0, 1.0))

        # 返回最後一層的結果 (或是回傳整個 list 視你的評測需求而定)
        return [layer_outputs[-1]]


# ===========================================================================
# Drop-in Model replacement (same public surface as model.model.Model)
# ===========================================================================
class Model_Simplified:
    def __init__(self, local_rank=-1, resume_path=None, resume_epoch=0,
                 load_path=None, training=False):
        from model.model import Model as _OriginalModel
        assert not training, "Model_Simplified is eval-only."
        assert load_path is not None
        self._orig = _OriginalModel(load_path=load_path, training=False)
        self._simp = DMVFN_Simplified(self._orig.dmvfn)
        dev = next(self._orig.dmvfn.parameters()).device
        self._simp = self._simp.to(dev)
        self._simp.eval()

    @torch.no_grad()
    def eval(self, imgs, name='CityValDataset',
             scale_list=(4, 4, 4, 2, 2, 2, 1, 1, 1)):
        """Mirrors model.model.Model.eval() exactly; uses self._simp underneath."""
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
        elif name in ('KittiValDataset', 'DavisValDataset'):
            assert n == 9
            img0, img1 = imgs[:, 2], imgs[:, 3]
            for _ in range(5):
                merged = self._simp(torch.cat((img0, img1), 1),
                                    scale=scale_list, training=False)
                pred = merged[-1] if len(merged) > 0 else img0
                preds.append(pred)
                img0, img1 = img1, pred
        elif name == 'VimeoValDataset':
            assert n == 3
            merged = self._simp(torch.cat((imgs[:, 0], imgs[:, 1]), 1),
                                scale=scale_list, training=False)
            pred = merged[-1] if len(merged) > 0 else imgs[:, 0]
            preds.append(pred)
        elif name == 'single_test':
            merged = self._simp(imgs[0], scale=scale_list, training=False)
            return merged[-1] if len(merged) > 0 else imgs[:, 0]
        else:
            raise ValueError(f"Unknown dataset: {name}")
        return torch.stack(preds, 1)

    def device(self):
        self._orig.device()


# ===========================================================================
# CLI: report PSNR for all three variants, plus cross comparisons
# ===========================================================================
def _cli_compare():
    ap = argparse.ArgumentParser()
    ap.add_argument('--val_dataset', default='KittiValDataset',
                    choices=['CityValDataset', 'KittiValDataset',
                             'DavisValDataset', 'VimeoValDataset'])
    ap.add_argument('--load_path', required=True)
    ap.add_argument('--n_samples', type=int, default=20)
    ap.add_argument('--save_imgs', action='store_true', help='Save predicted images to disk?')
    args = ap.parse_args()

    if args.save_imgs:
        save_dir = "output_samples"
        os.makedirs(save_dir, exist_ok=True)
        print(f"[*] Picture will save to: {save_dir}")

    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.dirname(here))
    sys.path.insert(0, here)

    from torch.utils.data import DataLoader
    from model.model import Model as _OriginalModel

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    DSet = getattr(importlib.import_module('dataset.dataset'), args.val_dataset)
    val_loader = DataLoader(DSet(), batch_size=1, num_workers=1, pin_memory=True)

    model_orig = _OriginalModel(load_path=args.load_path, training=False)

    # Build three modes; all wrap the same DMVFN, so weights are shared
    model_simp = Model_Simplified(load_path=args.load_path, training=False)

    def run_mode(mode):
        DMVFN_Simplified.MODE = mode
        return mode

    print("=" * 70)
    print(f" dmvfn_simplified  |  dataset={args.val_dataset}  "
          f"samples={args.n_samples}")
    print(f" load_path={args.load_path}")
    print("=" * 70)
    print(" Comparing four fp32 paths vs ground truth:")
    print("   orig        : DMVFN(training=False), with Bernoulli routing")
    print("   no_routing  : same as orig but all 9 blocks active")
    print("   hw_faithful : routing-aware iterative forward + SUM ΔF + final warp")
    print("                 (matches what HW actually computes from dump data)")
    print("   hw_algo     : no-routing iterative forward + per-layer warp+blend")
    print("=" * 70)

    ALL_MODES = ('orig', 'no_routing', 'hw_faithful', 'hw_algo')
    results = {m: [] for m in ALL_MODES}
    hw_vs_orig = []
    hw_vs_noroute = []
    hwf_vs_orig = []

    for idx, (data_gpu, _) in enumerate(val_loader):
        if idx >= args.n_samples:
            break
        data_gpu = data_gpu.to(device, non_blocking=True) / 255.

        preds_by_mode = {}
        for m in ALL_MODES:
            run_mode(m)
            preds_by_mode[m] = model_simp.eval(data_gpu, args.val_dataset)[0]

        gt = data_gpu[0]
        n_pred = preds_by_mode['orig'].shape[0]
        for j in range(n_pred):
            gt_idx = 2 if args.val_dataset == 'VimeoValDataset' else (4 + j)
            target = gt[gt_idx]

            for m in ALL_MODES:
                mse = torch.mean((target - preds_by_mode[m][j]) ** 2).item()
                results[m].append(-10 * math.log10(max(mse, 1e-12)))

            if args.save_imgs:
                vutils.save_image(target, f"{save_dir}/sample_{idx:03d}_f{j}_GT.png")
                for m in ALL_MODES:
                    vutils.save_image(preds_by_mode[m][j], f"{save_dir}/sample_{idx:03d}_f{j}_{m}.png")

            mse_hw_orig = torch.mean(
                (preds_by_mode['hw_algo'][j] - preds_by_mode['orig'][j]) ** 2
            ).item()
            mse_hw_noroute = torch.mean(
                (preds_by_mode['hw_algo'][j] - preds_by_mode['no_routing'][j]) ** 2
            ).item()
            mse_hwf_orig = torch.mean(
                (preds_by_mode['hw_faithful'][j] - preds_by_mode['orig'][j]) ** 2
            ).item()
            hw_vs_orig.append(-10 * math.log10(max(mse_hw_orig, 1e-12)))
            hw_vs_noroute.append(-10 * math.log10(max(mse_hw_noroute, 1e-12)))
            hwf_vs_orig.append(-10 * math.log10(max(mse_hwf_orig, 1e-12)))

        if (idx + 1) % 5 == 0:
            n = len(results['orig'])
            o  = sum(results['orig'])       / n
            nr = sum(results['no_routing']) / n
            hf = sum(results['hw_faithful']) / n
            hw = sum(results['hw_algo'])    / n
            print(f"  [{idx+1:3d}/{args.n_samples}] "
                  f"orig={o:5.2f}dB  "
                  f"no_route={nr:5.2f}dB  "
                  f"hw_faithful={hf:5.2f}dB  "
                  f"hw_algo={hw:5.2f}dB")

    n = max(len(results['orig']), 1)
    print()
    print("=" * 70)
    print(f" RESULTS  ({n} predictions)")
    print("=" * 70)
    print(f"  orig        vs GT : {sum(results['orig'])      /n:6.3f} dB")
    print(f"  no_routing  vs GT : {sum(results['no_routing'])/n:6.3f} dB"
          f"   (routing cost: {(sum(results['orig'])-sum(results['no_routing']))/n:+.3f} dB)")
    print(f"  hw_faithful vs GT : {sum(results['hw_faithful'])/n:6.3f} dB"
          f"   (routing + sum-of-ΔF + final warp)")
    print(f"  hw_algo     vs GT : {sum(results['hw_algo'])   /n:6.3f} dB"
          f"   (no routing, per-layer warp)")
    print()
    print(f"  hw_faithful vs orig   : {sum(hwf_vs_orig)      /n:6.3f} dB  ← should be very high")
    print(f"  hw_algo vs orig       : {sum(hw_vs_orig)       /n:6.3f} dB")
    print(f"  hw_algo vs no_routing : {sum(hw_vs_noroute)    /n:6.3f} dB")
    print()
    print("  Interpretation:")
    print("  • hw_faithful vs orig should be ~inf dB (identical routing → same result).")
    print("    If it's low, there's a Bernoulli seed mismatch.")
    print("  • hw_faithful vs GT is the true PSNR target for your HW dump pipeline.")
    print("  • hw_algo vs GT shows the cost of ignoring routing entirely.")
    print("  • The gap hw_faithful→hw_algo shows the impact of routing mismatch.")
    print("=" * 70)


if __name__ == '__main__':
    _cli_compare()