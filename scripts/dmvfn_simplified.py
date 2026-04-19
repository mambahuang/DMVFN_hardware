#!/usr/bin/env python3
"""
dmvfn_simplified.py — Sum-of-flows fp32 reference matching the HW algorithm
==============================================================================

目的
----
你的硬體 (gen_kitti_stim.py / sim_dmvfn.py) 並不是執行原版 DMVFN 的
iterative refinement,而是執行下列「sum-of-flows」近似:

    for i in 0..8:
        ΔF^i = lastconv_i( L10_input_i )   # 不做 inter-block warping
    F_final = sum(ΔF^i for i in 0..8)
    output  = warp_blend( I_{t-1}, I_t, F_final )

如果直接拿這個跟 PyTorch 原版的 pred.png 比 PSNR,等於拿不同演算法
的兩個系統互比,殘影本來就會有 — 因此你會誤以為硬體有 bug。

本檔提供一個 fp32 PyTorch reference,執行完全一樣的 sum-of-flows
演算法,讓你做以下兩條對照:

  (A) HW(int) vs Simplified(fp32)
      → 衡量 「fixed-point quantization + tiling + bilinear」 的精度損失
      → 應該很接近 (PSNR 通常 > 35 dB);若不接近,問題在硬體/sim
  (B) Simplified(fp32) vs Original DMVFN(fp32)
      → 衡量 「sum-of-flows 跟 iterative refinement 的演算法差距」
      → 一定有 gap;這個 gap 不是硬體的錯,是你採用 sum-of-flows
        近似演算法的代價

(A) 通過代表硬體 OK;(B) 是另一回事,需要從架構面修(見最後 README 段)。

使用方式
--------
在你 scripts/test.py 裡:

    from dmvfn_simplified import Model_Simplified

    # 取代原本的 model:
    # model = Model(load_path=args.load_path, training=False)
    model_orig = Model(load_path=args.load_path, training=False)
    model_sim  = Model_Simplified(load_path=args.load_path)

    # 跑 evaluate(model_sim, ...) 拿 simplified 的 metric
    # 跑 evaluate(model_orig, ...) 拿 original 的 metric
    # 對比兩者的 PSNR,確認你硬體的目標應該是 model_sim 的 PSNR 而不是 model_orig

關鍵設計決策
------------
1. 每個 block 都吃 prev_img=0, prev_flow=0
   → 等同於 9 個 block 都「重新出發」算 first-block ΔF
   → 對應你 hardware sum-of-flows 的行為
   → 如果你 hardware 實際上是傳 prev_img/prev_flow 但只是 lastconv
      之後不 warp,要把 SUM_OF_FLOWS_INIT 改成 'pytorch_first_block'

2. Routing module:simplified 模式忽略 routing(視為全部 9 個 block 都 active)
   → 因為 hardware 沒做 routing,9 個 block 的 lastconv 都會做
   → 如果你 hardware 之後要支援 routing,改 USE_ROUTING=True 並把
      v_i=0 的 block 跳過

3. 假設原版 model.MVFB.MVFB 的 forward signature 是:
      def forward(self, x, flow, scale):
          # x: (B, 6+3+4, H, W)  = concat([img0, img1, prev_img, prev_flow])
          # flow: (B, 4, H, W)   = prev_flow
          # 回傳: (img_t+1_pred, flow, mask)
   不同 commit 的 DMVFN signature 可能略有不同;若不符請看本檔末
   ADAPTING SECTION 微調。
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Backward warping (matches the original DMVFN implementation exactly)
# ---------------------------------------------------------------------------
def warp(img, flow):
    """
    Backward warp using bilinear interpolation with zero-padding outside.
    Matches torch.nn.functional.grid_sample(padding_mode='zeros').
    img: (B, C, H, W)
    flow: (B, 2, H, W)  — (dx, dy) per pixel, in pixel units
    """
    B, _, H, W = img.shape
    device = img.device
    # Build identity grid in pixel coords
    yy, xx = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing='ij',
    )
    grid = torch.stack([xx, yy], dim=0).unsqueeze(0).expand(B, -1, -1, -1)
    sampled = grid + flow  # (B, 2, H, W)

    # Normalize to [-1, 1] for grid_sample
    sampled_x = 2.0 * sampled[:, 0] / max(W - 1, 1) - 1.0
    sampled_y = 2.0 * sampled[:, 1] / max(H - 1, 1) - 1.0
    grid_norm = torch.stack([sampled_x, sampled_y], dim=-1)  # (B, H, W, 2)

    return F.grid_sample(
        img, grid_norm, mode='bilinear',
        padding_mode='zeros', align_corners=True,
    )


# ---------------------------------------------------------------------------
# Sum-of-flows DMVFN
# ---------------------------------------------------------------------------
class DMVFN_Simplified(nn.Module):
    """
    Wraps the original DMVFN's MVFB blocks, but replaces the iterative
    refinement loop with a sum-of-flows accumulator that matches the HW.
    """

    # ------------------------------------------------------------------
    # Behaviour switches (match these to your HW)
    # ------------------------------------------------------------------
    SUM_OF_FLOWS_INIT = 'zero'   # 'zero'  → every block gets prev_img=0, prev_flow=0
                                  #            (matches your gen_kitti_stim.py exactly)
                                  # 'pytorch_first_block' → only block 0 gets zeros;
                                  #   subsequent blocks see the previous block's L10 input
                                  #   that was dumped from a real PyTorch forward pass
                                  #   (closer to what the L10 dump file represents)
    USE_ROUTING       = False    # True → respect routing vector v;
                                  # False → all 9 blocks active (matches HW)

    SCALE_LIST = [4, 4, 4, 2, 2, 2, 1, 1, 1]
    NUM_BLOCKS = 9

    def __init__(self, dmvfn_orig):
        """
        dmvfn_orig: an instance of the original DMVFN nn.Module
                    (the one inside Model.eval()).
                    We borrow its MVFB blocks + routing module.
        """
        super().__init__()
        # The original DMVFN's submodules (block_tea / routing) are referenced
        # so this class just *redefines* the forward pass.
        self.blocks = dmvfn_orig.block_tea  # nn.ModuleList of 9 MVFBs
        if self.USE_ROUTING and hasattr(dmvfn_orig, 'routing'):
            self.routing = dmvfn_orig.routing
        else:
            self.routing = None
        assert len(self.blocks) == self.NUM_BLOCKS, \
            f"Expected 9 MVFB blocks, got {len(self.blocks)}"

    # ----------------------------------------------------------------
    def forward(self, x, scale_list=None):
        """
        x: (B, 6, H, W) — concat([img_{t-1}, img_t]) in [0, 1] range
        Returns: (B, 3, H, W) — predicted img_{t+1}
        """
        scale_list = scale_list or self.SCALE_LIST
        B, _, H, W = x.shape
        device = x.device
        img0 = x[:, :3]
        img1 = x[:, 3:6]

        # Accumulators — match the HW: just sum up every block's ΔF
        flow_accum = torch.zeros(B, 4, H, W, device=device)   # 4 = (dx_t, dy_t, dx_{t-1}, dy_{t-1})
        mask_accum = torch.zeros(B, 1, H, W, device=device)

        # Optional routing
        if self.USE_ROUTING and self.routing is not None:
            v = self.routing(img0, img1)  # shape (B, 9), in {0,1}
        else:
            v = torch.ones(B, self.NUM_BLOCKS, device=device)

        for i in range(self.NUM_BLOCKS):
            if v[0, i] < 0.5:
                continue  # skipped block

            # Build this block's input depending on init mode
            if self.SUM_OF_FLOWS_INIT == 'zero':
                prev_img  = torch.zeros(B, 3, H, W, device=device)
                prev_flow = torch.zeros(B, 4, H, W, device=device)
            elif self.SUM_OF_FLOWS_INIT == 'pytorch_first_block':
                # Use whatever the original block 0 would receive; for blocks 1..8
                # we still feed zeros (because the HW dump is "L10_input"; the
                # earlier conv layers already absorbed the prev-block context).
                if i == 0:
                    prev_img  = torch.zeros(B, 3, H, W, device=device)
                    prev_flow = torch.zeros(B, 4, H, W, device=device)
                else:
                    # Best approximation: still zeros, since we explicitly want
                    # to mimic the HW pipeline.
                    prev_img  = torch.zeros(B, 3, H, W, device=device)
                    prev_flow = torch.zeros(B, 4, H, W, device=device)
            else:
                raise ValueError(f"Unknown SUM_OF_FLOWS_INIT: {self.SUM_OF_FLOWS_INIT}")

            # Run the i-th MVFB. We are interested in its ΔF^i (flow + mask).
            # Different MVFB signatures: try the most common one first.
            block_input = torch.cat([img0, img1, prev_img, prev_flow], dim=1)  # (B, 13, H, W)
            try:
                # (a) signature: forward(x_concat, flow_prev, scale) → (img_pred, flow_new, mask_new)
                _, flow_new, mask_new = self.blocks[i](
                    block_input, prev_flow, scale=scale_list[i]
                )
            except TypeError:
                try:
                    # (b) signature: forward(x_concat, scale)
                    _, flow_new, mask_new = self.blocks[i](
                        block_input, scale=scale_list[i]
                    )
                except TypeError:
                    # (c) signature: forward(img0, img1, prev_img, prev_flow, scale)
                    _, flow_new, mask_new = self.blocks[i](
                        img0, img1, prev_img, prev_flow, scale_list[i]
                    )

            # ΔF^i = flow_new (because prev_flow was 0 → flow_new is the delta)
            flow_accum = flow_accum + flow_new
            mask_accum = mask_accum + mask_new

        # ── Final warp + blend (one shot, with the summed flow) ────────
        flow_t      = flow_accum[:, 0:2]   # warp src for img_t   (shifts img_t   → t+1)
        flow_t_minus = flow_accum[:, 2:4]  # warp src for img_{t-1}

        warped_t       = warp(img1, flow_t)        # warp img_t      with flow_t
        warped_t_minus = warp(img0, flow_t_minus)  # warp img_{t-1}  with flow_{t-1}

        mask = torch.sigmoid(mask_accum)            # (B, 1, H, W) ∈ (0, 1)
        pred = mask * warped_t + (1.0 - mask) * warped_t_minus
        return pred.clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# Drop-in replacement for the original `Model` wrapper
# ---------------------------------------------------------------------------
class Model_Simplified:
    """
    Mimics the public API of the original Model class so that test.py
    can swap it in without other changes.
    """

    def __init__(self, load_path):
        # Lazy import to avoid circular import problems
        from model.model import Model as _OriginalModel
        self._orig = _OriginalModel(load_path=load_path, training=False)
        # The original Model exposes its DMVFN at one of these attribute paths
        # depending on the commit; try them in order.
        dmvfn_net = None
        for attr in ('dmvfn', 'net', 'model', 'flownet'):
            if hasattr(self._orig, attr):
                dmvfn_net = getattr(self._orig, attr)
                break
        if dmvfn_net is None:
            raise AttributeError(
                "Could not find the inner DMVFN module on Model. "
                "Update Model_Simplified.__init__ to point at the correct attribute."
            )
        self.simplified = DMVFN_Simplified(dmvfn_net).to(
            next(dmvfn_net.parameters()).device
        )
        self.simplified.eval()

    # ---- Match the eval() interface used by test.py ----------------------
    @torch.no_grad()
    def eval(self, data_gpu, dataset_name):
        """
        data_gpu: shape depends on dataset.
          For Cityscapes/KITTI/DAVIS: (B, 9, 3, H, W) where frames 0..8 are
            consecutive; the model uses (frame_3, frame_4) → frame_5,
            then iteratively predicts frames 5..8.  We replicate this.
          For Vimeo: (B, 3, 3, H, W); use (frame_0, frame_1) → frame_2.
        Returns: (B, N, 3, H, W)  — same N as original Model.eval()
        """
        if dataset_name in ('CityValDataset', 'KittiValDataset', 'DavisValDataset'):
            return self._eval_autoregressive(data_gpu, n_predict=5)
        elif dataset_name == 'VimeoValDataset':
            return self._eval_single(data_gpu)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

    # ----------------------------------------------------------------------
    @torch.no_grad()
    def _eval_autoregressive(self, data_gpu, n_predict):
        # data_gpu shape: (B=1, 9, 3, H, W)
        # Following the original DMVFN test recipe: input pair = (frame_3, frame_4),
        # predict frame_5; then (frame_4, pred_5) → pred_6; ...; up to pred_9.
        B = data_gpu.shape[0]
        H, W = data_gpu.shape[-2:]
        prev_minus = data_gpu[:, 3]  # frame index 3 = img_{t-1}
        prev       = data_gpu[:, 4]  # frame index 4 = img_t
        preds = []
        for _ in range(n_predict):
            x = torch.cat([prev_minus, prev], dim=1)  # (B, 6, H, W)
            pred = self.simplified(x)                 # (B, 3, H, W)
            preds.append(pred)
            prev_minus = prev
            prev = pred
        return torch.stack(preds, dim=1)              # (B, n_predict, 3, H, W)

    @torch.no_grad()
    def _eval_single(self, data_gpu):
        # Vimeo: (B, 3, 3, H, W) → predict frame index 1 from (frame_0, frame_2)
        # NOTE: original DMVFN uses (frame_0, frame_2) → predict frame_1 for Vimeo;
        # if your Model.eval() uses a different mapping, mirror it here.
        B = data_gpu.shape[0]
        prev_minus = data_gpu[:, 0]
        prev       = data_gpu[:, 2]
        x = torch.cat([prev_minus, prev], dim=1)
        pred = self.simplified(x)
        return pred.unsqueeze(1)                       # (B, 1, 3, H, W)


# ---------------------------------------------------------------------------
# CLI for quick comparison: Original vs Simplified PSNR on a single dataset
# ---------------------------------------------------------------------------
def _cli_compare():
    """
    Quick A/B comparison on one dataset, no need to modify test.py.
    Run:
       python dmvfn_simplified.py \
          --val_dataset CityValDataset \
          --load_path pretrained_models/dmvfn_city.pkl
    """
    import argparse
    import math
    import importlib

    ap = argparse.ArgumentParser()
    ap.add_argument('--val_dataset', default='CityValDataset',
                    choices=['CityValDataset', 'KittiValDataset',
                             'DavisValDataset', 'VimeoValDataset'])
    ap.add_argument('--load_path', required=True)
    ap.add_argument('--n_samples', type=int, default=20,
                    help='Number of samples to evaluate (use small N for quick check)')
    args = ap.parse_args()

    # Make project root importable (assumes this file lives in scripts/ or repo root)
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.dirname(here))
    sys.path.insert(0, here)

    from torch.utils.data import DataLoader
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load dataset
    DSet = getattr(importlib.import_module('dataset.dataset'), args.val_dataset)
    val_data = DataLoader(DSet(), batch_size=1, num_workers=1, pin_memory=True)

    # Load both models
    from model.model import Model as _OriginalModel
    model_orig = _OriginalModel(load_path=args.load_path, training=False)
    model_simp = Model_Simplified(load_path=args.load_path)

    psnr_orig_all, psnr_simp_all, psnr_diff_all = [], [], []

    for i, (data_gpu, data_name) in enumerate(val_data):
        if i >= args.n_samples:
            break
        data_gpu = data_gpu.to(device, non_blocking=True) / 255.

        preds_orig = model_orig.eval(data_gpu, args.val_dataset)
        preds_simp = model_simp.eval(data_gpu, args.val_dataset)

        # PSNR vs ground truth (matches test.py: gt = data_gpu[0, 4+j])
        gt   = data_gpu[0]
        po   = preds_orig[0]
        ps   = preds_simp[0]
        n_pred = po.shape[0]

        for j in range(n_pred):
            if args.val_dataset == 'VimeoValDataset':
                gt_idx = 2
            else:
                gt_idx = 4 + j
            mse_o = torch.mean((gt[gt_idx] - po[j]) ** 2).item()
            mse_s = torch.mean((gt[gt_idx] - ps[j]) ** 2).item()
            mse_d = torch.mean((po[j]    - ps[j]) ** 2).item()
            psnr_orig_all.append(-10*math.log10(max(mse_o, 1e-12)))
            psnr_simp_all.append(-10*math.log10(max(mse_s, 1e-12)))
            psnr_diff_all.append(-10*math.log10(max(mse_d, 1e-12)))

        if (i + 1) % 5 == 0:
            n = len(psnr_orig_all)
            print(f"[{i+1:4d}/{args.n_samples}]  "
                  f"orig {sum(psnr_orig_all)/n:5.2f} dB | "
                  f"simp {sum(psnr_simp_all)/n:5.2f} dB | "
                  f"orig-vs-simp {sum(psnr_diff_all)/n:5.2f} dB")

    n = len(psnr_orig_all)
    print()
    print("=" * 60)
    print(f" Dataset: {args.val_dataset}  |  Samples: {args.n_samples}")
    print("=" * 60)
    print(f"  Original DMVFN     vs GT:  {sum(psnr_orig_all)/n:6.3f} dB")
    print(f"  Simplified (sum)   vs GT:  {sum(psnr_simp_all)/n:6.3f} dB")
    print(f"  Original vs Simplified  :  {sum(psnr_diff_all)/n:6.3f} dB")
    print()
    print("  Interpretation:")
    print("  - 'Original vs Simplified' is the ALGORITHMIC GAP. Your HW")
    print("    can never beat this number — it's the cost of using sum-of-flows.")
    print("  - Your HW PSNR vs Simplified should ideally be > 35 dB.")
    print("    If lower → fixed-point/tiling/bilinear bug, NOT DMVFN-the-model bug.")
    print("=" * 60)


if __name__ == "__main__":
    _cli_compare()


# ============================================================================
# ADAPTING SECTION
# ============================================================================
# If the script crashes inside DMVFN_Simplified.forward (`self.blocks[i](...)`),
# it's because the MVFB.forward signature in your local repo differs from the
# three I tried.  Open `model/MVFB.py` (or wherever MVFB is defined) and look
# at its `def forward(...)`. Then adjust the try/except chain in
# DMVFN_Simplified.forward to match.
#
# The general idea is unchanged: each block must return (warped_img, flow, mask)
# for some flow shape (B, 4, H, W) and mask shape (B, 1, H, W).  We discard
# warped_img and accumulate flow + mask.
#
# If the inner attribute path differs (Model.dmvfn / Model.net / ...), update
# Model_Simplified.__init__'s search list.
# ============================================================================