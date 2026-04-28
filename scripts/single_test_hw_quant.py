#!/usr/bin/env python3
"""
single_test_hw_quant.py — single image pair interpolation with hw_quant mode
=============================================================================
Runs the same fixed-point simulation as sim_dmvfn.py (weight×256 INT MAC,
Q11.10 flow with float coord bilinear warp, linear sigmoid MASK_SHIFT=7)
on a single pair of input frames.

Usage:
  python scripts/single_test_hw_quant.py \
      --image_0_path  path/to/frame0.png \
      --image_1_path  path/to/frame1.png \
      --load_path     pretrained_models/dmvfn_kitti.pkl \
      --output_path   pred_hw_quant.png

Note:
  The output pixel values match sim_dmvfn.py within ±1 LSB.
  The only difference is warp coordinate precision:
    hw_quant  : float coords (after floor(acc >> FLOW_SHIFT) / FRAC_ONE)
    sim_dmvfn : Q11.10 integer coords (full RTL-equivalent)
"""

import os
import sys
import cv2
import torch
import random
import argparse
import numpy as np

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_path)
sys.path.insert(0, os.path.join(root_path, 'scripts'))

from dmvfn_simplified import DMVFN_Simplified, Model_Simplified

DMVFN_Simplified.MODE = 'hw_quant'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
seed = 1234
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def evaluate(model, args):
    with torch.no_grad():
        img0 = cv2.imread(args.image_0_path)
        img1 = cv2.imread(args.image_1_path)
        if img0 is None or img1 is None:
            raise FileNotFoundError("One or both input images not found.")

        t0 = torch.from_numpy(img0.transpose(2, 0, 1).astype('float32'))
        t1 = torch.from_numpy(img1.transpose(2, 0, 1).astype('float32'))
        imgs = torch.cat([t0, t1], dim=0).unsqueeze(0).unsqueeze(0).to(device) / 255.

        pred = model.eval(imgs, 'single_test')   # (1, 3, H, W) float [0,1]
        pred_np = (pred.squeeze().cpu().numpy() * 255).transpose(1, 2, 0).astype(np.uint8)
        cv2.imwrite(args.output_path, pred_np)
        print(f"Saved: {args.output_path}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--image_0_path', required=True,  type=str)
    ap.add_argument('--image_1_path', required=True,  type=str)
    ap.add_argument('--load_path',    required=True,  type=str)
    ap.add_argument('--output_path',  default='pred_hw_quant.png', type=str)
    args = ap.parse_args()

    model = Model_Simplified(load_path=args.load_path, training=False)
    evaluate(model, args)
