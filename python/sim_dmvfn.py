#!/usr/bin/env python3
"""
sim_dmvfn.py -- Pure-Python fixed-point simulation of DMVFN 9-block pipeline
==============================================================================
Reads dumped FM + weights from MVFB/stim_data/{dataset}/ and simulates the
full RTL-equivalent fixed-point forward pass, outputting the final blended frame.

Replaces the need to run RTL simulation every time you want to check output
quality. Runs in ~1-2 minutes on CPU instead of ~1 hour for full RTL sim.

Usage:
  cd DMVFN_sram
  python python/sim_dmvfn.py --dataset kitti
  python python/sim_dmvfn.py --dataset cityscapes

Inputs (flat directory, e.g. MVFB/stim_data/KITTI/):
  kitti_b{i}_L10_input.txt        FM for block i   (ic_padded*FM_H rows)
  kitti_b{i}_lastconv_weight.txt  weights           (KH*KW*OC_FILE rows)
  kitti_img{0,1}_bank{0-3}_{R,G,B}.txt  source image SRAM (RTL-addr order)
  kitti_config.txt                dataset parameters
  kitti_tile_params.txt           tile (t,ty,tx,ih,iw,oh,ow,px,bx,by,fshift)
  kitti_golden_blend_{R,G,B}.txt  golden blend (optional, for comparison)

Outputs:
  python/sim_out/{dataset}/sim_final.png
  python/sim_out/{dataset}/sim_comparison.png   (src0 | sim | src1)
"""

import numpy as np
from PIL import Image
import os, sys, time, argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(SCRIPT_DIR)

# ---------------------------------------------------------------------------
# Fixed-point constants (shared for both datasets)
# ---------------------------------------------------------------------------
COORD_FRAC_W = 10
FRAC_ONE     = 1 << COORD_FRAC_W   # 1024
FRAC_MASK    = FRAC_ONE - 1
FLOW_SHIFT   = 16 - COORD_FRAC_W   # 6
KH = KW      = 4
STRIDE       = 2
TILE_STRIDE  = 4
TILE_IH = TILE_IW = TILE_STRIDE + 2   # 6
TILE_OH = TILE_OW = (TILE_IH - 1) * STRIDE + KH   # 14
TILE_PX      = TILE_OH * TILE_OW   # 196
VALID_H = VALID_W = 8
VALID_OFFSET = 3
OC_FILE      = 8   # number of OC entries in weight file (first 5 used)
OC_USE       = 5   # flow_x0, flow_y0, flow_x1, flow_y1, mask

# Per-dataset name prefixes
DATASET_PREFIX = {
    'kitti':      'kitti',
    'cityscapes': 'city',
}

# Per-dataset block IC config: (ic_actual, ic_padded) for each group
# G1=blocks 0-2, G2=blocks 3-5, G3=blocks 6-8
# Both KITTI and Cityscapes share the same IC structure (same model architecture)
BLOCK_IC = {
    'kitti': {
        0: (48, 64), 1: (48, 64), 2: (48, 64),
        3: (28, 32), 4: (28, 32), 5: (28, 32),
        6: (19, 32), 7: (19, 32), 8: (19, 32),
    },
    'cityscapes': {
        0: (48, 64), 1: (48, 64), 2: (48, 64),
        3: (28, 32), 4: (28, 32), 5: (28, 32),
        6: (19, 32), 7: (19, 32), 8: (19, 32),
    },
}


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
def load_config(filepath):
    """Parse kitti_config.txt / city_config.txt.
    Format: G1_IC G2_IC G3_IC OC KH KW STRIDE CROP_H CROP_W
            NUM_TILES TOTAL_BLEND_PX MASK_SHIFT SRC_W SRC_H NUM_BLOCKS OUT_DEPTH
    Returns dict.
    """
    with open(filepath) as f:
        vals = list(map(int, f.read().split()))
    keys = ['g1_ic','g2_ic','g3_ic','oc','kh','kw','stride',
            'crop_h','crop_w','num_tiles','total_blend_px','mask_shift',
            'src_w','src_h','num_blocks','out_depth']
    return dict(zip(keys, vals))


def load_tile_params(filepath):
    """Load tile_params.txt → list of dicts."""
    tiles = []
    with open(filepath) as f:
        for line in f:
            parts = list(map(int, line.split()))
            if len(parts) < 11:
                continue
            tiles.append({
                't':  parts[0], 'ty': parts[1], 'tx': parts[2],
                'ih': parts[3], 'iw': parts[4],
                'oh': parts[5], 'ow': parts[6], 'px': parts[7],
                'bx': parts[8], 'by': parts[9], 'fshift': parts[10],
            })
    return tiles


def load_fm(filepath, ic_padded, fm_h, fm_w):
    """Load FM file (ic_padded*fm_h rows, each row has fm_w values).
    Returns (ic_padded, fm_h, fm_w) int64.
    """
    with open(filepath) as f:
        lines = [l.strip() for l in f if l.strip()]
    assert len(lines) == ic_padded * fm_h, (
        f"{os.path.basename(filepath)}: expected {ic_padded*fm_h} lines, got {len(lines)}")
    fm = np.zeros((ic_padded, fm_h, fm_w), dtype=np.int64)
    for idx, line in enumerate(lines):
        c, y = divmod(idx, fm_h)
        vals = list(map(int, line.split()))
        fm[c, y, :len(vals)] = vals[:fm_w]
    return fm


def load_weight(filepath, ic_padded, kh, kw, oc_use=OC_USE, oc_file=OC_USE):
    """Load weight file (kh*kw*oc_file rows, each row has ic_padded values).
    Returns (ic_padded, oc_use, kh, kw) int64.
    """
    with open(filepath) as f:
        lines = [l.strip() for l in f if l.strip()]
    assert len(lines) == kh * kw * oc_file, (
        f"{os.path.basename(filepath)}: expected {kh*kw*oc_file} lines, got {len(lines)}")
    weight = np.zeros((ic_padded, oc_use, kh, kw), dtype=np.int64)
    idx = 0
    for k in range(kh * kw):
        ky, kx = k // kw, k % kw
        for oc in range(oc_file):
            vals = list(map(int, lines[idx].split()))
            if oc < oc_use:
                for ic in range(min(len(vals), ic_padded)):
                    weight[ic, oc, ky, kx] = vals[ic]
            idx += 1
    return weight


def load_image_sram(data_dir, prefix, img_idx, src_h, src_w):
    """Load RTL-addr-ordered image bank files.
    Files: {prefix}_img{img_idx}_bank{0-3}_{R,G,B}.txt
    Each file has BANK_ADDR_DEPTH lines of 2-hex-digit values.
    Returns (src_h, src_w, 3) uint8 RGB.
    """
    bank_dim_w = (src_w // 2 - 1).bit_length()

    # Load all 4 banks × 3 channels → addr-indexed arrays
    banks = {}  # (bn, ch) → np.array[addr]
    for bn in range(4):
        for ch, ch_name in enumerate(['R', 'G', 'B']):
            fpath = os.path.join(data_dir,
                                 f"{prefix}_img{img_idx}_bank{bn}_{ch_name}.txt")
            with open(fpath) as f:
                vals = [int(l.strip(), 16) for l in f if l.strip()]
            banks[(bn, ch)] = np.array(vals, dtype=np.uint8)

    img = np.zeros((src_h, src_w, 3), dtype=np.uint8)
    for y in range(src_h):
        for x in range(src_w):
            bn   = (y % 2) * 2 + (x % 2)
            bx   = x >> 1
            by_i = y >> 1
            addr = (by_i << bank_dim_w) | bx
            for ch in range(3):
                arr = banks[(bn, ch)]
                img[y, x, ch] = arr[addr] if addr < len(arr) else 0
    return img


def build_sram_from_image(img_hw, src_w):
    """(H, W) uint8 → list of 4 dicts {addr: int} matching RTL bank layout."""
    H, W = img_hw.shape
    bank_dim_w = (src_w // 2 - 1).bit_length()
    sram = [dict() for _ in range(4)]
    for y in range(H):
        for x in range(W):
            bn   = (y % 2) * 2 + (x % 2)
            addr = ((y // 2) << bank_dim_w) | (x // 2)
            sram[bn][addr] = int(img_hw[y, x])
    return sram


def build_sram_from_rtl_files(data_dir, prefix, img_idx, src_w):
    """Load 4 banks per channel directly from RTL addr-ordered files.
    Returns (sram_channels, bank_dim_w, bank_addr_w).
    sram_channels: list of 3 channels, each a list of 4 dicts {addr: int}.
    bank_addr_w: derived from file line count (log2).
    """
    bank_dim_w = (src_w // 2 - 1).bit_length()
    sram_channels = []
    bank_addr_w = 16   # default, updated from file line count
    for ch, ch_name in enumerate(['R', 'G', 'B']):
        sram = []
        for bn in range(4):
            fpath = os.path.join(data_dir,
                                 f"{prefix}_img{img_idx}_bank{bn}_{ch_name}.txt")
            d = {}
            with open(fpath) as f:
                lines = f.readlines()
            # Derive bank_addr_w from number of entries (first channel, first bank)
            if ch == 0 and bn == 0:
                import math
                n_entries = sum(1 for l in lines if l.strip())
                bank_addr_w = int(math.log2(n_entries))
            for addr, line in enumerate(lines):
                line = line.strip()
                if line:
                    d[addr] = int(line, 16)
            sram.append(d)
        sram_channels.append(sram)
    return sram_channels, bank_dim_w, bank_addr_w


# ---------------------------------------------------------------------------
# TConv2d (tile-wise, matching RTL)
# ---------------------------------------------------------------------------
def tconv2d_tile(tile_fm, weight):
    """(ic, TILE_IH, TILE_IW) × (ic, oc, kh, kw) → (oc, TILE_OH, TILE_OW) int64."""
    n_ic, ih, iw = tile_fm.shape
    _, n_oc, kh, kw = weight.shape
    oh = (ih - 1) * STRIDE + kh
    ow = (iw - 1) * STRIDE + kw
    out = np.zeros((n_oc, oh, ow), dtype=np.int64)
    for ky in range(kh):
        for kx in range(kw):
            w = weight[:, :, ky, kx]   # (ic, oc)
            contrib = np.einsum('io,ihw->ohw',
                                w.astype(np.int64), tile_fm.astype(np.int64))
            out[:, ky:ky + ih*STRIDE:STRIDE,
                   kx:kx + iw*STRIDE:STRIDE] += contrib
    return out


def extract_tile_fm(fm, ty, tx, fm_h, fm_w):
    """Extract (ic, TILE_IH, TILE_IW) tile patch, zero-pad OOB."""
    ic = fm.shape[0]
    row_start = ty * TILE_STRIDE - 1
    col_start = tx * TILE_STRIDE - 1
    tile = np.zeros((ic, TILE_IH, TILE_IW), dtype=np.int64)
    r0 = max(row_start, 0);  r1 = min(row_start + TILE_IH, fm_h)
    c0 = max(col_start, 0);  c1 = min(col_start + TILE_IW, fm_w)
    ri0 = r0 - row_start;    ri1 = ri0 + (r1 - r0)
    ci0 = c0 - col_start;    ci1 = ci0 + (c1 - c0)
    tile[:, ri0:ri1, ci0:ci1] = fm[:, r0:r1, c0:c1]
    return tile


# ---------------------------------------------------------------------------
# Fixed-point warp helpers (matching RTL)
# ---------------------------------------------------------------------------
def to_signed_32(val):
    val = int(val) & 0xFFFFFFFF
    if val >= (1 << 31):
        val -= (1 << 32)
    return val


def compute_warp_coord(flow_raw, fshift, base, pixel_pos, coord_mask):
    s       = to_signed_32(flow_raw)
    shifted = (s >> fshift) & coord_mask
    px_q    = (pixel_pos & ((coord_mask >> COORD_FRAC_W))) << COORD_FRAC_W
    return (base + px_q + shifted) & coord_mask


def bilinear_hw(sram_banks, coord_x, coord_y, coord_int_w, bank_dim_w,
                bank_addr_w=17, src_w=832, src_h=256):
    """Match RTL bilinear interpolation with border-mode clamping.

    Matches Backward_Warping_Top RTL behaviour: coordinates are clamped to
    [0, (SRC-1)<<FRAC] before address/weight generation, equivalent to
    PyTorch grid_sample(padding_mode='border', align_corners=True).
    """
    COORD_W = coord_int_w + COORD_FRAC_W

    # Sign-extend full coordinate
    cx_s = coord_x if coord_x < (1 << (COORD_W - 1)) else coord_x - (1 << COORD_W)
    cy_s = coord_y if coord_y < (1 << (COORD_W - 1)) else coord_y - (1 << COORD_W)

    # Border-mode clamp to [0, (SRC-1)<<FRAC]
    max_cx = (src_w - 1) << COORD_FRAC_W
    max_cy = (src_h - 1) << COORD_FRAC_W
    cx_c = max(0, min(cx_s, max_cx))
    cy_c = max(0, min(cy_s, max_cy))

    x_int = cx_c >> COORD_FRAC_W
    y_int = cy_c >> COORD_FRAC_W
    alpha = cx_c & FRAC_MASK
    beta  = cy_c & FRAC_MASK
    ia, ib = FRAC_ONE - alpha, FRAC_ONE - beta

    x0, x1 = x_int, x_int + 1
    y0, y1 = y_int, y_int + 1

    def read_px(x, y):
        # Border clamp for edge neighbours (weight = 0 in these cases)
        x = max(0, min(x, src_w - 1))
        y = max(0, min(y, src_h - 1))
        bn   = (y & 1) * 2 + (x & 1)
        bx   = (x >> 1) & ((1 << bank_dim_w) - 1)
        by   = (y >> 1) & ((1 << bank_dim_w) - 1)
        addr = (by << bank_dim_w) | bx
        return sram_banks[bn].get(addr, 0)

    p_TL = read_px(x0, y0)
    p_TR = read_px(x1, y0)
    p_BL = read_px(x0, y1)
    p_BR = read_px(x1, y1)

    s = p_TL*ia*ib + p_TR*alpha*ib + p_BL*ia*beta + p_BR*alpha*beta
    return ((s + (1 << (2*COORD_FRAC_W - 1))) >> (2*COORD_FRAC_W)) & 0xFFFF


# ---------------------------------------------------------------------------
# Reconstruct final image from golden blend
# ---------------------------------------------------------------------------
def reconstruct_from_golden(golden_r, golden_g, golden_b, tiles,
                             target_h, target_w):
    img = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    pixel_idx = 0
    for tile in tiles:
        ty, tx = tile['ty'], tile['tx']
        oh, ow, n_px = tile['oh'], tile['ow'], tile['px']
        tile_r = np.array(golden_r[pixel_idx:pixel_idx+n_px],
                          dtype=np.uint8).reshape(oh, ow)
        tile_g = np.array(golden_g[pixel_idx:pixel_idx+n_px],
                          dtype=np.uint8).reshape(oh, ow)
        tile_b = np.array(golden_b[pixel_idx:pixel_idx+n_px],
                          dtype=np.uint8).reshape(oh, ow)
        pixel_idx += n_px
        out_y = ty * VALID_H
        out_x = tx * VALID_W
        if out_y + VALID_H <= target_h and out_x + VALID_W <= target_w:
            img[out_y:out_y+VALID_H, out_x:out_x+VALID_W, 0] = \
                tile_r[VALID_OFFSET:VALID_OFFSET+VALID_H, VALID_OFFSET:VALID_OFFSET+VALID_W]
            img[out_y:out_y+VALID_H, out_x:out_x+VALID_W, 1] = \
                tile_g[VALID_OFFSET:VALID_OFFSET+VALID_H, VALID_OFFSET:VALID_OFFSET+VALID_W]
            img[out_y:out_y+VALID_H, out_x:out_x+VALID_W, 2] = \
                tile_b[VALID_OFFSET:VALID_OFFSET+VALID_H, VALID_OFFSET:VALID_OFFSET+VALID_W]
    return img


def compute_psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    return 10.0 * np.log10(255.0**2 / mse)


# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------
def simulate(dataset, out_dir, use_routing=False):
    t0 = time.time()

    prefix   = DATASET_PREFIX[dataset]
    subdir   = 'KITTI' if dataset == 'kitti' else 'cityscapes'
    data_dir = os.path.join(ROOT_DIR, 'MVFB', 'stim_data', subdir)
    os.makedirs(out_dir, exist_ok=True)

    # ── 1. Load config ────────────────────────────────────────────────────
    cfg_path = os.path.join(data_dir, f"{prefix}_config.txt")
    if not os.path.exists(cfg_path):
        print(f"ERROR: {cfg_path} not found"); sys.exit(1)
    cfg = load_config(cfg_path)
    SRC_H  = cfg['src_h']
    SRC_W  = cfg['src_w']
    FM_H   = cfg['crop_h']
    FM_W   = cfg['crop_w']
    MASK_SHIFT   = cfg['mask_shift']
    NUM_TILES    = cfg['num_tiles']
    TOTAL_BLEND  = cfg['total_blend_px']
    # COORD_INT_W must match the gen scripts (KITTI=11, Cityscapes=12).
    # Derived from max source dimension: bit_length() + 1 sign/guard bit.
    COORD_INT_W  = max(SRC_W, SRC_H).bit_length() + 1
    INT_MASK     = (1 << COORD_INT_W) - 1
    BANK_DIM_W   = (SRC_W // 2 - 1).bit_length()
    TILES_Y = FM_H // TILE_STRIDE
    TILES_X = FM_W // TILE_STRIDE

    # ── Load routing_ref (optional) ──────────────────────────────────────
    routing_ref = [1] * 9
    if use_routing:
        src_subdir = 'cityscapes' if dataset == 'cityscapes' else 'KITTI'
        rref_path = os.path.join(ROOT_DIR, src_subdir, 'phase15', 'routing_ref.txt')
        if os.path.exists(rref_path):
            with open(rref_path) as f:
                routing_ref = [int(x) for x in f.read().split()]
            print(f"  routing_ref loaded: {routing_ref}  ({sum(routing_ref)}/9 blocks active)")
        else:
            print(f"  [WARN] routing_ref.txt not found at {rref_path}, using all 9 blocks")

    print("=" * 65)
    print(f" DMVFN Python Simulator — {dataset.upper()}"
          + (" [routing]" if use_routing else " [all-9-blocks]"))
    print(f"   SRC: {SRC_H}x{SRC_W}  FM: {FM_H}x{FM_W}")
    print(f"   Tiles: {TILES_Y}x{TILES_X} = {NUM_TILES}")
    print(f"   MASK_SHIFT={MASK_SHIFT}  COORD_INT_W={COORD_INT_W}")
    print("=" * 65)

    # ── 2. Load tile params ───────────────────────────────────────────────
    tp_path = os.path.join(data_dir, f"{prefix}_tile_params.txt")
    tiles = load_tile_params(tp_path)
    print(f"  Loaded {len(tiles)} tile params")

    # ── 3. Load FMs and weights ───────────────────────────────────────────
    print("\nLoading FMs and weights...")
    block_ic = BLOCK_IC[dataset]
    fms, weights, biases = [], [], []  # 🌟 新增 biases 陣列
    for bidx in range(9):
        ic_act, ic_pad = block_ic[bidx]
        fm_path = os.path.join(data_dir, f"{prefix}_b{bidx}_L10_input.txt")
        wt_path = os.path.join(data_dir, f"{prefix}_b{bidx}_lastconv_weight.txt")
        bias_path = os.path.join(data_dir, f"{prefix}_b{bidx}_L10_bias.txt") # 🌟 Bias 路徑

        fms.append(load_fm(fm_path, ic_pad, FM_H, FM_W))
        weights.append(load_weight(wt_path, ic_pad, KH, KW))

        # 🌟 新增：讀取 Bias，如果檔案不存在就預設給 0
        if os.path.exists(bias_path):
            with open(bias_path, 'r') as f:
                bias_data = [int(x) for x in f.read().strip().split()]
            biases.append(bias_data)
        else:
            biases.append([0] * OC_USE)

        print(f"  Block{bidx}: FM{fms[-1].shape}  wt{weights[-1].shape}"
              f"  range=[{int(fms[-1][:ic_act].min())}, {int(fms[-1][:ic_act].max())}]")

    # ── 4. Load source images as float numpy arrays ───────────────────────
    print("\nLoading source images from SRAM bank files...")
    sram0_channels, bank_dim_w, bank_addr_w = build_sram_from_rtl_files(
        data_dir, prefix, 0, SRC_W)
    sram1_channels, _, _                    = build_sram_from_rtl_files(
        data_dir, prefix, 1, SRC_W)
    print(f"  SRAM loaded (4 banks × 3 ch, BANK_DIM_W={bank_dim_w}, BANK_ADDR_W={bank_addr_w})")

    def sram_to_img(sram_ch):
        img = np.zeros((SRC_H, SRC_W, 3), dtype=np.uint8)
        for y in range(SRC_H):
            for x in range(SRC_W):
                bn   = (y % 2) * 2 + (x % 2)
                bx_  = x >> 1
                by_  = y >> 1
                addr = (by_ << bank_dim_w) | bx_
                for c in range(3):
                    img[y, x, c] = sram_ch[c][bn].get(addr, 0)
        return img

    print("  Reconstructing source images from SRAM...")
    img0_rgb = sram_to_img(sram0_channels)
    img1_rgb = sram_to_img(sram1_channels)
    Image.fromarray(img0_rgb).save(os.path.join(out_dir, 'src_img0.png'))
    Image.fromarray(img1_rgb).save(os.path.join(out_dir, 'src_img1.png'))
    print(f"  Saved src_img0.png, src_img1.png")

    COORD_W    = COORD_INT_W + COORD_FRAC_W
    COORD_MASK = (1 << COORD_W) - 1

    # ── 5. Per-tile tconv + fixed-point warp + linear sigmoid blend ──────────
    print(f"\nProcessing {NUM_TILES} tiles (tconv + fp warp + linear sigmoid)...")

    # Per-channel SRAM banks for fixed-point bilinear warp
    sram0 = []
    sram1 = []
    for ch in range(3):
        sram0.append(sram0_channels[ch])
        sram1.append(sram1_channels[ch])

    out_img = np.zeros((SRC_H, SRC_W, 3), dtype=np.uint8)
    sim_R, sim_G, sim_B = [], [], []

    for tile in tiles:
        ty    = tile['ty']
        tx    = tile['tx']
        bx    = tile['bx']
        by    = tile['by']
        fshift = tile['fshift']

        raw_outs = []
        for bidx in range(9):
            if not routing_ref[bidx]:
                # Block skipped by routing: contribute zero ΔF/Δmask
                raw_outs.append(np.zeros((OC_USE, TILE_OH, TILE_OW), dtype=np.int64))
                continue
            tile_fm = extract_tile_fm(fms[bidx], ty, tx, FM_H, FM_W)
            raw     = tconv2d_tile(tile_fm, weights[bidx])
            bias_q16 = np.array(biases[bidx], dtype=np.int64) * 256
            raw += bias_q16[:, None, None]
            raw     = np.clip(raw, -(1 << 31), (1 << 31) - 1).astype(np.int64)
            raw_outs.append(raw)

        acc_flow = sum(r[:4] for r in raw_outs)
        acc_flow = np.clip(acc_flow, -(1 << 31), (1 << 31) - 1).astype(np.int64)
        acc_mask = sum(r[4:5] for r in raw_outs)
        acc_mask = np.clip(acc_mask, -(1 << 31), (1 << 31) - 1).astype(np.int64)

        out_y0 = ty * VALID_H
        out_x0 = tx * VALID_W

        tile_pixels_R = np.zeros((TILE_OH, TILE_OW), dtype=np.uint8)
        tile_pixels_G = np.zeros((TILE_OH, TILE_OW), dtype=np.uint8)
        tile_pixels_B = np.zeros((TILE_OH, TILE_OW), dtype=np.uint8)

        for py in range(TILE_OH):
            for px in range(TILE_OW):
                fx_t   = int(acc_flow[0, py, px])
                fy_t   = int(acc_flow[1, py, px])
                fx_t1  = int(acc_flow[2, py, px])
                fy_t1  = int(acc_flow[3, py, px])
                mask_v = int(acc_mask[0, py, px])

                # Q11.10 warp coordinates — matches gen_city_stim compute_warp_coord
                s0x = to_signed_32(fx_t);  s0y = to_signed_32(fy_t)
                s1x = to_signed_32(fx_t1); s1y = to_signed_32(fy_t1)
                px_q = (px & (INT_MASK)) << COORD_FRAC_W
                py_q = (py & (INT_MASK)) << COORD_FRAC_W
                cx_t  = (bx + px_q + ((s0x >> fshift) & COORD_MASK)) & COORD_MASK
                cy_t  = (by + py_q + ((s0y >> fshift) & COORD_MASK)) & COORD_MASK
                cx_t1 = (bx + px_q + ((s1x >> fshift) & COORD_MASK)) & COORD_MASK
                cy_t1 = (by + py_q + ((s1y >> fshift) & COORD_MASK)) & COORD_MASK

                # Fixed-point integer bilinear warp (matches golden / RTL)
                wR0 = bilinear_hw(sram0[0], cx_t,  cy_t,  COORD_INT_W, BANK_DIM_W, src_w=SRC_W, src_h=SRC_H)
                wG0 = bilinear_hw(sram0[1], cx_t,  cy_t,  COORD_INT_W, BANK_DIM_W, src_w=SRC_W, src_h=SRC_H)
                wB0 = bilinear_hw(sram0[2], cx_t,  cy_t,  COORD_INT_W, BANK_DIM_W, src_w=SRC_W, src_h=SRC_H)
                wR1 = bilinear_hw(sram1[0], cx_t1, cy_t1, COORD_INT_W, BANK_DIM_W, src_w=SRC_W, src_h=SRC_H)
                wG1 = bilinear_hw(sram1[1], cx_t1, cy_t1, COORD_INT_W, BANK_DIM_W, src_w=SRC_W, src_h=SRC_H)
                wB1 = bilinear_hw(sram1[2], cx_t1, cy_t1, COORD_INT_W, BANK_DIM_W, src_w=SRC_W, src_h=SRC_H)

                # Linear sigmoid approximation (matches golden/RTL: MASK_SHIFT)
                sm_val   = to_signed_32(mask_v)
                mask_sh  = sm_val >> MASK_SHIFT
                mask_q   = min(max(mask_sh + (FRAC_ONE >> 1), 0), FRAC_ONE)
                comp     = FRAC_ONE - mask_q

                blR = ((wR0 * mask_q + wR1 * comp) >> COORD_FRAC_W) & 0xFF
                blG = ((wG0 * mask_q + wG1 * comp) >> COORD_FRAC_W) & 0xFF
                blB = ((wB0 * mask_q + wB1 * comp) >> COORD_FRAC_W) & 0xFF

                tile_pixels_R[py, px] = blR
                tile_pixels_G[py, px] = blG
                tile_pixels_B[py, px] = blB

                # Write only the VALID_H×VALID_W interior pixels into output
                if (VALID_OFFSET <= py < VALID_OFFSET + VALID_H and
                        VALID_OFFSET <= px < VALID_OFFSET + VALID_W):
                    oy = out_y0 + (py - VALID_OFFSET)
                    ox = out_x0 + (px - VALID_OFFSET)
                    if 0 <= oy < SRC_H and 0 <= ox < SRC_W:
                        out_img[oy, ox] = (blR, blG, blB)

        for v in tile_pixels_R.flatten(): sim_R.append(int(v))
        for v in tile_pixels_G.flatten(): sim_G.append(int(v))
        for v in tile_pixels_B.flatten(): sim_B.append(int(v))

        t_idx = tile['t']
        if t_idx % 200 == 0 or t_idx == NUM_TILES - 1:
            print(f"  tile {t_idx+1}/{NUM_TILES}  [{time.time()-t0:.1f}s]")

    sim_img = out_img

    # ── 6. Save output image ──────────────────────────────────────────────
    sim_suffix = '_routing' if use_routing else ''
    sim_out = os.path.join(out_dir, f'sim_final{sim_suffix}.png')
    Image.fromarray(sim_img).save(sim_out)
    print(f"\n  Saved: {sim_out}")

    # ── 7. Compare vs golden (if available) ───────────────────────────────
    golden_exists = all(
        os.path.exists(os.path.join(data_dir, f"{prefix}_golden_blend_{ch}.txt"))
        for ch in ('R', 'G', 'B')
    )
    if golden_exists:
        print("\nComparing vs golden blend...")
        golden_r, golden_g, golden_b = [], [], []
        for ch_name, lst in [('R', golden_r), ('G', golden_g), ('B', golden_b)]:
            fpath = os.path.join(data_dir, f"{prefix}_golden_blend_{ch_name}.txt")
            with open(fpath) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        lst.append(int(line, 16))

        golden_img = reconstruct_from_golden(
            golden_r, golden_g, golden_b, tiles, SRC_H, SRC_W)

        diff = np.abs(sim_img.astype(np.int16) - golden_img.astype(np.int16))
        n_total = SRC_H * SRC_W * 3
        n_exact = int(np.sum(diff == 0))
        n_off1  = int(np.sum(diff == 1))
        n_offgt1= int(np.sum(diff > 1))
        max_dif = int(diff.max())
        psnr    = compute_psnr(sim_img, golden_img)

        print(f"  Exact match:  {n_exact}/{n_total} ({100*n_exact/n_total:.1f}%)")
        print(f"  Off-by-1:     {n_off1}")
        print(f"  Off-by->1:    {n_offgt1}")
        print(f"  Max diff:     {max_dif}")
        print(f"  PSNR:         {psnr:.2f} dB")

        if n_offgt1 == 0:
            print("  -> Sim matches golden (bit-exact or off-by-1 rounding)")
        else:
            print(f"  -> WARNING: {n_offgt1} subpixels differ by >1")

        golden_out = os.path.join(out_dir, 'golden_result.png')
        Image.fromarray(golden_img).save(golden_out)
        diff_vis = np.clip(diff.astype(np.uint8) * 4, 0, 255)
        Image.fromarray(diff_vis).save(os.path.join(out_dir, 'sim_vs_golden_diff.png'))
    else:
        print("  (No golden blend files found — skipping comparison)")

    # ── 8. Side-by-side comparison ────────────────────────────────────────
    gap = 4
    canvas = Image.new('RGB', (SRC_W * 3 + gap * 2, SRC_H), (40, 40, 40))
    canvas.paste(Image.fromarray(img0_rgb), (0, 0))
    canvas.paste(Image.fromarray(sim_img),  (SRC_W + gap, 0))
    canvas.paste(Image.fromarray(img1_rgb), (SRC_W * 2 + gap * 2, 0))
    comp_out = os.path.join(out_dir, f'sim_comparison{sim_suffix}.png')
    canvas.save(comp_out)
    print(f"\n  Side-by-side (src0 | sim | src1): {comp_out}")
    print(f"\n{'='*65}")
    print(f"  Done in {time.time()-t0:.1f}s")
    print(f"  {os.path.basename(sim_out)}: {sim_out}")
    print(f"  {os.path.basename(comp_out)}: {comp_out}")
    print(f"{'='*65}")

    return sim_img  # return for cross-comparison


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description='DMVFN fixed-point Python simulator')
    ap.add_argument('--dataset', choices=['kitti', 'cityscapes'], default='kitti')
    ap.add_argument('--out_dir', default=None,
                    help='Output directory (default: python/sim_out/{dataset})')
    ap.add_argument('--routing', action='store_true',
                    help='Run with routing_ref (skip blocks where ref=0)')
    ap.add_argument('--compare', action='store_true',
                    help='Run both all-9-blocks and routing, then save diff image')
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.join(SCRIPT_DIR, 'sim_out', args.dataset)

    if args.compare:
        print("\n[COMPARE MODE] Running all-9-blocks then routing...\n")
        img_all = simulate(args.dataset, out_dir, use_routing=False)
        img_rte = simulate(args.dataset, out_dir, use_routing=True)
        if img_all is not None and img_rte is not None:
            diff = np.abs(img_all.astype(np.int16) - img_rte.astype(np.int16))
            psnr = compute_psnr(img_all, img_rte)
            diff_vis = np.clip(diff.astype(np.uint8) * 8, 0, 255)
            diff_path = os.path.join(out_dir, 'routing_vs_all9_diff.png')
            Image.fromarray(diff_vis).save(diff_path)
            # side-by-side: all9 | routing | diff×8
            gap = 4
            SRC_H, SRC_W = img_all.shape[:2]
            canvas = Image.new('RGB', (SRC_W * 3 + gap * 2, SRC_H), (40, 40, 40))
            canvas.paste(Image.fromarray(img_all), (0, 0))
            canvas.paste(Image.fromarray(img_rte), (SRC_W + gap, 0))
            canvas.paste(Image.fromarray(diff_vis),(SRC_W * 2 + gap * 2, 0))
            side_path = os.path.join(out_dir, 'routing_vs_all9_side.png')
            canvas.save(side_path)
            print(f"\n{'='*65}")
            print(f"  [COMPARE] all-9-blocks vs routing")
            print(f"  PSNR between the two: {psnr:.2f} dB")
            print(f"  routing_ref active blocks: {sum(1 for x in open(os.path.join(ROOT_DIR, 'cityscapes' if args.dataset=='cityscapes' else 'KITTI', 'phase15', 'routing_ref.txt')).read().split() if x=='1')}/9")
            print(f"  Diff image (×8):  {diff_path}")
            print(f"  Side-by-side:     {side_path}")
            print(f"{'='*65}")
    else:
        simulate(args.dataset, out_dir, use_routing=args.routing)


if __name__ == '__main__':
    main()
