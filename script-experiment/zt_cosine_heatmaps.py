#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import argparse
from pathlib import Path
from typing import Dict, Tuple, Any, List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


# -------------------------
# IO helpers
# -------------------------
def load_bank(pt_path: str) -> torch.Tensor:
    """
    Load z_T bank from a .pt file.
    Supports:
      - raw tensor
      - dict checkpoint with common keys (including your 'zT_bank')
      - fallback: first tensor in dict with shape [16,4,64,64] or [*,4,64,64] and N==16
    """
    obj = torch.load(pt_path, map_location="cpu")  # torch.load loads the pickled object :contentReference[oaicite:1]{index=1}

    if isinstance(obj, torch.Tensor):
        z = obj

    elif isinstance(obj, dict):
        # 1) try common keys (add your key here)
        key_candidates = [
            "zT_bank",  # <-- your checkpoint uses this
            "zT", "z_t", "z_T",
            "latents", "noise", "z",
        ]
        z = None
        for k in key_candidates:
            if k in obj and isinstance(obj[k], torch.Tensor):
                z = obj[k]
                break

        # 2) fallback: search any tensor that looks like [16,4,64,64]
        if z is None:
            for k, v in obj.items():
                if isinstance(v, torch.Tensor) and v.ndim == 4 and tuple(v.shape[1:]) == (4, 64, 64):
                    if int(v.shape[0]) == 16:
                        z = v
                        break

        if z is None:
            raise KeyError(f"[load_bank] no tensor found in dict: keys={list(obj.keys())}")

    else:
        raise TypeError(f"[load_bank] unsupported content type: {type(obj)}")

    # validate shape
    if z.ndim != 4 or tuple(z.shape[1:]) != (4, 64, 64):
        raise ValueError(f"[load_bank] expect [16,4,64,64], got {tuple(z.shape)}")
    if int(z.shape[0]) != 16:
        raise ValueError(f"[load_bank] expect N=16, got N={int(z.shape[0])}")

    return z.float().contiguous()


def make_gaussian_bank(seed: int, shape=(16, 4, 64, 64)) -> torch.Tensor:
    """
    Deterministic Gaussian bank on CPU using a seeded generator.
    """
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    return torch.randn(shape, generator=g, device="cpu", dtype=torch.float32).contiguous()


# -------------------------
# cosine metric (your definition)
# -------------------------
def bank_cosine_mean(a: torch.Tensor, b: torch.Tensor, device: str = "cpu") -> float:
    """
    a,b: [16,4,64,64]
    Compute cosine for each index-aligned pair, then mean over 16:
        mean_i cos(flat(a[i]), flat(b[i]))
    """
    if a.shape != b.shape:
        raise ValueError(f"[bank_cosine_mean] shape mismatch: {a.shape} vs {b.shape}")

    a2 = a.view(a.shape[0], -1).to(device=device)
    b2 = b.view(b.shape[0], -1).to(device=device)

    # cosine_similarity along feature dim (dim=1), output [16], then mean
    # torch.nn.functional.cosine_similarity API: returns cosine similarity computed along dim. :contentReference[oaicite:4]{index=4}
    sim = F.cosine_similarity(a2, b2, dim=1, eps=1e-8).mean()
    return float(sim.detach().cpu().item())


# -------------------------
# plotting
# -------------------------
def plot_heatmap(mat: np.ndarray, row_labels: List[str], col_labels: List[str], title: str, out_png: str):
    """
    Annotated heatmap using matplotlib imshow + text annotations. :contentReference[oaicite:5]{index=5}
    """
    fig, ax = plt.subplots(figsize=(8, 6), dpi=160)
    im = ax.imshow(mat, aspect="auto")

    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right")
    ax.set_yticklabels(row_labels)

    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Cosine similarity (mean over 16)", rotation=90)

    # annotate
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.4f}", ha="center", va="center", fontsize=8)

    ax.set_xlabel("Columns")
    ax.set_ylabel("Rows")
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


# -------------------------
# main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, required=True, help="output directory for csv/xlsx/png")
    ap.add_argument("--device", type=str, default="cpu", help="cpu or cuda (cosine compute only)")
    ap.add_argument("--gauss_seed0", type=int, default=12345, help="Gaussian col-0 seed")
    ap.add_argument("--gauss_cols", type=int, default=4, help="number of Gaussian columns (default 4)")
    ap.add_argument("--gauss_seed_step", type=int, default=1, help="seed increment per column (default 1)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- fixed paths from your message ----
    paths = {
        "TR_w": "/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/latents_experiment/generate_TR_w.pt",
        "TR_w_att": "/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/latents_experiment/generate_TR_w_att_0_88.pt",

        "GS_w": "/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/latents_experiment/generate_GS_w.pt",
        "GS_w_att": "/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/latents_experiment/generate_GS_w_att.pt",

        "PRC_w": "/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/latents_experiment/generate_PRC_w.pt",
        "PRC_w_att": "/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/latents_experiment/generate_PRC_w_att_0_87.pt",

        "T2S_w": "/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/latents_experiment/generate_T2S_w.pt",
        "T2S_w_att": "/home/yancy/work/dm_backdoor_latent_space/experiment-1_19/latents_experiment/generate_T2S_w_att.pt",
    }

    # order (keep stable for paper tables)
    methods = ["TR", "GS", "PRC", "T2S"]
    rows_att = [f"{m}_w_att" for m in methods]
    cols_w = [f"{m}_w" for m in methods]

    # load banks
    banks: Dict[str, torch.Tensor] = {}
    for k, p in paths.items():
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Missing pt: {p}")
        banks[k] = load_bank(p)

    # -------------------------
    # Table A: w_att (rows) vs w (cols)  => 4x4 full
    # -------------------------
    matA = np.zeros((len(rows_att), len(cols_w)), dtype=np.float32)
    for i, r in enumerate(rows_att):
        for j, c in enumerate(cols_w):
            a = banks[r]   # [16,4,64,64]
            b = banks[c]
            matA[i, j] = bank_cosine_mean(a, b, device=args.device)

    dfA = pd.DataFrame(matA, index=rows_att, columns=cols_w)
    dfA.to_csv(out_dir / "cos_watt_vs_w.csv", float_format="%.6f")
    dfA.to_excel(out_dir / "cos_watt_vs_w.xlsx", engine="openpyxl")  # pandas to_excel :contentReference[oaicite:6]{index=6}
    plot_heatmap(matA, rows_att, cols_w, "Cosine similarity: w_att vs w", str(out_dir / "heatmap_watt_vs_w.png"))

    # -------------------------
    # Table B: w_att (rows) vs Gaussian columns (fixed seeds) => 4x4 full
    # Each column j uses Gaussian bank with seed = gauss_seed0 + j*step
    # -------------------------
    gauss_cols = int(args.gauss_cols)
    col_gauss = [f"Gauss_seed{args.gauss_seed0 + j * args.gauss_seed_step}" for j in range(gauss_cols)]

    matB = np.zeros((len(rows_att), gauss_cols), dtype=np.float32)
    for j in range(gauss_cols):
        seed = int(args.gauss_seed0 + j * args.gauss_seed_step)
        gbank = make_gaussian_bank(seed=seed)  # CPU
        for i, r in enumerate(rows_att):
            matB[i, j] = bank_cosine_mean(banks[r], gbank, device=args.device)

    dfB = pd.DataFrame(matB, index=rows_att, columns=col_gauss)
    dfB.to_csv(out_dir / "cos_watt_vs_gauss.csv", float_format="%.6f")
    dfB.to_excel(out_dir / "cos_watt_vs_gauss.xlsx", engine="openpyxl")
    plot_heatmap(matB, rows_att, col_gauss, "Cosine similarity: w_att vs Gaussian (fixed seeds)", str(out_dir / "heatmap_watt_vs_gauss.png"))

    print("[DONE]")
    print(" - cos_watt_vs_w.csv / .xlsx / heatmap_watt_vs_w.png")
    print(" - cos_watt_vs_gauss.csv / .xlsx / heatmap_watt_vs_gauss.png")
    print(f"Output dir: {out_dir}")


if __name__ == "__main__":
    main()
