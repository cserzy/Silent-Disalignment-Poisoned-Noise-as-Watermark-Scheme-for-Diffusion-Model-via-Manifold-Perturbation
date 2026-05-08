#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OMS latent repair utility (SD-like 4D latents only).

Core transform (structured orthogonal):
  x -> x[:, perm] -> block-wise right-multiply by Q_b -> inverse permute

Interpretation of Q:
  - Q is a structured orthogonal transform, not a dense projection matrix.
  - Its structure is:
      Q = P^T diag(Q_1, ..., Q_B) P
  - P is the random permutation matrix induced by q_seed.
  - Each Q_b is a block-wise orthogonal matrix learned by Procrustes fitting.
  - Forward repair uses Q, while inverse repair uses Q^T or the corresponding
    blended inverse when alpha-mixing is enabled.

Modes:
  - fit_apply:   fit Q from (source, target), then apply forward Q to source
  - apply_only:  load existing Q, apply forward Q to source
  - invert_only: load existing Q, apply inverse transform (Q^T block-wise)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


COMMON_LATENT_KEYS: Sequence[str] = (
    "latents",
    "zT_bank",
    "zT",
    "z_t",
    "z_T",
    "noise",
    "z",
)


def fail(msg: str) -> None:
    raise SystemExit(f"[ERR] {msg}")


def log(msg: str, verbose: bool = True) -> None:
    if verbose:
        print(msg, flush=True)


def ensure_parent(path: Path) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        fail(f"Cannot create parent directory for {path}: {e}")


def resolve_dtype(dtype_str: str) -> torch.dtype:
    d = str(dtype_str).lower().strip()
    if d == "fp16":
        return torch.float16
    if d == "bf16":
        return torch.bfloat16
    if d == "fp32":
        return torch.float32
    fail(f"Unsupported --dtype={dtype_str}. Use one of: fp16 / bf16 / fp32")
    raise AssertionError


def resolve_device(device_str: str) -> torch.device:
    dv = str(device_str).lower().strip()
    if dv == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dv == "cpu":
        return torch.device("cpu")
    if dv == "cuda":
        if not torch.cuda.is_available():
            fail("--device cuda was requested, but CUDA is not available.")
        return torch.device("cuda")
    fail(f"Unsupported --device={device_str}. Use one of: auto / cpu / cuda")
    raise AssertionError


def pick_latent_key(payload: Dict[str, Any], latent_key: Optional[str]) -> str:
    if latent_key:
        if latent_key not in payload:
            fail(
                f"--latent_key={latent_key} not found in dict pt. "
                f"Available keys: {list(payload.keys())}"
            )
        if not isinstance(payload[latent_key], torch.Tensor):
            fail(
                f"--latent_key={latent_key} exists but is not a torch.Tensor "
                f"(type={type(payload[latent_key])})."
            )
        return latent_key

    for key in COMMON_LATENT_KEYS:
        if key in payload and isinstance(payload[key], torch.Tensor):
            return key

    tensor_keys = [k for k, v in payload.items() if isinstance(v, torch.Tensor)]
    fail(
        "Cannot determine latent tensor key in dict pt. "
        f"Tried common keys={list(COMMON_LATENT_KEYS)}; "
        f"tensor_keys={tensor_keys}; all_keys={list(payload.keys())}. "
        "Please pass --latent_key explicitly."
    )
    raise AssertionError


def validate_4d_latent(t: torch.Tensor, tag: str) -> None:
    if not isinstance(t, torch.Tensor):
        fail(f"{tag}: latent is not a torch.Tensor (got {type(t)}).")
    if t.ndim != 4:
        fail(
            f"{tag}: expected SD-like 4D latent [N,C,H,W], got shape={tuple(t.shape)}. "
            "Packed/non-4D latents (e.g. FLUX packed latent) are not supported in this script."
        )


def load_pt_with_latent(pt_path: str, latent_key: Optional[str]) -> Dict[str, Any]:
    p = Path(pt_path).expanduser()
    if not p.is_file():
        fail(f"Input pt not found: {p}")

    try:
        obj = torch.load(str(p), map_location="cpu")
    except Exception as e:
        fail(f"Failed to load pt: {p} ({e})")

    if isinstance(obj, torch.Tensor):
        latent = obj
        container_type = "tensor"
        key = None
    elif isinstance(obj, dict):
        key = pick_latent_key(obj, latent_key)
        latent = obj[key]
        container_type = "dict"
    else:
        fail(f"Unsupported pt content type: {type(obj)} (expect Tensor or dict)")

    validate_4d_latent(latent, f"{p}")

    return {
        "path": str(p),
        "obj": obj,
        "container_type": container_type,
        "latent_key": key,
        "latent": latent.detach().cpu(),
    }


def save_pt_with_replaced_latent(pack: Dict[str, Any], new_latent: torch.Tensor, out_pt: str) -> Path:
    out_path = Path(out_pt).expanduser()
    ensure_parent(out_path)

    if pack["container_type"] == "tensor":
        to_save = new_latent
    else:
        to_save = dict(pack["obj"])
        to_save[pack["latent_key"]] = new_latent

    try:
        torch.save(to_save, str(out_path))
    except Exception as e:
        fail(f"Failed to save output pt to {out_path}: {e}")

    if not out_path.is_file():
        fail(f"Failed to save output pt: {out_path}")
    return out_path


def flatten_latent_4d(latent_4d: torch.Tensor) -> torch.Tensor:
    # [N,C,H,W] -> [N,D]
    return latent_4d.contiguous().view(latent_4d.shape[0], -1)


def unflatten_latent_2d(latent_2d: torch.Tensor, shape_4d: Tuple[int, int, int, int]) -> torch.Tensor:
    return latent_2d.view(*shape_4d)


def make_perm(num_dims: int, q_seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build a reproducible random coordinate permutation from q_seed.

    The returned `perm` corresponds to the coordinate shuffling induced by the
    permutation matrix P, and `inv_perm` corresponds to P^{-1}. In the
    structured OMS transform, this step performs coordinate shuffling before
    block-wise orthogonal transforms are applied.
    """
    rng = np.random.RandomState(int(q_seed))
    perm = torch.from_numpy(rng.permutation(num_dims).astype(np.int64))
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(num_dims, dtype=torch.int64)
    return perm, inv_perm


def make_block_slices(num_dims: int, block_size: int) -> List[Tuple[int, int]]:
    if block_size <= 0:
        fail(f"--block_size must be > 0, got {block_size}")
    blocks: List[Tuple[int, int]] = []
    s = 0
    while s < num_dims:
        e = min(s + block_size, num_dims)
        blocks.append((s, e))
        s = e
    return blocks


def fit_blockwise_procrustes(
    x_perm: torch.Tensor,
    y_perm: torch.Tensor,
    block_size: int,
) -> Tuple[List[torch.Tensor], List[float]]:
    """
    Fit the block-wise orthogonal factors of the structured OMS transform.

    Inputs `x_perm` and `y_perm` are both shaped [N, D]. The D-dimensional
    coordinates are partitioned into contiguous blocks of size `block_size`.
    For each block b, solve the orthogonal Procrustes problem

        min ||X_b Q_b - Y_b||_F,   subject to Q_b^T Q_b = I.

    Using the closed-form SVD solution:

        M_b = X_b^T Y_b = U Σ V^T,
        Q_b = U V^T.

    This is the core step in OMS for learning a structured orthogonal
    transform without materializing a dense D x D matrix.
    """
    if x_perm.shape != y_perm.shape:
        fail(f"fit_blockwise_procrustes shape mismatch: X={tuple(x_perm.shape)}, Y={tuple(y_perm.shape)}")

    _, d = x_perm.shape
    blocks = make_block_slices(d, block_size)

    q_blocks: List[torch.Tensor] = []
    ortho_errors: List[float] = []

    for s, e in blocks:
        xb = x_perm[:, s:e]  # [N,db]
        yb = y_perm[:, s:e]  # [N,db]

        # Row-vector right-multiply formulation:
        # min || X_b Q_b - Y_b ||_F, s.t. Q_b^T Q_b = I
        # M = X_b^T Y_b, M = U S V^T, Q_b = U V^T
        m = xb.transpose(0, 1) @ yb

        try:
            u, _, vh = torch.linalg.svd(m, full_matrices=False)
        except Exception as e:
            fail(f"SVD failed at block [{s}:{e}] with shape {tuple(m.shape)}: {e}")

        qb = u @ vh

        eye = torch.eye(qb.shape[0], device=qb.device, dtype=qb.dtype)
        ortho_err = torch.linalg.norm(qb.transpose(0, 1) @ qb - eye, ord="fro").item()

        q_blocks.append(qb.detach().cpu().to(torch.float32))
        ortho_errors.append(float(ortho_err))

    return q_blocks, ortho_errors


def apply_structured_q(
    x_2d: torch.Tensor,
    perm: torch.Tensor,
    inv_perm: torch.Tensor,
    q_blocks: Sequence[torch.Tensor],
    inverse: bool,
) -> torch.Tensor:
    """
    Apply the structured orthogonal transform Q or its inverse.

    Forward mode implements

        x -> x[:, perm] -> block-wise right multiply by Q_b -> x[:, inv_perm].

    In inverse mode, each block matrix Q_b is replaced by Q_b^T. Because every
    Q_b is orthogonal, Q_b^{-1} = Q_b^T, so the inverse transform is obtained
    by transposing each block rather than recomputing a separate factorization.
    """
    if x_2d.ndim != 2:
        fail(f"apply_structured_q expects 2D tensor [N,D], got shape={tuple(x_2d.shape)}")

    n, d = x_2d.shape
    if perm.numel() != d or inv_perm.numel() != d:
        fail(
            f"Permutation size mismatch: D={d}, len(perm)={perm.numel()}, len(inv_perm)={inv_perm.numel()}"
        )

    dev = x_2d.device
    perm_dev = perm.to(device=dev)
    inv_perm_dev = inv_perm.to(device=dev)

    x_perm = x_2d[:, perm_dev]

    out_chunks: List[torch.Tensor] = []
    offset = 0
    for qb_cpu in q_blocks:
        db = int(qb_cpu.shape[0])
        xb = x_perm[:, offset:offset + db]
        qb = qb_cpu.to(device=dev, dtype=x_2d.dtype)
        q_use = qb.transpose(0, 1) if inverse else qb
        out_chunks.append(xb @ q_use)
        offset += db

    if offset != d:
        fail(f"Block dimensions do not sum to D: sum_db={offset}, D={d}")

    x_perm_out = torch.cat(out_chunks, dim=1) if len(out_chunks) > 1 else out_chunks[0]
    x_out = x_perm_out[:, inv_perm_dev]
    if x_out.shape != (n, d):
        fail(f"Unexpected transformed shape: got {tuple(x_out.shape)}, expected {(n, d)}")
    return x_out


def compute_pair_metrics(x: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
    if x.shape != y.shape:
        fail(f"compute_pair_metrics shape mismatch: X={tuple(x.shape)}, Y={tuple(y.shape)}")

    diff = x - y
    mse = (diff * diff).mean().item()
    cos = F.cosine_similarity(x, y, dim=1, eps=1e-8).mean().item()
    y_fro = torch.linalg.norm(y, ord="fro")
    rel_fro = (torch.linalg.norm(diff, ord="fro") / (y_fro + 1e-12)).item()
    return {
        "mse": float(mse),
        "cosine": float(cos),
        "relative_fro": float(rel_fro),
    }


def compute_norm_stats(x: torch.Tensor, xq: torch.Tensor, y: Optional[torch.Tensor]) -> Dict[str, float]:
    out = {
        "mean_norm_x": float(torch.linalg.norm(x, dim=1).mean().item()),
        "mean_norm_xq": float(torch.linalg.norm(xq, dim=1).mean().item()),
    }
    if y is not None:
        out["mean_norm_y"] = float(torch.linalg.norm(y, dim=1).mean().item())
    return out


def to_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, np.integer)):
        return bool(v)
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(v)


def _safe_float(v: Any, default: float) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def _load_json_file(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None
    return None


def resolve_forward_aux_from_q_and_meta(
    q_obj: Dict[str, Any],
    q_pt_path: str,
    q_meta_json: str,
) -> Dict[str, Any]:
    aux = {
        "blend_alpha": 1.0,
        "match_target_std": False,
        "rescale_factor": 1.0,
        "std_before_rescale": None,
        "std_target": None,
        "eps": 1e-12,
        "source": {
            "blend_alpha": "default",
            "match_target_std": "default",
            "rescale_factor": "default",
            "std_before_rescale": "default",
            "std_target": "default",
            "eps": "default",
        },
    }

    if "blend_alpha" in q_obj:
        aux["blend_alpha"] = _safe_float(q_obj.get("blend_alpha"), 1.0)
        aux["source"]["blend_alpha"] = "q_pt"
    if "match_target_std" in q_obj:
        aux["match_target_std"] = to_bool(q_obj.get("match_target_std"))
        aux["source"]["match_target_std"] = "q_pt"

    stdq = q_obj.get("std_match", None)
    if isinstance(stdq, dict):
        if "enabled" in stdq:
            aux["match_target_std"] = to_bool(stdq.get("enabled"))
            aux["source"]["match_target_std"] = "q_pt.std_match"
        if "rescale_factor" in stdq and stdq.get("rescale_factor") is not None:
            aux["rescale_factor"] = _safe_float(stdq.get("rescale_factor"), 1.0)
            aux["source"]["rescale_factor"] = "q_pt.std_match"
        if "std_before_rescale" in stdq:
            aux["std_before_rescale"] = _safe_float(stdq.get("std_before_rescale"), 0.0)
            aux["source"]["std_before_rescale"] = "q_pt.std_match"
        if "std_target" in stdq:
            aux["std_target"] = _safe_float(stdq.get("std_target"), 0.0)
            aux["source"]["std_target"] = "q_pt.std_match"
        if "eps" in stdq:
            aux["eps"] = _safe_float(stdq.get("eps"), 1e-12)
            aux["source"]["eps"] = "q_pt.std_match"

    meta_obj: Optional[Dict[str, Any]] = None
    qpt = Path(q_pt_path).expanduser()
    candidates: List[Path] = []
    if str(q_meta_json).strip():
        candidates.append(Path(q_meta_json).expanduser())
    candidates.append(qpt.with_suffix(".json"))
    candidates.append(qpt.with_name(f"{qpt.stem}_meta.json"))
    candidates.append(qpt.with_name(f"{qpt.stem}.meta.json"))

    seen = set()
    for cand in candidates:
        ck = str(cand.resolve()) if cand.exists() else str(cand)
        if ck in seen:
            continue
        seen.add(ck)
        m = _load_json_file(cand)
        if isinstance(m, dict):
            meta_obj = m
            break

    if isinstance(meta_obj, dict):
        if aux["source"]["blend_alpha"] == "default" and "blend_alpha" in meta_obj:
            aux["blend_alpha"] = _safe_float(meta_obj.get("blend_alpha"), 1.0)
            aux["source"]["blend_alpha"] = "meta_json"
        if aux["source"]["match_target_std"] == "default" and "match_target_std" in meta_obj:
            aux["match_target_std"] = to_bool(meta_obj.get("match_target_std"))
            aux["source"]["match_target_std"] = "meta_json"

        stdm = meta_obj.get("std_match", None)
        if isinstance(stdm, dict):
            if aux["source"]["match_target_std"] == "default" and "enabled" in stdm:
                aux["match_target_std"] = to_bool(stdm.get("enabled"))
                aux["source"]["match_target_std"] = "meta_json.std_match"
            if aux["source"]["rescale_factor"] == "default" and stdm.get("rescale_factor") is not None:
                aux["rescale_factor"] = _safe_float(stdm.get("rescale_factor"), 1.0)
                aux["source"]["rescale_factor"] = "meta_json.std_match"
            if aux["source"]["std_before_rescale"] == "default" and "std_before_rescale" in stdm:
                aux["std_before_rescale"] = _safe_float(stdm.get("std_before_rescale"), 0.0)
                aux["source"]["std_before_rescale"] = "meta_json.std_match"
            if aux["source"]["std_target"] == "default" and "std_target" in stdm:
                aux["std_target"] = _safe_float(stdm.get("std_target"), 0.0)
                aux["source"]["std_target"] = "meta_json.std_match"
            if aux["source"]["eps"] == "default" and "eps" in stdm:
                aux["eps"] = _safe_float(stdm.get("eps"), 1e-12)
                aux["source"]["eps"] = "meta_json.std_match"

    return aux


def invert_blended_structured_q(
    y_2d: torch.Tensor,
    perm: torch.Tensor,
    inv_perm: torch.Tensor,
    q_blocks: Sequence[torch.Tensor],
    alpha: float,
    verbose: bool,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Invert the alpha-blended structured transform used in OMS repair.

    When `blend_alpha < 1`, the forward map is not a pure orthogonal transform:

        y = (1 - alpha) x + alpha xQ
          = x ((1 - alpha) I + alpha Q).

    Therefore the inverse cannot be computed by simply applying Q^T. Instead,
    each block solves the corresponding linear system induced by

        M_b = (1 - alpha) I + alpha Q_b.

    The implementation uses `torch.linalg.solve` on each block to avoid forming
    an explicit inverse matrix.
    """
    if y_2d.ndim != 2:
        fail(f"invert_blended_structured_q expects [N,D], got {tuple(y_2d.shape)}")
    if not (0.0 <= alpha <= 1.0):
        fail(f"alpha must be in [0,1], got {alpha}")

    n, d = y_2d.shape
    if perm.numel() != d or inv_perm.numel() != d:
        fail(f"Permutation size mismatch in blended inverse: D={d}, perm={perm.numel()}, inv_perm={inv_perm.numel()}")

    dev = y_2d.device
    perm_dev = perm.to(device=dev)
    inv_perm_dev = inv_perm.to(device=dev)
    y_perm = y_2d[:, perm_dev]

    x_blocks: List[torch.Tensor] = []
    offset = 0
    conds: List[float] = []
    sv_mins: List[float] = []
    solved_all = True

    for bi, qb_cpu in enumerate(q_blocks):
        db = int(qb_cpu.shape[0])
        yb = y_perm[:, offset:offset + db]
        qb = qb_cpu.to(device=dev, dtype=y_2d.dtype)

        eye = torch.eye(db, device=dev, dtype=y_2d.dtype)
        mb = (1.0 - alpha) * eye + alpha * qb

        try:
            sv = torch.linalg.svdvals(mb)
            sv_max = float(sv.max().item())
            sv_min = float(sv.min().item())
            cond = float(sv_max / (sv_min + 1e-30))
        except Exception as e:
            fail(f"Failed SVD stability check at block={bi}, dim={db}, alpha={alpha}: {e}")

        conds.append(cond)
        sv_mins.append(sv_min)
        if sv_min < 1e-10:
            fail(
                f"Blended inverse matrix near singular at block={bi}, dim={db}, alpha={alpha}. "
                f"sv_min={sv_min:.3e}, cond={cond:.3e}"
            )

        # Row-vector right-multiply inverse:
        # y = x M  =>  y^T = M^T x^T  =>  x^T = solve(M^T, y^T)
        try:
            xb_t = torch.linalg.solve(mb.transpose(0, 1), yb.transpose(0, 1))
            xb = xb_t.transpose(0, 1)
        except Exception as e:
            solved_all = False
            fail(f"Block solve failed at block={bi}, dim={db}, alpha={alpha}: {e}")

        if verbose:
            print(
                f"[INVERT][block {bi:03d}] dim={db} sv_min={sv_min:.3e} cond={cond:.3e}",
                flush=True,
            )

        x_blocks.append(xb)
        offset += db

    if offset != d:
        fail(f"Block dims do not sum to D in blended inverse: sum_db={offset}, D={d}")

    x_perm = torch.cat(x_blocks, dim=1) if len(x_blocks) > 1 else x_blocks[0]
    x_2d = x_perm[:, inv_perm_dev]
    if x_2d.shape != (n, d):
        fail(f"Unexpected blended inverse output shape: {tuple(x_2d.shape)} vs {(n, d)}")

    solve_summary = {
        "num_blocks": int(len(q_blocks)),
        "all_blocks_solved": bool(solved_all),
        "cond_mean": float(np.mean(conds)) if len(conds) > 0 else None,
        "cond_max": float(np.max(conds)) if len(conds) > 0 else None,
        "sv_min_min": float(np.min(sv_mins)) if len(sv_mins) > 0 else None,
    }
    return x_2d, solve_summary


def run_blended_inverse_with_fallback(
    y_2d_cpu: torch.Tensor,
    perm: torch.Tensor,
    inv_perm: torch.Tensor,
    q_blocks: Sequence[torch.Tensor],
    alpha: float,
    device: torch.device,
    verbose: bool,
) -> Tuple[torch.Tensor, Dict[str, Any], str]:
    try:
        y_dev = y_2d_cpu.to(device=device, dtype=torch.float32)
        x_dev, summary = invert_blended_structured_q(
            y_dev,
            perm=perm,
            inv_perm=inv_perm,
            q_blocks=q_blocks,
            alpha=alpha,
            verbose=verbose,
        )
        return x_dev.cpu(), summary, str(device)
    except RuntimeError as e:
        if device.type == "cuda" and "out of memory" in str(e).lower():
            log("[WARN] CUDA OOM during blended inverse. Falling back to CPU.", verbose=verbose)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            x_cpu, summary = invert_blended_structured_q(
                y_2d_cpu.to(torch.float32),
                perm=perm,
                inv_perm=inv_perm,
                q_blocks=q_blocks,
                alpha=alpha,
                verbose=verbose,
            )
            return x_cpu.cpu(), summary, "cpu"
        raise


def compute_latent_stats_4d(latent_4d: torch.Tensor) -> Dict[str, Any]:
    validate_4d_latent(latent_4d, "stats_only input")
    z = latent_4d.detach().cpu().to(torch.float32)
    n, c, h, w = z.shape
    z_flat = z.view(n, -1)

    per_sample_norms = torch.linalg.norm(z_flat, dim=1)
    channel_stats: List[Dict[str, float]] = []
    for ch in range(c):
        ch_t = z[:, ch, :, :]
        channel_stats.append(
            {
                "channel": int(ch),
                "mean": float(ch_t.mean().item()),
                "std": float(ch_t.std(unbiased=False).item()),
                "min": float(ch_t.min().item()),
                "max": float(ch_t.max().item()),
            }
        )

    return {
        "latent_shape": [int(n), int(c), int(h), int(w)],
        "has_nan": bool(torch.isnan(z).any().item()),
        "has_inf": bool(torch.isinf(z).any().item()),
        "global_stats": {
            "mean": float(z.mean().item()),
            "std": float(z.std(unbiased=False).item()),
            "min": float(z.min().item()),
            "max": float(z.max().item()),
            "abs_mean": float(z.abs().mean().item()),
        },
        "per_channel_stats": channel_stats,
        "per_sample_norms": [float(v) for v in per_sample_norms.tolist()],
        "per_sample_norm_stats": {
            "mean": float(per_sample_norms.mean().item()),
            "std": float(per_sample_norms.std(unbiased=False).item()),
            "min": float(per_sample_norms.min().item()),
            "max": float(per_sample_norms.max().item()),
        },
        "flattened_stats": {
            "mean": float(z_flat.mean().item()),
            "std": float(z_flat.std(unbiased=False).item()),
        },
    }


def print_latent_stats(stats: Dict[str, Any]) -> None:
    gs = stats["global_stats"]
    fs = stats["flattened_stats"]
    ns = stats["per_sample_norm_stats"]
    print("[STATS] latent_shape:", tuple(stats["latent_shape"]), flush=True)
    print(
        "[STATS] global: "
        f"mean={gs['mean']:.8f}, std={gs['std']:.8f}, min={gs['min']:.8f}, "
        f"max={gs['max']:.8f}, abs_mean={gs['abs_mean']:.8f}",
        flush=True,
    )
    print(
        "[STATS] flattened: "
        f"mean={fs['mean']:.8f}, std={fs['std']:.8f}",
        flush=True,
    )
    print(
        "[STATS] per-sample norm summary: "
        f"mean={ns['mean']:.8f}, std={ns['std']:.8f}, min={ns['min']:.8f}, max={ns['max']:.8f}",
        flush=True,
    )
    print(
        f"[STATS] has_nan={stats['has_nan']}, has_inf={stats['has_inf']}",
        flush=True,
    )
    for ch in stats["per_channel_stats"]:
        print(
            f"[STATS] ch{ch['channel']}: "
            f"mean={ch['mean']:.8f}, std={ch['std']:.8f}, min={ch['min']:.8f}, max={ch['max']:.8f}",
            flush=True,
        )


def run_apply_with_fallback(
    x_2d_cpu: torch.Tensor,
    perm: torch.Tensor,
    inv_perm: torch.Tensor,
    q_blocks: Sequence[torch.Tensor],
    inverse: bool,
    device: torch.device,
    verbose: bool,
) -> Tuple[torch.Tensor, str]:
    try:
        x_dev = x_2d_cpu.to(device=device, dtype=torch.float32)
        out_dev = apply_structured_q(
            x_dev,
            perm=perm,
            inv_perm=inv_perm,
            q_blocks=q_blocks,
            inverse=inverse,
        )
        return out_dev.cpu(), str(device)
    except RuntimeError as e:
        if device.type == "cuda" and "out of memory" in str(e).lower():
            log("[WARN] CUDA OOM during transform. Falling back to CPU.", verbose=verbose)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            out_cpu = apply_structured_q(
                x_2d_cpu.to(dtype=torch.float32),
                perm=perm,
                inv_perm=inv_perm,
                q_blocks=q_blocks,
                inverse=inverse,
            )
            return out_cpu.cpu(), "cpu"
        raise


def run_fit_with_fallback(
    x_fit_cpu: torch.Tensor,
    y_fit_cpu: torch.Tensor,
    perm: torch.Tensor,
    block_size: int,
    device: torch.device,
    verbose: bool,
) -> Tuple[List[torch.Tensor], List[float], str]:
    try:
        x_dev = x_fit_cpu.to(device=device, dtype=torch.float32)
        y_dev = y_fit_cpu.to(device=device, dtype=torch.float32)

        perm_dev = perm.to(device=device)
        x_perm = x_dev[:, perm_dev]
        y_perm = y_dev[:, perm_dev]
        q_blocks, ortho_errors = fit_blockwise_procrustes(
            x_perm=x_perm,
            y_perm=y_perm,
            block_size=block_size,
        )
        return q_blocks, ortho_errors, str(device)
    except RuntimeError as e:
        if device.type == "cuda" and "out of memory" in str(e).lower():
            log("[WARN] CUDA OOM during fitting. Falling back to CPU.", verbose=verbose)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            x_perm_cpu = x_fit_cpu[:, perm]
            y_perm_cpu = y_fit_cpu[:, perm]
            q_blocks, ortho_errors = fit_blockwise_procrustes(
                x_perm=x_perm_cpu,
                y_perm=y_perm_cpu,
                block_size=block_size,
            )
            return q_blocks, ortho_errors, "cpu"
        raise


def validate_q_pack(q_pack: Dict[str, Any], block_mode: str) -> None:
    required = ["perm", "inv_perm", "q_blocks", "block_size", "block_mode", "q_seed"]
    missing = [k for k in required if k not in q_pack]
    if missing:
        fail(f"Q file missing required keys: {missing}")

    if str(q_pack["block_mode"]) != str(block_mode):
        fail(
            f"Q block_mode mismatch: q_pt has {q_pack['block_mode']}, "
            f"but --block_mode={block_mode}."
        )

    if not isinstance(q_pack["perm"], torch.Tensor) or q_pack["perm"].ndim != 1:
        fail("Q file field 'perm' must be a 1D torch.Tensor")
    if not isinstance(q_pack["inv_perm"], torch.Tensor) or q_pack["inv_perm"].ndim != 1:
        fail("Q file field 'inv_perm' must be a 1D torch.Tensor")

    q_blocks = q_pack["q_blocks"]
    if isinstance(q_blocks, torch.Tensor):
        # allows stacked [B,d,d] only when all blocks same dim
        if q_blocks.ndim != 3:
            fail("If q_blocks is a tensor, expected shape [num_blocks,d,d].")
    elif isinstance(q_blocks, (list, tuple)):
        if len(q_blocks) == 0:
            fail("Q file has empty q_blocks.")
        for i, qb in enumerate(q_blocks):
            if not isinstance(qb, torch.Tensor):
                fail(f"q_blocks[{i}] is not a tensor (got {type(qb)}).")
            if qb.ndim != 2 or qb.shape[0] != qb.shape[1]:
                fail(f"q_blocks[{i}] must be square 2D tensor, got shape={tuple(qb.shape)}")
    else:
        fail(f"Unsupported q_blocks type: {type(q_blocks)}")


def normalize_q_blocks(q_blocks_any: Any) -> List[torch.Tensor]:
    if isinstance(q_blocks_any, torch.Tensor):
        # [B,d,d] -> list
        return [q_blocks_any[i].detach().cpu().to(torch.float32) for i in range(q_blocks_any.shape[0])]
    return [qb.detach().cpu().to(torch.float32) for qb in q_blocks_any]


def load_q_pt(q_pt: str, block_mode: str) -> Dict[str, Any]:
    p = Path(q_pt).expanduser()
    if not p.is_file():
        fail(f"Q pt not found: {p}")

    try:
        obj = torch.load(str(p), map_location="cpu")
    except Exception as e:
        fail(f"Failed to load q_pt: {p} ({e})")

    if not isinstance(obj, dict):
        fail(f"q_pt must be a dict, got {type(obj)}")

    validate_q_pack(obj, block_mode=block_mode)

    q_blocks = normalize_q_blocks(obj["q_blocks"])
    perm = obj["perm"].detach().cpu().to(torch.int64)
    inv_perm = obj["inv_perm"].detach().cpu().to(torch.int64)

    d = int(perm.numel())
    if int(inv_perm.numel()) != d:
        fail(f"perm/inv_perm length mismatch: {d} vs {inv_perm.numel()}")

    sum_db = int(sum(int(qb.shape[0]) for qb in q_blocks))
    if sum_db != d:
        fail(f"Q block dims sum mismatch: sum_db={sum_db}, D={d}")

    inv_from_perm = torch.empty_like(perm)
    inv_from_perm[perm] = torch.arange(d, dtype=torch.int64)
    if not torch.equal(inv_from_perm, inv_perm):
        fail("Q file has inconsistent perm and inv_perm (inv_perm is not the inverse of perm).")

    obj["q_blocks"] = q_blocks
    obj["perm"] = perm
    obj["inv_perm"] = inv_perm
    return obj


def build_meta_base(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "mode": args.mode,
        "source_pt": args.in_pt,
        "target_pt": args.target_pt,
        "q_pt": args.q_pt,
        "q_meta_json": args.q_meta_json,
        "q_seed": int(args.q_seed),
        "block_mode": args.block_mode,
        "block_size": int(args.block_size),
        "blend_alpha": float(args.blend_alpha),
        "match_target_std": bool(args.match_target_std),
        "device": args.device,
        "dtype": args.dtype,
        "strict_shape": bool(args.strict_shape),
        "latent_key": args.latent_key,
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="OMS (Orthogonal-based Manifold Shuffling) latent repair for SD-like 4D latents"
    )
    ap.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["fit_apply", "apply_only", "invert_only", "stats_only"],
    )
    ap.add_argument("--in_pt", type=str, required=True, help="Input latent pt (Tensor or dict)")
    ap.add_argument("--target_pt", type=str, default="", help="Target GS-like pt (required in fit_apply)")
    ap.add_argument("--q_pt", type=str, default="", help="Existing Q pt (required in apply_only/invert_only)")
    ap.add_argument(
        "--q_meta_json",
        type=str,
        default="",
        help="Optional meta json for q_pt (invert_only fallback for blend/std-match params).",
    )

    ap.add_argument("--out_pt", type=str, default="", help="Output repaired/inverted pt")
    ap.add_argument("--out_q_pt", type=str, default="", help="Output Q pt (required in fit_apply)")
    ap.add_argument("--out_meta_json", type=str, default="", help="Output meta json (optional; default next to out_pt)")

    ap.add_argument("--q_seed", type=int, default=12345)
    ap.add_argument("--block_mode", type=str, default="flat_chunk", choices=["flat_chunk"])
    ap.add_argument("--block_size", type=int, default=256)
    ap.add_argument(
        "--blend_alpha",
        type=float,
        default=1.0,
        help="Forward blend strength: x_mix=(1-alpha)*x + alpha*x_q (fit_apply/apply_only only)",
    )
    ap.add_argument(
        "--match_target_std",
        type=int,
        default=0,
        choices=[0, 1],
        help="Only for fit_apply: rescale x_mix by std(target)/std(x_mix).",
    )

    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--dtype", type=str, default="fp32", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--latent_key", type=str, default="", help="Optional: force dict latent key")
    ap.add_argument("--strict_shape", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--verbose", action="store_true")

    args = ap.parse_args()

    if args.blend_alpha < 0.0 or args.blend_alpha > 1.0:
        fail(f"--blend_alpha must be in [0, 1], got {args.blend_alpha}")

    if args.mode == "stats_only":
        # stats_only only needs --in_pt (already required)
        return args

    if not str(args.out_pt).strip():
        fail(f"mode={args.mode} requires --out_pt.")

    if args.mode == "fit_apply":
        if not str(args.target_pt).strip():
            fail("mode=fit_apply requires --target_pt.")
        if not str(args.out_q_pt).strip():
            fail("mode=fit_apply requires --out_q_pt.")
    elif args.mode in {"apply_only", "invert_only"}:
        if not str(args.q_pt).strip():
            fail(f"mode={args.mode} requires --q_pt.")

    return args


def main() -> None:
    args = parse_args()
    t0_all = time.time()

    target_dtype = resolve_dtype(args.dtype)
    compute_device = resolve_device(args.device)
    latent_key = args.latent_key if str(args.latent_key).strip() else None

    if args.mode == "stats_only":
        out_meta_path = Path(args.out_meta_json).expanduser() if str(args.out_meta_json).strip() else None
    else:
        out_pt_path = Path(args.out_pt).expanduser()
        out_meta_path = (
            Path(args.out_meta_json).expanduser()
            if str(args.out_meta_json).strip()
            else out_pt_path.parent / f"{out_pt_path.stem}.oms_meta.json"
        )

    log(f"[INFO] mode={args.mode}", verbose=True)
    log(f"[INFO] compute_device={compute_device}", verbose=True)

    src_pack = load_pt_with_latent(args.in_pt, latent_key=latent_key)
    src_lat = src_pack["latent"].to(torch.float32)
    src_shape = tuple(int(x) for x in src_lat.shape)
    n_src, c, h, w = src_shape
    d = int(c * h * w)

    meta: Dict[str, Any] = build_meta_base(args)
    meta["latent_shape"] = [n_src, c, h, w]
    meta["num_samples"] = int(n_src)
    meta["num_dims"] = int(d)

    if args.mode == "stats_only":
        stats = compute_latent_stats_4d(src_lat)
        print_latent_stats(stats)
        meta["stats"] = stats
        meta["time_cost_sec"] = {"total": float(time.time() - t0_all)}

        if out_meta_path is not None:
            ensure_parent(out_meta_path)
            try:
                with out_meta_path.open("w", encoding="utf-8") as f:
                    json.dump(meta, f, ensure_ascii=False, indent=2)
            except Exception as e:
                fail(f"Failed to save out_meta_json to {out_meta_path}: {e}")
            print(f"[OK] saved meta json: {out_meta_path}", flush=True)
        else:
            print("[OK] stats_only finished (no out_meta_json requested).", flush=True)
        return

    if args.mode == "fit_apply":
        tgt_pack = load_pt_with_latent(args.target_pt, latent_key=latent_key)
        tgt_lat = tgt_pack["latent"].to(torch.float32)
        tgt_shape = tuple(int(x) for x in tgt_lat.shape)

        if args.strict_shape:
            if src_shape != tgt_shape:
                fail(
                    f"strict shape check failed: source shape={src_shape}, target shape={tgt_shape}."
                )
            n_fit = n_src
        else:
            if src_shape[1:] != tgt_shape[1:]:
                fail(
                    "--strict_shape=False still requires matching C/H/W. "
                    f"source={src_shape}, target={tgt_shape}"
                )
            n_fit = min(src_shape[0], tgt_shape[0])
            if n_fit <= 0:
                fail(f"No samples to fit: source N={src_shape[0]}, target N={tgt_shape[0]}")
            log(
                f"[WARN] strict_shape disabled, using first {n_fit} samples for fitting "
                f"(source N={src_shape[0]}, target N={tgt_shape[0]}).",
                verbose=True,
            )

        x_fit = flatten_latent_4d(src_lat[:n_fit])
        y_fit = flatten_latent_4d(tgt_lat[:n_fit])

        perm, inv_perm = make_perm(d, args.q_seed)

        fit_start = time.time()
        q_blocks, ortho_errors, used_fit_device = run_fit_with_fallback(
            x_fit_cpu=x_fit,
            y_fit_cpu=y_fit,
            perm=perm,
            block_size=args.block_size,
            device=compute_device,
            verbose=args.verbose,
        )
        fit_sec = time.time() - fit_start

        xq_fit, _ = run_apply_with_fallback(
            x_2d_cpu=x_fit,
            perm=perm,
            inv_perm=inv_perm,
            q_blocks=q_blocks,
            inverse=False,
            device=compute_device,
            verbose=args.verbose,
        )

        before = compute_pair_metrics(x_fit, y_fit)
        after = compute_pair_metrics(xq_fit, y_fit)
        norm_stats = compute_norm_stats(x_fit, xq_fit, y_fit)

        ortho_mean = float(np.mean(ortho_errors)) if len(ortho_errors) > 0 else 0.0
        ortho_max = float(np.max(ortho_errors)) if len(ortho_errors) > 0 else 0.0

        print("[FIT] metrics (flattened [N,D]):", flush=True)
        print(f"  mse_before={before['mse']:.8f}", flush=True)
        print(f"  mse_after ={after['mse']:.8f}", flush=True)
        print(f"  cos_before={before['cosine']:.8f}", flush=True)
        print(f"  cos_after ={after['cosine']:.8f}", flush=True)
        print(f"  relfro_before={before['relative_fro']:.8f}", flush=True)
        print(f"  relfro_after ={after['relative_fro']:.8f}", flush=True)
        print(
            "  norm_mean: "
            f"||x||={norm_stats['mean_norm_x']:.6f}, "
            f"||xQ||={norm_stats['mean_norm_xq']:.6f}, "
            f"||y||={norm_stats['mean_norm_y']:.6f}",
            flush=True,
        )
        print(
            f"  orthogonality_error: mean={ortho_mean:.8e}, max={ortho_max:.8e}",
            flush=True,
        )

        x_all = flatten_latent_4d(src_lat)
        xq_all, used_apply_device_all = run_apply_with_fallback(
            x_2d_cpu=x_all,
            perm=perm,
            inv_perm=inv_perm,
            q_blocks=q_blocks,
            inverse=False,
            device=compute_device,
            verbose=args.verbose,
        )

        blend_alpha = float(args.blend_alpha)
        x_mix_all = (1.0 - blend_alpha) * x_all + blend_alpha * xq_all
        blend_stats: Optional[Dict[str, Any]] = None
        if blend_alpha != 1.0:
            norm_xq = compute_norm_stats(x_all, xq_all, y=None)
            norm_xmix = compute_norm_stats(x_all, x_mix_all, y=None)
            blend_stats = {"xq_norm_stats": norm_xq, "xmix_norm_stats": norm_xmix}
            print(
                "[BLEND] "
                f"alpha={blend_alpha:.6f} | mean||xQ||={norm_xq['mean_norm_xq']:.6f}, "
                f"mean||x_mix||={norm_xmix['mean_norm_xq']:.6f}",
                flush=True,
            )

        std_match_info: Optional[Dict[str, float]] = None
        if bool(args.match_target_std):
            std_before = float(x_mix_all.std(unbiased=False).item())
            y_all = flatten_latent_4d(tgt_lat)
            std_target = float(y_all.std(unbiased=False).item())
            rescale_factor = float(std_target / (std_before + 1e-12))
            x_final_all = x_mix_all * rescale_factor
            std_after = float(x_final_all.std(unbiased=False).item())
            std_match_info = {
                "std_before_rescale": std_before,
                "std_target": std_target,
                "rescale_factor": rescale_factor,
                "std_after_rescale": std_after,
            }
            print(
                "[STD-MATCH] "
                f"before={std_before:.8f}, target={std_target:.8f}, "
                f"factor={rescale_factor:.8f}, after={std_after:.8f}",
                flush=True,
            )
        else:
            x_final_all = x_mix_all
            std_match_info = {
                "enabled": False,
                "std_before_rescale": float(x_mix_all.std(unbiased=False).item()),
                "std_target": None,
                "rescale_factor": 1.0,
                "std_after_rescale": float(x_mix_all.std(unbiased=False).item()),
                "eps": 1e-12,
            }

        if std_match_info is not None and "enabled" not in std_match_info:
            std_match_info["enabled"] = True
            std_match_info["eps"] = 1e-12

        repaired = unflatten_latent_2d(x_final_all, src_shape).to(dtype=target_dtype)
        out_pt_saved = save_pt_with_replaced_latent(src_pack, repaired, args.out_pt)

        q_pack = {
            "format": "oms_structured_q_v1",
            "description": (
                "Structured Q storage: x -> x[:,perm] -> blockwise right-multiply Q_b -> x[:,inv_perm]. "
                "q_blocks + perm store the structured orthogonal component Q. "
                "When blend_alpha=1 and match_target_std=False, forward application is equivalent to "
                "applying this orthogonal Q without materializing dense D x D. "
                "When blend_alpha or std matching is enabled, the final repaired latent additionally "
                "includes alpha-mixing and/or global scale matching on top of Q."
            ),
            "mode": "fit_apply",
            "q_seed": int(args.q_seed),
            "block_mode": args.block_mode,
            "block_size": int(args.block_size),
            "blend_alpha": float(args.blend_alpha),
            "match_target_std": bool(args.match_target_std),
            "std_match": std_match_info,
            "num_blocks": int(len(q_blocks)),
            "perm": perm.cpu().to(torch.int64),
            "inv_perm": inv_perm.cpu().to(torch.int64),
            "q_blocks": [qb.cpu().to(torch.float32) for qb in q_blocks],
            "latent_shape": [int(v) for v in src_shape],
            "num_samples": int(n_fit),
            "num_dims": int(d),
            "fit_stats": {
                "mse_before": before["mse"],
                "mse_after": after["mse"],
                "cosine_before": before["cosine"],
                "cosine_after": after["cosine"],
                "relative_fro_before": before["relative_fro"],
                "relative_fro_after": after["relative_fro"],
                "mean_norm_x": norm_stats["mean_norm_x"],
                "mean_norm_xq": norm_stats["mean_norm_xq"],
                "mean_norm_y": norm_stats["mean_norm_y"],
                "orthogonality_error_mean": ortho_mean,
                "orthogonality_error_max": ortho_max,
                "orthogonality_errors_per_block": ortho_errors,
            },
            "fit_device": used_fit_device,
            "apply_device": used_apply_device_all,
            "fit_time_sec": float(fit_sec),
            "created_unix": float(time.time()),
        }

        out_q_path = Path(args.out_q_pt).expanduser()
        ensure_parent(out_q_path)
        try:
            torch.save(q_pack, str(out_q_path))
        except Exception as e:
            fail(f"Failed to save out_q_pt to {out_q_path}: {e}")
        if not out_q_path.is_file():
            fail(f"Failed to save out_q_pt: {out_q_path}")

        meta.update(
            {
                "target_pt": args.target_pt,
                "q_pt": str(out_q_path),
                "fit_num_samples": int(n_fit),
                "fit_device": used_fit_device,
                "apply_device": used_apply_device_all,
                "pre_fit_metrics": {
                    "mse": before["mse"],
                    "cosine": before["cosine"],
                    "relative_fro": before["relative_fro"],
                },
                "post_fit_metrics": {
                    "mse": after["mse"],
                    "cosine": after["cosine"],
                    "relative_fro": after["relative_fro"],
                },
                "norm_stats": norm_stats,
                "blend_stats": blend_stats,
                "std_match": std_match_info,
                "orthogonality": {
                    "error_mean": ortho_mean,
                    "error_max": ortho_max,
                    "errors_per_block": ortho_errors,
                },
                "time_cost_sec": {
                    "fit": float(fit_sec),
                    "total": float(time.time() - t0_all),
                },
                "out_pt": str(out_pt_saved),
                "out_q_pt": str(out_q_path),
            }
        )

        print(f"[OK] saved repaired pt: {out_pt_saved}", flush=True)
        print(f"[OK] saved Q pt:        {out_q_path}", flush=True)

    else:
        q_obj = load_q_pt(args.q_pt, block_mode=args.block_mode)
        perm = q_obj["perm"]
        inv_perm = q_obj["inv_perm"]
        q_blocks = q_obj["q_blocks"]
        meta["q_seed"] = int(q_obj.get("q_seed", meta["q_seed"]))
        meta["block_mode"] = str(q_obj.get("block_mode", meta["block_mode"]))
        meta["block_size"] = int(q_obj.get("block_size", meta["block_size"]))
        if bool(args.match_target_std):
            print(f"[WARN] --match_target_std is only effective in fit_apply; ignored in mode={args.mode}.", flush=True)

        if int(perm.numel()) != d:
            fail(
                f"Q dimension mismatch with source latent: q_D={perm.numel()}, source_D={d}. "
                f"source shape={src_shape}"
            )

        q_latent_shape = q_obj.get("latent_shape", None)
        if q_latent_shape is not None and len(q_latent_shape) == 4:
            q_shape = tuple(int(v) for v in q_latent_shape)
            if args.strict_shape and src_shape[1:] != q_shape[1:]:
                fail(
                    "strict shape check failed between source and q_pt latent shape: "
                    f"source={src_shape}, q_pt={q_shape}"
                )

        x_all = flatten_latent_4d(src_lat)
        inverse_flag = args.mode == "invert_only"
        blend_stats: Optional[Dict[str, Any]] = None
        inverse_info: Dict[str, Any] = {}

        if inverse_flag:
            aux = resolve_forward_aux_from_q_and_meta(
                q_obj=q_obj,
                q_pt_path=args.q_pt,
                q_meta_json=args.q_meta_json,
            )
            alpha_eff = float(aux["blend_alpha"])
            match_eff = bool(aux["match_target_std"])
            rescale_factor = float(aux["rescale_factor"])

            if not (0.0 <= alpha_eff <= 1.0):
                fail(f"Invalid effective blend_alpha for invert_only: {alpha_eff}")

            if match_eff and abs(rescale_factor) < 1e-12:
                fail(
                    f"Invalid rescale_factor for invert_only (too close to zero): {rescale_factor}. "
                    "Cannot undo std scaling safely."
                )

            if match_eff:
                x_unscaled = x_all / rescale_factor
            else:
                x_unscaled = x_all

            if alpha_eff == 1.0:
                x_final, used_apply_device = run_apply_with_fallback(
                    x_2d_cpu=x_unscaled,
                    perm=perm,
                    inv_perm=inv_perm,
                    q_blocks=q_blocks,
                    inverse=True,
                    device=compute_device,
                    verbose=args.verbose,
                )
                solve_summary = None
                inverse_mode = "pure_Q_inverse" if not match_eff else "blended_plus_rescale_inverse"
            else:
                x_final, solve_summary, used_apply_device = run_blended_inverse_with_fallback(
                    y_2d_cpu=x_unscaled,
                    perm=perm,
                    inv_perm=inv_perm,
                    q_blocks=q_blocks,
                    alpha=alpha_eff,
                    device=compute_device,
                    verbose=args.verbose,
                )
                inverse_mode = "blended_inverse" if not match_eff else "blended_plus_rescale_inverse"
                print(
                    "[INVERT] blended solve summary: "
                    f"num_blocks={solve_summary['num_blocks']}, "
                    f"all_blocks_solved={solve_summary['all_blocks_solved']}, "
                    f"cond_mean={solve_summary['cond_mean']:.3e}, "
                    f"cond_max={solve_summary['cond_max']:.3e}, "
                    f"sv_min_min={solve_summary['sv_min_min']:.3e}",
                    flush=True,
                )

            print(
                "[INVERT] "
                f"mode={inverse_mode}, "
                f"blend_alpha={alpha_eff:.6f}, "
                f"match_target_std={match_eff}, "
                f"rescale_factor={rescale_factor:.8f}",
                flush=True,
            )

            rec_global = {
                "mean": float(x_final.mean().item()),
                "std": float(x_final.std(unbiased=False).item()),
                "min": float(x_final.min().item()),
                "max": float(x_final.max().item()),
            }
            rec_norm = torch.linalg.norm(x_final, dim=1)
            rec_norm_stats = {
                "mean": float(rec_norm.mean().item()),
                "std": float(rec_norm.std(unbiased=False).item()),
            }
            print(
                "[INVERT] recovered stats: "
                f"mean={rec_global['mean']:.8f}, std={rec_global['std']:.8f}, "
                f"min={rec_global['min']:.8f}, max={rec_global['max']:.8f}, "
                f"norm_mean={rec_norm_stats['mean']:.8f}, norm_std={rec_norm_stats['std']:.8f}",
                flush=True,
            )

            inverse_info = {
                "inverse_mode": inverse_mode,
                "blend_alpha": alpha_eff,
                "match_target_std": match_eff,
                "rescale_factor": rescale_factor,
                "param_source": aux.get("source", {}),
                "solve_summary": solve_summary,
                "recovered_global_stats_flattened": rec_global,
                "recovered_norm_stats_flattened": rec_norm_stats,
            }
        else:
            x_q, used_apply_device = run_apply_with_fallback(
                x_2d_cpu=x_all,
                perm=perm,
                inv_perm=inv_perm,
                q_blocks=q_blocks,
                inverse=False,
                device=compute_device,
                verbose=args.verbose,
            )
            blend_alpha = float(args.blend_alpha)
            x_mix = (1.0 - blend_alpha) * x_all + blend_alpha * x_q
            x_final = x_mix
            if blend_alpha != 1.0:
                norm_xq = compute_norm_stats(x_all, x_q, y=None)
                norm_xmix = compute_norm_stats(x_all, x_mix, y=None)
                blend_stats = {"xq_norm_stats": norm_xq, "xmix_norm_stats": norm_xmix}
                print(
                    "[BLEND] "
                    f"alpha={blend_alpha:.6f} | mean||xQ||={norm_xq['mean_norm_xq']:.6f}, "
                    f"mean||x_mix||={norm_xmix['mean_norm_xq']:.6f}",
                    flush=True,
                )

        out_norm = compute_norm_stats(x_all, x_final, y=None)

        repaired = unflatten_latent_2d(x_final, src_shape).to(dtype=target_dtype)
        out_pt_saved = save_pt_with_replaced_latent(src_pack, repaired, args.out_pt)

        meta.update(
            {
                "q_pt": args.q_pt,
                "apply_device": used_apply_device,
                "inverse": bool(inverse_flag),
                "norm_stats": out_norm,
                "blend_stats": blend_stats,
                "inverse_info": inverse_info,
                "time_cost_sec": {
                    "total": float(time.time() - t0_all),
                },
                "out_pt": str(out_pt_saved),
            }
        )

        print(
            "[APPLY] norm sanity: "
            f"mean||x||={out_norm['mean_norm_x']:.6f}, "
            f"mean||x_out||={out_norm['mean_norm_xq']:.6f}",
            flush=True,
        )
        print(f"[OK] saved output pt: {out_pt_saved}", flush=True)

    ensure_parent(out_meta_path)
    try:
        with out_meta_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
    except Exception as e:
        fail(f"Failed to save out_meta_json to {out_meta_path}: {e}")

    print(f"[OK] saved meta json: {out_meta_path}", flush=True)


if __name__ == "__main__":
    main()
