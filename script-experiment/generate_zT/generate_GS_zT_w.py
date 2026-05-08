#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, argparse, math, hashlib
from typing import Optional
import numpy as np
import torch
from Crypto.Cipher import ChaCha20
from scipy.stats import norm  # ppf

# åºå® SD2.1 latent å½¢ç¶
C, H, W = 4, 64, 64

def bits_bin_to_bytes(bits: str) -> bytes:
    bits = bits.strip()
    if not set(bits) <= {"0","1"}:
        raise ValueError("äºè¿å¶å¯é¥/nonce åªè½åå« 0/1")
    if len(bits) % 8 != 0:
        raise ValueError("äºè¿å¶ä¸²é¿åº¦å¿é¡»æ¯ 8 çåæ°")
    out = bytearray()
    for i in range(0, len(bits), 8):
        out.append(int(bits[i:i+8], 2))
    return bytes(out)

def parse_key_32bytes(args) -> bytes:
    # äºæ¥ï¼--key_ones / --key_hex / --key_bin
    cnt = int(args.key_ones) + (args.key_hex is not None) + (args.key_bin is not None)
    if cnt != 1:
        raise ValueError("å¯é¥è¾å¥éä¸ä»éä¸ç§ï¼--key_ones æ --key_hex æ --key_bin")
    if args.key_ones:
        return b"\xff" * 32  # 32 å­èå¨ 1ï¼å³ 256 ä¸ªæ¯ç¹ 1ï¼
    if args.key_hex is not None:
        hx = args.key_hex.strip().lower()
        if len(hx) != 64 or any(ch not in "0123456789abcdef" for ch in hx):
            raise ValueError("key_hex å¿é¡»æ¯ 64 ä¸ªåå­è¿å¶å­ç¬¦ï¼=32å­èï¼")
        return bytes.fromhex(hx)
    if args.key_bin is not None:
        bs = args.key_bin.strip()
        if len(bs) != 256:
            raise ValueError("key_bin å¿é¡»æ¯ 256 ä½ 01 ä¸²ï¼=32å­èï¼")
        return bits_bin_to_bytes(bs)
    raise AssertionError

def parse_nonce_12bytes(args) -> bytes:
    # ä»»éå¶ä¸ï¼è¥åä¸ä¼ åç»ä¸ä¸ªåºå® nonceï¼ä»ç¨äºå¤ç°å®éªï¼
    if args.nonce_zero:
        return b"\x00" * 12
    if args.nonce_hex is not None:
        hx = args.nonce_hex.strip().lower()
        if len(hx) != 24 or any(ch not in "0123456789abcdef" for ch in hx):
            raise ValueError("nonce_hex å¿é¡»æ¯ 24 ä¸ªåå­è¿å¶å­ç¬¦ï¼=12å­èï¼")
        return bytes.fromhex(hx)
    if args.nonce_bin is not None:
        bs = args.nonce_bin.strip()
        if len(bs) != 96:
            raise ValueError("nonce_bin å¿é¡»æ¯ 96 ä½ 01 ä¸²ï¼=12å­èï¼")
        return bits_bin_to_bytes(bs)
    # åºå® nonceï¼ä»ä¸ºå®éªå¤ç°æ¹ä¾¿ï¼çäº§ä¸å»ºè®®åºå®/å¤ç¨ï¼
    return b"GS-fixed-nc!"  # 12 bytes

def make_base_bits(k_bits: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, 2, size=(k_bits,), dtype=np.int8)

def diffuse_bits_to_chw(base_bits: np.ndarray, fc: int, fhw: int) -> np.ndarray:
    assert base_bits.ndim == 1
    c0 = C // fc; h0 = H // fhw; w0 = W // fhw
    assert base_bits.size == c0 * h0 * w0
    sd_small = base_bits.reshape(c0, h0, w0)
    sd_c = np.tile(sd_small, (fc, 1, 1))     # (C, h0, w0)
    sd_hw = np.tile(sd_c, (1, fhw, fhw))     # (C, H, W)
    return sd_hw.astype(np.int8)

def chacha20_xor_bits(bits_c_hw: np.ndarray, key32: bytes, nonce12: bytes) -> np.ndarray:
    """Official bit packing (MSB-first) + ChaCha20 keystream XOR.

    We pack/unpack with NumPy packbits/unpackbits using bitorder='big' (default).
    This matches the official GS implementation that relies on np.packbits."""
    if bits_c_hw.dtype != np.uint8:
        flat = bits_c_hw.reshape(-1).astype(np.uint8)
    else:
        flat = bits_c_hw.reshape(-1)
    nbits = int(flat.size)
    # pack MSB-first (bitorder='big' is NumPy default)
    in_bytes = np.packbits(flat, bitorder='big').tobytes()
    nbytes = len(in_bytes)
    cipher = ChaCha20.new(key=key32, nonce=nonce12)  # 32B key + 12B nonce
    stream = cipher.encrypt(bytes(nbytes))
    out_bytes = bytes(a ^ b for a, b in zip(in_bytes, stream))
    # unpack MSB-first and truncate padding bits
    out_bits = np.unpackbits(np.frombuffer(out_bytes, dtype=np.uint8), bitorder='big')[:nbits]
    return out_bits.astype(np.int8).reshape(bits_c_hw.shape)

def sample_latents_from_bits(m_bits: np.ndarray, n_samples: int, l: int = 1, rng: np.random.Generator | None = None) -> torch.Tensor:
    """Distribution-preserving sampling for GS (l=1).

    rng: numpy Generator for reproducible sampling of U(0,1)."""
    assert l == 1
    y = m_bits.astype(np.int8)  # (C,H,W) in {0,1}
    if rng is None:
        rng = np.random.default_rng()
    latents = []
    for _ in range(n_samples):
        u = rng.random((C, H, W), dtype=np.float64)
        z = norm.ppf((u + y) * 0.5)
        latents.append(torch.from_numpy(z).to(torch.float32))
    return torch.stack(latents, dim=0)  # [n,4,64,64]

def main():
    p = argparse.ArgumentParser()
    # ââ ç´æ¥è¾å¥ 32 å­è keyï¼äºæ¥ä¸éä¸ï¼ââ
    p.add_argument("--key_ones", action="store_true", help="ä½¿ç¨ 32 å­èå¨ 1 çå¯é¥ï¼= 256 ä½å¨ 1ï¼")
    p.add_argument("--key_hex", type=str, help="64ä½åå­è¿å¶å¯é¥ï¼=32å­èï¼")
    p.add_argument("--key_bin", type=str, help="256ä½ 01 ä¸²å¯é¥ï¼=32å­èï¼")
    # ââ nonce éæ©ï¼å¯éå¶ä¸ï¼ä¸ä¼ åç¨åºå®å¼ï¼ä»ä¾å¤ç°å®éªï¼ââ
    p.add_argument("--nonce_hex", type=str, help="24ä½åå­è¿å¶ï¼=12å­èï¼")
    p.add_argument("--nonce_bin", type=str, help="96ä½ 01 ä¸²ï¼=12å­èï¼")
    p.add_argument("--nonce_zero", action="store_true", help="ä½¿ç¨ 12 å­èå¨ 0 ç nonceï¼å®éªå¯ç¨ï¼çäº§ç¦ç¨ï¼")
    # å¶å®åæ°
    p.add_argument("--out", required=True, type=str, help="è¾åº .pt è·¯å¾")
    p.add_argument("--n", type=int, default=16, help="ç°å latent æ°ï¼é»è®¤16")
    p.add_argument("--ch", type=int, default=1, help="channel_copyï¼é»è®¤1ï¼")
    p.add_argument("--hw", type=int, default=8, help="hw_copyï¼é»è®¤8ï¼")
    p.add_argument("--seed", type=int, default=12345, help="éæºç§å­ï¼æ§å¶æ°´å°æ©æ£çåºç¡ä½ï¼")
    p.add_argument("--latent_seed", type=int, default=None, help="éæ ·åªå£°éæºç§å­ï¼æ§å¶æ¯ä¸ª latent çå¹å¼éæ ·ï¼ï¼é»è®¤ seed+100000")
    args = p.parse_args()

    key32 = parse_key_32bytes(args)
    nonce12 = parse_nonce_12bytes(args)

    rng = np.random.default_rng(args.seed)
    k_bits = (C * H * W) // (args.ch * args.hw * args.hw)  # å®¹é=256bitï¼é»è®¤ï¼
    base_bits = make_base_bits(k_bits, rng)
    sd = diffuse_bits_to_chw(base_bits, fc=args.ch, fhw=args.hw)  # (C,H,W)

    m_bits = chacha20_xor_bits(sd, key32=key32, nonce12=nonce12)
    latent_seed = (int(args.seed) + 100000) if args.latent_seed is None else int(args.latent_seed)
    rng_lat = np.random.default_rng(latent_seed)
    latents = sample_latents_from_bits(m_bits, n_samples=args.n, l=1, rng=rng_lat)

    meta = dict(
        method="GaussianShading",
        key_repr=("ones" if args.key_ones else ("hex" if args.key_hex else "bin")),
        key_sha256=hashlib.sha256(key32).hexdigest(),
        nonce_zero=args.nonce_zero,
        nonce_hex=(args.nonce_hex or (nonce12.hex() if (not args.nonce_zero and args.nonce_hex is None and args.nonce_bin is None) else None)),
        fc=args.ch, fhw=args.hw, l=1, C=C, H=H, W=W, n=args.n, seed=args.seed,
        note="pass latents*pipe.scheduler.init_noise_sigma to diffusers."
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save({"latents": latents, "meta": meta}, args.out)
    print(f"[OK] saved {latents.shape} to {args.out}")

if __name__ == "__main__":
    main()
