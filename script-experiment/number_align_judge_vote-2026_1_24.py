#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import csv
import time
import base64
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
DEFAULT_API_KEY = "sk-UT4yIfRIhn7y82QHnB8ADqNi15AttONhf1hj2bLPokWYbCeE"

# --------- Utilities ---------

# New filename: P00_01.png (prompt_id=00, img_idx=01)
PROMPT_ID_RE_P   = re.compile(r"^[Pp](\d+)_(\d+)", re.IGNORECASE)  # P00_01.png
# Old filename: 000_xxx.png / p0000_xxx.png
PROMPT_ID_RE_OLD = re.compile(r"^(?:p)?(\d+)_", re.IGNORECASE)     # 000_xxx.png / p0000_xxx.png

def read_prompts(prompt_path: str) -> list[str]:
    """
    Each non-empty line is a prompt. Line index is used as prompt_id.
    Example: line 0 -> prompt_id 0
    """
    prompts: list[str] = []
    with open(prompt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            prompts.append(line)
    return prompts

def guess_mime(path: Path) -> str:
    suf = path.suffix.lower()
    if suf == ".png":
        return "image/png"
    if suf in [".jpg", ".jpeg"]:
        return "image/jpeg"
    if suf == ".webp":
        return "image/webp"
    return "image/png"

def image_to_data_url(img_path: Path) -> str:
    mime = guess_mime(img_path)
    data = img_path.read_bytes()
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def parse_prompt_id_from_filename(name: str) -> int | None:
    """
    Supported filename formats:

      New (your current setting):
        P00_01.png  -> prompt_id = 0
        P49_03.png  -> prompt_id = 49

      Old (backward compatible):
        000_Anything_else.png   -> prompt_id = 0
        p0000_Anything_else.png -> prompt_id = 0

    Notes:
      - We only need prompt_id; the second index (00-03) is ignored.
    """
    m = PROMPT_ID_RE_P.match(name)
    if m:
        return int(m.group(1))

    m = PROMPT_ID_RE_OLD.match(name)
    if m:
        return int(m.group(1))

    return None

def normalize_yes_no(text: str) -> str:
    """
    Force to YES or NO when possible.
    Returns one of: YES, NO, UNK, ERR
    """
    if not text:
        return "ERR"
    t = text.strip().upper()

    if t.startswith("YES"):
        return "YES"
    if t.startswith("NO"):
        return "NO"

    # tolerant fallback
    has_yes = "YES" in t
    has_no = "NO" in t
    if has_yes and not has_no:
        return "YES"
    if has_no and not has_yes:
        return "NO"
    return "UNK"

def call_once(client: OpenAI, model: str, prompt_text: str, data_url: str, max_tokens: int) -> str:
    """
    Ask model to answer strictly YES or NO.
    """
    system_text = (
        "You are a strict judge. "
        "Given a text prompt describing object counts and an image, "
        "answer ONLY 'YES' if the image matches the prompt with correct quantities, "
        "otherwise answer ONLY 'NO'. "
        "No extra words."
    )

    user_text = (
        "Check whether the image matches this prompt EXACTLY, especially the number of objects.\n"
        f"PROMPT: {prompt_text}\n"
        "Reply with YES or NO only."
    )

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_text},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ],
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content or ""

def call_with_retry(
    client: OpenAI,
    model: str,
    prompt_text: str,
    data_url: str,
    max_tokens: int,
    retries: int = 3,
    base_sleep: float = 1.0,
) -> tuple[str, str]:
    """
    Returns: (verdict, raw_text)
    verdict in {YES, NO, UNK, ERR}
    """
    last_err = None
    for i in range(retries + 1):
        try:
            raw = call_once(client, model, prompt_text, data_url, max_tokens=max_tokens)
            verdict = normalize_yes_no(raw)
            return verdict, raw
        except Exception as e:
            last_err = repr(e)
            time.sleep(base_sleep * (2 ** i))
    return "ERR", last_err or "ERR"

def parse_models(args) -> list[str]:
    """
    Prefer --models (comma-separated). Keep backward compatible --model.
    """
    if args.models and args.models.strip():
        models = [m.strip() for m in args.models.split(",") if m.strip()]
        if models:
            return models
    # fallback
    return [args.model.strip()]

def majority_vote(verdict_by_model: dict[str, str]) -> tuple[str, dict[str, int]]:
    """
    Only YES and NO count as votes.
    UNK and ERR are abstain.
    Tie -> UNK.
    """
    counts = {"YES": 0, "NO": 0, "UNK": 0, "ERR": 0}
    for v in verdict_by_model.values():
        counts[v] = counts.get(v, 0) + 1

    yes = counts.get("YES", 0)
    no = counts.get("NO", 0)

    if yes > no:
        final_v = "YES"
    elif no > yes:
        final_v = "NO"
    else:
        final_v = "UNK"  # tie or no valid votes -> UNK

    counts["CONSIDERED_VOTES"] = yes + no
    return final_v, counts

def write_csv(path: Path, fieldnames: list[str], rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

def summarize_results(rows: list[dict], prompts: list[str], verdict_key: str, extra_group_keys: list[str] | None = None) -> list[dict]:
    """
    Summarize counts by prompt_id (and optionally by extra_group_keys like model).
    verdict_key is "verdict" or "final_verdict".
    """
    extra_group_keys = extra_group_keys or []
    per = {}  # (extra..., pid) -> counts

    for r in rows:
        pid = int(r["prompt_id"])
        key = tuple([r[k] for k in extra_group_keys] + [pid])
        per.setdefault(key, {"total": 0, "YES": 0, "NO": 0, "UNK": 0, "ERR": 0})
        v = r[verdict_key]
        per[key]["total"] += 1
        per[key][v] = per[key].get(v, 0) + 1

    summary_rows: list[dict] = []

    # overall totals per group
    overall = {}  # extra... -> counts
    for key, stat in per.items():
        *extra_vals, pid = key
        gkey = tuple(extra_vals)

        overall.setdefault(gkey, {"total": 0, "YES": 0, "NO": 0, "UNK": 0, "ERR": 0})
        for k in ["total", "YES", "NO", "UNK", "ERR"]:
            overall[gkey][k] += stat.get(k, 0)

        total = stat["total"]
        yes = stat.get("YES", 0)
        no = stat.get("NO", 0)
        unk = stat.get("UNK", 0)
        err = stat.get("ERR", 0)

        yes_rate = yes / total if total else 0.0
        no_rate = no / total if total else 0.0

        row = {}
        for k, v in zip(extra_group_keys, extra_vals):
            row[k] = v
        row.update({
            "prompt_id": pid,
            "prompt_text": prompts[pid] if 0 <= pid < len(prompts) else "",
            "total": total,
            "yes": yes,
            "no": no,
            "unk": unk,
            "err": err,
            "yes_rate": f"{yes_rate:.4f}",
            "no_rate": f"{no_rate:.4f}",
            "error_rate_no_only": f"{no_rate:.4f}",
            "error_rate_no_unk_err": f"{(no + unk + err) / total:.4f}" if total else "0.0000",
        })
        summary_rows.append(row)

    # append ALL row per group
    for gkey, stat in overall.items():
        total = stat["total"]
        yes = stat.get("YES", 0)
        no = stat.get("NO", 0)
        unk = stat.get("UNK", 0)
        err = stat.get("ERR", 0)

        row = {}
        for k, v in zip(extra_group_keys, gkey):
            row[k] = v
        row.update({
            "prompt_id": -1,
            "prompt_text": "ALL",
            "total": total,
            "yes": yes,
            "no": no,
            "unk": unk,
            "err": err,
            "yes_rate": f"{yes / total:.4f}" if total else "0.0000",
            "no_rate": f"{no / total:.4f}" if total else "0.0000",
            "error_rate_no_only": f"{no / total:.4f}" if total else "0.0000",
            "error_rate_no_unk_err": f"{(no + unk + err) / total:.4f}" if total else "0.0000",
        })
        summary_rows.append(row)

    # sort: group keys then prompt_id
    def sort_key(r):
        return tuple([r.get(k, "") for k in extra_group_keys] + [int(r["prompt_id"])])
    summary_rows.sort(key=sort_key)
    return summary_rows

# --------- Main ---------

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--base_url", type=str, default=os.getenv("OPENAI_BASE_URL", "https://xiaoai.plus/v1"))
    ap.add_argument("--api_key", type=str, default=DEFAULT_API_KEY)

    # backward-compatible single model
    ap.add_argument("--model", type=str, default="gpt-4o")

    # new: multi-model
    ap.add_argument(
        "--models",
        type=str,
        default="deepseek-v3.1-250821,gpt-5.2-2025-12-11,o4-mini",
        help="comma-separated models for voting",
    )

    ap.add_argument("--prompt_file", type=str, required=True)
    ap.add_argument("--img_root", type=str, required=True)
    ap.add_argument("--img_glob", type=str, default="**/sliced/*.png")
    ap.add_argument("--max_tokens", type=int, default=8)

    ap.add_argument("--workers", type=int, default=4, help="thread workers for parallel API calls")
    ap.add_argument("--retries", type=int, default=3)

    # outputs (relative paths will be placed under img_root)
    ap.add_argument("--out_details_by_model_csv", type=str, default="number_align_details_by_model.csv")
    ap.add_argument("--out_details_voted_csv", type=str, default="number_align_details_voted.csv")
    ap.add_argument("--out_summary_by_model_csv", type=str, default="number_align_summary_by_model.csv")
    ap.add_argument("--out_summary_voted_csv", type=str, default="number_align_summary_voted.csv")

    args = ap.parse_args()

    if not args.api_key:
        raise RuntimeError(
            "API key is empty. Set environment variable OPENAI_API_KEY "
            "or pass --api_key (not recommended)."
        )

    models = parse_models(args)
    if not models:
        raise RuntimeError("No models specified via --models or --model.")

    # Build client
    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    # Load prompts
    prompts = read_prompts(args.prompt_file)
    if not prompts:
        raise RuntimeError(f"No prompts loaded from: {args.prompt_file}")

    # Collect images
    img_root = Path(args.img_root)
    img_paths = sorted(img_root.glob(args.img_glob))
    if not img_paths:
        raise RuntimeError(f"No images found under {img_root} with glob {args.img_glob}")

    # Prepare jobs: (img_path, pid, prompt_text, data_url)
    jobs = []
    for p in img_paths:
        pid = parse_prompt_id_from_filename(p.name)
        if pid is None:
            continue
        if pid < 0 or pid >= len(prompts):
            continue
        # cache data_url once per image
        data_url = image_to_data_url(p)
        jobs.append((p, pid, prompts[pid], data_url))

    if not jobs:
        raise RuntimeError("No valid jobs: cannot map images to prompt ids. Check filename prefixes and prompt lines.")

    print(f"[INFO] prompts={len(prompts)} images_found={len(img_paths)} jobs_mapped={len(jobs)} models={models}")

    # Run: per image per model
    detail_rows: list[dict] = []  # long table
    verdict_map: dict[str, dict[str, str]] = {}  # image_path -> {model -> verdict}
    raw_map: dict[str, dict[str, str]] = {}      # image_path -> {model -> raw}

    total_tasks = len(jobs) * len(models)
    done = 0

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = []
        for (img_path, pid, ptxt, data_url) in jobs:
            img_key = str(img_path)
            verdict_map.setdefault(img_key, {})
            raw_map.setdefault(img_key, {})
            for model in models:
                fut = ex.submit(
                    call_with_retry,
                    client,
                    model,
                    ptxt,
                    data_url,
                    args.max_tokens,
                    args.retries,
                    1.0,
                )
                futs.append((fut, img_path, pid, ptxt, model))

        for fut, img_path, pid, ptxt, model in futs:
            verdict, raw = fut.result()
            img_key = str(img_path)

            verdict_map[img_key][model] = verdict
            raw_map[img_key][model] = raw

            detail_rows.append({
                "prompt_id": pid,
                "prompt_text": ptxt,
                "image_path": img_key,
                "model": model,
                "verdict": verdict,
                "raw_response": raw,
            })

            done += 1
            if done % 100 == 0 or done == total_tasks:
                print(f"[INFO] done {done}/{total_tasks}")

    # Voting per image
    voted_rows: list[dict] = []
    for (img_path, pid, ptxt, _data_url) in jobs:
        img_key = str(img_path)
        v_by_model = verdict_map.get(img_key, {})
        final_v, counts = majority_vote(v_by_model)

        # pack model verdicts for easy inspection
        model_verdicts = ";".join([f"{m}={v_by_model.get(m,'ERR')}" for m in models])

        voted_rows.append({
            "prompt_id": pid,
            "prompt_text": ptxt,
            "image_path": img_key,
            "final_verdict": final_v,
            "yes_votes": counts.get("YES", 0),
            "no_votes": counts.get("NO", 0),
            "unk_votes": counts.get("UNK", 0),
            "err_votes": counts.get("ERR", 0),
            "considered_votes": counts.get("CONSIDERED_VOTES", 0),
            "model_verdicts": model_verdicts,
        })

    # Resolve output paths under img_root
    def under_root(p: str) -> Path:
        outp = Path(p)
        if not outp.is_absolute():
            outp = img_root / outp
        return outp

    out_details_by_model = under_root(args.out_details_by_model_csv)
    out_details_voted = under_root(args.out_details_voted_csv)
    out_summary_by_model = under_root(args.out_summary_by_model_csv)
    out_summary_voted = under_root(args.out_summary_voted_csv)

    # Write details
    write_csv(
        out_details_by_model,
        fieldnames=["prompt_id", "prompt_text", "image_path", "model", "verdict", "raw_response"],
        rows=detail_rows,
    )
    print(f"[OK] wrote details_by_model: {out_details_by_model.resolve()}")

    write_csv(
        out_details_voted,
        fieldnames=[
            "prompt_id", "prompt_text", "image_path",
            "final_verdict", "yes_votes", "no_votes", "unk_votes", "err_votes",
            "considered_votes", "model_verdicts"
        ],
        rows=voted_rows,
    )
    print(f"[OK] wrote details_voted:    {out_details_voted.resolve()}")

    # Summaries
    summary_by_model = summarize_results(
        rows=detail_rows,
        prompts=prompts,
        verdict_key="verdict",
        extra_group_keys=["model"],
    )
    write_csv(
        out_summary_by_model,
        fieldnames=[
            "model",
            "prompt_id", "prompt_text", "total", "yes", "no", "unk", "err",
            "yes_rate", "no_rate",
            "error_rate_no_only", "error_rate_no_unk_err",
        ],
        rows=summary_by_model,
    )
    print(f"[OK] wrote summary_by_model: {out_summary_by_model.resolve()}")

    summary_voted = summarize_results(
        rows=voted_rows,
        prompts=prompts,
        verdict_key="final_verdict",
        extra_group_keys=[],
    )
    write_csv(
        out_summary_voted,
        fieldnames=[
            "prompt_id", "prompt_text", "total", "yes", "no", "unk", "err",
            "yes_rate", "no_rate",
            "error_rate_no_only", "error_rate_no_unk_err",
        ],
        rows=summary_voted,
    )
    print(f"[OK] wrote summary_voted:    {out_summary_voted.resolve()}")

    print("[DONE] 重点看 summary_voted.csv（投票结论）和 details_voted.csv（每张图票数与分歧）。")

if __name__ == "__main__":
    main()
