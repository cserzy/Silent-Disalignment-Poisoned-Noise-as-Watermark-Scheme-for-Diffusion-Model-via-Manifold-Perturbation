#!/usr/bin/env bash
set -uo pipefail

ROOT=/home/yancy/work/dm_backdoor_latent_space
EXP_ROOT=${ROOT}/experiment-1_19
TEMPLATE=${EXP_ROOT}/result_summary/修复水印参数汇总模版-2026年4月19日.xlsx
OUT=${EXP_ROOT}/result_summary/AltDiff-dongman-OMS修复汇总-2026年4月26日.xlsx

python - <<'PY'
import json
import math
import shutil
from pathlib import Path
from openpyxl import load_workbook

ROOT = Path("/home/yancy/work/dm_backdoor_latent_space")
EXP_ROOT = ROOT / "experiment-1_19"
TEMPLATE = EXP_ROOT / "result_summary/修复水印参数汇总模版-2026年4月19日.xlsx"
OUT = EXP_ROOT / "result_summary/AltDiff-dongman-OMS修复汇总-2026年4月26日.xlsx"

runs = {
    "GS": {
        "black": EXP_ROOT / "imgs/vis_alt_gs_oms_safeon_w_att_dongman_seed12345/black_detect/black_ratio_alt_summary.json",
        "detect": EXP_ROOT / "imgs/vis_alt_gs_oms_w_att_dongman_seed12345/detect_gs_alt_oms/gs_detect_invert_oms_decode_alt_summary.json",
        "type": "gs",
    },
    "TR": {
        "black": EXP_ROOT / "imgs/vis_alt_tr_oms_safeon_w_att_0_88_dongman_seed12345/black_detect/black_ratio_alt_summary.json",
        "detect": EXP_ROOT / "imgs/vis_alt_tr_oms_w_att_0_88_dongman_seed12345/detect_tr_alt_oms/treering_detect_alt_oms_img.json",
        "type": "tr",
    },
    "T2S": {
        "black": EXP_ROOT / "imgs/vis_alt_t2s_oms_safeon_w_att_dongman_seed12345/black_detect/black_ratio_alt_summary.json",
        "detect": EXP_ROOT / "imgs/vis_alt_t2s_oms_w_att_dongman_seed12345/detect_t2s_alt_oms/detect_t2s_alt_oms.json",
        "type": "t2s",
    },
}

def load_json(p):
    if not p.exists():
        print(f"[WARN] missing: {p}")
        return {}
    try:
        return json.loads(p.read_text())
    except Exception as e:
        print(f"[WARN] failed to read json: {p} | {e}")
        return {}

def find_first(d, keys):
    if not isinstance(d, dict):
        return None
    for k in keys:
        if k in d:
            return d[k]
    for v in d.values():
        if isinstance(v, dict):
            r = find_first(v, keys)
            if r is not None:
                return r
    return None

def to_pct(x):
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None

def pct_fmt(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    return float(x)

def extract_black_rate(path):
    d = load_json(path)
    return to_pct(find_first(d, ["black_rate", "safety_trigger_rate"]))

def extract_detect(path, typ):
    d = load_json(path)
    if typ == "gs":
        det = find_first(d, ["detected_rate", "detect_rate"])
        bit = find_first(d, ["bit_acc_mean", "bit_acc"])
        return to_pct(det), to_pct(bit)
    if typ == "tr":
        s = d.get("summary", d)
        det = find_first(s, ["ok_rate", "detected_rate", "detect_rate"])
        if det is None and isinstance(d, dict) and "detail_rows" in d:
            rows = d.get("detail_rows") or []
            vals = []
            for r in rows:
                if isinstance(r, dict) and "ok" in r:
                    vals.append(float(r["ok"]))
                elif isinstance(r, (list, tuple)):
                    header = d.get("detail_header") or []
                    if "ok" in header:
                        vals.append(float(r[header.index("ok")]))
            if vals:
                det = sum(vals) / len(vals)
        return to_pct(det), None
    if typ == "t2s":
        s = find_first(d, ["summary_image_inversion_alt_oms"])
        if not isinstance(s, dict):
            s = d.get("summary", d)
        det = find_first(s, ["detected_mean", "detected_rate"])
        bit = find_first(s, ["acc_msg_oracle_mean"])
        return to_pct(det), to_pct(bit)
    return None, None

def find_alt_rows(ws):
    rows = {}
    current_model = None
    for r in range(1, ws.max_row + 1):
        a = ws.cell(r, 1).value
        b = ws.cell(r, 2).value
        if a:
            current_model = str(a).strip()
        if current_model == "AltDiff" and b:
            rows[str(b).strip()] = r
    return rows

def find_rev_cols(ws):
    # 模板表头：C:E=w_att, F:H=rev, I:K=Diff.
    # rev 下方顺序：NSFW, Detection_Rate, Bit_Acc
    for c in range(1, ws.max_column + 1):
        v1 = ws.cell(1, c).value
        if v1 and str(v1).strip().lower() == "rev":
            return {
                "NSFW": c,
                "Detection_Rate": c + 1,
                "Bit_Acc": c + 2,
            }
    return {"NSFW": 6, "Detection_Rate": 7, "Bit_Acc": 8}

if not TEMPLATE.exists():
    raise FileNotFoundError(TEMPLATE)

OUT.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(TEMPLATE, OUT)

wb = load_workbook(OUT)
ws = wb.active

rows = find_alt_rows(ws)
cols = find_rev_cols(ws)

print("[INFO] target rows:", rows)
print("[INFO] rev cols:", cols)

for method, cfg in runs.items():
    row = rows.get(method)
    if row is None:
        print(f"[WARN] no AltDiff row for {method}, skip")
        continue

    black_rate = extract_black_rate(cfg["black"])
    det_rate, bit_acc = extract_detect(cfg["detect"], cfg["type"])

    ws.cell(row, cols["NSFW"]).value = pct_fmt(black_rate)
    ws.cell(row, cols["Detection_Rate"]).value = pct_fmt(det_rate)
    ws.cell(row, cols["Bit_Acc"]).value = pct_fmt(bit_acc) if bit_acc is not None else None

    for c in [cols["NSFW"], cols["Detection_Rate"], cols["Bit_Acc"]]:
        ws.cell(row, c).number_format = "0.00%"

    print(f"[{method}] black_rate={black_rate} detected={det_rate} bit_acc={bit_acc}")

wb.save(OUT)
print(f"[OK] wrote: {OUT}")
PY
