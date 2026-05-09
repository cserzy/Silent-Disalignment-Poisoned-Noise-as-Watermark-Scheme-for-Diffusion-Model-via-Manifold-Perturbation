# PNaW Release: Scripts and Latent Assets

This repository releases a compact public subset of the code and assets used in our **PNaW (Poisoned Noise-as-Watermark)** experiments from the paper:

> **Silent Disalignment: Poisoned Noise-as-Watermark Scheme for Diffusion Model via Manifold Perturbation**

It is organized around the same three-step workflow used in the paper: generate a latent bank `z_T`, synthesize images from the saved `.pt` latents, and run the corresponding watermark detector or inversion-based evaluation.

This release includes:

- attacked or watermarked latent banks stored as `.pt`
- scripts for generating images from released latent banks
- watermark detection and inversion utilities
- representative prompt files and latent assets used in our experiments

## Quick Start

1. Create the recommended environment:

```bash
conda env create -f environment.hijacking.yml
conda activate Hijacking
```

2. Check the main released assets:

- [`latents_experiment/`](latents_experiment/)
- [`prompt/`](prompt/)
- [`script-experiment/`](script-experiment/)

3. Pick a workflow below:

- generate attacked latent banks in [1. Generate Attacked Latent Banks](#1-generate-attacked-latent-banks)
- generate images from `.pt` latents in [2. Generate Images from Released `.pt` Latents](#2-generate-images-from-released-pt-latents)
- run watermark detection in [3. Detection](#3-detection)

## Repository Layout

- `latents_experiment/`: released `.pt` latent banks and a small number of associated metadata files
- `prompt/`: released prompt sets, including `AdultSuggestive.txt` and `AnimeMinor.txt`
- `script-experiment/generate_zT/`: scripts for generating attacked watermark latents
- `script-experiment/`: scripts for generating images from latent banks
- `script-experiment/detect/`: watermark detectors and related inversion utilities
- `script-experiment/nsfw_score_report_ring_wm_only_exposed_only-12.29.py`: NSFW scoring script used in our evaluation pipeline
- `third_party/T2SMark/`: bundled T2SMark dependency snapshot used by our codebase

## Dependencies

These scripts are built around the official watermark implementations and common diffusion tooling. In practice, you should prepare:

- Diffusers-based Stable Diffusion checkpoints such as SD v1.4, SD v1.5, and SD v2.1
- AltDiffusion for the AltDiffusion examples
- the bundled [`third_party/T2SMark/`](third_party/T2SMark/) dependency snapshot
- the helper modules already released in this repository, such as `prc.py` and `pseudogaussians.py`

Some watermark components are vendored directly into this repository, while others are integrated into our released attack and detection scripts.

Some pipelines also require scheme-specific side products that are not fully bundled in this compact release:

- PRC detection needs watermark metadata generated during watermark construction
- T2S detection needs the corresponding cluster metadata `.pt`

For environment setup, this repository now includes two exported dependency files at the repository root:

- `environment.hijacking.yml`: the closest snapshot of our conda environment
- `requirements.txt`: a pip-style dependency list exported from the same environment

We recommend using the conda environment file first:

```bash
conda env create -f environment.hijacking.yml
conda activate Hijacking
```

If you prefer a pip-based setup, you can instead install:

```bash
pip install -r requirements.txt
```

## Released Prompt Files

We renamed the public prompt files to simpler names:

- `prompt/AdultSuggestive.txt`
- `prompt/AnimeMinor.txt`

When adapting old internal scripts, replace old `prompts/...txt` paths with these new names.

## 1. Generate Attacked Latent Banks

All commands below are examples. Replace `/path/to/...` with your local checkpoint paths as needed.

### 1.1 GS

```bash
CUDA_VISIBLE_DEVICES=0 python script-experiment/generate_zT/generate_GS_zT_w_att.py \
  --model_id /path/to/checkpoints/sd1-4-diffusers \
  --prompts prompt/AdultSuggestive.txt \
  --outdir tmp_gs \
  --margin 0.3 \
  --steps 30 --cfg 7.5 --height 512 --width 512 \
  --ssc_d_wm 256 \
  --gs_seed 12345 \
  --gs_key_hex aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa \
  --gs_nonce_zero \
  --gs_ch 4 --gs_hw 4 \
  --lambda1 0.88 \
  --export_zt_only \
  --n_zt 16 --zt_seed 12345 \
  --export_latents_dir latents_experiment \
  --export_latents_name generate_GS_w_att.pt
```

### 1.2 T2S

```bash
CUDA_VISIBLE_DEVICES=0 python script-experiment/generate_zT/generate_T2S_zT_w_att.py \
  --model_id /path/to/checkpoints/sd1-4-diffusers \
  --prompts prompt/AdultSuggestive.txt \
  --cluster_pt latents_experiment/generate_T2S_w.pt \
  --outdir tmp_t2s \
  --K 16 \
  --seed 12345 \
  --lam1 0.88 \
  --ssc_cal_N 12 --ssc_energy_ratio 0.90 --ssc_mini_steps 6 \
  --t2s_tau 0.674 \
  --export_latents_dir latents_experiment \
  --out_pt_name generate_T2S_w_att.pt \
  --out_pre_pt_name generate_T2S_w_att_pre.pt \
  --meta_pt_name generate_T2S_w_att_meta.pt \
  --meta_json_name generate_T2S_w_att_meta.json
```

### 1.3 PRC

```bash
CUDA_VISIBLE_DEVICES=0 python script-experiment/generate_zT/generate_PRC_zT_w_att.py \
  --model_id /path/to/checkpoints/sd1-4-diffusers \
  --prompts prompt/AnimeMinor.txt \
  --outdir tmp_prc \
  --negative_prompt "" \
  --steps 50 --cfg 7.5 --height 512 --width 512 \
  --rows 4 --cols 4 --gen_bs 4 \
  --seed 12345 \
  --ssc_N_cal 12 --ssc_energy_ratio 0.900 --ssc_mini_steps 6 \
  --ssc_d_sens_max 256 --ssc_d_wm 256 \
  --reuse_ssc 1 \
  --prc_message_length <message_length> --prc_error_prob 0.01 \
  --master_key <your_prc_master_key> \
  --lam1 0.89 \
  --save_zT 0 \
  --export_zT16_only 1 \
  --export_latents_dir latents_experiment \
  --export_latents_name generate_PRC_w_att_clip.pt \
  --wm_meta_subdir wm_meta_prc
```

Note:

- PRC detection needs the watermark metadata generated during this stage

### 1.4 TR

```bash
CUDA_VISIBLE_DEVICES=0 python script-experiment/generate_zT/generate_TR_zT_w_att.py \
  --model_id /path/to/checkpoints/sd1-4-diffusers \
  --prompts prompt/AdultSuggestive.txt \
  --outdir latents_experiment \
  --out_pt latents_experiment/generate_TR_w_att.pt \
  --height 512 --width 512 \
  --lam1 0.88 --seed 12345 \
  --ssc_N_cal 12 --ssc_mini_steps 6 \
  --tr_w_seed 12345 \
  --tr_w_pattern ring \
  --tr_w_mask_shape circle \
  --tr_w_radius 9 \
  --tr_w_channel -1 \
  --tr_w_injection complex
```

## 2. Generate Images from Released `.pt` Latents

### 2.1 Stable Diffusion v1.4 / v1.5 / v2.1

Use [`script-experiment/gen_from_zT_bank_multi_models-1_19.py`](script-experiment/gen_from_zT_bank_multi_models-1_19.py).

Example with GS:

```bash
CUDA_VISIBLE_DEVICES=0 python script-experiment/gen_from_zT_bank_multi_models-1_19.py \
  --model_id /path/to/checkpoints/sd1-4-diffusers \
  --prompts prompt/AdultSuggestive.txt \
  --zT_pt latents_experiment/generate_GS_w_att.pt \
  --outdir outputs/vis_sd14_GS_w_att_seed12345 \
  --steps 50 --cfg 7.5 --height 512 --width 512 \
  --n_per_prompt 4 --start_latent 0 \
  --dtype fp16 --seed 12345 \
  --negative_prompt ""
```

You can similarly swap `--zT_pt` to other released latents, for example:

- `latents_experiment/generate_T2S_w_att.pt`
- `latents_experiment/generate_TR_w_att.pt`
- `latents_experiment/generate_PRC_w_att_clip.pt`
- `latents_experiment/generate_GS_w_att_delrepair.pt`
- `latents_experiment/generate_T2S_w_att_delssc.pt`

### 2.2 AltDiffusion Safe-Off

Use [`script-experiment/gen_from_zT_bank_alt_diffusion-1_19.py`](script-experiment/gen_from_zT_bank_alt_diffusion-1_19.py) with `--disable_safety_checker`.

```bash
CUDA_VISIBLE_DEVICES=0 python script-experiment/gen_from_zT_bank_alt_diffusion-1_19.py \
  --model_id /path/to/checkpoints/AltDiffusion-fp16 \
  --prompts prompt/AdultSuggestive.txt \
  --zT_pt latents_experiment/generate_T2S_w_att_delrepair.pt \
  --outdir outputs/vis_alt_t2s_delrepair_seed12345 \
  --steps 50 --cfg 7.5 --height 512 --width 512 \
  --device cuda --dtype fp16 \
  --n_per_prompt 4 --start_latent 0 --seed 12345 \
  --disable_safety_checker
```

### 2.3 AltDiffusion Safe-On

Run the same script without `--disable_safety_checker`.

```bash
CUDA_VISIBLE_DEVICES=0 python script-experiment/gen_from_zT_bank_alt_diffusion-1_19.py \
  --model_id /path/to/checkpoints/AltDiffusion-fp16 \
  --prompts prompt/AdultSuggestive.txt \
  --zT_pt latents_experiment/generate_T2S_w_att_delrepair.pt \
  --outdir outputs/vis_alt_safeon_t2s_delrepair_seed12345 \
  --steps 50 --cfg 7.5 --height 512 --width 512 \
  --device cuda --dtype fp16 \
  --n_per_prompt 4 --start_latent 0 --seed 12345
```

## 3. Detection

### 3.1 Standard GS

```bash
CUDA_VISIBLE_DEVICES=0 python script-experiment/detect/detect_GS.py \
  --model_id /path/to/checkpoints/sd1-4-diffusers \
  --run_dir outputs/vis_sd14_GS_w_att_seed12345 \
  --out_dir outputs/vis_sd14_GS_w_att_seed12345/detect_gs \
  --save_zt
```

### 3.2 Standard TR

```bash
CUDA_VISIBLE_DEVICES=0 python script-experiment/detect/detect_TR.py \
  --img_dir outputs/vis_sd14_TR_w_att_seed12345/sliced \
  --model_id /path/to/checkpoints/sd1-4-diffusers \
  --out_xlsx outputs/vis_sd14_TR_w_att_seed12345/sd14_TR_detect.xlsx
```

### 3.3 Standard T2S

The public README uses `detect_T2S.py` as the main T2S detector entry. In this release, it corresponds to the compatibility-oriented detector.

```bash
CUDA_VISIBLE_DEVICES=0 python script-experiment/detect/detect_T2S.py \
  --cluster_meta_pt latents_experiment/generate_T2S_w_att_delrepair_meta.pt \
  --images_glob 'outputs/vis_sd14_T2S_w_att_seed12345/sliced/*.png' \
  --model_id /path/to/checkpoints/sd1-4-diffusers \
  --out_json outputs/vis_sd14_T2S_w_att_seed12345/sd14_T2S_w_att_detect.json \
  --out_csv outputs/vis_sd14_T2S_w_att_seed12345/sd14_T2S_w_att_detect.csv \
  --fp16 \
  --num_inversion_steps 10 \
  --inv_guidance 1.0 \
  --resize 512
```

### 3.4 Standard PRC

Use the standard PRC detector:

- `script-experiment/detect/prc_detect_global_official_align-1_21_fixdim-meg-z_T.py`

Example:

```bash
CUDA_VISIBLE_DEVICES=0 python script-experiment/detect/prc_detect_global_official_align-1_21_fixdim-meg-z_T.py \
  --run_dir outputs/vis_sd14_PRC_w_att_seed12345 \
  --model_id /path/to/checkpoints/sd1-4-diffusers \
  --steps 50 \
  --guidance 7.5 \
  --dtype fp32 \
  --var 1.5 \
  --fpr 1e-2 \
  --inv_bs 1 \
  --debug
```

### 3.5 AltDiffusion GS

```bash
CUDA_VISIBLE_DEVICES=0 python script-experiment/detect/detect_GS_alt.py \
  --model_id /path/to/checkpoints/AltDiffusion-fp16 \
  --run_dir outputs/vis_alt_ablate_gs_delrepair_seed12345 \
  --out_dir outputs/vis_alt_ablate_gs_delrepair_seed12345/detect_gs_alt \
  --dtype fp16 \
  --inv_steps 50 \
  --save_zt
```

### 3.6 AltDiffusion TR

```bash
CUDA_VISIBLE_DEVICES=0 python script-experiment/detect/detect_TR_alt.py \
  --img_dir outputs/vis_alt_ablate_tr_delrepair_seed12345 \
  --model_id /path/to/checkpoints/AltDiffusion-fp16 \
  --mode img \
  --detect_prompt empty \
  --guidance_scale 1.0 \
  --steps 50 \
  --inv_steps 50 \
  --fp16 1 \
  --out_dir outputs/vis_alt_ablate_tr_delrepair_seed12345/detect_tr_alt \
  --save_zt
```

### 3.7 AltDiffusion T2S

```bash
CUDA_VISIBLE_DEVICES=0 python script-experiment/detect/detect_T2S_alt.py \
  --model_id /path/to/checkpoints/AltDiffusion-fp16 \
  --run_dir outputs/vis_alt_ablate_t2s_delrepair_seed12345 \
  --out_dir outputs/vis_alt_ablate_t2s_delrepair_seed12345/detect_t2s_alt \
  --cluster_meta_pt latents_experiment/generate_T2S_w_att_delrepair_meta.pt \
  --dtype fp16 \
  --inv_steps 50 \
  --save_zt \
  --compute_auc
```

### 3.8 AltDiffusion PRC

Use the AltDiffusion PRC detector:

- `script-experiment/detect/prc_detect_alt_global_official_align.py`

Example:

```bash
CUDA_VISIBLE_DEVICES=0 python script-experiment/detect/prc_detect_alt_global_official_align.py \
  --model_id /path/to/checkpoints/AltDiffusion-fp16 \
  --run_dir outputs/vis_alt_ablate_prc_delrepair_seed12345 \
  --meta_root /path/to/generated_prc_meta_root \
  --dtype fp16 \
  --inv_steps 50 \
  --inv_bs 1 \
  --fpr 1e-2 \
  --master_key <your_prc_master_key> \
  --message_length <message_length> \
  --max_bp_iter 5000 \
  --save_zt \
  --save_zt_dir outputs/vis_alt_ablate_prc_delrepair_seed12345/detect_prc_alt/latents_prc_alt \
  --out_csv outputs/vis_alt_ablate_prc_delrepair_seed12345/detect_prc_alt/detect_results_prcGLOBAL_alt.csv
```

## 4. NSFW Scoring

We also release the NSFW scoring script used in our evaluation:

- [`script-experiment/nsfw_score_report_ring_wm_only_exposed_only-12.29.py`](script-experiment/nsfw_score_report_ring_wm_only_exposed_only-12.29.py)

Single-run example:

```bash
python script-experiment/nsfw_score_report_ring_wm_only_exposed_only-12.29.py \
  --manifests outputs/vis_alt_gs_w_att_seed12345/sliced/manifest.csv \
  --out_dir outputs/vis_alt_gs_w_att_seed12345/nsfw_report \
  --report_out outputs/vis_alt_gs_w_att_seed12345/nsfw_report/report.xlsx \
  --threshold 0.6 \
  --sweep 0.2,0.3,0.4,0.5,0.6,0.7,0.8
```

Parallel scoring style, following our batch script:

```bash
MAX_JOBS=4
THRESH=0.6
SWEEP="0.2,0.3,0.4,0.5,0.6,0.7,0.8"

python script-experiment/nsfw_score_report_ring_wm_only_exposed_only-12.29.py \
  --manifests outputs/vis_alt_gs_w_att_seed12345/sliced/manifest.csv \
  --out_dir outputs/vis_alt_gs_w_att_seed12345/nsfw_report \
  --report_out outputs/vis_alt_gs_w_att_seed12345/nsfw_report/report.xlsx \
  --threshold "${THRESH}" \
  --sweep "${SWEEP}"
```

For larger batches, you can wrap the same command in your own shell loop or scheduler.

## 5. OMS / Repair Pipeline

This release also includes the OMS-related repair utility:

- [`script-experiment/oms_repair_pt.py`](script-experiment/oms_repair_pt.py)

Representative example:

```bash
python script-experiment/oms_repair_pt.py \
  --mode fit_apply \
  --in_pt latents_experiment/generate_GS_w_att.pt \
  --target_pt latents_experiment/generate_GAUSS_w_aligned_vis.pt \
  --out_pt latents_experiment/generate_GS_w_att_oms_gauss_aligned.pt \
  --out_q_pt latents_experiment/oms_Q_GS_w_att_to_gauss_aligned.pt \
  --out_meta_json latents_experiment/oms_Q_GS_w_att_to_gauss_aligned.json \
  --q_seed 12345 \
  --block_size 64 \
  --blend_alpha 0.2 \
  --match_target_std 1 \
  --device cpu \
  --dtype fp32 \
  --verbose
```

## 6. Notes

- This public release is intentionally compact. Some internal experiment folders, cached metadata, and auxiliary artifacts are not included.
- PRC-related detection requires metadata generated together with the watermark construction stage.
- T2S detection requires the matching cluster metadata file for the chosen latent bank.
