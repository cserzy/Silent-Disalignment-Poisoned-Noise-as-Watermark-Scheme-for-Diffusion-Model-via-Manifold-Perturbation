#!/usr/bin/env python3

import inspect
from pathlib import Path

import torch
from diffusers import AutoencoderKL, PNDMScheduler, UNet2DConditionModel
from diffusers.pipelines.deprecated.alt_diffusion import AltDiffusionPipeline
from diffusers.pipelines.deprecated.alt_diffusion.modeling_roberta_series import (
    RobertaSeriesModelWithTransformation,
)
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from PIL import Image
from transformers import CLIPImageProcessor, XLMRobertaTokenizer


PROJECT_ROOT = Path("/home/yancy/work/dm_backdoor_latent_space/experiment-1_19")
MODEL_DIR = Path("/home/yancy/work/dm_backdoor_latent_space/checkpoints/AltDiffusion-fp16")
PROMPTS_PATH = Path(
    "/home/yancy/work/dm_backdoor_latent_space/prompts/prompts_in_train_v3.anchored-kuan-breast-50.txt"
)
LATENTS_PATH = PROJECT_ROOT / "latents_experiment" / "generate_GS_w.pt"
OUT_DIR = PROJECT_ROOT / "imgs" / "debug_alt_manual_load"


def log(msg: str) -> None:
    print(msg, flush=True)


def load_first_prompt(path: Path) -> str:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                return line
    raise RuntimeError(f"No non-empty prompt found in {path}")


def load_first_latent(path: Path) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu")

    if torch.is_tensor(obj):
        latents = obj
    elif isinstance(obj, dict):
        if "latents" in obj and torch.is_tensor(obj["latents"]):
            latents = obj["latents"]
        else:
            raise KeyError(f"No tensor under 'latents' in {path}; keys={list(obj.keys())}")
    else:
        raise TypeError(f"Unsupported latent container type: {type(obj)}")

    if latents.ndim != 4:
        raise ValueError(f"Expected 4D latent tensor, got shape={tuple(latents.shape)}")

    first = latents[0:1].contiguous().float()
    if tuple(first.shape) != (1, 4, 64, 64):
        raise ValueError(f"Expected first latent shape (1, 4, 64, 64), got {tuple(first.shape)}")

    return first


def build_pipeline(model_dir: Path, dtype: torch.dtype) -> AltDiffusionPipeline:
    log(f"[load] tokenizer <- {model_dir / 'tokenizer'}")
    tokenizer = XLMRobertaTokenizer.from_pretrained(model_dir, subfolder="tokenizer")

    log(f"[load] text_encoder <- {model_dir / 'text_encoder'}")
    text_encoder = RobertaSeriesModelWithTransformation.from_pretrained(
        model_dir, subfolder="text_encoder", torch_dtype=dtype
    )

    log(f"[load] vae <- {model_dir / 'vae'}")
    vae = AutoencoderKL.from_pretrained(model_dir, subfolder="vae", torch_dtype=dtype)

    log(f"[load] unet <- {model_dir / 'unet'}")
    unet = UNet2DConditionModel.from_pretrained(model_dir, subfolder="unet", torch_dtype=dtype)

    log(f"[load] scheduler <- {model_dir / 'scheduler'}")
    scheduler = PNDMScheduler.from_pretrained(model_dir, subfolder="scheduler")

    log(f"[load] feature_extractor <- {model_dir / 'feature_extractor'}")
    feature_extractor = CLIPImageProcessor.from_pretrained(model_dir, subfolder="feature_extractor")

    log(f"[load] safety_checker <- {model_dir / 'safety_checker'}")
    safety_checker = StableDiffusionSafetyChecker.from_pretrained(
        model_dir, subfolder="safety_checker", torch_dtype=dtype
    )

    pipe = AltDiffusionPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler,
        safety_checker=safety_checker,
        feature_extractor=feature_extractor,
        image_encoder=None,
        requires_safety_checker=True,
    )
    return pipe


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16

    log(f"[env] device={device} dtype={dtype}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    prompt = load_first_prompt(PROMPTS_PATH)
    latents = load_first_latent(LATENTS_PATH)
    log(f"[prompt] {prompt}")
    log(f"[latents] loaded from {LATENTS_PATH}")
    log(f"[latents] shape={tuple(latents.shape)} dtype={latents.dtype}")

    pipe = build_pipeline(MODEL_DIR, dtype=dtype)
    call_sig = inspect.signature(pipe.__call__)
    log(f"[check] __call__ has latents={'latents' in call_sig.parameters}")
    log(f"[check] prepare_latents exists={hasattr(pipe, 'prepare_latents')}")

    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    latents = latents.to(device=device, dtype=dtype)
    log(f"[latents] moved_to_device shape={tuple(latents.shape)} dtype={latents.dtype}")
    log(f"[scheduler] init_noise_sigma={pipe.scheduler.init_noise_sigma}")

    result = pipe(
        prompt=prompt,
        height=512,
        width=512,
        num_inference_steps=20,
        guidance_scale=7.5,
        num_images_per_prompt=1,
        latents=latents,
        output_type="pil",
        return_dict=True,
    )

    image = result.images[0]
    if not isinstance(image, Image.Image):
        raise TypeError(f"Expected PIL image, got {type(image)}")

    out_path = OUT_DIR / "manual_load_alt_test.png"
    image.save(out_path)
    log(f"[done] saved image -> {out_path}")
    log(f"[done] nsfw_content_detected={result.nsfw_content_detected}")


if __name__ == "__main__":
    main()
