"""
train_lora.py
Simple LoRA fine-tune for Stable Diffusion using Hugging Face diffusers + PEFT + Accelerate.

Dataset expectation:
- ml/data/images/*.jpg  (or png)
- ml/data/labels.csv    (two columns: filename,prompt)  (no header) OR optional auto-generated prompts.

Note: Run on a machine with GPU(s), CUDA, and a working accelerate config.
"""

import os
import random
import yaml
import argparse
from pathlib import Path
from PIL import Image
from dataclasses import dataclass
from typing import List, Optional

import torch
from torch.utils.data import Dataset, DataLoader

from transformers import CLIPTokenizer
from diffusers import UNet2DConditionModel, AutoencoderKL, DDPMScheduler, StableDiffusionPipeline
from diffusers import DPMSolverMultistepScheduler
from diffusers.optimization import get_scheduler
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Simple dataset
class ImagePromptDataset(Dataset):
    def __init__(self, images_dir: str, labels_path: Optional[str], tokenizer, resolution=512):
        self.images_dir = Path(images_dir)
        self.tokenizer = tokenizer
        self.resolution = resolution
        self.items = []

        if labels_path and Path(labels_path).exists():
            with open(labels_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(",", 1)
                    fname = parts[0].strip()
                    prompt = parts[1].strip() if len(parts) > 1 else ""
                    fp = self.images_dir / fname
                    if fp.exists():
                        self.items.append((str(fp), prompt))
        else:
            # fallback: include all images with empty prompts
            for fp in sorted(self.images_dir.glob("*.*")):
                self.items.append((str(fp), ""))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, prompt = self.items[idx]
        img = Image.open(path).convert("RGB")
        img = img.resize((self.resolution, self.resolution), resample=Image.LANCZOS)
        # convert to tensor normalized [0,1]
        image = (torch.from_numpy((torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))).numpy()).float())
        # but simpler: use PIL->numpy via to bytes would be messy. We'll use transforms in collate.
        return {"image_path": path, "prompt": prompt, "pil": img}

def collate_fn(examples, tokenizer, resolution):
    # preprocess images -> latent space handled by VAE encoder in training loop to reduce memory here.
    pixel_values = []
    input_ids = []
    for ex in examples:
        # basic pixel normalization with PIL -> tensor via torchvision would be nicer, but avoid extra deps.
        arr = torch.ByteTensor(torch.ByteStorage.from_buffer(ex["pil"].tobytes()))
        # not using raw arr; instead convert via torchvision transforms would be simpler in practice.
        # Here we'll pass PIL images downstream; training loop will use VAE to encode.
        pixel_values.append(ex["pil"])
        tokens = tokenizer(ex["prompt"], padding="max_length", truncation=True, max_length=tokenizer.model_max_length, return_tensors="pt")
        input_ids.append(tokens.input_ids[0])
    return {"pixel_pils": pixel_values, "input_ids": torch.stack(input_ids)}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="ml/config.yaml")
    parser.add_argument("--images_dir", type=str, default="ml/data/images")
    parser.add_argument("--labels", type=str, default="ml/data/labels.csv")
    parser.add_argument("--out_dir", type=str, default="./ml/lora_checkpoints")
    parser.add_argument("--hf_token", type=str, default=None)
    return parser.parse_args()

def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config, "r"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # Tokenizer (CLIP)
    tokenizer = CLIPTokenizer.from_pretrained(cfg["base_model"], subfolder="tokenizer", use_fast=True)

    # Load base models (mainly to create target modules for LoRA)
    unet = UNet2DConditionModel.from_pretrained(cfg["base_model"], subfolder="unet").to(device)
    text_encoder = None
    try:
        # many SD checkpoints have text encoder under 'text_encoder' subfolder
        from transformers import CLIPTextModel
        text_encoder = CLIPTextModel.from_pretrained(cfg["base_model"], subfolder="text_encoder").to(device)
    except Exception as e:
        print("Warning: couldn't load text encoder: ", e)

    # Prepare model for kbit training (optional) and configure LoRA
    unet = prepare_model_for_kbit_training(unet)

    lora_config = LoraConfig(
        r=cfg.get("lora_r", 16),
        lora_alpha=cfg.get("lora_alpha", 32),
        target_modules=["attn1", "attn2", "q_proj", "v_proj", "k_proj"],  # rough heuristics; adjust per model
        lora_dropout=cfg.get("lora_dropout", 0.05),
        bias="none",
        task_type="CAUSAL_LM"  # placeholder; diffusers/peft uses this less strictly for LoRA
    )

    unet = get_peft_model(unet, lora_config)

    if text_encoder is not None:
        # Optionally apply LoRA to text encoder too (comment out if problematic)
        te_lora_config = LoraConfig(
            r=cfg.get("lora_r", 8),
            lora_alpha=cfg.get("lora_alpha", 32),
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM"
        )
        text_encoder = get_peft_model(text_encoder, te_lora_config)

    # Prepare VAE and scheduler to compute training losses in latent space
    vae = AutoencoderKL.from_pretrained(cfg["base_model"], subfolder="vae").to(device)
    noise_scheduler = DDPMScheduler.from_pretrained(cfg["base_model"], subfolder="scheduler")

    # Dataset
    ds = ImagePromptDataset(args.images_dir, args.labels, tokenizer, resolution=cfg.get("resolution", 512))
    # simple dataloader which hands PIL images to loop
    loader = DataLoader(ds, batch_size=cfg.get("train_batch_size", 1), shuffle=True, collate_fn=lambda ex: collate_fn(ex, tokenizer, cfg.get("resolution",512)), num_workers=cfg.get("num_workers", 4))

    # Accelerator wraps training
    accelerator = Accelerator(mixed_precision=cfg.get("mixed_precision", "fp16"))
    device = accelerator.device
    print("Accelerator device:", device)

    # Optimizer: only LoRA params (peft exposes trainable params)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, unet.parameters()), lr=cfg.get("learning_rate", 1e-4))

    unet, optimizer, loader = accelerator.prepare(unet, optimizer, loader)

    # Training loop (simpler):
    global_step = 0
    max_train_steps = cfg.get("max_train_steps", 1500)
    gradient_accumulation_steps = cfg.get("gradient_accumulation_steps", 1)

    os.makedirs(args.out_dir, exist_ok=True)

    for epoch in range(9999):
        for batch in loader:
            # Encode images with VAE to latents
            pil_images = batch["pixel_pils"]
            prompts = batch["input_ids"].to(device)

            # Convert PIL images to pixel tensors via simple transform (PIL->tensor)
            pixel_tensors = []
            for pil in pil_images:
                arr = torch.from_numpy((torch.ByteTensor(torch.ByteStorage.from_buffer(pil.tobytes()))).numpy())
                # This cheap conversion is not ideal; in practice use torchvision.transforms to convert to CHW float
                pil_tensor = torch.tensor(np.array(pil)).permute(2,0,1).float() / 255.0
                pixel_tensors.append(pil_tensor)
            pixel_tensors = torch.stack(pixel_tensors).to(device)

            with torch.no_grad():
                latents = vae.encode(pixel_tensors * 2 - 1).latent_dist.sample() * 0.18215

            # Sample noise and timesteps
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Get text embeddings from tokenizer and text encoder (if available) or from tokenizer + CLIP
            if text_encoder is not None:
                text_embeds = text_encoder(prompts)[0]
            else:
                # fallback simple embedding: zeros (not ideal)
                text_embeds = torch.zeros((latents.shape[0], 77, 768), device=device)

            # Forward through UNet
            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states=text_embeds).sample

            # Compute MSE loss with target noise
            loss = torch.nn.functional.mse_loss(model_pred, noise)
            loss = loss / gradient_accumulation_steps
            accelerator.backward(loss)

            if accelerator.sync_gradients:
                torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)

            if (global_step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            global_step += 1

            if global_step % cfg.get("logging_steps", 50) == 0:
                print(f"step {global_step} | loss: {loss.item():.4f}")

            if global_step % cfg.get("save_steps", 500) == 0:
                # Save PEFT/LoRA weights
                save_path = Path(args.out_dir) / f"lora-step-{global_step}"
                unet.save_pretrained(save_path)
                if text_encoder is not None:
                    text_encoder.save_pretrained(save_path / "text_encoder")
                print("Saved checkpoint to", save_path)

            if global_step >= max_train_steps:
                print("Reached max steps, finishing")
                save_path = Path(args.out_dir) / f"lora-final"
                unet.save_pretrained(save_path)
                if text_encoder is not None:
                    text_encoder.save_pretrained(save_path / "text_encoder")
                return

if __name__ == "__main__":
    main()
