"""
generate_from_lora.py
Loads the base stable-diffusion pipeline and applies LoRA weights (from peft lora checkpoint)
Generates images from a prompt and saves to disk.
"""

import argparse
from pathlib import Path
import torch
from diffusers import StableDiffusionPipeline
from peft import PeftModel, PeftConfig
from transformers import CLIPTokenizer, CLIPTextModel

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--lora_checkpoint", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="./ml/generated")
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pipe = StableDiffusionPipeline.from_pretrained(args.base_model, torch_dtype=torch.float16 if device=="cuda" else torch.float32)
    pipe = pipe.to(device)

    # Load LoRA weights: peft stores a small state dict. We can merge them into the unet or load via peft.PeftModel if that fits.
    # For simplicity, we will try to load from path into UNet if present:
    lora_path = Path(args.lora_checkpoint)
    if not lora_path.exists():
        raise FileNotFoundError("LoRA checkpoint not found: " + str(lora_path))

    # peft has utilities: PeftModel.from_pretrained(...) -> but with diffusers/UNet shapes this can need custom code
    # We'll try a helper merge (this may need adaptation depending on how you saved the LoRA)
    try:
        from peft import PeftModel
        pipe.unet = PeftModel.from_pretrained(pipe.unet, str(lora_path)).to(device)
        print("Applied LoRA to UNet via PeftModel.")
    except Exception as e:
        print("Couldn't apply LoRA via PeftModel out-of-the-box:", e)
        print("Proceeding without LoRA merge — ensure you saved compatible PEFT weights.")

    if args.seed is not None:
        generator = torch.Generator(device=device).manual_seed(args.seed)
    else:
        generator = None

    images = pipe(prompt=args.prompt, height=args.height, width=args.width, num_inference_steps=args.num_inference_steps, generator=generator).images

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(images):
        out_path = out_dir / f"gen_{i}.png"
        img.save(out_path)
        print("Saved", out_path)

if __name__ == "__main__":
    main()
