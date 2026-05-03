import os
import glob
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

HP_DIR = "results_sdxl/sdxl_hp"
OUTPUT_DIR = "results_sdxl/sdxl_hp/comparison"
PRUNED_SUBDIR = "pruned/pruned_20/pruned"

LABEL_HEIGHT = 80 
FONT_SIZE = 32


def short_label(exp_name):
    """Extract key HP params from experiment folder name for a compact label."""
    parts = exp_name.split("_")
    beta = next((parts[i+1] for i, p in enumerate(parts) if p == "beta"), "?")
    masking = next((parts[i+1:] for i, p in enumerate(parts) if p == "masking"), "?")
    masking_str = "_".join(masking) if masking != "?" else "?"
    loss_idx = next((i for i, p in enumerate(parts) if p == "loss"), -1)
    loss = f"recon={parts[loss_idx+1]},reg={parts[loss_idx+2]}" if loss_idx >= 0 else "?"
    lr_idx = next((i for i, p in enumerate(parts) if p == "lr"), -1)
    lr = "_".join(parts[lr_idx+1:lr_idx+4]) if lr_idx >= 0 else "?"
    return f"beta={beta}. lr={lr}. loss={loss}.\nmasking={masking_str}"


def get_font(size):
    return ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", size)


def make_grid(images_with_labels, output_path):
    w, h = images_with_labels[0][0].size
    font = get_font(FONT_SIZE)
    n = len(images_with_labels)

    canvas = Image.new("RGB", (w * n, h + LABEL_HEIGHT), color=(240, 240, 240))
    draw = ImageDraw.Draw(canvas)

    for i, (img, label) in enumerate(images_with_labels):
        canvas.paste(img, (i * w, LABEL_HEIGHT))
        draw.multiline_text((i * w + 4, 4), label, fill=(0, 0, 0), font=font, spacing=2)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    canvas.save(output_path)
    print(f"Saved: {output_path}")


def main():
    exp_dirs = sorted(Path(HP_DIR).iterdir())

    # Collect all images grouped by prompt
    prompt_images = {}  # prompt_stem -> [(img, label), ...]

    for exp_dir in exp_dirs:
        pruned_dir = exp_dir / PRUNED_SUBDIR
        if not pruned_dir.exists():
            print(f"Missing: {pruned_dir}")
            continue

        label = short_label(exp_dir.name)
        for img_path in sorted(pruned_dir.glob("*.png")):
            # Use prompt text (strip timestamp suffix) as key
            stem = img_path.stem.rsplit("_seed_", 1)[0]
            if stem not in prompt_images:
                prompt_images[stem] = []
            prompt_images[stem].append((Image.open(img_path).convert("RGB"), label))

    for prompt_stem, images_with_labels in prompt_images.items():
        if len(images_with_labels) != len(exp_dirs):
            print(f"Warning: {prompt_stem} has {len(images_with_labels)}/{len(exp_dirs)} images")
        safe_name = prompt_stem.replace(" ", "_").replace("/", "_")[:60]
        make_grid(images_with_labels, f"{OUTPUT_DIR}/{safe_name}.png")


if __name__ == "__main__":
    main()
