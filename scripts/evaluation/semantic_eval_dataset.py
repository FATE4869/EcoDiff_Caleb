import os

os.environ["TORCH_HOME"] = "/gpfs/projects/shlneuroai/caleb/torch_cache/"
import torch
import torchvision.transforms.functional as TF
from torchmetrics.image.fid import FrechetInceptionDistance
from diffusers.models import UNet2DConditionModel, FluxTransformer2DModel, SD3Transformer2DModel
from accelerate import PartialState
from PIL import Image
from tqdm import tqdm
import argparse
import pickle

from sdib.utils import create_pipeline, load_pipeline, get_precision
from sdib.data.eval_dataset import EvalDataset


def load_model(args, torch_dtype, device):
    if args.save_pth is not None:
        pipe = create_pipeline(
            args.model, device, torch_dtype,
            save_pt=args.save_pth, lambda_threshold=args.lambda_threshold
        )
    else:
        pipe = load_pipeline(args.model, torch_dtype, disable_progress_bar=True)
        if args.pruned_model_pt is not None:
            with open(args.pruned_model_pt, "rb") as f:
                model = pickle.load(f)
            model.to(get_precision(args.mix_precision))
            if hasattr(pipe, "unet"):
                assert isinstance(model, UNet2DConditionModel)
                pipe.unet = model
            else:
                assert isinstance(model, (FluxTransformer2DModel, SD3Transformer2DModel))
                pipe.transformer = model
    pipe.to(device)
    return pipe


def semantic_eval(args):
    if args.mix_precision == "bf16":
        torch_dtype = torch.bfloat16
    else:
        raise ValueError(f"torch dtype {args.mix_precision} not supported")

    distributed_state = PartialState()
    device = distributed_state.device

    dataset_dir = "/mmfs1/gscratch/shlneuroai/zheng94/dataset/"
    eval_ds = EvalDataset(data_dir=dataset_dir, dataset_name=args.dataset_name, max_size=args.max_size)

    if distributed_state.is_main_process:
        print(f"Loaded {len(eval_ds)} samples from {args.dataset_name}")
        print(f"Running on {distributed_state.num_processes} GPU(s)")

    os.makedirs(args.save_dir, exist_ok=True)
    real_dir = os.path.join(args.save_dir, "real")
    gen_dir = os.path.join(args.save_dir, "generated")
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(gen_dir, exist_ok=True)

    # Stagger model loading to avoid CPU RAM OOM when multiple processes load simultaneously
    with distributed_state.main_process_first():
        pipe = load_model(args, torch_dtype, device)
    generator = torch.Generator(device=device).manual_seed(args.seed)

    all_indices = list(range(len(eval_ds)))
    with distributed_state.split_between_processes(all_indices) as local_indices:
        for idx in tqdm(local_indices, disable=not distributed_state.is_local_main_process):
            sample = eval_ds[idx]
            caption = sample["text"]
            real_pil = sample["image"].convert("RGB").resize((args.image_size, args.image_size))

            with torch.no_grad():
                gen_pil = pipe(
                    caption,
                    num_inference_steps=args.num_intervention_steps,
                    generator=generator,
                ).images[0].resize((args.image_size, args.image_size))

            real_pil.save(os.path.join(real_dir, f"{idx:05d}.png"))
            gen_pil.save(os.path.join(gen_dir, f"{idx:05d}.png"))

    # wait for all processes to finish generating
    distributed_state.wait_for_everyone()

    # compute FID on main process only
    if distributed_state.is_main_process:
        print("Computing FID...")
        fid = FrechetInceptionDistance(feature=64)

        for idx in tqdm(range(len(eval_ds))):
            real_pil = Image.open(os.path.join(real_dir, f"{idx:05d}.png")).convert("RGB")
            gen_pil = Image.open(os.path.join(gen_dir, f"{idx:05d}.png")).convert("RGB")

            real_t = (TF.to_tensor(real_pil).unsqueeze(0) * 255).to(torch.uint8)
            gen_t = (TF.to_tensor(gen_pil).unsqueeze(0) * 255).to(torch.uint8)

            fid.update(real_t, real=True)
            fid.update(gen_t, real=False)

        fid_score = fid.compute().item()
        print(f"FID score: {fid_score:.4f}")

        if args.save_pth is not None:
            log_txt = os.path.join(os.path.dirname(args.save_pth), "semantic_eval.txt")
        else:
            os.makedirs("results", exist_ok=True)
            log_txt = "results/semantic_eval.txt"

        with open(log_txt, "a") as f:
            model_label = args.pruned_model_pt or args.save_pth or "original"
            f.write(f"[{args.dataset_name}] model={model_label}  FID={fid_score:.4f}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Image semantic evaluation with EvalDataset")
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--save_dir", "-s", type=str, default="./results/generated_images")
    parser.add_argument("--mix_precision", type=str, default="bf16")
    parser.add_argument("--num_intervention_steps", type=int, default=50)
    parser.add_argument("--model", type=str, default="sdxl", help="sdxl | sd2 | sd1 ...")
    parser.add_argument("--dataset_name", type=str, default="coco", help="coco | flickr | laion")
    parser.add_argument("--max_size", type=int, default=5000, help="number of samples to evaluate")
    parser.add_argument("--image_size", type=int, default=512, help="resize images to this size for FID")
    parser.add_argument("--save_pth", "-sp", type=str, default=None, help="path to hooker .pth for pruned model")
    parser.add_argument("--pruned_model_pt", type=str, default=None, help="path to pruned model .pkl")
    parser.add_argument("--lambda_threshold", "-lt", type=float, default=0.01)

    args = parser.parse_args()
    semantic_eval(args)
