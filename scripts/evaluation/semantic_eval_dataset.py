import gc
import io
import os
from dotenv import load_dotenv
load_dotenv()

os.environ["TORCH_HOME"] = os.getenv("TORCH_HOME")
import torch
import torchvision.transforms.functional as TF
from torchmetrics.image.fid import FrechetInceptionDistance
from diffusers.models import UNet2DConditionModel, FluxTransformer2DModel, SD3Transformer2DModel
from open_clip import create_model_and_transforms, get_tokenizer
from accelerate import PartialState
from PIL import Image
from tqdm import tqdm
import argparse
import pickle

from sdib.utils import create_pipeline, load_pipeline, get_precision, get_total_params
from sdib.data.eval_dataset import EvalDataset


def load_model(args, torch_dtype, device):

    if args.save_pth is not None:
        pipe = create_pipeline(
            args.model, device, torch_dtype,
            save_pt=args.save_pth, lambda_threshold=args.lambda_threshold
        )
        print(f"Loaded pruned model from {args.save_pth} with lambda_threshold={args.lambda_threshold}")
    else:
        pipe = load_pipeline(args.model, torch_dtype, disable_progress_bar=True)
        if args.pruned_model_pt is not None:
            class CpuUnpickler(pickle.Unpickler):
                def find_class(self, module, name):
                    if module == "torch.storage" and name == "_load_from_bytes":
                        return lambda b: torch.load(io.BytesIO(b), map_location="cpu", weights_only=False)
                    return super().find_class(module, name)
            with open(args.pruned_model_pt, "rb") as f:
                model = pickle.load(f)
                # model = CpuUnpickler(f).load()
            model.to(get_precision(args.mix_precision))
            if hasattr(pipe, "unet"):
                assert isinstance(model, UNet2DConditionModel)
                del pipe.unet
                pipe.unet = model
            else:
                assert isinstance(model, (FluxTransformer2DModel, SD3Transformer2DModel))
                del pipe.transformer
                pipe.transformer = model
            gc.collect()
            print(f"Loaded pruned model from {args.pruned_model_pt}")
        else:
            print(f"Loaded original model {args.model} without pruning")
    pipe.to(device)
    return pipe


def semantic_eval(args):
    if args.mix_precision == "bf16":
        torch_dtype = torch.bfloat16
    else:
        raise ValueError(f"torch dtype {args.mix_precision} not supported")

    distributed_state = PartialState()
    device = distributed_state.device

    dataset_dir = os.getenv("DATASET_DIR")
    eval_ds = EvalDataset(data_dir=dataset_dir, dataset_name=args.dataset_name, max_size=args.max_size)

    if distributed_state.is_main_process:
        print(f"Loaded {len(eval_ds)} samples from {args.dataset_name}")
        print(f"Running on {distributed_state.num_processes} GPU(s)")

    os.makedirs(args.save_dir, exist_ok=True)
    real_dir = os.path.join(args.save_dir, "real")
    gen_dir = os.path.join(args.save_dir, "generated")
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(gen_dir, exist_ok=True)

    n_expected = len(eval_ds)
    images_exist = (
        len(os.listdir(real_dir)) >= n_expected
        and len(os.listdir(gen_dir)) >= n_expected
    )
    # images_exist = True
    if images_exist:
        if distributed_state.is_main_process:
            print(f"Found {n_expected} existing images in {args.save_dir}, skipping generation.")
    else:
        pipe = load_model(args, torch_dtype, device)
        if distributed_state.is_main_process:
            if hasattr(pipe, "unet"):
                total_params = sum(p.numel() for p in pipe.unet.parameters())
            else:
                total_params = sum(p.numel() for p in pipe.transformer.parameters())
            # total_params = get_total_params(pipe.unet if hasattr(pipe, "unet") else pipe.transformer)
            print(f"Model has {total_params/1e9:.2f}B parameters")
            print("Generating synthetic images with the model...")
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
        fid = FrechetInceptionDistance(feature=2048)
        original_list, gen_list = [], []
        for image_name in tqdm(os.listdir(real_dir)):
        # for idx in tqdm(range(len(eval_ds))):
            real_pil = Image.open(os.path.join(real_dir, image_name)).convert("RGB")
            gen_pil = Image.open(os.path.join(gen_dir, image_name)).convert("RGB")

            real_t = (TF.to_tensor(real_pil).unsqueeze(0) * 255).to(torch.uint8)
            gen_t = (TF.to_tensor(gen_pil).unsqueeze(0) * 255).to(torch.uint8)
            original_list.append(real_t)
            gen_list.append(gen_t)
            if len(original_list) % 500 == 0 and len(original_list) > 0:
                fid.update(torch.cat(original_list), real=True)
                fid.update(torch.cat(gen_list), real=False)
                original_list, gen_list = [], []
        if original_list: # in case there are remaining images not divisible by 500
            fid.update(torch.cat(original_list), real=True)
            fid.update(torch.cat(gen_list), real=False)
        fid_score = fid.compute().item()
        print(f"FID score: {fid_score:.4f}")

        log_txt = os.path.join(args.save_dir, "semantic_eval.txt")
        with open(log_txt, "a") as f:
            model_label = args.pruned_model_pt or args.save_dir or "original"
            f.write(f"[{args.dataset_name}] model={model_label}  FID={fid_score:.4f}\n")


def clip_eval(args):
    distributed_state = PartialState()
    device = distributed_state.device

    dataset_dir = os.getenv("DATASET_DIR")
    eval_ds = EvalDataset(data_dir=dataset_dir, dataset_name=args.dataset_name, max_size=args.max_size)

    gen_dir = os.path.join(args.save_dir, "generated")
    if not os.path.exists(gen_dir) or len(os.listdir(gen_dir)) < len(eval_ds):
        raise RuntimeError(f"Generated images not found in {gen_dir}. Run semantic_eval first.")

    if distributed_state.is_main_process:
        print("Computing CLIP score (mean cosine similarity between generated images and captions)...")
        clip_model, _, preprocess = create_model_and_transforms(
            model_name=args.clip_backbone, pretrained=args.clip_pretrained,
            cache_dir=os.getenv("CLIP_CACHE_DIR"),
        )
        clip_model = clip_model.to(device).eval()
        tokenizer = get_tokenizer(args.clip_backbone)

        clip_scores = []
        with torch.no_grad():
            for idx in tqdm(range(len(eval_ds))):
                caption = eval_ds[idx]["text"]
                gen_pil = Image.open(os.path.join(gen_dir, f"{idx:05d}.png")).convert("RGB")

                image_input = preprocess(gen_pil).unsqueeze(0).to(device)
                text_input = tokenizer([caption]).to(device)

                image_features = clip_model.encode_image(image_input)
                text_features = clip_model.encode_text(text_input)

                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                cosine_sim = (image_features * text_features).sum().item()
                clip_scores.append(cosine_sim)

        clip_score = 100.0 * sum(clip_scores) / len(clip_scores)
        print(f"CLIP score (x100): {clip_score:.4f}")

        log_txt = os.path.join(args.save_dir, "semantic_eval.txt")
        with open(log_txt, "a") as f:
            model_label = args.pruned_model_pt or args.save_dir or "original"
            f.write(f"[{args.dataset_name}] model={model_label}  CLIP score (x100)={clip_score:.4f}\n")


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
    parser.add_argument("--clip_backbone", type=str, default="ViT-B-16",help="clip model type, available ViT-B-16, ViT-L-14")
    parser.add_argument("--clip_pretrained", type=str, default="laion400m_e32")
    parser.add_argument("--task", type=str, default="fid", help="fid | clip | all")
    args = parser.parse_args()
    if args.task == "fid":
        semantic_eval(args)
    elif args.task == "clip":
        clip_eval(args)
    elif args.task == "all":
        semantic_eval(args)
        clip_eval(args)
    else:
        raise ValueError(f"Unknown task: {args.task}")
