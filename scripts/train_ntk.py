import logging
import os
from dotenv import load_dotenv
from huggingface_hub import login
load_dotenv()
login(token=os.environ["HF_TOKEN"])

import omegaconf
import torch
import torch.nn.functional as F
import tqdm
import yaml
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from diffusers.optimization import get_scheduler
from torch.utils.data import DataLoader

import argparse
from contextlib import contextmanager
import wandb
from sdib.data import DiTDataset, PromptImageDataset
from sdib.hooks import CrossAttentionExtractionHook, FeedForwardHooker, NormHooker
from sdib.utils import (
    calculate_mask_sparsity,
    calculate_reg_loss,
    get_precision,
    load_config,
    load_pipeline,
    save_image_binarize_seed,
    save_image_seed,
)


# ---------------------------------------------------------------------------
# NTK helpers
# ---------------------------------------------------------------------------

@contextmanager
def ntk_masking_context(hookers, use_sigmoid_lambda: bool = False):
    """Switch hookers to 'binary' masking with explicit leaf mask tensors.

    Differentiating w.r.t. these leaf tensors avoids the hard-concrete
    saturation problem (hard_concrete(λ=5) clamps to 1 → zero gradient).

    Original model  (use_sigmoid_lambda=False):
        mask = ones  [requires_grad=True]
        → J = ∂f/∂mask evaluated at full activations; always non-zero.

    Masked model    (use_sigmoid_lambda=True):
        mask = sigmoid(λ)  [leaf via autograd chain through λ]
        → J = ∂f_masked/∂mask; gradient flows back to λ through sigmoid.

    Yields the flat list of mask leaf tensors (ref_params) to use as the
    `lambdas` argument of compute_ntk_matrix.
    """
    saved = [(list(h.lambs), h.masking) for h in hookers]
    ref_params = []
    for hooker in hookers:
        if use_sigmoid_lambda:
            new_lambs = [torch.sigmoid(l) for l in hooker.lambs]
        else:
            new_lambs = [torch.ones_like(l).requires_grad_(True) for l in hooker.lambs]
        hooker.lambs = new_lambs
        hooker.masking = "binary"
        ref_params.extend(new_lambs)
    try:
        yield ref_params
    finally:
        for hooker, (orig_lambs, orig_masking) in zip(hookers, saved):
            hooker.lambs = orig_lambs
            hooker.masking = orig_masking


def compute_ntk_matrix(
    outputs: torch.Tensor,
    mask_params: list,
    proj_dim: int = 16,
    proj: torch.Tensor = None,
    create_graph: bool = False,
) -> torch.Tensor:
    """Compute the empirical NTK matrix w.r.t. mask_params.

    K[i,j] = <∂f(x_i)/∂mask, ∂f(x_j)/∂mask>

    mask_params are per-head/per-neuron scale tensors set up by
    ntk_masking_context.  For the original model they are all-ones leaves;
    for the masked model they are sigmoid(λ) so gradient flows to λ.

    create_graph=True is required when computing K_masked so that
    ∂L_ntk/∂λ is non-zero (K_masked depends on λ through the nonlinear
    forward pass and the sigmoid mask values).
    """
    B = outputs.shape[0]
    out_flat = outputs.reshape(B, -1).float()
    D = out_flat.shape[1]

    if proj is None:
        gen = torch.Generator(device=outputs.device).manual_seed(0)
        proj = torch.randn(D, proj_dim, generator=gen,
                           device=outputs.device, dtype=torch.float32)
        proj = F.normalize(proj, dim=0)
    else:
        proj = proj.to(device=outputs.device, dtype=torch.float32)

    projected = out_flat @ proj  # [B, proj_dim], differentiable w.r.t. mask_params

    J_list = []
    for i in range(B):
        row = []
        for k in range(proj_dim):
            is_last = (i == B - 1 and k == proj_dim - 1)
            g = torch.autograd.grad(
                projected[i, k], mask_params,
                retain_graph=(not is_last) or create_graph,
                create_graph=create_graph,
                allow_unused=True,
            )
            row.append(torch.cat([
                gg.flatten().float() if gg is not None
                else torch.zeros(p.numel(), dtype=torch.float32, device=p.device)
                for gg, p in zip(g, mask_params)
            ]))
        J_list.append(torch.stack(row))  # [proj_dim, P]

    J = torch.stack(J_list)              # [B, proj_dim, P]
    J_flat = J.reshape(B, -1)            # [B, proj_dim * P]

    norms = J_flat.norm(dim=1, keepdim=True).clamp(min=1e-8)
    J_flat = J_flat / norms

    return J_flat @ J_flat.T             # [B, B]


def precompute_original_jacobians(
    pipe,
    dataset,
    hookers,
    cfg,
    device,
    seed,
    proj_dim: int = 16,
    save_dir: str = "jacobians",
):
    """
    Precompute per-step Jacobian vectors for the original (unmasked) model
    over all samples in the dataset.  Results are saved to disk as a list of dicts:

        [{'J': tensor[proj_dim, P], 'proj': tensor[D, proj_dim],
          'z_in': tensor}, ...]   # len = n_steps

    P = total number of mask units (heads + neurons across all masked layers).
    J = ∂f(x)/∂mask evaluated at mask=1 (full-model, no pruning).  Using
    all-ones leaves (not hard_concrete(λ)) avoids the saturation problem:
    hard_concrete(init_lambda=5) clamps to 1 → zero gradient.  At mask=1 the
    gradient ∂f/∂mask_h is the sensitivity of the output to each head/neuron,
    which is always non-zero for active layers.

    hookers: list of hooker objects (cross_attn, ff, norm).
    Projection matrices are seeded by step_idx for reproducibility.
    """
    os.makedirs(save_dir, exist_ok=True)

    if hasattr(pipe, 'transformer') and hasattr(pipe.transformer, 'enable_gradient_checkpointing'):
        pipe.transformer.enable_gradient_checkpointing()

    # Use ntk_masking_context: switches masking to "binary" and sets lambs to
    # all-ones leaf tensors.  autograd.grad differentiates w.r.t. these leaves
    # (not λ), so the Jacobian is ∂f/∂mask at mask=1 — always non-zero.
    # The context is entered once; the same all-ones leaves are reused across
    # all samples (each forward pass builds an independent graph via autograd.grad).
    with ntk_masking_context(hookers, use_sigmoid_lambda=False) as ref_params:
        try:
            for sample_idx in tqdm.tqdm(range(len(dataset)), desc="Precomputing Jacobians"):
                save_path = os.path.join(save_dir, f"jacobian_{sample_idx}.pt")
                if os.path.exists(save_path):
                    continue
                data = dataset[sample_idx]
                prompt = [data["prompt"]] if isinstance(data["prompt"], str) else data["prompt"]

                g_cpu = torch.Generator(device.type).manual_seed(seed)
                with torch.no_grad():
                    prep = pipe.inference_preparation_phase(
                        prompt,
                        generator=g_cpu,
                        num_inference_steps=cfg.trainer.num_intervention_steps,
                        output_type="latent",
                    )

                steps_data = []
                for step_idx, t in tqdm.tqdm(
                    enumerate(prep.timesteps),
                    total=len(prep.timesteps),
                    desc=f"  sample {sample_idx} steps",
                    leave=False,
                ):
                    with torch.set_grad_enabled(True):
                        out_latents = pipe.inference_with_grad_denoising_step(step_idx, t, prep)

                    D = out_latents.reshape(1, -1).shape[1]
                    gen = torch.Generator().manual_seed(step_idx)
                    proj = F.normalize(torch.randn(D, proj_dim, generator=gen), dim=0).to(device)

                    out_flat = out_latents.reshape(1, -1).float()
                    projected = out_flat @ proj  # [1, proj_dim]

                    J_step = torch.zeros(proj_dim, sum(p.numel() for p in ref_params),
                                         dtype=torch.float32)
                    for k in tqdm.tqdm(range(proj_dim), desc="   proj_dim", leave=False):
                        is_last = (k == proj_dim - 1)
                        g = torch.autograd.grad(
                            projected[0, k], ref_params,
                            retain_graph=not is_last, create_graph=False,
                            allow_unused=True,
                        )
                        J_step[k] = torch.cat([
                            gg.flatten().float().cpu() if gg is not None
                            else torch.zeros(p.numel(), dtype=torch.float32)
                            for gg, p in zip(g, ref_params)
                        ])
                    steps_data.append({
                        'J': J_step,
                        'proj': proj.cpu(),
                        'z_in': prep.latents.detach().cpu(),
                    })
                    prep.latents = out_latents.detach()
                    torch.cuda.empty_cache()
                torch.save(steps_data, save_path)
        finally:
            pass  # ntk_masking_context __exit__ restores hookers

    if hasattr(pipe, 'transformer') and hasattr(pipe.transformer, 'disable_gradient_checkpointing'):
        pipe.transformer.disable_gradient_checkpointing()


def compute_k_orig_from_jacobians(jacobian_dir, indices, step_idx, device, proj_indices=None):
    """
    Load precomputed Jacobians for a batch of sample indices at a given
    denoising step and return the row-normalised [B, B] NTK matrix.

    proj_indices: optional 1-D LongTensor of row indices into the stored
                  [precompute_proj_dim, P] Jacobian.  Use this to subsample
                  to args.ntk_proj_dim directions at training time, passing
                  the same indices to compute_ntk_matrix so that both
                  K_orig and K_masked use identical projection directions.
    """
    Js = [torch.load(os.path.join(jacobian_dir, f"jacobian_{i}.pt"),
                     weights_only=False)[step_idx]['J']
          for i in indices]
    J_batch = torch.stack(Js).to(device)        # [B, precompute_proj_dim, P]
    if proj_indices is not None:
        J_batch = J_batch[:, proj_indices, :]   # [B, ntk_proj_dim, P]
    J_flat = J_batch.reshape(len(indices), -1).float()
    norms = J_flat.norm(dim=1, keepdim=True).clamp(min=1e-8)
    J_flat = J_flat / norms
    return J_flat @ J_flat.T                    # [B, B]



# ---------------------------------------------------------------------------
# Combined pruning + NTK loss
# ---------------------------------------------------------------------------

def pruning_loss(
    reconstruction_loss_func,
    image_pt,
    image,
    cross_attn_hooker,
    ff_hooker,
    device,
    torch_dtype,
    cfg,
    logger=None,
    norm_hooker=None,
    ntk_lambda: float = 0.0,
    ntk_proj_dim: int = 16,
    K_orig: torch.Tensor = None,
):
    """
    Combined loss = reconstruction + beta * regularisation + ntk_lambda * NTK.

    NTK loss: MSE between the lambda-NTK of the masked model (K_masked) and
    the lambda-NTK of the original model evaluated at lambda=1 (K_orig).

        K[i,j] = <J_lambda f(x_i), J_lambda f(x_j)>

    K_orig must be precomputed and passed in (detached); K_masked is computed
    here from `image["images"]` which is differentiable w.r.t. the current
    lambda parameters.

    Args:
        K_orig: [B, B] NTK matrix of the original model (detached). Required
                when ntk_lambda > 0.  If None and ntk_lambda > 0, the NTK
                term is skipped with a warning.
    """
    loss_reconstruct = reconstruction_loss_func(image_pt, image["images"])
    attn_loss_reg = torch.tensor(0.0, device=device, dtype=torch_dtype)
    if cfg.loss.use_attn_reg:
        attn_loss_reg = calculate_reg_loss(
            attn_loss_reg,
            cross_attn_hooker.lambs,
            cfg.loss.reg,
            mean=cfg.loss.mean,
            reg=cfg.loss.lambda_reg,
            reg_alpha=cfg.loss.reg_alpha,
            reg_beta=cfg.loss.reg_beta,
        )
    ff_loss_reg = torch.tensor(0.0, device=device, dtype=torch_dtype)
    if cfg.loss.use_ffn_reg:
        ff_loss_reg = calculate_reg_loss(
            ff_loss_reg,
            ff_hooker.lambs,
            cfg.loss.reg,
            mean=cfg.loss.mean,
            reg=cfg.loss.lambda_reg,
            reg_alpha=cfg.loss.reg_alpha,
            reg_beta=cfg.loss.reg_beta,
        )
    loss_reg = attn_loss_reg + ff_loss_reg
    norm_loss_reg = torch.tensor(0.0, device=device, dtype=torch_dtype)
    if norm_hooker:
        norm_loss_reg = calculate_reg_loss(
            ff_loss_reg,
            norm_hooker.lambs,
            cfg.loss.reg,
            mean=cfg.loss.mean,
            reg=cfg.loss.lambda_reg,
            reg_alpha=cfg.loss.reg_alpha,
            reg_beta=cfg.loss.reg_beta,
        )
        loss_reg = loss_reg + norm_loss_reg

    # NTK loss is computed per-step in the training loop; pruning_loss only
    # handles reconstruction + regularisation.  loss_ntk is passed in as a
    # pre-computed value (or 0) and added to the total below.
    loss_ntk = torch.tensor(0.0, device=device, dtype=torch_dtype)

    loss = (
        loss_reconstruct
        + cfg.trainer.beta * loss_reg
        + ntk_lambda * loss_ntk
    )

    if logger:
        log_output = (
            f"ff_loss_reg: {ff_loss_reg.item()}"
            f" attn_loss_reg: {attn_loss_reg.item()}"
            f" loss_ntk: {loss_ntk.item()}"
        )
        if norm_hooker:
            log_output += f" norm_loss_reg: {norm_loss_reg.item()}"
        logger.info(log_output)

    return loss, loss_reconstruct, loss_reg, loss_ntk


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    cfg = load_config(args.cfg)
    args.ntk_proj_dim = cfg.data.ntk_proj_dim
    args.precompute_proj_dim = cfg.data.precompute_proj_dim
    device = torch.device(cfg.trainer.device)
    with open(args.validation_prompts_path, "r") as f:
        validation_prompts = yaml.safe_load(f)
    if cfg.trainer.model == "dit":
        validation_prompts = [1]
    else:
        validation_prompts = validation_prompts

    if cfg.logger.type == "wandb":
        config = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        config["ntk_lambda"] = args.ntk_lambda
        config["ntk_proj_dim"] = args.ntk_proj_dim
        import time
        timestr = time.strftime("%Y%m%d-%H%M%S")
        name = f"{cfg.logger.notes}_ntk_{timestr}"
        run = wandb.init(
            project=cfg.logger.project,
            notes=cfg.logger.notes,
            tags=cfg.logger.tags,
            config=config,
            name=name,
        )

    logger = logging.getLogger(__name__)
    filename = f"{cfg.logger.output_dir}/{cfg.logger.project}/{cfg.logger.notes}/report.log"
    os.makedirs(f"{cfg.logger.output_dir}/{cfg.logger.project}/{cfg.logger.notes}", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(filename), logging.StreamHandler()],
    )
    logger.info(f"Validation prompts: {validation_prompts}")
    logger.info(f"NTK lambda: {args.ntk_lambda}  proj_dim: {args.ntk_proj_dim}")

    accelerator_project_config = ProjectConfiguration(
        project_dir=cfg.logger.output_dir, logging_dir=cfg.logger.output_dir
    )
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator_log_with = "all" if cfg.logger.type == "csv" else cfg.logger.type
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.accelerator.gradient_accumulation_steps,
        log_with=accelerator_log_with,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    if args.islaunch:
        device = accelerator.device

    seed = cfg.trainer.seed
    set_seed(seed)
    g_cpu = torch.Generator(device.type).manual_seed(seed)

    torch_dtype = get_precision(cfg.trainer.precision)

    pipe = load_pipeline(cfg.trainer.model, torch_dtype, cfg.trainer.disable_progress_bar)
    pipe.to(device)
    pipe.vae.requires_grad_(False)
    if cfg.trainer.model in ["sd3", "dit", "flux", "flux_dev"]:
        pipe.transformer.requires_grad_(False)
    else:
        pipe.unet.requires_grad_(False)

    if cfg.trainer.model == "dit":
        train_dataset = DiTDataset(
            pipe=pipe,
            save_dir=cfg.data.save_dir,
            device=device,
            size=cfg.data.size,
            num_inference_steps=cfg.trainer.num_intervention_steps,
            seed=seed,
        )
    else:
        train_dataset = PromptImageDataset(
            metadata=cfg.data.metadata,
            pipe=pipe,
            num_inference_steps=cfg.trainer.num_intervention_steps,
            save_dir=cfg.data.save_dir,
            device=device,
            seed=seed,
            size=cfg.data.size,
        )
        try:
            batch_size = cfg.data.batch_size
            logger.info(f"Batch size: {batch_size}")
        except Exception as e:
            logger.info(f"Error: {e}, setting batch size to 1")
            batch_size = 1
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    if cfg.logger.type == "wandb":
        img = save_image_seed(pipe, validation_prompts, cfg.trainer.num_intervention_steps, device, seed, save_dir=None)
        wandb.log({"image": [wandb.Image(i) for i in img]})
    else:
        path = os.path.join(args.save_dir, cfg.logger.project, cfg.logger.notes, "images")
        val_path = os.path.join(path, "validation", "initial_image")
        save_image_seed(pipe, validation_prompts, cfg.trainer.num_intervention_steps, device, seed, save_dir=val_path)
        train_path = os.path.join(path, "train", "initial_image")
        save_image_seed(
            pipe, train_dataset[0]["prompt"], cfg.trainer.num_intervention_steps, device, seed, save_dir=train_path
        )

    if cfg.loss.reconstruct == 1:
        reconstruction_loss_func = torch.nn.L1Loss(reduction="mean")
    elif cfg.loss.reconstruct == 2:
        reconstruction_loss_func = torch.nn.MSELoss()
    else:
        raise ValueError(f"Reconstruction loss {cfg.loss.reconstruct} not supported")

    cross_attn_hooker = CrossAttentionExtractionHook(
        pipe,
        regex=cfg.trainer.regex,
        dtype=torch_dtype,
        head_num_filter=cfg.trainer.head_num_filter,
        masking=cfg.trainer.masking,
        dst=cfg.logger.save_lambda_path.attn,
        epsilon=cfg.trainer.epsilon,
        model_name=cfg.trainer.model,
        attn_name=cfg.trainer.attn_name,
        use_log=cfg.trainer.use_log,
        eps=cfg.trainer.masking_eps,
    )
    cross_attn_hooker.add_hooks(init_value=cfg.trainer.init_lambda)
    lamda_block_names = cross_attn_hooker.get_lambda_block_names

    ff_hooker = FeedForwardHooker(
        pipe,
        regex=cfg.trainer.regex,
        dtype=torch_dtype,
        masking=cfg.trainer.masking,
        dst=cfg.logger.save_lambda_path.ffn,
        epsilon=cfg.trainer.epsilon,
        eps=cfg.trainer.masking_eps,
        use_log=cfg.trainer.use_log,
    )
    ff_hooker.add_hooks(init_value=cfg.trainer.init_lambda)
    ff_lambda_block_names = ff_hooker.get_lambda_block_names

    if cfg.trainer.n_lr != 0:
        norm_hooker = NormHooker(
            pipe,
            regex=cfg.trainer.regex,
            dtype=torch_dtype,
            masking=cfg.trainer.masking,
            dst=cfg.logger.save_lambda_path.norm,
            epsilon=cfg.trainer.epsilon,
            eps=cfg.trainer.masking_eps,
            use_log=cfg.trainer.use_log,
        )
        norm_hooker.add_hooks(init_value=cfg.trainer.init_lambda)
        norm_lambda_block_names = norm_hooker.get_lambda_block_names
    else:
        norm_hooker = None
        norm_lambda_block_names = []

    logger.info(f"Initializing lambda to be {cfg.trainer.init_lambda}")
    _ = pipe(validation_prompts, generator=g_cpu, num_inference_steps=1)
    if args.load_lambda:
        cross_attn_hooker.binary = False
        ff_hooker.binary = False
        cross_attn_hooker.load(device=device)
        ff_hooker.load(device=device)
        for i, lambs in enumerate(cross_attn_hooker.lambs):
            lambs = lambs.detach().clone().requires_grad_(True)
            lambs.to(device)
            cross_attn_hooker.lambs[i] = lambs
        for i, lambs in enumerate(ff_hooker.lambs):
            lambs = lambs.detach().clone().requires_grad_(True)
            lambs.to(device)
            ff_hooker.lambs[i] = lambs

    params = [
        {"params": cross_attn_hooker.lambs, "lr": cfg.trainer.attn_lr},
        {"params": ff_hooker.lambs, "lr": cfg.trainer.ff_lr},
    ]
    if cfg.trainer.n_lr != 0:
        params += ({"params": norm_hooker.lambs, "lr": cfg.trainer.n_lr},)

    optimizer = torch.optim.AdamW(params, lr=cfg.trainer.lr)
    lr_scheduler = get_scheduler(
        cfg.lr_scheduler.type,
        optimizer=optimizer,
        num_warmup_steps=cfg.lr_scheduler.warmup_steps,
        num_training_steps=cfg.trainer.epochs * len(dataloader) // cfg.trainer.accumulate_grad_batches,
        num_cycles=cfg.lr_scheduler.num_cycles,
        power=cfg.lr_scheduler.power,
    )

    # ---- Precompute original model Jacobians (once, before training) ----
    jacobian_dir = os.path.join(cfg.data.save_dir, "jacobians")
    if args.ntk_lambda > 0.0:
        all_cached = all(
            os.path.exists(os.path.join(jacobian_dir, f"jacobian_{i}.pt"))
            for i in range(len(train_dataset))
        )
        if not all_cached:
            logger.info(f"Precomputing original model Jacobians (lambda={cfg.trainer.init_lambda})...")
            hookers_list = [cross_attn_hooker, ff_hooker] + ([norm_hooker] if norm_hooker else [])
            precompute_original_jacobians(
                pipe, train_dataset, hookers_list,
                cfg, device, seed, args.precompute_proj_dim, jacobian_dir,
            )
            torch.cuda.empty_cache()
            logger.info(f"Jacobians saved to {jacobian_dir}")
        else:
            logger.info(f"Using cached Jacobians from {jacobian_dir}")

    # Enable gradient checkpointing on the transformer to cut per-step activation
    # memory from ~15-20 GB down to ~3-5 GB. Must be set before accelerator.prepare
    # so it is not lost after DDP wrapping.
    transformer = getattr(pipe, 'transformer', getattr(pipe, 'unet', None))
    if transformer is not None and hasattr(transformer, 'enable_gradient_checkpointing'):
        transformer.enable_gradient_checkpointing()
        logger.info("Enabled transformer gradient checkpointing for training.")
    # Sliced VAE decoding avoids a large temporary allocation during image reconstruction.
    if hasattr(pipe, 'enable_vae_slicing'):
        pipe.enable_vae_slicing()
        logger.info("Enabled VAE slicing.")

    pipe, optimizer, lr_scheduler = accelerator.prepare(pipe, optimizer, lr_scheduler)
    logger.info("Start Training ...")

    torch.cuda.empty_cache()
    optimizer.zero_grad()

    _norm_hooker = norm_hooker  # shorthand used below

    total_step = cfg.trainer.epochs * len(dataloader)
    with tqdm.tqdm(total=total_step) as pbar:
        for i in range(cfg.trainer.epochs):
            for idx, data in enumerate(dataloader):
                image_pt = data["image"]
                prompt = data["prompt"]
                indices = data["idx"].tolist() if torch.is_tensor(data["idx"]) else list(data["idx"])
                import pdb; pdb.set_trace()
                if cfg.trainer.grad_checkpointing:
                    # ---- Compute K_masked at all steps using precomputed K_orig ----
                    loss_ntk = torch.tensor(0.0, device=device, dtype=torch_dtype)
                    if args.ntk_lambda > 0.0 and len(indices) >= 1:
                        g_cpu_ntk = torch.Generator(device.type).manual_seed(seed)
                        with torch.no_grad():
                            prep_ntk = pipe.inference_preparation_phase(
                                prompt,
                                generator=g_cpu_ntk,
                                num_inference_steps=cfg.trainer.num_intervention_steps,
                                output_type="latent",
                            )
                        hookers_list = [cross_attn_hooker, ff_hooker] + ([_norm_hooker] if _norm_hooker else [])
                        n_steps = len(prep_ntk.timesteps)
                        ntk_accum = torch.tensor(0.0, device=device, dtype=torch.float32)
                        for step_idx, t in enumerate(prep_ntk.timesteps):
                            # Load all per-step data for the batch in one pass to avoid
                            # re-reading files and to extract z_in, proj, J together.
                            batch_step_data = [
                                torch.load(
                                    os.path.join(jacobian_dir, f"jacobian_{i}.pt"),
                                    weights_only=False,
                                )[step_idx]
                                for i in indices
                            ]
                            # Use original model's z_k so K_masked and K_orig are evaluated
                            # at the same input point, making the comparison principled.
                            prep_ntk.latents = torch.cat(
                                [d['z_in'] for d in batch_step_data], dim=0
                            ).to(device)
                            proj_full = batch_step_data[0]['proj'].to(device)
                            proj_indices = torch.randperm(proj_full.shape[1], device=device)[:args.ntk_proj_dim]
                            proj_t = proj_full[:, proj_indices]
                            # K_orig from precomputed Jacobians (mask=1 reference)
                            J_batch = torch.stack([d['J'] for d in batch_step_data]).to(device)
                            J_sub = J_batch[:, proj_indices, :]
                            J_flat = J_sub.reshape(len(indices), -1).float()
                            norms = J_flat.norm(dim=1, keepdim=True).clamp(min=1e-8)
                            K_orig_t = (J_flat / norms) @ (J_flat / norms).T
                            # K_masked: forward with sigmoid(λ) masks, differentiate w.r.t.
                            # mask values so gradient flows back to λ via sigmoid.
                            with ntk_masking_context(hookers_list, use_sigmoid_lambda=True) as ntk_params:
                                with torch.set_grad_enabled(True):
                                    out_latents_ntk = pipe.inference_with_grad_denoising_step(
                                        step_idx, t, prep_ntk
                                    )
                                K_masked_t = compute_ntk_matrix(
                                    out_latents_ntk, ntk_params, args.ntk_proj_dim,
                                    proj_t, create_graph=True,
                                )
                            logger.info(f"Step {step_idx}, t: {t}. K_orig_t: {K_orig_t.detach().cpu().numpy()} K_masked_t: {K_masked_t.detach().cpu().numpy()}")
                            ntk_accum = ntk_accum + F.mse_loss(
                                K_masked_t, K_orig_t.to(dtype=K_masked_t.dtype)
                            )
                        loss_ntk = (ntk_accum / n_steps).to(torch_dtype)
                        del out_latents_ntk, K_masked_t, K_orig_t, proj_t, ntk_accum
                        torch.cuda.empty_cache()

                    # ---- Masked model: checkpointed forward + backward ----
                    g_cpu = torch.Generator(device.type).manual_seed(seed)
                    with torch.no_grad():
                        preparation_phase_output = pipe.inference_preparation_phase(
                            prompt,
                            generator=g_cpu,
                            num_inference_steps=cfg.trainer.num_intervention_steps,
                            output_type="latent",
                        )
                        intermediate_latents = [preparation_phase_output.latents]
                        timesteps = preparation_phase_output.timesteps
                        for timesteps_idx, t in enumerate(timesteps):
                            latents = pipe.inference_with_grad_denoising_step(
                                timesteps_idx, t, preparation_phase_output
                            )
                            preparation_phase_output.latents = latents
                            intermediate_latents.append(latents)
                        intermediate_latents.pop()
                        latents.requires_grad = True

                    with torch.set_grad_enabled(True):
                        prompt_embeds = preparation_phase_output.prompt_embeds
                        image = pipe.inference_with_grad_aft_denoising(
                            latents, prompt_embeds, g_cpu, "latent", True, device
                        )
                        loss, loss_reconstruct, loss_reg, _ = pruning_loss(
                            reconstruction_loss_func,
                            image_pt,
                            image,
                            cross_attn_hooker,
                            ff_hooker,
                            device,
                            torch_dtype,
                            cfg,
                            logger=None,  # logged below with the correct loss_ntk
                            norm_hooker=_norm_hooker,
                            ntk_lambda=0.0,  # NTK loss computed per-step above
                        )
                        loss = loss + args.ntk_lambda * loss_ntk
                        logger.info(
                            f"loss_reconstruct: {loss_reconstruct.item()}"
                            f" loss_reg: {loss_reg.item()}"
                            f" loss_ntk: {loss_ntk.item()}"
                        )
                        accelerator.backward(loss)
                        grad = latents.grad.detach()

                    timesteps = preparation_phase_output.timesteps
                    if cfg.trainer.n_lr == 0:
                        trainable_lambs = cross_attn_hooker.lambs + ff_hooker.lambs
                    else:
                        trainable_lambs = cross_attn_hooker.lambs + ff_hooker.lambs + norm_hooker.lambs
                    for rev_idx, t in enumerate(reversed(timesteps)):
                        current_latents = intermediate_latents[-(rev_idx + 1)].detach()
                        intermediate_latents[-(rev_idx + 1)] = None  # free immediately
                        current_latents.requires_grad = True
                        step_idx = len(timesteps) - rev_idx - 1
                        with torch.set_grad_enabled(True):
                            preparation_phase_output.latents = current_latents
                            latents = pipe.inference_with_grad_denoising_step(
                                step_idx, t, preparation_phase_output, step_index=step_idx,
                            )
                            # Compute lamb grads and latent grad in one pass to avoid
                            # holding the computation graph twice (no retain_graph needed).
                            all_grads = torch.autograd.grad(
                                latents,
                                trainable_lambs + [current_latents],
                                grad_outputs=grad,
                                retain_graph=False,
                            )
                            lamb_grads = all_grads[:len(trainable_lambs)]
                            for lamb, lamb_grad in zip(trainable_lambs, lamb_grads):
                                if lamb.grad is None:
                                    lamb.grad = lamb_grad
                                else:
                                    lamb.grad += lamb_grad
                            grad = all_grads[len(trainable_lambs)]
                            del all_grads, lamb_grads, latents
                        torch.cuda.empty_cache()
                    del intermediate_latents, grad
                    torch.cuda.empty_cache()
                else:
                    # ---- Masked model: step by step for per-step NTK ----
                    g_cpu = torch.Generator(device.type).manual_seed(seed)
                    with torch.no_grad():
                        prep = pipe.inference_preparation_phase(
                            prompt,
                            generator=g_cpu,
                            num_inference_steps=cfg.trainer.num_intervention_steps,
                            output_type="latent",
                        )

                    hookers_list = [cross_attn_hooker, ff_hooker] + ([_norm_hooker] if _norm_hooker else [])
                    B = prep.latents.shape[0]
                    n_steps = len(prep.timesteps)
                    loss_ntk = torch.tensor(0.0, device=device, dtype=torch.float32)

                    # ---- Per-step NTK: use original model's z_k so K_masked and K_orig
                    #      are evaluated at the same input point each step. ----
                    if args.ntk_lambda > 0.0 and B >= 2:
                        for step_idx, t in enumerate(prep.timesteps):
                            batch_step_data = [
                                torch.load(
                                    os.path.join(jacobian_dir, f"jacobian_{i}.pt"),
                                    weights_only=False,
                                )[step_idx]
                                for i in indices
                            ]
                            prep.latents = torch.cat(
                                [d['z_in'] for d in batch_step_data], dim=0
                            ).to(device)
                            proj_full = batch_step_data[0]['proj'].to(device)
                            proj_indices = torch.randperm(proj_full.shape[1], device=device)[:args.ntk_proj_dim]
                            proj_t = proj_full[:, proj_indices]
                            J_batch = torch.stack([d['J'] for d in batch_step_data]).to(device)
                            J_sub = J_batch[:, proj_indices, :]
                            J_flat = J_sub.reshape(len(indices), -1).float()
                            norms = J_flat.norm(dim=1, keepdim=True).clamp(min=1e-8)
                            K_orig_t = (J_flat / norms) @ (J_flat / norms).T
                            with ntk_masking_context(hookers_list, use_sigmoid_lambda=True) as ntk_params:
                                with torch.set_grad_enabled(True):
                                    out_latents = pipe.inference_with_grad_denoising_step(
                                        step_idx, t, prep
                                    )
                                K_masked_t = compute_ntk_matrix(
                                    out_latents, ntk_params, args.ntk_proj_dim,
                                    proj_t, create_graph=True,
                                )
                            loss_ntk = loss_ntk + F.mse_loss(
                                K_masked_t, K_orig_t.to(dtype=K_masked_t.dtype)
                            )

                    loss_ntk = (loss_ntk / n_steps).to(torch_dtype)

                    # ---- Reconstruction: run masked model's own trajectory ----
                    # Use a fresh prep so the NTK loop's z_k overwrites don't affect
                    # the latents seen during reconstruction.
                    g_cpu = torch.Generator(device.type).manual_seed(seed)
                    with torch.no_grad():
                        prep_recon = pipe.inference_preparation_phase(
                            prompt,
                            generator=g_cpu,
                            num_inference_steps=cfg.trainer.num_intervention_steps,
                            output_type="latent",
                        )
                        for step_idx, t in enumerate(prep_recon.timesteps):
                            out_latents = pipe.inference_with_grad_denoising_step(
                                step_idx, t, prep_recon
                            )
                            prep_recon.latents = out_latents.detach()

                    final_latents = out_latents.detach().requires_grad_(True)
                    with torch.set_grad_enabled(True):
                        image = pipe.inference_with_grad_aft_denoising(
                            final_latents, prep_recon.prompt_embeds, g_cpu, "latent", True, device
                        )

                    loss, loss_reconstruct, loss_reg, _ = pruning_loss(
                        reconstruction_loss_func,
                        image_pt,
                        image,
                        cross_attn_hooker,
                        ff_hooker,
                        device,
                        torch_dtype,
                        cfg,
                        norm_hooker=_norm_hooker,
                        ntk_lambda=0.0,  # NTK loss computed per-step above
                    )
                    loss = loss + args.ntk_lambda * loss_ntk
                    accelerator.backward(loss)

                if (idx * batch_size) % cfg.trainer.accumulate_grad_batches == 0:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                if (idx * batch_size) % cfg.logger.plot_interval == 0:
                    if cfg.logger.type == "wandb":
                        wandb.log(
                            {
                                "loss_reconstruct": loss_reconstruct,
                                "loss_reg": loss_reg,
                                "loss_ntk": loss_ntk,
                                "loss": loss,
                                "lr": lr_scheduler.get_last_lr()[0],
                                "vram": torch.cuda.max_memory_allocated(device) / 1024**3,
                            },
                            commit=False,
                        )
                        for index, lamb in enumerate(cross_attn_hooker.lambs):
                            heads = [f"head_{j}" for j in range(lamb.shape[0])]
                            data_table = [[h, l] for h, l in zip(heads, lamb.detach().float().cpu().numpy())]
                            table = wandb.Table(data=data_table, columns=["head", "lambda"])
                            wandb.log(
                                {
                                    f"lambda_{lamda_block_names[index]}": wandb.plot.bar(
                                        table, "head", "lambda", title=f"lambda_{lamda_block_names[index]}"
                                    )
                                },
                                commit=False,
                            )
                        img_continues_mask = save_image_seed(
                            pipe, validation_prompts, cfg.trainer.num_intervention_steps, device, seed, save_dir=None
                        )
                        img_discrete_mask = save_image_binarize_seed(
                            pipe,
                            [cross_attn_hooker, ff_hooker],
                            validation_prompts,
                            cfg.trainer.num_intervention_steps,
                            device,
                            seed,
                            save_dir=None,
                        )
                        wandb.log(
                            {
                                "image with continuous mask": [wandb.Image(i) for i in img_continues_mask],
                                "image with discrete mask": [wandb.Image(i) for i in img_discrete_mask],
                            }
                        )
                    else:
                        path = os.path.join(args.save_dir, cfg.logger.project, cfg.logger.notes, "images")
                        val_path = os.path.join(path, "validation", f"epoch_{i}_step_{idx}")
                        train_path = os.path.join(path, "train", f"epoch_{i}_step_{idx}")
                        for save_path, prompts in zip(
                            [val_path, train_path], [validation_prompts, train_dataset[0]["prompt"]]
                        ):
                            save_image_seed(
                                pipe,
                                prompts,
                                cfg.trainer.num_intervention_steps,
                                device,
                                seed,
                                save_dir=os.path.join(save_path, "continues mask"),
                            )
                            torch.cuda.empty_cache()
                            hookers = [cross_attn_hooker, ff_hooker]
                            if cfg.trainer.n_lr != 0:
                                hookers.append(norm_hooker)
                            save_image_binarize_seed(
                                pipe,
                                hookers,
                                prompts,
                                cfg.trainer.num_intervention_steps,
                                device,
                                seed,
                                save_dir=os.path.join(save_path, "discrete mask"),
                            )
                            torch.cuda.empty_cache()

                    for n, lamb in zip(lamda_block_names, cross_attn_hooker.lambs):
                        logger.info(f"lambda in {n}: {lamb.clamp(min=0).tolist()}")
                    for n, lamb in zip(ff_lambda_block_names, ff_hooker.lambs):
                        logger.info(
                            f"lambda {n}: max {lamb.max().item()}, min {lamb.min().item()}, mean {lamb.mean().item()}"
                        )
                    if cfg.trainer.n_lr != 0:
                        for n, lamb in zip(norm_lambda_block_names, norm_hooker.lambs):
                            logger.info(
                                f"lambda {n}: max {lamb.max().item()}, "
                                + f"min {lamb.min().item()}, mean {lamb.mean().item()}"
                            )

                    masking_threshold = 0
                    remain_head, total_head, sparsity = calculate_mask_sparsity(cross_attn_hooker, masking_threshold)
                    ff_remain_head, ff_total_head, ff_sparsity = calculate_mask_sparsity(ff_hooker, masking_threshold)
                    if cfg.trainer.n_lr != 0:
                        norm_remain_head, norm_total_head, norm_sparsity = calculate_mask_sparsity(
                            norm_hooker, masking_threshold
                        )
                    else:
                        norm_remain_head, norm_total_head, norm_sparsity = 0, 0, 0

                    logger.info(
                        f"mask sparsity for threshold {masking_threshold}: "
                        f"{remain_head}/{total_head}, {sparsity:.2%} \n"
                        f"ff_mask sparsity for threshold {masking_threshold}: "
                        f"{ff_remain_head}/{ff_total_head}, {ff_sparsity:.2%} \n"
                        f"norm_mask sparsity for threshold {masking_threshold}: "
                        f"{norm_remain_head}/{norm_total_head}, {norm_sparsity:.2%} \n"
                    )
                    logger.info(
                        f"loss_reconstruct: {loss_reconstruct}, loss_reg: {loss_reg}, "
                        f"loss_ntk: {loss_ntk}, total_loss: {loss}"
                    )

                    cross_attn_hooker.save(os.path.join("lambda", f"epoch_{i}_step_{idx}_attn.pt"))
                    ff_hooker.save(os.path.join("lambda", f"epoch_{i}_step_{idx}_ff.pt"))
                    if cfg.trainer.n_lr != 0:
                        norm_hooker.save(os.path.join("lambda", f"epoch_{i}_step_{idx}_norm.pt"))
                    logger.info(f"epoch: {i}, step: {idx}: saving lambda")
                pbar.update()

        logger.info(f"epoch {i+1}/{cfg.trainer.epochs}")

        hookers = [cross_attn_hooker, ff_hooker]
        if cfg.trainer.n_lr != 0:
            hookers.append(norm_hooker)

        if cfg.logger.type == "wandb":
            logger.info("Saving final image to wandb ...")
            img = save_image_binarize_seed(
                pipe,
                hookers,
                validation_prompts,
                cfg.trainer.num_intervention_steps,
                device,
                seed,
                save_dir=None,
            )
            wandb.log({"image": [wandb.Image(i) for i in img]})
            run.finish()
            wandb.finish()
            logger.info("Done")
        else:
            path = os.path.join(args.save_dir, cfg.logger.project, cfg.logger.notes, "images")
            train_path = os.path.join(path, "train", "final_image")
            val_path = os.path.join(path, "validation", "final_image")
            prompts = [validation_prompts, train_dataset[0]["prompt"]]
            for save_path, prompt in zip([val_path, train_path], prompts):
                save_image_binarize_seed(
                    pipe,
                    hookers,
                    prompt,
                    cfg.trainer.num_intervention_steps,
                    device,
                    seed,
                    save_dir=save_path,
                )

    logger.info(f"Training finished with cfg:{args.cfg}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train EcoDiff with NTK alignment loss")
    parser.add_argument("--cfg", type=str, default="configs/sdxl.yaml")
    parser.add_argument(
        "--validation_prompts_path", "-v", type=str,
        default="configs/validation_prompts_small.yaml",
    )
    parser.add_argument("--save_dir", "-s", type=str, default="./results")
    parser.add_argument("--notes", type=str, default="ntk_run")
    parser.add_argument("--islaunch", action="store_true")
    parser.add_argument("--task", "-t", type=str, default="general")
    parser.add_argument("--load_lambda", "-l", action="store_true")
    # parser.add_argument(
    #     "--jacobian_dir", type=str, default=None,
    #     help="Directory for precomputed Jacobians. Defaults to <save_dir>/jacobians.",
    # )
    parser.add_argument(
        "--ntk_lambda", type=float, default=0.1,
        help="Weight for the NTK alignment loss term (0 = disabled).",
    )
    # parser.add_argument(
    #     "--ntk_proj_dim", type=int, default=16,
    #     help=(
    #         "Number of projection directions sampled at training time for the "
    #         "masked-model NTK computation. Must be <= precompute_proj_dim. "
    #         "Reduce if training-time memory is tight."
    #     ),
    # )
    # parser.add_argument(
    #     "--precompute_proj_dim", type=int, default=64,
    #     help=(
    #         "Number of projection directions stored during Jacobian precomputation. "
    #         "A larger value gives a richer reference; at training time only "
    #         "ntk_proj_dim of these are randomly sampled per step."
    #     ),
    # )
    args = parser.parse_args()
    main(args)
