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

def compute_ntk_kernel(J: torch.Tensor) -> torch.Tensor:
    """Cosine-normalised NTK matrix from a batch of Jacobian tensors.

    K[i,j] = cosine(<J[i].flatten()>, <J[j].flatten()>)

    J: [B, proj_dim, P] — Jacobian of proj_dim scalar outputs w.r.t. P params.
    """
    J_flat = J.reshape(J.shape[0], -1)  # [B, proj_dim * P]
    norms = J_flat.norm(dim=1, keepdim=True).clamp(min=1e-8)
    J_flat = J_flat / norms
    return J_flat @ J_flat.T  # [B, B]


@contextmanager
def no_masking_context(hookers):
    """Temporarily disable masking on all hookers (original model forward)."""
    orig = [h.masking for h in hookers]
    for h in hookers:
        h.masking = "no_masking"
    try:
        yield
    finally:
        for h, m in zip(hookers, orig):
            h.masking = m


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

    Yields the flat list of mask leaf tensors (ref_params).
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


@contextmanager
def detached_mask_context(hookers):
    """Switch hookers to 'binary' masking with detached sigmoid(λ) leaf tensors.

    Yields mask_params: flat list of sigmoid(λ).detach().requires_grad_(True) tensors.
    These serve as Jacobian computation targets without connecting the Jacobian
    graph to λ.  After computing J_masked with autograd.grad w.r.t. mask_params,
    apply sigma_prime = sigmoid(λ)·(1−sigmoid(λ)) weighting (from the original
    λ tensors, collected before entering this context) to restore gradient flow.
    """
    saved = [(list(h.lambs), h.masking) for h in hookers]
    mask_params = []
    for hooker, (orig_lambs, _) in zip(hookers, saved):
        new_lambs = [torch.sigmoid(l).detach().requires_grad_(True) for l in orig_lambs]
        hooker.lambs = new_lambs
        hooker.masking = "binary"
        mask_params.extend(new_lambs)
    try:
        yield mask_params
    finally:
        for hooker, (orig_lambs, orig_masking) in zip(hookers, saved):
            hooker.lambs = orig_lambs
            hooker.masking = orig_masking


def precompute_original_jacobians(
    pipe,
    dataset,
    hookers,
    cfg,
    device,
    seed,
    proj_dim: int = 4,
    save_dir: str = "jacobians",
):
    """
    Precompute per-step Jacobian vectors for the original (unmasked) model
    over all samples in the dataset.  Results are saved to disk as a list of dicts:

        [{'J_orig': tensor[1, proj_dim, P], 'proj': tensor[D, proj_dim],
          'z_in': tensor}, ...]   # len = n_steps

    J_orig: Jacobian ∂(out_flat @ proj[:, k])/∂mask |_{mask=ones},
            shape [1, proj_dim, P] where P = total number of mask parameters.
            Used at training time to compute K_orig = compute_ntk_kernel(J_orig_sub).
    proj: projection matrix [D, proj_dim] seeded by step_idx for reproducibility.
    z_in: input latent for this step (ensures K_masked is evaluated at the same point).

    hookers: list of hooker objects (cross_attn, ff, norm).
    """
    os.makedirs(save_dir, exist_ok=True)

    # Temporarily enable GC to bound activation memory during proj_dim Jacobian
    # backward passes per step.
    transformer = getattr(pipe, 'transformer', getattr(pipe, 'unet', None))
    gc_enabled = False
    if transformer is not None and hasattr(transformer, 'enable_gradient_checkpointing'):
        transformer.enable_gradient_checkpointing()
        gc_enabled = True

    # Binary masking with mask=sigmoid(λ_init) leaves: same operating point as
    # training, so K_orig ≈ K_eff at t=0 and the loss grows only as λ diverges.
    with detached_mask_context(hookers) as ref_params:
        for sample_idx in tqdm.tqdm(range(len(dataset)), desc="Precomputing Jacobians"):
            save_path = os.path.join(save_dir, f"jacobian_{sample_idx}.pt")
            if os.path.exists(save_path):
                continue
            data = dataset[sample_idx]
            prompt = [data["prompt"]] if isinstance(data["prompt"], str) else data["prompt"]

            g_cpu = torch.Generator(device.type).manual_seed(seed + sample_idx)
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
                # Grad enabled so autograd.grad can trace through the forward pass.
                with torch.set_grad_enabled(True):
                    out_latents = pipe.inference_with_grad_denoising_step(step_idx, t, prep)

                D = out_latents.reshape(1, -1).shape[1]
                gen = torch.Generator().manual_seed(step_idx)
                proj = F.normalize(torch.randn(D, proj_dim, generator=gen), dim=0).to(device)
                out_flat = out_latents.reshape(1, -1).float()
                projected = out_flat @ proj  # [1, proj_dim]

                # proj_dim backward passes; retain graph until the last one.
                J_rows = []
                for k in range(proj_dim):
                    g = torch.autograd.grad(
                        projected[0, k],
                        ref_params,
                        retain_graph=(k < proj_dim - 1),
                        create_graph=False,
                        allow_unused=True,
                    )
                    J_rows.append(torch.cat([
                        gg.flatten().float() if gg is not None
                        else torch.zeros(p.numel(), dtype=torch.float32, device=device)
                        for gg, p in zip(g, ref_params)
                    ]))
                J = torch.stack(J_rows).unsqueeze(0)  # [1, proj_dim, P]

                steps_data.append({
                    'J_orig': J.detach().cpu(),
                    'proj': proj.cpu(),
                    'z_in': prep.latents.detach().cpu(),
                })
                prep.latents = out_latents.detach()
                torch.cuda.empty_cache()
            torch.save(steps_data, save_path)

    if gc_enabled:
        transformer.disable_gradient_checkpointing()




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
        verbose=args.verbose
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
        verbose=args.verbose
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
            verbose=args.verbose
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
                if cfg.trainer.grad_checkpointing:
                    # ---- Compute K_masked at all steps using precomputed K_orig ----
                    loss_ntk = torch.tensor(0.0, device=device, dtype=torch_dtype)
                    if args.ntk_lambda > 0.0 and len(indices) >= 2:
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
                        B = len(indices)
                        all_lambs = [l for hooker in hookers_list for l in hooker.lambs]
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
                            # K_orig: NTK kernel from precomputed Jacobians.
                            J_orig = torch.cat(
                                [d['J_orig'] for d in batch_step_data], dim=0
                            ).to(device).float()  # [B, precompute_proj_dim, P]
                            J_orig_sub = J_orig[:, proj_indices.cpu(), :]  # [B, ntk_proj_dim, P]
                            K_orig_t = compute_ntk_kernel(J_orig_sub).detach()

                            # K_eff: NTK kernel of masked model, differentiable via sigma_prime.
                            # detached_mask_context gives sigmoid(λ).detach() leaves for J computation.
                            with detached_mask_context(hookers_list) as mask_params:
                                with torch.set_grad_enabled(True):
                                    out_latents_ntk = pipe.inference_with_grad_denoising_step(
                                        step_idx, t, prep_ntk
                                    )
                                out_flat = out_latents_ntk.reshape(B, -1).float()
                                projected = out_flat @ proj_t  # [B, ntk_proj_dim]
                                n_grads = B * args.ntk_proj_dim
                                grad_count = 0
                                J_masked_rows = []
                                for b in range(B):
                                    J_row = []
                                    for k in range(args.ntk_proj_dim):
                                        grad_count += 1
                                        g = torch.autograd.grad(
                                            projected[b, k], mask_params,
                                            retain_graph=(grad_count < n_grads),
                                            create_graph=False,
                                            allow_unused=True,
                                        )
                                        J_row.append(torch.cat([
                                            gg.flatten().float() if gg is not None
                                            else torch.zeros(p.numel(), dtype=torch.float32, device=device)
                                            for gg, p in zip(g, mask_params)
                                        ]))
                                    J_masked_rows.append(torch.stack(J_row))  # [ntk_proj_dim, P]
                                J_masked = torch.stack(J_masked_rows).detach()  # [B, ntk_proj_dim, P]
                            # Context exited; hooks restored. J_masked is fully detached.

                            # sigma_prime = σ'(λ): differentiable w.r.t. original λ.
                            sigma_prime = torch.cat([
                                (torch.sigmoid(l) * (1.0 - torch.sigmoid(l))).flatten().float()
                                for l in all_lambs
                            ], dim=0)  # [P]
                            J_eff = J_masked * sigma_prime.view(1, 1, -1)  # [B, ntk_proj_dim, P]
                            K_eff = compute_ntk_kernel(J_eff)  # [B, B]
                            ntk_step_loss = F.mse_loss(K_eff, K_orig_t.to(dtype=K_eff.dtype))
                            # Backward flows only through sigma_prime → λ; no GC recomputation.
                            (args.ntk_lambda * ntk_step_loss / n_steps).backward()
                            if args.verbose:
                                K_o = K_orig_t.cpu().float().numpy()                                                                              
                                K_e = K_eff.detach().cpu().float().numpy()
                                logger.info(                                                                                                      
                                    f"Step {step_idx}, t={t}\n"                                                                                 
                                    f"  K_orig :\n{K_o}\n"                                                                                        
                                    f"  K_eff  :\n{K_e}\n"                                                                                        
                                    f"  diff   :\n{K_e - K_o}\n"            
                                    f"  ntk_loss: {ntk_step_loss.item():.6f}"                                                                     
                                )     
                            ntk_accum = ntk_accum + ntk_step_loss.detach()
                        loss_ntk = (ntk_accum / n_steps).to(torch_dtype)
                        del J_orig, J_masked, J_eff, K_eff, K_orig_t, proj_t, ntk_accum
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
                        # NTK loss already backpropagated per step inside ntk_masking_context
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
                            bptt_scale = 1.0 / cfg.trainer.accumulate_grad_batches
                            for lamb, lamb_grad in zip(trainable_lambs, lamb_grads):
                                if lamb.grad is None:
                                    lamb.grad = lamb_grad * bptt_scale
                                else:
                                    lamb.grad += lamb_grad * bptt_scale
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
                    all_lambs = [l for hooker in hookers_list for l in hooker.lambs]

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

                            # K_orig: NTK kernel from precomputed Jacobians.
                            J_orig = torch.cat(
                                [d['J_orig'] for d in batch_step_data], dim=0
                            ).to(device).float()  # [B, precompute_proj_dim, P]
                            J_orig_sub = J_orig[:, proj_indices.cpu(), :]  # [B, ntk_proj_dim, P]
                            K_orig_t = compute_ntk_kernel(J_orig_sub).detach()

                            # K_eff: NTK kernel of masked model, differentiable via sigma_prime.
                            with detached_mask_context(hookers_list) as mask_params:
                                with torch.set_grad_enabled(True):
                                    out_latents = pipe.inference_with_grad_denoising_step(
                                        step_idx, t, prep
                                    )
                                out_flat = out_latents.reshape(B, -1).float()
                                projected = out_flat @ proj_t  # [B, ntk_proj_dim]
                                n_grads = B * args.ntk_proj_dim
                                grad_count = 0
                                J_masked_rows = []
                                for b in range(B):
                                    J_row = []
                                    for k in range(args.ntk_proj_dim):
                                        grad_count += 1
                                        g = torch.autograd.grad(
                                            projected[b, k], mask_params,
                                            retain_graph=(grad_count < n_grads),
                                            create_graph=False,
                                            allow_unused=True,
                                        )
                                        J_row.append(torch.cat([
                                            gg.flatten().float() if gg is not None
                                            else torch.zeros(p.numel(), dtype=torch.float32, device=device)
                                            for gg, p in zip(g, mask_params)
                                        ]))
                                    J_masked_rows.append(torch.stack(J_row))  # [ntk_proj_dim, P]
                                J_masked = torch.stack(J_masked_rows).detach()  # [B, ntk_proj_dim, P]
                            # Context exited; hooks restored. J_masked is fully detached.

                            sigma_prime = torch.cat([
                                (torch.sigmoid(l) * (1.0 - torch.sigmoid(l))).flatten().float()
                                for l in all_lambs
                            ], dim=0)  # [P]
                            J_eff = J_masked * sigma_prime.view(1, 1, -1)  # [B, ntk_proj_dim, P]
                            K_eff = compute_ntk_kernel(J_eff)  # [B, B]
                            ntk_step_loss = F.mse_loss(K_eff, K_orig_t.to(dtype=K_eff.dtype))
                            (args.ntk_lambda * ntk_step_loss / n_steps).backward()
                            loss_ntk = loss_ntk + ntk_step_loss.detach()

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
                    # NTK loss already backpropagated per step inside ntk_masking_context
                    accelerator.backward(loss)

                if idx % cfg.trainer.accumulate_grad_batches == 0:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                if idx % cfg.logger.plot_interval == 0:
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
    parser.add_argument(
        "--verbose", action="store_true",
        help="Log per-step K_orig / K_eff matrices and NTK loss.",
    )
    args = parser.parse_args()
    main(args)
