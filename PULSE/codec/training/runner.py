"""Single-GPU / torchrun training with stage-aware checkpoint resume."""
import argparse
import contextlib
import json
import os
from pathlib import Path
import random

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from codec.checkpoint import save_model
from codec.config import TRAINING_SIZES, build_model, load_config, recipe_path, validate_recipe
from codec.models.activations import InputDomainHardSigmoid, LeakyHardSigmoid
from codec.optim.soap import SOAP
from codec.training.data import ImageDataset, read_records
from codec.training.losses import DinoDiscriminator, RDObjective, discriminator_loss
from codec.utils.model_io import atomic_torch_save, load_native_state_dict


def worker_seed(worker_id):
    """Match the reference Lightning worker seed sequence without its runtime."""
    rank = int(os.environ.get("RANK", 0))
    base_seed = torch.initial_seed() - worker_id
    value = (base_seed << 32) | (worker_id << 16) | rank
    seeds = []
    for _ in range(4):
        value = (value * 6364136223846793005 + 1) & ((1 << 64) - 1)
        seeds.append(value)
    torch.manual_seed(seeds[0])
    random.seed((seeds[1] << 32) | seeds[2])
    np.random.seed(np.random.SeedSequence([base_seed, worker_id, rank]).generate_state(4))


def configure_training_runtime():
    # SOAP's FP32 matrix products also need the reference precision policy.
    torch.set_float32_matmul_precision("medium")
    torch.backends.cudnn.allow_tf32 = True


def configure_training_model(model, stage):
    # The original LR MSE run precedes the input-domain surrogate used by HR
    # and perceptual training. Both gates have identical forward arithmetic.
    gate = {
        "mse_lr": LeakyHardSigmoid,
        "mse_hr": InputDomainHardSigmoid,
        "perceptual_lr": InputDomainHardSigmoid,
        "perceptual_hr": InputDomainHardSigmoid,
        "perceptual_roi": InputDomainHardSigmoid,
    }[stage]
    model.decoder.renderer.gate_activation = gate()


def seed_stage(seed, model_init_rng):
    """Each stage starts like a fresh seeded process loading the previous weights."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_rng_state(model_init_rng)


def freeze_heads(discriminator, freeze):
    for name, parameter in discriminator.named_parameters():
        parameter.requires_grad_(not freeze and not name.startswith("backbone."))


def make_optimizer(parameters, settings, lr):
    settings = dict(settings)
    kind = settings.pop("name")
    settings.pop("lr", None)
    if "betas" in settings:
        settings["betas"] = tuple(settings["betas"])
    return {"soap": SOAP, "adam": torch.optim.Adam}[kind](
        parameters, lr=lr, **settings)


def training_objective(model, criterion, image, metadata, qp, discriminator=None, *, disc_update=False):
    """One original-style G or D objective; optimizer/accumulation live outside."""
    if disc_update:
        with torch.no_grad():
            reconstruction, latent, _ = model(image, qp=qp, return_codec_res=True)
        fake = discriminator(reconstruction, latent.detach())
        real = discriminator(image * 2 - 1, latent.detach())
        loss = discriminator_loss(real, fake)
        return loss, {"loss": loss.detach(), "disc_loss": loss.detach()}
    reconstruction, latent, rates = model(image, qp=qp, return_codec_res=True)
    # The discriminator's generator term is a per-image mean of local logits.
    # Evaluate this after the reconstruction/ROI terms, matching the reference
    # autograd accumulation order (important for BF16 bitwise comparisons).
    gan = None if discriminator is None else lambda: -discriminator(
        reconstruction, latent.detach()).reshape(image.shape[0], -1).mean(dim=1)
    return criterion(reconstruction, image, rates, qp, metadata, gan)


def main(objective):
    parser = argparse.ArgumentParser(description=f"PULSE {objective} training")
    parser.add_argument("--size", choices=TRAINING_SIZES[objective], default="base")
    parser.add_argument("--config", help="override the size/objective YAML recipe")
    parser.add_argument("--train-data", help="image directory, TXT list or JSONL")
    parser.add_argument("--hr-data", help="high-resolution training images")
    parser.add_argument("--roi-data", help="Stage-II JSONL with faces/texts boxes")
    parser.add_argument("--data-root", help="root for relative manifest paths")
    parser.add_argument("--output")
    parser.add_argument("--check-config", action="store_true", help="validate and print the full recipe without training")
    parser.add_argument("--init", help="initialize backbone from a model bundle")
    parser.add_argument("--resume", help="resume training-state.pt, including optimizers")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--micro-batch-size", type=int)
    parser.add_argument("--save-every", type=int, default=10000)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--max-steps", type=int, help="limit optimizer steps for a smoke run")
    parser.add_argument("--crop-size", type=int, help="override all crop sizes for a smoke run")
    parser.add_argument("--stop-after-stage", help="stop after the named stage")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--face-teacher")
    parser.add_argument("--ocr-teacher")
    parser.add_argument("--dino-source", default="facebookresearch/dinov2")
    args = parser.parse_args()
    recipe = validate_recipe(load_config(args.config or recipe_path(objective, args.size)))
    if recipe["objective"] != objective or recipe["size"] != args.size:
        parser.error("recipe objective/size must match the command")
    if args.check_config:
        print(json.dumps(recipe, indent=2))
        return
    if not args.train_data or not args.output:
        parser.error("--train-data and --output are required for training")
    args.seed = recipe["runtime"]["seed"] if args.seed is None else args.seed
    if args.init and args.resume:
        parser.error("--init and --resume are mutually exclusive")
    for key in ("save_every", "log_every", "threads"):
        if getattr(args, key) < 1:
            parser.error(f"--{key.replace('_', '-')} must be positive")
    if args.workers < 0 or (args.max_steps is not None and args.max_steps < 1):
        parser.error("--workers must be nonnegative and --max-steps positive")
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}" if args.device == "cuda" or
                          (world > 1 and args.device.startswith("cuda")) else args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.set_num_threads(args.threads)
    if world > 1:
        torch.distributed.init_process_group("nccl" if device.type == "cuda" else "gloo")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    configure_training_runtime()
    if args.stop_after_stage and args.stop_after_stage not in {s["name"] for s in recipe["stages"]}:
        parser.error("--stop-after-stage is not a stage in this recipe")
    if Path(args.output).exists() and any(Path(args.output).iterdir()) and not args.resume:
        raise FileExistsError("output directory is not empty; use --resume or a new directory")
    config = dict(size=args.size, objective=objective,
                  architecture=recipe["architecture"])
    model = build_model(config).to(device)
    model_init_rng = torch.get_rng_state()
    resume = torch.load(args.resume, map_location="cpu", weights_only=True) if args.resume else None
    if resume:
        if resume["recipe"] != recipe:
            raise ValueError("resume recipe does not match; use --init for a new recipe")
        model.load_state_dict(resume["model"], strict=True)
    elif args.init:
        state = load_native_state_dict(Path(args.init) / "model.pt")
        state = {k: v for k, v in state.items() if not k.startswith("entropy_model.meta_prior.")}
        model.load_state_dict(state, strict=True)
    wrapped = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None,
                  find_unused_parameters=True) if world > 1 else model
    root = Path(args.output)
    if rank == 0:
        root.mkdir(parents=True, exist_ok=True)
        (root / "recipe.json").write_text(json.dumps(recipe, indent=2))
        (root / "runtime.json").write_text(json.dumps(dict(
            arguments=vars(args), world_size=world, torch=torch.__version__,
            float32_matmul_precision=torch.get_float32_matmul_precision(),
            cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
            loader=dict(drop_last=False, persistent_workers=args.workers > 0,
                        prefetch_factor=8 if args.workers else None),
        ), indent=2))
    global_step = int(resume["global_step"]) if resume else 0
    start_stage = int(resume["stage_index"]) if resume else 0
    first_step = int(resume["step_in_stage"]) if resume else 0
    executed_steps = 0
    for stage_index, stage in enumerate(recipe["stages"]):
        if stage_index < start_stage:
            continue
        configure_training_model(model, stage["name"])
        seed_stage(args.seed, model_init_rng)
        source = {"lr": args.train_data, "hr": args.hr_data,
                  "roi": args.roi_data}.get(stage["data"])
        if not source:
            raise ValueError(f"stage {stage['name']} requires --{stage['data']}-data")
        records = read_records(source, args.data_root)
        if stage.get("loss", {}).get("roi") and not any(r.get("faces") or r.get("texts") for r in records):
            raise ValueError("ROI stage requires annotated images in --roi-data")
        dataset = ImageDataset(records, args.crop_size or stage["crop"], **stage["augment"])
        global_batch = int(stage["batch"])
        if global_batch % world:
            raise ValueError("global stage batch size must be divisible by WORLD_SIZE")
        per_rank = global_batch // world
        micro = args.micro_batch_size or per_rank
        if micro < 1 or per_rank % micro:
            raise ValueError("--micro-batch-size must divide the per-rank batch size")
        accum = per_rank // micro
        if len(dataset) < micro * world:
            raise ValueError("training dataset is smaller than one distributed microbatch")
        sampler = DistributedSampler(dataset, world, rank, shuffle=True) if world > 1 else None
        loader = DataLoader(dataset, batch_size=micro, sampler=sampler, shuffle=sampler is None,
                            drop_last=False, num_workers=args.workers, pin_memory=device.type == "cuda",
                            worker_init_fn=worker_seed, persistent_workers=args.workers > 0,
                            **({"prefetch_factor": 8} if args.workers else {}))
        criterion = RDObjective(stage["loss"], model.qp_num,
                                face_teacher=args.face_teacher, ocr_teacher=args.ocr_teacher).to(device)
        opt_settings = stage["optimizer"]
        optimizer = make_optimizer(model.parameters(), opt_settings, stage["lr"])
        discriminator = None
        if stage["loss"].get("gan", 0):
            discriminator = DinoDiscriminator(model.M, args.dino_source).to(device).train()
            d_wrapped = DDP(discriminator, device_ids=[local_rank] if device.type == "cuda" else None) if world > 1 else discriminator
            d_settings = stage["disc_optimizer"]
            d_optimizer = make_optimizer(
                [p for p in discriminator.parameters() if p.requires_grad],
                d_settings, d_settings.get("lr", stage["lr"]))
        if resume and stage_index == start_stage:
            if resume["optimizer_name"] != opt_settings["name"]:
                raise ValueError("resume optimizer does not match")
            optimizer.load_state_dict(resume["optimizer"])
            if discriminator is not None:
                discriminator.load_state_dict(resume["discriminator"], strict=False)
                d_optimizer.load_state_dict(resume["d_optimizer"])
            random.setstate(resume["python_rng"])
            torch.set_rng_state(resume["torch_rng"].cpu())
            if device.type == "cuda" and resume.get("cuda_rng") is not None:
                torch.cuda.set_rng_state(resume["cuda_rng"].cpu(), device)
        else:
            first_step = 0
        epoch, iterator = 0, iter(loader)
        model.train()
        if rank == 0:
            print(json.dumps(dict(stage=stage["name"], status="started",
                                  images=len(dataset), crop=dataset.crop_size,
                                  batch=global_batch, micro_batch=micro,
                                  accumulation=accum, steps=stage["steps"])), flush=True)
        for step in range(first_step, stage["steps"]):
            lr = stage["lr"] * (stage["lr_gamma"] ** sum(step >= m for m in stage["milestones"]))
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            if discriminator is not None:
                d_optimizer.zero_grad(set_to_none=True)
            disc_update = discriminator is not None and (step + 1) % stage.get("disc_step", 2) == 0
            if discriminator is not None:
                freeze_heads(discriminator, not disc_update)
            logs = {}
            for micro_index in range(accum):
                try:
                    image, metadata = next(iterator)
                except StopIteration:
                    epoch += 1
                    if sampler:
                        sampler.set_epoch(epoch)
                    iterator = iter(loader)
                    image, metadata = next(iterator)
                image = image.to(device, non_blocking=True)
                metadata = {k: v.to(device, non_blocking=True) for k, v in metadata.items()}
                qp = torch.randint(model.qp_num, (image.shape[0],), device=device)
                active = d_wrapped if disc_update else wrapped
                sync = contextlib.nullcontext if world == 1 or micro_index == accum - 1 else active.no_sync
                amp = lambda: torch.autocast(device.type, dtype=torch.bfloat16)
                with sync(), amp():
                    loss, metrics = training_objective(
                        model if disc_update else wrapped, criterion, image, metadata, qp,
                        d_wrapped if disc_update else discriminator, disc_update=disc_update)
                    (loss / accum).backward()
                for key, value in metrics.items():
                    logs[key] = logs.get(key, 0.0) + float(value) / accum
            if not all(np.isfinite(value) for value in logs.values()):
                raise FloatingPointError(f"non-finite loss at step {global_step}: {logs}")
            clip = recipe["runtime"]["gradient_clip"]
            if disc_update:
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), clip, error_if_nonfinite=True)
                d_optimizer.step()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip, error_if_nonfinite=True)
                optimizer.step()
            global_step += 1
            executed_steps += 1
            finished = step + 1 == stage["steps"]
            limited = args.max_steps is not None and executed_steps >= args.max_steps
            if rank == 0 and (global_step % args.log_every == 0 or limited or finished):
                message = dict(stage=stage["name"], step=step + 1, global_step=global_step, lr=lr, **logs)
                print(json.dumps(message), flush=True)
                with (root / "train.jsonl").open("a") as stream:
                    stream.write(json.dumps(message) + "\n")
            if rank == 0 and (global_step % args.save_every == 0 or finished or limited):
                save_model(model, config, root / "model")
                training = dict(recipe=recipe, model=model.state_dict(), optimizer=optimizer.state_dict(),
                                optimizer_name=opt_settings["name"], stage_index=stage_index,
                                step_in_stage=step + 1, global_step=global_step,
                                python_rng=random.getstate(), torch_rng=torch.get_rng_state(),
                                cuda_rng=torch.cuda.get_rng_state(device) if device.type == "cuda" else None)
                if discriminator is not None:
                    training["discriminator"] = {k: v for k, v in discriminator.state_dict().items()
                                                 if not k.startswith("backbone.")}
                    training["d_optimizer"] = d_optimizer.state_dict()
                atomic_torch_save(training, root / "training-state.pt")
            if limited:
                if world > 1:
                    torch.distributed.barrier()
                    torch.distributed.destroy_process_group()
                return
        if rank == 0:
            save_model(model, config, root / stage["name"])
        resume = None
        if args.stop_after_stage == stage["name"]:
            break
    if world > 1:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()
