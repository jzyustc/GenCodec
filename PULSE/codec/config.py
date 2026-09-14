"""The three receiver budgets used by PULSE."""
from copy import deepcopy
import json
from pathlib import Path
from importlib.resources import files

import yaml

from codec.models.pulse import PULSE

SIZES = {
    "s": dict(M=256, z_ch=80, hidden_size=96, hidden_size_x=32,
              hx_bottleneck=16, mlp_ratio=2.0, num_cond_blocks=2,
              entropy_channels=128, dico_block_variant="plain"),
    "base": dict(M=320, z_ch=96, hidden_size=144, hidden_size_x=48,
                 hx_bottleneck=48, mlp_ratio=272 / 144, num_cond_blocks=2,
                 entropy_channels=128, dico_block_variant="ca"),
    "l": dict(M=320, z_ch=144, hidden_size=288, hidden_size_x=96,
              hx_bottleneck=48, mlp_ratio=800 / 288, num_cond_blocks=4,
              entropy_channels=256, dico_block_variant="ca"),
}


TRAINING_SIZES = {"mse": ("s", "base"), "perceptual": ("base", "l")}


def architecture(size, **overrides):
    if size not in SIZES:
        raise ValueError(f"size must be one of {tuple(SIZES)}, got {size!r}")
    unknown = overrides.keys() - SIZES[size].keys()
    if unknown:
        raise TypeError(f"unknown architecture parameters: {sorted(unknown)}")
    return {**deepcopy(SIZES[size]), **overrides}


def build_model(config):
    """Construct only the public PULSE class, never an arbitrary YAML class."""
    args = dict(config["architecture"])
    if args.keys() != SIZES[config["size"]].keys():
        raise ValueError("model config must contain only the complete size-dependent parameters")
    return PULSE(**args)


def load_config(path):
    path = Path(path)
    with path.open(encoding="utf-8") as stream:
        config = json.load(stream) if path.suffix == ".json" else yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("configuration must be a mapping")
    return config


def recipe_path(objective, size=None):
    root = Path(__file__).resolve().parents[1] / "configs"
    if not root.is_dir():
        root = files("codec.recipes")
    return root / objective / f"{size}.yaml" if size else root / f"{objective}.yaml"


def validate_recipe(recipe):
    if recipe["objective"] not in TRAINING_SIZES or recipe["size"] not in TRAINING_SIZES[recipe["objective"]]:
        raise ValueError("invalid training size/objective")
    if not recipe.get("stages"):
        raise ValueError("recipe has no stages")
    if recipe["architecture"].keys() != SIZES[recipe["size"]].keys():
        raise ValueError("architecture must contain only the complete size-dependent parameters")
    if recipe["runtime"]["precision"] != "bf16":
        raise ValueError("the source-aligned training recipe uses BF16")
    for stage in recipe["stages"]:
        for key in ("optimizer", "augment", "lr_gamma"):
            if key not in stage:
                raise ValueError(f"stage {stage['name']} is missing {key}")
        if min(stage["crop"], stage["batch"], stage["steps"], stage["lr"]) <= 0 or stage["crop"] % 64:
            raise ValueError("invalid stage crop/batch/steps/learning rate")
        if stage["loss"]["lambda_min"] <= 0 or stage["loss"]["lambda_max"] < stage["loss"]["lambda_min"]:
            raise ValueError("invalid lambda range")
        if any(m <= 0 or m >= stage["steps"] for m in stage["milestones"]):
            raise ValueError("milestones must be strictly within the stage")
        if stage.get("disc_step", 2) < 2:
            raise ValueError("disc_step must leave at least one generator step")
    return recipe
