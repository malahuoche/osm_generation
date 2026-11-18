"""Small CPU inference bridge to pinned, unmodified official MIA sources.

Only the package initializers that register training frameworks/datasets are
skipped. The network, camera, projection, and image transforms execute the
original source files. No substitute network or trained parameter is used.
"""

from pathlib import Path
import contextlib
import hashlib
import io
import logging
import os
import sys
import types

ROOT = Path(os.environ.get("BEV_OSM_CACHE", str(Path.home() / ".cache/bev-osm")))
SOURCE = Path(os.environ.get("MIA_REPO", "third_party/MapItAnywhere")).resolve()
COMMIT = "6dbfaa569c7f384204a9c4c5b915414c1c87451f"
WEIGHT_SHA256 = "51a063fa03c24984bcf4bbd15d64011d83f72af209727bc6a5c3caee5aea2735"
CLASSES = ["road", "crossing", "sidewalk", "building", "terrain", "parking"]
GT_CHANNELS = [0, 1, 2, 4, 6, 7]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def setup_imports():
    if not (SOURCE / "mapper").is_dir():
        raise FileNotFoundError(f"Set MIA_REPO to the official checkout: {SOURCE}")
    os.environ.setdefault("XFORMERS_DISABLED", "1")
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
    for name, path in [
        ("mapper", SOURCE / "mapper"),
        ("mapper.data", SOURCE / "mapper/data"),
    ]:
        if name not in sys.modules:
            module = types.ModuleType(name)
            module.__path__ = [str(path)]
            module.__file__ = str(path / "__init__.py")
            module.__package__ = name
            sys.modules[name] = module
    sys.modules["mapper"].logger = logging.getLogger("mapper")
    sys.modules["mapper"].repo_dir = SOURCE
    sys.modules["mapper"].EXPERIMENTS_PATH = SOURCE / "experiments"
    sys.modules["mapper"].DATASETS_PATH = SOURCE / "datasets"


def model_config():
    # Resolve the official pretrain YAML without importing the training runner.
    from omegaconf import OmegaConf

    setup_imports()
    from mapper.models.schema import ModelConfiguration

    conf_dir = SOURCE / "mapper/conf"
    model = OmegaConf.load(conf_dir / "model/mapper.yaml")
    del model["defaults"]
    encoder = OmegaConf.load(conf_dir / "model/image_encoder/dino.yaml")
    pretrain = OmegaConf.load(conf_dir / "pretrain.yaml")
    root = OmegaConf.create({"data": {"pixel_per_meter": 2}, "model": model})
    root.model.image_encoder = encoder
    root.model.loss = OmegaConf.merge(root.model.loss, pretrain.model.loss)
    resolved = OmegaConf.to_container(root.model, resolve=True)
    config = OmegaConf.merge(OmegaConf.structured(ModelConfiguration), resolved)
    config.image_encoder.backbone.pretrained = False
    return config


def preprocess(image_rgb, camera_dict, roll_pitch):
    """Official MIA dataset path: gravity alignment, longest side 512, pad."""
    import numpy as np
    import torch

    setup_imports()
    from mapper.utils.wrappers import Camera
    from mapper.data.image import rectify_image, resize_image, pad_image

    camera = Camera.from_dict(camera_dict).float()
    image = (
        torch.from_numpy(np.ascontiguousarray(image_rgb)).permute(2, 0, 1).float() / 255
    )
    if tuple(image.shape[-2:][::-1]) != (camera_dict["width"], camera_dict["height"]):
        raise ValueError("Image dimensions disagree with camera metadata")
    roll, pitch = map(float, roll_pitch)
    image, valid = rectify_image(image, camera, roll, pitch)
    image, _, camera, valid = resize_image(
        image, 512, fn=max, camera=camera, valid=valid
    )
    # Upstream pad_image prints debug shapes. Silence only these prints.
    with contextlib.redirect_stdout(io.StringIO()):
        image, valid, camera = pad_image(image, 512, camera, valid)
    return {"image": image[None], "valid": valid[None], "camera": torch.stack([camera])}


def read_array(path):
    import numpy as np

    value = np.load(path, allow_pickle=False)
    if isinstance(value, np.lib.npyio.NpzFile):
        with value as archive:
            if len(archive.files) != 1:
                raise ValueError(f"Expected one array in {path}")
            return archive[archive.files[0]]
    return value


def load_gt(semantic_path, flood_path):
    """Official front 50x50 m crop and six-channel selection; npz supported."""
    import numpy as np

    semantic = read_array(semantic_path)
    flood = read_array(flood_path)
    if (
        semantic.ndim != 3
        or semantic.shape[-1] < 8
        or flood.shape != semantic.shape[:2]
    ):
        raise ValueError("Unexpected source label or flood-mask shape")
    h, w = semantic.shape[:2]
    cy, cx = h // 2, w // 2
    if cy < 100 or cx < 50:
        raise ValueError("Source GT is too small for the official crop")
    selection = np.s_[cy - 100 : cy, cx - 50 : cx + 50]
    labels = semantic[selection][..., GT_CHANNELS]
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("GT must contain binary layers")
    crop = flood[selection].astype(np.float32)
    confidence = (crop - crop.min()) / (crop.max() - crop.min() + 1e-6)
    return labels.astype(bool), confidence, crop


def load_model(checkpoint, threads=4):
    import collections
    import typing
    import torch
    from omegaconf import DictConfig, ListConfig, OmegaConf
    from omegaconf.base import Metadata, ContainerMetadata
    from omegaconf.nodes import AnyNode

    setup_imports()
    from mapper.models import get_model

    checkpoint = Path(checkpoint)
    digest = sha256(checkpoint)
    if digest != WEIGHT_SHA256:
        raise ValueError("Checkpoint differs from verified author release")
    torch.set_num_threads(threads)
    torch.manual_seed(0)
    # Only allow the specific configuration types present in this known file.
    safe = [
        DictConfig,
        ListConfig,
        Metadata,
        ContainerMetadata,
        AnyNode,
        int,
        dict,
        list,
        typing.Any,
        collections.defaultdict,
    ]
    with torch.serialization.safe_globals(safe):
        archive = torch.load(
            checkpoint, map_location="cpu", weights_only=True, mmap=True
        )
    config = model_config()
    hp = archive["hyper_parameters"]
    hp = OmegaConf.to_container(hp, resolve=True) if OmegaConf.is_config(hp) else hp
    for name in [
        "num_classes",
        "latent_dim",
        "z_max",
        "x_max",
        "pixel_per_meter",
        "num_scale_bins",
    ]:
        if hp["model"][name] != config[name]:
            raise ValueError(f"Checkpoint/model config mismatch: {name}")
    for name, expected in [
        ("resize_image", 512),
        ("pad_to_square", True),
        ("rectify_pitch", True),
        ("gravity_align", True),
    ]:
        if hp["data"][name] != expected:
            raise ValueError(f"Checkpoint/preprocessing mismatch: {name}")
    if any(not k.startswith("model.") for k in archive["state_dict"]):
        raise ValueError("Unrecognized non-model checkpoint parameters")
    state = {k.removeprefix("model."): v for k, v in archive["state_dict"].items()}
    model = get_model("map_perception_net")(config)
    # Published checkpoint contains an old confidence head absent from the
    # current official SegmentationHead. Official custom inference uses
    # strict=False. Here allow only these three known extras, then strictly
    # match every parameter used by the current official inference network.
    legacy_names = {
        "decoder.confidence_head.0.weight",
        "decoder.confidence_head.3.weight",
        "decoder.confidence_head.3.bias",
    }
    unexpected = set(state) - set(model.state_dict())
    if unexpected != legacy_names:
        raise ValueError(
            f"Unexpected checkpoint/source difference: {sorted(unexpected)}"
        )
    legacy = {k: list(state.pop(k).shape) for k in sorted(legacy_names)}
    keys = model.load_state_dict(state, strict=True)
    model = model.cpu().eval()
    report = {
        "checkpoint_sha256": digest,
        "checkpoint_bytes": checkpoint.stat().st_size,
        "official_commit": COMMIT,
        "strict_load": True,
        "state_tensors": len(state),
        "missing_keys": list(keys.missing_keys),
        "unexpected_keys": list(keys.unexpected_keys),
        "published_state_tensors": len(archive["state_dict"]),
        "excluded_legacy_confidence_head": legacy,
        "compatibility_note": "Current official inference code has no confidence head; only these three legacy tensors are excluded, all active parameters load strictly.",
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "epoch": archive.get("epoch"),
        "global_step": archive.get("global_step"),
        "config": OmegaConf.to_container(config, resolve=True),
        "checkpoint_hyperparameters": hp,
        "torch_version": torch.__version__,
        "device": "cpu",
        "threads": threads,
        "seed": 0,
        "initializers_skipped": ["mapper", "mapper.data"],
        "source_network_and_transforms_modified": False,
    }
    return model, report
