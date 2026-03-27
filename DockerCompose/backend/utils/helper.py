import os
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn

import monai
from monai.networks.nets import EfficientNetBN
from monai.visualize import GradCAM
from monai.transforms import (
    Compose,
    LoadImageD,
    EnsureChannelFirstD,
    ScaleIntensityRangeD,
    ResizeD,
    EnsureTypeD,
    MapTransform,
)


# ---------------------------------------------------------------------------
# Custom transform
# ---------------------------------------------------------------------------

class ToGrayscaleD(MapTransform):
    """
    Ensures image is 1-channel. If loaded as RGB/multi-channel, converts by
    channel mean. Expects channel-first [C, H, W].
    """
    def __call__(self, data):
        d = dict(data)
        x = d["image"]
        if x.shape[0] > 1:
            x = x.mean(axis=0, keepdims=True)
        d["image"] = x
        return d


# ---------------------------------------------------------------------------
# Low-level helpers (called once at startup, not per request)
# ---------------------------------------------------------------------------

def _build_preprocess(img_size: tuple) -> Compose:
    """Build the deterministic preprocessing pipeline. Call once and reuse."""
    return Compose([
        LoadImageD(keys="image", image_only=True, reader="PILReader"),
        EnsureChannelFirstD(keys="image"),
        ToGrayscaleD(keys="image"),
        ScaleIntensityRangeD(
            keys="image", a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=True
        ),
        ResizeD(keys="image", spatial_size=img_size),
        EnsureTypeD(keys="image", track_meta=False),
    ])


def _find_last_conv_name(model: nn.Module) -> str:
    """Return the name of the last Conv2d layer. Call once and cache."""
    last = None
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            last = name
    if last is None:
        raise RuntimeError("No Conv2d layer found in model.")
    return last


def _disable_inplace_relu(model: nn.Module) -> None:
    for mod in model.modules():
        if isinstance(mod, nn.ReLU):
            mod.inplace = False


def _load_efficientnet(ckpt_path: str, device: torch.device) -> tuple:
    """Load checkpoint, build model, return (model, labels, img_size)."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    labels = ckpt["labels"]
    img_size = tuple(ckpt.get("img_size", (320, 320)))

    model = EfficientNetBN(
        model_name="efficientnet-b0",
        spatial_dims=2,
        in_channels=1,
        pretrained=False,
        num_classes=len(labels),
    ).to(device)

    model.load_state_dict(ckpt["model"])
    _disable_inplace_relu(model)
    model.eval()
    return model, labels, img_size


# ---------------------------------------------------------------------------
# ModelManager — single instance, lives for the lifetime of the process
# ---------------------------------------------------------------------------

class ModelManager:
    """
    Loads and owns the model, preprocessing pipeline, and GradCAM generator.
    All expensive initialization happens once in __init__; infer() is fast.
    """

    def __init__(self, ckpt_path: str, device: torch.device | None = None, amp: bool = True):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        # AMP only makes sense on CUDA
        self._amp = amp and self.device.type == "cuda"

        # --- one-time model load ---
        self.model, self.labels, self.img_size = _load_efficientnet(ckpt_path, self.device)
        self.label_to_idx: dict[str, int] = {l: i for i, l in enumerate(self.labels)}

        # --- one-time preprocessing pipeline ---
        self._preprocess = _build_preprocess(self.img_size)

        # --- cache last conv name, build GradCAM once ---
        self._last_conv = _find_last_conv_name(self.model)
        self._cam_gen = GradCAM(nn_module=self.model, target_layers=self._last_conv)

        # --- CUDA warm-up: compile kernels before first real request ---
        self._warmup()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def infer(self, image_path: str, class_idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run inference + Grad-CAM for one image.

        Returns
        -------
        probs : np.ndarray [C]        sigmoid probabilities for all classes
        cam   : np.ndarray [H, W]     normalised Grad-CAM heatmap (0..1)
        img   : np.ndarray [H, W]     preprocessed input image (0..1)
        """
        x = self._preprocess_image(image_path)  # [1,1,H,W] on device

        # FP16 inference via AMP
        with torch.no_grad():
            with torch.amp.autocast(self.device.type, enabled=self._amp):
                logits = self.model(x)
                probs = torch.sigmoid(logits)[0].cpu().numpy()

        # Grad-CAM requires gradients — run outside no_grad
        cam = self._cam_gen(x, class_idx=class_idx)   # [1,1,h,w]
        cam = cam[0, 0].detach().cpu().numpy()
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        img = x[0, 0].detach().cpu().numpy()          # [H,W] 0..1
        return probs, cam, img

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _preprocess_image(self, image_path: str) -> torch.Tensor:
        """Preprocess a single image to [1,1,H,W] on the target device."""
        sample = {"image": image_path}
        x = self._preprocess(sample)["image"]   # [1,H,W]
        return x.unsqueeze(0).to(self.device)   # [1,1,H,W]

    def _warmup(self) -> None:
        """Forward pass with a dummy tensor to trigger CUDA kernel compilation."""
        dummy = torch.zeros(1, 1, *self.img_size, device=self.device)
        with torch.no_grad():
            with torch.amp.autocast(self.device.type, enabled=self._amp):
                self.model(dummy)


# ---------------------------------------------------------------------------
# Image saving utilities (unchanged API, called per request — cheap ops)
# ---------------------------------------------------------------------------

def save_heatmap(img: np.ndarray, cam: np.ndarray, output_path: str, alpha: float = 0.4) -> None:
    """
    Overlay Grad-CAM heatmap on the original image and save.

    img : [H, W] float32 in [0, 1]
    cam : [h, w] float32 in [0, 1]  (will be upsampled to img size)
    """
    cam_t = torch.tensor(cam)[None, None, ...]
    cam_interp = torch.nn.functional.interpolate(
        cam_t, size=img.shape, mode="bilinear", align_corners=False
    )[0, 0].numpy()

    # Orientation correction (flip vertical then rotate 90° CW)
    img = np.ascontiguousarray(np.rot90(np.flipud(img), k=-1))
    cam_interp = np.ascontiguousarray(np.rot90(np.flipud(cam_interp), k=-1))

    height, width = img.shape
    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
    ax.axis("off")
    ax.imshow(img, cmap="gray", origin="upper")
    ax.imshow(cam_interp, cmap="jet", alpha=alpha, vmin=0, vmax=1.0, origin="upper")
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0, transparent=False)
    plt.close(fig)


def save_image(img: np.ndarray, output_path: str) -> None:
    """Save a preprocessed [H, W] float image to disk."""
    fig, ax = plt.subplots(figsize=(img.shape[1] / 100, img.shape[0] / 100), dpi=100)
    ax.axis("off")
    ax.imshow(img, cmap="gray")
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def clear_temp_storage(temp_dir: str) -> None:
    """Delete all files in temp_dir, creating it if it doesn't exist."""
    if os.path.exists(temp_dir):
        for filename in os.listdir(temp_dir):
            file_path = os.path.join(temp_dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}: {e}")
    else:
        os.makedirs(temp_dir, exist_ok=True)
