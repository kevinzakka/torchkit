"""Tools for visualizing resnet feature maps and ViT attention weights."""

from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from torchvision import transforms as T


def _load_model(
    model_name: str,
    device: torch.device,
):
    if "dino" in model_name:
        assert model_name in [
            "dino_vits16",
            "dino_vits8",
            "dino_vitb16",
            "dino_vitb8",
            "dino_resnet50",
        ], f"{model_name} is not a valid DINO pretrained model."
        model = torch.hub.load("facebookresearch/dino:main", model_name)
        normalize = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    elif "resnet" in model_name:
        model = getattr(torchvision.models, model_name)(pretrained=True)
        normalize = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    model.to(device)
    return model, normalize


def visualize_attention(
    model_name: str,
    image: Union[str, np.ndarray],
    image_size: Tuple[int, int] = (480, 480),
    resnet_layer_idx: Optional[int] = -2,
) -> np.ndarray:
    # Load the model.
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model, normalize = _load_model(model_name, device)

    # Create pre-processing pipeline.
    preprocess = T.Compose(
        [
            T.Resize(image_size),
            T.ToTensor(),
            normalize,
        ]
    )

    if isinstance(image, str):
        img = Image.open(image)
    else:
        img = Image.fromarray(image)
    img_tensor = preprocess(img)

    if "dino" in model_name and "resnet" not in model_name:
        patch_size = model.patch_embed.patch_size
        w = img_tensor.shape[1] - img_tensor.shape[1] % patch_size
        h = img_tensor.shape[2] - img_tensor.shape[2] % patch_size
        img_tensor = img_tensor[:, :w, :h].unsqueeze(0).to(device)
        w_featmap = img_tensor.shape[-2] // patch_size
        h_featmap = img_tensor.shape[-1] // patch_size

        attentions = model.get_last_selfattention(img_tensor)
        nh = attentions.shape[1]

        attentions = attentions[0, :, 0, 1:].reshape(nh, -1)
        attentions = attentions.reshape(nh, w_featmap, h_featmap)

        attentions = (
            nn.functional.interpolate(
                attentions.unsqueeze(0),
                scale_factor=patch_size,
                mode="nearest",
            )[0]
            .cpu()
            .numpy()
        )
    else:
        layers = list(model.children())[:resnet_layer_idx]
        model = nn.Sequential(*layers)

        img_tensor = img_tensor.unsqueeze(0).to(device)
        img_h, img_w = img_tensor.shape[-2], img_tensor.shape[-1]

        with torch.no_grad():
            attentions = model(img_tensor)

        # Average over all feature maps.
        attentions = attentions.mean(dim=1)

        scale_factor_h = img_h / attentions.shape[-2]
        scale_factor_w = img_w / attentions.shape[-1]
        attentions = (
            nn.functional.interpolate(
                attentions.unsqueeze(0),
                scale_factor=(scale_factor_h, scale_factor_w),
                mode="nearest",
            )[0]
            .cpu()
            .numpy()
        )

    return attentions
