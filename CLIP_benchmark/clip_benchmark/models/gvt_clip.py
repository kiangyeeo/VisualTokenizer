import os
import sys
from pathlib import Path
from typing import Optional

import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _ensure_gvt_importable() -> None:
    gvt_root = Path(os.environ.get("GVT_ROOT", _repo_root() / "GVT" / "gvt")).resolve()
    if str(gvt_root) not in sys.path:
        sys.path.insert(0, str(gvt_root))


def _default_gvt_checkpoint() -> str:
    path = _repo_root() / "checkpoints" / "gvt.pth"
    return str(path) if path.exists() else ""


class ProjectionHead(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.expects_tokens = False
        self.norm = nn.LayerNorm(input_dim)
        self.proj = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.norm(x))


class AttentionPoolProjectionHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_query_tokens: int = 1,
        num_heads: int = 8,
        drop_cls_token: bool = True,
    ):
        super().__init__()
        self.expects_tokens = True
        self.drop_cls_token = drop_cls_token
        self.query = nn.Parameter(torch.empty(1, num_query_tokens, input_dim))
        self.attn = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(input_dim)
        self.proj = nn.Linear(input_dim, output_dim)
        nn.init.normal_(self.query, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError(f"Attention pooling expects visual tokens [B, N, D], got {tuple(tokens.shape)}")
        if self.drop_cls_token and tokens.shape[1] > 1:
            tokens = tokens[:, 1:]
        query = self.query.expand(tokens.shape[0], -1, -1)
        pooled, _ = self.attn(query, tokens, tokens, need_weights=False)
        pooled = pooled.mean(dim=1)
        return self.proj(self.norm(pooled))


def pool_tokens(tokens: torch.Tensor, pooling: str = "mean_no_cls") -> torch.Tensor:
    if tokens.ndim == 2:
        return tokens
    if tokens.ndim != 3:
        raise ValueError(f"Expected visual tokens with shape [B, N, D], got {tuple(tokens.shape)}")

    if pooling == "cls":
        return tokens[:, 0]
    if pooling == "mean":
        return tokens.mean(dim=1)
    if pooling == "mean_no_cls":
        return tokens[:, 1:].mean(dim=1) if tokens.shape[1] > 1 else tokens.mean(dim=1)
    raise ValueError("pooling must be one of: cls, mean, mean_no_cls")


def build_gvt_visual_tokenizer(checkpoint_path: Optional[str] = None) -> nn.Module:
    _ensure_gvt_importable()
    from gvt.modules.visual_tokenizers import build_visual_tokenizer

    config = {
        "visual_tokenizer_type": os.environ.get("GVT_TOKENIZER_TYPE", "gvt"),
        "visual_tokenizer_path": checkpoint_path or os.environ.get("GVT_TOKENIZER_PATH", _default_gvt_checkpoint()),
        "visual_tokenizer_dim": int(os.environ.get("GVT_TOKENIZER_DIM", "1024")),
    }
    tokenizer = build_visual_tokenizer(config)
    tokenizer.eval()
    for param in tokenizer.parameters():
        param.requires_grad = False
    return tokenizer


def get_gvt_transform():
    from PIL import Image
    from torchvision.transforms import Compose, Normalize, Resize, ToTensor

    try:
        from torchvision.transforms import InterpolationMode
        bicubic = InterpolationMode.BICUBIC
    except ImportError:
        bicubic = Image.BICUBIC

    size = int(os.environ.get("GVT_IMAGE_SIZE", "224"))
    return Compose(
        [
            Resize((size, size), interpolation=bicubic),
            lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ]
    )


def _clip_text_dim(text_model: nn.Module, tokenizer, device: torch.device) -> int:
    tokenized = tokenizer(["a photo"]).to(device)
    with torch.no_grad():
        return int(text_model.encode_text(tokenized).shape[-1])


def load_projection(
    path: Optional[str],
    input_dim: int,
    output_dim: int,
    device: torch.device,
) -> nn.Module:
    payload = None
    projector_type = os.environ.get("GVT_PROJECTOR_TYPE", "linear")
    num_query_tokens = int(os.environ.get("GVT_NUM_QUERY_TOKENS", "1"))
    num_heads = int(os.environ.get("GVT_ATTN_HEADS", "8"))
    drop_cls_token = os.environ.get("GVT_DROP_CLS_TOKEN", "1") != "0"

    if path:
        payload = torch.load(path, map_location="cpu")
        if isinstance(payload, dict):
            projector_type = payload.get("projector_type", projector_type)
            num_query_tokens = int(payload.get("num_query_tokens", num_query_tokens))
            num_heads = int(payload.get("attn_heads", num_heads))
            drop_cls_token = bool(payload.get("drop_cls_token", drop_cls_token))

    if projector_type == "attention":
        projector = AttentionPoolProjectionHead(
            input_dim=input_dim,
            output_dim=output_dim,
            num_query_tokens=num_query_tokens,
            num_heads=num_heads,
            drop_cls_token=drop_cls_token,
        )
    elif projector_type == "linear":
        projector = ProjectionHead(input_dim=input_dim, output_dim=output_dim)
    else:
        raise ValueError("GVT_PROJECTOR_TYPE must be 'linear' or 'attention'")

    if path:
        state_dict = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
        projector.load_state_dict(state_dict)
    else:
        print(
            "WARNING: GVT_PROJECTION_PATH is not set. "
            "Using a randomly initialized projection head; scores are only useful for smoke tests."
        )
    return projector.to(device)


class GVTCLIPModel(nn.Module):
    def __init__(
        self,
        visual_tokenizer: nn.Module,
        projector: nn.Module,
        text_model: nn.Module,
        pooling: str = "mean_no_cls",
    ):
        super().__init__()
        self.visual_tokenizer = visual_tokenizer
        self.projector = projector
        self.text_model = text_model
        self.pooling = pooling

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        tokens = self.visual_tokenizer(images)
        if getattr(self.projector, "expects_tokens", False):
            features = self.projector(tokens)
        else:
            pooled = pool_tokens(tokens, self.pooling)
            features = self.projector(pooled)
        return F.normalize(features, dim=-1)

    def encode_text(self, tokenized_texts: torch.Tensor) -> torch.Tensor:
        return self.text_model.encode_text(tokenized_texts)


def load_gvt_clip(
    model_name: str = "ViT-B-32-quickgelu",
    pretrained: str = "laion400m_e32",
    cache_dir: str = None,
    device="cpu",
):
    device = torch.device(device)
    text_model, _, _ = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
        cache_dir=cache_dir,
    )
    text_model = text_model.to(device).eval()
    for param in text_model.parameters():
        param.requires_grad = False

    text_tokenizer = open_clip.get_tokenizer(model_name)
    output_dim = _clip_text_dim(text_model, text_tokenizer, device)

    visual_tokenizer = build_gvt_visual_tokenizer().to(device)
    input_dim = int(getattr(visual_tokenizer, "output_dim", os.environ.get("GVT_TOKENIZER_DIM", 1024)))
    projection_path = os.environ.get("GVT_PROJECTION_PATH", "")
    projector = load_projection(projection_path, input_dim, output_dim, device)

    model = GVTCLIPModel(
        visual_tokenizer=visual_tokenizer,
        projector=projector,
        text_model=text_model,
        pooling=os.environ.get("GVT_POOLING", "mean_no_cls"),
    ).to(device)
    return model, get_gvt_transform(), text_tokenizer
