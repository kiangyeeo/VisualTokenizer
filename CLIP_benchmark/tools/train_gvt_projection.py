import argparse
import os
from pathlib import Path

import open_clip
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from clip_benchmark.datasets.builder import build_dataset
from clip_benchmark.models.gvt_clip import (
    AttentionPoolProjectionHead,
    ProjectionHead,
    build_gvt_visual_tokenizer,
    get_gvt_transform,
    pool_tokens,
)


class DualTransformDataset(Dataset):
    def __init__(self, dataset, gvt_transform, clip_transform):
        self.dataset = dataset
        self.gvt_transform = gvt_transform
        self.clip_transform = clip_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        image = sample[0]
        return self.gvt_transform(image), self.clip_transform(image)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a lightweight projection from GVT visual tokens to OpenCLIP image embedding space."
    )
    parser.add_argument("--dataset", default="cifar10")
    parser.add_argument("--dataset-root", default="/cache/ma-user/VisualTokenizer/data/clip_benchmark")
    parser.add_argument("--split", default="train")
    parser.add_argument("--task", default="zeroshot_classification")
    parser.add_argument("--openclip-model", default="ViT-B-32-quickgelu")
    parser.add_argument("--openclip-pretrained", default="laion400m_e32")
    parser.add_argument("--model-cache-dir", default="/cache/ma-user/VisualTokenizer/checkpoints/open_clip")
    parser.add_argument("--gvt-checkpoint", default="/cache/ma-user/VisualTokenizer/checkpoints/gvt.pth")
    parser.add_argument("--output", default="/cache/ma-user/VisualTokenizer/checkpoints/gvt_clip_projection.pt")
    parser.add_argument("--projector-type", default="attention", choices=["linear", "attention"])
    parser.add_argument("--pooling", default="mean_no_cls", choices=["cls", "mean", "mean_no_cls"])
    parser.add_argument("--num-query-tokens", type=int, default=1)
    parser.add_argument("--attn-heads", type=int, default=8)
    parser.add_argument("--keep-cls-token", action="store_true")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=0, help="Use a prefix subset for quick smoke tests. 0 means full split.")
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (not args.no_amp) and device.type == "cuda"

    os.environ["GVT_TOKENIZER_PATH"] = args.gvt_checkpoint

    target_model, _, clip_transform = open_clip.create_model_and_transforms(
        args.openclip_model,
        pretrained=args.openclip_pretrained,
        cache_dir=args.model_cache_dir,
    )
    target_model = target_model.to(device).eval()
    for param in target_model.parameters():
        param.requires_grad = False

    visual_tokenizer = build_gvt_visual_tokenizer(args.gvt_checkpoint).to(device).eval()
    for param in visual_tokenizer.parameters():
        param.requires_grad = False

    gvt_transform = get_gvt_transform()
    base_dataset = build_dataset(
        dataset_name=args.dataset,
        root=args.dataset_root,
        transform=None,
        split=args.split,
        download=True,
        task=args.task,
    )
    if args.max_samples > 0:
        base_dataset = Subset(base_dataset, range(min(args.max_samples, len(base_dataset))))
    dataset = DualTransformDataset(base_dataset, gvt_transform, clip_transform)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    input_dim = int(getattr(visual_tokenizer, "output_dim", 1024))
    with torch.no_grad():
        sample = next(iter(dataloader))[1][:1].to(device)
        output_dim = int(target_model.encode_image(sample).shape[-1])

    if args.projector_type == "attention":
        projector = AttentionPoolProjectionHead(
            input_dim=input_dim,
            output_dim=output_dim,
            num_query_tokens=args.num_query_tokens,
            num_heads=args.attn_heads,
            drop_cls_token=not args.keep_cls_token,
        ).to(device)
    else:
        projector = ProjectionHead(input_dim=input_dim, output_dim=output_dim).to(device)
    optimizer = torch.optim.AdamW(projector.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(args.epochs):
        projector.train()
        total_loss = 0.0
        total_count = 0
        progress = tqdm(dataloader, desc=f"epoch {epoch + 1}/{args.epochs}")
        for gvt_images, clip_images in progress:
            gvt_images = gvt_images.to(device, non_blocking=True)
            clip_images = clip_images.to(device, non_blocking=True)

            with torch.no_grad(), torch.autocast(device_type=device.type, enabled=use_amp):
                target = F.normalize(target_model.encode_image(clip_images), dim=-1)
                tokens = visual_tokenizer(gvt_images)

            with torch.autocast(device_type=device.type, enabled=use_amp):
                if getattr(projector, "expects_tokens", False):
                    pred = F.normalize(projector(tokens), dim=-1)
                else:
                    pred = F.normalize(projector(pool_tokens(tokens, args.pooling)), dim=-1)
                loss = 1.0 - (pred * target).sum(dim=-1).mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            batch_size = gvt_images.shape[0]
            total_loss += float(loss.detach()) * batch_size
            total_count += batch_size
            progress.set_postfix(loss=total_loss / max(total_count, 1))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": projector.state_dict(),
            "projector_type": args.projector_type,
            "input_dim": input_dim,
            "output_dim": output_dim,
            "pooling": args.pooling,
            "num_query_tokens": args.num_query_tokens,
            "attn_heads": args.attn_heads,
            "drop_cls_token": not args.keep_cls_token,
            "dataset": args.dataset,
            "split": args.split,
            "openclip_model": args.openclip_model,
            "openclip_pretrained": args.openclip_pretrained,
            "gvt_checkpoint": args.gvt_checkpoint,
        },
        output,
    )
    print(f"Saved projection to {output}")


if __name__ == "__main__":
    main()
