"""Build the paired ImageNet/INVAE layout from priml's extracted ImageNet source.

This is a straightforward single-device preparer. It uses the existing
ImageNet source and synset labels; the reference's preprocessed dataset can
also be used directly without running this command.
"""

from __future__ import annotations

from pathlib import Path

import argparse
import json

from PIL import Image

import numpy as np
import torch

from priml.data.processors.labels import ImagenetSynsetToIndex
from priml.data.sources.extracted_imagenet import ExtractedImageNetSource
from priml.model.third_party.invae import encode_image, load_invae


def center_crop(image: Image.Image, size: int) -> Image.Image:
    """Dhariwal-style center crop: box downsample, bicubic resize, center cut."""
    while min(image.size) >= 2 * size:
        image = image.resize(
            (image.width // 2, image.height // 2), Image.Resampling.BOX
        )
    scale = size / min(image.size)
    image = image.resize(
        (round(image.width * scale), round(image.height * scale)),
        Image.Resampling.BICUBIC,
    )
    left = (image.width - size) // 2
    top = (image.height - size) // 2
    return image.crop((left, top, left + size, top + size))


def prepare(
    source: Path,
    output: Path,
    *,
    resolution: int = 256,
    checkpoint: Path | None = None,
    device: str = "cuda",
    limit: int | None = None,
) -> int:
    """Write PNGs, sampled 32-channel INVAE arrays, and dataset.json labels."""
    if resolution not in (256, 512):
        raise ValueError("resolution must be 256 or 512")
    source_config = ExtractedImageNetSource.Config(working_dir=source, split="train")
    image_source = source_config.make()
    labels = ImagenetSynsetToIndex.Config().make()
    vae = load_invae(checkpoint, device=device)
    metadata: list[list[str | int]] = []
    for index, record in enumerate(labels(iter(image_source))):
        if limit is not None and index >= limit:
            break
        sample = dict(record)
        class_id = sample.get("label")
        if not isinstance(class_id, int):
            raise TypeError(f"ImageNet synset was not mapped to an integer: {class_id}")
        file_path = sample.get("file_path")
        if not isinstance(file_path, str):
            raise TypeError(f"ImageNet record has no file path: {file_path}")
        relative = Path(f"{index // 1000:05d}") / f"img{index:08d}"
        image_path = output / "images" / relative.with_suffix(".png")
        latent_path = output / "vae-in" / relative.with_suffix(".npy")
        image_path.parent.mkdir(parents=True, exist_ok=True)
        latent_path.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(file_path) as opened:
            cropped = center_crop(opened.convert("RGB"), resolution)
        cropped.save(image_path)
        image = (
            torch.from_numpy(np.asarray(cropped).copy())
            .permute(2, 0, 1)[None]
            .to(device)
        )
        latent = encode_image(vae, image).cpu().numpy()
        np.save(latent_path, latent)
        metadata.append(
            [latent_path.relative_to(output / "vae-in").as_posix(), class_id]
        )
    (output / "vae-in" / "dataset.json").write_text(
        json.dumps({"labels": metadata}), encoding="utf-8"
    )
    return len(metadata)


def main() -> None:
    """Parse preparation options and write the processed dataset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, required=True, help="Extracted ImageNet root"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Processed dataset root"
    )
    parser.add_argument("--resolution", type=int, default=256, choices=(256, 512))
    parser.add_argument("--checkpoint", type=Path, help="Local e2e-invae-400k.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, help="Prepare only this many images")
    args = parser.parse_args()
    print(
        prepare(
            args.source,
            args.output,
            resolution=args.resolution,
            checkpoint=args.checkpoint,
            device=args.device,
            limit=args.limit,
        )
    )


if __name__ == "__main__":
    main()
