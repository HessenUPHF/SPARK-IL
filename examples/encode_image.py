import argparse
from pathlib import Path

import torch

from spark_il import SparkILEncoder


def main():
    parser = argparse.ArgumentParser(description="Extract SPARK-IL fused image embeddings.")
    parser.add_argument("--checkpoint", default="checkpoints/model1.pth")
    parser.add_argument("--config", default=None)
    parser.add_argument("--images", nargs="+", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", default="embeddings.pt")
    args = parser.parse_args()
    encoder = SparkILEncoder.from_pretrained(args.checkpoint, args.config, args.device)
    embeddings = encoder.encode(args.images, batch_size=args.batch_size)
    if embeddings.shape != (len(args.images), 768) or not torch.isfinite(embeddings).all():
        raise RuntimeError("Expected finite embeddings with shape [number of images, 768].")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(embeddings, output)
    print(f"Checkpoint loaded successfully. Embedding shape: {tuple(embeddings.shape)}")
    print(f"Saved raw fused embeddings to {output}")


if __name__ == "__main__":
    main()
