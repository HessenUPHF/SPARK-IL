import json
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from .model import DualSpectralViT_KAN


class SparkILEncoder:
    def __init__(self, model, device):
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.model.requires_grad_(False)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    @classmethod
    def from_pretrained(cls, checkpoint_path, config_path=None, device=None):
        checkpoint_path = Path(checkpoint_path)
        if checkpoint_path.is_dir():
            checkpoint_path = checkpoint_path / "model1.pth"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        config_path = Path(config_path) if config_path else checkpoint_path.with_name("config1.json")
        with config_path.open() as stream:
            config = json.load(stream)
        parameters = {key: config[key] for key in ("embed_dim", "num_heads", "n_bands")}
        if parameters["embed_dim"] != 768:
            raise ValueError("The released ViT-L/14 encoder requires embed_dim=768.")
        if parameters["num_heads"] <= 0 or 768 % parameters["num_heads"]:
            raise ValueError("num_heads must be a positive divisor of 768.")
        if not 1 <= parameters["n_bands"] <= 768:
            raise ValueError("n_bands must be between 1 and 768.")
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model = DualSpectralViT_KAN(**parameters, pretrained=None)
        model.load_state_dict(state_dict, strict=True)
        del state_dict
        return cls(model, device)

    @torch.inference_mode()
    def encode(self, images, batch_size=8):
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        if isinstance(images, (str, Path, Image.Image)):
            images = [images]
        else:
            images = list(images)
        if not images:
            return torch.empty((0, 768), dtype=torch.float32)
        batches = []
        self.model.eval()
        for start in range(0, len(images), batch_size):
            tensors = []
            for item in images[start:start + batch_size]:
                if isinstance(item, Image.Image):
                    tensors.append(self.transform(item.convert("RGB")))
                else:
                    with Image.open(item) as image:
                        tensors.append(self.transform(image.convert("RGB")))
            batch = torch.stack(tensors).to(self.device)
            embeddings = self.model(batch, return_fft=True)
            batches.append(embeddings.float().cpu())
        return torch.cat(batches, dim=0)
