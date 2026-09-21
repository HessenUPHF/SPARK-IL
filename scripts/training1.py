import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import math
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
import json
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import open_clip
import copy
import torch.utils.data
from torch.utils.data import TensorDataset

class KANLinear(nn.Module):
    def __init__(self, in_features, out_features, num_experts=4, dropout=0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_experts = num_experts
        

        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        

        expert_hidden = max(4, out_features // 2)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features, expert_hidden),
                nn.SiLU(),
                nn.Linear(expert_hidden, out_features)
            ) for _ in range(num_experts)
        ])
        

        self.expert_weights = nn.Linear(in_features, num_experts)
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_features)
        
        self.reset_parameters()

    def reset_parameters(self):
        if self.weight.numel() > 0:
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None and self.bias.numel() > 0:
            fan_in = self.in_features if self.in_features > 0 else 1
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        if x.numel() == 0:
            return torch.zeros(x.shape[0], self.out_features, device=x.device)
            
        base_out = F.linear(x, self.weight, self.bias)
        expert_weights = F.softmax(self.expert_weights(x), dim=-1)
        
        expert_outputs = []
        for expert in self.experts:
            expert_out = expert(x)
            expert_outputs.append(expert_out.unsqueeze(-1))
        
        expert_outputs = torch.cat(expert_outputs, dim=-1)
        expert_weights = expert_weights.unsqueeze(1)
        mixed_expert_out = torch.sum(expert_outputs * expert_weights, dim=-1)
        
        out = base_out + mixed_expert_out
        return self.norm(out)

class SwiGLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.w1 = nn.Linear(dim, dim)
        self.w2 = nn.Linear(dim, dim)
    def forward(self, x):
        return F.silu(self.w1(x)) * self.w2(x)

class MultiBandFFTBlock(nn.Module):
    def __init__(self, dim, n_bands):
        super().__init__()
        self.dim = dim
        self.n_bands = n_bands
        
        base_size = dim // n_bands
        remainder = dim % n_bands
        sizes = [base_size + 1 if i < remainder else base_size for i in range(n_bands)]
        
        total = sum(sizes)
        if total != dim:
            sizes[-1] += (dim - total)
        
        print(f" FFT Block: {dim} dim to {sizes} band sizes ({n_bands} bands)")
        
        self.band_slices = []
        start = 0
        for size in sizes:
            end = start + size
            self.band_slices.append(slice(start, end))
            start = end
        
        self.band_kan = nn.ModuleList([KANLinear(size, size) for size in sizes])
        self.band_act = nn.ModuleList([SwiGLU(size) for size in sizes])
        self.fuse = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        if x.dim() != 2:
            raise ValueError(f"Expected 2D input, got {x.dim()}D")
        
        batch_size, seq_len = x.shape
        if seq_len != self.dim:
            if seq_len < self.dim:
                x = F.pad(x, (0, self.dim - seq_len))
            else:
                x = x[:, :self.dim]
        
        Fz = torch.fft.fft(x, dim=-1)
        mag = torch.abs(Fz)
        mag = torch.log1p(mag)
        
        band_outputs = []
        for sl, kan, act in zip(self.band_slices, self.band_kan, self.band_act):
            band_data = mag[:, sl]
            processed_band = act(kan(band_data))
            band_outputs.append(processed_band)
        
        z = torch.cat(band_outputs, dim=-1)
        return self.fuse(self.norm(z))

class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, q, kv):
        B, Nq, D = q.size()
        _, Nk, _ = kv.size()
        
        residual = q.squeeze(1)
        
        q = self.q_proj(q).reshape(B, Nq, self.num_heads, D // self.num_heads).transpose(1, 2)
        k = self.k_proj(kv).reshape(B, Nk, self.num_heads, D // self.num_heads).transpose(1, 2)
        v = self.v_proj(kv).reshape(B, Nk, self.num_heads, D // self.num_heads).transpose(1, 2)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, Nq, D)
        out = self.out_proj(out)
        
        out = out.squeeze(1) + residual
        return self.norm(out)

class DualSpectralViT_KAN(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12, n_bands=4):
        super().__init__()
        
        print(" Loading ViT-L/14 from OpenCLIP...")
        self.vit, _, _ = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')
        

        frozen_params = 0
        unfrozen_params = 0
        for name, param in self.vit.named_parameters():
            if 'visual.transformer.resblocks.22' in name or \
               'visual.transformer.resblocks.23' in name:
                param.requires_grad = True
                unfrozen_params += param.numel()
            else:
                param.requires_grad = False
                frozen_params += param.numel()
            
        print(f" Loaded ViT-L/14 (OpenCLIP) - Partially Frozen")
        print(f"   Frozen ViT params: {frozen_params:,}")
        print(f"   Unfrozen ViT params (last 2 blocks): {unfrozen_params:,}")
        print(f" ViT-L/14 output dimension: {self.vit.visual.output_dim}")
        

        self.rgb_proj = nn.Linear(3*224*224, embed_dim)
        

        self.fft_rgb = nn.Sequential(
            MultiBandFFTBlock(embed_dim, n_bands),
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            MultiBandFFTBlock(embed_dim, n_bands)
        )
        
        self.fft_feat = nn.Sequential(
            MultiBandFFTBlock(embed_dim, n_bands),
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            MultiBandFFTBlock(embed_dim, n_bands)
        )
        
        self.cross_attn = CrossAttentionBlock(embed_dim, num_heads)
        
        self.proj_head = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, embed_dim),
        )
        
        self.cls = nn.Linear(embed_dim, 1)
        nn.init.normal_(self.cls.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.cls.bias, 0.1)
        print(f" Classifier initialized with bias: {self.cls.bias.item():.4f}")

    def forward(self, x, return_fft=False, return_cross_attention_features=False):
        B = x.size(0)
        

        vit_embeds = self.vit.encode_image(x)
        

        rgb_flat = x.flatten(1)
        rgb_projected = self.rgb_proj(rgb_flat)
        z_fft1 = self.fft_rgb(rgb_projected)
        z_fft2 = self.fft_feat(vit_embeds)

        cross_fused = self.cross_attn(z_fft1.unsqueeze(1), z_fft2.unsqueeze(1))


        fused = cross_fused + 0.2 * z_fft1 + 0.2 * z_fft2
        
        proj = self.proj_head(fused)
        

        logits = self.cls(proj)
        

        if return_cross_attention_features:
            return fused, z_fft1, z_fft2
        if return_fft:
            return fused
        
        return logits, proj, proj

def set_seed(seed=42):
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def build_loader(data_path, img_size=224, bs=16, val_split=0.2, seed=42, num_workers=4):
    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    ds = datasets.ImageFolder(data_path, transform=tfm)
    

    class_counts = {}
    for _, label in ds:
        class_counts[label] = class_counts.get(label, 0) + 1
    print(f" Class distribution: {class_counts}")
    
    n_val = max(1, int(len(ds) * val_split))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(seed))
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=True,
                            num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader

def save_full_model1(model, save_dir, cfg):
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, "model1.pth"))
    with open(os.path.join(save_dir, "config1.json"), "w") as f:
        json.dump(cfg, f, indent=2)

class ReplayBuffer:
    def __init__(self, max_samples_per_tech=300):
        self.max_samples_per_tech = max_samples_per_tech
        self.buffer = {}
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def add_technique_samples(self, tech_name, dataset):
        """Add 300 samples (150 real + 150 fake) from a technique to buffer"""
        print(f" Building replay buffer for {tech_name}...")
        
        real_paths, real_labels = [], []
        fake_paths, fake_labels = [], []
        

        for idx in range(len(dataset)):
            path, label = dataset.samples[idx]
            if label == 0 and len(real_paths) < 150:
                real_paths.append(path)
                real_labels.append(0)
            elif label == 1 and len(fake_paths) < 150:
                fake_paths.append(path)
                fake_labels.append(1)
            
            if len(real_paths) >= 150 and len(fake_paths) >= 150:
                break
        

        if real_paths and fake_paths:
            all_paths = real_paths + fake_paths
            all_labels = real_labels + fake_labels
            
            self.buffer[tech_name] = (all_paths, all_labels)
            print(f" Added {len(all_paths)} samples ({len(real_paths)} real, {len(fake_paths)} fake) to replay buffer for {tech_name}")
        else:
            print(f" Could not collect balanced samples for {tech_name}")
    
    def get_replay_batch(self, current_tech, batch_size=32, device="cuda"):
        """Get a mixed replay batch from all previous techniques"""
        replay_imgs, replay_lbls = [], []
        
        for tech_name, (paths, labels) in self.buffer.items():
            if tech_name != current_tech:

                indices = np.random.choice(len(paths), size=min(8, len(paths)), replace=False)
                
                for idx in indices:
                    try:

                        img = self.load_image(paths[idx])
                        replay_imgs.append(img)
                        replay_lbls.append(labels[idx])
                    except Exception as e:
                        continue
        
        if replay_imgs and len(replay_imgs) >= 16:
            replay_imgs = torch.stack(replay_imgs).to(device)
            replay_lbls = torch.tensor(replay_lbls, device=device).float().unsqueeze(1)
            return replay_imgs, replay_lbls
        return None, None
    
    def load_image(self, path):
        """Load and transform single image"""
        img = datasets.folder.default_loader(path)
        return self.transform(img)

def train_incremental1(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("\n GPU Configuration:")
    if torch.cuda.is_available():
        print(f"   Found {torch.cuda.device_count()} GPU(s):")
        for i in range(torch.cuda.device_count()):
            print(f"   - GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"     Memory: {torch.cuda.get_device_properties(i).total_memory/1024/1024/1024:.1f} GB")
    else:
        print("   No GPU available, running on CPU")
    print("")
    
    set_seed(args.seed)
    
    techniques = [t.strip() for t in args.techniques.split(",")]
    
    print(" Initializing model with PARTIALLY FROZEN OpenCLIP ViT-L/14...")
    model = DualSpectralViT_KAN(
        embed_dim=768,
        num_heads=12,
        n_bands=args.bands
    ).to(device)
    

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f" Total parameters: {total_params:,}")
    print(f" Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
    

    trainable_params_list = [p for p in model.parameters() if p.requires_grad]
    
    criterion = nn.BCEWithLogitsLoss()
    

    replay_buffer = ReplayBuffer(max_samples_per_tech=300)
    teacher_model = None
    technique_stats = []

    print(f" Starting FIXED incremental training on {len(techniques)} techniques:")
    print(f"   Techniques: {techniques}")

    for tech_idx, tech in enumerate(techniques):
        print(f"\n [{tech_idx+1}/{len(techniques)}] Training on technique: {tech}")
        
        data_path = os.path.join(args.data_root, tech)
        if not os.path.exists(data_path):
            print(f" Technique path not found: {data_path}")
            continue
            
        print(f" Loading data from: {data_path}")
        

        tfm = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        full_dataset = datasets.ImageFolder(data_path, transform=tfm)
        
        train_loader, val_loader = build_loader(
            data_path, args.img_size, args.bs,
            args.val_split, args.seed, args.num_workers
        )
        
        print(f" Training samples: {len(train_loader.dataset)}, Validation samples: {len(val_loader.dataset)}")
        

        if tech_idx > 0:
            teacher_model = copy.deepcopy(model)
            teacher_model.eval()
            for param in teacher_model.parameters():
                param.requires_grad = False
            print(" Created NEW frozen teacher model for distillation")
        

        optimizer = torch.optim.AdamW(
            trainable_params_list, 
            lr=args.lr,
            weight_decay=5e-5
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

        tech_best_val_acc = 0.0
        tech_best_f1 = 0.0


        for epoch in range(1, args.epochs + 1):

            model.train()
            total_loss, correct, total = 0, 0, 0
            total_cls_loss, total_distill_loss = 0, 0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]")
            
            for batch_idx, (imgs, lbls) in enumerate(pbar):
                imgs = imgs.to(device)
                y = lbls.float().unsqueeze(1).to(device)
                

                mixed_imgs, mixed_lbls = imgs, y
                if tech_idx > 0 and batch_idx % 2 == 0:
                    replay_imgs, replay_lbls = replay_buffer.get_replay_batch(tech, batch_size=32, device=device)
                    if replay_imgs is not None:

                        mixed_imgs = torch.cat([imgs, replay_imgs])
                        mixed_lbls = torch.cat([y, replay_lbls])
                
                optimizer.zero_grad()
                

                logits, embeddings, raw_proj = model(mixed_imgs)
                

                loss_cls = criterion(logits, mixed_lbls)
                

                loss_distill = 0.0
                if teacher_model is not None:
                    with torch.no_grad():
                        teacher_logits, teacher_embeddings, _ = teacher_model(imgs)
                    

                    loss_embed = F.mse_loss(embeddings[:len(imgs)], teacher_embeddings)
                    

                    loss_logits = F.kl_div(
                        F.log_softmax(logits[:len(imgs)], dim=-1),
                        F.softmax(teacher_logits, dim=-1),
                        reduction='batchmean'
                    )
                    

                    loss_distill = 2.0 * loss_embed + 0.3 * loss_logits
                

                loss_reg = 0.0
                for param in trainable_params_list:
                    loss_reg += param.norm(2)
                loss_reg = 5e-6 * loss_reg
                

                loss = loss_cls + loss_distill + loss_reg
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params_list, max_norm=1.0)
                optimizer.step()

                preds = (torch.sigmoid(logits) > 0.5).float()
                correct += (preds[:len(y)] == y).sum().item()
                total += y.size(0)
                total_loss += loss.item() * y.size(0)
                total_cls_loss += loss_cls.item() * y.size(0)
                total_distill_loss += loss_distill.item() * y.size(0) if loss_distill > 0 else 0
                
                pbar.set_postfix(
                    loss=total_loss/total, 
                    acc=100*correct/total,
                    distill=total_distill_loss/total if total_distill_loss > 0 else 0
                )

            avg_train_loss = total_loss / total
            train_acc = 100 * correct / total


            model.eval()
            val_loss, val_correct, val_total = 0, 0, 0
            all_preds, all_labels = [], []
            
            with torch.no_grad():
                for imgs, lbls in val_loader:
                    imgs = imgs.to(device)
                    y = lbls.float().unsqueeze(1).to(device)
                    
                    logits, _, _ = model(imgs)
                    loss = criterion(logits, y)
                    
                    probs = torch.sigmoid(logits)
                    preds = (probs > 0.5).float()
                    
                    val_correct += (preds == y).sum().item()
                    val_total += y.size(0)
                    val_loss += loss.item() * imgs.size(0)
                    
                    all_preds.extend(preds.cpu().numpy().flatten())
                    all_labels.extend(y.cpu().numpy().flatten())

            avg_val_loss = val_loss / val_total if val_total > 0 else 0
            val_acc = 100 * val_correct / val_total if val_total > 0 else 0
            val_f1 = f1_score(all_labels, all_preds, zero_division=0)

            print(f" Technique {tech} | Epoch {epoch}")
            print(f"   Train - Loss: {avg_train_loss:.4f}, Acc: {train_acc:.2f}%")
            print(f"   Val - Loss: {avg_val_loss:.4f} | Acc: {val_acc:.2f}% | F1: {val_f1:.3f}")
            
            if tech_idx > 0:
                print(f"    Distill Loss: {total_distill_loss/total:.6f}")
                print(f"    Replay: Active (50/50 ratio)")

            scheduler.step()


            if val_acc > tech_best_val_acc:
                tech_best_val_acc = val_acc
                tech_best_f1 = val_f1


        replay_buffer.add_technique_samples(tech, full_dataset)
        

        teacher_model = copy.deepcopy(model)
        teacher_model.eval()
        for param in teacher_model.parameters():
            param.requires_grad = False
        print(" Created NEW teacher model for next technique")


        technique_stats.append({
            "Technique": tech,
            "Best_Val_Accuracy": tech_best_val_acc,
            "Best_Val_F1": tech_best_f1,
            "Final_Epoch": args.epochs,
        })

        print(f" Technique {tech} completed: Best Val Acc: {tech_best_val_acc:.2f}%, Best F1: {tech_best_f1:.3f}")
        print(f"   Replay buffer now contains {len(replay_buffer.buffer)} techniques")


    save_full_model1(model, args.out, {
        "embed_dim": 768,
        "num_heads": 12,
        "n_bands": args.bands,
        "techniques": techniques,
        "training_mode": "fixed_incremental_learning",
        "epochs_per_technique": args.epochs,
        "replay_ratio": "50/50",
        "distillation_weights": "embedding=2.0, logits=0.3"
    })
    

    report_df = pd.DataFrame(technique_stats)
    report_df.to_csv(os.path.join(args.out, "incremental_training_report1.csv"), index=False)
    

    avg_acc = report_df["Best_Val_Accuracy"].mean()
    avg_f1 = report_df["Best_Val_F1"].mean()
    
    print(f"\n FIXED Incremental Learning finished!")
    print(f"   Final model saved to: {args.out}/model1.pth")
    print(f"   Average Val Accuracy: {avg_acc:.2f}%")
    print(f"   Average Val F1: {avg_f1:.3f}")
    print(f"   Replay buffer: {len(replay_buffer.buffer)} techniques with 300 samples each")
    print(f"   Key fixes applied: 50/50 replay ratio, path-based buffer, new teacher models")
    
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DualSpectralViT-KAN Training1 - FIXED Incremental Learning")
    

    parser.add_argument("--data_root", type=str, required=True,
                        help="Root folder with subfolders per technique")
    parser.add_argument("--techniques", type=str, required=True,
                        help="Comma-separated list of techniques to train sequentially")
    parser.add_argument("--out", type=str, required=True,
                        help="Output directory for models and reports")
    

    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of epochs per technique")
    parser.add_argument("--bs", type=int, default=64,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--val_split", type=float, default=0.2,
                        help="Validation split ratio")
    parser.add_argument("--img_size", type=int, default=224,
                        help="Image size")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of DataLoader workers")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    

    parser.add_argument("--embed_dim", type=int, default=768,
                        help="Embedding dimension")
    parser.add_argument("--heads", type=int, default=12,
                        help="Number of attention heads")
    parser.add_argument("--bands", type=int, default=4,
                        help="Number of frequency bands")
    
    args = parser.parse_args()
    
    print(f" STARTING FIXED INCREMENTAL LEARNING")
    print(f"   Critical fixes applied:")
    print(f"   - 50/50 replay ratio (32 new + 32 replay)")
    print(f"   - Path-based replay buffer (no GPU RAM issues)")
    print(f"   - New teacher model for each technique")
    print(f"   - Optimal distillation weights (embedding=2.0, logits=0.3)")
    print(f"   - Keep skip connections (scaled to 0.2)")
    print(f"   - Lower regularization (weight_decay=5e-5, reg_lambda=5e-6)")
    
    train_incremental1(args)