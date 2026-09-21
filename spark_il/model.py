import math
import torch
from torch import nn
import torch.nn.functional as F
import open_clip

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
    def __init__(self, embed_dim=768, num_heads=12, n_bands=4, pretrained="openai"):
        super().__init__()
        
        print(" Loading ViT-L/14 from OpenCLIP...")
        self.vit, _, _ = open_clip.create_model_and_transforms('ViT-L-14', pretrained=pretrained)
        

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

