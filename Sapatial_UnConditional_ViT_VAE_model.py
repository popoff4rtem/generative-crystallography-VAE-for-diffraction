import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange, repeat

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=(250, 480), patch_size=(10, 16), in_channels=1, embed_dim=512):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        
        self.proj = nn.Conv2d(in_channels, embed_dim, 
                             kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)  # (B, embed_dim, grid_h, grid_w)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x
    
class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, num_patches):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        
    def forward(self, x):
        return x + self.pos_embed

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # Self-attention
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        
        # MLP
        x_norm = self.norm2(x)
        mlp_out = self.mlp(x_norm)
        x = x + mlp_out
        
        return x    

class ViTEncoderForLatentMaps(nn.Module):
    def __init__(self, img_size=(240, 480), patch_size=(10, 16), in_channels=1, 
                 embed_dim=512, depth=8, num_heads=8, mlp_ratio=4.0, 
                 latent_channels=256, dropout=0.1):
        super().__init__()
        
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.num_patches = self.patch_embed.num_patches
        self.grid_size = self.patch_embed.grid_size
        
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self.dropout = nn.Dropout(dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Project to latent maps [B, latent_channels, grid_h, grid_w]
        self.latent_proj = nn.Linear(embed_dim, latent_channels)
        
    def forward(self, x):
        # Patch embedding
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)
        
        # Add positional encoding
        x = x + self.pos_embed
        x = self.dropout(x)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        
        # Reshape to spatial grid and project to latent channels
        B, num_patches, embed_dim = x.shape
        h, w = self.grid_size
        x = x.view(B, h, w, embed_dim).permute(0, 3, 1, 2)  # (B, embed_dim, h, w)
        latent_maps = self.latent_proj(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)  # (B, latent_channels, h, w)
        
        return latent_maps

class ViTDecoderForLatentMaps(nn.Module):
    def __init__(self, img_size=(240, 480), patch_size=(10, 16), in_channels=1,
                 embed_dim=512, depth=8, num_heads=8, mlp_ratio=4.0,
                 latent_channels=256, dropout=0.1):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.latent_channels = latent_channels
        
        # Project latent maps to sequence
        self.latent_proj = nn.Conv2d(latent_channels, embed_dim, 1)
        
        # Learnable decoder tokens
        self.decoder_tokens = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self.dropout = nn.Dropout(dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Patch reconstruction
        self.patch_pred = nn.Linear(embed_dim, patch_size[0] * patch_size[1] * in_channels)
        
    def forward(self, latent_maps):
        B, C, H, W = latent_maps.shape
        
        # Project latent maps to embedding space
        x = self.latent_proj(latent_maps)  # (B, embed_dim, H, W)
        x = x.flatten(2).transpose(1, 2)   # (B, num_patches, embed_dim)
        
        # Add decoder tokens and positional encoding
        x = x + self.decoder_tokens
        x = x + self.pos_embed
        x = self.dropout(x)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
            
        x = self.norm(x)
        
        # Predict patches
        patches = self.patch_pred(x)  # (B, num_patches, patch_size[0]*patch_size[1])
        
        # Reshape to image
        patches = patches.view(B, self.grid_size[0], self.grid_size[1], 
                             self.patch_size[0], self.patch_size[1])
        
        # Rearrange to image format
        img = rearrange(patches, 'b h w p1 p2 -> b (h p1) (w p2)')
        img = img.unsqueeze(1)  # Add channel dimension
        
        return img

class ViTVAEForLatentMaps(nn.Module):
    def __init__(self, img_size=(240, 480), patch_size=(10, 16), in_channels=1,
                 embed_dim=512, depth=8, num_heads=8, mlp_ratio=4.0,
                 latent_channels=256, dropout=0.1):
        super().__init__()
        
        self.encoder = ViTEncoderForLatentMaps(img_size, patch_size, in_channels, embed_dim,
                                             depth, num_heads, mlp_ratio, latent_channels, dropout)
        
        self.decoder = ViTDecoderForLatentMaps(img_size, patch_size, in_channels, embed_dim,
                                             depth, num_heads, mlp_ratio, latent_channels, dropout)
        
        # Для вариационности - проекции для μ и σ
        self.fc_mu = nn.Conv2d(latent_channels, latent_channels, 1)
        self.fc_logvar = nn.Conv2d(latent_channels, latent_channels, 1)
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def encode(self, x):
        latent_maps = self.encoder(x)
        mu, logvar = self.fc_mu(latent_maps), self.fc_logvar(latent_maps)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar
        
    def decode(self, z):
        return self.decoder(z)
        
    def forward(self, x):
        z, mu, logvar = self.encode(x)
        recon_x = self.decode(z)
        return recon_x, mu, logvar