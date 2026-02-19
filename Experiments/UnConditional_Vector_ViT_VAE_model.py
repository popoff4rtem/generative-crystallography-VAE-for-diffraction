import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# -----------------------------
# Patch Embedding
# -----------------------------
class PatchEmbedding(nn.Module):
    def __init__(
        self,
        img_size=(250, 480),
        patch_size=(10, 16),
        in_channels=1,
        embed_dim=512
    ):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1]
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.proj(x)                      # (B, D, Gh, Gw)
        x = x.flatten(2).transpose(1, 2)      # (B, N, D)
        return x


# -----------------------------
# Positional Encoding
# -----------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, num_patches):
        super().__init__()
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, embed_dim)
        )

    def forward(self, x):
        return x + self.pos_embed


# -----------------------------
# Transformer Block
# -----------------------------
class TransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        mlp_ratio=4.0,
        dropout=0.1
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # Self-attention
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out

        # MLP
        x_norm = self.norm2(x)
        x = x + self.mlp(x_norm)

        return x


# -----------------------------
# ViT Encoder (Unconditional)
# -----------------------------
class ViTEncoder(nn.Module):
    def __init__(
        self,
        img_size=(250, 480),
        patch_size=(10, 16),
        in_channels=1,
        embed_dim=512,
        depth=8,
        num_heads=8,
        mlp_ratio=4.0,
        latent_dim=256,
        dropout=0.1
    ):
        super().__init__()

        self.patch_embed = PatchEmbedding(
            img_size, patch_size, in_channels, embed_dim
        )
        self.num_patches = self.patch_embed.num_patches

        self.pos_embed = PositionalEncoding(embed_dim, self.num_patches)
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        self.fc_mean = nn.Linear(embed_dim, latent_dim)
        self.fc_logvar = nn.Linear(embed_dim, latent_dim)

    def forward(self, x):
        x = self.patch_embed(x)     # (B, N, D)
        x = self.pos_embed(x)
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        x = x.mean(dim=1)           # Global average pooling

        mean = self.fc_mean(x)
        logvar = self.fc_logvar(x)

        return mean, logvar


# -----------------------------
# ViT Decoder (Unconditional)
# -----------------------------
class ViTDecoder(nn.Module):
    def __init__(
        self,
        img_size=(250, 480),
        patch_size=(10, 16),
        in_channels=1,
        embed_dim=512,
        depth=8,
        num_heads=8,
        mlp_ratio=4.0,
        latent_dim=256,
        dropout=0.1
    ):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1]
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        # ⚠️ Более стабильная проекция латента
        self.latent_proj = nn.Linear(latent_dim, embed_dim)

        self.decoder_tokens = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim)
        )

        self.pos_embed = PositionalEncoding(embed_dim, self.num_patches)
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        self.patch_pred = nn.Linear(
            embed_dim,
            patch_size[0] * patch_size[1] * in_channels
        )

    def forward(self, z):
        B = z.size(0)

        # (B, D) -> (B, N, D)
        x = self.latent_proj(z).unsqueeze(1)
        x = x.repeat(1, self.num_patches, 1)

        x = x + self.decoder_tokens
        x = self.pos_embed(x)
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        patches = self.patch_pred(x)
        patches = patches.view(
            B,
            self.grid_size[0],
            self.grid_size[1],
            self.patch_size[0],
            self.patch_size[1]
        )

        img = rearrange(
            patches, 'b h w p1 p2 -> b (h p1) (w p2)'
        )

        return img.unsqueeze(1)


# -----------------------------
# Full ViT-VAE
# -----------------------------
class ViTVAE(nn.Module):
    def __init__(
        self,
        img_size=(250, 480),
        patch_size=(10, 16),
        in_channels=1,
        embed_dim=512,
        depth=8,
        num_heads=8,
        mlp_ratio=4.0,
        latent_dim=256,
        dropout=0.1
    ):
        super().__init__()

        self.encoder = ViTEncoder(
            img_size, patch_size, in_channels,
            embed_dim, depth, num_heads,
            mlp_ratio, latent_dim, dropout
        )

        self.decoder = ViTDecoder(
            img_size, patch_size, in_channels,
            embed_dim, depth, num_heads,
            mlp_ratio, latent_dim, dropout
        )

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x):
        mean, logvar = self.encoder(x)
        z = self.reparameterize(mean, logvar)
        recon_x = self.decoder(z)
        return recon_x, mean, logvar
    
    def encode(self, x):
        mean, logvar = self.encoder(x)
        z = self.reparameterize(mean, logvar)
        return z, mean, logvar

    def decode(self, z):
        return self.decoder(z)