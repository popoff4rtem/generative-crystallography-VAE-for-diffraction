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

class ConditionBlock(nn.Module):
    def __init__(self, embed_dim, num_classes, mode="add", num_heads=8, dropout=0.1):
        super().__init__()
        assert mode in ["add", "cross_attn"]
        self.mode = mode

        self.class_embed = nn.Embedding(num_classes, embed_dim)

        if mode == "cross_attn":
            self.cross_attn = nn.MultiheadAttention(
                embed_dim, num_heads, dropout=dropout, batch_first=True
            )
            self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, class_labels):
        """
        x: (B, N, D)
        class_labels: (B,)
        """
        class_emb = self.class_embed(class_labels)  # (B, D)

        if self.mode == "add":
            return x + class_emb.unsqueeze(1)

        # cross-attention
        class_token = class_emb.unsqueeze(1)  # (B, 1, D)

        attn_out, _ = self.cross_attn(
            query=x,
            key=class_token,
            value=class_token
        )

        return self.norm(x + attn_out)

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

class ViTEncoder(nn.Module):
    def __init__(self, img_size=(250, 480), patch_size=(10, 16), in_channels=1,
                 embed_dim=512, depth=8, num_heads=8, mlp_ratio=4.0,
                 num_classes=30, latent_dim=256, dropout=0.1, condition_type="add", condition_mode="single"):
        super().__init__()

        assert condition_mode in ["single", "multi"]
        self.condition_mode = condition_mode

        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.num_patches = self.patch_embed.num_patches

        self.condition = ConditionBlock(
            embed_dim,
            num_classes,
            mode=condition_type,
            num_heads=num_heads,
            dropout=dropout
        )

        self.pos_embed = PositionalEncoding(embed_dim, self.num_patches)
        self.dropout = nn.Dropout(dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # Latent projection
        self.latent_mean = nn.Linear(embed_dim, latent_dim)
        self.latent_logvar = nn.Linear(embed_dim, latent_dim)

    def forward(self, x, class_labels):
        # Patch embedding
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)

        if self.condition_mode == "single":
            # Add class conditioning and positional encoding
            x = self.condition(x, class_labels)
            x = self.pos_embed(x)
            x = self.dropout(x)

            # Transformer blocks
            for block in self.blocks:
                x = block(x)
        else:
            # Multi mode: ОДИН И ТОТ ЖЕ condition блок применяется перед каждым блоком
            x = self.pos_embed(x)
            x = self.dropout(x)

            # Transformer blocks
            for block in self.blocks:
                x = self.condition(x, class_labels)
                x = block(x)

        x = self.norm(x)

        # Global average pooling
        x = x.mean(dim=1)  # (B, embed_dim)

        # Latent distribution
        mean = self.latent_mean(x)
        logvar = self.latent_logvar(x)

        return mean, logvar

class ViTDecoder(nn.Module):
    def __init__(self, img_size=(250, 480), patch_size=(10, 16), in_channels=1,
                 embed_dim=512, depth=8, num_heads=8, mlp_ratio=4.0,
                 num_classes=30, latent_dim=256, dropout=0.1, condition_type="add", condition_mode="single"):
        super().__init__()

        assert condition_mode in ["single", "multi"]
        self.condition_mode = condition_mode

        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.latent_dim = latent_dim

        # Project latent to initial sequence
        self.latent_proj = nn.Linear(latent_dim, embed_dim * self.num_patches)

        # Learnable decoder tokens
        self.decoder_tokens = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))

        self.condition = ConditionBlock(
            embed_dim,
            num_classes,
            mode=condition_type,
            num_heads=num_heads,
            dropout=dropout
        )

        self.pos_embed = PositionalEncoding(embed_dim, self.num_patches)
        self.dropout = nn.Dropout(dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # Patch reconstruction
        self.patch_pred = nn.Linear(embed_dim, patch_size[0] * patch_size[1] * in_channels)

    def forward(self, z, class_labels):
        B = z.shape[0]

        # Project latent to initial sequence and reshape
        x = self.latent_proj(z)  # (B, embed_dim * num_patches)
        x = x.view(B, self.num_patches, -1)  # (B, num_patches, embed_dim)

        # Add decoder tokens and conditioning
        x = x + self.decoder_tokens

        if self.condition_mode == "single":
            x = self.condition(x, class_labels)
            x = self.pos_embed(x)
            x = self.dropout(x)

            # Transformer blocks
            for block in self.blocks:
                x = block(x)

        else:
            x = self.pos_embed(x)
            x = self.dropout(x)

            # Transformer blocks
            for block in self.blocks:
                x = self.condition(x, class_labels)
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

class ViTVAE(nn.Module):
    def __init__(self, img_size=(250, 480), patch_size=(10, 16), in_channels=1,
                 embed_dim=512, depth=8, num_heads=8, mlp_ratio=4.0,
                 num_classes=30, latent_dim=256, dropout=0.1, condition_type="add", condition_mode="single"):
        super().__init__()

        self.encoder = ViTEncoder(img_size, patch_size, in_channels, embed_dim,
                                depth, num_heads, mlp_ratio, num_classes,
                                latent_dim, dropout, condition_type, condition_mode)

        self.decoder = ViTDecoder(img_size, patch_size, in_channels, embed_dim,
                                depth, num_heads, mlp_ratio, num_classes,
                                latent_dim, dropout, condition_type, condition_mode)

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def forward(self, x, class_labels):
        mean, logvar = self.encoder(x, class_labels)
        z = self.reparameterize(mean, logvar)
        recon_x = self.decoder(z, class_labels)
        return recon_x, mean, logvar

    def encode(self, x, class_labels):
        mean, logvar = self.encoder(x, class_labels)
        z = self.reparameterize(mean, logvar)
        return z, mean, logvar

    def decode(self, z, class_labels):
        return self.decoder(z, class_labels)

def vae_loss(x_recon, x, mu, logvar, beta=1.0):
    recon_loss = F.mse_loss(x_recon, x, reduction='mean')
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    total_loss = recon_loss + beta * kl_loss
    return total_loss, recon_loss, kl_loss

def vae_loss_free_bits(x_recon, x, mu, logvar, free_bits_threshold=0.1, beta=1.0):
    recon_loss = F.mse_loss(x_recon, x, reduction='mean')
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    # Free bits применяем к КАЖДОМУ латентному элементу
    kl_loss = torch.clamp(kl_loss, min=free_bits_threshold)
    total_loss = recon_loss + beta * kl_loss
    return total_loss, recon_loss, kl_loss