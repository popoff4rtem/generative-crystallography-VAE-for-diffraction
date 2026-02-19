import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PatchEmbed(nn.Module):
    """Разбиваем латентные карты на патчи"""
    def __init__(self, in_channels=256, patch_size=(3, 6), hidden_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, hidden_dim, 
                             kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        # x: [batch, 256, 15, 30]
        x = self.proj(x)  # [batch, hidden_dim, 5, 5] для patch_size=(3,6)
        batch_size, channels, height, width = x.shape
        x = x.flatten(2)  # [batch, channels, height*width]
        x = x.transpose(1, 2)  # [batch, height*width, channels]
        
        return x

class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_dim, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.frequency_embedding_size = frequency_embedding_size

   # В TimestepEmbedder.forward
    def forward(self, t):
        t = t.float()
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

class ClassEmbedder(nn.Module):
    def __init__(self, num_classes, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(num_classes, hidden_dim)
        
    def forward(self, labels):
        return self.embed(labels)

class DiTBlock(nn.Module):
    """Базовый блок DiT с conditioning"""
    def __init__(self, hidden_dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.attn = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True, dropout=0.1
        )
        self.norm2 = nn.LayerNorm(hidden_dim, eps=1e-6)
        
        mlp_hidden_dim = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, hidden_dim)
        )
        
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 6 * hidden_dim),
        )
        
    def forward(self, x, c):
        # c - conditioning vector [batch, hidden_dim]
        
        # Get modulation parameters
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        
        # Modulation for attention
        x_mod = self.norm1(x)
        x_mod = x_mod * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        
        # Self-attention with gate
        attn_out, _ = self.attn(x_mod, x_mod, x_mod)
        x = x + gate_msa.unsqueeze(1) * attn_out
        
        # Modulation for MLP
        x_mod = self.norm2(x)
        x_mod = x_mod * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        
        # MLP with gate
        mlp_out = self.mlp(x_mod)
        x = x + gate_mlp.unsqueeze(1) * mlp_out
        
        return x
    
class SimpleDiTBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True, dropout=0.1)
        self.norm2 = nn.LayerNorm(hidden_dim, eps=1e-6)
        
        mlp_hidden_dim = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_hidden_dim, hidden_dim),
            nn.Dropout(0.1)
        )
        
        # Простой conditioning вместо сложной модуляции
        self.condition_proj = nn.Linear(hidden_dim, hidden_dim * 2)
        
    def forward(self, x, c):
        # Condition projection
        scale_shift = self.condition_proj(c).chunk(2, dim=1)
        scale, shift = scale_shift[0].unsqueeze(1), scale_shift[1].unsqueeze(1)
        
        # Attention с residual
        residual = x
        x = self.norm1(x)
        x = x * (1 + scale) + shift  # Simple conditioning
        attn_out, _ = self.attn(x, x, x)
        x = residual + attn_out
        
        # MLP с residual
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = residual + x
        
        return x
    
class CrossAttentionDiTBlock(nn.Module):
    """DiT блок с cross-attention для conditioning"""
    def __init__(self, hidden_dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        # Self-attention часть
        self.norm1 = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True, dropout=0.1)
        
        # Cross-attention часть
        self.norm2 = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True, dropout=0.1)
        
        # MLP
        self.norm3 = nn.LayerNorm(hidden_dim, eps=1e-6)
        mlp_hidden_dim = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_hidden_dim, hidden_dim),
            nn.Dropout(0.1)
        )
        
    def forward(self, x, conditioning):
        # x: [batch, num_patches, hidden_dim]
        # conditioning: [batch, 2, hidden_dim] (t_emb + class_emb)
        
        # Self-attention
        residual = x
        x = self.norm1(x)
        attn_out, _ = self.self_attn(x, x, x)
        x = residual + attn_out
        
        # Cross-attention с conditioning
        residual = x
        x = self.norm2(x)
        
        # conditioning как key/value, x как query
        cross_attn_out, _ = self.cross_attn(
            query=x, 
            key=conditioning, 
            value=conditioning
        )
        x = residual + cross_attn_out
        
        # MLP
        residual = x
        x = self.norm3(x)
        x = self.mlp(x)
        x = residual + x
        
        return x

class DiffusionTransformer(nn.Module):
    """DiT для латентных 2D карт с 2D positional embedding"""
    def __init__(self,
                 in_size=(25, 30), 
                 in_channels=256,
                 patch_size=(3, 6),
                 hidden_dim=768,
                 depth=12,
                 num_heads=12,
                 num_classes=30):
        super().__init__()
        
        # Patch embedding
        self.patch_embed = PatchEmbed(in_channels, patch_size, hidden_dim)
        h, w = in_size[0] // patch_size[0], in_size[1] // patch_size[1]  # 5x5
        
        # Learnable 2D positional embeddings
        self.pos_embed_h = nn.Parameter(torch.randn(1, h, 1, hidden_dim) * 0.02)
        self.pos_embed_w = nn.Parameter(torch.randn(1, 1, w, hidden_dim) * 0.02)
        
        # Conditioning embedders
        self.t_embedder = TimestepEmbedder(hidden_dim)
        self.c_embedder = ClassEmbedder(num_classes, hidden_dim)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_dim, num_heads) for _ in range(depth)
        ])
        
        # Final layers
        self.norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.proj_out = nn.Linear(hidden_dim, patch_size[0] * patch_size[1] * in_channels)
        
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.h, self.w = h, w

        self.initialize_weights()

    def initialize_weights(self):
        # Инициализация позиционных эмбеддингов
        nn.init.normal_(self.pos_embed_h, std=0.02)
        nn.init.normal_(self.pos_embed_w, std=0.02)
        
        # Инициализация линейных слоев
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)

    def forward(self, x, t, class_labels):
        # x: [batch, 256, 15, 30]
        # t: [batch]
        # class_labels: [batch]
        
        # Patch embedding
        x = self.patch_embed(x)  # [batch, num_patches, hidden_dim]
        
        # Добавляем 2D positional embedding
        # Превращаем последовательность патчей обратно в сетку h×w
        B = x.size(0)
        x = x.view(B, self.h, self.w, -1)
        x = x + self.pos_embed_h + self.pos_embed_w
        x = x.view(B, self.h * self.w, -1)  # снова в последовательность
        
        # Conditioning
        t_emb = self.t_embedder(t)
        c_emb = self.c_embedder(class_labels)
        # c = F.normalize(t_emb + c_emb, dim=-1)
        c = t_emb + c_emb
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x, c)
            
        # Final projection
        x = self.norm(x)
        x = self.proj_out(x)  # [batch, num_patches, patch_dim]
        
        # Reshape обратно в 2D
        batch_size, num_patches, patch_dim = x.shape
        h, w = self.h, self.w
        x = x.view(batch_size, h, w, self.patch_size[0], self.patch_size[1], self.in_channels)
        x = x.permute(0, 5, 1, 3, 2, 4)
        x = x.contiguous().view(batch_size, self.in_channels, h * self.patch_size[0], w * self.patch_size[1])
        
        return x


class CrossAttentionDiffusionTransformer(nn.Module):
    """DiT с cross-attention conditioning"""
    def __init__(self,
                 in_size=(25, 30), 
                 in_channels=256,
                 patch_size=(3, 3),
                 hidden_dim=512,
                 depth=8,
                 num_heads=8,
                 num_classes=30):
        super().__init__()
        
        # Patch embedding (оставляем как было)
        self.patch_embed = PatchEmbed(in_channels, patch_size, hidden_dim)
        h, w = in_size[0] // patch_size[0], in_size[1] // patch_size[1]  # 5x5
        
        # Positional embeddings
        self.pos_embed_h = nn.Parameter(torch.randn(1, h, 1, hidden_dim) * 0.02)
        self.pos_embed_w = nn.Parameter(torch.randn(1, 1, w, hidden_dim) * 0.02)
        
        # Conditioning embedders
        self.t_embedder = TimestepEmbedder(hidden_dim)
        self.c_embedder = ClassEmbedder(num_classes, hidden_dim)
        
        # Transformer blocks с cross-attention
        self.blocks = nn.ModuleList([
            CrossAttentionDiTBlock(hidden_dim, num_heads) for _ in range(depth)
        ])
        
        # Final layers
        self.norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.proj_out = nn.Linear(hidden_dim, patch_size[0] * patch_size[1] * in_channels)
        
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.h, self.w = h, w
        
        self.initialize_weights()

    def initialize_weights(self):
        nn.init.normal_(self.pos_embed_h, std=0.02)
        nn.init.normal_(self.pos_embed_w, std=0.02)
        
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)

    def forward(self, x, t, class_labels):
        # x: [batch, 256, 15, 30]
        # t: [batch]
        # class_labels: [batch]
        
        # Patch embedding
        x = self.patch_embed(x)  # [batch, num_patches, hidden_dim]
        
        # Positional embedding
        B = x.size(0)
        x = x.view(B, self.h, self.w, -1)
        x = x + self.pos_embed_h + self.pos_embed_w
        x = x.view(B, self.h * self.w, -1)
        
        # Подготавливаем conditioning
        t_emb = self.t_embedder(t)  # [batch, hidden_dim]
        c_emb = self.c_embedder(class_labels)  # [batch, hidden_dim]
        
        # Объединяем conditioning векторы
        # [batch, 2, hidden_dim] - 2 conditioning вектора (time + class)
        conditioning = torch.stack([t_emb, c_emb], dim=1)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x, conditioning)
            
        # Final projection
        x = self.norm(x)
        x = self.proj_out(x)
        
        # Reshape обратно в 2D
        batch_size, num_patches, patch_dim = x.shape
        h, w = self.h, self.w
        x = x.view(batch_size, h, w, self.patch_size[0], self.patch_size[1], self.in_channels)
        x = x.permute(0, 5, 1, 3, 2, 4)
        x = x.contiguous().view(batch_size, self.in_channels, h * self.patch_size[0], w * self.patch_size[1])
        
        return x