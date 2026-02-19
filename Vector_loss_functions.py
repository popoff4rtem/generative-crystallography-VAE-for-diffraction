import torch
import torch.nn as nn
import torch.nn.functional as F

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