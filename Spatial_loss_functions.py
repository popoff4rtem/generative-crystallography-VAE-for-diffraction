import torch
import torch.nn as nn
import torch.nn.functional as F

def vae_loss(x_recon, x, mu, logvar, beta=1.0):
    recon_loss = F.mse_loss(x_recon, x, reduction='mean')
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    total_loss = recon_loss + beta * kl_loss
    return total_loss, recon_loss, kl_loss

def spatial_vae_loss_optimal(
    x_recon, x,
    mu, logvar,
    free_bits_channel=0.05,
    beta=0.5,
    recon_weight=1.0
):
    """
    Оптимальный VAE loss для spatial ViT-VAE (diffraction-friendly)
    
    mu, logvar: [B, C, H, W]
    """

    # --- Reconstruction ---
    recon_loss = F.mse_loss(x_recon, x, reduction='mean')

    # --- KL per element ---
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())  # [B,C,H,W]

    # --- Aggregate spatially → channel-wise ---
    kl_channel = kl.mean(dim=(2, 3))   # [B, C]

    # --- Free bits per channel ---
    kl_channel = torch.clamp(kl_channel, min=free_bits_channel)

    # --- Final KL ---
    kl_loss = kl_channel.mean()

    total_loss = recon_weight * recon_loss + beta * kl_loss

    return total_loss, recon_loss, kl_loss

def class_compactness_loss(mu_global, labels, eps=1e-6):
    """
    mu_global: [B, C]
    labels: [B]
    """
    loss = 0.0
    count = 0

    for cls in torch.unique(labels):
        mask = labels == cls
        if mask.sum() < 2:
            continue

        mu_c = mu_global[mask]                      # [Nc, C]
        center = mu_c.mean(dim=0, keepdim=True)     # [1, C]

        loss += ((mu_c - center) ** 2).mean()
        count += 1

    if count > 0:
        loss = loss / count
    else:
        loss = torch.tensor(0.0, device=mu_global.device)

    return loss

def spatial_vae_loss_with_clustering_pull(
    x_recon, x,
    mu, logvar,
    labels,
    free_bits_channel=0.05,
    beta=0.5,
    gamma=0.05,
    recon_weight=1.0
):
    # --- Reconstruction ---
    recon_loss = F.mse_loss(x_recon, x, reduction='mean')

    # --- KL ---
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    kl_channel = kl.mean(dim=(2,3))
    kl_channel = torch.clamp(kl_channel, min=free_bits_channel)
    kl_loss = kl_channel.mean()

    # --- Clustering ---
    mu_global = mu.mean(dim=(2,3))
    cluster_loss = class_compactness_loss(mu_global, labels)

    total_loss = (
        recon_weight * recon_loss
        + beta * kl_loss
        + gamma * cluster_loss
    )

    return total_loss, recon_loss, kl_loss, cluster_loss

def class_separation_loss(mu_global, labels, margin=1.0):
    centers = []

    for cls in torch.unique(labels):
        mask = labels == cls
        if mask.sum() < 2:
            continue
        centers.append(mu_global[mask].mean(dim=0))

    if len(centers) < 2:
        return torch.tensor(0.0, device=mu_global.device)

    centers = torch.stack(centers)  # [K, D]

    loss = 0.0
    count = 0

    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            dist = torch.norm(centers[i] - centers[j], p=2)
            loss += torch.clamp(margin - dist, min=0.0) ** 2
            count += 1

    return loss / count

def spatial_vae_loss_with_clustering_push_pull(
    x_recon, x,
    mu, logvar,
    labels,
    free_bits_channel=0.05,
    beta=0.5,
    gamma_pull=0.05,
    gamma_push=0.01,
    margin=1.0,
    recon_weight=1.0
):
    # --- Reconstruction ---
    recon_loss = F.mse_loss(x_recon, x, reduction='mean')

    # --- KL ---
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    kl_channel = kl.mean(dim=(2,3))
    kl_channel = torch.clamp(kl_channel, min=free_bits_channel)
    kl_loss = kl_channel.mean()

    # --- Latent ---
    mu_global = mu.mean(dim=(2,3))

    pull_loss = class_compactness_loss(mu_global, labels)
    push_loss = class_separation_loss(mu_global, labels, margin=margin)

    cluster_loss = gamma_pull * pull_loss + gamma_push * push_loss

    total_loss = (
        recon_weight * recon_loss
        + beta * kl_loss
        + cluster_loss
    )

    return total_loss, recon_loss, kl_loss, pull_loss, push_loss

# losses.py
def basic_vae_loss(recon, x, mu, logvar, labels=None, kl_weight=1.0, **_):
    loss, recon_loss, kl_loss = vae_loss(recon, x, mu, logvar, kl_weight)
    return loss, {"recon": recon_loss, "kl": kl_loss}


def clustering_pull_loss(recon, x, mu, logvar, labels, beta, gamma, **static):
    loss, recon_loss, kl_loss, cluster_loss = spatial_vae_loss_with_clustering_pull(
        x_recon=recon,
        x=x,
        mu=mu,
        logvar=logvar,
        labels=labels,
        beta=beta,
        gamma=gamma,
        **static  # free_bits_channel, recon_weight и т.д.
    )
    return loss, {"recon": recon_loss, "kl": kl_loss, "cluster": cluster_loss}


def clustering_push_pull_loss(recon, x, mu, logvar, labels, beta, gamma_pull, gamma_push, **static):
    (
        loss,
        recon_loss,
        kl_loss,
        pull_loss,
        push_loss,
    ) = spatial_vae_loss_with_clustering_push_pull(
        x_recon=recon,
        x=x,
        mu=mu,
        logvar=logvar,
        labels=labels,
        beta=beta,
        gamma_pull=gamma_pull,
        gamma_push=gamma_push,
        **static
    )
    return loss, {
        "recon": recon_loss,
        "kl": kl_loss,
        "pull": pull_loss,
        "push": push_loss,
    }