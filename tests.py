# tests.py
"""
Модуль с тестами / диагностиками латентного пространства для VAE-модели.

Содержит три независимых класса:
1. LatentSpaceBasicStats     — базовая статистика (mean, std, min, max, nan/inf)
2. LatentSpaceVisualizer     — визуализация с понижением размерности (t-SNE / UMAP / PCA)
3. LatentSpaceClusteringEval — метрики кластеризации, KL, linear probe, MI-proxy
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from typing import Optional, Literal, List, Dict, Any
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap

class LatentSpaceBasicStats:
    """
    Класс для сбора и вывода базовой статистики латентных представлений.
    Работает как в conditional, так и в unconditional режиме.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device | str,
        mode: Literal["Conditional", "Unconditional"] = "Conditional",
    ):
        self.model = model
        self.device = device
        self.mode = mode
        self.model.eval()

    @torch.no_grad()
    def collect(self, dataloader: torch.utils.data.DataLoader) -> torch.Tensor:
        latent_samples = []

        for batch in tqdm(dataloader, desc="Collecting latent stats"):
            if self.mode == "Conditional":
                if len(batch) < 2:
                    raise ValueError("Conditional mode expects (data, labels)")
                data, labels = batch
                data = data.to(self.device)
                labels = labels.to(self.device)
                z, _, _ = self.model.encode(data, labels)
            else:
                data = batch[0] if isinstance(batch, (list, tuple)) else batch
                data = data.to(self.device)
                z, _, _ = self.model.encode(data)

            latent_samples.append(z.cpu())

        return torch.cat(latent_samples, dim=0)

    def diagnose(self, dataloader: torch.utils.data.DataLoader) -> Dict[str, Any]:
        latents = self.collect(dataloader)

        stats = {
            "shape": tuple(latents.shape),
            "mean": float(latents.mean()),
            "std": float(latents.std()),
            "min": float(latents.min()),
            "max": float(latents.max()),
            "has_nan": bool(torch.isnan(latents).any()),
            "has_inf": bool(torch.isinf(latents).any()),
        }

        # шум для сравнения
        noise = torch.randn_like(latents)
        stats["noise_std_reference"] = float(noise.std())

        print("═" * 50)
        print("LATENT SPACE — BASE DIAGNOSTIC")
        print(f"Mode: {self.mode}")
        print(f"Shape: {stats['shape']}")
        print(f"Mean: {stats['mean']:.4f}")
        print(f"Std : {stats['std']:.4f}")
        print(f"Min : {stats['min']:.4f}")
        print(f"Max : {stats['max']:.4f}")
        print(f"NaN : {stats['has_nan']}")
        print(f"Inf : {stats['has_inf']}")
        print(f"Noise std (ref): {stats['noise_std_reference']:.4f}")
        print("═" * 50)

        return stats


class LatentSpaceVisualizer:
    """
    Класс для визуализации латентного пространства после понижения размерности.
    Поддерживает t-SNE, UMAP, PCA.
    Ожидает метки классов (даже в unconditional режиме).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device | str,
        mode: Literal["Conditional", "Unconditional"] = "Conditional",
    ):
        self.model = model
        self.device = device
        self.mode = mode
        self.model.eval()

    @torch.no_grad()
    def collect_vectors(
        self, dataloader: torch.utils.data.DataLoader
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Собирает глобальные векторы (mean по пространственным осям)"""
        all_z_vec, all_labels, all_stds, all_norms = [], [], [], []

        for data_batch, label_batch in tqdm(dataloader, desc="Collecting for viz"):
            data = data_batch.to(self.device)
            labels = label_batch.to(self.device)

            z, mu, logvar = self._encode(data, labels)

            z_vec = torch.mean(z, dim=(2, 3)).cpu().numpy()
            mu_vec = torch.mean(mu, dim=(2, 3)).cpu().numpy()
            logvar_vec = torch.mean(logvar, dim=(2, 3)).cpu().numpy()

            all_z_vec.append(z_vec)
            all_norms.append(np.linalg.norm(mu_vec, axis=1))
            all_stds.append(np.exp(0.5 * logvar_vec).mean(axis=1))
            all_labels.append(labels.cpu().numpy())

        return (
            np.vstack(all_z_vec),
            np.concatenate(all_labels),
            np.concatenate(all_stds),
            np.concatenate(all_norms),
        )

    def _encode(self, data: torch.Tensor, labels: torch.Tensor):
        if self.mode == "Conditional":
            return self.model.encode(data, labels)
        else:
            return self.model.encode(data)

    def visualize(
        self,
        dataloader: torch.utils.data.DataLoader,
        method: Literal["tsne", "umap", "pca"] = "tsne",
        class_names: Optional[List[str]] = None,
        figsize: tuple = (15, 5),
        save_path: Optional[str] = None,
    ):
        z_vec, labels, stds, norms = self.collect_vectors(dataloader)

        # Понижение размерности
        if method == "tsne":
            reducer = TSNE(n_components=2, perplexity=30, random_state=42)
        elif method == "umap":
            reducer = umap.UMAP(n_components=2, random_state=42)
        else:
            reducer = PCA(n_components=2)

        lat2d = reducer.fit_transform(z_vec)

        # Цветовая карта для классов
        unique_labels = np.unique(labels)
        n_classes = len(unique_labels)
        base_colors = (
            list(plt.cm.tab20.colors)
            + list(plt.cm.tab20b.colors)
            + list(plt.cm.tab20c.colors)
        )
        cmap = ListedColormap(base_colors[:n_classes])

        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # 1. По классам
        scatter = axes[0].scatter(
            lat2d[:, 0],
            lat2d[:, 1],
            c=labels,
            cmap=cmap,
            s=10,
            vmin=-0.5,
            vmax=n_classes - 0.5,
        )
        cbar = fig.colorbar(scatter, ax=axes[0], ticks=range(n_classes))
        if class_names and len(class_names) >= n_classes:
            cbar.ax.set_yticklabels(class_names[:n_classes])
        else:
            cbar.ax.set_yticklabels([str(i) for i in range(n_classes)])
        axes[0].set_title("Latent by Class")

        # 2. По стандартному отклонению
        s = axes[1].scatter(lat2d[:, 0], lat2d[:, 1], c=stds, cmap="viridis", s=10)
        fig.colorbar(s, ax=axes[1])
        axes[1].set_title("Latent by Std")

        # 3. По норме mu
        s = axes[2].scatter(lat2d[:, 0], lat2d[:, 1], c=norms, cmap="plasma", s=10)
        fig.colorbar(s, ax=axes[2])
        axes[2].set_title("Latent by Norm")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved to: {save_path}")
        plt.show()

        return {"z_2d": lat2d, "labels": labels, "stds": stds, "norms": norms}


class LatentSpaceClusteringEval:
    """
    Класс для вычисления метрик качества латентного пространства:
    - Кластеризационные метрики (Silhouette, Calinski-Harabasz, Davies-Bouldin)
    - Linear probe accuracy (кросс-валидация)
    - Прокси mutual information через логистическую регрессию
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device | str,
        mode: Literal["Conditional", "Unconditional"] = "Conditional",
    ):
        self.model = model
        self.device = device
        self.mode = mode
        self.model.eval()

    @torch.no_grad()
    def collect_statistics(
        self, dataloader: torch.utils.data.DataLoader
    ) -> Dict[str, np.ndarray]:
        latents, mus, logvars, labels_all = [], [], [], []

        for data_batch, label_batch in tqdm(dataloader, desc="Collecting stats"):
            data = data_batch.to(self.device)
            labels = label_batch.to(self.device)

            z, mu, logvar = self._encode(data, labels)

            z_vec = torch.mean(z, dim=(2, 3)).cpu()
            mu_vec = torch.mean(mu, dim=(2, 3)).cpu()
            logvar_vec = torch.mean(logvar, dim=(2, 3)).cpu()

            latents.append(z_vec)
            mus.append(mu_vec)
            logvars.append(logvar_vec)
            labels_all.append(labels.cpu())

        return {
            "z": torch.cat(latents).numpy(),
            "mu": torch.cat(mus).numpy(),
            "logvar": torch.cat(logvars).numpy(),
            "labels": torch.cat(labels_all).numpy(),
        }

    def _encode(self, data: torch.Tensor, labels: torch.Tensor):
        if self.mode == "Conditional":
            return self.model.encode(data, labels)
        else:
            return self.model.encode(data)

    @staticmethod
    def clustering_metrics(z: np.ndarray, labels: np.ndarray) -> Dict:
        if len(np.unique(labels)) < 2:
            return {"error": "Less than 2 unique classes → clustering metrics undefined"}
        return {
            "silhouette": float(silhouette_score(z, labels)),
            "calinski_harabasz": float(calinski_harabasz_score(z, labels)),
            "davies_bouldin": float(davies_bouldin_score(z, labels)),
        }

    @staticmethod
    def linear_probe(z: np.ndarray, labels: np.ndarray, n_splits: int = 5) -> Dict:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        accs = []
        for train_idx, test_idx in skf.split(z, labels):
            clf = LogisticRegression(max_iter=4000, n_jobs=-1)
            clf.fit(z[train_idx], labels[train_idx])
            accs.append(accuracy_score(labels[test_idx], clf.predict(z[test_idx])))
        return {
            "linear_probe_mean": float(np.mean(accs)),
            "linear_probe_std": float(np.std(accs)),
        }

    @staticmethod
    def mi_proxy(z: np.ndarray, labels: np.ndarray) -> Dict:
        X_tr, X_te, y_tr, y_te = train_test_split(
            z, labels, test_size=0.3, stratify=labels, random_state=42
        )
        clf = LogisticRegression(max_iter=4000, n_jobs=-1)
        clf.fit(X_tr, y_tr)
        return {"mi_proxy_acc": float(accuracy_score(y_te, clf.predict(X_te)))}

    def evaluate(self, dataloader: torch.utils.data.DataLoader) -> Dict[str, Any]:
        stats = self.collect_statistics(dataloader)

        z = stats["z"]
        mu = stats["mu"]
        logvar = stats["logvar"]
        labels = stats["labels"]

        results = {
            "clustering": self.clustering_metrics(z, labels),
            "linear_probe": self.linear_probe(z, labels),
            "mi_proxy": self.mi_proxy(z, labels),
        }

        print("═" * 50)
        print("LATENT SPACE — CLUSTERING METRICS")
        print(f"Mode: {self.mode}")
        print("Clustering:", results["clustering"])
        print("Linear probe:", results["linear_probe"])
        print("MI proxy:", results["mi_proxy"])
        print("═" * 50)

        return results