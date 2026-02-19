# physical_diffraction_metrics.py
"""
Класс для вычисления и анализа физических метрик расходимости 
между предсказанными и реальными дифракционными паттернами.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from typing import Optional, Dict, Any, Tuple, Literal

# предполагаем, что DiffractionMetricsCalculator уже импортирован
from Diffraction_metrics_Calculator import DiffractionMetricsCalculator


class PhysicalDiffractionEvaluator:
    """
    Вычисляет метрики физической расходимости (integral intensity, peak shape и др.)
    между декодированными (сгенерированными) и реальными дифракционными изображениями.
    
    Особенности:
    - Поддерживает conditional и unconditional режимы декодера
    - Работает с нормализацией [-1,1] → [0,1]
    - Собирает статистику по всем батчам
    - Строит гистограммы распределений ошибок
    - Выводит mean / std / min / max / кол-во образцов
    """

    def __init__(
        self,
        model: torch.nn.Module,
        calculator: DiffractionMetricsCalculator,
        device: torch.device | str,
        mode: Literal["Conditional", "Unconditional"] = "Conditional",
        latent_channels: int = 32,
        latent_spatial: Tuple[int, int] = (25, 20),   # (H, W) latent map
        denorm: bool = True,                          # перевод из [-1,1] в [0,1]
        clamp: bool = True,
    ):
        self.model = model
        self.calculator = calculator
        self.device = device
        self.mode = mode
        self.latent_shape = (latent_channels, *latent_spatial)
        self.denorm = denorm
        self.clamp = clamp

        self.model.eval()

    def _denorm_and_clamp(self, x: torch.Tensor) -> torch.Tensor:
        if not self.denorm:
            return x
        x = (x + 1.0) / 2.0
        if self.clamp:
            x = x.clamp_(0.0, 1.0)
        return x

    @torch.no_grad()
    def collect_metrics(
        self,
        dataloader: torch.utils.data.DataLoader,
        noise_std: float = 1.0,           # масштаб шума для сэмплирования z
        desc: str = "Physical metrics",
    ) -> Dict[str, np.ndarray]:
        """
        Собирает метрики по всему датасету.
        Возвращает словарь со списками/массивами значений для каждой метрики.
        """
        all_metrics = {
            "integral": [],
            "shape":    [],
            # можно добавить другие ключи из DiffractionMetricsCalculator
        }

        for data_batch, label_batch in tqdm(dataloader, desc=desc):
            data = data_batch.to(self.device)
            labels = label_batch.to(self.device) if self.mode == "Conditional" else None

            # Генерация z ~ N(0, I)
            bs = data.size(0)
            z = torch.randn(bs, *self.latent_shape, device=self.device) * noise_std

            # Декодирование
            if self.mode == "Conditional":
                generated = self.model.decoder(z, labels)
            else:
                generated = self.model.decoder(z)

            # Приведение к [0,1]
            gt_01    = self._denorm_and_clamp(data)
            gen_01   = self._denorm_and_clamp(generated)

            # Вычисление метрик по каждому образцу в батче
            for pred_img, gt_img in zip(gen_01, gt_01):
                metrics_dict = self.calculator(
                    batch_pred_2d = pred_img.unsqueeze(0),   # [1, 1, H, W] или [1, C, H, W]
                    batch_true_2d = gt_img.unsqueeze(0),
                    peak_params_pred = {"scale": False},
                    peak_params_true = {"scale": False},
                    tol = 0.05,
                )

                # Предполагаем, что возвращаются списки/массивы по пикам
                int_vals = np.asarray(metrics_dict.get("Integral Intensity", []))
                shape_vals = np.asarray(metrics_dict.get("Shape", []))

                if len(int_vals) > 0:
                    all_metrics["integral"].append(int_vals)
                if len(shape_vals) > 0:
                    all_metrics["shape"].append(shape_vals)

        # Объединяем всё в один массив по каждой метрике
        for k in list(all_metrics.keys()):
            if all_metrics[k]:
                all_metrics[k] = np.concatenate(all_metrics[k], axis=0)
            else:
                all_metrics[k] = np.array([])

        return all_metrics

    @staticmethod
    def summarize_array(
        data: np.ndarray,
        name: str,
        print_stats: bool = True
    ) -> Dict[str, float]:
        if data.size == 0:
            if print_stats:
                print(f"{name}: no valid values")
            return {"mean": np.nan, "std": np.nan, "min": np.nan, "max": np.nan, "count": 0}

        mean_val = float(np.mean(data))
        std_val  = float(np.std(data, ddof=1))
        min_val  = float(np.min(data))
        max_val  = float(np.max(data))
        count    = len(data)

        if print_stats:
            print(f"{name}:")
            print(f"  Mean:   {mean_val:.4f}")
            print(f"  Std:    {std_val:.4f}")
            print(f"  Min:    {min_val:.4f}")
            print(f"  Max:    {max_val:.4f}")
            print(f"  Samples: {count}")
            print("-" * 50)

        return {
            "mean": mean_val,
            "std": std_val,
            "min": min_val,
            "max": max_val,
            "count": count,
        }

    @staticmethod
    def summarize_errors(
        errs: np.ndarray,
        name: str = "Error"
    ) -> Dict[str, float]:
        if errs.size == 0:
            return {"mean": np.nan, "median": np.nan, "p95": np.nan}
        return {
            "mean":   float(np.mean(errs)),
            "median": float(np.median(errs)),
            "p95":    float(np.percentile(errs, 95)),
        }

    def plot_distributions(
        self,
        all_metrics: Dict[str, np.ndarray],
        metrics_keys: list = ["integral", "shape"],
        bins: Optional[Dict] = None,
        xlims: Optional[Dict] = None,
        figsize: Tuple[float, float] = (15, 5),
        title: str = "Physical Metrics Error Distributions",
        save_path: Optional[str] = None,
    ):
        if bins is None:
            bins = {"integral": 200, "shape": 150}
        if xlims is None:
            xlims = {"integral": None, "shape": None}

        sns.set_theme(style="white")

        fig, axes = plt.subplots(1, len(metrics_keys), figsize=figsize)
        fig.suptitle(title, fontsize=16)

        color = "#1f77b4"

        for ax, metric in zip(axes if len(metrics_keys) > 1 else [axes], metrics_keys):
            data = all_metrics.get(metric, np.array([]))
            data = data[np.isfinite(data)]

            if data.size == 0:
                ax.text(0.5, 0.5, "No valid data", ha="center", va="center")
                ax.set_title(metric)
                continue

            sns.histplot(
                data,
                bins=bins.get(metric, 100),
                stat="density",
                kde=True,
                color=color,
                alpha=0.35,
                ax=ax,
                label=metric.capitalize()
            )

            ax.set_title(metric.capitalize())
            ax.set_xlabel("Absolute Error")
            ax.set_ylabel("Density")
            ax.tick_params(direction="in")
            ax.grid(False)

            if xlims.get(metric) is not None:
                ax.set_xlim(*xlims[metric])

            err_stats = self.summarize_errors(data)
            ax.text(
                0.02, 0.98,
                f"mean  = {err_stats['mean']:.3g}\n"
                f"median = {err_stats['median']:.3g}\n"
                f"p95    = {err_stats['p95']:.3g}",
                transform=ax.transAxes,
                ha="left", va="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.85)
            )
            ax.legend()

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=160, bbox_inches="tight")
            print(f"Plot saved → {save_path}")
        plt.show()

    def evaluate(
        self,
        dataloader: torch.utils.data.DataLoader,
        noise_std: float = 1.0,
        plot: bool = True,
        print_summary: bool = True,
        plot_title: str = "Physical Divergence Metrics",
        save_plot: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Основной метод: сбор + статистика + (опционально) графики.
        """
        print(f"Evaluating physical diffraction metrics ({self.mode} mode)...")
        print(f"Latent shape: {self.latent_shape}")

        metrics = self.collect_metrics(
            dataloader,
            noise_std=noise_std,
            desc="Physical divergence eval"
        )

        results = {}

        if print_summary:
            print("\n" + "="*60)
            print("INTEGRATED INTENSITY DIVERGENCE")
            results["integral"] = self.summarize_array(
                metrics.get("integral", np.array([])), "Integral intensity"
            )

            print("\nPEAK SHAPE DIVERGENCE")
            results["shape"] = self.summarize_array(
                metrics.get("shape", np.array([])), "Peak shape"
            )
            print("="*60 + "\n")

        if plot:
            self.plot_distributions(
                metrics,
                metrics_keys=["integral", "shape"],
                title=plot_title,
                save_path=save_plot
            )

        return {
            "raw_metrics": metrics,
            "summary": results
        }