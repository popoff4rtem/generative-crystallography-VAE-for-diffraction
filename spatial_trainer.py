# trainer.py
import torch
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from collections import defaultdict
from typing import Callable, Dict, Optional, Literal

class VAETrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable,
        mode: Literal["Conditional", "Unconditional"] = "Conditional",
        device: Optional[str] = None,
        grad_accum_steps: int = 1,
        use_amp: bool = True,
        param_schedulers: Optional[Dict[str, Callable[[int], float]]] = None,
        static_params: Optional[Dict] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        require_labels_in_loss: bool = True,          # ← новый флаг
    ):
        """
        VAE Trainer с поддержкой conditional/unconditional forward и обязательными labels для loss.
        
        Args:
            mode: влияет только на вызов модели
                "Conditional"    → model(data, labels)
                "Unconditional"  → model(data)
            require_labels_in_loss: если True — ожидается, что в каждом батче есть labels,
                                    иначе ошибка. Полезно когда clustering loss всегда нужен.
        """
        self.mode = mode
        if mode not in ("Conditional", "Unconditional"):
            raise ValueError("mode must be 'Conditional' or 'Unconditional'")

        self.require_labels_in_loss = require_labels_in_loss

        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.grad_accum_steps = grad_accum_steps
        self.use_amp = use_amp
        self.param_schedulers = param_schedulers or {}
        self.static_params = static_params or {}
        self.scheduler = scheduler
        self.scaler = GradScaler() if use_amp else None

    def train(self, num_epochs: int):
        for epoch in range(num_epochs):
            avg_metrics = self._train_one_epoch(epoch)
            current_params = {name: sched(epoch) for name, sched in self.param_schedulers.items()}
            self._print_epoch(epoch, avg_metrics, current_params)

            if self.scheduler is not None:
                self.scheduler.step()   # или self.scheduler.step(avg_metrics['loss'])

    def _train_one_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        epoch_metrics = defaultdict(float)
        n_batches = len(self.train_loader)

        current_params = {name: sched(epoch) for name, sched in self.param_schedulers.items()}

        for step, batch in enumerate(tqdm(self.train_loader, desc=f"Epoch {epoch:03d}")):

            # Распаковка батча
            if len(batch) >= 2:
                data, labels = batch[:2]
            else:
                data = batch[0]
                labels = None

            data = data.to(self.device)

            if self.require_labels_in_loss and labels is None:
                raise ValueError(
                    f"Batch {step} does not contain labels, but require_labels_in_loss=True"
                )

            if labels is not None:
                labels = labels.to(self.device)

            # Forward модели в зависимости от режима
            if self.mode == "Conditional":
                if labels is None:
                    raise ValueError("Conditional mode requires labels in batch")
                recon, mu, logvar = self.model(data, labels)
            else:
                recon, mu, logvar = self.model(data)

            with autocast(enabled=self.use_amp):
                total_loss, components = self.loss_fn(
                    recon=recon,
                    x=data,
                    mu=mu,
                    logvar=logvar,
                    labels=labels,                    # всегда передаём (может быть None)
                    **current_params,
                    **self.static_params,
                )

            loss_for_backward = total_loss / self.grad_accum_steps

            if self.use_amp:
                self.scaler.scale(loss_for_backward).backward()
            else:
                loss_for_backward.backward()

            # Метрики — от полной (несмасштабированной) потери
            epoch_metrics["loss"] += total_loss.item()
            for name, val in components.items():
                epoch_metrics[name] += val.item()

            # Шаг оптимизатора только когда накопили
            if (step + 1) % self.grad_accum_steps == 0:
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()

        # Последний неполный шаг
        if (step + 1) % self.grad_accum_steps != 0:
            if self.use_amp:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad()

        return {k: v / n_batches for k, v in epoch_metrics.items()}

    def _print_epoch(self, epoch: int, metrics: Dict[str, float], params: Dict[str, float]):
        parts = [f"Loss: {metrics['loss']:.4f}"]
        for k, v in metrics.items():
            if k == "loss":
                continue
            nice_name = k.replace("_loss", "").replace("_", " ").title()
            parts.append(f"{nice_name}: {v:.4f}")

        param_str = ", ".join(f"{k}={v:.4f}" for k, v in params.items()) if params else ""
        if param_str:
            param_str = f" | {param_str}"

        mode_tag = f"[{self.mode}]" if self.mode == "Unconditional" else ""
        print(f"Epoch {epoch:03d} {mode_tag}| {' | '.join(parts)}{param_str}")