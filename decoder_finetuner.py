# decoder_finetuner.py
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from typing import Optional, Literal, Callable

class DecoderFinetuner:
    """
    Тренер для фазы fine-tuning только декодера VAE.
    
    Особенности:
    - Замораживает энкодер (и головы mu/logvar) на старте
    - Сбрасывает состояние оптимизатора для замороженных параметров
    - Использует только reconstruction loss (по умолчанию MSE)
    - Поддерживает mixed precision (AMP)
    - Работает в двух режимах вызова модели: Conditional / Unconditional
    - Минимум зависимостей, максимально прозрачный код
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        mode: Literal["Conditional", "Unconditional"] = "Conditional",
        recon_loss_fn: Callable = F.mse_loss,
        device: Optional[str] = None,
        use_amp: bool = True,
        freeze_encoder_on_init: bool = True,
    ):
        self.mode = mode
        if mode not in ("Conditional", "Unconditional"):
            raise ValueError("mode must be 'Conditional' or 'Unconditional'")

        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.recon_loss_fn = recon_loss_fn
        self.use_amp = use_amp
        self.scaler = GradScaler() if use_amp else None

        if freeze_encoder_on_init:
            self.freeze_encoder_and_reset_optimizer()

    def freeze_encoder_and_reset_optimizer(self):
        """Замораживает энкодер и головы латентного распределения + очищает их состояние в оптимизаторе"""
        self.model.encoder.eval()
        if hasattr(self.model, 'fc_mu'):
            self.model.fc_mu.eval()
        if hasattr(self.model, 'fc_logvar'):
            self.model.fc_logvar.eval()

        frozen_modules = [self.model.encoder]
        if hasattr(self.model, 'fc_mu'):
            frozen_modules.append(self.model.fc_mu)
        if hasattr(self.model, 'fc_logvar'):
            frozen_modules.append(self.model.fc_logvar)

        for module in frozen_modules:
            for p in module.parameters():
                p.requires_grad = False
                # Очистка состояния оптимизатора, если параметр там был
                if p in self.optimizer.state:
                    self.optimizer.state[p].clear()

        print("Encoder (and mu/logvar heads) frozen and optimizer state cleared.")

    def train(self, num_epochs: int):
        for epoch in range(num_epochs):
            avg_loss = self._train_one_epoch(epoch)
            self._print_epoch(epoch, avg_loss)

    def _train_one_epoch(self, epoch: int) -> float:
        self.model.decoder.train()   # только декодер в train-режиме
        total_loss = 0.0
        n_batches = len(self.train_loader)

        for step, batch in enumerate(tqdm(self.train_loader, desc=f"Decoder FT Epoch {epoch:03d}")):

            # Распаковка батча
            data, labels = self._prepare_batch(batch)

            # Энкодер → z (без градиентов)
            with torch.no_grad():
                if self.mode == "Conditional":
                    z, mu, logvar = self.model.encode(data, labels)
                else:
                    z, mu, logvar = self.model.encode(data)

            # Декодер → реконструкция
            with autocast(enabled=self.use_amp):
                if self.mode == "Conditional":
                    x_recon = self.model.decoder(z, labels)
                else:
                    x_recon = self.model.decoder(z)

                loss = self.recon_loss_fn(x_recon, data)

            # Backward + шаг
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            self.optimizer.zero_grad()

            total_loss += loss.item()

        return total_loss / n_batches

    def _prepare_batch(self, batch) -> tuple:
        if len(batch) >= 2:
            data, labels = batch[:2]
        else:
            data = batch[0]
            labels = None

        data = data.to(self.device)
        if labels is not None:
            labels = labels.to(self.device)

        if self.mode == "Conditional" and labels is None:
            raise ValueError("Conditional mode requires labels in batch")

        return data, labels

    def _print_epoch(self, epoch: int, avg_loss: float):
        mode_tag = f"[{self.mode}]" if self.mode == "Unconditional" else ""
        print(f"Decoder FT Epoch {epoch:03d} {mode_tag}| Loss: {avg_loss:.6f}")