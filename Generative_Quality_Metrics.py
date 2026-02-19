import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.stats import entropy
from scipy.linalg import sqrtm
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from tqdm import tqdm
import matplotlib.pyplot as plt


def denorm(x):
    """Приводит изображения из [-1,1] → [0,1]"""
    return (x + 1) / 2


class PSNR_SSIM_Evaluator:
    """
    Вычисление PSNR и SSIM с бутстрэпом доверительных интервалов.
    
    Использование:
        evaluator = PSNR_SSIM_Evaluator(device)
        mean_psnr, ci_psnr, mean_ssim, ci_ssim = evaluator.compute(model, data_loader)
    """
    def __init__(self, device, num_bootstraps=1000, seed=42):
        self.device = device
        self.num_bootstraps = num_bootstraps
        self.seed = seed
        
        self.psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
        self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    @torch.no_grad()
    def compute(self, model, data_loader):
        model.eval()
        
        psnr_values = []
        ssim_values = []
        
        for data, labels in tqdm(data_loader, desc="PSNR/SSIM"):
            data = data.to(self.device)
            labels = labels.to(self.device)
            
            z = torch.randn(data.size(0), 32, 25, 30).to(self.device)
            generated = model.decoder(z, labels)
            
            data_01  = denorm(data).clamp(0, 1)
            gen_01   = denorm(generated).clamp(0, 1)
            
            psnr_val = self.psnr_metric(data_01, gen_01)
            ssim_val = self.ssim_metric(data_01, gen_01)
            
            psnr_values.append(psnr_val.item())
            ssim_values.append(ssim_val.item())
        
        psnr_values = np.array(psnr_values)
        ssim_values = np.array(ssim_values)
        N = len(psnr_values)
        
        # Бутстрэп
        np.random.seed(self.seed)
        boot_PSNR = []
        boot_SSIM = []
        
        for _ in tqdm(range(self.num_bootstraps), desc="Bootstrap PSNR/SSIM"):
            idx = np.random.choice(N, size=N, replace=True)
            boot_PSNR.append(np.mean(psnr_values[idx]))
            boot_SSIM.append(np.mean(ssim_values[idx]))
        
        # Результаты
        mean_psnr, ci_low_p, ci_high_p = np.mean(boot_PSNR), *np.percentile(boot_PSNR, [2.5, 97.5])
        mean_ssim, ci_low_s, ci_high_s = np.mean(boot_SSIM), *np.percentile(boot_SSIM, [2.5, 97.5])
        
        print(f"Mean PSNR: {mean_psnr:.4f} dB  [95% CI: {ci_low_p:.4f} – {ci_high_p:.4f}]")
        print(f"Mean SSIM: {mean_ssim:.4f}     [95% CI: {ci_low_s:.4f} – {ci_high_s:.4f}]")
        
        return mean_psnr, (ci_low_p, ci_high_p), mean_ssim, (ci_low_s, ci_high_s)


class InceptionScoreEvaluator:
    """
    Вычисление Inception Score (IS) с бутстрэпом.
    
    Использование:
        evaluator = InceptionScoreEvaluator(device)
        mean_is, ci_is = evaluator.compute(classifier, model, data_loader)
    """
    def __init__(self, device, num_bootstraps=1000, seed=42):
        self.device = device
        self.num_bootstraps = num_bootstraps
        self.seed = seed

    @torch.no_grad()
    def compute(self, classifier, model, data_loader):
        classifier.eval()
        model.eval()
        
        preds = []
        for data, labels in tqdm(data_loader, desc="Collecting IS predictions"):
            data = data.to(self.device)
            labels = labels.to(self.device)
            z = torch.randn(data.size(0), 32, 25, 30).to(self.device)
            generated = model.decoder(z, labels)
            outputs = classifier(generated)
            probs = F.softmax(outputs, dim=1).cpu().numpy()
            preds.append(probs)
        
        preds = np.concatenate(preds, axis=0)
        N = preds.shape[0]
        
        # Бутстрэп
        np.random.seed(self.seed)
        boot_scores = []
        for _ in tqdm(range(self.num_bootstraps), desc="Bootstrap IS"):
            idx = np.random.choice(N, size=N, replace=True)
            boot_preds = preds[idx]
            py = np.mean(boot_preds, axis=0)
            kl_divs = [entropy(p_yx, py) for p_yx in boot_preds]
            score = np.exp(np.mean(kl_divs))
            boot_scores.append(score)
        
        mean_is = np.mean(boot_scores)
        ci_low, ci_high = np.percentile(boot_scores, [2.5, 97.5])
        
        print(f"Mean Inception Score: {mean_is:.3f}  [95% CI: {ci_low:.3f} – {ci_high:.3f}]")
        
        return mean_is, (ci_low, ci_high)


class FeatureExtractor(nn.Module):
    def __init__(self, classifier):
        super().__init__()
        # Берём всё кроме последнего линейного слоя
        self.features = nn.Sequential(*list(classifier.children())[:-1])
    
    def forward(self, x):
        x = self.features(x)
        return x.view(x.size(0), -1)  # [B, 512]


class FID_Evaluator:
    """
    Вычисление per-class FID + средний FID с бутстрэпом.
    Также рисует красивую гистограмму.
    
    Использование:
        evaluator = FID_Evaluator(device, class_names)
        overall_mean_fid = evaluator.compute(classifier, model, data_loader)
    """
    def __init__(self, device, class_names, num_bootstraps=300, seed=42, feat_dim=512):
        self.device = device
        self.class_names = class_names          # список строк, например crystals
        self.num_classes = len(class_names)
        self.num_bootstraps = num_bootstraps
        self.seed = seed
        self.feat_dim = feat_dim

    @staticmethod
    def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
        diff = mu1 - mu2
        covmean = sqrtm(sigma1.dot(sigma2)).real
        return diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean) + eps

    @torch.no_grad()
    def compute(self, classifier, model, data_loader):
        model.eval()
        extractor = FeatureExtractor(classifier).to(self.device)
        extractor.eval()

        real_features_dict = {c: [] for c in range(self.num_classes)}
        gen_features_dict  = {c: [] for c in range(self.num_classes)}

        for data, labels in tqdm(data_loader, desc="Collecting FID features"):
            data = data.to(self.device)
            labels = labels.to(self.device)
            
            real_feats = extractor(data).cpu().numpy()
            
            z = torch.randn(data.size(0), 32, 25, 30).to(self.device)
            generated = model.decoder(z, labels)
            gen_feats = extractor(generated).cpu().numpy()
            
            for i in range(data.size(0)):
                c = labels[i].item()
                real_features_dict[c].append(real_feats[i])
                gen_features_dict[c].append(gen_feats[i])

        # Преобразуем в numpy массивы
        for c in range(self.num_classes):
            if real_features_dict[c]:
                real_features_dict[c] = np.stack(real_features_dict[c])
            else:
                real_features_dict[c] = np.empty((0, self.feat_dim))
                
            if gen_features_dict[c]:
                gen_features_dict[c] = np.stack(gen_features_dict[c])
            else:
                gen_features_dict[c] = np.empty((0, self.feat_dim))

        # Бутстрэп FID по классам
        np.random.seed(self.seed)
        per_class_results = {}

        for c in range(self.num_classes):
            real = real_features_dict[c]
            gen  = gen_features_dict[c]
            N_real = len(real)
            N_gen  = len(gen)

            if N_real < 2 or N_gen < 2:
                per_class_results[c] = {'mean': np.nan, 'ci_low': np.nan, 'ci_high': np.nan}
                continue

            boot_fids = []
            for _ in tqdm(range(self.num_bootstraps), desc=f"FID class {self.class_names[c]}", leave=False):
                idx_r = np.random.choice(N_real, N_real, replace=True)
                idx_g = np.random.choice(N_gen,  N_gen,  replace=True)

                mu_r = np.mean(real[idx_r], axis=0)
                sg_r = np.cov(real[idx_r], rowvar=False)

                mu_g = np.mean(gen[idx_g], axis=0)
                sg_g = np.cov(gen[idx_g], rowvar=False)

                fid = self.frechet_distance(mu_r, sg_r, mu_g, sg_g)
                boot_fids.append(fid)

            mean_fid = np.mean(boot_fids)
            ci_low, ci_high = np.percentile(boot_fids, [2.5, 97.5])
            per_class_results[c] = {'mean': mean_fid, 'ci_low': ci_low, 'ci_high': ci_high}

        # Вывод и график
        means = []
        errors_low = []
        errors_high = []

        print("\nPer-class FID:")
        for c in range(self.num_classes):
            res = per_class_results.get(c, {'mean': np.nan, 'ci_low': np.nan, 'ci_high': np.nan})
            m, l, h = res['mean'], res['ci_low'], res['ci_high']
            print(f"Class {self.class_names[c]:<20}  FID = {m:.3f}  [95% CI: {l:.3f} – {h:.3f}]")
            
            means.append(m)
            errors_low.append(m - l if not np.isnan(l) else 0)
            errors_high.append(h - m if not np.isnan(h) else 0)

        overall_mean = np.nanmean(means)
        print(f"\nOverall mean FID (across classes): {overall_mean:.3f}\n")

        # Гистограмма
        x = np.arange(self.num_classes)
        errors = np.array([errors_low, errors_high])

        plt.figure(figsize=(14, 7))
        plt.bar(x, means, yerr=errors, capsize=5, alpha=0.75, color='cornflowerblue', edgecolor='navy')
        plt.axhline(overall_mean, color='darkred', ls='--', lw=2, label=f'Overall mean FID: {overall_mean:.2f}')
        
        plt.xticks(x, self.class_names, rotation=90, ha='right', fontsize=9)
        plt.xlabel('Class')
        plt.ylabel('Mean FID')
        plt.title('Per-class FID with 95% Confidence Intervals')
        plt.legend()
        plt.tight_layout()
        plt.show()

        return overall_mean