import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class UnConditionalVAE(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        
        #nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        
        # ---- Encoder ----
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = 4, stride = 2, padding = 1),
            nn.ReLU(),
            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 4, stride = 2, padding = 1),
            nn.ReLU(),
            nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 4, stride = 2, padding = 1),
            nn.ReLU(),
            nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 4, stride = 2, padding = 1),
            nn.ReLU()
        )
        self.flatten_dim = 256 * (250 // 16) * (480 // 16)  # после 4-х свёрток stride=2
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)
        
        # ---- Decoder ----
        self.fc_decode = nn.Linear(latent_dim, self.flatten_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels = 256, out_channels = 128, kernel_size = 4, stride = 2, padding = 1),
            nn.ConstantPad2d((0, 0, 1, 0), 0),  # паддинг (left, right, top, bottom)
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels = 128, out_channels = 64, kernel_size = 4, stride = 2, padding = 1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels = 64, out_channels = 32, kernel_size = 4, stride = 2, padding = 1),
            nn.ConstantPad2d((0, 0, 1, 0), 0),  # паддинг (left, right, top, bottom)
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels = 32, out_channels = 1, kernel_size = 4, stride = 2, padding = 1),
            nn.Tanh()
        )
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):

        # ---- Encoder ----
        h = self.encoder(x)
        h_flat = h.view(h.size(0), -1)
        mu, logvar = self.fc_mu(h_flat), self.fc_logvar(h_flat)
        z = self.reparameterize(mu, logvar)
        
        # ---- Decoder ----
        h_dec = self.fc_decode(z)
        h_dec = h_dec.view(h_dec.size(0), 256, (250 // 16), (480 // 16))
        x_recon = self.decoder(h_dec)
        return x_recon, mu, logvar

class SpatialVAE(nn.Module):
    def __init__(self, latent_channels=256):
        super().__init__()
        # Энкодер: [B, 1, 240, 480] → [B, 256, 15, 30] (compression 16x)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = 4, stride = 2, padding = 1),
            nn.ReLU(),
            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 4, stride = 2, padding = 1),
            nn.ReLU(),
            nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 4, stride = 2, padding = 1),
            nn.ReLU(),
            nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 4, stride = 2, padding = 1),
            nn.ReLU(),
            nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU()
        )
        
        # Проекции для μ и σ
        self.fc_mu = nn.Conv2d(latent_channels, latent_channels, 1)
        self.fc_logvar = nn.Conv2d(latent_channels, latent_channels, 1)
        
        # Декодер: [B, 256, 15, 30] → [B, 1, 240, 480]
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels = 256, out_channels = 128, kernel_size = 4, stride = 2, padding = 1),
            nn.ConstantPad2d((0, 0, 1, 0), 0),  # паддинг (left, right, top, bottom)
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels = 128, out_channels = 64, kernel_size = 4, stride = 2, padding = 1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels = 64, out_channels = 32, kernel_size = 4, stride = 2, padding = 1),
            nn.ConstantPad2d((0, 0, 1, 0), 0),  # паддинг (left, right, top, bottom)
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels = 32, out_channels = 1, kernel_size = 4, stride = 2, padding = 1),
            nn.Tanh()
        )
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def encode(self, x):
        h = self.encoder(x)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        return self.reparameterize(mu, logvar), mu, logvar
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        
        # ---- Encoder ----
        h = self.encoder(x)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        
        # ---- Decoder ----
        x_recon = self.decoder(z)
        return x_recon, mu, logvar
    

class ResNetAutoEncoder(nn.Module):
    def __init__(self, latent_channels=256, pretrained=False):
        super().__init__()
        # Загружаем ResNet-18
        resnet = models.resnet18(pretrained=pretrained)
        # Адаптируем входной слой для 1 канала (вместо 3)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if pretrained:
            # Перенос весов для первого слоя (усредняем по каналам RGB)
            pretrained_weights = resnet.conv1.weight.mean(dim=1, keepdim=True)
            self.conv1.weight = nn.Parameter(pretrained_weights)
        
        # Остальная часть ResNet-18 (без fc-слоя)
        self.encoder = nn.Sequential(
            self.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2
        )

        self.layer3_enc = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True) 
        )

        self.layer4_enc = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True) 
        )

        # Проекции для μ и σ
        self.fc_mu = nn.Conv2d(512, latent_channels, 1)
        self.fc_logvar = nn.Conv2d(512, latent_channels, 1)

        self.layer4_dec = nn.Sequential(
            nn.Conv2d(latent_channels, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True) 
        )

        self.layer3_dec = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True) 
        )

        self.decoder = nn.Sequential(
            self.layer4_dec,
            self.layer3_dec,
            nn.ConstantPad2d((0, 0, 1, 0), 0),  # паддинг (left, right, top, bottom)
            nn.ConvTranspose2d(in_channels = 128, out_channels = 64, kernel_size = 4, stride = 2, padding = 1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels = 64, out_channels = 32, kernel_size = 4, stride = 2, padding = 1),
            nn.ConstantPad2d((0, 0, 1, 0), 0),  # паддинг (left, right, top, bottom)
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels = 32, out_channels = 1, kernel_size = 4, stride = 2, padding = 1),
            nn.Tanh()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # Энкодер
        h = self.encoder(x)

        # Масштабируем до нужного размера
        h = h[:, :, :-2, :]
        h = self.layer3_enc(h)
        h = self.layer4_enc(h)

        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)

        # Декодер
        x_recon = self.decoder(z)

        return x_recon, mu, logvar

    def encode(self, x):
        # Энкодер
        h = self.encoder(x)

        # Масштабируем до нужного размера
        h = h[:, :, :-2, :]
        h = self.layer3_enc(h)
        h = self.layer4_enc(h)

        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        return self.reparameterize(mu, logvar), mu, logvar
    
    def decode(self, z):
        return self.decoder(z)

class ResNetAutoEncoder(nn.Module):
    def __init__(self, latent_channels=256, pretrained=False):
        super().__init__()
        # Загружаем ResNet-18
        resnet = models.resnet18(pretrained=pretrained)
        # Адаптируем входной слой для 1 канала (вместо 3)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if pretrained:
            # Перенос весов для первого слоя (усредняем по каналам RGB)
            pretrained_weights = resnet.conv1.weight.mean(dim=1, keepdim=True)
            self.conv1.weight = nn.Parameter(pretrained_weights)
        
        # Остальная часть ResNet-18 (без fc-слоя)
        self.encoder = nn.Sequential(
            self.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2
        )

        self.down = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(8, 1), stride=(1, 2), padding=(0, 0)),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True) 
        )

        self.layer3_enc = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True) 
        )

        self.layer4_enc = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True) 
        )

        # self.latent_down = LatentDownsampler(512)

        # Проекции для μ и σ
        self.fc_mu = nn.Conv2d(512, latent_channels, 1)
        self.fc_logvar = nn.Conv2d(512, latent_channels, 1)

        #self.latent_up = LatentUpsampler(latent_channels)

        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels=latent_channels, out_channels=latent_channels, kernel_size=(8, 1), stride=(1, 2), padding=(1, 0)),
            nn.ConstantPad2d((0, 1, 0, 0), 0),  # паддинг (left, right, top, bottom)
            nn.BatchNorm2d(latent_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(latent_channels, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True) 
        )

        self.layer4_dec = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True) 
        )

        self.layer3_dec = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True) 
        )

        self.decoder = nn.Sequential(
            self.layer4_dec,
            self.layer3_dec,
            nn.ConstantPad2d((0, 0, 1, 0), 0),  # паддинг (left, right, top, bottom)
            nn.ConvTranspose2d(in_channels = 128, out_channels = 64, kernel_size = 4, stride = 2, padding = 1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels = 64, out_channels = 32, kernel_size = 4, stride = 2, padding = 1),
            nn.ConstantPad2d((0, 0, 1, 0), 0),  # паддинг (left, right, top, bottom)
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels = 32, out_channels = 1, kernel_size = 4, stride = 2, padding = 1),
            nn.Tanh()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # Энкодер
        h = self.encoder(x)
        h = self.down(h)
        h = self.layer3_enc(h)
        h = self.layer4_enc(h)

        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)

        # Декодер
        z = self.up(z)
        x_recon = self.decoder(z)

        return x_recon, mu, logvar

    def encode(self, x):
        # Энкодер
        h = self.encoder(x)
        h = self.down(h)
        h = self.layer3_enc(h)
        h = self.layer4_enc(h)

        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        return self.reparameterize(mu, logvar), mu, logvar
    
    def decode(self, z):
        z = self.up(z)
        return self.decoder(z)