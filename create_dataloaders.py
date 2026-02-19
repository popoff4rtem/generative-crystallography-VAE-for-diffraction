# create_dataloaders.py

import pickle
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def load_and_filter_dataframe(
    pkl_path='./datasets/seg_maps_dataset.pkl',
    allowed_stats=(1e8, 2e8, 5e8)
):
    """
    Загружает pickle-файл и фильтрует по значениям Stats.
    """
    with open(pkl_path, 'rb') as f:
        df = pickle.load(f)

    df = df[df['Stats'].isin(allowed_stats)].copy()

    return df


def prepare_datasets(
    df,
    test_size=0.1,
    random_state=42,
    batch_size=10,
    device='cpu'
):
    """
    Готовит train/test DataLoader'ы без какой-либо интерполяции.
    Работает с оригинальными размерами матриц.

    Возвращает:
        train_loader, test_loader, label_encoder, crystals (np.array)
    """

    # Список уникальных кристаллов (отсортированный для удобства)
    crystals = np.sort(df['Crystal'].unique())

    # Кодируем метки классов
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(df['Crystal'])

    # Features — матрицы дифракции
    X = df['Matrix'].values  # массив объектов (списков/массивов)

    # Разделение
    (X_train, X_test,
     y_train, y_test) = train_test_split(
        X, y_encoded,
        test_size=test_size,
        random_state=random_state,
        stratify=y_encoded  # сохраняем баланс классов
    )

    # Преобразование в тензоры [N, 1, H, W]
    def matrices_to_tensor(matrices_list):
        # предполагаем, что все матрицы имеют одинаковый размер
        arr = np.stack(matrices_list)           # [N, H, W]
        arr = arr[:, np.newaxis, :, :]          # [N, 1, H, W]
        tensor = torch.from_numpy(arr).float()
        tensor = tensor * 2.0 - 1.0             # в [-1, +1]
        return tensor.to(device)

    Diff_train_tensor = matrices_to_tensor(X_train)
    Diff_test_tensor  = matrices_to_tensor(X_test)

    y_train_tensor = torch.from_numpy(y_train).long().to(device)
    y_test_tensor  = torch.from_numpy(y_test).long().to(device)

    # Датасеты
    train_dataset = TensorDataset(Diff_train_tensor, y_train_tensor)
    test_dataset  = TensorDataset(Diff_test_tensor, y_test_tensor)

    # DataLoader'ы
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == 'cuda')
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == 'cuda')
    )

    return train_loader, test_loader, label_encoder, crystals

def get_dataloaders(
    pkl_path='./datasets/seg_maps_dataset.pkl',
    allowed_stats=(1e8, 2e8, 5e8),
    test_size=0.1,
    batch_size=10,
    random_state=42,
    device=None
):
    """
    Основная функция-обёртка.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif isinstance(device, str):
        device = torch.device(device)           # ← главное исправление
    elif not isinstance(device, torch.device):
        raise ValueError("device должен быть str ('cuda'/'cpu') или torch.device")

    df = load_and_filter_dataframe(pkl_path, allowed_stats)
    return prepare_datasets(
        df=df,
        test_size=test_size,
        random_state=random_state,
        batch_size=batch_size,
        device=device
    )