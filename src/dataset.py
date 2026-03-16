import os
import glob
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split

# Meaningful beat symbols in MIT-BIH (excluding noise/note markers like '+', '~', etc.)
NOISE_SYMBOLS = {'+', '~', '|', 'x', ']', '[', '!', '"', '@'}

class ECGDataset(Dataset):
    """
    PyTorch Dataset for 1D ECG signals.
    Output shape: (1, window_size) — ready for 1D CNN.
    """
    def __init__(self, signals, labels):
        # Z-score normalization per beat (mean=0, std=1)
        # Prevents exploding/vanishing gradients during training
        signals = signals.astype(np.float32)
        mean = signals.mean(axis=1, keepdims=True)
        std  = signals.std(axis=1, keepdims=True) + 1e-8  
        signals = (signals - mean) / std

        self.signals = torch.tensor(signals, dtype=torch.float32).unsqueeze(1)  
        self.labels  = torch.tensor(labels,  dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.signals[idx], self.labels[idx]


def extract_beats(ekg_path, ann_path, window_size=360):
    """
    Reads EKG + annotation files for a single patient,
    and extracts fixed-window beats around each QRS peak.

    Args:
        ekg_path: path to '{id}_ekg.csv'
        ann_path: path to '{id}_annotations_1.csv'
        window_size: window size in samples (360 = 1 second @ 360 Hz)

    Returns:
        signals: np.array shape (N, window_size)
        labels:  np.array shape (N,)  — 0: Normal, 1: Arrhythmia
    """
    # --- Read ECG signal ---
    ekg_df = pd.read_csv(ekg_path, low_memory=False)

    signal_col = ekg_df.columns[1]
    signal_values = ekg_df[signal_col].values.astype(np.float64)

    # --- Read annotations ---
    ann_df = pd.read_csv(ann_path, low_memory=False)
    ann_df.columns = ann_df.columns.str.strip()  

    half = window_size // 2
    signals, labels = [], []

    for _, row in ann_df.iterrows():
        peak_idx = int(row['index'])
        symbol   = str(row['annotation_symbol']).strip()

        # Skip noise/note symbols
        if symbol in NOISE_SYMBOLS:
            continue

        # Check file boundaries
        if peak_idx - half < 0 or peak_idx + half >= len(signal_values):
            continue

        beat = signal_values[peak_idx - half : peak_idx + half]  # (window_size,)
        label = 0 if symbol == 'N' else 1                         # Binary: Normal / Arrhythmia

        signals.append(beat)
        labels.append(label)

    return np.array(signals, dtype=np.float32), np.array(labels, dtype=np.int64)


def load_fl_data(data_dir, num_real_clients=16, max_records=None, window_size=360):
    """
    Reads all patient records and splits them among FL clients.

    Args:
        data_dir:        'data/raw' directory
        num_real_clients: number of real FL clients to split into
        max_records:     None = all records, int = first N records (for quick testing)
        window_size:     beat window size

    Returns:
        client_loaders: List[DataLoader]  — one per real client
        test_loader:    DataLoader        — global test set
    """
    # Find {id}_ekg.csv + {id}_annotations_1.csv pairs for each patient
    ekg_files = sorted(glob.glob(os.path.join(data_dir, '*_ekg.csv')))

    if max_records is not None:
        ekg_files = ekg_files[:max_records]

    all_signals, all_labels = [], []

    for ekg_path in ekg_files:
        # Derive annotation file path: 100_ekg.csv → 100_annotations_1.csv
        base_id  = os.path.basename(ekg_path).replace('_ekg.csv', '')
        ann_path = os.path.join(data_dir, f'{base_id}_annotations_1.csv')

        if not os.path.exists(ann_path):
            print(f"[SKIP] Annotation not found: {ann_path}")
            continue

        print(f"[READ] Processing {base_id}...")
        sig, lbl = extract_beats(ekg_path, ann_path, window_size)

        if len(sig) == 0:
            print(f"[WARN] No beats extracted for {base_id}, skipping.")
            continue

        all_signals.append(sig)
        all_labels.append(lbl)

    X = np.concatenate(all_signals, axis=0)
    y = np.concatenate(all_labels,  axis=0)

    # --- Class distribution report ---
    n_normal      = (y == 0).sum()
    n_arrhythmia  = (y == 1).sum()
    print(f"\nTotal beats: {len(X)} | Normal: {n_normal} ({n_normal/len(X)*100:.1f}%) | Arrhythmia: {n_arrhythmia} ({n_arrhythmia/len(X)*100:.1f}%)")

    # --- Train / Test split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y  
    )

    split_size = len(X_train) // num_real_clients
    client_loaders = []

    for i in range(num_real_clients):
        start = i * split_size
        end   = start + split_size
        X_c, y_c = X_train[start:end], y_train[start:end]
        ds = ECGDataset(X_c, y_c)

        class_counts  = np.bincount(y_c, minlength=2).astype(np.float32)
        class_weights = 1.0 / (class_counts + 1e-8)
        sample_weights = class_weights[y_c]
        sampler = WeightedRandomSampler(
            weights=torch.tensor(sample_weights, dtype=torch.float32),
            num_samples=len(y_c),
            replacement=True
        )

        loader = DataLoader(ds, batch_size=32, sampler=sampler)
        client_loaders.append(loader)

    test_loader = DataLoader(
        ECGDataset(X_test, y_test),
        batch_size=64,
        shuffle=False
    )

    return client_loaders, test_loader


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    RAW_DIR  = os.path.join(BASE_DIR, "data", "raw")

    client_loaders, test_loader = load_fl_data(RAW_DIR, num_real_clients=16, max_records=5)

    print(f"\nNumber of clients created  : {len(client_loaders)}")
    batch_x, batch_y = next(iter(client_loaders[0]))
    print(f"First client batch shape   : {batch_x.shape}")   
    print(f"First client label sample  : {batch_y[:8]}")
    print(f"Test loader batch count    : {len(test_loader)}")