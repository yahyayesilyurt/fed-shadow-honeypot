import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
from skimage.metrics import structural_similarity as ssim


def _to_numpy(x) -> np.ndarray:
    """Converts a torch tensor or NumPy array to a flat float32 NumPy array."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(np.float32)
    return np.array(x, dtype=np.float32)


def _ensure_dir(path: str):
    """Creates directory if it doesn't exist."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


def calculate_utility_metrics(y_true, y_pred) -> dict:
    """
    Measures the diagnostic success of the global model.

    Returns:
        {
          "accuracy": float,  # overall accuracy
          "f1_score": float,  # weighted F1 (reliable for imbalanced classes)
        }
    """
    y_true = _to_numpy(y_true).astype(int)
    y_pred = _to_numpy(y_pred).astype(int)

    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    return {"accuracy": acc, "f1_score": f1}


def calculate_privacy_leakage(original_signal, reconstructed_signal) -> dict:
    """
    Measures how well the attacker reconstructed the original signal.

    Interpretation:
    - High MSE  -> Attacker failed (Honeypot defense working)
    - Low MSE   -> Attacker succeeded (Privacy breach)
    - High SSIM -> Signals are structurally similar (bad)
    - Low SSIM  -> Signals are structurally different (good, defense working)

    Returns:
        {
          "mse":  float,  # Mean Squared Error
          "ssim": float,  # Structural Similarity Index [-1, 1]
        }
    """
    orig  = _to_numpy(original_signal).flatten()
    recon = _to_numpy(reconstructed_signal).flatten()

    mse_val = mean_squared_error(orig, recon)

    # SSIM: data_range = dynamic range for 1D signal
    data_range = orig.max() - orig.min()
    if data_range < 1e-8:
        # Constant signal (all values same) -> comparison meaningless
        ssim_val = 1.0
    else:
        ssim_val = ssim(orig, recon, data_range=float(data_range))

    return {"mse": float(mse_val), "ssim": float(ssim_val)}


def plot_ecg_comparison(
    original,
    reconstructed,
    title: str = "Gradient Inversion Attack Result",
    save_path: str = None
):
    """
    Compares the original ECG signal with the attacker's reconstructed signal.
    """
    orig  = _to_numpy(original).flatten()
    recon = _to_numpy(reconstructed).flatten()

    # Metrics — add to title
    metrics    = calculate_privacy_leakage(orig, recon)
    full_title = f"{title}\nMSE: {metrics['mse']:.4f} | SSIM: {metrics['ssim']:.4f}"

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(orig,  label='Original ECG (Real)',          color='blue',  linewidth=2)
    ax.plot(recon, label="Attacker's Reconstruction (Fake)", color='red',   linewidth=1.5, linestyle='--')

    ax.set_title(full_title, fontsize=12)
    ax.set_xlabel('Time (360 Hz sample index)')
    ax.set_ylabel('Amplitude (Normalized)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        _ensure_dir(save_path)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Plot] Saved: {save_path}")
    else:
        plt.show()

    plt.close(fig)



def plot_training_history(history: dict, save_path: str = None):
    """
    Plots the metric trends by round during FL training.

    Args:
        history: {
            "rounds":    [1, 2, 3, ...],
            "loss":      [0.5, 0.4, ...],
            "accuracy":  [0.80, 0.85, ...],
            "f1_score":  [0.78, 0.83, ...],   # optional
        }
        save_path: file path to save (None -> show on screen)
    """
    rounds   = history.get("rounds",   [])
    loss     = history.get("loss",     [])
    accuracy = history.get("accuracy", [])
    f1       = history.get("f1_score", [])

    has_f1   = len(f1) > 0
    n_plots  = 3 if has_f1 else 2

    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))

    # Loss
    axes[0].plot(rounds, loss, marker='o', color='tomato', linewidth=2)
    axes[0].set_title("Global Model — Loss")
    axes[0].set_xlabel("FL Round")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(rounds, accuracy, marker='s', color='steelblue', linewidth=2)
    axes[1].set_title("Global Model — Accuracy")
    axes[1].set_xlabel("FL Round")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, alpha=0.3)

    # F1 Score (optional)
    if has_f1:
        axes[2].plot(rounds, f1, marker='^', color='seagreen', linewidth=2)
        axes[2].set_title("Global Model — F1 Score (weighted)")
        axes[2].set_xlabel("FL Round")
        axes[2].set_ylabel("F1 Score")
        axes[2].set_ylim(0, 1)
        axes[2].grid(True, alpha=0.3)

    plt.suptitle("Federated Learning — Training History", fontsize=13, y=1.02)
    plt.tight_layout()

    if save_path:
        _ensure_dir(save_path)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Plot] Saved: {save_path}")
    else:
        plt.show()

    plt.close(fig)
