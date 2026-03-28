import torch
import flwr as fl
import numpy as np
from collections import OrderedDict
from src.model import train_model, test_model, ECGNet


def get_parameters(net) -> list[np.ndarray]:
    """Returns model weights as a list of NumPy arrays."""
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_parameters(net, parameters: list[np.ndarray]):
    """Loads a list of NumPy arrays into the model's state_dict."""
    params_dict  = zip(net.state_dict().keys(), parameters)
    state_dict   = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


class ECGClient(fl.client.NumPyClient):
    """
    FL client that performs local training on real ECG data.
    """
    def __init__(self, net: ECGNet, train_loader, test_loader, device: torch.device):
        self.net          = net
        self.train_loader = train_loader
        self.test_loader  = test_loader
        self.device       = device

    def get_parameters(self, config):
        return get_parameters(self.net)

    def fit(self, parameters, config):
        set_parameters(self.net, parameters)

        train_loss, train_acc = train_model(
            self.net, self.train_loader, epochs=1, device=self.device
        )

        num_samples = len(self.train_loader.dataset)

        return (
            get_parameters(self.net),
            num_samples,
            {
                "train_loss":  float(train_loss),
                "train_acc":   float(train_acc),
            }
        )

    def evaluate(self, parameters, config):
        set_parameters(self.net, parameters)

        loss, accuracy, f1 = test_model(self.net, self.test_loader, self.device)

        return (
            float(loss),
            len(self.test_loader.dataset),
            {
                "accuracy": float(accuracy),
                "f1_score": float(f1),
            }
        )


class HoneypotClient(fl.client.NumPyClient):
    """
    Fake FL client that generates poisoned gradients.

    How it works:
    - Does not use real data and performs no training.
    - Adds high-variance Gaussian noise on top of the global weights.
    - When an attacker reconstructs these gradients, they receive meaningless signals.
    - The server identifies honeypots by a private client registry, not by any flag in the updates.
    """

    NOISE_SCALE = 5.0

    def __init__(self, net: ECGNet, device: torch.device):
        self.net    = net
        self.device = device

    def get_parameters(self, config):
        return get_parameters(self.net)

    def fit(self, parameters, config):
        poisoned = [
            param + np.random.normal(loc=0.0, scale=self.NOISE_SCALE, size=param.shape).astype(param.dtype)
            for param in parameters
        ]

        return poisoned, 1, {}

    def evaluate(self, parameters, config):
        return 0.0, 1, {"accuracy": 0.0, "f1_score": 0.0}