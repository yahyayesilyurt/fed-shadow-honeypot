import torch
import torch.nn.functional as F
import numpy as np
from src.utils import calculate_privacy_leakage


class GradientInversionAttacker:
    """
    Simulates a Deep Leakage from Gradients (DLG) attack.

    Realistic scenario:
    - The attacker listens to network traffic and steals gradients from all clients.
    - These gradients are a mixture of real + honeypot updates.
    - The attacker tries to reconstruct the original ECG signal from mixed gradients.
    - Honeypots disrupt this reconstruction process.
    """

    def __init__(self, global_model, device: torch.device):
        self.model  = global_model
        self.device = device


    def _to_tensor_list(self, gradients) -> list[torch.Tensor]:
        """
        Moves a list of NumPy arrays or tensors onto the target device.
        """
        result = []
        for g in gradients:
            if isinstance(g, np.ndarray):
                g = torch.tensor(g, dtype=torch.float32)
            result.append(g.to(self.device))
        return result


    def _estimate_label(self, target_gradients: list[torch.Tensor]) -> torch.Tensor:
        last_grad = target_gradients[-2]   # fc2.weight gradients → (num_classes, hidden)
        estimated = last_grad.mean(dim=1).argmin().item()
        print(f"  [Attacker] Estimated label: {estimated}")
        return torch.tensor([estimated], dtype=torch.long, device=self.device)


    def reconstruct_signal(
        self,
        target_gradients,
        target_label=None,
        iterations: int = 500,
        lr: float = 0.01,
        tv_weight: float = 1e-4,
        original_signal=None,   
    ) -> dict:
        """
        Runs the gradient inversion optimization loop.

        Args:
            target_gradients: Stolen gradients from the network (list of tensor or ndarray).
            target_label:     Target class label. If None, estimated automatically.
            iterations:       Number of optimization steps.
            lr:               Adam learning rate (0.01 is more stable for DLG).
            tv_weight:        Total Variation regularization weight.
                              Smooths the signal and enforces realistic ECG shape.
            original_signal:  Real ECG (if available, MSE/SSIM are computed).

        Returns:
            {
              "reconstructed": np.ndarray (1, 360),
              "final_loss":    float,
                            "metrics":       dict  (if original_signal is provided)
            }
        """
        print("\n[Attacker] Starting Gradient Inversion attack...")
        self.model.eval()

        target_grads = self._to_tensor_list(target_gradients)

        if target_label is not None:
            dummy_label = torch.tensor([target_label], dtype=torch.long, device=self.device)
        else:
            dummy_label = self._estimate_label(target_grads)

        dummy_data = torch.randn((1, 1, 360), device=self.device, requires_grad=True)

        optimizer = torch.optim.Adam([dummy_data], lr=lr)

        final_loss = float("inf")

        for it in range(iterations):
            optimizer.zero_grad()
            self.model.zero_grad()

            dummy_pred   = self.model(dummy_data)
            dummy_loss   = F.cross_entropy(dummy_pred, dummy_label)
            dummy_grads  = torch.autograd.grad(
                dummy_loss, self.model.parameters(), create_graph=True
            )

            grad_dist = sum(
                ((dg - tg) ** 2).sum()
                for dg, tg in zip(dummy_grads, target_grads)
            )

            tv_loss = torch.sum(torch.abs(dummy_data[:, :, 1:] - dummy_data[:, :, :-1]))

            total_loss = grad_dist + tv_weight * tv_loss
            total_loss.backward()
            optimizer.step()

            with torch.no_grad():
                dummy_data.clamp_(-3.0, 3.0)

            final_loss = grad_dist.item()

            if (it + 1) % 100 == 0:
                print(f"  [Iter {it+1:>4}/{iterations}] "
                      f"Grad Dist: {final_loss:.6f} | "
                      f"TV Loss: {tv_loss.item():.6f}")

        print("[Attacker] Optimization completed.")

        reconstructed_np = dummy_data.detach().cpu().numpy()   

        result = {
            "reconstructed": reconstructed_np,
            "final_loss":    final_loss,
            "metrics":       {},
        }
        if original_signal is not None:
            result["metrics"] = calculate_privacy_leakage(original_signal, reconstructed_np)
            print(f"  [Attacker] MSE : {result['metrics']['mse']:.4f}")
            print(f"  [Attacker] SSIM: {result['metrics']['ssim']:.4f}")

        return result
