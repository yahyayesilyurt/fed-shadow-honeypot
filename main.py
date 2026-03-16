import os
import torch
import flwr as fl
import numpy as np

from src.dataset  import load_fl_data
from src.model    import ECGNet
from src.client   import ECGClient, HoneypotClient, get_parameters, set_parameters
from src.server   import ShadowHoneypotStrategy
from src.attacker import GradientInversionAttacker
from src.utils    import plot_ecg_comparison, plot_training_history

NUM_REAL_CLIENTS = 16
NUM_HONEYPOTS    = 4
TOTAL_CLIENTS    = NUM_REAL_CLIENTS + NUM_HONEYPOTS
NUM_ROUNDS       = 10        


def run_attack_demo(device: torch.device, test_loader, output_dir: str = "results"):
    """
    Compares three scenarios:
    A) Attack on real client gradients                 -> Privacy breach (high SSIM)
    B) Attack on honeypot gradients                    -> Defense works (low SSIM)
    C) Attack on mixed (real + honeypot) gradients     -> Real threat model
    """
    os.makedirs(output_dir, exist_ok=True)
    print("\n" + "="*60)
    print("GRADIENT INVERSION ATTACK DEMO")
    print("="*60)

    global_model = ECGNet().to(device)
    attacker     = GradientInversionAttacker(global_model, device)

    real_signals, real_labels = next(iter(test_loader))
    target_signal = real_signals[0:1].to(device)          
    target_label  = real_labels[0].item()
    print(f"Target: {'Arrhythmia' if target_label == 1 else 'Normal'} (Class {target_label})")

    # --- Compute real client gradients ---
    global_model.zero_grad()
    real_pred  = global_model(target_signal)
    real_loss  = torch.nn.functional.cross_entropy(
        real_pred, torch.tensor([target_label], device=device)
    )
    real_gradients = list(torch.autograd.grad(real_loss, global_model.parameters()))

    # -- Scenario A: Real gradient attack --
    print("\n[A] Attacking real client gradients...")
    result_real = attacker.reconstruct_signal(
        real_gradients,
        target_label=None,      
        iterations=1000,
        original_signal=target_signal.cpu().numpy(),
    )
    plot_ecg_comparison(
        original=target_signal.cpu().numpy(),
        reconstructed=result_real["reconstructed"],
        title="Scenario A - Real Client Attack (Defense OFF)",
        save_path=os.path.join(output_dir, "attack_real.png"),
    )

    # -- Scenario B: Honeypot gradient attack --
    print("\n[B] Attacking honeypot gradients...")
    honeypot_gradients = [
        g + torch.randn_like(g) * HoneypotClient.NOISE_SCALE
        for g in real_gradients
    ]
    result_honeypot = attacker.reconstruct_signal(
        honeypot_gradients,
        target_label=None,
        iterations=1000,
        original_signal=target_signal.cpu().numpy(),
    )
    plot_ecg_comparison(
        original=target_signal.cpu().numpy(),
        reconstructed=result_honeypot["reconstructed"],
        title="Scenario B - Honeypot Attack (Defense ON)",
        save_path=os.path.join(output_dir, "attack_honeypot.png"),
    )

    # -- Scenario C: Mixed gradient attack (real threat model) --
    # The attacker sees both real and honeypot updates on the network
    # and cannot distinguish them - so averages both
    print("\n[C] Attacking mixed (real + honeypot) gradients...")
    mixed_gradients = [
        (rg + hg) / 2.0
        for rg, hg in zip(real_gradients, honeypot_gradients)
    ]
    result_mixed = attacker.reconstruct_signal(
        mixed_gradients,
        target_label=None,
        iterations=1000,
        original_signal=target_signal.cpu().numpy(),
    )
    plot_ecg_comparison(
        original=target_signal.cpu().numpy(),
        reconstructed=result_mixed["reconstructed"],
        title="Scenario C - Mixed Gradient Attack (Real Threat Model)",
        save_path=os.path.join(output_dir, "attack_mixed.png"),
    )

    # -- Summary table --------------------------------------------------------
    print("\n" + "="*60)
    print("ATTACK RESULTS SUMMARY")
    print(f"{'Scenario':<40} {'MSE':>10} {'SSIM':>10}")
    print("-"*60)
    for label, result in [
        ("A - Real client (defense off)", result_real),
        ("B - Honeypot (defense on)",     result_honeypot),
        ("C - Mixed gradients",            result_mixed),
    ]:
        m = result["metrics"]
        print(f"{label:<40} {m['mse']:>10.4f} {m['ssim']:>10.4f}")
    print("="*60)
    print("Expected: low MSE/high SSIM in A, high MSE/low SSIM in B and C")

def run_fl_simulation(client_fn, num_clients, num_rounds, strategy, device, test_loader):
    """
    Manual FL loop without Ray/Flower simulation.
    Windows-compatible, with full control over each round.
    """
    from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
    from src.model import ECGNet, test_model
    import numpy as np

    # Initialize all clients
    clients = [client_fn(str(i)) for i in range(num_clients)]

    # Global model - initial weights
    global_net = ECGNet().to(device)
    global_params = get_parameters(global_net)  # List[np.ndarray]

    # Track history
    history = {"rounds": [], "loss": [], "accuracy": [], "f1_score": []}

    for round_num in range(1, num_rounds + 1):
        print(f"\n{'='*60}")
        print(f"FL ROUND {round_num}/{num_rounds}")
        print(f"{'='*60}")

        fit_results = []
        for client in clients:
            # Flower NumPyClient.fit() → (params, num_examples, metrics)
            updated_params, num_examples, metrics = client.numpy_client.fit(
                global_params, config={}
            )
            fit_results.append((updated_params, num_examples, metrics))

        # Send to strategy in Flower format
        from flwr.common import FitRes, Parameters
        from flwr.server.client_proxy import ClientProxy

        # Create mock proxy + FitRes objects for aggregate_fit
        flower_results = []
        for i, (params, num_ex, metrics) in enumerate(fit_results):
            fit_res = FitRes(
                status=None,
                parameters=ndarrays_to_parameters(params),
                num_examples=num_ex,
                metrics=metrics,
            )
            flower_results.append((clients[i], fit_res))

        aggregated_params, fit_metrics = strategy.aggregate_fit(
            server_round=round_num,
            results=flower_results,
            failures=[],
        )

        if aggregated_params is None:
            print(f"[WARN] Round {round_num} skipped!")
            continue

        global_params = parameters_to_ndarrays(aggregated_params)
        set_parameters(global_net, global_params)

        from src.model import test_model
        loss, accuracy, f1 = test_model(global_net, test_loader, device)

        print(f"\n[Round {round_num}] Global Model Results:")
        print(f"  Loss    : {loss:.4f}")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1 Score: {f1:.4f}")

        history["rounds"].append(round_num)
        history["loss"].append(loss)
        history["accuracy"].append(accuracy)
        history["f1_score"].append(f1)

    return global_net, history


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("\n" + "="*60)
    print("LOADING DATASET")
    print("="*60)
    RAW_DIR = os.path.join(os.getcwd(), "data", "raw")
    client_loaders, test_loader = load_fl_data(
        RAW_DIR,
        num_real_clients=NUM_REAL_CLIENTS,  
        max_records=None,                   # use None for full data
    )

    def client_fn(cid: str) -> fl.client.Client:
        client_id = int(cid)
        net = ECGNet().to(device)
        if client_id < NUM_REAL_CLIENTS:
            return ECGClient(net, client_loaders[client_id], test_loader, device).to_client()
        else:
            return HoneypotClient(net, device).to_client()

    strategy = ShadowHoneypotStrategy(
        num_real_clients=NUM_REAL_CLIENTS,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=TOTAL_CLIENTS,
        min_evaluate_clients=NUM_REAL_CLIENTS,
        min_available_clients=TOTAL_CLIENTS,
    )

    print("\n" + "="*60)
    print("STARTING FEDERATED LEARNING")
    print("="*60)
    global_net, history = run_fl_simulation(
        client_fn=client_fn,
        num_clients=TOTAL_CLIENTS,
        num_rounds=NUM_ROUNDS,
        strategy=strategy,
        device=device,
        test_loader=test_loader,
    )

    if history["rounds"]:
        plot_training_history(
            history=history,
            save_path=os.path.join("results", "fl_training_history.png"),
        )

    run_attack_demo(device, test_loader, output_dir="results")


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn', force=True)
    main()